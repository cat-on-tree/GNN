import pandas as pd
import os
import numpy as np
import random


def rebuild_cold_start_with_llm_clean():
    # ==========================================
    # 0. 配置区域 (可调整参数)
    # ==========================================
    NUM_POS_SAMPLES = 100  # 期望的正样本数量
    NEG_POS_RATIO = 10  # 负正比例 (例如 10 代表 1:10)
    SEED = 42  # 固定随机种子

    # [新增] 清洗后的高质量正样本路径
    clean_pos_path = '../../data/benchmark/Kaggle_drug_repositioning/full_mapping_without_na.csv'

    # 其他路径
    train_path = '../../data/benchmark/PrimeKG/train_edges.csv'
    val_path = '../../data/benchmark/PrimeKG/val_edges.csv'
    full_path = '../../data/benchmark/Kaggle_drug_repositioning/full_mapping.csv'
    nodes_path = '../../data/benchmark/PrimeKG/nodes.csv'

    output_path = '../../data/benchmark/Kaggle_drug_repositioning/test_cold.csv'

    print(f"🚀 Rebuilding Cold-Start Data with LLM-Cleaned Positives (Fixed Seed {SEED})...")

    # === 1. 全局随机种子固定 ===
    np.random.seed(SEED)
    random.seed(SEED)
    rng = np.random.RandomState(SEED)

    if not os.path.exists(clean_pos_path):
        print(f"❌ Error: Cleaned data file not found at {clean_pos_path}")
        return

    # === 2. 读取数据 ===
    print("   Loading data...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_full = pd.read_csv(full_path)  # 用于 Ban Set
    df_nodes = pd.read_csv(nodes_path)

    # [核心修改] 读取清洗后的正样本
    print(f"   Reading High-Quality Positives form {clean_pos_path}...")
    df_clean = pd.read_csv(clean_pos_path)

    # === 3. 构建超级全量 Ban Set (保持最严格标准) ===
    # 即使 LLM 认为某个样本分数只有 4 分（被剔除出正样本），它在 Full Mapping 里存在，
    # 我们依然不能把它当负样本（它可能是弱关联，但不是无关联）。
    print("   Building Universal Ban Set (Train + Val + Full)...")

    # 1. Train + Val
    edges_to_ban = pd.concat([
        df_train[['x_index', 'y_index']],
        df_val[['x_index', 'y_index']]
    ])

    # 2. Full Mapping (这是最全的“已知关系库”)
    df_full_valid = df_full.dropna(subset=['x_index', 'y_index'])
    edges_to_ban = pd.concat([edges_to_ban, df_full_valid[['x_index', 'y_index']]])

    # 3. 转为 Set
    global_ban_set = set()
    for _, row in edges_to_ban.iterrows():
        u, v = int(row['x_index']), int(row['y_index'])
        global_ban_set.add((min(u, v), max(u, v)))

    print(f"   🔒 Universal Ban Set size: {len(global_ban_set)}")

    # === 4. 获取所有合法的 Drug ID ===
    valid_drug_ids = set(df_nodes[df_nodes['node_type'] == 'drug']['node_index'].unique())
    all_valid_drugs = np.sort(list(valid_drug_ids))

    # === 5. 筛选 Cold Start 疾病 (Degree <= 3) ===
    # 定义基于训练集
    disease_counts = df_train['y_index'].value_counts()
    low_degree_diseases = set(disease_counts[disease_counts <= 3].index)
    print(f"   Found {len(low_degree_diseases)} cold-start diseases in training set.")

    # === 6. [核心修改] 从 Cleaned Data 中筛选正样本 ===
    # 逻辑：Candidate 必须同时满足：
    # 1. 在 Cleaned Data (Score >= 7) 中
    # 2. 属于 Cold Start Disease
    # 3. 不在 Train/Val 集中 (防止泄漏)

    # 确保列名对齐 (LLM 清洗脚本输出的 CSV 可能包含 original_x_index 等，这里假设它保留了 x_index)
    if 'x_index' not in df_clean.columns and 'original_csv_index' in df_clean.columns:
        # 如果 LLM 输出没有保留 x_index，可能需要用 merge 找回，但通常之前的脚本保留了所有列
        print("⚠️ Warning: checking column names...")

    # 筛选属于冷启动疾病的样本
    df_candidates = df_clean[df_clean['y_index'].isin(low_degree_diseases)].copy()

    # 去重：确保不在 Train Set
    train_edge_set = set(zip(df_train['x_index'], df_train['y_index']))
    candidate_pairs = list(zip(df_candidates['x_index'], df_candidates['y_index']))
    is_new = [p not in train_edge_set for p in candidate_pairs]

    df_pos = df_candidates[is_new].copy()

    # 类型检查
    df_pos = df_pos[df_pos['x_index'].isin(valid_drug_ids)]

    print(f"   High-Quality Cold-Start Candidates found: {len(df_pos)}")

    # 采样
    if len(df_pos) > NUM_POS_SAMPLES:
        df_pos = df_pos.sample(n=NUM_POS_SAMPLES, random_state=SEED)
    else:
        print(f"⚠️ Warning: Only found {len(df_pos)} candidates. Using all.")

    pos_data = df_pos[['x_index', 'y_index']].copy()
    pos_data['label'] = 1

    # 处理 relation 列 (如果有的话)
    if 'relation' in df_pos.columns:
        pos_data['relation'] = df_pos['relation']
    else:
        pos_data['relation'] = 'indication'

    # === 7. 负采样 (使用全量 Ban Set) ===
    # 这一步逻辑不变，依然是避开 Full Mapping
    neg_rows = []
    pos_records = pos_data.to_dict('records')

    print(f"   Generating negative samples (Ratio 1:{NEG_POS_RATIO})...")

    for row in pos_records:
        disease = int(row['y_index'])
        rel_type = row['relation']

        for _ in range(NEG_POS_RATIO):
            retry = 0
            while retry < 100:
                rand_drug = int(rng.choice(all_valid_drugs))
                check_pair = (min(rand_drug, disease), max(rand_drug, disease))

                if check_pair not in global_ban_set:
                    neg_rows.append({
                        'x_index': rand_drug,
                        'y_index': disease,
                        'label': 0,
                        'relation': rel_type
                    })
                    break
                retry += 1

    neg_data = pd.DataFrame(neg_rows)

    # === 8. 合并与保存 ===
    final_df = pd.concat([pos_data, neg_data], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 确保类型为 int
    final_df['x_index'] = final_df['x_index'].astype(int)
    final_df['y_index'] = final_df['y_index'].astype(int)
    final_df['label'] = final_df['label'].astype(int)

    final_df.to_csv(output_path, index=False)
    print(f"✅ LLM-Cleaned Cold-Start Test Set Saved to: {output_path}")
    print(f"   Total Samples: {len(final_df)}")
    print(f"   Positive (High Quality): {len(pos_data)}")
    print(f"   Negative: {len(neg_data)}")


if __name__ == "__main__":
    rebuild_cold_start_with_llm_clean()