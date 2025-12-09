import pandas as pd
import os
import numpy as np
import random


def rebuild_correctly_fixed_seed_strict():
    print("🚀 Rebuilding Cold-Start Data with STRICT TYPE CONSTRAINT (Fixed Seed 42)...")

    # === 1. 全局随机种子固定 ===
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    # === 2. 路径 ===
    train_path = '../../data/benchmark/PrimeKG/train_edges.csv'
    full_path = '../../data/benchmark/Kaggle_drug_repositioning/full_mapping.csv'
    nodes_path = '../../data/benchmark/PrimeKG/nodes.csv'  # 新增：需要读取nodes文件来获取类型
    output_path = '../../data/benchmark/Kaggle_drug_repositioning/test_cold.csv'

    if not os.path.exists(full_path) or not os.path.exists(nodes_path):
        print("❌ Full mapping or nodes file not found.")
        return

    # === 3. 读取数据 ===
    print("   Loading data...")
    df_train = pd.read_csv(train_path)
    df_full = pd.read_csv(full_path)
    df_nodes = pd.read_csv(nodes_path)

    # === 4. 获取所有合法的 Drug ID ===
    print("   Filtering valid drug IDs from nodes.csv...")
    # 筛选 node_type 为 'drug' 的节点 ID
    valid_drug_ids = set(df_nodes[df_nodes['node_type'] == 'drug']['node_index'].unique())

    # 还要确保这些 drug ID 确实存在于我们的索引体系中（取交集以防万一）
    # 这里我们假设 train 中的 x_index 包含了大部分药物，或者直接用 nodes 的药物 ID
    # 为了安全起见，我们只使用那些确实被标记为 drug 的 ID 作为负采样池

    # 将 set 转为排序好的 list/array 供 random.choice 使用
    # !!! 修正处：变量名修改为 valid_drug_ids !!!
    all_valid_drugs = np.sort(list(valid_drug_ids))

    print(f"   Found {len(all_valid_drugs)} valid drug nodes.")

    # === 5. 筛选 Cold Start 疾病 (Degree <= 3) ===
    disease_counts = df_train['y_index'].value_counts()
    low_degree_diseases = set(disease_counts[disease_counts <= 3].index)

    # === 6. 筛选正样本候选 ===
    # 检查是否有 relation 列
    has_relation = 'relation' in df_full.columns

    df_candidates = df_full[df_full['y_index'].isin(low_degree_diseases)].copy()

    # 去重：排除训练集中已有的边
    train_edge_set = set(zip(df_train['x_index'], df_train['y_index']))
    candidate_pairs = list(zip(df_candidates['x_index'], df_candidates['y_index']))
    is_new = [p not in train_edge_set for p in candidate_pairs]

    df_pos = df_candidates[is_new].copy()

    # 【新增检查】确保正样本里的 x 也是 drug (以防万一 full_mapping 里混入了杂质)
    df_pos = df_pos[df_pos['x_index'].isin(valid_drug_ids)]

    # 采样正样本 (如果超过 2000 条)
    if len(df_pos) > 2000:
        df_pos = df_pos.sample(n=2000, random_state=SEED)

    print(f"   Positives selected: {len(df_pos)}")

    # 构造正样本 DataFrame
    pos_data = df_pos[['x_index', 'y_index']].copy()
    pos_data['label'] = 1
    if has_relation:
        pos_data['relation'] = df_pos['relation']
    else:
        pos_data['relation'] = 'indication'

    # === 7. 负采样 (带类型约束) ===
    neg_rows = []

    # 更新现存边集合 (训练集 + 刚才选出的测试集正样本)
    existing_edges = train_edge_set.union(set(zip(pos_data['x_index'], pos_data['y_index'])))

    pos_records = pos_data.to_dict('records')

    print("   Generating negative samples with strict type constraints...")

    rng = np.random.RandomState(SEED)

    for row in pos_records:
        disease = row['y_index']
        rel_type = row['relation']

        # 尝试采样负样本
        for _ in range(100):
            # 从合法的 drug 列表中随机选
            rand_drug = rng.choice(all_valid_drugs)

            # 确保不构成已知的边
            if (rand_drug, disease) not in existing_edges:
                neg_rows.append({
                    'x_index': rand_drug,
                    'y_index': disease,
                    'label': 0,
                    'relation': rel_type
                })
                break
        else:
            print(f"⚠️ Warning: Could not find neg sample for disease {disease}")

    neg_data = pd.DataFrame(neg_rows)

    # === 8. 合并与保存 ===
    final_df = pd.concat([pos_data, neg_data], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    final_df['x_index'] = final_df['x_index'].astype(int)
    final_df['y_index'] = final_df['y_index'].astype(int)
    final_df['label'] = final_df['label'].astype(int)

    final_df.to_csv(output_path, index=False)
    print(f"✅ Reproducible Cold-Start Test Set Saved to: {output_path}")
    print(f"   Total Samples: {len(final_df)}")
    print(f"   Positive: {len(pos_data)}")
    print(f"   Negative: {len(neg_data)}")


if __name__ == "__main__":
    rebuild_correctly_fixed_seed_strict()