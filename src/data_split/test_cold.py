import pandas as pd
import os
import numpy as np
import random


def rebuild_correctly_fixed_seed():
    print("🚀 Rebuilding Cold-Start Data (Fixed Seed 42)...")

    # === 1. 全局随机种子固定 ===
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    # === 2. 路径 ===
    train_path = '../../data/benchmark/PrimeKG/train_edges.csv'
    full_path = '../../data/benchmark/Kaggle_drug_repositioning/full_mapping.csv'
    output_path = '../../data/benchmark/Kaggle_drug_repositioning/test_cold.csv'

    if not os.path.exists(full_path):
        print("❌ Full mapping not found.")
        return

    # === 3. 读取数据 ===
    df_train = pd.read_csv(train_path)
    df_full = pd.read_csv(full_path)

    # === 4. 筛选 Cold Start 疾病 (Degree <= 3) ===
    disease_counts = df_train['y_index'].value_counts()
    low_degree_diseases = set(disease_counts[disease_counts <= 3].index)

    # === 5. 筛选正样本候选 ===
    # 检查是否有 relation 列
    has_relation = 'relation' in df_full.columns

    df_candidates = df_full[df_full['y_index'].isin(low_degree_diseases)].copy()

    # 去重：排除训练集中已有的边
    # 使用 set 加速查找
    train_edge_set = set(zip(df_train['x_index'], df_train['y_index']))
    candidate_pairs = list(zip(df_candidates['x_index'], df_candidates['y_index']))
    is_new = [p not in train_edge_set for p in candidate_pairs]

    df_pos = df_candidates[is_new].copy()

    # 采样正样本 (如果超过 2000 条)
    if len(df_pos) > 2000:
        df_pos = df_pos.sample(n=2000, random_state=SEED)  # 固定种子

    print(f"   Positives selected: {len(df_pos)}")

    # 构造正样本 DataFrame
    pos_data = df_pos[['x_index', 'y_index']].copy()
    pos_data['label'] = 1
    if has_relation:
        pos_data['relation'] = df_pos['relation']
    else:
        pos_data['relation'] = 'indication'

    # === 6. 负采样 (固定逻辑) ===
    neg_rows = []
    all_drugs = np.sort(df_train['x_index'].unique())  # 排序以确保索引一致

    # 更新现存边集合 (训练集 + 刚才选出的测试集正样本)
    # 任何真实存在的边都不能作为负样本
    existing_edges = train_edge_set.union(set(zip(pos_data['x_index'], pos_data['y_index'])))

    pos_records = pos_data.to_dict('records')

    print("   Generating negative samples...")

    # 为了保证可复现，我们使用一个确定的随机状态生成器
    rng = np.random.RandomState(SEED)

    for row in pos_records:
        disease = row['y_index']
        rel_type = row['relation']

        # 尝试采样负样本
        # 为了避免无限循环(虽然不太可能)，设置最大尝试次数
        for _ in range(100):
            rand_drug = rng.choice(all_drugs)
            if (rand_drug, disease) not in existing_edges:
                neg_rows.append({
                    'x_index': rand_drug,
                    'y_index': disease,
                    'label': 0,
                    'relation': rel_type
                })
                break
        else:
            # 如果100次都碰撞了(极罕见)，为了保持数据平衡，还是要硬塞一个
            # 或者跳过。这里选择跳过，但这会破坏 1:1 平衡。
            # 考虑到图很稀疏，这种情况概率极低。
            print(f"⚠️ Warning: Could not find neg sample for disease {disease}")

    neg_data = pd.DataFrame(neg_rows)

    # === 7. 合并与保存 ===
    final_df = pd.concat([pos_data, neg_data], ignore_index=True)
    # 最后的打乱也必须固定种子
    final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 类型转换确保整洁
    final_df['x_index'] = final_df['x_index'].astype(int)
    final_df['y_index'] = final_df['y_index'].astype(int)
    final_df['label'] = final_df['label'].astype(int)

    final_df.to_csv(output_path, index=False)
    print(f"✅ Reproducible Cold-Start Test Set Saved to: {output_path}")
    print(f"   Total Samples: {len(final_df)}")
    print(f"   Positive: {len(pos_data)}")
    print(f"   Negative: {len(neg_data)}")


if __name__ == "__main__":
    rebuild_correctly_fixed_seed()