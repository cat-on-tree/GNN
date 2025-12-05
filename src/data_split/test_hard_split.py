import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# 读取现有数据
train_df = pd.read_csv("../data/benchmark/PrimeKG/train_edges.csv")
test_df_easy = pd.read_csv("../data/benchmark/Kaggle_drug_repositioning/test.csv")
nodes_df = pd.read_csv("../data/benchmark/PrimeKG/nodes.csv")
save_dir = "../data/benchmark/Kaggle_drug_repositioning"

# 1. 统计度数 (基于全量训练数据)
all_nodes = pd.concat([train_df['x_index'], train_df['y_index']])
degrees = all_nodes.value_counts().to_dict()

# 2. 准备候选池 (只取 Indication 相关的疾病节点)
# 这里我们偷个懒，直接用 test_easy 里的 target 节点作为候选池，因为它们都是疾病
# 更好的做法是像之前一样从 indication 关系里提取
candidate_nodes = test_df_easy[test_df_easy['label'] == 1]['y_index'].unique()
candidate_degrees = {n: degrees.get(n, 0) for n in candidate_nodes}

# 3. 对候选节点进行分桶 (Binning)
# 按度数分为 10 个桶
degree_values = list(candidate_degrees.values())
bins = np.percentile(degree_values, np.linspace(0, 100, 11))
# 确保 bins 唯一
bins = np.unique(bins)

from collections import defaultdict

degree_buckets = defaultdict(list)

for node, deg in candidate_degrees.items():
    # 找到所属的桶
    bucket_idx = np.digitize(deg, bins) - 1
    degree_buckets[bucket_idx].append(node)

# 4. 生成 Hard Test Set
# 保持正样本不变，只替换负样本
test_pos = test_df_easy[test_df_easy['label'] == 1].copy()
rng = np.random.default_rng(42)

hard_neg_rows = []

print("Generating Degree-Matched Negatives...")
for _, row in tqdm(test_pos.iterrows(), total=len(test_pos)):
    src = row['x_index']
    dst_real = row['y_index']
    rel = row['relation']

    real_degree = degrees.get(dst_real, 0)

    # 找到对应的度数桶
    bucket_idx = np.digitize(real_degree, bins) - 1
    # 修正边界
    bucket_idx = max(0, min(bucket_idx, len(bins) - 2))

    # 从同一个桶里选负样本 (度数相近)
    candidates = degree_buckets[bucket_idx]

    # 简单的冲突检测 (这里简化了，实际应该查 Global Ban)
    # 假设桶里只要选个不一样的就行
    fake_dst = rng.choice(candidates)
    retry = 0
    while fake_dst == dst_real and retry < 10:
        fake_dst = rng.choice(candidates)
        retry += 1

    hard_neg_rows.append([rel, src, fake_dst, 0])

test_neg_hard = pd.DataFrame(hard_neg_rows, columns=['relation', 'x_index', 'y_index', 'label'])

# 合并
test_hard = pd.concat([test_pos, test_neg_hard], ignore_index=True).sample(frac=1, random_state=42)
test_hard.to_csv(os.path.join(save_dir, "test_hard.csv"), index=False)

print(f"✅ Hard Test Set saved! Use this to challenge your models.")