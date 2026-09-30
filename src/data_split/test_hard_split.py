import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict

# ==========================================
# 0. 配置与路径
# ==========================================
# 期望配置
NUM_POS_SAMPLES = 100  # 期望的正样本数量
NEG_POS_RATIO = 10  # 负正比例 (1:10)
SEED = 42

# 路径
input_clean_pos = "../../data/benchmark/Kaggle_drug_repositioning/full_mapping_without_na.csv"  # 正样本来源
input_raw_full = "../../data/benchmark/Kaggle_drug_repositioning/full_mapping.csv"  # Ban Set 来源
train_path = "../../data/benchmark/PrimeKG/train_edges.csv"
val_path = "../../data/benchmark/PrimeKG/val_edges.csv"
nodes_path = "../../data/benchmark/PrimeKG/nodes.csv"
save_dir = "../../data/benchmark/Kaggle_drug_repositioning"
os.makedirs(save_dir, exist_ok=True)

rng = np.random.default_rng(SEED)

print(f"🚀 Generating STRICT Degree-Matched Test Set...")
print(f"   Target: {NUM_POS_SAMPLES} Positives, Ratio 1:{NEG_POS_RATIO}")

# ==========================================
# 1. 准备节点类型与度数信息
# ==========================================
print(">>> Step 1: Loading Nodes & Calculating Degrees...")
df_nodes = pd.read_csv(nodes_path)
valid_disease_ids = set(df_nodes[df_nodes['node_type'] == 'disease']['node_index'].unique())
valid_drug_ids = set(df_nodes[df_nodes['node_type'] == 'drug']['node_index'].unique())

# 读取训练/验证集 (用于计算度数 + Ban Set)
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)

# 计算度数 (仅基于已知图谱 Train+Val)
all_known_edges = pd.concat([df_train, df_val])
all_nodes_series = pd.concat([all_known_edges['x_index'], all_known_edges['y_index']])
degrees = all_nodes_series.value_counts().to_dict()

# 构建 {度数: [Valid Disease List]} 映射表
degree_lookup = defaultdict(list)
for node in valid_disease_ids:
    d = degrees.get(node, 0)
    degree_lookup[d].append(node)
# 转 numpy array 加速
degree_lookup_np = {k: np.array(v) for k, v in degree_lookup.items()}

# ==========================================
# 2. 构建超级禁忌表 (Global Ban Set)
# ==========================================
print(">>> Step 2: Building Comprehensive Global Ban Set...")

# A. 训练集 + 验证集
ban_sources = [df_train, df_val]

# B. 全量原始数据 (用于排除潜在假负例)
# 关键处理：只有 x_index 和 y_index 都存在的行才有资格进入 Ban Set
df_raw = pd.read_csv(input_raw_full)
df_raw_valid = df_raw.dropna(subset=['x_index', 'y_index']).copy()
# 确保类型为 int，以便后续匹配
df_raw_valid['x_index'] = df_raw_valid['x_index'].astype(int)
df_raw_valid['y_index'] = df_raw_valid['y_index'].astype(int)
ban_sources.append(df_raw_valid)

# 合并所有源
df_ban_all = pd.concat(ban_sources)

# 构建 Set: (min(u,v), max(u,v)) 无向匹配
# 这样最安全，不管谁是头谁是尾，只要连过就不做负样本
global_ban_set = set(zip(
    df_ban_all[['x_index', 'y_index']].min(axis=1),
    df_ban_all[['x_index', 'y_index']].max(axis=1)
))

print(f"   Total unique edges banned: {len(global_ban_set)}")

# ==========================================
# 3. 采样正样本 (来自 Clean Source)
# ==========================================
print(f">>> Step 3: Sampling {NUM_POS_SAMPLES} Positives...")
df_clean = pd.read_csv(input_clean_pos)
df_clean['x_index'] = df_clean['x_index'].astype(int)
df_clean['y_index'] = df_clean['y_index'].astype(int)

# 过滤 1: 类型正确 (Drug -> Disease)
mask_type = df_clean['x_index'].isin(valid_drug_ids) & df_clean['y_index'].isin(valid_disease_ids)
candidates = df_clean[mask_type].copy()

# 过滤 2: 不能在 Train/Val 中 (防止数据泄漏)
# 这里用一个简单的 set 查重
train_val_pairs = set(zip(df_train['x_index'], df_train['y_index'])) | \
                  set(zip(df_val['x_index'], df_val['y_index']))


def is_leak(row):
    return (row['x_index'], row['y_index']) in train_val_pairs


candidates['is_leak'] = candidates.apply(is_leak, axis=1)
clean_candidates = candidates[~candidates['is_leak']].copy()

# 采样
if len(clean_candidates) > NUM_POS_SAMPLES:
    test_pos = clean_candidates.sample(n=NUM_POS_SAMPLES, random_state=SEED).reset_index(drop=True)
else:
    test_pos = clean_candidates.reset_index(drop=True)

# 格式化正样本
test_pos = test_pos[['x_index', 'y_index']].copy()
test_pos['label'] = 1
test_pos['relation'] = 'indication'

print(f"   Final Positives: {len(test_pos)}")

# ==========================================
# 4. 生成度匹配负样本 (1:N)
# ==========================================
target_neg_count = len(test_pos) * NEG_POS_RATIO
print(f">>> Step 4: Generating Negatives (Target: {target_neg_count})...")

hard_neg_rows = []
tolerance_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

for _, row in tqdm(test_pos.iterrows(), total=len(test_pos)):
    src = int(row['x_index'])  # Drug
    dst_real = int(row['y_index'])  # Disease

    target_deg = degrees.get(dst_real, 0)
    created_count = 0
    retry = 0

    # 尝试生成 N 个负样本
    while created_count < NEG_POS_RATIO and retry < 50:
        found_one = False

        for tol in tolerance_levels:
            min_d = int(target_deg * (1 - tol))
            max_d = int(target_deg * (1 + tol))

            candidates_list = [degree_lookup_np[d] for d in range(min_d, max_d + 1) if d in degree_lookup_np]
            if not candidates_list: continue

            pool = np.concatenate(candidates_list)

            # 在当前容忍度下尝试 10 次
            for _ in range(10):
                fake_dst = int(rng.choice(pool))

                if fake_dst == dst_real: continue

                # 🔥 查 Global Ban Set
                check_pair = (min(src, fake_dst), max(src, fake_dst))

                if check_pair not in global_ban_set:
                    hard_neg_rows.append({
                        'relation': 'indication',
                        'x_index': src,
                        'y_index': fake_dst,
                        'label': 0
                    })
                    created_count += 1
                    found_one = True
                    break

            if found_one: break  # 跳出 tol 循环，生成下一个负样本

        if not found_one: retry += 1

test_neg = pd.DataFrame(hard_neg_rows)

# ==========================================
# 5. 保存
# ==========================================
print(">>> Step 5: Saving...")
# 合并
test_final = pd.concat([test_pos, test_neg], ignore_index=True)
test_final = test_final.sample(frac=1, random_state=SEED).reset_index(drop=True)

# 确保列顺序和类型
cols = ['relation', 'x_index', 'y_index', 'label']
test_final = test_final[cols]
test_final['x_index'] = test_final['x_index'].astype(int)
test_final['y_index'] = test_final['y_index'].astype(int)
test_final['label'] = test_final['label'].astype(int)

# 保存
filename = "test_hard.csv"
output_path = os.path.join(save_dir, filename)
test_final.to_csv(output_path, index=False)

print(f"🎉 Success! Saved to {output_path}")
print(f"   Positives: {len(test_pos)}")
print(f"   Negatives: {len(test_neg)}")
print(f"   Ratio: 1:{len(test_neg) / len(test_pos):.2f}")