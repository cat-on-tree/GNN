import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict

# ==========================================
# 1. 配置与加载
# ==========================================
save_dir = "../../data/benchmark/PrimeKG/"
# 输入：你提取的 indication 专用测试候选集
input_candidates_path = os.path.join(save_dir, "val_indication.csv")
train_path = os.path.join(save_dir, "train_edges.csv")
val_path = os.path.join(save_dir, "val_edges.csv")
nodes_path = os.path.join(save_dir, "nodes.csv")
output_path = os.path.join(save_dir, "test_hard.csv")

seed = 42
rng = np.random.default_rng(seed)

print("🚀 Generating Strict Degree-Matched Test Set...")
print(">>> Step 0: Loading Data & Types...")

# 加载节点类型
df_nodes = pd.read_csv(nodes_path)
valid_drug_ids = set(df_nodes[df_nodes['node_type'] == 'drug']['node_index'].unique())
valid_disease_ids = set(df_nodes[df_nodes['node_type'] == 'disease']['node_index'].unique())

print(f"   Valid Drugs: {len(valid_drug_ids)}")
print(f"   Valid Diseases: {len(valid_disease_ids)}")

# 加载边数据
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_candidates = pd.read_csv(input_candidates_path)

# ==========================================
# 2. 强制方向校正 (Drug -> Disease)
# ==========================================
print("\n>>> Step 1: Enforcing strictly Drug->Disease orientation...")

clean_rows = []
dropped_count = 0

for _, row in df_candidates.iterrows():
    u, v = int(row['x_index']), int(row['y_index'])
    r = row['relation']

    # 情况 A: 顺序正确
    if u in valid_drug_ids and v in valid_disease_ids:
        clean_rows.append({'relation': r, 'x_index': u, 'y_index': v, 'label': 1})

    # 情况 B: 顺序反了 (Disease, Drug) -> 翻转
    elif v in valid_drug_ids and u in valid_disease_ids:
        clean_rows.append({'relation': r, 'x_index': v, 'y_index': u, 'label': 1})

    # 情况 C: 类型不对 (比如 Drug-Drug 或 Disease-Disease)，丢弃
    else:
        dropped_count += 1

test_pos = pd.DataFrame(clean_rows)
print(f"   Original Candidates: {len(df_candidates)}")
print(f"   Cleaned (Drug->Disease): {len(test_pos)}")
if dropped_count > 0:
    print(f"   Dropped {dropped_count} invalid edges.")

# ==========================================
# 3. 统计度数 (只统计 Disease)
# ==========================================
print("\n>>> Step 2: Calculating Exact Degrees for Diseases...")
# PrimeKG 是无向逻辑，疾病的度数 = 它在 x 列出现的次数 + 它在 y 列出现的次数
# 我们需要统计 Train + Val 的全量图
all_edges = pd.concat([df_train, df_val])

# 简单暴力法：把 x 和 y 拼起来统计
all_nodes_series = pd.concat([all_edges['x_index'], all_edges['y_index']])
degrees_map = all_nodes_series.value_counts().to_dict()

# 构建查找表 {degree: [disease_list]}
degree_lookup = defaultdict(list)
for node in valid_disease_ids:
    d = degrees_map.get(node, 0)
    degree_lookup[d].append(node)

# 转 numpy 以便快速采样
degree_lookup_np = {k: np.array(v) for k, v in degree_lookup.items()}
print(f"   Degree buckets created. Max degree found: {max(degree_lookup_np.keys()) if degree_lookup_np else 0}")

# ==========================================
# 4. 构建 Ban Set (防止采样到真样本)
# ==========================================
print("\n>>> Step 3: Building Global Ban Set...")
# Ban Set 包含 Train + Val + Test 本身
existing_edges = pd.concat([
    df_train[df_train['label'] == 1],
    df_val[df_val['label'] == 1],
    test_pos
])

global_ban_set = set(zip(
    existing_edges['relation'],
    existing_edges[['x_index', 'y_index']].min(axis=1),
    existing_edges[['x_index', 'y_index']].max(axis=1)
))

# ==========================================
# 5. 生成度匹配负样本
# ==========================================
print("\n>>> Step 4: Generating Hard Negatives (Tail Replacement)...")
hard_neg_rows = []
skipped_count = 0

# 容忍度策略：先找完全一样，没有就找相似
tolerance_levels = [0.0, 0.05, 0.1, 0.2]  # 0%, 5%, 10%, 20%

pos_records = test_pos.to_dict('records')

for row in tqdm(pos_records):
    drug = row['x_index']
    real_disease = row['y_index']
    rel = row['relation']

    # 目标度数
    target_deg = degrees_map.get(real_disease, 0)

    found = False
    for tol in tolerance_levels:
        min_d = int(target_deg * (1 - tol))
        max_d = int(target_deg * (1 + tol))

        # 收集候选池
        candidates_arrays = []
        for d in range(min_d, max_d + 1):
            if d in degree_lookup_np:
                candidates_arrays.append(degree_lookup_np[d])

        if not candidates_arrays:
            continue

        pool = np.concatenate(candidates_arrays)

        # 尝试采样
        for _ in range(20):
            fake_disease = rng.choice(pool)

            # 必须不是自己，且不构成已知边
            check_u, check_v = min(drug, fake_disease), max(drug, fake_disease)

            if fake_disease != real_disease and (rel, check_u, check_v) not in global_ban_set:
                hard_neg_rows.append({
                    'relation': rel,
                    'x_index': drug,  # 保持是 Drug
                    'y_index': fake_disease,  # 替换为度数相似的 Fake Disease
                    'label': 0
                })
                found = True
                break
        if found:
            break

    if not found:
        skipped_count += 1

print(f"   Generated {len(hard_neg_rows)} hard negatives.")
if skipped_count > 0:
    print(f"   Skipped {skipped_count} due to no suitable candidate.")

# ==========================================
# 6. 保存
# ==========================================
print("\n>>> Step 5: Saving...")
df_neg = pd.DataFrame(hard_neg_rows)

# 为了对齐，我们只保留找到了负样本的正样本
# 这里通过索引对齐比较麻烦，简单点直接 concat 即可
# 但如果追求严谨的 Pair，可以把上面的逻辑改成 yield (pos, neg)
# 鉴于 Indication 数量不多，我们允许极少量损失，只保存配对成功的

# 这里我直接保存所有成功的负样本，和原始正样本混合
# 注意：这可能导致 pos 比 neg 多几个（如果 skip > 0）
# 对于 AUC 计算没影响，影响几乎可以忽略

test_hard = pd.concat([test_pos, df_neg], ignore_index=True)
test_hard = test_hard.sample(frac=1, random_state=seed).reset_index(drop=True)

test_hard.to_csv(output_path, index=False)
print(f"✅ Saved to {output_path}")
print(f"   Total Samples: {len(test_hard)}")