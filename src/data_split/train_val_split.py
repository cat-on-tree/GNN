import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict

# ================= 配置 =================
input_path = "../data/benchmark/PrimeKG/edges.csv"
nodes_path = "../data/benchmark/PrimeKG/nodes.csv"
save_dir = "../data/benchmark/PrimeKG"
os.makedirs(save_dir, exist_ok=True)

seed = 42
val_ratio = 0.1
rng = np.random.default_rng(seed)

# ================= 1. 读取数据 =================
print(">>> Step 1: Loading data...")
df = pd.read_csv(input_path)
nodes_df = pd.read_csv(nodes_path)
num_nodes = int(nodes_df['node_index'].max() + 1)

print(f"Original edges: {len(df)}")

# ================= 2. 无向图标准化与去重 =================
print(">>> Step 2: Normalization (Undirected)...")
# 这一步防止 A->B 和 B->A 的泄漏
df['src'] = df[['x_index', 'y_index']].min(axis=1)
df['dst'] = df[['x_index', 'y_index']].max(axis=1)
unique_edges = df.drop_duplicates(subset=['relation', 'src', 'dst']).copy()
# 彻底打乱
unique_edges = unique_edges.sample(frac=1, random_state=seed).reset_index(drop=True)

# ================= 3. 划分正样本 =================
print(">>> Step 3: Splitting Positives...")
split_idx = int(len(unique_edges) * (1 - val_ratio))
train_pos = unique_edges.iloc[:split_idx].copy()
val_pos = unique_edges.iloc[split_idx:].copy()
train_pos['label'] = 1
val_pos['label'] = 1

print(f"Train Pos: {len(train_pos)} | Val Pos: {len(val_pos)}")

# ================= 4. 构建类型约束候选池 (Type-Constrained Pool) =================
# 【核心修改】: 统计每种 Relation 允许连接的节点集合
print(">>> Step 4: Building Type-Constrained Candidate Pools...")

rel_to_candidates = defaultdict(set)

# 使用全量正样本来统计合法的节点类型分布
# 因为做了 min-max 标准化，我们把 src 和 dst 都加入候选池
all_rels = unique_edges['relation'].values
all_src = unique_edges['src'].values
all_dst = unique_edges['dst'].values

for r, u, v in zip(all_rels, all_src, all_dst):
    rel_to_candidates[r].add(u)
    rel_to_candidates[r].add(v)

# 转为 numpy array 以便快速采样
pool_dict = {}
for r, nodes in rel_to_candidates.items():
    pool_dict[r] = np.array(list(nodes))

print(f"Candidate pools built for {len(pool_dict)} relations.")

# ================= 5. 构建 Global Ban Set =================
# 防止负样本撞上已知的正样本
all_pos_df = pd.concat([train_pos, val_pos])
global_ban_set = set(zip(all_pos_df['relation'], all_pos_df['src'], all_pos_df['dst']))


# ================= 6. 类型约束负采样函数 =================
def generate_type_constrained_negatives(pos_df, ban_set, candidate_pools, rng, desc):
    print(f"Generating Type-Constrained Negatives for {desc}...")

    neg_rows = []

    # 按 relation 分组处理，提高效率
    grouped = pos_df.groupby('relation')

    for r, group in tqdm(grouped, total=len(grouped), desc=desc):
        # 如果该关系没有候选池（极罕见），跳过
        if r not in candidate_pools:
            continue

        candidates = candidate_pools[r]
        pool_size = len(candidates)

        # 提取当前组的头节点
        srcs = group['src'].values
        num_samples = len(srcs)

        # 批量随机采样：从该关系的【合法候选池】里选
        rand_indices = rng.integers(0, pool_size, size=num_samples)
        fake_dsts = candidates[rand_indices]

        for i in range(num_samples):
            h = srcs[i]
            t_fake = fake_dsts[i]

            check_u, check_v = min(h, t_fake), max(h, t_fake)

            # 冲突检查 (自环 或 撞正样本)
            retry = 0
            max_retry = 5
            while (check_u == check_v or (r, check_u, check_v) in ban_set) and retry < max_retry:
                # 重采
                t_fake = candidates[rng.integers(0, pool_size)]
                check_u, check_v = min(h, t_fake), max(h, t_fake)
                retry += 1

            if retry < max_retry:
                # 成功：存入 x, y, label=0
                neg_rows.append([r, h, t_fake, 0])

    return pd.DataFrame(neg_rows, columns=['relation', 'x_index', 'y_index', 'label'])


# ================= 7. 执行负采样 =================
train_neg = generate_type_constrained_negatives(train_pos, global_ban_set, pool_dict, rng, "Train")
val_neg = generate_type_constrained_negatives(val_pos, global_ban_set, pool_dict, rng, "Val")

# ================= 8. 保存 =================
print(">>> Step 5: Saving...")


# 构造最终 DataFrame
def format_final(pos, neg):
    pos_fmt = pd.DataFrame({
        'relation': pos['relation'],
        'x_index': pos['src'],
        'y_index': pos['dst'],
        'label': 1
    })
    final = pd.concat([pos_fmt, neg], ignore_index=True)
    # 打乱
    return final.sample(frac=1, random_state=seed).reset_index(drop=True)


train_final = format_final(train_pos, train_neg)
val_final = format_final(val_pos, val_neg)

train_final.to_csv(os.path.join(save_dir, "train_edges.csv"), index=False)
val_final.to_csv(os.path.join(save_dir, "val_edges.csv"), index=False)

print(f"✅ Done! Train Size: {len(train_final)}, Val Size: {len(val_final)}")