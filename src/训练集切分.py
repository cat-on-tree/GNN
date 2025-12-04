import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ========== 配置 ==========
input_path = "../data/benchmark/PrimeKG/edges.csv"
nodes_path = "../data/benchmark/PrimeKG/nodes.csv"
save_dir = "../data/benchmark/PrimeKG"
os.makedirs(save_dir, exist_ok=True)

seed = 42
val_ratio = 0.1
num_neg_ratio = 1

# ========== 1. 读取数据 ==========
print("Loading edges...")
df = pd.read_csv(input_path)
print(f"Original edges: {len(df)}")

nodes_df = pd.read_csv(nodes_path)
num_nodes = int(nodes_df['node_index'].max() + 1)
print(f"Total nodes: {num_nodes}")

# ========== 2. 解决泄漏 (Leakage Solution) ==========
print("Handling undirected edges to prevent leakage...")

# 构造去重键
df['src'] = df[['x_index', 'y_index']].min(axis=1)
df['dst'] = df[['x_index', 'y_index']].max(axis=1)

# 去重
unique_edges = df.drop_duplicates(subset=['relation', 'src', 'dst']).copy()
print(f"Unique undirected edges: {len(unique_edges)}")

# 打乱
rng = np.random.default_rng(seed)
perm = rng.permutation(len(unique_edges))
unique_edges = unique_edges.iloc[perm].reset_index(drop=True)

# ========== 3. 划分训练/验证集 ==========
split_idx = int(len(unique_edges) * (1 - val_ratio))
train_base = unique_edges.iloc[:split_idx].copy()
val_base = unique_edges.iloc[split_idx:].copy()

# 添加 Label
train_base['label'] = 1
val_base['label'] = 1

print(f"Train base (pos): {len(train_base)}")
print(f"Val base (pos): {len(val_base)}")


# ========== 4. 生成 Hard Negatives ==========
def generate_hard_negatives(pos_df, num_nodes, rng):
    print(f"Generating hard negatives for {len(pos_df)} samples...")

    # 使用 set 进行快速查找 (relation, min_node, max_node)
    # 确保查找时也是无向的逻辑
    pos_set = set(zip(pos_df['relation'], pos_df['src'], pos_df['dst']))

    neg_src = []
    neg_dst = []
    neg_rel = []

    # 取出 numpy array 加速
    src_arr = pos_df['src'].values
    dst_arr = pos_df['dst'].values
    rel_arr = pos_df['relation'].values

    batch_size = 100000
    num_samples = len(pos_df)

    for i in tqdm(range(0, num_samples, batch_size)):
        end = min(i + batch_size, num_samples)

        b_src = src_arr[i:end]
        b_rel = rel_arr[i:end]

        # 随机生成假尾节点
        b_neg_dst = rng.integers(0, num_nodes, size=len(b_src))

        for j in range(len(b_src)):
            h, t_fake, r = b_src[j], b_neg_dst[j], b_rel[j]

            # 检查冲突 (转为无向 min-max 后查表)
            check_u, check_v = min(h, t_fake), max(h, t_fake)

            # 拒绝采样：如果是自环，或者该边在正样本中存在
            while check_u == check_v or (r, check_u, check_v) in pos_set:
                t_fake = rng.integers(0, num_nodes)
                check_u, check_v = min(h, t_fake), max(h, t_fake)

            # 存入结果 (统一存为 min-max 格式，保持清洁)
            neg_src.append(check_u)
            neg_dst.append(check_v)
            neg_rel.append(r)

    return pd.DataFrame({
        'relation': neg_rel,
        'x_index': neg_src,  # 直接叫最终列名
        'y_index': neg_dst,  # 直接叫最终列名
        'label': 0
    })


train_neg = generate_hard_negatives(train_base, num_nodes, rng)
val_neg = generate_hard_negatives(val_base, num_nodes, rng)

# ========== 5. 合并与保存 (修复 Bug 的关键步骤) ==========
print("Merging dataframes...")

# 【关键修改】
# 我们只提取需要的列，并且把 src/dst 重命名为 x_index/y_index
# 这样 train_pos 和 train_neg 就拥有完全一致的列结构：['relation', 'x_index', 'y_index', 'label']
columns_to_keep = ['relation', 'src', 'dst', 'label']

train_pos = train_base[columns_to_keep].rename(columns={'src': 'x_index', 'dst': 'y_index'}).reset_index(drop=True)
val_pos = val_base[columns_to_keep].rename(columns={'src': 'x_index', 'dst': 'y_index'}).reset_index(drop=True)

train_neg = train_neg.reset_index(drop=True)  # train_neg 已经是 x_index, y_index 了
val_neg = val_neg.reset_index(drop=True)

# 再次确认列名一致
print(f"Train Pos Cols: {train_pos.columns.tolist()}")
print(f"Train Neg Cols: {train_neg.columns.tolist()}")

# 安全合并
train_final = pd.concat([train_pos, train_neg], ignore_index=True)
val_final = pd.concat([val_pos, val_neg], ignore_index=True)

# 打乱
train_final = train_final.sample(frac=1, random_state=seed).reset_index(drop=True)
val_final = val_final.sample(frac=1, random_state=seed).reset_index(drop=True)

print("Saving files...")
train_final.to_csv(os.path.join(save_dir, "train_edges.csv"), index=False)
val_final.to_csv(os.path.join(save_dir, "val_edges.csv"), index=False)

print("Done! Files saved to:", save_dir)