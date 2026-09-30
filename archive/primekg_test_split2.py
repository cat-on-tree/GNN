import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ==========================================
# 1. 配置区域
# ==========================================
# 【修改 1】输入路径改为上一步生成的 Indication 测试集
save_dir = "../../data/benchmark/PrimeKG/"
input_clean_path = os.path.join(save_dir, "val_indication.csv")

# 训练集和验证集路径
train_path = os.path.join(save_dir, "train_edges.csv")
val_path = os.path.join(save_dir, "val_edges.csv")
nodes_path = os.path.join(save_dir, "nodes.csv")

# 输出路径
output_path = os.path.join(save_dir, "test.csv")

# 采样配置
# 【修改 2】测试集数量设为 None，表示"有多少用多少"（对应那 891 条），或者你可以指定具体数字
num_test_samples = None
seed = 42
rng = np.random.default_rng(seed)

# ==========================================
# 2. 读取数据与构建类型白名单
# ==========================================
print(">>> Step 0: Loading data & Node Types...")
df_nodes = pd.read_csv(nodes_path)
# 获取所有合法的 Drug ID 和 Disease ID
valid_drug_ids = set(df_nodes[df_nodes['node_type'] == 'drug']['node_index'].unique())
valid_disease_ids = set(df_nodes[df_nodes['node_type'] == 'disease']['node_index'].unique())
print(f"   Valid Drugs: {len(valid_drug_ids)}")
print(f"   Valid Diseases: {len(valid_disease_ids)}")

# 读取刚刚提取的 indication 测试集
if not os.path.exists(input_clean_path):
    raise FileNotFoundError(f"找不到输入文件: {input_clean_path}")

df_test = pd.read_csv(input_clean_path)
print(f"   Loaded Test Candidates: {len(df_test)}")

# ==========================================
# 3. 严格清洗正样本 (Strict Filtering)
# ==========================================
print("\n>>> Step 1: Cleaning Positives...")

# 1. 类型检查：Head 必须是 Drug，Tail 必须是 Disease
# PrimeKG 中通常较小的索引在 x, 较大的在 y，但不保证 x 总是 drug。
# 我们需要确保 (Drug, Disease) 的顺序。
valid_rows = []
for _, row in df_test.iterrows():
    u, v = int(row['x_index']), int(row['y_index'])

    # 判断方向
    if u in valid_drug_ids and v in valid_disease_ids:
        # 顺序正确: Drug -> Disease
        valid_rows.append({'relation': row['relation'], 'x_index': u, 'y_index': v, 'label': 1})
    elif v in valid_drug_ids and u in valid_disease_ids:
        # 顺序反了: Disease -> Drug，我们要把它翻转回来，便于统一处理
        valid_rows.append({'relation': row['relation'], 'x_index': v, 'y_index': u, 'label': 1})
    # else: 类型不匹配（非 Drug-Disease），丢弃

test_pos_typed = pd.DataFrame(valid_rows)
print(f"   Valid Type (Drug->Disease) Candidates: {len(test_pos_typed)}")

# 2. 训练集查重 (只查 Train，不查 Val)
# 【核心逻辑修正】：因为这些数据来自 Val，所以肯定在 Val 里。我们只看它们是否不幸也在 Train 里。
print("   Checking leakage against TRAIN set only...")
df_train = pd.read_csv(train_path)
# 只需要 Train 的正样本做 Ban List
train_pos_df = df_train[df_train['label'] == 1]
train_ban_set = set(zip(
    train_pos_df['relation'],
    train_pos_df[['x_index', 'y_index']].min(axis=1),
    train_pos_df[['x_index', 'y_index']].max(axis=1)
))

clean_rows = []
leak_count = 0
for _, row in tqdm(test_pos_typed.iterrows(), total=len(test_pos_typed), desc="Checking Train leakage"):
    # 无向图比较：使用 min-max
    check_tuple = (row['relation'], min(row['x_index'], row['y_index']), max(row['x_index'], row['y_index']))
    if check_tuple in train_ban_set:
        leak_count += 1
    else:
        clean_rows.append(row)

test_pos_clean = pd.DataFrame(clean_rows).reset_index(drop=True)
print(f"   Removed {leak_count} edges leaking in Train. Final Positives: {len(test_pos_clean)}")

# ==========================================
# 4. 确定最终正样本数量
# ==========================================
# 如果没有指定数量，就全用；如果指定了，就采样
if num_test_samples is None or num_test_samples > len(test_pos_clean):
    test_pos_final = test_pos_clean
    real_sample_num = len(test_pos_clean)
else:
    test_pos_final = test_pos_clean.sample(n=num_test_samples, random_state=seed).reset_index(drop=True)
    real_sample_num = num_test_samples

print(f"\n>>> Step 2: Selected {real_sample_num} positives for testing.")

# ==========================================
# 5. 生成负样本 (Strict Hard Negatives)
# ==========================================
print("\n>>> Step 3: Generating STRICT Negative Samples...")

# 负样本 Ban List：必须包含 Train + Val + Test本身
# 【注意】：这里必须加回 Val，因为我们要确保生成的“假药”不是真的有效（即不在 Val 或 Train 中）
df_val = pd.read_csv(val_path)
val_pos_df = df_val[df_val['label'] == 1]

# 合并所有已知正样本作为“禁区”
existing_pos = pd.concat([train_pos_df, val_pos_df, test_pos_final])
global_ban_set = set(zip(
    existing_pos['relation'],
    existing_pos[['x_index', 'y_index']].min(axis=1),
    existing_pos[['x_index', 'y_index']].max(axis=1)
))

# 候选池：所有疾病 ID
candidate_diseases = np.array(list(valid_disease_ids))
pool_size = len(candidate_diseases)

neg_rows = []
src_drugs = test_pos_final['x_index'].values
relations = test_pos_final['relation'].values

# 预随机采样
rand_indices = rng.integers(0, pool_size, size=len(src_drugs))
potential_diseases = candidate_diseases[rand_indices]

for i in tqdm(range(len(src_drugs)), desc="Negative Sampling"):
    h = src_drugs[i]  # Drug
    r = relations[i]
    t_fake = potential_diseases[i]  # Disease

    # 冲突检查
    check_u, check_v = min(h, t_fake), max(h, t_fake)

    retry = 0
    # 只要这个组合在 全局禁区 中，就重采
    while (check_u == check_v or (r, check_u, check_v) in global_ban_set) and retry < 50:
        t_fake = candidate_diseases[rng.integers(0, pool_size)]
        check_u, check_v = min(h, t_fake), max(h, t_fake)
        retry += 1

    if retry < 50:
        neg_rows.append({'relation': r, 'x_index': h, 'y_index': t_fake, 'label': 0})
    else:
        # 极罕见情况：该药物几乎治愈了所有已知疾病，无法采样负样本
        pass

test_neg = pd.DataFrame(neg_rows)

# ==========================================
# 6. 保存结果
# ==========================================
print("\n>>> Step 4: Saving...")
# 1:1 混合
test_final = pd.concat([test_pos_final, test_neg], ignore_index=True)
test_final = test_final.sample(frac=1, random_state=seed).reset_index(drop=True)

test_final.to_csv(output_path, index=False)
print(f"🎉 Success! Test set saved to: {output_path}")
print(f"   Positives: {len(test_pos_final)}")
print(f"   Negatives: {len(test_neg)}")
print(f"   Total: {len(test_final)}")