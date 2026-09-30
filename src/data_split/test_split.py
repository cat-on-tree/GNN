import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ==========================================
# 1. 配置区域
# ==========================================
input_clean_path = "../../data/benchmark/Kaggle_drug_repositioning/full_mapping_without_na.csv"
input_raw_full_path = "../../data/benchmark/Kaggle_drug_repositioning/full_mapping.csv"  # 🔥 新增：用于Ban Set的全量数据
train_path = "../../data/benchmark/PrimeKG/train_edges.csv"
val_path = "../../data/benchmark/PrimeKG/val_edges.csv"
nodes_path = "../../data/benchmark/PrimeKG/nodes.csv"
save_dir = "../../data/benchmark/Kaggle_drug_repositioning"
os.makedirs(save_dir, exist_ok=True)

target_relation = 'indication'

# 🔥【核心配置】设置样本数量和比例
num_pos_samples = 100  # 正样本数量
neg_pos_ratio = 10  # 负正比例 (1:10)

seed = 42
rng = np.random.default_rng(seed)

# ==========================================
# 2. 读取数据与构建类型白名单
# ==========================================
print(">>> Step 0: Loading data & Node Types...")
df_nodes = pd.read_csv(nodes_path)
valid_drug_ids = set(df_nodes[df_nodes['node_type'] == 'drug']['node_index'].unique())
valid_disease_ids = set(df_nodes[df_nodes['node_type'] == 'disease']['node_index'].unique())

print(f"   Valid Drugs: {len(valid_drug_ids)}")
print(f"   Valid Diseases: {len(valid_disease_ids)}")

df_test = pd.read_csv(input_clean_path)
df_test['x_index'] = df_test['x_index'].astype(int)
df_test['y_index'] = df_test['y_index'].astype(int)

# ==========================================
# 3. 严格清洗正样本 (Strict Filtering)
# ==========================================
print("\n>>> Step 1: Cleaning Positives (Strict Type Check)...")
test_pos_raw = pd.DataFrame({
    'relation': target_relation,
    'x_index': df_test['x_index'],
    'y_index': df_test['y_index'],
    'label': 1
})

is_head_drug = test_pos_raw['x_index'].isin(valid_drug_ids)
is_tail_disease = test_pos_raw['y_index'].isin(valid_disease_ids)
valid_type_mask = is_head_drug & is_tail_disease
test_pos_typed = test_pos_raw[valid_type_mask].copy()

try:
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
except FileNotFoundError:
    print("❌ Error: Train or Val file not found.")
    exit(1)

# 构建基础的 Train + Val 集合用于去重 (Leakage Check)
existing_pos_train_val = pd.concat([
    df_train[df_train['label'] == 1],
    df_val[df_val['label'] == 1]
])

train_val_set = set(zip(
    existing_pos_train_val['relation'],
    existing_pos_train_val[['x_index', 'y_index']].min(axis=1),
    existing_pos_train_val[['x_index', 'y_index']].max(axis=1)
))

# 筛选正样本：必须不在 Train/Val 中
valid_rows = []
leak_count = 0
for _, row in tqdm(test_pos_typed.iterrows(), total=len(test_pos_typed), desc="Checking leakage"):
    check_tuple = (row['relation'], min(row['x_index'], row['y_index']), max(row['x_index'], row['y_index']))
    if check_tuple in train_val_set:
        leak_count += 1
    else:
        valid_rows.append(row)

test_pos_clean = pd.DataFrame(valid_rows).reset_index(drop=True)
print(f"Removed {leak_count} leaked edges. Clean candidates: {len(test_pos_clean)}")

# ==========================================
# 4. 双向最大覆盖抽样 (正样本)
# ==========================================
print(f"\n>>> Step 2: Sampling {num_pos_samples} positives...")
if len(test_pos_clean) <= num_pos_samples:
    test_pos_sampled = test_pos_clean
else:
    pool = test_pos_clean.copy().sample(frac=1, random_state=seed).reset_index(drop=True)
    final_selection = []
    covered_drugs = set()
    covered_diseases = set()
    turn = 0
    pbar = tqdm(total=num_pos_samples, desc="Sampling Positives")

    while len(final_selection) < num_pos_samples and not pool.empty:
        found = False
        remove_list = []
        for idx, row in pool.iterrows():
            if len(final_selection) >= num_pos_samples: break
            drug, disease = row['x_index'], row['y_index']
            is_new_drug = drug not in covered_drugs
            is_new_disease = disease not in covered_diseases
            should = False
            if turn == 0:
                should = is_new_drug or (is_new_disease and len(final_selection) < num_pos_samples * 0.9)
            else:
                should = is_new_disease or (is_new_drug and len(final_selection) < num_pos_samples * 0.9)

            if should:
                final_selection.append(row)
                covered_drugs.add(drug);
                covered_diseases.add(disease)
                remove_list.append(idx)
                pbar.update(1);
                found = True;
                turn = 1 - turn
        if remove_list: pool = pool.drop(remove_list)
        if not found:
            rem = num_pos_samples - len(final_selection)
            if rem > 0:
                fill = pool.sample(n=rem, random_state=seed)
                for _, r in fill.iterrows(): final_selection.append(r); pbar.update(1)
            break
    pbar.close()
    test_pos_sampled = pd.DataFrame(final_selection).reset_index(drop=True)

# ==========================================
# 5. 生成比例失衡的负样本 (Imbalanced Negatives)
# ==========================================
num_neg_target = len(test_pos_sampled) * neg_pos_ratio
print(f"\n>>> Step 3: Generating {num_neg_target} Negatives (Ratio 1:{neg_pos_ratio})...")

candidate_pool = np.array(list(valid_disease_ids))
pool_size = len(candidate_pool)

# 🔥【核心修改：构建超级禁忌表 Global Ban Set】
print("   Building comprehensive Global Ban Set (Train + Val + RAW Full Mapping)...")

# 1. 基础：Train + Val
# (已经有了 train_val_set)

# 2. 扩充：Full Mapping (RAW) - 包含那些有缺失值但有Index的行
df_raw = pd.read_csv(input_raw_full_path)
# 只保留有 index 的行
df_raw_valid = df_raw.dropna(subset=['x_index', 'y_index']).copy()
df_raw_valid['x_index'] = df_raw_valid['x_index'].astype(int)
df_raw_valid['y_index'] = df_raw_valid['y_index'].astype(int)

# 构建全量 Set (忽略 relation，只看连边)
# 使用 (min, max) 确保无向匹配
full_mapping_set = set(zip(
    df_raw_valid[['x_index', 'y_index']].min(axis=1),
    df_raw_valid[['x_index', 'y_index']].max(axis=1)
))

# 3. 合并：Global Ban = Train/Val Set (带relation) + Full Raw Set (只看边)
# 为了方便查重，我们统一逻辑：负采样时，不仅要查 relation，更要查这两个点是否在 Full Mapping 里连过
# 因此，最严格的 ban set 应该是一个只包含 (u, v) 的集合

strict_ban_pairs = set()

# 添加 Train/Val
for _, row in existing_pos_train_val.iterrows():
    u, v = int(row['x_index']), int(row['y_index'])
    strict_ban_pairs.add((min(u, v), max(u, v)))

# 添加 Full Raw
strict_ban_pairs.update(full_mapping_set)

print(f"   Total unique forbidden pairs: {len(strict_ban_pairs)}")

neg_rows = []
src_arr = test_pos_sampled['x_index'].values
rel_arr = test_pos_sampled['relation'].values

# 生成负样本
for i in tqdm(range(len(src_arr)), desc="Negative Sampling"):
    h = src_arr[i]
    r = rel_arr[i]

    # 对这一个正样本，生成 neg_pos_ratio 个负样本
    for _ in range(neg_pos_ratio):

        # 尝试生成
        retry = 0
        found = False
        while retry < 50:
            t_fake = candidate_pool[rng.integers(0, pool_size)]

            # 检查1: 不能自环
            if h == t_fake:
                retry += 1
                continue

            # 检查2: 是否在 Global Ban Set 中
            check_pair = (min(h, t_fake), max(h, t_fake))

            if check_pair not in strict_ban_pairs:
                neg_rows.append([r, h, t_fake, 0])
                found = True
                break

            retry += 1

        if not found:
            # 极罕见情况：找不到负样本
            pass

test_neg = pd.DataFrame(neg_rows, columns=['relation', 'x_index', 'y_index', 'label'])

# ==========================================
# 6. 保存
# ==========================================
print("\n>>> Step 4: Saving...")
test_final = pd.concat([test_pos_sampled, test_neg], ignore_index=True)
test_final = test_final.sample(frac=1, random_state=seed).reset_index(drop=True)

filename = "test.csv"
output_path = os.path.join(save_dir, filename)

test_final.to_csv(output_path, index=False)
print(f"🎉 Success! Saved to {output_path}")
print(f"   Positives: {len(test_pos_sampled)}")
print(f"   Negatives: {len(test_neg)}")
print(f"   Actual Ratio: 1 : {len(test_neg) / len(test_pos_sampled):.2f}")
print(f"   Ban Set Source: Train + Val + Full Raw Mapping (Strict Mode)")