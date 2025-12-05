import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ==========================================
# 1. 配置区域
# ==========================================
input_clean_path = "../data/benchmark/Kaggle_drug_repositioning/full_mapping_without_na.csv"
train_path = "../data/benchmark/PrimeKG/train_edges.csv"
val_path = "../data/benchmark/PrimeKG/val_edges.csv"
nodes_path = "../data/benchmark/PrimeKG/nodes.csv"
save_dir = "../data/benchmark/Kaggle_drug_repositioning"
os.makedirs(save_dir, exist_ok=True)

target_relation = 'indication'
num_test_samples = 500
seed = 42
rng = np.random.default_rng(seed)

# ==========================================
# 2. 读取数据与预检查
# ==========================================
print(">>> Step 0: Loading data...")
df_test = pd.read_csv(input_clean_path)
df_test['x_index'] = df_test['x_index'].astype(int)
df_test['y_index'] = df_test['y_index'].astype(int)

# 构造初始测试集候选池
test_pos_all = pd.DataFrame({
    'relation': target_relation,
    'x_index': df_test['x_index'],
    'y_index': df_test['y_index'],
    'label': 1
})

try:
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
except FileNotFoundError:
    print("❌ Error: Train or Val file not found. Please run split script first.")
    exit(1)

# ==========================================
# 3. 构建全图正样本禁忌表 (Global Ban Set)
# ==========================================
print("\n>>> Step 1: Building Global Ban Set (Undirected)...")
existing_pos = pd.concat([
    df_train[df_train['label'] == 1],
    df_val[df_val['label'] == 1]
])

train_val_set = set(zip(
    existing_pos['relation'],
    existing_pos[['x_index', 'y_index']].min(axis=1),
    existing_pos[['x_index', 'y_index']].max(axis=1)
))
print(f"Existing known edges (Train+Val): {len(train_val_set)}")

# ==========================================
# 4. 防泄漏清洗
# ==========================================
print("\n>>> Step 2: Cleaning Test Candidates...")
valid_rows = []
leak_count = 0
for _, row in tqdm(test_pos_all.iterrows(), total=len(test_pos_all), desc="Checking leakage"):
    check_tuple = (row['relation'], min(row['x_index'], row['y_index']), max(row['x_index'], row['y_index']))
    if check_tuple in train_val_set:
        leak_count += 1
    else:
        valid_rows.append(row)

test_pos_clean = pd.DataFrame(valid_rows).reset_index(drop=True)
print(f"Removed {leak_count} leaked edges. Clean candidates: {len(test_pos_clean)}")

# ==========================================
# 5. 双向最大覆盖抽样 (正样本)
# ==========================================
print(f"\n>>> Step 3: Sampling {num_test_samples} positives...")
# ... (此处逻辑不变，省略中间细节以节省篇幅，直接用结果) ...
if len(test_pos_clean) <= num_test_samples:
    test_pos_sampled = test_pos_clean
else:
    pool = test_pos_clean.copy().sample(frac=1, random_state=seed).reset_index(drop=True)
    final_selection = []
    covered_drugs = set();
    covered_diseases = set()
    turn = 0
    pbar = tqdm(total=num_test_samples, desc="Sampling")

    while len(final_selection) < num_test_samples and not pool.empty:
        found = False
        remove_list = []
        for idx, row in pool.iterrows():
            if len(final_selection) >= num_test_samples: break
            drug, disease = row['x_index'], row['y_index']
            is_new_drug = drug not in covered_drugs
            is_new_disease = disease not in covered_diseases
            should = False
            if turn == 0:
                should = is_new_drug or (is_new_disease and len(final_selection) < num_test_samples * 0.9)
            else:
                should = is_new_disease or (is_new_drug and len(final_selection) < num_test_samples * 0.9)

            if should:
                final_selection.append(row)
                covered_drugs.add(drug);
                covered_diseases.add(disease)
                remove_list.append(idx)
                pbar.update(1);
                found = True;
                turn = 1 - turn
        if remove_list: pool = pool.drop(remove_list)
        if not found:  # 填补剩余
            rem = num_test_samples - len(final_selection)
            if rem > 0:
                fill = pool.sample(n=rem, random_state=seed)
                for _, r in fill.iterrows(): final_selection.append(r); pbar.update(1)
            break
    pbar.close()
    test_pos_sampled = pd.DataFrame(final_selection).reset_index(drop=True)

# ==========================================
# 6. 生成 Hard Negatives (类型约束版!)
# ==========================================
print("\n>>> Step 4: Generating Type-Constrained Hard Negatives...")

# 【新增逻辑】构建候选池：只允许 "indication" 关系在训练集中连过的节点 (即所有疾病)
# 注意：这里我们利用训练集的数据分布来获知哪些节点是疾病
print("   > Building candidate pool for 'indication'...")
indication_edges = existing_pos[existing_pos['relation'] == target_relation]

# 收集所有出现过的 Disease 节点
# 假设数据是规范的 Drug(x) -> Disease(y)，那么 y 列就是所有的疾病
# 为了保险，我们把 x 和 y 都算上，再依赖图谱本身的二分特性
candidates = set(indication_edges['y_index'].values) | set(indication_edges['x_index'].values)
candidate_pool = np.array(list(candidates))
print(f"   > Indication target pool size: {len(candidate_pool)} (Likely Diseases)")

# 更新 Global Ban
test_candidates_set = set(zip(
    test_pos_clean['relation'],
    test_pos_clean[['x_index', 'y_index']].min(axis=1),
    test_pos_clean[['x_index', 'y_index']].max(axis=1)
))
global_ban_set = train_val_set.union(test_candidates_set)

neg_rows = []
src_arr = test_pos_sampled['x_index'].values
rel_arr = test_pos_sampled['relation'].values

# 预采样：只从 candidate_pool 里选！
pool_size = len(candidate_pool)
rand_indices = rng.integers(0, pool_size, size=len(src_arr))
b_neg_dst = candidate_pool[rand_indices]

for j in tqdm(range(len(src_arr)), desc="Negative Sampling"):
    h, t_fake, r = src_arr[j], b_neg_dst[j], rel_arr[j]
    check_u, check_v = min(h, t_fake), max(h, t_fake)

    # 冲突检查
    retry = 0
    while (check_u == check_v or (r, check_u, check_v) in global_ban_set) and retry < 10:
        # 重采：必须从 candidate_pool 里选
        t_fake = candidate_pool[rng.integers(0, pool_size)]
        check_u, check_v = min(h, t_fake), max(h, t_fake)
        retry += 1

    if retry < 10:
        neg_rows.append([r, h, t_fake, 0])

test_neg = pd.DataFrame(neg_rows, columns=['relation', 'x_index', 'y_index', 'label'])

# ==========================================
# 7. 保存
# ==========================================
print("\n>>> Step 5: Saving...")
test_final = pd.concat([test_pos_sampled, test_neg], ignore_index=True)
test_final = test_final.sample(frac=1, random_state=seed).reset_index(drop=True)
output_path = os.path.join(save_dir, "test.csv")
test_final.to_csv(output_path, index=False)
print(f"🎉 Success! Type-Constrained Test set saved to {output_path}")
print(f"   Positives: {len(test_pos_sampled)}, Negatives: {len(test_neg)}")