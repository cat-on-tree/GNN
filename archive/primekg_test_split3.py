import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ==========================================
# 1. 配置区域
# ==========================================
save_dir = "../../data/benchmark/PrimeKG/"
input_candidates_path = os.path.join(save_dir, "val_indication.csv")
train_path = os.path.join(save_dir, "train_edges.csv")
nodes_path = os.path.join(save_dir, "nodes.csv")
output_path = os.path.join(save_dir, "test_cold.csv")

# 【核心配置】
DRUG_DEGREE_THRESHOLD = 50
DISEASE_DEGREE_THRESHOLD = 50

seed = 42
rng = np.random.default_rng(seed)

# ==========================================
# 2. 数据加载与预处理
# ==========================================
print(f"🚀 Generating Mixed Cold-Start Test Set (Threshold <= {DRUG_DEGREE_THRESHOLD})...")

df_candidates = pd.read_csv(input_candidates_path)
df_train = pd.read_csv(train_path)
df_nodes = pd.read_csv(nodes_path)

# --- 新增：构建类型查找集合 ---
print(">>> Building Node Type Sets...")
valid_drug_ids = set(df_nodes[df_nodes['node_type'] == 'drug']['node_index'].unique())
valid_disease_ids = set(df_nodes[df_nodes['node_type'] == 'disease']['node_index'].unique())
drug_pool = np.array(list(valid_drug_ids))
disease_pool = np.array(list(valid_disease_ids))

print(f"   Valid Drugs: {len(valid_drug_ids)}")
print(f"   Valid Diseases: {len(valid_disease_ids)}")

print(">>> Calculating Degrees in Train Graph...")
# 统计药物度数 (x_index)
train_drug_counts = df_train['x_index'].value_counts()
# 统计疾病度数 (y_index)
train_disease_counts = df_train['y_index'].value_counts()


# 定义冷门节点集合
def is_cold_drug(node_idx):
    return train_drug_counts.get(node_idx, 0) <= DRUG_DEGREE_THRESHOLD


def is_cold_disease(node_idx):
    return train_disease_counts.get(node_idx, 0) <= DISEASE_DEGREE_THRESHOLD


# ==========================================
# 3. 筛选混合冷启动正样本 (并强制纠正方向)
# ==========================================
print(">>> Filtering Positives (Drug is Cold OR Disease is Cold)...")

cold_pos_rows = []
skipped_invalid_type = 0

for _, row in df_candidates.iterrows():
    u_raw, v_raw = int(row['x_index']), int(row['y_index'])

    # --- 核心修改：强制方向纠正 (Drug -> Disease) ---
    drug_id, disease_id = None, None

    # 情况1: u是药, v是病
    if u_raw in valid_drug_ids and v_raw in valid_disease_ids:
        drug_id, disease_id = u_raw, v_raw
    # 情况2: v是药, u是病 (反向)
    elif v_raw in valid_drug_ids and u_raw in valid_disease_ids:
        drug_id, disease_id = v_raw, u_raw
    else:
        # 类型不对 (比如 Drug-Gene), 跳过
        skipped_invalid_type += 1
        continue

    # 检查冷启动条件
    flag_cold_drug = is_cold_drug(drug_id)
    flag_cold_disease = is_cold_disease(disease_id)

    if flag_cold_drug or flag_cold_disease:
        cold_pos_rows.append({
            'relation': row['relation'],
            'x_index': drug_id,  # 确保 x 是 Drug
            'y_index': disease_id,  # 确保 y 是 Disease
            'label': 1,
            'cold_type': 'drug' if flag_cold_drug else 'disease'
        })

print(f"   Skipped {skipped_invalid_type} rows with invalid types.")
df_cold_pos = pd.DataFrame(cold_pos_rows)
print(f"   Selected {len(df_cold_pos)} mixed cold-start edges (Corrected Direction).")

if len(df_cold_pos) == 0:
    print("❌ No samples found. Please increase threshold.")
    exit()

# ==========================================
# 4. 准备负采样资源
# ==========================================
print(">>> Preparing Ban Set...")
# Global Ban (Train + Val 的正样本)
val_path = os.path.join(save_dir, "val_edges.csv")
df_val = pd.read_csv(val_path)
all_known = pd.concat([df_train[df_train['label'] == 1], df_val[df_val['label'] == 1]])

# 构建 Ban Set 时使用 min-max 标准化，因为查重是不分方向的
global_ban = set(zip(
    all_known['relation'],
    all_known[['x_index', 'y_index']].min(axis=1),
    all_known[['x_index', 'y_index']].max(axis=1)
))

# ==========================================
# 5. 策略性负采样 (Adaptive Negative Sampling)
# ==========================================
print(">>> Generating Negatives (Adaptive Strategy)...")

neg_rows = []
pos_records = df_cold_pos.to_dict('records')

for row in tqdm(pos_records):
    # 这里取出来的 x 已经是 Drug, y 已经是 Disease
    u_drug, v_disease, r = row['x_index'], row['y_index'], row['relation']
    cold_type = row['cold_type']

    retry = 0
    found = False

    while retry < 50:
        if cold_type == 'drug':
            # 策略：替换 Tail (Disease) -> 保持 Drug 不变
            v_fake = rng.choice(disease_pool)

            check_u, check_v = min(u_drug, v_fake), max(u_drug, v_fake)

            # 确保 v_fake 不是原来的病，且这条边不存在
            if v_fake != v_disease and (r, check_u, check_v) not in global_ban:
                neg_rows.append({
                    'relation': r,
                    'x_index': u_drug,  # Drug
                    'y_index': v_fake,  # Fake Disease
                    'label': 0
                })
                found = True
                break

        else:  # cold_type == 'disease'
            # 策略：替换 Head (Drug) -> 保持 Disease 不变
            u_fake = rng.choice(drug_pool)

            check_u, check_v = min(u_fake, v_disease), max(u_fake, v_disease)

            # 确保 u_fake 不是原来的药
            if u_fake != u_drug and (r, check_u, check_v) not in global_ban:
                neg_rows.append({
                    'relation': r,
                    'x_index': u_fake,  # Fake Drug
                    'y_index': v_disease,  # Disease
                    'label': 0
                })
                found = True
                break

        retry += 1

    # 如果极小概率找不到负样本，为了保持平衡，对应的正样本最好也别要了？
    # 这里为了简单起见，如果找不到就不加负样本（会导致正负不平衡）。
    # 或者你可以选择在这里不 break，而是记录 index 后续删除对应的正样本。

df_neg = pd.DataFrame(neg_rows)

# ==========================================
# 6. 保存
# ==========================================
print(">>> Saving...")
# 剔除辅助列 cold_type
df_cold_pos = df_cold_pos.drop(columns=['cold_type'])

# 确保 1:1 (如果有没采样成功的，截断正样本)
min_len = min(len(df_cold_pos), len(df_neg))
final_df = pd.concat([df_cold_pos.iloc[:min_len], df_neg.iloc[:min_len]], ignore_index=True)

# 打乱
final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)

# 强制类型转换
final_df['x_index'] = final_df['x_index'].astype(int)
final_df['y_index'] = final_df['y_index'].astype(int)
final_df['label'] = final_df['label'].astype(int)

final_df.to_csv(output_path, index=False)
print(f"✅ Saved to {output_path}")
print(f"   Positives: {min_len}")
print(f"   Negatives: {min_len}")
print(f"   Total: {len(final_df)}")
print("   NOTE: x_index is GUARANTEED to be Drug, y_index is GUARANTEED to be Disease.")