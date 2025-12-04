import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ==========================================
# 1. 配置区域
# ==========================================
# 原始测试集路径 (除去 NA 后的 mapping 文件)
input_clean_path = "../data/benchmark/Kaggle_drug_repositioning/full_mapping_without_na.csv"

# 训练集和验证集路径 (用于防泄漏检查)
train_path = "../data/benchmark/PrimeKG/train_edges.csv"
val_path = "../data/benchmark/PrimeKG/val_edges.csv"

# 节点表 (用于获取总节点数)
nodes_path = "../data/benchmark/PrimeKG/nodes.csv"

# 输出目录
save_dir = "../data/benchmark/Kaggle_drug_repositioning"
os.makedirs(save_dir, exist_ok=True)

# 目标关系类型 (PrimeKG 中药物治疗疾病的关系通常叫 indication)
target_relation = 'indication'

# 抽样参数
num_test_samples = 500  # 正样本数量 (负样本会自动生成相同数量)
seed = 42  # 固定随机种子
rng = np.random.default_rng(seed)

# ==========================================
# 2. 预检查
# ==========================================
print(">>> Step 0: Checking prerequisites...")
try:
    df_train = pd.read_csv(train_path)
    if target_relation not in df_train['relation'].unique():
        print(f"❌ 错误: 训练集中不存在关系 '{target_relation}'。请检查关系名称。")
        exit(1)
    print(f"✅ 确认: 训练集中包含 '{target_relation}'")
except FileNotFoundError:
    print(f"❌ 错误: 找不到训练集文件 {train_path}")
    exit(1)

# ==========================================
# 3. 读取并标准化测试集
# ==========================================
print("\n>>> Step 1: Loading test mapping...")
df_test = pd.read_csv(input_clean_path)
# 确保 ID 为整数
df_test['x_index'] = df_test['x_index'].astype(int)
df_test['y_index'] = df_test['y_index'].astype(int)

# 构造初始正样本池
test_pos_all = pd.DataFrame({
    'relation': target_relation,
    'x_index': df_test['x_index'],
    'y_index': df_test['y_index'],
    'label': 1
})
print(f"Original test pairs loaded: {len(test_pos_all)}")

# ==========================================
# 4. 防泄漏清洗 (Leakage Removal)
# ==========================================
print("\n>>> Step 2: Removing edges that exist in Train/Val...")
try:
    df_val = pd.read_csv(val_path)
except FileNotFoundError:
    print("❌ 找不到验证集")
    exit(1)

# 合并训练和验证集的正样本，构建“已存在边”的集合
existing_pos = pd.concat([
    df_train[df_train['label'] == 1],
    df_val[df_val['label'] == 1]
])

# 使用 set 存储 (relation, min_id, max_id) 以处理潜在的无向性或方向混淆
existing_set = set(zip(
    existing_pos['relation'],
    existing_pos[['x_index', 'y_index']].min(axis=1),
    existing_pos[['x_index', 'y_index']].max(axis=1)
))

# 过滤测试集
valid_rows = []
leak_count = 0
for _, row in tqdm(test_pos_all.iterrows(), total=len(test_pos_all), desc="Checking leakage"):
    # 检查当前测试边是否已存在
    check_tuple = (row['relation'], min(row['x_index'], row['y_index']), max(row['x_index'], row['y_index']))
    if check_tuple in existing_set:
        leak_count += 1
    else:
        valid_rows.append(row)

test_pos_clean = pd.DataFrame(valid_rows).reset_index(drop=True)
print(f"Removed {leak_count} leaked edges.")
print(f"Clean test candidates available: {len(test_pos_clean)}")

# ==========================================
# 5. 双向最大覆盖抽样 (Bi-directional Sampling)
# ==========================================
print(f"\n>>> Step 3: Sampling {num_test_samples} positives (Maximizing Drug & Disease Diversity)...")

if len(test_pos_clean) <= num_test_samples:
    print(f"⚠️ 可用样本不足 {num_test_samples}，使用全部数据。")
    test_pos_sampled = test_pos_clean
else:
    # 准备工作
    pool = test_pos_clean.copy()
    # 先打乱池子，保证随机性
    pool = pool.sample(frac=1, random_state=seed).reset_index(drop=True)

    final_selection = []
    covered_drugs = set()
    covered_diseases = set()

    # 贪心策略：交替寻找能带来“新药物”或“新疾病”覆盖的样本
    turn = 0  # 0: 优先找新 Drug, 1: 优先找新 Disease

    # 进度条
    pbar = tqdm(total=num_test_samples, desc="Sampling")

    while len(final_selection) < num_test_samples and not pool.empty:

        found_in_this_scan = False
        rows_to_remove = []

        # 遍历当前池子
        for idx, row in pool.iterrows():
            if len(final_selection) >= num_test_samples:
                break

            drug = row['x_index']
            disease = row['y_index']

            is_new_drug = drug not in covered_drugs
            is_new_disease = disease not in covered_diseases

            should_pick = False

            # 决策逻辑
            if turn == 0:  # 轮到 Drug 回合
                if is_new_drug:
                    should_pick = True
                elif is_new_disease and len(final_selection) < num_test_samples * 0.9:
                    # 如果没新药了，有新病也行，但留点余地
                    should_pick = True
            else:  # 轮到 Disease 回合
                if is_new_disease:
                    should_pick = True
                elif is_new_drug and len(final_selection) < num_test_samples * 0.9:
                    should_pick = True

            if should_pick:
                final_selection.append(row)
                covered_drugs.add(drug)
                covered_diseases.add(disease)
                rows_to_remove.append(idx)
                pbar.update(1)
                found_in_this_scan = True

                # 切换回合
                turn = 1 - turn

        # 从池中移除已选
        if rows_to_remove:
            pool = pool.drop(rows_to_remove)

        # 如果一整轮扫描都没找到能增加覆盖率的样本（说明剩下的全是旧药旧病）
        # 直接随机填充剩余名额
        if not found_in_this_scan:
            remaining_cnt = num_test_samples - len(final_selection)
            if remaining_cnt > 0:
                # print(f"Coverage saturated. Randomly filling {remaining_cnt}...")
                random_fill = pool.sample(n=remaining_cnt, random_state=seed)
                for _, row in random_fill.iterrows():
                    final_selection.append(row)
                    pbar.update(1)
            break

    pbar.close()
    test_pos_sampled = pd.DataFrame(final_selection).reset_index(drop=True)

    # 打印覆盖统计
    n_drugs = test_pos_sampled['x_index'].nunique()
    n_diseases = test_pos_sampled['y_index'].nunique()
    print(f"✅ 抽样完成。覆盖统计: {n_drugs} 种药物, {n_diseases} 种疾病。")

# ==========================================
# 6. 生成 Hard Negatives
# ==========================================
print("\n>>> Step 4: Generating Hard Negatives...")
nodes_df = pd.read_csv(nodes_path)
num_nodes = int(nodes_df['node_index'].max() + 1)

# 构建全局禁忌表 (Global Ban List)
# 包含: 训练集 + 验证集 + 测试集所有候选正例 (不仅仅是抽中的这500个)
# 目的: 防止生成的负例恰好是真实的阳性样本
test_pos_set = set(zip(
    test_pos_clean['relation'],
    test_pos_clean[['x_index', 'y_index']].min(axis=1),
    test_pos_clean[['x_index', 'y_index']].max(axis=1)
))
global_ban_set = existing_set.union(test_pos_set)

neg_src = []
neg_dst = []
neg_rel = []

src_arr = test_pos_sampled['x_index'].values
rel_arr = test_pos_sampled['relation'].values

# 随机生成初始负节点池
b_neg_dst = rng.integers(0, num_nodes, size=len(src_arr))

for j in tqdm(range(len(src_arr)), desc="Negative Sampling"):
    h, t_fake, r = src_arr[j], b_neg_dst[j], rel_arr[j]

    check_u, check_v = min(h, t_fake), max(h, t_fake)

    # 拒绝采样循环
    # 如果生成的 (h, t_fake) 在禁忌表中，或者是自环，就重采
    while check_u == check_v or (r, check_u, check_v) in global_ban_set:
        t_fake = rng.integers(0, num_nodes)
        check_u, check_v = min(h, t_fake), max(h, t_fake)

    neg_src.append(h)
    neg_dst.append(t_fake)
    neg_rel.append(r)

test_neg = pd.DataFrame({
    'relation': neg_rel,
    'x_index': neg_src,
    'y_index': neg_dst,
    'label': 0
})

# ==========================================
# 7. 合并与保存
# ==========================================
print("\n>>> Step 5: Saving final dataset...")

# 合并正负样本
test_final = pd.concat([test_pos_sampled, test_neg], ignore_index=True)

# 最终打乱
test_final = test_final.sample(frac=1, random_state=seed).reset_index(drop=True)

output_filename = "test.csv"
output_path = os.path.join(save_dir, output_filename)

test_final.to_csv(output_path, index=False)

print(f"🎉 成功! 文件已保存至: {output_path}")
print(f"   总样本数: {len(test_final)}")
print(f"   正样本 (Label=1): {len(test_pos_sampled)}")
print(f"   负样本 (Label=0): {len(test_neg)}")
print("Done.")