import pandas as pd
import os

# ================= 配置 =================
save_dir = "../../data/benchmark/PrimeKG/"
val_path = os.path.join(save_dir, "val_edges.csv")

# ================= 1. 读取 Validation 数据 =================
print(">>> Loading validation data...")
if not os.path.exists(val_path):
    print(f"错误: 找不到文件 {val_path}，请确保上一段切分脚本已执行完毕。")
else:
    val_df = pd.read_csv(val_path)
    print(f"Validation Set Total Size: {len(val_df)}")

    # ================= 2. 识别 Indication 关系 =================
    # PrimeKG 中表示适应症的关系通常是 'indication'
    # 注意：我们只关心 label=1 的正样本，因为负样本（label=0）是随机生成的假关系

    target_relation = 'indication'

    # 筛选条件：关系是 indication 且 label 是 1 (真实存在的边)
    indication_mask = (val_df['relation'] == target_relation) & (val_df['label'] == 1)

    indication_test_df = val_df[indication_mask].copy()

    # 同时查看一下整个 val 集中各类关系的分布（只看正样本）
    print("\n>>> Relation Distribution in Validation Set (Positive Samples Only):")
    print(val_df[val_df['label'] == 1]['relation'].value_counts())

    # ================= 3. 统计结果 =================
    count = len(indication_test_df)
    print(f"\n✅ 统计完成！")
    print(f"Validation 集中 '{target_relation}' (Label=1) 的数量为: {count}")

    if count > 0:
        # ================= 4. 提取并保存 =================
        #保存为单独的测试集
        output_path = os.path.join(save_dir, "val_indication.csv")
        indication_test_df.to_csv(output_path, index=False)
        print(f"已将这 {count} 条 Indication 数据单独保存至: {output_path}")