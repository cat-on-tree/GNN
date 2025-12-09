import pandas as pd
import os

# 1. 定义文件路径
nodes_path = '../../data/benchmark/PrimeKG/nodes.csv'
test_files = ['../../data/benchmark/Kaggle_drug_repositioning/test.csv', '../../data/benchmark/Kaggle_drug_repositioning/test_cold.csv', '../../data/benchmark/Kaggle_drug_repositioning/test_hard.csv']
allowed_types = ['drug', 'disease']  # 定义允许的节点类型

# 2. 加载 nodes.csv
print(f"Loading {nodes_path}...")
try:
    # 只需要 index, type, name 这几列
    nodes_df = pd.read_csv(nodes_path, usecols=['node_index', 'node_type', 'node_name'])
except ValueError:
    # 如果列名不对，尝试读取所有列并打印列名供调试
    nodes_df = pd.read_csv(nodes_path)
    print("Error: 列名不匹配，nodes.csv 的列名为:", nodes_df.columns)
    exit()


# 3. 定义处理函数
def process_test_file(file_path, nodes_df):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}: File not found.")
        return

    print(f"\nProcessing {file_path}...")
    df = pd.read_csv(file_path)
    original_count = len(df)

    # --- 步骤 A: 映射 X 节点 (通常是 Drug) ---
    # 将 df 的 x_index 与 nodes_df 的 node_index 进行左连接
    df = df.merge(nodes_df, left_on='x_index', right_on='node_index', how='left')
    df.rename(columns={'node_name': 'x_name', 'node_type': 'x_type'}, inplace=True)
    df.drop(columns=['node_index'], inplace=True)  # 删除多余的索引列

    # --- 步骤 B: 映射 Y 节点 (通常是 Disease) ---
    # 将 df 的 y_index 与 nodes_df 的 node_index 进行左连接
    df = df.merge(nodes_df, left_on='y_index', right_on='node_index', how='left')
    df.rename(columns={'node_name': 'y_name', 'node_type': 'y_type'}, inplace=True)
    df.drop(columns=['node_index'], inplace=True)

    # --- 步骤 C: 检查是否有 ID 没找到对应的名字 (Mapping Check) ---
    if df['x_name'].isnull().any() or df['y_name'].isnull().any():
        print(f"⚠️ Warning: Some indices in {file_path} could not be found in nodes.csv!")
        # 打印出没找到的行数
        missing_x = df['x_name'].isnull().sum()
        missing_y = df['y_name'].isnull().sum()
        print(f"  - Missing X names: {missing_x}")
        print(f"  - Missing Y names: {missing_y}")

    # --- 步骤 D: 类型校验 (Type Validation) ---
    # 检查 x_type 和 y_type 是否都在 allowed_types 中
    # 注意：这里我们放宽一点，只检查是否是 drug 或 disease，不强制 x 必须是 drug
    # 因为有时候可能是 disease-drug 的反向关系，或者 drug-drug

    valid_mask = (df['x_type'].isin(allowed_types)) & (df['y_type'].isin(allowed_types))
    invalid_rows = df[~valid_mask]

    if not invalid_rows.empty:
        print(f"❌ Validation Failed in {file_path}!")
        print(f"  Found {len(invalid_rows)} rows with non-drug/disease types.")
        print("  Invalid types found:")
        print(invalid_rows[['x_type', 'y_type']].value_counts())

        # 可选：你可以选择删除这些行，或者只是标记。这里我们保留但给出警告。
    else:
        print(f"✅ Validation Passed: All mapped nodes are strictly 'drug' or 'disease'.")

    # --- 步骤 E: 格式化输出 ---
    # 重新排列列，把名字放在前面，方便查看
    output_columns = ['relation', 'x_name', 'y_name', 'label', 'x_type', 'y_type', 'x_index', 'y_index']
    df_final = df[output_columns]

    # 保存为新文件，例如 test_llm.csv
    new_filename = file_path.replace('.csv', '_llm.csv')
    df_final.to_csv(new_filename, index=False)
    print(f"💾 Saved processed file to: {new_filename}")


# 4. 执行处理
for f in test_files:
    process_test_file(f, nodes_df)