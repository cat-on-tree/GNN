import pandas as pd


def standardize_name(name):
    """去除引号、空格并转小写"""
    if pd.isna(name):
        return ""
    return str(name).strip('"').strip().lower()


def extract_names_from_train(train_path):
    train_df = pd.read_csv(train_path, dtype=str)
    drug_names = set()
    disease_names = set()
    for idx, row in train_df.iterrows():
        # 假定训练集具有 'x_type', 'x_name', 'y_type', 'y_name' 格式
        if "x_type" in row and "x_name" in row:
            if row["x_type"].lower() == "drug":
                drug_names.add(standardize_name(row["x_name"]))
            elif row["x_type"].lower() == "disease":
                disease_names.add(standardize_name(row["x_name"]))
        if "y_type" in row and "y_name" in row:
            if row["y_type"].lower() == "drug":
                drug_names.add(standardize_name(row["y_name"]))
            elif row["y_type"].lower() == "disease":
                disease_names.add(standardize_name(row["y_name"]))
    return drug_names, disease_names


def extract_names_from_test(test_path):
    test_df = pd.read_csv(test_path, dtype=str)
    drug_names = set()
    disease_names = set()
    # 假定测试集有 'DrugName', 'DiseaseName'
    for idx, row in test_df.iterrows():
        drug_names.add(standardize_name(row.get("DrugName", "")))
        disease_names.add(standardize_name(row.get("DiseaseName", "")))
    return drug_names, disease_names


def main():
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"
    drug_train, disease_train = extract_names_from_train(train_path)
    drug_test, disease_test = extract_names_from_test(test_path)

    missing_drugs = sorted([name for name in drug_test if name and name not in drug_train])
    missing_diseases = sorted([name for name in disease_test if name and name not in disease_train])

    print(f"测试集drug名总数: {len(drug_test)}，其中未在训练集中出现的drug数: {len(missing_drugs)}")
    print(f"测试集disease名总数: {len(disease_test)}，其中未在训练集中出现的disease数: {len(missing_diseases)}")

    if missing_drugs:
        print("未见于训练集的drug name示例:", missing_drugs[:10])
    if missing_diseases:
        print("未见于训练集的disease name示例:", missing_diseases[:10])

    # 可输出到文件
    with open("../data/missing_drugs.txt", "w", encoding="utf-8") as f:
        for name in missing_drugs:
            f.write(name + "\n")
    with open("../data/missing_diseases.txt", "w", encoding="utf-8") as f:
        for name in missing_diseases:
            f.write(name + "\n")
    print("所有未见drug和disease名字已分别保存到 missing_drugs.txt 和 missing_diseases.txt")


if __name__ == "__main__":
    main()