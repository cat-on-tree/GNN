import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

# -------- 配置路径 ---------
test_path = "../data/benchmark/valmapping.csv"  # 测试集路径
model_path = "../model/RGCN/rgcn_best_weights.pt"
node_map_path = "../model/RGCN/node_map.pkl"
encode_save_path = "../model/RGCN/graph_encoding.pkl"
output_pred_csv = "../data/evaluation/RGCN/test_pred_results.csv"
roc_svg = "../data/evaluation/RGCN/test_roc_curve.svg"
unseen_nodes_file = "../data/evaluation/RGCN/unseen_names.txt"

# --------- 加载编码与模型 ----------
with open(encode_save_path, "rb") as f:
    encoding = pickle.load(f)
node_map = encoding["node_map"]
edge_index = torch.tensor(encoding["edge_list"], dtype=torch.long).t()
edge_type = torch.tensor(encoding["edge_type_list"], dtype=torch.long)
rel2id = encoding["rel2id"]
num_nodes = len(node_map)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_rel, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_rel))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_rel))
        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_rel))
    def forward(self, node_idx, edge_index, edge_type):
        x = self.embedding(node_idx)
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = torch.relu(x)
        return x

# --------- 加载模型权重 ----------
model = RGCN(num_nodes=num_nodes, hidden_dim=64, num_rel=len(rel2id)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --------- 构建所有节点embedding ---------
edge_index = edge_index.to(device)
edge_type = edge_type.to(device)
with torch.no_grad():
    node_indices = torch.arange(num_nodes, dtype=torch.long, device=device)
    node_emb = model(node_indices, edge_index, edge_type)

# --------- 处理测试集与预测 ---------
test_df = pd.read_csv(test_path, dtype=str).fillna('')
results = []
unseen_names = set()

for idx, row in test_df.iterrows():
    drug_name = row["DrugName"].strip('"').strip()
    disease_name = row["DiseaseName"].strip('"').strip()
    drug_key = ("drug", drug_name)
    disease_key = ("disease", disease_name)
    label = int(row["label"]) if "label" in row else None

    # 节点名称标准化处理
    drug_idx = node_map.get(drug_key, None)
    disease_idx = node_map.get(disease_key, None)
    record = {
        "DrugID": row["DrugID"],
        "DrugName": drug_name,
        "DiseaseID": row["DiseaseID"],
        "DiseaseName": disease_name,
        "label": label
    }
    if drug_idx is None or disease_idx is None:
        record["prob"] = np.nan
        missing_note = []
        if drug_idx is None:
            missing_note.append(f"Drug: {drug_name}")
        if disease_idx is None:
            missing_note.append(f"Disease: {disease_name}")
        record["note"] = "; ".join(missing_note)
        unseen_names.update(missing_note)
    else:
        prob = torch.sigmoid(torch.dot(node_emb[drug_idx], node_emb[disease_idx])).item()
        record["prob"] = prob
        record["note"] = ""
    results.append(record)

# --------- 统计并输出未见节点 ---------
with open(unseen_nodes_file, "w", encoding="utf-8") as f:
    f.write("以下节点在训练集未出现，测试行已忽略：\n")
    for name in sorted(unseen_names):
        f.write(name + "\n")
print(f"未见于训练集的Name节点已存为: {unseen_nodes_file}")

# --------- 输出预测结果及评估 ---------
out_df = pd.DataFrame(results)
out_df.to_csv(output_pred_csv, index=False)
print(f"测试集预测结果已输出至: {output_pred_csv}")

# --------- 计算AUROC与绘制ROC曲线 ---------
val_rows = out_df.dropna(subset=["prob"])  # 仅有分数的才评估
if "label" in val_rows and val_rows["label"].nunique() == 2:
    y_true = val_rows["label"].astype(int)
    y_score = val_rows["prob"].astype(float)
    auroc = roc_auc_score(y_true, y_score)
    print(f"Test AUROC (忽略未见节点): {auroc:.4f}")

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color='C0', label=f'ROC curve (AUC={auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='C1', linestyle='--', label="Random guess")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on Test Set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_svg)
    print(f"ROC曲线已保存到: {roc_svg}")
else:
    print("Warning: 测试集label标签不满足两类或无可评估分数，无法计算AUROC和ROC曲线。")