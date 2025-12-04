import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# ========== 1. 路径与配置 ==========
nodes_path = "../data/benchmark/PrimeKG/nodes.csv"
train_path = "../data/benchmark/PrimeKG/train_edges.csv"
val_path = "../data/benchmark/PrimeKG/val_edges.csv"

model_dir = "../model/RGCN"
plot_dir = "../data/training_process/RGCN"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

best_model_path = os.path.join(model_dir, "rgcn_best_weights.pt")
final_model_path = os.path.join(model_dir, "rgcn_final_weights.pt")
train_val_curve_csv = os.path.join(plot_dir, "train_val_curve.csv")
val_curve_svg = os.path.join(plot_dir, "val_curve.svg")
loss_curve_svg = os.path.join(plot_dir, "train_val_loss_curve.svg")

# --- 性能优化参数 ---
batch_size = 4096
num_neighbors = [20, 10]  # 采样邻居数
hidden_dim = 128
max_epochs = 100
patience = 10
lr = 0.001
num_workers = 0  # Windows 上建议先设为 0，Linux 可以设为 4 或 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 2. 数据加载 ==========
print("Loading data...")
nodes_df = pd.read_csv(nodes_path)
num_nodes = nodes_df['node_index'].max() + 1

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

relation_types = sorted(train_df['relation'].unique().tolist())
rel2id = {r: i for i, r in enumerate(relation_types)}
num_rels = len(rel2id)

# --- 构建 PyG Data 对象 ---
print("Constructing Graph Data...")
# 显式确保 num_nodes 是 int 类型
num_nodes = int(nodes_df['node_index'].max() + 1)

pos_train_df = train_df[train_df['label'] == 1]
edge_index = torch.stack([
    torch.tensor(pos_train_df['x_index'].values, dtype=torch.long),
    torch.tensor(pos_train_df['y_index'].values, dtype=torch.long)
], dim=0)
edge_type = torch.tensor([rel2id[r] for r in pos_train_df['relation'].values], dtype=torch.long)

data = Data(
    edge_index=edge_index,
    edge_type=edge_type,
    num_nodes=num_nodes,
    x=torch.arange(num_nodes, dtype=torch.long)
)

# 验证数据对象完整性 (可选)
# data.validate(raise_on_error=True)

# --- 定义用于训练的边 ---
print("Preparing Training Loader...")
train_edge_label_index = torch.stack([
    torch.tensor(train_df['x_index'].values, dtype=torch.long),
    torch.tensor(train_df['y_index'].values, dtype=torch.long)
], dim=0)
train_edge_label = torch.tensor(train_df['label'].values, dtype=torch.float)

train_rel_tensor = torch.tensor([rel2id[r] for r in train_df['relation'].values], dtype=torch.long)

# 训练加载器
train_loader = LinkNeighborLoader(
    data,
    num_neighbors=num_neighbors,
    edge_label_index=train_edge_label_index,
    edge_label=train_edge_label,
    batch_size=batch_size,
    shuffle=True,
    neg_sampling_ratio=0,
    num_workers=num_workers
)

# 验证集加载器 (同样加速)
print("Preparing Validation Loader...")
val_edge_label_index = torch.stack([
    torch.tensor(val_df['x_index'].values, dtype=torch.long),
    torch.tensor(val_df['y_index'].values, dtype=torch.long)
], dim=0)
val_edge_label = torch.tensor(val_df['label'].values, dtype=torch.float)
val_rel_tensor = torch.tensor([rel2id.get(r, 0) for r in val_df['relation'].values], dtype=torch.long)

val_loader = LinkNeighborLoader(
    data,
    num_neighbors=num_neighbors,
    edge_label_index=val_edge_label_index,
    edge_label=val_edge_label,
    batch_size=batch_size,
    shuffle=False,
    neg_sampling_ratio=0
)


# ========== 3. 模型定义 ==========
class RGCN_DistMult(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_rel, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.relation_embedding = nn.Embedding(num_rel, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_rel))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_rel))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_rel))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_type):
        # x 是子图中节点的全局 ID
        h = self.embedding(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, edge_type)
            h = bn(h)
            h = torch.relu(h)
            h = torch.dropout(h, p=self.dropout, train=self.training)
        return h

    def score(self, node_emb, src, tgt, rel):
        return (node_emb[src] * self.relation_embedding(rel) * node_emb[tgt]).sum(dim=1)


model = RGCN_DistMult(num_nodes, hidden_dim, num_rels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

# ========== 4. 极速训练流程 ==========
best_val_auc = 0
patience_counter = 0
train_loss_hist, val_loss_hist, val_auc_hist = [], [], []

print("Starting Fast Training...")

for epoch in range(max_epochs):
    model.train()
    total_loss = 0
    total_examples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")

    for batch in pbar:
        batch = batch.to(device)

        # batch.n_id: 子图中包含的所有节点的全局 ID
        # batch.edge_index: 子图的边结构 (Local ID)
        # batch.edge_label_index: 我们要预测的那批边的 Local ID (Src, Tgt)
        # batch.input_id: 这批样本在原始 train_df 中的索引 (用于查 Relation)

        # 1. 计算子图 Embedding
        # 注意：这里传入 batch.n_id 给 embedding layer 取数
        h = model(batch.n_id, batch.edge_index, batch.edge_type)

        # 2. 获取当前 Batch 边的信息
        # LinkNeighborLoader 已经帮我们把 edge_label_index 映射为 Local ID 了
        # 所以直接用 h[src_local] 即可，无需手动 map
        src_local = batch.edge_label_index[0]
        tgt_local = batch.edge_label_index[1]

        # 获取 Relation (需要回到 CPU 查原始 Tensor，或者把 train_rel_tensor 放到 GPU)
        # 由于 input_id 在 GPU 上，为了速度，我们可以把 train_rel_tensor 放在 CPU
        # 只有当前 batch 需要的才搬运
        batch_input_ids = batch.input_id.cpu()
        batch_rel = train_rel_tensor[batch_input_ids].to(device)
        batch_lbl = batch.edge_label.to(device)

        # 3. 计算 Loss
        scores = model.score(h, src_local, tgt_local, batch_rel)
        loss = criterion(scores, batch_lbl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_samples = batch.edge_label.numel()
        total_loss += loss.item() * num_samples
        total_examples += num_samples

    avg_train_loss = total_loss / total_examples

    # --- 验证 ---
    model.eval()
    val_preds = []
    val_targets = []
    val_loss_sum = 0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            h = model(batch.n_id, batch.edge_index, batch.edge_type)

            src_local = batch.edge_label_index[0]
            tgt_local = batch.edge_label_index[1]

            batch_input_ids = batch.input_id.cpu()
            batch_rel = val_rel_tensor[batch_input_ids].to(device)
            batch_lbl = batch.edge_label.to(device)

            scores = model.score(h, src_local, tgt_local, batch_rel)
            val_loss_sum += criterion(scores, batch_lbl).item()

            val_preds.append(torch.sigmoid(scores).cpu().numpy())
            val_targets.append(batch_lbl.cpu().numpy())
            val_batches += 1

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_auc = roc_auc_score(val_targets, val_preds)
    avg_val_loss = val_loss_sum / val_batches

    print(f"Epoch {epoch + 1}: Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f}")

    train_loss_hist.append(avg_train_loss)
    val_loss_hist.append(avg_val_loss)
    val_auc_hist.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping")
        break

# 保存图表 (代码同前)
curve_df = pd.DataFrame({
    "epoch": list(range(1, len(train_loss_hist) + 1)),
    "train_loss": train_loss_hist,
    "val_loss": val_loss_hist,
    "val_auc": val_auc_hist
})
curve_df.to_csv(train_val_curve_csv, index=False)
plt.figure(figsize=(6, 4))
plt.plot(curve_df["epoch"], curve_df["val_auc"], label="Val AUC", c='C1')
plt.savefig(val_curve_svg)
plt.figure(figsize=(6, 4))
plt.plot(curve_df["epoch"], curve_df["train_loss"], label="Train Loss", c='C0')
plt.plot(curve_df["epoch"], curve_df["val_loss"], label="Val Loss", c='C2')
plt.savefig(loss_curve_svg)