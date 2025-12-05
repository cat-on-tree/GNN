import pandas as pd
import torch
import os
import sys
import logging
import time
import numpy as np
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve
from tqdm import tqdm


# 配置日志 (如果需要独立调用)
def setup_logger(model_name):
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger()


def load_and_build_data(nodes_path, train_path, val_path, test_path, test_hard_path=None):
    print("Loading data...")
    nodes_df = pd.read_csv(nodes_path)
    num_nodes = int(nodes_df['node_index'].max() + 1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    all_relations = set(train_df['relation'].unique()) | \
                    set(val_df['relation'].unique()) | \
                    set(test_df['relation'].unique())

    test_hard_df = None
    if test_hard_path and os.path.exists(test_hard_path):
        print(f"Found Hard Test Set at: {test_hard_path}")
        test_hard_df = pd.read_csv(test_hard_path)
        all_relations = all_relations | set(test_hard_df['relation'].unique())

    relation_types = sorted(list(all_relations))
    rel2id = {r: i for i, r in enumerate(relation_types)}
    num_rels = len(rel2id)
    print(f"Total unique relations: {num_rels}")

    pos_train_df = train_df[train_df['label'] == 1]
    edge_index = torch.stack([
        torch.tensor(pos_train_df['x_index'].values, dtype=torch.long),
        torch.tensor(pos_train_df['y_index'].values, dtype=torch.long)
    ], dim=0)
    edge_type = torch.tensor([rel2id[r] for r in pos_train_df['relation'].values], dtype=torch.long)

    data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                x=torch.arange(num_nodes, dtype=torch.long))

    def get_tensors(df):
        edge_label_index = torch.stack([
            torch.tensor(df['x_index'].values, dtype=torch.long),
            torch.tensor(df['y_index'].values, dtype=torch.long)
        ], dim=0)
        edge_label = torch.tensor(df['label'].values, dtype=torch.float)
        rel_tensor = torch.tensor([rel2id[r] for r in df['relation'].values], dtype=torch.long)
        return edge_label_index, edge_label, rel_tensor

    datasets = {
        'train': get_tensors(train_df),
        'val': get_tensors(val_df),
        'test': get_tensors(test_df),
        'test_hard': get_tensors(test_hard_df) if test_hard_df is not None else None
    }
    return data, datasets, num_nodes, num_rels, rel2id


def create_loader(data, dataset_tensors, batch_size, num_neighbors, shuffle=False):
    if dataset_tensors is None: return None
    edge_label_index, edge_label, _ = dataset_tensors
    return LinkNeighborLoader(
        data, num_neighbors=num_neighbors, edge_label_index=edge_label_index,
        edge_label=edge_label, batch_size=batch_size, shuffle=shuffle, neg_sampling_ratio=0
    )


def train_model(model, train_loader, val_loader, train_rel, val_rel, device, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = 0
    patience_counter = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0;
        total_examples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['max_epochs']}")
        for batch in pbar:
            batch = batch.to(device)
            h = model(batch.n_id, batch.edge_index, batch.edge_type)
            src, tgt = batch.edge_label_index
            batch_rel = train_rel[batch.input_id.cpu()].to(device)
            batch_lbl = batch.edge_label.to(device)

            scores = model.score(h, src, tgt, batch_rel)
            loss = criterion(scores, batch_lbl)

            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
            total_loss += loss.item() * batch.edge_label.numel()
            total_examples += batch.edge_label.numel()

        avg_train_loss = total_loss / total_examples
        avg_val_loss, val_auc, _, _ = evaluate(model, val_loader, val_rel, device, verbose=False)

        print(f"Epoch {epoch + 1}: Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val AUC={val_auc:.4f}")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), config['best_model_path'])
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            print("Early stopping.")
            break

    return history


def evaluate(model, loader, rel_tensor, device, save_path=None, verbose=True):
    model.eval()
    preds, targets = [], []
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    if verbose:
        iter_loader = tqdm(loader, desc="Evaluating")
    else:
        iter_loader = loader

    with torch.no_grad():
        for batch in iter_loader:
            batch = batch.to(device)
            h = model(batch.n_id, batch.edge_index, batch.edge_type)
            src, tgt = batch.edge_label_index
            batch_rel = rel_tensor[batch.input_id.cpu()].to(device)
            batch_lbl = batch.edge_label.to(device)

            scores = model.score(h, src, tgt, batch_rel)
            total_loss += criterion(scores, batch_lbl).item()

            preds.append(torch.sigmoid(scores).cpu().numpy())
            targets.append(batch_lbl.cpu().numpy())

    all_preds = np.concatenate(preds)
    all_targets = np.concatenate(targets)

    auc = roc_auc_score(all_targets, all_preds)

    if verbose and save_path:
        all_labels = (all_preds >= 0.5).astype(int)
        acc = accuracy_score(all_targets, all_labels)
        f1 = f1_score(all_targets, all_labels)
        precision = precision_score(all_targets, all_labels)
        recall = recall_score(all_targets, all_labels)

        # 【修改 1】: 打印结果到日志
        print("\n" + "=" * 30)
        print(f"Results for: {os.path.basename(save_path)}")
        print(f"AUC       : {auc:.4f}")
        print(f"Accuracy  : {acc:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print("=" * 30 + "\n")

        # 【修改 2】: 保存 metrics 为 TXT 格式
        with open(save_path + "_metrics.txt", "w") as f:
            f.write(f"AUC: {auc:.6f}\n")
            f.write(f"Accuracy: {acc:.6f}\n")
            f.write(f"F1 Score: {f1:.6f}\n")
            f.write(f"Precision: {precision:.6f}\n")
            f.write(f"Recall: {recall:.6f}\n")

        # 【修改 3】: 保存 Raw Data 并增加 FPR/TPR 以便直接作图
        # 计算 ROC 曲线坐标
        fpr, tpr, thresholds = roc_curve(all_targets, all_preds)

        # 为了方便保存，我们需要把 fpr/tpr 拉长或插值，但通常 raw data 保存原始预测值最灵活
        # 用户需要直接作图，我们保存两份：
        # 1. 原始预测值 (raw_pred.csv) -> 适合重新计算任何指标
        # 2. ROC 绘图数据 (roc_curve.csv) -> 适合直接拖进 Excel/Origin 画图

        # A. 保存原始预测 (Raw)
        raw_df = pd.DataFrame({'y_true': all_targets, 'y_score': all_preds})
        raw_df.to_csv(save_path + "_raw_pred.csv", index=False)

        # B. 保存 ROC 曲线坐标 (Plot Data)
        # FPR (X-axis), TPR (Y-axis)
        roc_df = pd.DataFrame({
            'FPR (1-Specificity)': fpr,
            'TPR (Sensitivity)': tpr,
            'Threshold': thresholds
        })
        roc_df.to_csv(save_path + "_roc_curve_data.csv", index=False)

    return total_loss / len(loader), auc, all_preds, all_targets