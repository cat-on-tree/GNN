import pandas as pd
import torch
import os
import sys
import logging
import time
import numpy as np
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve
)
from tqdm import tqdm


# 配置日志
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


# ==============================================================================
#  【核心】 Ranking 指标计算函数 (与 LLM 对齐)
# ==============================================================================
def calculate_ranking_metrics(y_true, y_score, max_k=None):
    """
    一次性计算各类 Ranking 指标 (Top-K Precision, Enrichment Factor, Cumulative Gain)
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    df = pd.DataFrame({'label': y_true, 'score': y_score})
    # 按分数降序排列
    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    total_pos = sum(y_true)
    total_n = len(y_true)
    baseline_prec = total_pos / total_n

    if max_k is None: max_k = total_n

    k_list, prec_list, ef_list, cum_gain_list = [], [], [], []
    cumulative_pos = 0

    # 遍历计算直到 max_k 或 数据集尽头
    limit = min(max_k, len(df))

    for k in range(1, limit + 1):
        is_pos = df.loc[k - 1, 'label']
        cumulative_pos += is_pos

        prec = cumulative_pos / k
        ef = prec / baseline_prec if baseline_prec > 0 else 0
        gain = cumulative_pos

        k_list.append(k)
        prec_list.append(prec)
        ef_list.append(ef)
        cum_gain_list.append(gain)

    return pd.DataFrame({
        'k': k_list,
        'precision_at_k': prec_list,
        'enrichment_factor': ef_list,
        'cumulative_gain': cum_gain_list
    })


# ==============================================================================
#  数据加载与构建
# ==============================================================================
def load_and_build_data(nodes_path, train_path, val_path, test_path, test_hard_path=None, test_cold_path=None):
    print("Loading data...")
    nodes_df = pd.read_csv(nodes_path)
    num_nodes = int(nodes_df['node_index'].max() + 1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    all_relations = set(train_df['relation'].unique()) | \
                    set(val_df['relation'].unique()) | \
                    set(test_df['relation'].unique())

    # 加载 Hard 测试集
    test_hard_df = None
    if test_hard_path and os.path.exists(test_hard_path):
        print(f"Found Hard Test Set at: {test_hard_path}")
        test_hard_df = pd.read_csv(test_hard_path)
        all_relations = all_relations | set(test_hard_df['relation'].unique())

    # 加载 Cold Start 测试集，并处理列顺序问题
    test_cold_df = None
    if test_cold_path and os.path.exists(test_cold_path):
        print(f"Found Cold Start Test Set at: {test_cold_path}")
        df_temp = pd.read_csv(test_cold_path)

        required_cols = {'x_index', 'y_index', 'label', 'relation'}
        if required_cols.issubset(df_temp.columns):
            test_cold_df = df_temp
            all_relations = all_relations | set(test_cold_df['relation'].unique())
        else:
            print(f"⚠️ Warning: {test_cold_path} missing required columns. Expected {required_cols}")

    relation_types = sorted(list(all_relations))
    rel2id = {r: i for i, r in enumerate(relation_types)}
    num_rels = len(rel2id)
    print(f"Total unique relations: {num_rels}")

    # 构建全图数据
    pos_train_df = train_df[train_df['label'] == 1]
    edge_index = torch.stack([
        torch.tensor(pos_train_df['x_index'].values, dtype=torch.long),
        torch.tensor(pos_train_df['y_index'].values, dtype=torch.long)
    ], dim=0)
    edge_type = torch.tensor([rel2id[r] for r in pos_train_df['relation'].values], dtype=torch.long)

    data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                x=torch.arange(num_nodes, dtype=torch.long))

    def get_tensors(df):
        if df is None: return None
        if not {'x_index', 'y_index', 'label', 'relation'}.issubset(df.columns):
            return None

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
        'test_hard': get_tensors(test_hard_df),
        'test_cold': get_tensors(test_cold_df)
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

    # 1. 基础指标 (无论是否 verbose 都计算)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.0

    # 2. 仅在正式评估阶段 (save_path不为空) 计算复杂指标并生成 CSV/TXT
    if verbose and save_path:
        model_prefix = os.path.basename(save_path)  # e.g., "standard"

        # AUPRC
        auprc = average_precision_score(all_targets, all_preds)

        # Threshold Metrics (F1, Acc, Recall, Precision)
        all_labels = (all_preds >= 0.5).astype(int)
        acc = accuracy_score(all_targets, all_labels)
        f1 = f1_score(all_targets, all_labels)
        precision = precision_score(all_targets, all_labels)
        recall = recall_score(all_targets, all_labels)

        # Ranking Metrics (Top-K, EF, CG)
        df_ranking = calculate_ranking_metrics(all_targets, all_preds, max_k=len(all_targets))

        # --- 打印 Summary ---
        print("\n" + "=" * 40)
        print(f"📊 Model Evaluation Summary: {model_prefix}")
        print("=" * 40)
        print(f"   AUPRC (Avg Precision) : {auprc:.4f}")
        print(f"   AUROC                 : {auc:.4f}")
        print(f"   F1-Score (th=0.5)     : {f1:.4f}")
        print(f"   Accuracy (th=0.5)     : {acc:.4f}")
        print("-" * 40)

        top_ks = [10, 50, 100]
        top_k_vals = {}
        for k in top_ks:
            if k <= len(df_ranking):
                val = df_ranking.loc[k - 1, 'precision_at_k']
                print(f"   Precision @ Top-{k:<3}   : {val:.4f}")
                top_k_vals[k] = val
            else:
                top_k_vals[k] = 0.0
        print("=" * 40 + "\n")

        # --- 保存 1: Summary TXT ---
        with open(save_path + "_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Dataset: {model_prefix}\n")
            f.write(f"Total Samples: {len(all_targets)}\n")
            f.write(f"Pos Ratio: {sum(all_targets) / len(all_targets):.4f}\n\n")
            f.write(f"AUROC: {auc:.6f}\n")
            f.write(f"AUPRC: {auprc:.6f}\n")
            f.write(f"F1-Score (th=0.5): {f1:.6f}\n")
            f.write(f"Accuracy (th=0.5): {acc:.6f}\n")
            f.write(f"Precision (th=0.5): {precision:.6f}\n")
            f.write(f"Recall (th=0.5): {recall:.6f}\n\n")
            f.write("=== Top-K Metrics ===\n")
            for k in top_ks:
                f.write(f"Precision @ Top-{k}: {top_k_vals[k]:.6f}\n")

        # --- 保存 2: ROC Curve Data (FPR, TPR) ---
        fpr, tpr, roc_thresholds = roc_curve(all_targets, all_preds)
        df_roc = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': roc_thresholds})
        df_roc.to_csv(save_path + "_roc_curve.csv", index=False)

        # --- 保存 3: PRC Curve Data (Recall, Precision) ---
        precs, recs, prc_thresholds = precision_recall_curve(all_targets, all_preds)
        # 补齐阈值长度
        prc_thresholds = np.append(prc_thresholds, 1.0)
        df_prc = pd.DataFrame({'Recall': recs, 'Precision': precs, 'Threshold': prc_thresholds})
        df_prc.to_csv(save_path + "_prc_curve.csv", index=False)

        # --- 保存 4: EF & Cumulative Gain (前20个点) ---
        df_ranking_top20 = df_ranking.head(20).copy()
        df_ranking_top20[['k', 'enrichment_factor']].to_csv(save_path + "_ef_top20.csv", index=False)
        df_ranking_top20[['k', 'cumulative_gain']].to_csv(save_path + "_cumgain_top20.csv", index=False)

        # --- 保存 5: Top-K Bar Data ---
        df_topk_bar = pd.DataFrame({
            'Top_K': [f'Top-{k}' for k in top_ks],
            'Precision': [top_k_vals[k] for k in top_ks]
        })
        df_topk_bar.to_csv(save_path + "_topk_bar.csv", index=False)

        # --- 保存 6: 原始预测 (Raw) ---
        raw_df = pd.DataFrame({'y_true': all_targets, 'y_score': all_preds})
        raw_df.to_csv(save_path + "_raw_pred.csv", index=False)

    return total_loss / len(loader), auc, all_preds, all_targets