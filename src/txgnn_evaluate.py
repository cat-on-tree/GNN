import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
import time
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
from scipy import interpolate

# === 路径修正与引用 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import load_and_build_data, create_loader

try:
    from models.txgnn_model import TxGNNModel
except ImportError:
    try:
        from models.txgnn import TxGNNModel
    except ImportError:
        print("❌ Error: Could not import TxGNNModel.")
        sys.exit(1)


# ==============================================================================
#  Logger 类
# ==============================================================================
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


# ==============================================================================
#  核心辅助函数 (插值对齐 + Ranking)
# ==============================================================================
def get_batch_edge_indices(batch):
    if hasattr(batch, 'edge_label_index'):
        return batch.edge_label_index[0], batch.edge_label_index[1]
    return batch.src, batch.dst


def get_batch_label(batch):
    if hasattr(batch, 'edge_label') and batch.edge_label is not None:
        return batch.edge_label
    elif hasattr(batch, 'y') and batch.y is not None:
        return batch.y
    elif hasattr(batch, 'label') and batch.label is not None:
        return batch.label
    raise AttributeError("Batch object has no valid label attribute")


def interpolate_curve(x, y, num_points=1000):
    """
    对曲线进行线性插值，使其变为固定的行数 (num_points)。
    """
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    _, unique_indices = np.unique(x_sorted, return_index=True)
    x_unique = x_sorted[unique_indices]
    y_unique = y_sorted[unique_indices]

    if len(x_unique) < 2: return x, y

    f = interpolate.interp1d(x_unique, y_unique, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(min(x_unique), max(x_unique), num_points)
    y_new = f(x_new)
    y_new = np.clip(y_new, 0.0, 1.0)
    return x_new, y_new


def calculate_ranking_metrics(y_true, y_score, max_k=None):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    df = pd.DataFrame({'label': y_true, 'score': y_score})
    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    total_pos = sum(y_true)
    total_n = len(y_true)
    baseline_prec = total_pos / total_n

    if max_k is None: max_k = total_n

    k_list, prec_list, ef_list, cum_gain_list = [], [], [], []
    cumulative_pos = 0

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
#  详细评估函数
# ==============================================================================
def evaluate_detailed(model, loader, rel_tensor, device, dataset_name, save_dir, prefix, force_rel_id=None):
    """
    force_rel_id:
      - None: 使用原本的 Batch input_id (适用于 Standard Test)
      - Int (e.g. 14): 强制所有关系 ID 为该值 (适用于 Cold/Hard Test)
    """
    model.eval()
    preds = []
    labels = []

    print(f"Evaluating {dataset_name} (Force Rel ID: {force_rel_id})...")

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {dataset_name}", leave=False):
            batch = batch.to(device)
            try:
                n_id = batch.n_id if hasattr(batch, 'n_id') else None

                # Forward Pass
                out = model(batch.x, batch.edge_index, batch.edge_type, batch_n_id=n_id)
                src, dst = get_batch_edge_indices(batch)

                # --- 关系 ID 处理逻辑 (关键修正) ---
                if force_rel_id is not None:
                    # 1. 如果指定了 ID (针对 Cold/Hard)，强制覆盖
                    batch_rel = torch.full((len(src),), force_rel_id, dtype=torch.long, device=device)
                else:
                    # 2. 如果没指定 (针对 Standard)，尝试使用原有的 Relation
                    if hasattr(batch, 'input_id'):
                        batch_rel = rel_tensor[batch.input_id.cpu()].to(device)
                    else:
                        # 兜底：如果都没有，才填 0
                        batch_rel = torch.zeros_like(src)
                # -----------------------------------

                # Score
                if hasattr(model, 'score'):
                    scores = model.score(out, src, dst, batch_rel)
                else:
                    scores = model.encoder.score(out, src, dst, batch_rel)

                prob = torch.sigmoid(scores)

                preds.append(prob.cpu())
                lbl = get_batch_label(batch)
                labels.append(lbl.cpu())

            except IndexError:
                continue

    if len(preds) == 0:
        print(f"⚠️ Warning: No predictions made for {dataset_name}")
        return

    all_preds = torch.cat(preds).numpy()
    all_targets = torch.cat(labels).numpy()

    # --- Metrics ---
    try:
        auroc = roc_auc_score(all_targets, all_preds)
    except:
        auroc = 0.0

    auprc = average_precision_score(all_targets, all_preds)

    pred_labels = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_targets, pred_labels)
    f1 = f1_score(all_targets, pred_labels)
    prec = precision_score(all_targets, pred_labels)
    rec = recall_score(all_targets, pred_labels)

    df_ranking = calculate_ranking_metrics(all_targets, all_preds, max_k=len(all_targets))

    print(f"\n========== Results: {dataset_name} ==========")
    print(f"AUPRC     : {auprc:.4f}")
    print(f"AUROC     : {auroc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print("-" * 30)

    top_ks = [10, 50, 100]
    top_k_vals = {}
    for k in top_ks:
        if k <= len(df_ranking):
            val = df_ranking.loc[k - 1, 'precision_at_k']
            print(f"Top-{k:<3} Prec: {val:.4f}")
            top_k_vals[k] = val
        else:
            top_k_vals[k] = 0.0
    print("=" * 40)

    # --- Save Artifacts ---
    os.makedirs(save_dir, exist_ok=True)
    base_path = os.path.join(save_dir, prefix)

    # Summary
    with open(f"{base_path}_summary.txt", "w", encoding='utf-8') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Relation ID Used: {force_rel_id}\n")
        f.write(f"Total Samples: {len(all_targets)}\n")
        f.write(f"Pos Ratio: {sum(all_targets) / len(all_targets):.4f}\n\n")
        f.write(f"AUROC: {auroc:.6f}\n")
        f.write(f"AUPRC: {auprc:.6f}\n")
        f.write(f"Accuracy (th=0.5): {acc:.6f}\n")
        f.write(f"F1 Score (th=0.5): {f1:.6f}\n")
        f.write(f"Precision (th=0.5): {prec:.6f}\n")
        f.write(f"Recall (th=0.5): {rec:.6f}\n\n")
        f.write("=== Top-K Metrics ===\n")
        for k in top_ks:
            f.write(f"Precision @ Top-{k}: {top_k_vals[k]:.6f}\n")

    # Raw Preds
    pd.DataFrame({'y_true': all_targets, 'y_score': all_preds}).to_csv(f"{base_path}_raw_pred.csv", index=False)

    # ROC (Aligned)
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    fpr_i, tpr_i = interpolate_curve(fpr, tpr, 1000)
    pd.DataFrame({'FPR': fpr_i, 'TPR': tpr_i}).to_csv(f"{base_path}_roc_curve.csv", index=False)

    # PRC (Aligned)
    precs, recs, _ = precision_recall_curve(all_targets, all_preds)
    recs_i, precs_i = interpolate_curve(recs, precs, 1000)
    pd.DataFrame({'Recall': recs_i, 'Precision': precs_i}).to_csv(f"{base_path}_prc_curve.csv", index=False)

    # EF & CumGain
    df_rank_20 = df_ranking.head(20).copy()
    df_rank_20[['k', 'enrichment_factor']].to_csv(f"{base_path}_ef_top20.csv", index=False)
    df_rank_20[['k', 'cumulative_gain']].to_csv(f"{base_path}_cumgain_top20.csv", index=False)

    # Top-K Bar
    pd.DataFrame({'Top_K': top_ks, 'Precision': [top_k_vals[k] for k in top_ks]}).to_csv(f"{base_path}_topk_bar.csv",
                                                                                         index=False)

    print(f"✨ Artifacts saved to {os.path.dirname(base_path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../model/TxGNN/txgnn_finetuned_best.pt')
    parser.add_argument('--sim_path', type=str, default='../model/TxGNN/txgnn_sim_data.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--target_rel_id', type=int, default=14, help='Forced ID for Cold/Hard sets')
    args = parser.parse_args()

    # Log Setup
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_dir, f"TxGNN_Final_Eval_{int(time.time())}.log"))

    print("\n" + "=" * 50)
    print("🚀 RESUMING PIPELINE: FINAL EVALUATION (Corrected Logic)")
    print("=" * 50)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Data
    nodes_path = '../data/benchmark/PrimeKG/nodes.csv'
    train_path = '../data/benchmark/PrimeKG/train_edges.csv'
    val_path = '../data/benchmark/PrimeKG/val_edges.csv'
    test_path = '../data/benchmark/Kaggle_drug_repositioning/test.csv'
    test_hard_path = '../data/benchmark/Kaggle_drug_repositioning/test_hard.csv'
    test_cold_path = '../data/benchmark/Kaggle_drug_repositioning/test_cold.csv'

    print("Loading data...")
    data, datasets, num_nodes, num_rels, _ = load_and_build_data(
        nodes_path, train_path, val_path, test_path,
        test_hard_path=test_hard_path, test_cold_path=test_cold_path
    )

    # Rel Tensors
    test_rel_tensor = datasets['test'][2]
    hard_rel_tensor = datasets['test_hard'][2] if datasets['test_hard'] else None
    cold_rel_tensor = datasets['test_cold'][2] if 'test_cold' in datasets and datasets['test_cold'] else None

    # Model
    print("Initializing TxGNN Model...")
    model = TxGNNModel(num_nodes, 128, num_rels, device=args.device).to(device)
    model.load_similarity(args.sim_path)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("✅ Model Loaded.")

    # Loaders
    batch_size = 2048
    test_loader = create_loader(data, datasets['test'], batch_size, [20, 10], False)
    hard_loader = create_loader(data, datasets['test_hard'], batch_size, [20, 10], False) if datasets[
        'test_hard'] else None
    cold_loader = create_loader(data, datasets['test_cold'], batch_size, [20, 10], False) if datasets[
        'test_cold'] else None

    save_dir = "../data/evaluation/TxGNN"

    # --- 1. Standard Test: 使用原始 ID (force_rel_id=None) ---
    print("\n>>> Running Standard Test (Original IDs)...")
    evaluate_detailed(model, test_loader, test_rel_tensor, device,
                      "Standard Test Set", save_dir, "standard",
                      force_rel_id=None)  # <--- 关键点：Standard 不强制 ID

    # --- 2. Hard Test: 强制使用 ID ---
    if hard_loader:
        print(f"\n>>> Running Hard Test (Force ID 7)...")
        evaluate_detailed(model, hard_loader, hard_rel_tensor, device,
                          "Hard Test Set", save_dir, "hard",
                          force_rel_id=7) # <--- 修改为 7

    # --- 3. Cold Test: 强制使用 ID 14 (保持不变) ---
    if cold_loader:
        print(f"\n>>> Running Cold Test (Force ID 14)...")
        evaluate_detailed(model, cold_loader, cold_rel_tensor, device,
                          "Cold Start Test Set", save_dir, "cold",
                          force_rel_id=14) # <--- 保持 14

    print(f"\n✨ All Evaluation Artifacts saved to {save_dir}")


if __name__ == "__main__":
    main()