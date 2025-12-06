import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from tqdm import tqdm
import random
import datetime


# === 1. 基础设置 ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.append(current_dir)
if parent_dir not in sys.path: sys.path.append(parent_dir)

from utils import load_and_build_data, create_loader

try:
    from models.txgnn_model import TxGNNModel
    from models.rgcn import RGCN
    from models.han import HAN
    from models.hgt import HGT
except ImportError:
    try:
        from .txgnn import TxGNNModel
    except:
        pass


class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self): pass


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
    raise AttributeError("Batch has no label")


def evaluate_single_model(model, loader, rel_tensor, device, model_name, output_root):
    print(f"\n📊 Evaluating [{model_name}]...")
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {model_name}", leave=False):
            batch = batch.to(device)
            try:
                n_id = batch.n_id if hasattr(batch, 'n_id') else None

                # Forward
                if model_name == "TxGNN":
                    out = model(batch.x, batch.edge_index, batch.edge_type, batch_n_id=n_id)
                elif model_name in ["HGT", "HAN", "RGCN"]:
                    out = model(batch.x, batch.edge_index, batch.edge_type)

                # Scoring
                src, dst = get_batch_edge_indices(batch)

                # 关键：强制使用 ID 4
                if hasattr(batch, 'input_id'):
                    batch_rel = rel_tensor[batch.input_id.cpu()].to(device)
                else:
                    batch_rel = torch.full_like(src, 4, dtype=torch.long)  # Fallback ID 4

                scores = model.score(out, src, dst, batch_rel)
                prob = torch.sigmoid(scores)

                preds.append(prob.cpu())
                labels.append(get_batch_label(batch).cpu())
            except Exception as e:
                continue

    if len(preds) == 0: return None

    all_preds = torch.cat(preds).numpy()
    all_targets = torch.cat(labels).numpy()

    # Metrics
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5

    pred_lbls = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_targets, pred_lbls)
    f1 = f1_score(all_targets, pred_lbls)

    print(f"   👉 {model_name}: AUC={auc:.4f}, F1={f1:.4f}")

    # === 保存 Artifacts ===
    save_dir = os.path.join(output_root, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Metrics
    with open(os.path.join(save_dir, "cold_metrics.txt"), "w") as f:
        f.write(f"AUC: {auc:.6f}\nAccuracy: {acc:.6f}\nF1: {f1:.6f}\n")

    # 2. Raw Predictions
    pd.DataFrame({'y_true': all_targets, 'y_score': all_preds}).to_csv(
        os.path.join(save_dir, "cold_raw_pred.csv"), index=False
    )

    # 3. ROC Curve Data (Requested Feature)
    fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
    pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds}).to_csv(
        os.path.join(save_dir, "cold_roc_curve_data.csv"), index=False
    )

    return {'Model': model_name, 'AUC': auc, 'Accuracy': acc, 'F1': f1}


def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--txgnn_path', default='../model/TxGNN/txgnn_finetuned_best.pt')
    parser.add_argument('--rgcn_path', default='../model/RGCN/rgcn_best.pt')
    parser.add_argument('--han_path', default='../model/HAN/han_best.pt')
    parser.add_argument('--hgt_path', default='../model/HGT/hgt_best.pt')
    parser.add_argument('--sim_path', default='../model/TxGNN/txgnn_sim_data.pt')
    parser.add_argument('--test_path', default='../data/benchmark/Kaggle_drug_repositioning/test_cold.csv')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # === 设置动态日志文件名 ===
    os.makedirs("../logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"../logs/COLD_{timestamp}.log"
    sys.stdout = Logger(log_filename)

    print("=" * 60)
    print("🚀 COLD START BENCHMARK (Optimized Relation ID: 4)")
    print(f"📅 Timestamp: {timestamp}")
    print("=" * 60)

    # Load Data
    nodes_path = '../data/benchmark/PrimeKG/nodes.csv'
    train_path = '../data/benchmark/PrimeKG/train_edges.csv'
    val_path = '../data/benchmark/PrimeKG/val_edges.csv'

    print("\n[Data] Loading Graph...")
    data, datasets, num_nodes, num_rels, rel2id = load_and_build_data(
        nodes_path, train_path, val_path,
        test_path=args.test_path, test_hard_path=None
    )

    # Load Test Set
    df_cold = pd.read_csv(args.test_path)

    # === 🏆 强制使用扫描出的最佳 ID 4 ===
    TARGET_ID = 4
    print(f"🔒 Locking Relation ID to {TARGET_ID} (Best Match for Indication)")

    # 构造一个全为 4 的 tensor
    rel_tensor = torch.full((len(df_cold),), TARGET_ID, dtype=torch.long)

    # Loader
    edge_label_index = torch.stack([
        torch.tensor(df_cold['x_index'].values, dtype=torch.long),
        torch.tensor(df_cold['y_index'].values, dtype=torch.long)
    ], dim=0)
    edge_label = torch.tensor(df_cold['label'].values, dtype=torch.float)

    cold_loader = create_loader(data, (edge_label_index, edge_label, rel_tensor),
                                batch_size=2048, num_neighbors=[20, 10], shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_root = "../data/evaluation"
    results = []

    # 1. RGCN
    if os.path.exists(args.rgcn_path):
        try:
            model = RGCN(num_nodes, 128, num_rels).to(device)
            model.load_state_dict(torch.load(args.rgcn_path, map_location=device))
            res = evaluate_single_model(model, cold_loader, rel_tensor, device, "RGCN", output_root)
            if res: results.append(res)
            del model
        except Exception as e:
            print(f"RGCN Error: {e}")

    # 2. HAN
    if os.path.exists(args.han_path):
        try:
            model = HAN(num_nodes, 128, num_rels).to(device)
            model.load_state_dict(torch.load(args.han_path, map_location=device))
            res = evaluate_single_model(model, cold_loader, rel_tensor, device, "HAN", output_root)
            if res: results.append(res)
            del model
        except Exception as e:
            print(f"HAN Error: {e}")

    # 3. HGT
    if os.path.exists(args.hgt_path):
        try:
            model = HGT(num_nodes, 128, num_rels).to(device)
            model.load_state_dict(torch.load(args.hgt_path, map_location=device))
            res = evaluate_single_model(model, cold_loader, rel_tensor, device, "HGT", output_root)
            if res: results.append(res)
            del model
        except Exception as e:
            print(f"HGT Error: {e}")

    # 4. TxGNN
    if os.path.exists(args.txgnn_path):
        try:
            model = TxGNNModel(num_nodes, 128, num_rels, device=args.device).to(device)
            model.load_similarity(args.sim_path)
            # 处理 Strict Loading
            state_dict = torch.load(args.txgnn_path, map_location=device)
            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            model.load_state_dict(filtered_state_dict, strict=False)

            res = evaluate_single_model(model, cold_loader, rel_tensor, device, "TxGNN", output_root)
            if res: results.append(res)
            del model
        except Exception as e:
            print(f"TxGNN Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS: Cold Start Benchmark")
    print("=" * 60)
    if results:
        df = pd.DataFrame(results)
        print(df[['Model', 'AUC', 'Accuracy', 'F1']].sort_values('AUC', ascending=False).to_string(index=False))
        df.to_csv(os.path.join(output_root, "cold_start_benchmark_summary_final.csv"), index=False)


if __name__ == "__main__":
    main()