import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve
from tqdm import tqdm

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


# === Logger 类 (用于追加日志) ===
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        # mode='a' 表示追加模式，不会覆盖旧日志
        self.log = open(filepath, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


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


def evaluate_detailed(model, loader, rel_tensor, device, dataset_name, save_dir, prefix):
    """
    详细评估函数，生成 metrics.txt, raw_pred.csv, roc_curve_data.csv
    """
    model.eval()
    preds = []
    labels = []

    print(f"Evaluating {dataset_name}...")
    with torch.no_grad():
        # 为了不破坏日志格式，这里 leave=False
        for batch in tqdm(loader, desc=f"Eval {dataset_name}", leave=False):
            batch = batch.to(device)
            try:
                n_id = batch.n_id if hasattr(batch, 'n_id') else None

                # Forward
                out = model(batch.x, batch.edge_index, batch.edge_type, batch_n_id=n_id)

                # Indices
                src, dst = get_batch_edge_indices(batch)

                # Relations
                if hasattr(batch, 'input_id'):
                    batch_rel = rel_tensor[batch.input_id.cpu()].to(device)
                else:
                    batch_rel = torch.zeros_like(src)

                # Score
                scores = model.score(out, src, dst, batch_rel)
                prob = torch.sigmoid(scores)

                preds.append(prob.cpu())
                lbl = get_batch_label(batch)
                labels.append(lbl.cpu())

            except IndexError:
                continue

    if len(preds) == 0:
        print(f"⚠️ Warning: No predictions made for {dataset_name}")
        return

    # 拼接结果
    all_preds = torch.cat(preds).numpy()
    all_targets = torch.cat(labels).numpy()

    # 计算指标
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.0

    pred_labels = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_targets, pred_labels)
    f1 = f1_score(all_targets, pred_labels)
    prec = precision_score(all_targets, pred_labels)
    rec = recall_score(all_targets, pred_labels)

    print(f"\n========== Results: {dataset_name} ==========")
    print(f"AUC       : {auc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("=" * 40)

    # === 保存详细文件 ===
    os.makedirs(save_dir, exist_ok=True)
    base_path = os.path.join(save_dir, prefix)

    # 1. 保存 Metrics TXT
    with open(f"{base_path}_metrics.txt", "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"AUC: {auc:.6f}\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"F1 Score: {f1:.6f}\n")
        f.write(f"Precision: {prec:.6f}\n")
        f.write(f"Recall: {rec:.6f}\n")
    print(f"📄 Saved metrics to {base_path}_metrics.txt")

    # 2. 保存 Raw Predictions CSV (y_true, y_score)
    raw_df = pd.DataFrame({
        'y_true': all_targets,
        'y_score': all_preds
    })
    raw_df.to_csv(f"{base_path}_raw_pred.csv", index=False)
    print(f"📄 Saved raw predictions to {base_path}_raw_pred.csv")

    # 3. 保存 ROC Curve Data CSV
    fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
    roc_df = pd.DataFrame({
        'FPR (1-Specificity)': fpr,
        'TPR (Sensitivity)': tpr,
        'Threshold': thresholds
    })
    roc_df.to_csv(f"{base_path}_roc_curve_data.csv", index=False)
    print(f"📄 Saved ROC curve data to {base_path}_roc_curve_data.csv")


def main():
    parser = argparse.ArgumentParser()
    # 关键路径配置
    parser.add_argument('--model_path', type=str, default='../model/TxGNN/txgnn_finetuned_best.pt',
                        help='Path to the fine-tuned model weights')
    parser.add_argument('--sim_path', type=str, default='../model/TxGNN/txgnn_sim_data.pt',
                        help='Path to similarity matrix')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # === 自动配置日志 ===
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)

    # 寻找最新的 TxGNN 日志文件
    logs = [f for f in os.listdir(log_dir) if f.startswith("TxGNN_") and f.endswith(".log")]

    if logs:
        # 按修改时间排序，找最新的
        latest_log = max(logs, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
        log_path = os.path.join(log_dir, latest_log)
        print(f"📝 Found existing log: {log_path}. Appending evaluation results...")
    else:
        # 如果没找到，新建一个
        log_path = os.path.join(log_dir, "TxGNN_eval_only.log")
        print(f"📝 No existing log found. Creating new log: {log_path}")

    # 重定向 stdout 到 Logger
    sys.stdout = Logger(log_path)

    print("\n" + "=" * 50)
    print("🚀 RESUMING PIPELINE: FINAL EVALUATION")
    print(f"📅 Time: {pd.Timestamp.now()}")
    print("=" * 50)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 结果保存目录
    eval_save_dir = "../data/evaluation/TxGNN"
    os.makedirs(eval_save_dir, exist_ok=True)

    # 1. 加载数据
    nodes_path = '../data/benchmark/PrimeKG/nodes.csv'
    train_path = '../data/benchmark/PrimeKG/train_edges.csv'
    val_path = '../data/benchmark/PrimeKG/val_edges.csv'
    test_path = '../data/benchmark/Kaggle_drug_repositioning/test.csv'
    test_hard_path = '../data/benchmark/Kaggle_drug_repositioning/test_hard.csv'  # 确保这里路径正确

    print("Loading data...")
    data, datasets, num_nodes, num_rels, _ = load_and_build_data(
        nodes_path, train_path, val_path, test_path, test_hard_path
    )

    # 提取 Rel Tensors
    test_rel_tensor = datasets['test'][2]
    hard_rel_tensor = datasets['test_hard'][2] if datasets['test_hard'] else None

    # 2. 初始化并加载模型
    print("Initializing TxGNN Model...")
    model = TxGNNModel(num_nodes, 128, num_rels, device=args.device).to(device)
    model.load_similarity(args.sim_path)

    print(f"Loading Fine-tuned Weights from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        return

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("✅ Model Loaded Successfully.")

    # 3. 创建 Loaders
    print("Creating Loaders...")
    # 使用 [20, 10] 或 [-1] 均可，这里保持一致性
    test_loader = create_loader(data, datasets['test'], batch_size=2048, num_neighbors=[20, 10], shuffle=False)
    hard_loader = create_loader(data, datasets['test_hard'], batch_size=2048, num_neighbors=[20, 10], shuffle=False)

    # 4. 执行评估
    print("\n" + "=" * 10 + " Starting Final Evaluation " + "=" * 10)

    # Standard Test
    evaluate_detailed(model, test_loader, test_rel_tensor, device,
                      dataset_name="Standard Test Set",
                      save_dir=eval_save_dir,
                      prefix="standard")  # 生成 standard_metrics.txt 等

    # Hard Test
    if hard_loader:
        evaluate_detailed(model, hard_loader, hard_rel_tensor, device,
                          dataset_name="Hard Test Set",
                          save_dir=eval_save_dir,
                          prefix="hard")  # 生成 hard_metrics.txt 等

    print(f"\n✨ All Evaluation Artifacts saved to {eval_save_dir}")


if __name__ == "__main__":
    main()