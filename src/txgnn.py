import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# === 路径修正与引用 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import load_and_build_data, create_loader

# 尝试导入 TxGNNModel
try:
    from models.txgnn_model import TxGNNModel
except ImportError:
    try:
        from models.txgnn import TxGNNModel
    except ImportError:
        print("❌ Error: Could not import TxGNNModel.")
        sys.exit(1)

import matplotlib

matplotlib.use('Agg')


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


def get_batch_edge_indices(batch):
    """获取边的起点和终点"""
    if hasattr(batch, 'edge_label_index'):
        return batch.edge_label_index[0], batch.edge_label_index[1]
    else:
        # Fallback
        return batch.src, batch.dst


def get_batch_label(batch):
    """精确获取标签，避免获取到 None"""
    # 优先检查 edge_label (PyG LinkNeighborLoader 标准)
    if hasattr(batch, 'edge_label') and batch.edge_label is not None:
        return batch.edge_label
    elif hasattr(batch, 'y') and batch.y is not None:
        return batch.y
    elif hasattr(batch, 'label') and batch.label is not None:
        return batch.label
    else:
        raise AttributeError("Batch object has no valid label attribute (edge_label, y, or label)")


def evaluate(model, loader, rel_tensor, device, dataset_name="Validation"):
    """
    评估函数
    :param rel_tensor: 该数据集对应的全量关系 Tensor (用于 lookup)
    """
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {dataset_name}", leave=False):
            batch = batch.to(device)
            try:
                n_id = batch.n_id if hasattr(batch, 'n_id') else None

                # 1. Forward (TxGNN Embedding)
                out = model(batch.x, batch.edge_index, batch.edge_type, batch_n_id=n_id)

                # 2. 获取 Edge Indices
                src, dst = get_batch_edge_indices(batch)

                # 3. 获取 Relation Type (关键修复)
                # 使用 batch.input_id 从全量 rel_tensor 中查找
                if hasattr(batch, 'input_id'):
                    batch_rel = rel_tensor[batch.input_id.cpu()].to(device)
                else:
                    # Fallback: 假设全图模式
                    print("Warning: No input_id in batch, using zero relations.")
                    batch_rel = torch.zeros_like(src)

                # 4. Score
                scores = model.score(out, src, dst, batch_rel)
                prob = torch.sigmoid(scores)

                preds.append(prob.cpu())

                # 5. Label
                lbl = get_batch_label(batch)
                labels.append(lbl.cpu())

            except IndexError:
                continue

    if len(preds) == 0:
        return 0, 0, 0, 0, 0

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0

    pred_labels = (preds > 0.5).astype(int)
    acc = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    prec = precision_score(labels, pred_labels)
    rec = recall_score(labels, pred_labels)

    return auc, acc, f1, prec, rec


def plot_training_curves(history, save_dir):
    df_hist = pd.DataFrame(history)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df_hist['epoch'], df_hist['loss'], 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df_hist['epoch'], df_hist['val_auc'], 'r-', label='Val AUC')
    plt.title('Validation AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.svg'))
    plt.close()


def train_txgnn_finetune(args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    process_dir = "../data/training_process/TxGNN"
    eval_dir = "../data/evaluation/TxGNN"
    model_dir = "../model/TxGNN"
    log_dir = "../logs"

    for d in [process_dir, eval_dir, model_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(log_dir, f"TxGNN_{timestamp}.log")
    sys.stdout = Logger(log_path)

    print("=" * 50)
    print(f"🚀 Start Fine-tuning: TxGNN")
    print(f"📂 Model Weights: {model_dir}")
    print("=" * 50)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Data
    nodes_path = '../data/benchmark/PrimeKG/nodes.csv'
    train_path = '../data/benchmark/PrimeKG/train_edges.csv'
    val_path = '../data/benchmark/PrimeKG/val_edges.csv'
    test_path = '../data/benchmark/Kaggle_drug_repositioning/test.csv'
    test_hard_path = '../data/benchmark/Kaggle_drug_repositioning/test_hard_degree_matched.csv'

    if not os.path.exists(args.sim_path):
        print(f"❌ Error: Similarity file not found at {args.sim_path}")
        return

    print("Loading data...")
    data, datasets, num_nodes, num_rels, _ = load_and_build_data(
        nodes_path, train_path, val_path, test_path, test_hard_path
    )

    # === 提取 Relation Tensors ===
    train_rel_tensor = datasets['train'][2]
    val_rel_tensor = datasets['val'][2]
    test_rel_tensor = datasets['test'][2]
    hard_rel_tensor = datasets['test_hard'][2] if datasets['test_hard'] else None

    print("Initializing TxGNN Model...")
    model = TxGNNModel(num_nodes, 128, num_rels, device=args.device).to(device)
    model.load_similarity(args.sim_path)

    print(f"Loading Pretrained RGCN Weights from {args.pretrained_path}...")
    if not os.path.exists(args.pretrained_path):
        print("❌ Pretrained model file does not exist.")
        return

    pretrained_dict = torch.load(args.pretrained_path, map_location=device)
    model_dict = model.state_dict()
    new_state_dict = {}

    reinit_count = 0
    for k, v in pretrained_dict.items():
        if 'relation_embedding' in k:
            reinit_count += 1
            continue
        new_key = 'encoder.' + k
        if new_key in model_dict:
            new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Weights Loaded. Re-initialized {reinit_count} decoder layers.")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Create Loaders
    train_loader = create_loader(data, datasets['train'], batch_size=1024, shuffle=True, num_neighbors=[20, 10])
    val_loader = create_loader(data, datasets['val'], batch_size=2048, shuffle=False, num_neighbors=[20, 10])

    print("\nStarting Fine-tuning Phase...")
    history = []
    best_val_auc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            try:
                n_id = batch.n_id if hasattr(batch, 'n_id') else None

                # 1. Forward
                out = model(batch.x, batch.edge_index, batch.edge_type, batch_n_id=n_id)

                # 2. Indices
                src, dst = get_batch_edge_indices(batch)

                # 3. Relations
                if hasattr(batch, 'input_id'):
                    batch_rel = train_rel_tensor[batch.input_id.cpu()].to(device)
                else:
                    batch_rel = torch.zeros_like(src)

                # 4. Score
                pred = model.score(out, src, dst, batch_rel)

                # 5. Label
                lbl = get_batch_label(batch)

                loss = loss_fn(pred, lbl)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            except IndexError:
                continue
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    continue
                raise e

        avg_loss = total_loss / len(train_loader)
        val_auc, _, _, _, _ = evaluate(model, val_loader, val_rel_tensor, device, "Validation")

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}")

        history.append({'epoch': epoch + 1, 'loss': avg_loss, 'val_auc': val_auc})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "txgnn_finetuned_best.pt"))
            print(f"   🌟 New Best Model Saved (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    pd.DataFrame(history).to_csv(os.path.join(process_dir, "train_val_history.csv"), index=False)
    plot_training_curves(history, process_dir)

    print("\n" + "=" * 10 + " Final Evaluation " + "=" * 10)
    model.load_state_dict(torch.load(os.path.join(model_dir, "txgnn_finetuned_best.pt")))

    # === 修复：加入 num_neighbors ===
    test_loader = create_loader(data, datasets['test'], batch_size=2048, num_neighbors=[20, 10], shuffle=False)
    hard_loader = create_loader(data, datasets['test_hard'], batch_size=2048, num_neighbors=[20, 10], shuffle=False)

    def evaluate_and_save(loader, rels, name, filename):
        if loader is None: return
        auc, acc, f1, prec, rec = evaluate(model, loader, rels, device, name)
        print(f"\n========== Evaluation: {name} ==========")
        print(f"AUC       : {auc:.4f}")

        res_df = pd.DataFrame([{
            'Model': 'TxGNN', 'Dataset': name,
            'AUC': auc, 'Accuracy': acc, 'F1': f1, 'Precision': prec, 'Recall': rec
        }])
        res_df.to_csv(os.path.join(eval_dir, filename), index=False)

    evaluate_and_save(test_loader, test_rel_tensor, "Standard Test Set", "txgnn_results_standard.csv")
    evaluate_and_save(hard_loader, hard_rel_tensor, "Hard Test Set (Degree Matched)", "txgnn_results_hard.csv")

    print(f"\n✨ All Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='../model/RGCN/rgcn_best.pt')
    parser.add_argument('--sim_path', type=str, default='../model/TxGNN/txgnn_sim_data.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_txgnn_finetune(args)