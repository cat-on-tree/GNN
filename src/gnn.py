import argparse
import os
import sys
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 引入你的工具函数和模型
from utils import load_and_build_data, create_loader, train_model, evaluate  # 这里的 evaluate 我们在下面会重写覆盖
from models.rgcn import RGCN
from models.hgt import HGT
from models.han import HAN

# 设置绘图风格
plt.style.use('ggplot')

# ==============================================================================
#  核心修复：带强制 ID 功能的 Evaluate 函数 (无需重建 Loader)
# ==============================================================================
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def get_batch_label(batch):
    if hasattr(batch, 'edge_label') and batch.edge_label is not None:
        return batch.edge_label
    elif hasattr(batch, 'y') and batch.y is not None:
        return batch.y
    return None


def interpolate_curve(x, y, num_points=1001):
    target_x = np.linspace(0, 1, num_points)
    if len(x) < 2: return target_x, np.zeros_like(target_x)

    # 处理降序
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]

    # 去重
    _, unique_indices = np.unique(x, return_index=True)
    x = x[unique_indices]
    y = y[unique_indices]

    interp_y = np.interp(target_x, x, y)
    return target_x, interp_y


def calculate_ranking_metrics(y_true, y_score, max_k=None):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    df = pd.DataFrame({'label': y_true, 'score': y_score})
    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    total_pos = sum(y_true)
    if max_k is None: max_k = len(y_true)

    k_list, prec_list, ef_list, cum_gain_list = [], [], [], []
    cumulative_pos = 0
    total_n = len(y_true)
    baseline_prec = total_pos / total_n if total_n > 0 else 0

    limit = min(max_k, len(df))
    for k in range(1, limit + 1):
        is_pos = df.loc[k - 1, 'label']
        cumulative_pos += is_pos
        prec = cumulative_pos / k
        ef = prec / baseline_prec if baseline_prec > 0 else 0

        k_list.append(k)
        prec_list.append(prec)
        ef_list.append(ef)
        cum_gain_list.append(cumulative_pos)

    return pd.DataFrame({
        'k': k_list, 'precision_at_k': prec_list,
        'enrichment_factor': ef_list, 'cumulative_gain': cum_gain_list
    })


def evaluate_with_force_id(model, loader, rel_tensor, device, save_path=None, force_rel_id=None):
    """
    参数:
      force_rel_id: 如果不为 None (例如 14)，则强制把 batch 中的关系 ID 替换为该值。
    """
    model.eval()
    preds, targets = [], []

    # 进度条
    dataset_name = os.path.basename(save_path) if save_path else "Eval"
    pbar = tqdm(loader, desc=f"Eval {dataset_name} (ForceID={force_rel_id})", leave=False)

    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)

            # --- 1. 获取输入 ---
            # 兼容不同模型的 forward 签名
            try:
                # 大多数 PyG 模型
                out = model(batch.n_id, batch.edge_index, batch.edge_type)
            except:
                # 某些模型可能需要 batch_n_id (如 TxGNN)
                out = model(batch.x, batch.edge_index, batch.edge_type)

            # --- 2. 关系 ID 修正逻辑 ---
            if hasattr(batch, 'edge_label_index'):
                src, dst = batch.edge_label_index
            else:
                src, dst = batch.src, batch.dst

            if force_rel_id is not None:
                # 【关键】强制构造全为 force_rel_id 的 Tensor
                batch_rel = torch.full((len(src),), force_rel_id, dtype=torch.long, device=device)
            else:
                # 【关键】使用原始数据自带的 ID
                if hasattr(batch, 'input_id'):
                    batch_rel = rel_tensor[batch.input_id.cpu()].to(device)
                else:
                    batch_rel = torch.zeros_like(src)

            # --- 3. 计算分数 ---
            scores = model.score(out, src, dst, batch_rel)

            # 确保 sigmoid (如果模型内部没做)
            # RGCN/HGT/HAN 的 score 通常是 logits
            scores = torch.sigmoid(scores)

            batch_lbl = get_batch_label(batch)
            if batch_lbl is None: continue

            preds.append(scores.cpu().numpy())
            targets.append(batch_lbl.cpu().numpy())

    if not preds: return 0, 0, [], []

    all_preds = np.concatenate(preds)
    all_targets = np.concatenate(targets)

    # --- 4. 计算指标与保存 ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Metrics
        try:
            auc_val = roc_auc_score(all_targets, all_preds)
        except:
            auc_val = 0.5
        auprc = average_precision_score(all_targets, all_preds)

        binary_preds = (all_preds >= 0.5).astype(int)
        acc = accuracy_score(all_targets, binary_preds)
        f1 = f1_score(all_targets, binary_preds)

        # Ranking
        df_rank = calculate_ranking_metrics(all_targets, all_preds, max_k=len(all_targets))

        # Top-K
        top_ks = [10, 50, 100]
        top_k_res = {}
        for k in top_ks:
            row = df_rank[df_rank['k'] == k]
            top_k_res[k] = row.iloc[0]['precision_at_k'] if not row.empty else 0.0

        # --- 保存文件 ---
        # 1. Summary
        with open(save_path + "_summary.txt", "w", encoding='utf-8') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Rel ID Used: {force_rel_id}\n")
            f.write(f"AUROC: {auc_val:.6f}\n")
            f.write(f"AUPRC: {auprc:.6f}\n")
            f.write(f"Accuracy: {acc:.6f}\n")
            f.write(f"F1: {f1:.6f}\n")
            for k in top_ks:
                f.write(f"Top-{k}: {top_k_res[k]:.6f}\n")

        # 2. Raw Pred
        pd.DataFrame({'y_true': all_targets, 'y_score': all_preds}).to_csv(save_path + "_raw_pred.csv", index=False)

        # 3. Ranking Metrics (Original)
        df_rank.to_csv(save_path + "_ranking_metrics.csv", index=False)

        # 4. ROC (Aligned)
        fpr, tpr, _ = roc_curve(all_targets, all_preds)
        fpr_i, tpr_i = interpolate_curve(fpr, tpr, 1001)
        pd.DataFrame({'FPR': fpr_i, 'TPR': tpr_i}).to_csv(save_path + "_roc_curve.csv", index=False)

        # 5. PRC (Aligned)
        precs, recs, _ = precision_recall_curve(all_targets, all_preds)
        recs_i, precs_i = interpolate_curve(recs, precs, 1001)
        pd.DataFrame({'Recall': recs_i, 'Precision': precs_i}).to_csv(save_path + "_prc_curve.csv", index=False)

        # 6. EF & CumGain (Top 20)
        df_top20 = df_rank.head(20).copy()
        df_top20[['k', 'enrichment_factor']].to_csv(save_path + "_ef_top20.csv", index=False)
        df_top20[['k', 'cumulative_gain']].to_csv(save_path + "_cumgain_top20.csv", index=False)

        # 7. Top-K Bar
        pd.DataFrame({'Top_K': top_ks, 'Precision': [top_k_res[k] for k in top_ks]}).to_csv(save_path + "_topk_bar.csv",
                                                                                            index=False)

        # 8. PNG Previews
        generate_visualizations(save_path, model.__class__.__name__, dataset_name)

        print(f"✅ Saved results to {save_path}")

    return 0.0, 0.0, all_preds, all_targets  # Dummy return for compat


def generate_visualizations(save_path_prefix, model_name, dataset_name):
    """简单画个图预览一下"""
    try:
        df_roc = pd.read_csv(save_path_prefix + "_roc_curve.csv")
        plt.figure(figsize=(4, 4))
        plt.plot(df_roc['FPR'], df_roc['TPR'])
        plt.title('ROC')
        plt.savefig(save_path_prefix + "_roc_curve.png")
        plt.close()
    except:
        pass


# Logger 类
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['RGCN', 'HAN', 'HGT'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--target_rel_id', type=int, default=14, help="ID for Hard/Cold sets")
    args = parser.parse_args()

    # 日志设置
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sys.stdout = Logger(os.path.join(log_dir, f"{args.model}_{timestamp}.log"))

    # 路径配置
    BASE_DATA = "../data/benchmark"
    nodes_path = f"{BASE_DATA}/PrimeKG/nodes.csv"
    train_path = f"{BASE_DATA}/PrimeKG/train_edges.csv"
    val_path = f"{BASE_DATA}/PrimeKG/val_edges.csv"
    test_path = f"{BASE_DATA}/Kaggle_drug_repositioning/test.csv"
    test_hard_path = f"{BASE_DATA}/Kaggle_drug_repositioning/test_hard.csv"
    test_cold_path = f"{BASE_DATA}/Kaggle_drug_repositioning/test_cold.csv"

    model_dir = f"../model/{args.model}"
    train_process_dir = f"../data/training_process/{args.model}"
    eval_dir = f"../data/evaluation/{args.model}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(train_process_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    config = {'max_epochs': 100, 'patience': 10, 'lr': 0.001,
              'best_model_path': os.path.join(model_dir, f"{args.model.lower()}_best.pt")}

    # 加载数据
    data, datasets, num_nodes, num_rels, _ = load_and_build_data(
        nodes_path, train_path, val_path, test_path,
        test_hard_path=test_hard_path, test_cold_path=test_cold_path
    )

    train_loader = create_loader(data, datasets['train'], 4096, [20, 10], shuffle=True)
    val_loader = create_loader(data, datasets['val'], 4096, [20, 10])
    test_loader = create_loader(data, datasets['test'], 4096, [20, 10])
    test_hard_loader = create_loader(data, datasets['test_hard'], 4096, [20, 10])
    test_cold_loader = create_loader(data, datasets['test_cold'], 4096, [20, 10]) if datasets['test_cold'] else None

    # 模型初始化
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.model == 'RGCN':
        model = RGCN(num_nodes, 128, num_rels).to(device)
    elif args.model == 'HGT':
        model = HGT(num_nodes, 128, num_rels, num_layers=2, num_heads=4).to(device)
    elif args.model == 'HAN':
        model = HAN(num_nodes, 128, num_rels, num_layers=2, num_heads=4).to(device)

    # 训练 (如果存在模型则跳过)
    if os.path.exists(config['best_model_path']):
        print(f"Skipping Training, loading {config['best_model_path']}")
        model.load_state_dict(torch.load(config['best_model_path']))
    else:
        history = train_model(model, train_loader, val_loader, datasets['train'][2], datasets['val'][2], device, config)
        pd.DataFrame(history).to_csv(os.path.join(train_process_dir, "train_log.csv"), index=False)
        model.load_state_dict(torch.load(config['best_model_path']))

    print("\n" + "=" * 40)
    print(f"EVALUATION PHASE (Rel ID: {args.target_rel_id} for Hard/Cold)")
    print("=" * 40)

    # 1. Standard: 不强制 ID
    evaluate_with_force_id(model, test_loader, datasets['test'][2], device,
                           save_path=os.path.join(eval_dir, "standard"),
                           force_rel_id=None)

    # 2. Hard: 强制 ID 7
    if test_hard_loader:
        print(f"\n=== Eval Hard (Force ID 7) ===")
        save_p = os.path.join(eval_dir, "hard")
        evaluate_with_force_id(model, test_hard_loader, datasets['test_hard'][2], device,
                               save_path=save_p,
                               force_rel_id=7) # <--- 修改为 7

    # === 3. Cold Test: Force ID 14 ===
    if test_cold_loader:
        print(f"\n=== Eval Cold (Force ID 14) ===")
        save_p = os.path.join(eval_dir, "cold")
        evaluate_with_force_id(model, test_cold_loader, datasets['test_cold'][2], device,
                               save_path=save_p,
                               force_rel_id=14) # <--- 保持 14


if __name__ == "__main__":
    main()