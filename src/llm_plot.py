import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

# 设置绘图风格
plt.style.use('ggplot')


# ==========================================
# 核心：插值对齐函数
# ==========================================
def interpolate_curve(x, y, num_points=1001):
    """
    将曲线 (x, y) 插值到固定的 num_points 个点上，使 X 轴在 [0, 1] 均匀分布。
    用于对齐不同模型的 ROC/PRC 数据行数。
    """
    # 定义标准的 X 轴 (0.000, 0.001, ... 1.000)
    target_x = np.linspace(0, 1, num_points)

    # np.interp 要求 x 必须是单调递增的
    # 如果输入的 x 是递减的（例如 PRC 的 Recall 某些时候），需要翻转
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]

    # 执行线性插值
    interp_y = np.interp(target_x, x, y)

    return target_x, interp_y


def calculate_ranking_metrics(y_true, y_score, max_k=None):
    # (保持原逻辑不变，略，下方完整代码包含)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", default="../data/evaluation/llm_plot")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    base_name = os.path.basename(args.input_jsonl)
    model_prefix = base_name.split('_')[0] if '_' in base_name else os.path.splitext(base_name)[0]
    save_dir = os.path.join(args.output_dir, model_prefix)
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 Processing {args.input_jsonl}...")

    # 读取数据
    labels, scores = [], []
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                l = item.get('label', item.get('ground_truth'))
                p = item.get('pred_prob')
                if l is not None and p is not None:
                    labels.append(int(l))
                    scores.append(float(p))
            except:
                continue

    if len(labels) == 0: return

    y_true = np.array(labels)
    y_score = np.array(scores)

    # 计算标量指标
    try:
        auroc = roc_auc_score(y_true, y_score)
    except:
        auroc = 0.0
    auprc = average_precision_score(y_true, y_score)
    y_pred_bin = (y_score >= args.threshold).astype(int)
    f1 = f1_score(y_true, y_pred_bin)
    acc = accuracy_score(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)

    # Ranking 指标
    df_ranking = calculate_ranking_metrics(y_true, y_score, max_k=len(y_true))
    top_ks = [10, 50, 100]
    top_k_results = {}
    for k in top_ks:
        if k <= len(df_ranking):
            top_k_results[k] = df_ranking.loc[k - 1, 'precision_at_k']
        else:
            top_k_results[k] = 0.0

    # =========================================================
    # 生成对齐的 Plot Data (CSV)
    # =========================================================

    # 1. ROC Curve (统一插值到 1001 行)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    interp_fpr, interp_tpr = interpolate_curve(fpr, tpr, num_points=1001)
    df_roc = pd.DataFrame({'FPR': interp_fpr, 'TPR': interp_tpr})
    df_roc.to_csv(os.path.join(save_dir, f"{model_prefix}_roc_curve.csv"), index=False)

    # 2. PRC Curve (统一插值到 1001 行)
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    # Precision-Recall Curve 的 Recall 通常是从 1 到 0，interpolate_curve 会自动处理翻转
    interp_recall, interp_precision = interpolate_curve(recalls, precisions, num_points=1001)
    df_prc = pd.DataFrame({'Recall': interp_recall, 'Precision': interp_precision})
    df_prc.to_csv(os.path.join(save_dir, f"{model_prefix}_prc_curve.csv"), index=False)

    # 3. EF & CumGain (Top 20 已经是整数对齐，直接保存)
    df_top20 = df_ranking.head(20).copy()
    # 强制 K 为整数列，方便作图
    df_top20['k'] = df_top20['k'].astype(int)

    df_top20[['k', 'enrichment_factor']].to_csv(
        os.path.join(save_dir, f"{model_prefix}_ef_top20.csv"), index=False
    )
    df_top20[['k', 'cumulative_gain']].to_csv(
        os.path.join(save_dir, f"{model_prefix}_cumgain_top20.csv"), index=False
    )

    # 4. Top-K Bar
    df_topk_bar = pd.DataFrame({
        'Top_K': [f'Top-{k}' for k in top_ks],
        'Precision': [top_k_results[k] for k in top_ks]
    })
    df_topk_bar.to_csv(os.path.join(save_dir, f"{model_prefix}_topk_bar.csv"), index=False)

    # 5. Summary TXT
    with open(os.path.join(save_dir, f"{model_prefix}_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_prefix}\n")
        f.write(f"AUROC: {auroc:.6f}\n")
        f.write(f"AUPRC: {auprc:.6f}\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"F1: {f1:.6f}\n")
        for k in top_ks:
            f.write(f"Top-{k}: {top_k_results[k]:.6f}\n")

    # 绘图 (PNG预览)
    plt.figure();
    plt.plot(interp_fpr, interp_tpr);
    plt.title('ROC');
    plt.savefig(os.path.join(save_dir, f"{model_prefix}_plot_roc.png"));
    plt.close()
    plt.figure();
    plt.plot(interp_recall, interp_precision);
    plt.title('PRC');
    plt.savefig(os.path.join(save_dir, f"{model_prefix}_plot_prc.png"));
    plt.close()

    print(f"✅ Saved aligned data to {save_dir}")


if __name__ == "__main__":
    main()