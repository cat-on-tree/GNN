import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 🔥 新增绘图库
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def calculate_precision_at_k(y_true, y_score, k):
    """计算 Top-K 的 Precision"""
    if k > len(y_true):
        k = len(y_true)

    # 构建 DataFrame 方便排序
    df = pd.DataFrame({'label': y_true, 'score': y_score})
    # 按分数降序排列
    df = df.sort_values('score', ascending=False)

    # 取前 K 个
    top_k = df.head(k)

    # 计算正样本比例
    precision = top_k['label'].sum() / k
    return precision


def main():
    parser = argparse.ArgumentParser(description="Calculate Imbalanced Metrics (AUPRC/P@K) for Logprobs")
    parser.add_argument("--input_jsonl", required=True,
                        help="Input JSONL file (e.g., qwen3-32b_prob_debug.jsonl)")
    parser.add_argument("--output_dir", default="../data/evaluation/llm_result_auprc",
                        help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary classification (default 0.5)")

    args = parser.parse_args()

    # 1. 准备路径
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 从文件名提取模型前缀
    base_name = os.path.basename(args.input_jsonl)
    if '_' in base_name:
        model_prefix = base_name.split('_')[0]
    else:
        model_prefix = os.path.splitext(base_name)[0]

    txt_path = os.path.join(args.output_dir, f"{model_prefix}_metrics_auprc.txt")
    raw_pred_path = os.path.join(args.output_dir, f"{model_prefix}_raw_pred.csv")
    pr_curve_csv_path = os.path.join(args.output_dir, f"{model_prefix}_pr_curve.csv")
    pr_curve_img_path = os.path.join(args.output_dir, f"{model_prefix}_pr_curve.png")  # 🔥 图片路径

    print(f"🚀 Loading data from {args.input_jsonl}...")

    # 2. 读取数据 (保持原有逻辑不变)
    labels = []
    scores = []
    valid_indices = []
    skipped_count = 0

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                idx = item.get('index')
                label = item.get('label')
                prob = item.get('pred_prob')

                # === 异常检测 ===
                if label is None:
                    label = item.get('ground_truth')

                if label is None:
                    print(f"⚠️ [Idx {idx}] Missing label. Skipping.")
                    skipped_count += 1
                    continue

                if prob is None:
                    print(f"⚠️ [Idx {idx}] Missing pred_prob. Treating as 0.0")
                    prob = 0.0

                labels.append(int(label))
                scores.append(float(prob))
                valid_indices.append(idx)

            except json.JSONDecodeError:
                print("❌ Found invalid JSON line. Skipping.")
                skipped_count += 1
                continue

    y_true = np.array(labels)
    y_score = np.array(scores)

    if len(y_true) == 0:
        print("❌ Error: No valid data found to calculate metrics.")
        return

    print(f"✅ Loaded {len(y_true)} valid samples.")
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} invalid samples.")

    # 3. 计算指标

    # 二值化预测
    y_pred_binary = (y_score >= args.threshold).astype(int)

    # AUPRC 计算
    try:
        if len(np.unique(y_true)) > 1:
            auprc = average_precision_score(y_true, y_score)
            baseline_auprc = sum(y_true) / len(y_true)
        else:
            auprc = 0.0
            baseline_auprc = 0.0
            print("⚠️ Warning: Only one class present. AUPRC set to 0.0.")
    except ValueError as e:
        auprc = 0.0
        baseline_auprc = 0.0
        print(f"⚠️ Warning: Could not calculate AUPRC. Error: {e}")

    # 常规指标
    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Top-K Precision
    k_values = [10, 50, 100, int(len(y_true) * 0.01), int(len(y_true) * 0.05)]
    p_at_k_results = {}
    for k in k_values:
        if k > 0:
            p_at_k = calculate_precision_at_k(y_true, y_score, k)
            p_at_k_results[f"P@{k}"] = p_at_k

    # 4. 生成报告
    report_content = (
        f"Imbalanced Evaluation Report: {model_prefix}\n"
        f"========================================\n"
        f"Timestamp: {pd.Timestamp.now()}\n"
        f"Input File: {base_name}\n"
        f"Total Samples: {len(y_true)}\n"
        f"Positive Samples: {sum(y_true)}\n"
        f"Negative Samples: {len(y_true) - sum(y_true)}\n"
        f"Ratio (Neg/Pos): {(len(y_true) - sum(y_true)) / max(1, sum(y_true)):.2f}\n"
        f"Threshold: {args.threshold}\n\n"
        f"Key Metrics (Imbalanced):\n"
        f"------------------------\n"
        f"AUPRC:           {auprc:.4f} (Baseline: {baseline_auprc:.4f})\n"
        f"Precision:       {precision:.4f}\n"
        f"Recall:          {recall:.4f}\n"
        f"F1 Score:        {f1:.4f}\n"
        f"Specificity:     {specificity:.4f}\n\n"
        f"Top-K Metrics (Screening Power):\n"
        f"------------------------------\n"
    )
    for k_name, val in p_at_k_results.items():
        report_content += f"{k_name:<10}: {val:.4f}\n"

    print("\n" + report_content)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"💾 Metrics saved to: {txt_path}")

    # 5. 保存详细预测表
    df_raw = pd.DataFrame({
        'index': valid_indices,
        'label': y_true,
        'pred_prob': y_score,
        'pred_class': y_pred_binary
    })
    df_raw.to_csv(raw_pred_path, index=False)

    # 6. 计算并保存 PR 曲线数据
    if len(np.unique(y_true)) > 1:
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_score)

        # 保存 CSV
        df_pr = pd.DataFrame({
            'precision': precision_curve[:-1],
            'recall': recall_curve[:-1],
            'threshold': thresholds_pr
        })
        df_pr.to_csv(pr_curve_csv_path, index=False)
        print(f"💾 PR Curve data saved to: {pr_curve_csv_path}")

        # === 🔥 7. 绘制 PR 曲线图片 ===
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='darkorange', lw=2,
                 label=f'PR Curve (AUPRC = {auprc:.3f})')

        # 绘制 Baseline (随机线)
        plt.plot([0, 1], [baseline_auprc, baseline_auprc], linestyle='--', color='navy',
                 label=f'Random Baseline ({baseline_auprc:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title(f'Precision-Recall Curve: {model_prefix}')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        plt.savefig(pr_curve_img_path, dpi=300)
        plt.close()  # 关闭画布释放内存
        print(f"🖼️  PR Curve Plot saved to: {pr_curve_img_path}")
    else:
        print("⚠️ Skipping PR curve generation (single class).")


if __name__ == "__main__":
    main()