import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve


def main():
    parser = argparse.ArgumentParser(description="Step 3: Calculate Metrics and Generate ROC Data")
    parser.add_argument("--input_jsonl", required=True,
                        help="Input JSONL file with scores (e.g., qwen3-8b_scored.jsonl)")
    parser.add_argument("--output_dir", default="../data/evaluation/llm_result", help="Directory to save results")

    args = parser.parse_args()

    # 1. 准备路径
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 从文件名提取前缀 (例如 qwen3-8b)
    base_name = os.path.basename(args.input_jsonl)
    if '_' in base_name:
        model_prefix = base_name.split('_')[0]
    else:
        model_prefix = os.path.splitext(base_name)[0]

    txt_path = os.path.join(args.output_dir, f"{model_prefix}_result.txt")
    raw_pred_path = os.path.join(args.output_dir, f"{model_prefix}_raw_pred.csv")
    roc_data_path = os.path.join(args.output_dir, f"{model_prefix}_roc_curve_data.csv")

    print(f"Loading data from {args.input_jsonl}...")

    # 2. 读取数据并处理异常
    labels = []
    scores = []
    valid_indices = []
    error_count = 0

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            idx = item.get('index')
            label = item.get('label')
            raw_score = item.get('pred_score')

            # === 异常处理核心逻辑 ===
            # 情况1: 缺失字段
            if label is None or raw_score is None:
                print(f"Warning: Sample index {idx} missing label or score. Skipping.")
                error_count += 1
                continue

            # 情况2: score 为 -1 (API 失败)
            if raw_score == -1:
                # 策略A (默认): 剔除该样本，不参与计算
                # print(f"Warning: Sample index {idx} has score -1 (API Error). Skipping.")
                # error_count += 1
                # continue

                # 策略B (可选): 视为 0 分 (如果需要保持样本总数不变，请取消下面两行的注释，并注释掉上面的 continue)
                raw_score = 0
                # print(f"Warning: Sample index {idx} has score -1. Treating as 0.")

            labels.append(int(label))
            # 归一化分数到 0-1 区间
            scores.append(float(raw_score) / 100.0)
            valid_indices.append(idx)

    # 转换为 numpy 数组
    y_true = np.array(labels)
    y_score = np.array(scores)

    if len(y_true) == 0:
        print("Error: No valid data found to calculate metrics.")
        return

    print(f"Total valid samples: {len(y_true)}")
    if error_count > 0:
        print(f"Skipped {error_count} invalid/error samples.")

    # 3. 计算指标
    # 为了计算 Accuracy, F1, Precision, Recall，我们需要一个阈值将概率转为 0/1
    # 这里默认使用 0.5 作为阈值 (对应原始分数 50分)
    threshold = 0.5
    y_pred_binary = (y_score >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError as e:
        auc = 0.0
        print(f"Warning: Could not calculate AUC (possibly only one class present). Error: {e}")

    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)

    # 4. 生成报告文本
    report_content = (
        f"Model Evaluation Report: {model_prefix}\n"
        f"========================================\n"
        f"Input File: {args.input_jsonl}\n"
        f"Total Samples: {len(y_true)}\n"
        f"Threshold: {threshold} (Raw Score 50)\n\n"
        f"Metrics:\n"
        f"--------\n"
        f"AUC:       {auc:.4f}\n"
        f"Accuracy:  {acc:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
    )

    print(report_content)

    # 保存 TXT
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Saved metrics to: {txt_path}")

    # 5. 保存 raw_pred (用于后续自定义分析)
    # 包含 index, label, score (normalized)
    df_raw = pd.DataFrame({
        'index': valid_indices,
        'label': y_true,
        'prob_score': y_score
    })
    df_raw.to_csv(raw_pred_path, index=False)
    print(f"Saved raw predictions to: {raw_pred_path}")

    # 6. 保存 roc_curve_data (直接用于绘图)
    # 包含 fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    df_roc = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds
    })
    df_roc.to_csv(roc_data_path, index=False)
    print(f"Saved ROC curve data to: {roc_data_path}")


if __name__ == "__main__":
    main()