import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Basic normalization
# ============================================================

def normalize_name(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def normalize_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


# ============================================================
# Load LLM JSONL results
# ============================================================

def read_jsonl(path):
    rows = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                obj["_source_jsonl"] = str(path)
                rows.append(obj)
            except Exception as e:
                print(f"[WARN] Failed to parse {path}:{line_no}: {e}")

    return rows


def discover_result_files(result_dir, pattern, exclude_keywords):
    result_dir = Path(result_dir)
    files = sorted(result_dir.glob(pattern))

    exclude_keywords = [x.lower() for x in exclude_keywords]
    kept = []

    for p in files:
        s = str(p).lower()
        if any(k in s for k in exclude_keywords):
            print(f"[INFO] Skip excluded result file: {p}")
            continue
        kept.append(p)

    if not kept:
        raise FileNotFoundError(
            f"No result JSONL files found in {result_dir} with pattern={pattern}"
        )

    return kept


def load_llm_results(result_dir, pattern, exclude_keywords):
    files = discover_result_files(result_dir, pattern, exclude_keywords)

    all_rows = []
    for p in files:
        rows = read_jsonl(p)
        print(f"Loaded {len(rows):,} rows from {p}")
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError("No LLM result rows loaded.")

    df = pd.DataFrame(all_rows)

    required = [
        "model",
        "test_file",
        "negative_sampling_strategy",
        "negative_ratio",
        "drug_name",
        "label",
        "pred_prob",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in LLM JSONL: {missing}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="coerce")
    df["negative_ratio"] = pd.to_numeric(df["negative_ratio"], errors="coerce")

    before = len(df)
    df = df.dropna(
        subset=[
            "model",
            "test_file",
            "negative_sampling_strategy",
            "negative_ratio",
            "drug_name",
            "label",
            "pred_prob",
        ]
    )
    after = len(df)

    if after < before:
        print(f"[WARN] Dropped {before - after:,} invalid LLM rows.")

    df["label"] = df["label"].astype(int)
    df["negative_ratio"] = df["negative_ratio"].astype(int)
    df["drug_name_norm"] = df["drug_name"].apply(normalize_name)

    return df


def apply_model_renames(df, rename_model):
    """
    Safe simultaneous rename.

    Example:
      --rename_model qwen3-8b-sft=qwen3-8b-kto qwen3-8b-kto=qwen3-8b-sft
    """
    if not rename_model:
        return df

    mapping = {}
    for item in rename_model:
        if "=" not in item:
            raise ValueError(f"Invalid --rename_model item: {item}. Expected OLD=NEW")
        old, new = item.split("=", 1)
        mapping[old] = new

    df = df.copy()
    df["model_original"] = df["model"]
    df["model"] = df["model"].map(mapping).fillna(df["model"])

    print("[INFO] Applied model renaming:")
    for old, new in mapping.items():
        print(f"  {old} -> {new}")

    return df


# ============================================================
# Compute training drug indication degree
# ============================================================

def load_nodes(nodes_csv):
    nodes = pd.read_csv(nodes_csv, low_memory=False)

    required = ["node_index", "node_id", "node_type", "node_name"]
    missing = [c for c in required if c not in nodes.columns]
    if missing:
        raise ValueError(f"Missing required columns in nodes.csv: {missing}")

    nodes = nodes[["node_index", "node_id", "node_type", "node_name"]].copy()
    nodes["node_index"] = pd.to_numeric(nodes["node_index"], errors="coerce")
    nodes = nodes.dropna(subset=["node_index"])
    nodes["node_index"] = nodes["node_index"].astype(int)

    nodes["node_id_norm"] = nodes["node_id"].apply(normalize_id)
    nodes["node_type"] = nodes["node_type"].astype(str)
    nodes["node_name"] = nodes["node_name"].astype(str)
    nodes["node_name_norm"] = nodes["node_name"].apply(normalize_name)

    return nodes


def compute_train_drug_indication_degree(train_csv, nodes_csv, output_degree_csv=None):
    """
    Compute drug indication degree from TxGNN train split only.

    Important:
      TxGNN train.csv x_idx/y_idx may be internal indices, not PrimeKG node_index.
      Therefore we map train.csv x_id/y_id to nodes.csv node_id.
    """
    train = pd.read_csv(train_csv, low_memory=False)
    nodes = load_nodes(nodes_csv)

    required = ["x_type", "x_id", "relation", "y_type", "y_id"]
    missing = [c for c in required if c not in train.columns]
    if missing:
        raise ValueError(f"Missing required columns in train.csv: {missing}")

    train = train.copy()
    train["relation"] = train["relation"].astype(str)
    train["x_type"] = train["x_type"].astype(str)
    train["y_type"] = train["y_type"].astype(str)
    train["x_id_norm"] = train["x_id"].apply(normalize_id)
    train["y_id_norm"] = train["y_id"].apply(normalize_id)

    node_x = nodes.rename(columns={
        "node_id_norm": "x_id_norm",
        "node_index": "x_node_index",
        "node_type": "x_node_type",
        "node_name": "x_name",
        "node_name_norm": "x_name_norm",
    })

    node_y = nodes.rename(columns={
        "node_id_norm": "y_id_norm",
        "node_index": "y_node_index",
        "node_type": "y_node_type",
        "node_name": "y_name",
        "node_name_norm": "y_name_norm",
    })

    train = train.merge(
        node_x[["x_id_norm", "x_node_type", "x_node_index", "x_name", "x_name_norm"]],
        on="x_id_norm",
        how="left",
    )

    train = train.merge(
        node_y[["y_id_norm", "y_node_type", "y_node_index", "y_name", "y_name_norm"]],
        on="y_id_norm",
        how="left",
    )

    # Keep rows whose mapped node types match endpoint types when mapping is available.
    train = train[
        (train["x_node_type"].isna() | (train["x_node_type"] == train["x_type"]))
        & (train["y_node_type"].isna() | (train["y_node_type"] == train["y_type"]))
    ].copy()

    # Direction-agnostic drug-disease indication edges.
    xy = train[
        (train["relation"] == "indication")
        & (train["x_type"] == "drug")
        & (train["y_type"] == "disease")
    ][["x_node_index", "x_name", "x_name_norm", "y_node_index", "y_name", "y_name_norm"]].copy()

    xy = xy.rename(columns={
        "x_node_index": "drug_node_index",
        "x_name": "drug_name",
        "x_name_norm": "drug_name_norm",
        "y_node_index": "disease_node_index",
        "y_name": "disease_name",
        "y_name_norm": "disease_name_norm",
    })

    yx = train[
        (train["relation"] == "indication")
        & (train["x_type"] == "disease")
        & (train["y_type"] == "drug")
    ][["y_node_index", "y_name", "y_name_norm", "x_node_index", "x_name", "x_name_norm"]].copy()

    yx = yx.rename(columns={
        "y_node_index": "drug_node_index",
        "y_name": "drug_name",
        "y_name_norm": "drug_name_norm",
        "x_node_index": "disease_node_index",
        "x_name": "disease_name",
        "x_name_norm": "disease_name_norm",
    })

    indication_edges = pd.concat([xy, yx], ignore_index=True)
    indication_edges = indication_edges.dropna(subset=["drug_node_index", "disease_node_index"])

    print(f"[INFO] Training indication drug-disease edges after node mapping: {len(indication_edges):,}")

    if len(indication_edges) == 0:
        raise ValueError(
            "No indication edges were mapped to nodes. "
            "Check whether train.csv x_id/y_id match nodes.csv node_id."
        )

    # All drug nodes, retaining drugs with zero indication degree.
    drug_nodes = nodes[nodes["node_type"] == "drug"][["node_index", "node_name", "node_name_norm"]].copy()
    drug_nodes = drug_nodes.rename(columns={
        "node_index": "drug_node_index",
        "node_name": "drug_name",
        "node_name_norm": "drug_name_norm",
    })

    degree = (
        indication_edges
        .groupby(["drug_node_index", "drug_name", "drug_name_norm"])["disease_node_index"]
        .nunique()
        .reset_index(name="drug_indication_degree")
    )

    degree = drug_nodes.merge(
        degree[["drug_node_index", "drug_indication_degree"]],
        on="drug_node_index",
        how="left",
    )

    degree["drug_indication_degree"] = degree["drug_indication_degree"].fillna(0).astype(int)
    degree["log1p_drug_indication_degree"] = np.log1p(
        degree["drug_indication_degree"].astype(float)
    )

    degree_by_name = (
        degree
        .groupby("drug_name_norm", as_index=False)
        .agg(
            drug_name=("drug_name", "first"),
            drug_indication_degree=("drug_indication_degree", "max"),
            log1p_drug_indication_degree=("log1p_drug_indication_degree", "max"),
        )
    )

    if output_degree_csv:
        output_degree_csv = Path(output_degree_csv)
        output_degree_csv.parent.mkdir(parents=True, exist_ok=True)
        degree_by_name.to_csv(output_degree_csv, index=False)
        print(f"[INFO] Saved training drug degree table to: {output_degree_csv}")

    print("[INFO] Drug indication degree summary:")
    print(degree_by_name["drug_indication_degree"].describe().to_string())

    return degree_by_name


# ============================================================
# Top-K false-positive degree enrichment
# ============================================================

def compute_topk_fp_enrichment(df, top_k=100):
    """
    For each model/test_file/ratio:
      1. Sort candidates by pred_prob descending.
      2. Select TopK.
      3. Among TopK, identify false positives: label == 0.
      4. Compute mean log1p drug degree among TopK false positives.
      5. Subtract mean log1p drug degree among all negatives.

    Main metric:
      fp_log_degree_enrichment =
        mean_log_degree_topk_false_positives - mean_log_degree_all_negatives
    """
    rows = []

    group_cols = [
        "model",
        "negative_sampling_strategy",
        "negative_ratio",
        "test_file",
    ]

    for keys, g in df.groupby(group_cols, dropna=False):
        model, strategy, ratio, test_file = keys

        g = g.sort_values("pred_prob", ascending=False, kind="mergesort").copy()
        top = g.head(min(top_k, len(g))).copy()

        neg = g[g["label"] == 0].copy()
        top_fp = top[top["label"] == 0].copy()
        top_tp = top[top["label"] == 1].copy()

        n_total = len(g)
        n_pos = int((g["label"] == 1).sum())
        n_neg = int((g["label"] == 0).sum())
        n_top = len(top)
        n_top_fp = len(top_fp)
        n_top_tp = len(top_tp)

        topk_precision = n_top_tp / n_top if n_top > 0 else np.nan
        topk_fp_rate = n_top_fp / n_top if n_top > 0 else np.nan

        all_neg_mean_log_degree = (
            float(neg["log1p_drug_indication_degree"].mean())
            if n_neg > 0 else np.nan
        )
        all_neg_mean_degree = (
            float(neg["drug_indication_degree"].mean())
            if n_neg > 0 else np.nan
        )

        top_fp_mean_log_degree = (
            float(top_fp["log1p_drug_indication_degree"].mean())
            if n_top_fp > 0 else np.nan
        )
        top_fp_mean_degree = (
            float(top_fp["drug_indication_degree"].mean())
            if n_top_fp > 0 else np.nan
        )

        # Difference on log degree scale.
        fp_log_degree_enrichment = (
            top_fp_mean_log_degree - all_neg_mean_log_degree
            if np.isfinite(top_fp_mean_log_degree) and np.isfinite(all_neg_mean_log_degree)
            else np.nan
        )

        # Raw degree ratio, using +1 to avoid division by zero instability.
        fp_degree_ratio_plus1 = (
            (top_fp_mean_degree + 1.0) / (all_neg_mean_degree + 1.0)
            if np.isfinite(top_fp_mean_degree) and np.isfinite(all_neg_mean_degree)
            else np.nan
        )

        # Also compute TopK all-candidate degree shift as a secondary diagnostic.
        top_mean_log_degree = (
            float(top["log1p_drug_indication_degree"].mean())
            if n_top > 0 else np.nan
        )
        all_candidate_mean_log_degree = (
            float(g["log1p_drug_indication_degree"].mean())
            if n_total > 0 else np.nan
        )
        top_all_log_degree_enrichment = (
            top_mean_log_degree - all_candidate_mean_log_degree
            if np.isfinite(top_mean_log_degree) and np.isfinite(all_candidate_mean_log_degree)
            else np.nan
        )

        rows.append({
            "model": model,
            "negative_sampling_strategy": strategy,
            "negative_ratio": ratio,
            "test_file": test_file,
            "top_k": top_k,
            "n_total": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_top": n_top,
            "n_top_tp": n_top_tp,
            "n_top_fp": n_top_fp,
            "topk_precision": topk_precision,
            "topk_fp_rate": topk_fp_rate,
            "all_neg_mean_log1p_drug_degree": all_neg_mean_log_degree,
            "all_neg_mean_drug_degree": all_neg_mean_degree,
            "topk_fp_mean_log1p_drug_degree": top_fp_mean_log_degree,
            "topk_fp_mean_drug_degree": top_fp_mean_degree,
            "fp_log_degree_enrichment": fp_log_degree_enrichment,
            "fp_degree_ratio_plus1": fp_degree_ratio_plus1,
            "topk_all_mean_log1p_drug_degree": top_mean_log_degree,
            "all_candidate_mean_log1p_drug_degree": all_candidate_mean_log_degree,
            "top_all_log_degree_enrichment": top_all_log_degree_enrichment,
        })

    return pd.DataFrame(rows)


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), 0.0

    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def summarize_enrichment(per_file):
    rows = []

    group_cols = [
        "model",
        "negative_sampling_strategy",
        "negative_ratio",
        "top_k",
    ]

    metrics = [
        "topk_precision",
        "topk_fp_rate",
        "n_top_fp",
        "all_neg_mean_log1p_drug_degree",
        "topk_fp_mean_log1p_drug_degree",
        "fp_log_degree_enrichment",
        "all_neg_mean_drug_degree",
        "topk_fp_mean_drug_degree",
        "fp_degree_ratio_plus1",
        "top_all_log_degree_enrichment",
    ]

    for keys, g in per_file.groupby(group_cols, dropna=False):
        model, strategy, ratio, top_k = keys

        row = {
            "model": model,
            "negative_sampling_strategy": strategy,
            "negative_ratio": ratio,
            "top_k": top_k,
            "n_files": int(g["test_file"].nunique()),
        }

        for m in metrics:
            mean, std = mean_std(g[m].values)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values([
        "negative_sampling_strategy",
        "negative_ratio",
        "model",
    ]).reset_index(drop=True)

    return summary


# ============================================================
# GraphPad export
# ============================================================

def export_graphpad_tables(per_file, summary, output_dir, model_order=None):
    output_dir = Path(output_dir)
    graphpad_dir = output_dir / "graphpad"
    graphpad_dir.mkdir(parents=True, exist_ok=True)

    pf = per_file.copy()
    sm = summary.copy()

    if model_order is None:
        model_order = sorted(pf["model"].dropna().unique().tolist())
    else:
        model_order = [m for m in model_order if m in set(pf["model"])]

    ratios = sorted(pf["negative_ratio"].dropna().unique().tolist())

    # Long replicate table.
    tidy_cols = [
        "model",
        "negative_sampling_strategy",
        "negative_ratio",
        "test_file",
        "top_k",
        "topk_precision",
        "topk_fp_rate",
        "n_top_fp",
        "all_neg_mean_log1p_drug_degree",
        "topk_fp_mean_log1p_drug_degree",
        "fp_log_degree_enrichment",
        "all_neg_mean_drug_degree",
        "topk_fp_mean_drug_degree",
        "fp_degree_ratio_plus1",
        "top_all_log_degree_enrichment",
    ]

    tidy = pf[tidy_cols].copy()
    tidy = tidy.sort_values(["negative_ratio", "model", "test_file"])

    tidy_path = graphpad_dir / "graphpad_tidy_topk_fp_enrichment_per_file.csv"
    tidy.to_csv(tidy_path, index=False)
    print(f"[INFO] Saved GraphPad tidy table to: {tidy_path}")

    # Replicate table for main metric.
    rep_rows = []

    for model in model_order:
        row = {"model": model}

        for ratio in ratios:
            vals = (
                pf[
                    (pf["model"] == model)
                    & (pf["negative_ratio"] == ratio)
                ]
                .sort_values("test_file")["fp_log_degree_enrichment"]
                .tolist()
            )

            max_reps = (
                pf[pf["negative_ratio"] == ratio]
                .groupby("model")["test_file"]
                .nunique()
                .max()
            )

            if pd.isna(max_reps):
                max_reps = 0

            for i, v in enumerate(vals, start=1):
                row[f"ratio_{ratio}_rep{i}"] = v

            for i in range(len(vals) + 1, int(max_reps) + 1):
                row[f"ratio_{ratio}_rep{i}"] = np.nan

        rep_rows.append(row)

    rep_df = pd.DataFrame(rep_rows)
    rep_path = graphpad_dir / "graphpad_replicates_fp_log_degree_enrichment.csv"
    rep_df.to_csv(rep_path, index=False)
    print(f"[INFO] Saved GraphPad replicate table to: {rep_path}")

    # Mean/SD table for main metric.
    mean_sd_rows = []

    for model in model_order:
        row = {"model": model}

        for ratio in ratios:
            tmp = sm[
                (sm["model"] == model)
                & (sm["negative_ratio"] == ratio)
            ]

            if len(tmp) == 0:
                row[f"ratio_{ratio}_mean"] = np.nan
                row[f"ratio_{ratio}_sd"] = np.nan
                row[f"ratio_{ratio}_n_files"] = np.nan
            else:
                rec = tmp.iloc[0]
                row[f"ratio_{ratio}_mean"] = rec["fp_log_degree_enrichment_mean"]
                row[f"ratio_{ratio}_sd"] = rec["fp_log_degree_enrichment_std"]
                row[f"ratio_{ratio}_n_files"] = rec["n_files"]

        mean_sd_rows.append(row)

    mean_sd_df = pd.DataFrame(mean_sd_rows)
    mean_sd_path = graphpad_dir / "graphpad_mean_sd_fp_log_degree_enrichment.csv"
    mean_sd_df.to_csv(mean_sd_path, index=False)
    print(f"[INFO] Saved GraphPad mean/SD table to: {mean_sd_path}")

    # Extra summary table with several metrics.
    summary_path = graphpad_dir / "graphpad_summary_all_metrics.csv"
    sm.to_csv(summary_path, index=False)
    print(f"[INFO] Saved GraphPad summary all metrics to: {summary_path}")


# ============================================================
# Plot
# ============================================================

def plot_enrichment(summary, output_png, output_pdf=None, model_order=None):
    plot_df = summary.copy()

    if plot_df.empty:
        raise ValueError("No data to plot.")

    if model_order is None:
        model_order = sorted(plot_df["model"].unique().tolist())
    else:
        model_order = [m for m in model_order if m in set(plot_df["model"])]

    ratios = sorted(plot_df["negative_ratio"].unique().tolist())

    ratio_colors = {
        1: "#4C78A8",
        4: "#F58518",
        9: "#54A24B",
    }

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(max(8, 0.75 * len(model_order)), 4.8),
    )

    width = 0.22
    x = np.arange(len(model_order))

    for j, ratio in enumerate(ratios):
        tmp = plot_df[plot_df["negative_ratio"] == ratio].set_index("model")

        means = []
        stds = []
        for m in model_order:
            if m in tmp.index:
                means.append(tmp.loc[m, "fp_log_degree_enrichment_mean"])
                stds.append(tmp.loc[m, "fp_log_degree_enrichment_std"])
            else:
                means.append(np.nan)
                stds.append(0.0)

        offset = (j - (len(ratios) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=f"1:{ratio}",
            color=ratio_colors.get(ratio, None),
            alpha=0.9,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=35, ha="right")
    ax.set_ylabel("Top100 FP degree enrichment\nΔ mean log1p(training drug degree)")
    ax.set_title("Top100 false-positive drug-degree enrichment")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Neg. ratio", loc="best")

    fig.tight_layout()

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved PNG figure to: {output_png}")

    if output_pdf:
        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
        print(f"[INFO] Saved PDF figure to: {output_pdf}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze TopK false-positive drug-degree enrichment for LLM results."
    )

    parser.add_argument(
        "--result_dir",
        default="data/result_llm",
        help="Directory containing LLM JSONL result files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern for JSONL files.",
    )
    parser.add_argument(
        "--exclude_keywords",
        nargs="*",
        default=["debug", "tmp"],
        help="Exclude result files whose path contains these keywords.",
    )

    parser.add_argument(
        "--train_csv",
        default="data/benchmark/PrimeKG/full_graph_42/train.csv",
        help="TxGNN training split CSV.",
    )
    parser.add_argument(
        "--nodes_csv",
        default="data/benchmark/PrimeKG/nodes.csv",
        help="PrimeKG nodes.csv.",
    )

    parser.add_argument(
        "--graph_group",
        default="pyg_style",
        help="Optional graph_group filter. Use 'ALL' to disable.",
    )
    parser.add_argument(
        "--strategy",
        default="degree_matched_head",
        help="Negative sampling strategy to analyze.",
    )
    parser.add_argument(
        "--ratios",
        nargs="*",
        type=int,
        default=[1, 4, 9],
        help="Negative ratios to include.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top-K threshold for false-positive enrichment. Default: 100.",
    )

    parser.add_argument(
        "--rename_model",
        nargs="*",
        default=None,
        help="Safe simultaneous rename pairs OLD=NEW.",
    )
    parser.add_argument(
        "--model_order",
        nargs="*",
        default=None,
        help="Optional order of models in plot and GraphPad tables.",
    )

    parser.add_argument(
        "--output_dir",
        default="data/result_llm/degree_bias_head/fp_enrichment",
        help="Output directory.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    degree_csv = output_dir / "train_drug_indication_degrees.csv"

    degree = compute_train_drug_indication_degree(
        train_csv=args.train_csv,
        nodes_csv=args.nodes_csv,
        output_degree_csv=degree_csv,
    )

    df = load_llm_results(
        result_dir=args.result_dir,
        pattern=args.pattern,
        exclude_keywords=args.exclude_keywords,
    )

    df = apply_model_renames(df, args.rename_model)

    # Filter target setting.
    df = df[df["negative_sampling_strategy"] == args.strategy].copy()
    df = df[df["negative_ratio"].isin(args.ratios)].copy()

    if args.graph_group != "ALL":
        if "graph_group" not in df.columns:
            raise ValueError("graph_group column not found in LLM results.")
        df = df[df["graph_group"] == args.graph_group].copy()

    if df.empty:
        raise ValueError("No LLM rows left after filtering.")

    # Merge drug training indication degree by normalized drug name.
    df = df.merge(
        degree[[
            "drug_name_norm",
            "drug_indication_degree",
            "log1p_drug_indication_degree",
        ]],
        on="drug_name_norm",
        how="left",
    )

    missing = df["drug_indication_degree"].isna().sum()
    if missing > 0:
        missing_drugs = sorted(
            df.loc[df["drug_indication_degree"].isna(), "drug_name"]
            .dropna()
            .unique()
            .tolist()
        )
        print(
            f"[WARN] Missing training degree for {missing:,} rows, "
            f"{len(missing_drugs):,} unique drugs."
        )
        print("[WARN] First 30 missing drugs:", missing_drugs[:30])

        # Missing means not mapped to PrimeKG nodes; drop rather than treating as zero.
        df = df.dropna(subset=["drug_indication_degree"]).copy()

    df["drug_indication_degree"] = df["drug_indication_degree"].astype(int)
    df["log1p_drug_indication_degree"] = df["log1p_drug_indication_degree"].astype(float)

    merged_csv = output_dir / "llm_results_with_train_drug_degree.csv"
    df.to_csv(merged_csv, index=False)
    print(f"[INFO] Saved merged analysis rows to: {merged_csv}")

    print("=" * 80)
    print("TopK false-positive drug-degree enrichment analysis")
    print("=" * 80)
    print(f"Rows: {len(df):,}")
    print(f"Models: {sorted(df['model'].unique().tolist())}")
    print(f"Strategy: {args.strategy}")
    print(f"TopK: {args.top_k}")
    print(f"Ratios: {sorted(df['negative_ratio'].unique().tolist())}")
    if "graph_group" in df.columns:
        print(f"Graph groups: {sorted(df['graph_group'].dropna().unique().tolist())}")
    print("=" * 80)

    per_file = compute_topk_fp_enrichment(df, top_k=args.top_k)
    summary = summarize_enrichment(per_file)

    per_file_csv = output_dir / f"top{args.top_k}_fp_degree_enrichment_per_file.csv"
    summary_csv = output_dir / f"top{args.top_k}_fp_degree_enrichment_summary.csv"

    per_file.to_csv(per_file_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"[INFO] Saved per-file enrichment table to: {per_file_csv}")
    print(f"[INFO] Saved summary enrichment table to: {summary_csv}")

    export_graphpad_tables(
        per_file=per_file,
        summary=summary,
        output_dir=output_dir,
        model_order=args.model_order,
    )

    png = output_dir / f"figure_C_top{args.top_k}_fp_degree_enrichment.png"
    pdf = output_dir / f"figure_C_top{args.top_k}_fp_degree_enrichment.pdf"

    plot_enrichment(
        summary=summary,
        output_png=png,
        output_pdf=pdf,
        model_order=args.model_order,
    )

    print("\nSummary table:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()