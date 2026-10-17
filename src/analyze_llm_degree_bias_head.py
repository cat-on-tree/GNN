import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def normalize_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_name(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


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
        raise FileNotFoundError(f"No result JSONL files found in {result_dir} with pattern={pattern}")

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
    df = df.dropna(subset=["model", "test_file", "negative_sampling_strategy", "negative_ratio", "drug_name", "label", "pred_prob"])
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
      Therefore we map train.csv x_id/y_id to nodes.csv node_id, together with node_type.
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

    # In rare cases, node_id alone may not be globally unique across node types.
    # Keep rows whose mapped node types match train.csv endpoint types.
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
    degree["log1p_drug_indication_degree"] = np.log1p(degree["drug_indication_degree"].astype(float))

    # Name-based merge table for LLM result drug_name.
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


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    try:
        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        return np.nan, np.nan


def compute_per_file_correlations(df):
    rows = []

    group_cols = [
        "model",
        "negative_sampling_strategy",
        "negative_ratio",
        "test_file",
    ]

    for keys, g in df.groupby(group_cols, dropna=False):
        model, strategy, ratio, test_file = keys

        subsets = [
            ("all", g),
            ("negative_only", g[g["label"] == 0]),
        ]

        for subset_name, gg in subsets:
            if len(gg) < 3:
                continue

            rho, p = safe_spearman(
                gg["pred_prob"].values,
                gg["log1p_drug_indication_degree"].values,
            )

            rows.append({
                "model": model,
                "negative_sampling_strategy": strategy,
                "negative_ratio": ratio,
                "test_file": test_file,
                "subset": subset_name,
                "n": len(gg),
                "spearman_rho": rho,
                "spearman_p": p,
                "mean_score": float(np.mean(gg["pred_prob"].values)),
                "mean_log1p_drug_degree": float(np.mean(gg["log1p_drug_indication_degree"].values)),
                "mean_drug_degree": float(np.mean(gg["drug_indication_degree"].values)),
            })

    return pd.DataFrame(rows)


def summarize_correlations(per_file):
    rows = []

    group_cols = [
        "model",
        "negative_sampling_strategy",
        "negative_ratio",
        "subset",
    ]

    for keys, g in per_file.groupby(group_cols, dropna=False):
        model, strategy, ratio, subset = keys
        vals = g["spearman_rho"].dropna().astype(float).values

        if len(vals) == 0:
            mean = np.nan
            std = np.nan
        elif len(vals) == 1:
            mean = float(vals[0])
            std = 0.0
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))

        rows.append({
            "model": model,
            "negative_sampling_strategy": strategy,
            "negative_ratio": ratio,
            "subset": subset,
            "n_files": int(g["test_file"].nunique()),
            "spearman_rho_mean": mean,
            "spearman_rho_std": std,
            "n_mean": float(g["n"].mean()),
            "mean_score_mean": float(g["mean_score"].mean()),
            "mean_drug_degree_mean": float(g["mean_drug_degree"].mean()),
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values([
        "subset",
        "negative_ratio",
        "model",
    ]).reset_index(drop=True)

    return summary


def export_graphpad_tables(per_file, summary, output_dir, model_order=None):
    """
    Export GraphPad/Prism-friendly tables.

    Outputs:
      1. graphpad_tidy_per_file.csv
         Long-format replicate table. Recommended for R/Python/GraphPad import.

      2. graphpad_mean_sd_wide.csv
         One row per model/subset, columns are ratio-specific mean/std.

      3. graphpad_replicates_{subset}.csv
         Prism-friendly replicate table:
           Model, ratio_1_rep1, ..., ratio_1_rep5, ratio_4_rep1, ...

      4. graphpad_mean_sd_{subset}.csv
         Prism-friendly mean/SD table:
           Model, ratio_1_mean, ratio_1_sd, ratio_4_mean, ratio_4_sd, ...
    """
    output_dir = Path(output_dir)
    graphpad_dir = output_dir / "graphpad"
    graphpad_dir.mkdir(parents=True, exist_ok=True)

    pf = per_file.copy()
    sm = summary.copy()

    if model_order is None:
        model_order = sorted(pf["model"].dropna().unique().tolist())
    else:
        model_order = [m for m in model_order if m in set(pf["model"])]

    # Stable ratio ordering.
    ratios = sorted(pf["negative_ratio"].dropna().unique().tolist())
    subsets = sorted(pf["subset"].dropna().unique().tolist())

    # ------------------------------------------------------------
    # 1. Long tidy per-file replicate table.
    # ------------------------------------------------------------
    tidy = pf[[
        "model",
        "negative_sampling_strategy",
        "negative_ratio",
        "subset",
        "test_file",
        "spearman_rho",
        "n",
        "mean_score",
        "mean_log1p_drug_degree",
        "mean_drug_degree",
    ]].copy()

    tidy = tidy.sort_values([
        "subset",
        "negative_ratio",
        "model",
        "test_file",
    ])

    tidy_path = graphpad_dir / "graphpad_tidy_per_file.csv"
    tidy.to_csv(tidy_path, index=False)
    print(f"[INFO] Saved GraphPad tidy replicate table to: {tidy_path}")

    # ------------------------------------------------------------
    # 2. Wide summary table: one row per model/subset.
    # ------------------------------------------------------------
    wide_rows = []

    for subset in subsets:
        for model in model_order:
            row = {
                "subset": subset,
                "model": model,
            }

            for ratio in ratios:
                tmp = sm[
                    (sm["subset"] == subset)
                    & (sm["model"] == model)
                    & (sm["negative_ratio"] == ratio)
                ]

                if len(tmp) == 0:
                    row[f"ratio_{ratio}_mean"] = np.nan
                    row[f"ratio_{ratio}_sd"] = np.nan
                    row[f"ratio_{ratio}_n_files"] = np.nan
                else:
                    rec = tmp.iloc[0]
                    row[f"ratio_{ratio}_mean"] = rec["spearman_rho_mean"]
                    row[f"ratio_{ratio}_sd"] = rec["spearman_rho_std"]
                    row[f"ratio_{ratio}_n_files"] = rec["n_files"]

            wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    wide_path = graphpad_dir / "graphpad_mean_sd_wide.csv"
    wide.to_csv(wide_path, index=False)
    print(f"[INFO] Saved GraphPad wide mean/SD table to: {wide_path}")

    # ------------------------------------------------------------
    # 3. Per-subset replicate and mean/SD files.
    # ------------------------------------------------------------
    for subset in subsets:
        pf_sub = pf[pf["subset"] == subset].copy()
        sm_sub = sm[sm["subset"] == subset].copy()

        # Replicate table.
        rep_rows = []

        for model in model_order:
            row = {
                "model": model,
            }

            for ratio in ratios:
                vals = (
                    pf_sub[
                        (pf_sub["model"] == model)
                        & (pf_sub["negative_ratio"] == ratio)
                    ]
                    .sort_values("test_file")["spearman_rho"]
                    .tolist()
                )

                # Usually five seeds/files.
                for i, v in enumerate(vals, start=1):
                    row[f"ratio_{ratio}_rep{i}"] = v

                # Fill to max replicate count in this subset/ratio.
                max_reps = (
                    pf_sub[pf_sub["negative_ratio"] == ratio]
                    .groupby("model")["test_file"]
                    .nunique()
                    .max()
                )

                if pd.isna(max_reps):
                    max_reps = 0

                for i in range(len(vals) + 1, int(max_reps) + 1):
                    row[f"ratio_{ratio}_rep{i}"] = np.nan

            rep_rows.append(row)

        rep_df = pd.DataFrame(rep_rows)
        rep_path = graphpad_dir / f"graphpad_replicates_{subset}.csv"
        rep_df.to_csv(rep_path, index=False)
        print(f"[INFO] Saved GraphPad replicate table for {subset} to: {rep_path}")

        # Mean/SD table.
        mean_sd_rows = []

        for model in model_order:
            row = {
                "model": model,
            }

            for ratio in ratios:
                tmp = sm_sub[
                    (sm_sub["model"] == model)
                    & (sm_sub["negative_ratio"] == ratio)
                ]

                if len(tmp) == 0:
                    row[f"ratio_{ratio}_mean"] = np.nan
                    row[f"ratio_{ratio}_sd"] = np.nan
                    row[f"ratio_{ratio}_n_files"] = np.nan
                else:
                    rec = tmp.iloc[0]
                    row[f"ratio_{ratio}_mean"] = rec["spearman_rho_mean"]
                    row[f"ratio_{ratio}_sd"] = rec["spearman_rho_std"]
                    row[f"ratio_{ratio}_n_files"] = rec["n_files"]

            mean_sd_rows.append(row)

        mean_sd_df = pd.DataFrame(mean_sd_rows)
        mean_sd_path = graphpad_dir / f"graphpad_mean_sd_{subset}.csv"
        mean_sd_df.to_csv(mean_sd_path, index=False)
        print(f"[INFO] Saved GraphPad mean/SD table for {subset} to: {mean_sd_path}")


def plot_correlation(summary, output_png, output_pdf=None, model_order=None):
    plot_df = summary[
        summary["negative_sampling_strategy"].eq("degree_matched_head")
        & summary["subset"].isin(["all", "negative_only"])
    ].copy()

    if plot_df.empty:
        raise ValueError("No data to plot.")

    if model_order is None:
        model_order = sorted(plot_df["model"].unique().tolist())
    else:
        model_order = [m for m in model_order if m in set(plot_df["model"])]

    ratios = sorted(plot_df["negative_ratio"].unique().tolist())
    subsets = ["all", "negative_only"]

    ratio_colors = {
        1: "#4C78A8",
        4: "#F58518",
        9: "#54A24B",
    }

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(max(10, 0.75 * len(model_order) * 2), 4.6),
        sharey=True,
    )

    width = 0.22
    x = np.arange(len(model_order))

    for ax, subset in zip(axes, subsets):
        sub = plot_df[plot_df["subset"] == subset].copy()

        for j, ratio in enumerate(ratios):
            tmp = sub[sub["negative_ratio"] == ratio].set_index("model")

            means = []
            stds = []
            for m in model_order:
                if m in tmp.index:
                    means.append(tmp.loc[m, "spearman_rho_mean"])
                    stds.append(tmp.loc[m, "spearman_rho_std"])
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
        ax.set_title("All samples" if subset == "all" else "Negative samples only")
        ax.set_ylabel("Spearman ρ(score, log1p drug degree)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].legend(title="Neg. ratio", loc="best")

    fig.suptitle("Score–drug popularity correlation under degree_matched_head")
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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze score-degree correlation for LLM results on degree_matched_head."
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
        help="Negative sampling strategy to analyze. Default: degree_matched_head.",
    )
    parser.add_argument(
        "--ratios",
        nargs="*",
        type=int,
        default=[1, 4, 9],
        help="Negative ratios to include.",
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
        help="Optional order of models in plot.",
    )

    parser.add_argument(
        "--output_dir",
        default="data/result_llm/degree_bias_head",
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

    # Filter main hard-negative setting.
    df = df[df["negative_sampling_strategy"] == args.strategy].copy()
    df = df[df["negative_ratio"].isin(args.ratios)].copy()

    if args.graph_group != "ALL":
        if "graph_group" not in df.columns:
            raise ValueError("graph_group column not found in LLM results.")
        df = df[df["graph_group"] == args.graph_group].copy()

    if df.empty:
        raise ValueError("No LLM rows left after filtering.")

    # Merge by normalized drug name.
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
        missing_drugs = sorted(df.loc[df["drug_indication_degree"].isna(), "drug_name"].dropna().unique().tolist())
        print(f"[WARN] Missing training degree for {missing:,} rows, {len(missing_drugs):,} unique drugs.")
        print("[WARN] First 30 missing drugs:", missing_drugs[:30])

        # Missing means not mapped to PrimeKG nodes; drop rather than treating as zero.
        df = df.dropna(subset=["drug_indication_degree"]).copy()

    df["drug_indication_degree"] = df["drug_indication_degree"].astype(int)
    df["log1p_drug_indication_degree"] = df["log1p_drug_indication_degree"].astype(float)

    merged_csv = output_dir / "llm_results_with_train_drug_degree_degree_matched_head.csv"
    df.to_csv(merged_csv, index=False)
    print(f"[INFO] Saved merged analysis rows to: {merged_csv}")

    print("=" * 80)
    print("Filtered analysis data")
    print("=" * 80)
    print(f"Rows: {len(df):,}")
    print(f"Models: {sorted(df['model'].unique().tolist())}")
    print(f"Strategy: {args.strategy}")
    print(f"Ratios: {sorted(df['negative_ratio'].unique().tolist())}")
    if "graph_group" in df.columns:
        print(f"Graph groups: {sorted(df['graph_group'].dropna().unique().tolist())}")
    print("=" * 80)

    per_file = compute_per_file_correlations(df)
    summary = summarize_correlations(per_file)

    per_file_csv = output_dir / "score_degree_spearman_per_file.csv"
    summary_csv = output_dir / "score_degree_spearman_summary.csv"

    per_file.to_csv(per_file_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"[INFO] Saved per-file correlation table to: {per_file_csv}")
    print(f"[INFO] Saved summary correlation table to: {summary_csv}")

    export_graphpad_tables(
        per_file=per_file,
        summary=summary,
        output_dir=output_dir,
        model_order=args.model_order,
    )

    png = output_dir / "figure_A_score_degree_spearman_degree_matched_head.png"
    pdf = output_dir / "figure_A_score_degree_spearman_degree_matched_head.pdf"

    plot_correlation(
        summary=summary,
        output_png=png,
        output_pdf=pdf,
        model_order=args.model_order,
    )

    print("\nSummary table:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()