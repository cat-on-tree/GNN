import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu


# ============================================================
# Normalization helpers
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
# Load LLM benchmark rows from one JSONL
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


def load_test_rows_from_jsonl(source_jsonl):
    rows = read_jsonl(source_jsonl)
    if not rows:
        raise ValueError(f"No rows loaded from {source_jsonl}")

    df = pd.DataFrame(rows)

    required = [
        "test_file",
        "graph_group",
        "negative_sampling_strategy",
        "negative_ratio",
        "drug_name",
        "disease_name",
        "label",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in JSONL: {missing}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["negative_ratio"] = pd.to_numeric(df["negative_ratio"], errors="coerce")

    df = df.dropna(subset=[
        "test_file",
        "graph_group",
        "negative_sampling_strategy",
        "negative_ratio",
        "drug_name",
        "disease_name",
        "label",
    ]).copy()

    df["label"] = df["label"].astype(int)
    df["negative_ratio"] = df["negative_ratio"].astype(int)
    df["drug_name_norm"] = df["drug_name"].apply(normalize_name)
    df["disease_name_norm"] = df["disease_name"].apply(normalize_name)

    # Test rows are identical across models, but JSONL can contain duplicates
    # if rerun/merged. Keep one row per test_file + row_index when available.
    if "row_index" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["test_file", "row_index"]).copy()
        after = len(df)
        if after < before:
            print(f"[INFO] Dropped {before - after:,} duplicated test rows by test_file,row_index.")
    else:
        before = len(df)
        df = df.drop_duplicates(subset=[
            "test_file",
            "drug_name",
            "disease_name",
            "label",
            "negative_sampling_strategy",
            "negative_ratio",
        ]).copy()
        after = len(df)
        if after < before:
            print(f"[INFO] Dropped {before - after:,} duplicated test rows by pair fields.")

    return df


# ============================================================
# Load PrimeKG nodes and training graph
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


def load_and_map_train_edges(train_csv, nodes_csv):
    """
    Map TxGNN train split x_id/y_id to PrimeKG node_index using nodes.csv node_id.

    Important:
      train.csv x_idx/y_idx may be TxGNN internal indices, so we do not rely on them.
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

    # Keep rows whose mapped node types match endpoint types where mapping is available.
    train = train[
        (train["x_node_type"].isna() | (train["x_node_type"] == train["x_type"]))
        & (train["y_node_type"].isna() | (train["y_node_type"] == train["y_type"]))
    ].copy()

    mapped = train.dropna(subset=["x_node_index", "y_node_index"]).copy()
    mapped["x_node_index"] = mapped["x_node_index"].astype(int)
    mapped["y_node_index"] = mapped["y_node_index"].astype(int)

    print(f"[INFO] Loaded train edges: {len(train):,}")
    print(f"[INFO] Mapped train edges with node_index: {len(mapped):,}")

    return mapped, nodes


# ============================================================
# Compute graph covariates from training graph
# ============================================================

def build_adjacency(mapped_train):
    adj = defaultdict(set)

    for x, y in zip(mapped_train["x_node_index"], mapped_train["y_node_index"]):
        if pd.isna(x) or pd.isna(y):
            continue
        x = int(x)
        y = int(y)
        if x == y:
            continue
        adj[x].add(y)
        adj[y].add(x)

    return adj


def compute_total_degree_table(nodes, adj):
    rows = []

    for _, r in nodes.iterrows():
        node_index = int(r["node_index"])
        rows.append({
            "node_index": node_index,
            "node_name_norm": r["node_name_norm"],
            "node_name": r["node_name"],
            "node_type": r["node_type"],
            "total_degree": len(adj.get(node_index, set())),
        })

    return pd.DataFrame(rows)


def compute_indication_degree_tables(mapped_train, nodes):
    """
    Direction-agnostic indication degree:
      drug_indication_degree = number of unique disease neighbors by indication in train.
      disease_indication_degree = number of unique drug neighbors by indication in train.
    """
    xy = mapped_train[
        (mapped_train["relation"] == "indication")
        & (mapped_train["x_type"] == "drug")
        & (mapped_train["y_type"] == "disease")
    ][[
        "x_node_index", "x_name", "x_name_norm",
        "y_node_index", "y_name", "y_name_norm",
    ]].copy()

    xy = xy.rename(columns={
        "x_node_index": "drug_node_index",
        "x_name": "drug_name",
        "x_name_norm": "drug_name_norm",
        "y_node_index": "disease_node_index",
        "y_name": "disease_name",
        "y_name_norm": "disease_name_norm",
    })

    yx = mapped_train[
        (mapped_train["relation"] == "indication")
        & (mapped_train["x_type"] == "disease")
        & (mapped_train["y_type"] == "drug")
    ][[
        "y_node_index", "y_name", "y_name_norm",
        "x_node_index", "x_name", "x_name_norm",
    ]].copy()

    yx = yx.rename(columns={
        "y_node_index": "drug_node_index",
        "y_name": "drug_name",
        "y_name_norm": "drug_name_norm",
        "x_node_index": "disease_node_index",
        "x_name": "disease_name",
        "x_name_norm": "disease_name_norm",
    })

    ind = pd.concat([xy, yx], ignore_index=True)
    ind = ind.dropna(subset=["drug_node_index", "disease_node_index"]).copy()
    ind["drug_node_index"] = ind["drug_node_index"].astype(int)
    ind["disease_node_index"] = ind["disease_node_index"].astype(int)

    print(f"[INFO] Training indication drug-disease edges after node mapping: {len(ind):,}")

    drug_deg = (
        ind.groupby(["drug_node_index", "drug_name", "drug_name_norm"])["disease_node_index"]
        .nunique()
        .reset_index(name="drug_indication_degree")
    )

    disease_deg = (
        ind.groupby(["disease_node_index", "disease_name", "disease_name_norm"])["drug_node_index"]
        .nunique()
        .reset_index(name="disease_indication_degree")
    )

    # Include zero-degree drug/disease nodes.
    drug_nodes = nodes[nodes["node_type"] == "drug"][["node_index", "node_name", "node_name_norm"]].copy()
    drug_nodes = drug_nodes.rename(columns={
        "node_index": "drug_node_index",
        "node_name": "drug_name",
        "node_name_norm": "drug_name_norm",
    })

    disease_nodes = nodes[nodes["node_type"] == "disease"][["node_index", "node_name", "node_name_norm"]].copy()
    disease_nodes = disease_nodes.rename(columns={
        "node_index": "disease_node_index",
        "node_name": "disease_name",
        "node_name_norm": "disease_name_norm",
    })

    drug_deg = drug_nodes.merge(
        drug_deg[["drug_node_index", "drug_indication_degree"]],
        on="drug_node_index",
        how="left",
    )
    drug_deg["drug_indication_degree"] = drug_deg["drug_indication_degree"].fillna(0).astype(int)

    disease_deg = disease_nodes.merge(
        disease_deg[["disease_node_index", "disease_indication_degree"]],
        on="disease_node_index",
        how="left",
    )
    disease_deg["disease_indication_degree"] = disease_deg["disease_indication_degree"].fillna(0).astype(int)

    # Name-based tables for LLM rows.
    drug_by_name = (
        drug_deg.groupby("drug_name_norm", as_index=False)
        .agg(
            drug_name=("drug_name", "first"),
            drug_node_index=("drug_node_index", "first"),
            drug_indication_degree=("drug_indication_degree", "max"),
        )
    )

    disease_by_name = (
        disease_deg.groupby("disease_name_norm", as_index=False)
        .agg(
            disease_name=("disease_name", "first"),
            disease_node_index=("disease_node_index", "first"),
            disease_indication_degree=("disease_indication_degree", "max"),
        )
    )

    return drug_by_name, disease_by_name


def make_node_covariate_tables(mapped_train, nodes):
    adj = build_adjacency(mapped_train)
    total_degree = compute_total_degree_table(nodes, adj)
    drug_ind, disease_ind = compute_indication_degree_tables(mapped_train, nodes)

    # Total degree by normalized name and type.
    drug_total = (
        total_degree[total_degree["node_type"] == "drug"]
        .groupby("node_name_norm", as_index=False)
        .agg(
            drug_total_degree=("total_degree", "max"),
            drug_total_node_index=("node_index", "first"),
        )
        .rename(columns={"node_name_norm": "drug_name_norm"})
    )

    disease_total = (
        total_degree[total_degree["node_type"] == "disease"]
        .groupby("node_name_norm", as_index=False)
        .agg(
            disease_total_degree=("total_degree", "max"),
            disease_total_node_index=("node_index", "first"),
        )
        .rename(columns={"node_name_norm": "disease_name_norm"})
    )

    drug_cov = drug_ind.merge(drug_total, on="drug_name_norm", how="left")
    disease_cov = disease_ind.merge(disease_total, on="disease_name_norm", how="left")

    drug_cov["drug_total_degree"] = drug_cov["drug_total_degree"].fillna(0).astype(int)
    disease_cov["disease_total_degree"] = disease_cov["disease_total_degree"].fillna(0).astype(int)

    return drug_cov, disease_cov, adj


# ============================================================
# Merge covariates into test set
# ============================================================

def add_covariates_to_test(df, drug_cov, disease_cov, adj):
    df = df.copy()

    df = df.merge(
        drug_cov[[
            "drug_name_norm",
            "drug_node_index",
            "drug_indication_degree",
            "drug_total_degree",
        ]],
        on="drug_name_norm",
        how="left",
    )

    df = df.merge(
        disease_cov[[
            "disease_name_norm",
            "disease_node_index",
            "disease_indication_degree",
            "disease_total_degree",
        ]],
        on="disease_name_norm",
        how="left",
    )

    missing_drug = df["drug_indication_degree"].isna().sum()
    missing_disease = df["disease_indication_degree"].isna().sum()

    if missing_drug > 0:
        missing = sorted(df.loc[df["drug_indication_degree"].isna(), "drug_name"].dropna().unique().tolist())
        print(f"[WARN] Missing drug covariates for {missing_drug:,} rows, {len(missing):,} unique drugs.")
        print("[WARN] First 20 missing drugs:", missing[:20])

    if missing_disease > 0:
        missing = sorted(df.loc[df["disease_indication_degree"].isna(), "disease_name"].dropna().unique().tolist())
        print(f"[WARN] Missing disease covariates for {missing_disease:,} rows, {len(missing):,} unique diseases.")
        print("[WARN] First 20 missing diseases:", missing[:20])

    # Drop rows that cannot be mapped.
    before = len(df)
    df = df.dropna(subset=[
        "drug_node_index",
        "disease_node_index",
        "drug_indication_degree",
        "disease_indication_degree",
        "drug_total_degree",
        "disease_total_degree",
    ]).copy()
    after = len(df)

    if after < before:
        print(f"[WARN] Dropped {before - after:,} test rows with missing covariates.")

    df["drug_node_index"] = df["drug_node_index"].astype(int)
    df["disease_node_index"] = df["disease_node_index"].astype(int)

    for c in [
        "drug_indication_degree",
        "disease_indication_degree",
        "drug_total_degree",
        "disease_total_degree",
    ]:
        df[c] = df[c].astype(float)

    # Shared neighbors in the training graph.
    shared_counts = []
    for d, dis in zip(df["drug_node_index"], df["disease_node_index"]):
        neigh_d = adj.get(int(d), set())
        neigh_dis = adj.get(int(dis), set())
        shared_counts.append(len(neigh_d & neigh_dis))

    df["shared_neighbor_count"] = shared_counts

    # Composite covariates.
    df["endpoint_indication_degree_sum"] = (
        df["drug_indication_degree"] + df["disease_indication_degree"]
    )
    df["endpoint_total_degree_sum"] = (
        df["drug_total_degree"] + df["disease_total_degree"]
    )

    # Log-transformed covariates for balance analysis.
    for c in [
        "drug_indication_degree",
        "disease_indication_degree",
        "drug_total_degree",
        "disease_total_degree",
        "endpoint_indication_degree_sum",
        "endpoint_total_degree_sum",
        "shared_neighbor_count",
    ]:
        df[f"log1p_{c}"] = np.log1p(df[c].astype(float))

    return df


# ============================================================
# Balance statistics
# ============================================================

def standardized_mean_difference(pos, neg):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    mean_pos = np.mean(pos)
    mean_neg = np.mean(neg)

    var_pos = np.var(pos, ddof=1) if len(pos) > 1 else 0.0
    var_neg = np.var(neg, ddof=1) if len(neg) > 1 else 0.0

    pooled_sd = np.sqrt((var_pos + var_neg) / 2.0)

    if pooled_sd == 0:
        if mean_pos == mean_neg:
            return 0.0
        return np.nan

    return float((mean_pos - mean_neg) / pooled_sd)


def safe_ks(pos, neg):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan

    try:
        res = ks_2samp(pos, neg, alternative="two-sided", mode="auto")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def safe_mannwhitney(pos, neg):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    try:
        res = mannwhitneyu(pos, neg, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return np.nan


def compute_balance_per_file(df, covariates):
    rows = []

    group_cols = [
        "negative_sampling_strategy",
        "negative_ratio",
        "test_file",
    ]

    for keys, g in df.groupby(group_cols, dropna=False):
        strategy, ratio, test_file = keys

        pos = g[g["label"] == 1]
        neg = g[g["label"] == 0]

        for cov in covariates:
            pos_vals = pos[cov].astype(float).values
            neg_vals = neg[cov].astype(float).values

            smd = standardized_mean_difference(pos_vals, neg_vals)
            ks_stat, ks_p = safe_ks(pos_vals, neg_vals)
            mw_p = safe_mannwhitney(pos_vals, neg_vals)

            rows.append({
                "negative_sampling_strategy": strategy,
                "negative_ratio": ratio,
                "test_file": test_file,
                "covariate": cov,
                "n_pos": len(pos_vals),
                "n_neg": len(neg_vals),
                "pos_mean": float(np.nanmean(pos_vals)) if len(pos_vals) else np.nan,
                "neg_mean": float(np.nanmean(neg_vals)) if len(neg_vals) else np.nan,
                "pos_median": float(np.nanmedian(pos_vals)) if len(pos_vals) else np.nan,
                "neg_median": float(np.nanmedian(neg_vals)) if len(neg_vals) else np.nan,
                "pos_std": float(np.nanstd(pos_vals, ddof=1)) if len(pos_vals) > 1 else 0.0,
                "neg_std": float(np.nanstd(neg_vals, ddof=1)) if len(neg_vals) > 1 else 0.0,
                "smd": smd,
                "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_p,
                "mannwhitney_p_value": mw_p,
            })

    return pd.DataFrame(rows)


def summarize_balance(per_file):
    rows = []

    group_cols = [
        "negative_sampling_strategy",
        "negative_ratio",
        "covariate",
    ]

    for keys, g in per_file.groupby(group_cols, dropna=False):
        strategy, ratio, covariate = keys

        row = {
            "negative_sampling_strategy": strategy,
            "negative_ratio": ratio,
            "covariate": covariate,
            "n_files": int(g["test_file"].nunique()),
            "n_pos_mean": float(g["n_pos"].mean()),
            "n_neg_mean": float(g["n_neg"].mean()),
            "pos_mean_mean": float(g["pos_mean"].mean()),
            "pos_mean_std": float(g["pos_mean"].std(ddof=1)) if len(g) > 1 else 0.0,
            "neg_mean_mean": float(g["neg_mean"].mean()),
            "neg_mean_std": float(g["neg_mean"].std(ddof=1)) if len(g) > 1 else 0.0,
            "smd_mean": float(g["smd"].mean()),
            "smd_std": float(g["smd"].std(ddof=1)) if len(g) > 1 else 0.0,
            "abs_smd_mean": float(g["abs_smd"].mean()),
            "abs_smd_max": float(g["abs_smd"].max()),
            "ks_statistic_mean": float(g["ks_statistic"].mean()),
            "ks_statistic_std": float(g["ks_statistic"].std(ddof=1)) if len(g) > 1 else 0.0,
            "mannwhitney_p_median": float(g["mannwhitney_p_value"].median()),
            "ks_p_median": float(g["ks_p_value"].median()),
        }

        # Common heuristic for SMD.
        if row["abs_smd_mean"] < 0.1:
            row["balance_interpretation"] = "well_balanced_abs_smd_lt_0.1"
        elif row["abs_smd_mean"] < 0.2:
            row["balance_interpretation"] = "small_imbalance_abs_smd_0.1_to_0.2"
        else:
            row["balance_interpretation"] = "notable_imbalance_abs_smd_ge_0.2"

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values([
        "negative_sampling_strategy",
        "negative_ratio",
        "abs_smd_mean",
        "covariate",
    ]).reset_index(drop=True)

    return summary


def make_supplementary_table(summary):
    """
    Publication-friendly table with rounded values.
    """
    table = summary.copy()

    covariate_label = {
        "log1p_drug_indication_degree": "log1p(drug indication degree)",
        "log1p_disease_indication_degree": "log1p(disease indication degree)",
        "log1p_drug_total_degree": "log1p(drug total degree)",
        "log1p_disease_total_degree": "log1p(disease total degree)",
        "log1p_endpoint_indication_degree_sum": "log1p(endpoint indication degree sum)",
        "log1p_endpoint_total_degree_sum": "log1p(endpoint total degree sum)",
        "log1p_shared_neighbor_count": "log1p(shared neighbor count)",
    }

    table["covariate_label"] = table["covariate"].map(covariate_label).fillna(table["covariate"])

    keep = [
        "negative_sampling_strategy",
        "negative_ratio",
        "covariate_label",
        "n_files",
        "n_pos_mean",
        "n_neg_mean",
        "pos_mean_mean",
        "neg_mean_mean",
        "smd_mean",
        "smd_std",
        "abs_smd_mean",
        "abs_smd_max",
        "ks_statistic_mean",
        "mannwhitney_p_median",
        "balance_interpretation",
    ]

    table = table[keep].copy()

    round_cols = [
        "n_pos_mean",
        "n_neg_mean",
        "pos_mean_mean",
        "neg_mean_mean",
        "smd_mean",
        "smd_std",
        "abs_smd_mean",
        "abs_smd_max",
        "ks_statistic_mean",
        "mannwhitney_p_median",
    ]

    for c in round_cols:
        table[c] = table[c].astype(float).round(4)

    return table


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Covariate balance analysis for degree-matched positive vs negative test pairs."
    )

    parser.add_argument(
        "--source_jsonl",
        default="data/result_llm/qwen3-8b.jsonl",
        help="One LLM JSONL result file used to recover test rows. Test rows should be identical across models.",
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
        help="Graph group to analyze. Use ALL to disable filter.",
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
        "--output_dir",
        default="data/result_llm/degree_bias_head/covariate_balance",
        help="Output directory.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading test rows")
    print("=" * 80)

    test_df = load_test_rows_from_jsonl(args.source_jsonl)

    test_df = test_df[test_df["negative_sampling_strategy"] == args.strategy].copy()
    test_df = test_df[test_df["negative_ratio"].isin(args.ratios)].copy()

    if args.graph_group != "ALL":
        test_df = test_df[test_df["graph_group"] == args.graph_group].copy()

    if test_df.empty:
        raise ValueError("No test rows left after filtering.")

    print(f"[INFO] Filtered test rows: {len(test_df):,}")
    print(f"[INFO] Strategy: {args.strategy}")
    print(f"[INFO] Ratios: {sorted(test_df['negative_ratio'].unique().tolist())}")
    print(f"[INFO] Test files: {test_df['test_file'].nunique()}")

    print("=" * 80)
    print("Loading and mapping training graph")
    print("=" * 80)

    mapped_train, nodes = load_and_map_train_edges(args.train_csv, args.nodes_csv)
    drug_cov, disease_cov, adj = make_node_covariate_tables(mapped_train, nodes)

    drug_cov_path = output_dir / "training_drug_covariates.csv"
    disease_cov_path = output_dir / "training_disease_covariates.csv"
    drug_cov.to_csv(drug_cov_path, index=False)
    disease_cov.to_csv(disease_cov_path, index=False)

    print(f"[INFO] Saved drug covariates to: {drug_cov_path}")
    print(f"[INFO] Saved disease covariates to: {disease_cov_path}")

    print("=" * 80)
    print("Merging covariates into test rows")
    print("=" * 80)

    analysis_df = add_covariates_to_test(test_df, drug_cov, disease_cov, adj)

    analysis_rows_path = output_dir / "degree_matched_test_rows_with_covariates.csv"
    analysis_df.to_csv(analysis_rows_path, index=False)
    print(f"[INFO] Saved analysis rows to: {analysis_rows_path}")

    covariates = [
        "log1p_drug_indication_degree",
        "log1p_disease_indication_degree",
        "log1p_drug_total_degree",
        "log1p_disease_total_degree",
        "log1p_endpoint_indication_degree_sum",
        "log1p_endpoint_total_degree_sum",
        "log1p_shared_neighbor_count",
    ]

    print("=" * 80)
    print("Computing positive/negative covariate balance")
    print("=" * 80)

    per_file = compute_balance_per_file(analysis_df, covariates)
    summary = summarize_balance(per_file)
    supp_table = make_supplementary_table(summary)

    per_file_path = output_dir / "covariate_balance_per_file.csv"
    summary_path = output_dir / "covariate_balance_summary.csv"
    supp_path = output_dir / "supplementary_table_A_covariate_balance.csv"
    xlsx_path = output_dir / "supplementary_table_A_covariate_balance.xlsx"

    per_file.to_csv(per_file_path, index=False)
    summary.to_csv(summary_path, index=False)
    supp_table.to_csv(supp_path, index=False)

    print(f"[INFO] Saved per-file balance table to: {per_file_path}")
    print(f"[INFO] Saved summary balance table to: {summary_path}")
    print(f"[INFO] Saved Supplementary Table A CSV to: {supp_path}")

    # Optional Excel output.
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            supp_table.to_excel(writer, index=False, sheet_name="Supplementary_Table_A")
            summary.to_excel(writer, index=False, sheet_name="Summary_Full")
            per_file.to_excel(writer, index=False, sheet_name="Per_File")
        print(f"[INFO] Saved Supplementary Table A XLSX to: {xlsx_path}")
    except Exception as e:
        print(f"[WARN] Could not save XLSX file: {e}")

    print("\nSupplementary Table A preview:")
    print(supp_table.to_string(index=False))


if __name__ == "__main__":
    main()