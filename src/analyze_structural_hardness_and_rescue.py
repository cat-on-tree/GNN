from scipy import sparse
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# Basic helpers
# ============================================================

def normalize_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_metric(func, y_true, y_score):
    try:
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(func(y_true, y_score))
    except Exception:
        return np.nan


def mean_sd(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().values
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def pairwise_win_rate(pos_scores, neg_scores):
    """
    P(score_positive > score_negative) + 0.5 * P(tie).

    Equivalent to AUROC for one positive group against one negative bin,
    but computed directly and robustly.
    """
    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_scores = np.asarray(neg_scores, dtype=float)

    pos_scores = pos_scores[np.isfinite(pos_scores)]
    neg_scores = neg_scores[np.isfinite(neg_scores)]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.nan

    neg_sorted = np.sort(neg_scores)

    wins = np.searchsorted(neg_sorted, pos_scores, side="left")
    ties_right = np.searchsorted(neg_sorted, pos_scores, side="right")
    ties = ties_right - wins

    return float(np.mean((wins + 0.5 * ties) / len(neg_sorted)))


# ============================================================
# Load nodes / graph
# ============================================================

def load_nodes(nodes_csv):
    nodes = pd.read_csv(nodes_csv, low_memory=False)

    required = ["node_index", "node_id", "node_type", "node_name"]
    missing = [c for c in required if c not in nodes.columns]
    if missing:
        raise ValueError(f"{nodes_csv} missing columns: {missing}")

    nodes = nodes[["node_index", "node_id", "node_type", "node_name"]].copy()
    nodes["node_index"] = pd.to_numeric(nodes["node_index"], errors="coerce")
    nodes = nodes.dropna(subset=["node_index"])
    nodes["node_index"] = nodes["node_index"].astype(int)

    nodes["node_id_norm"] = nodes["node_id"].apply(normalize_id)
    nodes["node_type"] = nodes["node_type"].astype(str)
    nodes["node_name"] = nodes["node_name"].astype(str)
    nodes["node_name_norm"] = nodes["node_name"].str.lower().str.strip()

    return nodes


def load_train_edges_with_node_index(train_csv, nodes_csv):
    train = pd.read_csv(train_csv, low_memory=False)
    nodes = load_nodes(nodes_csv)

    required = ["x_type", "x_id", "relation", "y_type", "y_id"]
    missing = [c for c in required if c not in train.columns]
    if missing:
        raise ValueError(f"{train_csv} missing columns: {missing}")

    train = train.copy()
    train["x_type"] = train["x_type"].astype(str)
    train["y_type"] = train["y_type"].astype(str)
    train["relation"] = train["relation"].astype(str)
    train["x_id_norm"] = train["x_id"].apply(normalize_id)
    train["y_id_norm"] = train["y_id"].apply(normalize_id)

    node_x = nodes.rename(columns={
        "node_id_norm": "x_id_norm",
        "node_index": "x_node_index",
        "node_type": "x_node_type",
        "node_name": "x_node_name",
    })

    node_y = nodes.rename(columns={
        "node_id_norm": "y_id_norm",
        "node_index": "y_node_index",
        "node_type": "y_node_type",
        "node_name": "y_node_name",
    })

    train = train.merge(
        node_x[["x_id_norm", "x_node_index", "x_node_type", "x_node_name"]],
        on="x_id_norm",
        how="left",
    )

    train = train.merge(
        node_y[["y_id_norm", "y_node_index", "y_node_type", "y_node_name"]],
        on="y_id_norm",
        how="left",
    )

    train = train[
        (train["x_node_type"].isna() | (train["x_node_type"] == train["x_type"]))
        & (train["y_node_type"].isna() | (train["y_node_type"] == train["y_type"]))
    ].copy()

    mapped = train.dropna(subset=["x_node_index", "y_node_index"]).copy()
    mapped["x_node_index"] = mapped["x_node_index"].astype(int)
    mapped["y_node_index"] = mapped["y_node_index"].astype(int)

    print(f"[INFO] Loaded train edges: {len(train):,}")
    print(f"[INFO] Mapped train edges: {len(mapped):,}")

    return mapped, nodes


def build_undirected_adjacency(mapped_train):
    adj = defaultdict(set)

    for x, y in zip(mapped_train["x_node_index"], mapped_train["y_node_index"]):
        x = int(x)
        y = int(y)
        if x == y:
            continue
        adj[x].add(y)
        adj[y].add(x)

    return adj


def build_name_to_node_index(nodes):
    """
    Fallback mapper by node_name + node_type.
    Only keeps unique names per type to avoid ambiguous mapping.
    """
    out = {}

    for node_type in ["drug", "disease"]:
        sub = nodes[nodes["node_type"] == node_type].copy()
        counts = sub["node_name_norm"].value_counts()

        unique_names = set(counts[counts == 1].index)
        sub = sub[sub["node_name_norm"].isin(unique_names)]

        for _, r in sub.iterrows():
            out[(node_type, r["node_name_norm"])] = int(r["node_index"])

    return out


# ============================================================
# Structural metrics
# ============================================================

def shortest_path_with_cutoff(adj, src, dst, cutoff=4):
    """
    BFS shortest path up to cutoff.
    Returns cutoff + 1 if no path found within cutoff.
    """
    if src is None or dst is None:
        return np.nan

    src = int(src)
    dst = int(dst)

    if src == dst:
        return 0

    if src not in adj or dst not in adj:
        return cutoff + 1

    q = deque([(src, 0)])
    seen = {src}

    while q:
        node, dist = q.popleft()

        if dist >= cutoff:
            continue

        for nb in adj.get(node, []):
            if nb == dst:
                return dist + 1
            if nb not in seen:
                seen.add(nb)
                q.append((nb, dist + 1))

    return cutoff + 1


def compute_pair_structural_metrics(
    pair_df,
    adj,
    name_to_node_index,
    shortest_cutoff=4,
    compute_shortest_path=False,
    batch_size=5000,
):
    """
    Sparse accelerated structural metric computation.

    Computes, in batches:
      - drug_total_degree
      - disease_total_degree
      - endpoint_total_degree_sum
      - shared_neighbor_count
      - jaccard_shared_neighbors
      - adamic_adar_score

    By default, shortest path is NOT computed because cutoff BFS on a large
    biomedical KG is very slow. If compute_shortest_path=True, it will fall
    back to the previous per-pair BFS behavior and may be slow.

    pair_df requires:
      test_file, row_index, drug_node_index, disease_node_index, drug_name, disease_name
    """

    print("[INFO] Resolving test-pair node indices")

    # --------------------------------------------------------
    # Resolve node indices first.
    # Prefer drug_node_index / disease_node_index.
    # Fallback to unique node_name mapping.
    # --------------------------------------------------------
    test_files = []
    row_indices = []
    drug_indices = []
    disease_indices = []

    for _, r in pair_df.iterrows():
        test_file = r["test_file"]
        row_index = int(r["row_index"])

        drug_idx = safe_int(r.get("drug_node_index"))
        disease_idx = safe_int(r.get("disease_node_index"))

        if drug_idx is None:
            dn = str(r.get("drug_name", "")).lower().strip()
            drug_idx = name_to_node_index.get(("drug", dn))

        if disease_idx is None:
            dis = str(r.get("disease_name", "")).lower().strip()
            disease_idx = name_to_node_index.get(("disease", dis))

        test_files.append(test_file)
        row_indices.append(row_index)
        drug_indices.append(drug_idx)
        disease_indices.append(disease_idx)

    n_pairs = len(pair_df)

    drug_arr = np.array(
        [-1 if x is None else int(x) for x in drug_indices],
        dtype=np.int64,
    )
    disease_arr = np.array(
        [-1 if x is None else int(x) for x in disease_indices],
        dtype=np.int64,
    )

    valid_pair_mask = (drug_arr >= 0) & (disease_arr >= 0)

    print(f"[INFO] Test pairs: {n_pairs:,}")
    print(f"[INFO] Pairs with resolved node indices: {int(valid_pair_mask.sum()):,}")

    # --------------------------------------------------------
    # Build sparse adjacency matrix from existing adj dict.
    # adj is already undirected from build_undirected_adjacency().
    # --------------------------------------------------------
    print("[INFO] Building sparse adjacency matrix")

    max_adj_node = 0
    nnz = 0
    for u, nbs in adj.items():
        if u > max_adj_node:
            max_adj_node = int(u)
        if nbs:
            m = max(nbs)
            if m > max_adj_node:
                max_adj_node = int(m)
        nnz += len(nbs)

    max_pair_node = -1
    if valid_pair_mask.any():
        max_pair_node = int(max(drug_arr[valid_pair_mask].max(), disease_arr[valid_pair_mask].max()))

    n_nodes = max(max_adj_node, max_pair_node) + 1

    print(f"[INFO] Sparse adjacency shape: {n_nodes:,} x {n_nodes:,}")
    print(f"[INFO] Sparse adjacency nnz: {nnz:,}")

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)

    cursor = 0
    for u, nbs in adj.items():
        n = len(nbs)
        if n == 0:
            continue
        rows[cursor:cursor + n] = int(u)
        cols[cursor:cursor + n] = np.fromiter(nbs, dtype=np.int32, count=n)
        cursor += n

    data = np.ones(nnz, dtype=np.float32)

    A = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    )

    A.eliminate_zeros()

    degree = np.diff(A.indptr).astype(np.float32)

    # --------------------------------------------------------
    # Adamic-Adar uses neighbor weight 1 / log(degree(z)).
    # For degree <= 1, define weight as 0 to avoid division issues.
    # --------------------------------------------------------
    aa_weight = np.zeros(n_nodes, dtype=np.float32)
    valid_degree = degree > 1
    aa_weight[valid_degree] = 1.0 / np.log(degree[valid_degree])

    # Column-weighted adjacency.
    # Aw[u, z] = A[u, z] * aa_weight[z]
    Aw = A.multiply(aa_weight.reshape(1, -1)).tocsr()

    # --------------------------------------------------------
    # Prepare output arrays.
    # --------------------------------------------------------
    drug_total_degree = np.full(n_pairs, np.nan, dtype=np.float64)
    disease_total_degree = np.full(n_pairs, np.nan, dtype=np.float64)
    endpoint_total_degree_sum = np.full(n_pairs, np.nan, dtype=np.float64)

    shared_neighbor_count = np.full(n_pairs, np.nan, dtype=np.float64)
    jaccard_shared_neighbors = np.full(n_pairs, np.nan, dtype=np.float64)
    adamic_adar_score = np.full(n_pairs, np.nan, dtype=np.float64)

    shortest_path_length_cutoff = np.full(n_pairs, np.nan, dtype=np.float64)
    inv_shortest_path_hardness = np.full(n_pairs, np.nan, dtype=np.float64)

    valid_positions = np.where(valid_pair_mask)[0]

    print("[INFO] Computing sparse structural metrics in batches")

    for start in range(0, len(valid_positions), batch_size):
        end = min(start + batch_size, len(valid_positions))
        pos = valid_positions[start:end]

        d = drug_arr[pos]
        s = disease_arr[pos]

        deg_d = degree[d].astype(np.float64)
        deg_s = degree[s].astype(np.float64)

        drug_total_degree[pos] = deg_d
        disease_total_degree[pos] = deg_s
        endpoint_total_degree_sum[pos] = deg_d + deg_s

        # shared neighbors = dot(A[drug], A[disease])
        # Efficient row-wise sparse intersection.
        shared = np.asarray(
            A[d].multiply(A[s]).sum(axis=1)
        ).ravel().astype(np.float64)

        shared_neighbor_count[pos] = shared

        union = deg_d + deg_s - shared
        jacc = np.divide(
            shared,
            union,
            out=np.zeros_like(shared, dtype=np.float64),
            where=union > 0,
        )
        jaccard_shared_neighbors[pos] = jacc

        # Adamic-Adar = dot(A_weighted[drug], A[disease])
        aa = np.asarray(
            Aw[d].multiply(A[s]).sum(axis=1)
        ).ravel().astype(np.float64)

        adamic_adar_score[pos] = aa

        if (start // batch_size) % 10 == 0:
            print(f"[INFO] Processed {end:,}/{len(valid_positions):,} resolved pairs")

    # --------------------------------------------------------
    # Optional shortest path.
    # Disabled by default because it is the slow step.
    # --------------------------------------------------------
    if compute_shortest_path:
        print("[WARN] compute_shortest_path=True: running slow per-pair cutoff BFS")

        for i in valid_positions:
            sp = shortest_path_with_cutoff(
                adj,
                int(drug_arr[i]),
                int(disease_arr[i]),
                cutoff=shortest_cutoff,
            )
            shortest_path_length_cutoff[i] = sp
            inv_shortest_path_hardness[i] = -float(sp) if pd.notna(sp) else np.nan

            if i % 5000 == 0:
                print(f"[INFO] Shortest path processed pair index {i:,}/{n_pairs:,}")

    out = pd.DataFrame({
        "test_file": test_files,
        "row_index": row_indices,
        "drug_node_index": np.where(drug_arr >= 0, drug_arr, np.nan),
        "disease_node_index": np.where(disease_arr >= 0, disease_arr, np.nan),

        "drug_total_degree": drug_total_degree,
        "disease_total_degree": disease_total_degree,
        "endpoint_total_degree_sum": endpoint_total_degree_sum,

        "shared_neighbor_count": shared_neighbor_count,
        "jaccard_shared_neighbors": jaccard_shared_neighbors,
        "adamic_adar_score": adamic_adar_score,

        "shortest_path_length_cutoff": shortest_path_length_cutoff,
        "inv_shortest_path_hardness": inv_shortest_path_hardness,
    })

    print("[INFO] Structural metric summary:")
    for c in [
        "drug_total_degree",
        "disease_total_degree",
        "endpoint_total_degree_sum",
        "shared_neighbor_count",
        "jaccard_shared_neighbors",
        "adamic_adar_score",
    ]:
        print(f"\n[INFO] {c}")
        print(out[c].describe().to_string())

    return out


# ============================================================
# Load predictions
# ============================================================

def load_llm_jsonl_files(llm_jsonl_paths, graph_group, strategy, ratios, source_format):
    rows = []
    ratios = set(int(x) for x in ratios)

    for p in llm_jsonl_paths:
        p = Path(p)
        if not p.exists():
            print(f"[WARN] LLM JSONL not found, skipping: {p}")
            continue

        print(f"[INFO] Reading LLM JSONL: {p}")

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)

                if r.get("graph_group") != graph_group:
                    continue
                if r.get("negative_sampling_strategy") != strategy:
                    continue
                if int(r.get("negative_ratio")) not in ratios:
                    continue
                if source_format is not None and r.get("source_format") != source_format:
                    continue

                rows.append({
                    "model": r.get("model"),
                    "model_family": "llm",
                    "test_file": r.get("test_file"),
                    "graph_group": r.get("graph_group"),
                    "negative_sampling_strategy": r.get("negative_sampling_strategy"),
                    "negative_ratio": int(r.get("negative_ratio")),
                    "negative_sampling_seed": safe_int(r.get("negative_sampling_seed")),
                    "row_index": int(r.get("row_index")),
                    "drug_name": r.get("drug_name"),
                    "disease_name": r.get("disease_name"),
                    "label": int(r.get("label")),
                    "score": safe_float(r.get("pred_prob")),
                    "x_idx": safe_int(r.get("x_idx")),
                    "y_idx": safe_int(r.get("y_idx")),
                    "original_x_index": safe_int(r.get("original_x_index")),
                    "original_y_index": safe_int(r.get("original_y_index")),
                    "source_file": str(p),
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No LLM rows loaded after filtering.")

    print(f"[INFO] Loaded LLM rows after filtering: {len(df):,}")
    print(f"[INFO] LLM models: {sorted(df['model'].dropna().unique().tolist())}")
    print(f"[INFO] LLM files: {df['test_file'].nunique():,}")

    return df


def load_gnn_predictions(gnn_csv, graph_group, strategy, ratios, gnn_models=None, gnn_score_col="pred_prob"):
    df = pd.read_csv(gnn_csv, low_memory=False)
    ratios = set(int(x) for x in ratios)

    required = [
        "model_name",
        "test_file",
        "graph_group",
        "negative_sampling_strategy",
        "negative_ratio",
        "negative_sampling_seed",
        "row_index",
        "label",
        gnn_score_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{gnn_csv} missing columns: {missing}")

    df = df[
        (df["graph_group"] == graph_group)
        & (df["negative_sampling_strategy"] == strategy)
        & (df["negative_ratio"].astype(int).isin(ratios))
    ].copy()

    if gnn_models is not None and len(gnn_models) > 0 and gnn_models != ["all"]:
        df = df[df["model_name"].isin(gnn_models)].copy()

    if df.empty:
        raise ValueError("No GNN rows loaded after filtering.")

    out = pd.DataFrame({
        "model": df["model_name"].astype(str),
        "model_family": "gnn",
        "test_file": df["test_file"].astype(str),
        "graph_group": df["graph_group"].astype(str),
        "negative_sampling_strategy": df["negative_sampling_strategy"].astype(str),
        "negative_ratio": df["negative_ratio"].astype(int),
        "negative_sampling_seed": pd.to_numeric(df["negative_sampling_seed"], errors="coerce").astype("Int64"),
        "row_index": df["row_index"].astype(int),
        "drug_name": df.get("x_name", pd.Series([np.nan] * len(df))).astype(str),
        "disease_name": df.get("y_name", pd.Series([np.nan] * len(df))).astype(str),
        "label": pd.to_numeric(df["label"], errors="raise").astype(int),
        "score": pd.to_numeric(df[gnn_score_col], errors="coerce"),
        "x_idx": pd.to_numeric(df.get("x_idx", np.nan), errors="coerce"),
        "y_idx": pd.to_numeric(df.get("y_idx", np.nan), errors="coerce"),
        "original_x_index": pd.to_numeric(df.get("original_x_index", np.nan), errors="coerce"),
        "original_y_index": pd.to_numeric(df.get("original_y_index", np.nan), errors="coerce"),
        "x_id": df.get("x_id", pd.Series([np.nan] * len(df))).astype(str),
        "y_id": df.get("y_id", pd.Series([np.nan] * len(df))).astype(str),
    })

    print(f"[INFO] Loaded GNN rows after filtering: {len(out):,}")
    print(f"[INFO] GNN models: {sorted(out['model'].dropna().unique().tolist())}")
    print(f"[INFO] GNN files: {out['test_file'].nunique():,}")

    return out


def build_pair_metadata_from_gnn(gnn_df):
    """
    Use GNN rows as canonical test-set metadata because they contain x_id/y_id/name/index.
    One row per test_file,row_index.
    """
    cols = [
        "test_file",
        "row_index",
        "negative_ratio",
        "negative_sampling_seed",
        "label",
        "drug_name",
        "disease_name",
        "x_idx",
        "y_idx",
        "original_x_index",
        "original_y_index",
    ]
    extra = [c for c in ["x_id", "y_id"] if c in gnn_df.columns]

    meta = (
        gnn_df[cols + extra]
        .drop_duplicates(subset=["test_file", "row_index"])
        .copy()
    )

    # For structural graph based on PrimeKG/node.csv, original_x_index/y_index are
    # usually the global node_index. Prefer them, fallback to x_idx/y_idx.
    meta["drug_node_index"] = pd.to_numeric(meta["original_x_index"], errors="coerce")
    meta["disease_node_index"] = pd.to_numeric(meta["original_y_index"], errors="coerce")

    meta.loc[meta["drug_node_index"].isna(), "drug_node_index"] = pd.to_numeric(
        meta.loc[meta["drug_node_index"].isna(), "x_idx"],
        errors="coerce",
    )
    meta.loc[meta["disease_node_index"].isna(), "disease_node_index"] = pd.to_numeric(
        meta.loc[meta["disease_node_index"].isna(), "y_idx"],
        errors="coerce",
    )

    return meta


# ============================================================
# Hardness binning
# ============================================================

def assign_negative_hardness_bins(df, structural_metric, group_cols):
    """
    Assign low/mid/high bins among negatives within each group.
    Higher metric = harder.

    For shortest path, use inv_shortest_path_hardness where higher means shorter path.
    """
    out_parts = []

    for _, g in df.groupby(group_cols, dropna=False):
        g = g.copy()
        g[f"{structural_metric}_hardness_bin"] = np.nan

        neg_mask = g["label"].astype(int) == 0
        neg = g.loc[neg_mask].copy()

        vals = pd.to_numeric(neg[structural_metric], errors="coerce")

        valid = vals.notna()
        if valid.sum() < 3:
            out_parts.append(g)
            continue

        # qcut on rank avoids errors when many ties.
        ranks = vals[valid].rank(method="first")

        try:
            bins = pd.qcut(
                ranks,
                q=3,
                labels=["low", "mid", "high"],
                duplicates="drop",
            )
        except Exception:
            out_parts.append(g)
            continue

        g.loc[neg.index[valid], f"{structural_metric}_hardness_bin"] = bins.astype(str).values
        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


# ============================================================
# Structural stratification analysis
# ============================================================

def compute_structural_stratification(
    pred_df,
    structural_metrics,
):
    """
    For each model / file / structural metric / hardness bin:
      positives are all positives in the file;
      negatives are negatives in the bin;
      compute pairwise win rate, score gap, AUROC, AUPRC.
    """
    per_file_rows = []

    base_group_cols = [
        "negative_ratio",
        "test_file",
    ]

    for metric in structural_metrics:
        bin_col = f"{metric}_hardness_bin"

        df_bin = assign_negative_hardness_bins(
            pred_df,
            structural_metric=metric,
            group_cols=base_group_cols,
        )

        for (model, ratio, test_file), g in df_bin.groupby(
            ["model", "negative_ratio", "test_file"],
            dropna=False,
        ):
            pos = g[g["label"].astype(int) == 1].copy()
            if pos.empty:
                continue

            pos_scores = pos["score"].astype(float).values

            for bin_name in ["low", "mid", "high"]:
                neg = g[
                    (g["label"].astype(int) == 0)
                    & (g[bin_col].astype(str) == bin_name)
                ].copy()

                if neg.empty:
                    continue

                neg_scores = neg["score"].astype(float).values

                y_true = np.concatenate([
                    np.ones(len(pos_scores), dtype=int),
                    np.zeros(len(neg_scores), dtype=int),
                ])
                y_score = np.concatenate([pos_scores, neg_scores])

                row = {
                    "model": model,
                    "negative_ratio": int(ratio),
                    "test_file": test_file,
                    "structural_metric": metric,
                    "hardness_bin": bin_name,
                    "n_pos": int(len(pos_scores)),
                    "n_neg": int(len(neg_scores)),
                    "pairwise_win_rate": pairwise_win_rate(pos_scores, neg_scores),
                    "score_gap": float(np.nanmean(pos_scores) - np.nanmean(neg_scores)),
                    "auroc_vs_bin": safe_metric(roc_auc_score, y_true, y_score),
                    "auprc_vs_bin": safe_metric(average_precision_score, y_true, y_score),
                    "neg_metric_mean": float(pd.to_numeric(neg[metric], errors="coerce").mean()),
                    "neg_metric_median": float(pd.to_numeric(neg[metric], errors="coerce").median()),
                }

                per_file_rows.append(row)

    per_file = pd.DataFrame(per_file_rows)

    if per_file.empty:
        raise ValueError("Structural stratification produced no rows.")

    summary_rows = []

    group_cols = [
        "model",
        "negative_ratio",
        "structural_metric",
        "hardness_bin",
    ]

    metrics = [
        "pairwise_win_rate",
        "score_gap",
        "auroc_vs_bin",
        "auprc_vs_bin",
        "n_pos",
        "n_neg",
        "neg_metric_mean",
        "neg_metric_median",
    ]

    for keys, g in per_file.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_files"] = int(g["test_file"].nunique())

        for m in metrics:
            mu, sd = mean_sd(g[m])
            row[f"{m}_mean"] = mu
            row[f"{m}_sd"] = sd

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(group_cols).reset_index(drop=True)

    return per_file, summary


# ============================================================
# Ranking / rescue-suppression analysis
# ============================================================

def add_within_file_ranks(pred_df):
    """
    Rank descending by score within model + test_file.
    rank 1 = highest score.
    """
    df = pred_df.copy()
    df["rank"] = (
        df.groupby(["model", "test_file"], dropna=False)["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return df


def make_rank_pivot(pred_ranked):
    """
    One row per test_file,row_index, with score/rank columns per model.
    """
    base_cols = [
        "test_file",
        "row_index",
        "negative_ratio",
        "negative_sampling_seed",
        "label",
        "drug_name",
        "disease_name",
        "shared_neighbor_count",
        "jaccard_shared_neighbors",
        "adamic_adar_score",
        "shortest_path_length_cutoff",
        "endpoint_total_degree_sum",
        "drug_total_degree",
        "disease_total_degree",
    ]

    base_cols = [c for c in base_cols if c in pred_ranked.columns]

    meta = (
        pred_ranked[base_cols]
        .drop_duplicates(subset=["test_file", "row_index"])
        .copy()
    )

    score_wide = pred_ranked.pivot_table(
        index=["test_file", "row_index"],
        columns="model",
        values="score",
        aggfunc="first",
    )
    score_wide.columns = [f"score__{c}" for c in score_wide.columns]

    rank_wide = pred_ranked.pivot_table(
        index=["test_file", "row_index"],
        columns="model",
        values="rank",
        aggfunc="first",
    )
    rank_wide.columns = [f"rank__{c}" for c in rank_wide.columns]

    wide = (
        meta
        .merge(score_wide.reset_index(), on=["test_file", "row_index"], how="left")
        .merge(rank_wide.reset_index(), on=["test_file", "row_index"], how="left")
    )

    return wide


def summarize_case_mask(df, mask, structural_cols):
    sub = df.loc[mask].copy()

    out = {
        "count": int(len(sub)),
    }

    for c in structural_cols:
        if c in sub.columns:
            out[f"{c}_mean"] = float(pd.to_numeric(sub[c], errors="coerce").mean()) if len(sub) else np.nan
            out[f"{c}_median"] = float(pd.to_numeric(sub[c], errors="coerce").median()) if len(sub) else np.nan

    return out


def compute_rescue_suppression(
    rank_wide,
    gnn_models,
    sft_model,
    kto_model,
    ablation_model,
    topk_list,
    max_cases_per_type=100,
):
    structural_cols = [
        "shared_neighbor_count",
        "jaccard_shared_neighbors",
        "adamic_adar_score",
        "shortest_path_length_cutoff",
        "endpoint_total_degree_sum",
        "drug_total_degree",
        "disease_total_degree",
    ]

    summary_rows = []
    case_rows = []

    available_cols = set(rank_wide.columns)

    needed_models = [sft_model, kto_model]
    if ablation_model:
        needed_models.append(ablation_model)

    for m in needed_models:
        if f"rank__{m}" not in available_cols:
            print(f"[WARN] Model missing from rank table: {m}")

    # --------------------------------------------------------
    # GNN -> SFT rescue for positives
    # --------------------------------------------------------
    for gnn_model in gnn_models:
        rank_gnn_col = f"rank__{gnn_model}"
        score_gnn_col = f"score__{gnn_model}"
        rank_sft_col = f"rank__{sft_model}"
        score_sft_col = f"score__{sft_model}"

        if rank_gnn_col not in available_cols or rank_sft_col not in available_cols:
            print(f"[WARN] Skipping GNN->SFT rescue for missing model: {gnn_model}")
            continue

        for K in topk_list:
            for (ratio, test_file), g in rank_wide.groupby(["negative_ratio", "test_file"], dropna=False):
                label = g["label"].astype(int)

                pos_total = int((label == 1).sum())

                sft_rescued = (
                    (label == 1)
                    & (pd.to_numeric(g[rank_gnn_col], errors="coerce") > K)
                    & (pd.to_numeric(g[rank_sft_col], errors="coerce") <= K)
                )

                gnn_top_pos = (
                    (label == 1)
                    & (pd.to_numeric(g[rank_gnn_col], errors="coerce") <= K)
                )

                sft_top_pos = (
                    (label == 1)
                    & (pd.to_numeric(g[rank_sft_col], errors="coerce") <= K)
                )

                row = {
                    "comparison": f"{gnn_model}_to_{sft_model}",
                    "case_type": "sft_rescued_positive_over_gnn",
                    "negative_ratio": int(ratio),
                    "test_file": test_file,
                    "K": int(K),
                    "n_positive_total": pos_total,
                    "gnn_top_positive_count": int(gnn_top_pos.sum()),
                    "sft_top_positive_count": int(sft_top_pos.sum()),
                    "sft_rescued_positive_count": int(sft_rescued.sum()),
                    "sft_rescued_positive_fraction": float(sft_rescued.sum() / pos_total) if pos_total else np.nan,
                }

                row.update({
                    f"rescued_{k}": v
                    for k, v in summarize_case_mask(g, sft_rescued, structural_cols).items()
                })

                summary_rows.append(row)

                cases = g.loc[sft_rescued].copy()
                if not cases.empty:
                    cases["rank_improvement"] = (
                        pd.to_numeric(cases[rank_gnn_col], errors="coerce")
                        - pd.to_numeric(cases[rank_sft_col], errors="coerce")
                    )
                    cases = cases.sort_values("rank_improvement", ascending=False).head(max_cases_per_type)

                    for _, c in cases.iterrows():
                        case_rows.append({
                            "comparison": f"{gnn_model}_to_{sft_model}",
                            "case_type": "sft_rescued_positive_over_gnn",
                            "negative_ratio": int(ratio),
                            "test_file": test_file,
                            "K": int(K),
                            "drug_name": c.get("drug_name"),
                            "disease_name": c.get("disease_name"),
                            "label": int(c.get("label")),
                            "rank_before": c.get(rank_gnn_col),
                            "rank_after": c.get(rank_sft_col),
                            "score_before": c.get(score_gnn_col),
                            "score_after": c.get(score_sft_col),
                            "rank_improvement": c.get("rank_improvement"),
                            **{col: c.get(col) for col in structural_cols if col in c.index},
                        })

    # --------------------------------------------------------
    # SFT -> KTO rescue positives and suppress negatives
    # --------------------------------------------------------
    rank_sft_col = f"rank__{sft_model}"
    score_sft_col = f"score__{sft_model}"
    rank_kto_col = f"rank__{kto_model}"
    score_kto_col = f"score__{kto_model}"

    if rank_sft_col in available_cols and rank_kto_col in available_cols:
        for K in topk_list:
            for (ratio, test_file), g in rank_wide.groupby(["negative_ratio", "test_file"], dropna=False):
                label = g["label"].astype(int)

                pos_total = int((label == 1).sum())
                neg_total = int((label == 0).sum())

                kto_rescued_pos = (
                    (label == 1)
                    & (pd.to_numeric(g[rank_sft_col], errors="coerce") > K)
                    & (pd.to_numeric(g[rank_kto_col], errors="coerce") <= K)
                )

                kto_suppressed_neg = (
                    (label == 0)
                    & (pd.to_numeric(g[rank_sft_col], errors="coerce") <= K)
                    & (pd.to_numeric(g[rank_kto_col], errors="coerce") > K)
                )

                kto_introduced_neg = (
                    (label == 0)
                    & (pd.to_numeric(g[rank_sft_col], errors="coerce") > K)
                    & (pd.to_numeric(g[rank_kto_col], errors="coerce") <= K)
                )

                sft_top_neg = (
                    (label == 0)
                    & (pd.to_numeric(g[rank_sft_col], errors="coerce") <= K)
                )

                kto_top_neg = (
                    (label == 0)
                    & (pd.to_numeric(g[rank_kto_col], errors="coerce") <= K)
                )

                row = {
                    "comparison": f"{sft_model}_to_{kto_model}",
                    "case_type": "kto_rescue_suppression_vs_sft",
                    "negative_ratio": int(ratio),
                    "test_file": test_file,
                    "K": int(K),
                    "n_positive_total": pos_total,
                    "n_negative_total": neg_total,
                    "kto_rescued_positive_count": int(kto_rescued_pos.sum()),
                    "kto_rescued_positive_fraction": float(kto_rescued_pos.sum() / pos_total) if pos_total else np.nan,
                    "sft_top_negative_count": int(sft_top_neg.sum()),
                    "kto_top_negative_count": int(kto_top_neg.sum()),
                    "kto_suppressed_negative_count": int(kto_suppressed_neg.sum()),
                    "kto_introduced_negative_count": int(kto_introduced_neg.sum()),
                    "net_topK_false_positive_reduction": int(kto_suppressed_neg.sum() - kto_introduced_neg.sum()),
                }

                row.update({
                    f"rescued_positive_{k}": v
                    for k, v in summarize_case_mask(g, kto_rescued_pos, structural_cols).items()
                })

                row.update({
                    f"suppressed_negative_{k}": v
                    for k, v in summarize_case_mask(g, kto_suppressed_neg, structural_cols).items()
                })

                row.update({
                    f"introduced_negative_{k}": v
                    for k, v in summarize_case_mask(g, kto_introduced_neg, structural_cols).items()
                })

                summary_rows.append(row)

                case_defs = [
                    ("kto_rescued_positive_over_sft", kto_rescued_pos, True),
                    ("kto_suppressed_negative_from_sft_topK", kto_suppressed_neg, False),
                    ("kto_introduced_negative_into_topK", kto_introduced_neg, False),
                ]

                for case_type, mask, is_positive in case_defs:
                    cases = g.loc[mask].copy()
                    if cases.empty:
                        continue

                    if is_positive:
                        cases["rank_improvement"] = (
                            pd.to_numeric(cases[rank_sft_col], errors="coerce")
                            - pd.to_numeric(cases[rank_kto_col], errors="coerce")
                        )
                        cases = cases.sort_values("rank_improvement", ascending=False).head(max_cases_per_type)
                    else:
                        cases["rank_change"] = (
                            pd.to_numeric(cases[rank_kto_col], errors="coerce")
                            - pd.to_numeric(cases[rank_sft_col], errors="coerce")
                        )
                        cases = cases.sort_values("rank_change", ascending=False).head(max_cases_per_type)

                    for _, c in cases.iterrows():
                        case_rows.append({
                            "comparison": f"{sft_model}_to_{kto_model}",
                            "case_type": case_type,
                            "negative_ratio": int(ratio),
                            "test_file": test_file,
                            "K": int(K),
                            "drug_name": c.get("drug_name"),
                            "disease_name": c.get("disease_name"),
                            "label": int(c.get("label")),
                            "rank_before": c.get(rank_sft_col),
                            "rank_after": c.get(rank_kto_col),
                            "score_before": c.get(score_sft_col),
                            "score_after": c.get(score_kto_col),
                            "rank_improvement_or_change": c.get("rank_improvement", c.get("rank_change")),
                            **{col: c.get(col) for col in structural_cols if col in c.index},
                        })

    # --------------------------------------------------------
    # Ablation -> KTO optional comparison
    # --------------------------------------------------------
    if ablation_model:
        rank_ab_col = f"rank__{ablation_model}"
        score_ab_col = f"score__{ablation_model}"

        if rank_ab_col in available_cols and rank_kto_col in available_cols:
            for K in topk_list:
                for (ratio, test_file), g in rank_wide.groupby(["negative_ratio", "test_file"], dropna=False):
                    label = g["label"].astype(int)

                    pos_total = int((label == 1).sum())
                    neg_total = int((label == 0).sum())

                    kto_rescued_pos = (
                        (label == 1)
                        & (pd.to_numeric(g[rank_ab_col], errors="coerce") > K)
                        & (pd.to_numeric(g[rank_kto_col], errors="coerce") <= K)
                    )

                    kto_suppressed_neg = (
                        (label == 0)
                        & (pd.to_numeric(g[rank_ab_col], errors="coerce") <= K)
                        & (pd.to_numeric(g[rank_kto_col], errors="coerce") > K)
                    )

                    kto_introduced_neg = (
                        (label == 0)
                        & (pd.to_numeric(g[rank_ab_col], errors="coerce") > K)
                        & (pd.to_numeric(g[rank_kto_col], errors="coerce") <= K)
                    )

                    row = {
                        "comparison": f"{ablation_model}_to_{kto_model}",
                        "case_type": "kto_rescue_suppression_vs_ablation",
                        "negative_ratio": int(ratio),
                        "test_file": test_file,
                        "K": int(K),
                        "n_positive_total": pos_total,
                        "n_negative_total": neg_total,
                        "kto_rescued_positive_count": int(kto_rescued_pos.sum()),
                        "kto_rescued_positive_fraction": float(kto_rescued_pos.sum() / pos_total) if pos_total else np.nan,
                        "kto_suppressed_negative_count": int(kto_suppressed_neg.sum()),
                        "kto_introduced_negative_count": int(kto_introduced_neg.sum()),
                        "net_topK_false_positive_reduction": int(kto_suppressed_neg.sum() - kto_introduced_neg.sum()),
                    }

                    summary_rows.append(row)

    per_file_summary = pd.DataFrame(summary_rows)
    cases = pd.DataFrame(case_rows)

    aggregate_rows = []

    if not per_file_summary.empty:
        id_cols = ["comparison", "case_type", "negative_ratio", "K"]
        numeric_cols = [
            c for c in per_file_summary.columns
            if c not in id_cols + ["test_file"]
            and pd.api.types.is_numeric_dtype(per_file_summary[c])
        ]

        for keys, g in per_file_summary.groupby(id_cols, dropna=False):
            row = dict(zip(id_cols, keys))
            row["n_files"] = int(g["test_file"].nunique())

            for c in numeric_cols:
                mu, sd = mean_sd(g[c])
                row[f"{c}_mean"] = mu
                row[f"{c}_sd"] = sd

            aggregate_rows.append(row)

    aggregate = pd.DataFrame(aggregate_rows)

    return per_file_summary, aggregate, cases


# ============================================================
# GraphPad exports
# ============================================================

def export_graphpad(struct_per_file, rescue_per_file, output_dir):
    graphpad_dir = Path(output_dir) / "graphpad"
    graphpad_dir.mkdir(parents=True, exist_ok=True)

    if not struct_per_file.empty:
        struct_per_file.to_csv(
            graphpad_dir / "graphpad_structural_hardness_per_file_tidy.csv",
            index=False,
        )

        # One tidy file for key metric.
        key = struct_per_file[[
            "model",
            "negative_ratio",
            "test_file",
            "structural_metric",
            "hardness_bin",
            "pairwise_win_rate",
            "score_gap",
            "auroc_vs_bin",
            "auprc_vs_bin",
        ]].copy()

        key.to_csv(
            graphpad_dir / "graphpad_pairwise_win_rate_by_hardness.csv",
            index=False,
        )

    if rescue_per_file is not None and not rescue_per_file.empty:
        rescue_per_file.to_csv(
            graphpad_dir / "graphpad_rescue_suppression_per_file_tidy.csv",
            index=False,
        )

    print(f"[INFO] Saved GraphPad data to: {graphpad_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Structural-hardness stratification and rescue/suppression analysis "
            "for GNN, SFT, and KTO predictions."
        )
    )

    parser.add_argument(
        "--llm-jsonl",
        nargs="*",
        default=[
            "data/result_llm/llama3-8b.jsonl",
            "data/result_llm/qwen3-8b.jsonl",
            "data/result_llm/qwen3-8b-ablation.jsonl",
            "data/result_llm/qwen3-8b-kto.jsonl",
            "data/result_llm/qwen3-8b-sft.jsonl",
            "data/result_llm/qwen3-235b-a22b.jsonl",
        ],
        help="LLM per-row JSONL prediction files.",
    )
    parser.add_argument(
        "--gnn-csv",
        default="data/result_gnn/gnn_fixed_test_predictions.csv",
        help="GNN per-row prediction CSV.",
    )
    parser.add_argument(
        "--train-csv",
        default="data/benchmark/PrimeKG/full_graph_42/train.csv",
        help="Training graph CSV.",
    )
    parser.add_argument(
        "--nodes-csv",
        default="data/benchmark/PrimeKG/nodes.csv",
        help="Node table CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/result_mechanism/structural_hardness_rescue",
        help="Output directory.",
    )

    parser.add_argument("--graph-group", default="pyg_style")
    parser.add_argument("--strategy", default="degree_matched_head")
    parser.add_argument("--ratios", nargs="*", type=int, default=[4, 9])
    parser.add_argument("--source-format", default="gnn")

    parser.add_argument(
        "--gnn-models",
        nargs="*",
        default=["all"],
        help="GNN models to include. Use 'all' for all models in the GNN CSV.",
    )
    parser.add_argument("--sft-model", default="qwen3-8b-sft")
    parser.add_argument("--kto-model", default="qwen3-8b-kto")
    parser.add_argument("--ablation-model", default="qwen3-8b-ablation")
    parser.add_argument("--base-model", default="qwen3-8b")

    parser.add_argument(
        "--topk",
        nargs="*",
        type=int,
        default=[10, 50, 100, 200],
        help="K values for rescue/suppression analysis.",
    )
    parser.add_argument(
        "--structural-metrics",
        nargs="*",
        default=[
            "shared_neighbor_count",
            "jaccard_shared_neighbors",
            "adamic_adar_score",
            "inv_shortest_path_hardness",
        ],
        help="Structural metrics used for hardness binning. Higher means harder.",
    )
    parser.add_argument(
        "--shortest-cutoff",
        type=int,
        default=4,
        help="Shortest path BFS cutoff. Unreached paths are assigned cutoff+1.",
    )
    parser.add_argument(
        "--max-cases-per-type",
        type=int,
        default=100,
        help="Maximum example cases saved per case type per file/K.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading training graph and node table")
    print("=" * 80)

    mapped_train, nodes = load_train_edges_with_node_index(
        train_csv=args.train_csv,
        nodes_csv=args.nodes_csv,
    )
    adj = build_undirected_adjacency(mapped_train)
    name_to_node_index = build_name_to_node_index(nodes)

    print(f"[INFO] Graph nodes with edges: {len(adj):,}")

    print("=" * 80)
    print("Loading predictions")
    print("=" * 80)

    llm_df = load_llm_jsonl_files(
        llm_jsonl_paths=args.llm_jsonl,
        graph_group=args.graph_group,
        strategy=args.strategy,
        ratios=args.ratios,
        source_format=args.source_format,
    )

    gnn_df = load_gnn_predictions(
        gnn_csv=args.gnn_csv,
        graph_group=args.graph_group,
        strategy=args.strategy,
        ratios=args.ratios,
        gnn_models=args.gnn_models,
        gnn_score_col="pred_prob",
    )

    common_files = sorted(set(llm_df["test_file"].unique()) & set(gnn_df["test_file"].unique()))
    if len(common_files) == 0:
        raise ValueError("No common test_file values between LLM and GNN predictions.")

    print(f"[INFO] Common test files: {len(common_files):,}")

    llm_df = llm_df[llm_df["test_file"].isin(common_files)].copy()
    gnn_df = gnn_df[gnn_df["test_file"].isin(common_files)].copy()

    print("=" * 80)
    print("Building canonical pair metadata from GNN predictions")
    print("=" * 80)

    pair_meta = build_pair_metadata_from_gnn(gnn_df)

    print(f"[INFO] Canonical test pairs: {len(pair_meta):,}")

    print("=" * 80)
    print("Computing structural metrics")
    print("=" * 80)

    struct_df = compute_pair_structural_metrics(
        pair_df=pair_meta,
        adj=adj,
        name_to_node_index=name_to_node_index,
        shortest_cutoff=args.shortest_cutoff,
    )

    struct_path = output_dir / "pair_structural_metrics.csv"
    struct_df.to_csv(struct_path, index=False)
    print(f"[INFO] Saved pair structural metrics: {struct_path}")

    # Merge structural metrics into predictions.
    pred_df = pd.concat([
        llm_df,
        gnn_df.drop(columns=["x_id", "y_id"], errors="ignore"),
    ], ignore_index=True)

    pred_df = pred_df.merge(
        struct_df,
        on=["test_file", "row_index"],
        how="left",
    )

    # Also merge canonical label/name metadata to keep everything consistent.
    pred_df = pred_df.merge(
        pair_meta[[
            "test_file",
            "row_index",
            "label",
            "drug_name",
            "disease_name",
        ]].rename(columns={
            "label": "canonical_label",
            "drug_name": "canonical_drug_name",
            "disease_name": "canonical_disease_name",
        }),
        on=["test_file", "row_index"],
        how="left",
    )

    # Prefer canonical metadata.
    pred_df["label"] = pred_df["canonical_label"].fillna(pred_df["label"]).astype(int)
    pred_df["drug_name"] = pred_df["canonical_drug_name"].fillna(pred_df["drug_name"])
    pred_df["disease_name"] = pred_df["canonical_disease_name"].fillna(pred_df["disease_name"])

    pred_df = pred_df.drop(columns=[
        "canonical_label",
        "canonical_drug_name",
        "canonical_disease_name",
    ])

    pred_df = pred_df.dropna(subset=["score"]).copy()

    combined_path = output_dir / "combined_predictions_with_structural_metrics.csv"
    pred_df.to_csv(combined_path, index=False)
    print(f"[INFO] Saved combined predictions: {combined_path}")

    print("=" * 80)
    print("Structural difficulty stratification analysis")
    print("=" * 80)

    structural_per_file, structural_summary = compute_structural_stratification(
        pred_df=pred_df,
        structural_metrics=args.structural_metrics,
    )

    structural_per_file_path = output_dir / "structural_hardness_per_file.csv"
    structural_summary_path = output_dir / "manuscript_structural_hardness_summary.csv"

    structural_per_file.to_csv(structural_per_file_path, index=False)
    structural_summary.to_csv(structural_summary_path, index=False)

    print(f"[INFO] Saved structural per-file: {structural_per_file_path}")
    print(f"[INFO] Saved structural summary: {structural_summary_path}")

    print("=" * 80)
    print("Rescue / suppression analysis")
    print("=" * 80)

    pred_ranked = add_within_file_ranks(pred_df)
    rank_wide = make_rank_pivot(pred_ranked)

    rank_wide_path = output_dir / "rank_wide_table.csv"
    rank_wide.to_csv(rank_wide_path, index=False)
    print(f"[INFO] Saved rank wide table: {rank_wide_path}")

    gnn_models = sorted(gnn_df["model"].dropna().unique().tolist())

    print(f"[INFO] GNN models for rescue analysis: {gnn_models}")
    print(f"[INFO] SFT model: {args.sft_model}")
    print(f"[INFO] KTO model: {args.kto_model}")
    print(f"[INFO] Ablation model: {args.ablation_model}")
    print(f"[INFO] TopK values: {args.topk}")

    rescue_per_file, rescue_summary, rescue_cases = compute_rescue_suppression(
        rank_wide=rank_wide,
        gnn_models=gnn_models,
        sft_model=args.sft_model,
        kto_model=args.kto_model,
        ablation_model=args.ablation_model,
        topk_list=args.topk,
        max_cases_per_type=args.max_cases_per_type,
    )

    rescue_per_file_path = output_dir / "rescue_suppression_per_file.csv"
    rescue_summary_path = output_dir / "supplementary_rescue_suppression_summary.csv"
    rescue_cases_path = output_dir / "supplementary_rescue_suppression_cases.csv"

    rescue_per_file.to_csv(rescue_per_file_path, index=False)
    rescue_summary.to_csv(rescue_summary_path, index=False)
    rescue_cases.to_csv(rescue_cases_path, index=False)

    print(f"[INFO] Saved rescue per-file: {rescue_per_file_path}")
    print(f"[INFO] Saved rescue summary: {rescue_summary_path}")
    print(f"[INFO] Saved rescue cases: {rescue_cases_path}")

    print("=" * 80)
    print("GraphPad export")
    print("=" * 80)

    export_graphpad(
        struct_per_file=structural_per_file,
        rescue_per_file=rescue_per_file,
        output_dir=output_dir,
    )

    print("=" * 80)
    print("Done")
    print("=" * 80)

    print("\nStructural hardness summary preview:")
    preview_cols = [
        "model",
        "negative_ratio",
        "structural_metric",
        "hardness_bin",
        "n_files",
        "pairwise_win_rate_mean",
        "pairwise_win_rate_sd",
        "score_gap_mean",
        "score_gap_sd",
        "auroc_vs_bin_mean",
        "auroc_vs_bin_sd",
    ]
    preview_cols = [c for c in preview_cols if c in structural_summary.columns]
    print(structural_summary[preview_cols].head(80).to_string(index=False))

    print("\nRescue/suppression summary preview:")
    if not rescue_summary.empty:
        preview_cols = [
            "comparison",
            "case_type",
            "negative_ratio",
            "K",
            "n_files",
            "sft_rescued_positive_count_mean",
            "kto_rescued_positive_count_mean",
            "kto_suppressed_negative_count_mean",
            "kto_introduced_negative_count_mean",
            "net_topK_false_positive_reduction_mean",
        ]
        preview_cols = [c for c in preview_cols if c in rescue_summary.columns]
        print(rescue_summary[preview_cols].head(80).to_string(index=False))
    else:
        print("[WARN] rescue_summary is empty.")


if __name__ == "__main__":
    main()