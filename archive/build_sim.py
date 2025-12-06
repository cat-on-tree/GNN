import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
import os
import sys
import time


def compute_real_jaccard_sim(data_folder, save_path):
    start_time = time.time()
    print(f"🚀 [High-Performance] Starting Full Jaccard Similarity Computation...")
    print(f"📂 Data Source: {data_folder}")
    print(f"💾 Save Target: {save_path}")

    # 1. 读取数据
    nodes_path = os.path.join(data_folder, 'nodes.csv')
    edges_path = os.path.join(data_folder, 'train_edges.csv')  # 严格使用训练集

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print(f"❌ Error: Data files not found at {data_folder}")
        return

    print("   Loading CSVs...")
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    num_nodes = len(nodes)
    print(f"   - Total Nodes: {num_nodes}")
    print(f"   - Training Edges: {len(edges)}")

    # 2. 识别疾病节点
    if 'node_type' in nodes.columns:
        disease_nodes = nodes[nodes['node_type'] == 'disease']
        disease_indices = disease_nodes['node_index'].values
        disease_indices.sort()  # 排序很重要
    else:
        raise ValueError("Column 'node_type' not found. Cannot identify diseases.")

    num_diseases = len(disease_indices)
    print(f"   - Disease Nodes: {num_diseases}")

    # 3. 构建稀疏邻接矩阵 (CSR Matrix)
    print("   Building Sparse Adjacency Matrix...")
    row = edges['x_index'].values
    col = edges['y_index'].values
    data = np.ones(len(row), dtype=np.float32)

    # 构建双向图 (Undirected)
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)  # Binary

    # 4. 计算交集 (Intersection)
    print("   ⚡ Calculating Intersection (A_disease @ A_disease.T)...")
    disease_adj = adj[disease_indices]
    intersection = disease_adj.dot(disease_adj.T)

    # 转为 Dense
    print("   Converting to Dense Matrix...")
    intersection_dense = intersection.toarray().astype(np.float32)

    # 5. 计算 Jaccard
    print("   ➗ Computing Jaccard Coefficient...")
    degrees = np.array(adj.sum(axis=1)).flatten().astype(np.float32)
    disease_degrees = degrees[disease_indices]

    deg_matrix = disease_degrees[:, None] + disease_degrees[None, :]
    union_matrix = deg_matrix - intersection_dense

    sim_matrix = intersection_dense / (union_matrix + 1e-9)
    np.fill_diagonal(sim_matrix, 1.0)

    # === Sanity Check (检查数据质量) ===
    print("\n   🔍 [Sanity Check]")
    print(f"      - Matrix Shape: {sim_matrix.shape}")
    print(f"      - Min Value: {sim_matrix.min():.4f}")
    print(f"      - Max Value: {sim_matrix.max():.4f}")
    print(f"      - Mean Value: {sim_matrix.mean():.4f}")
    # 检查有多少非对角元素非零
    non_zero_ratio = (np.count_nonzero(sim_matrix) - num_diseases) / (num_diseases * num_diseases) * 100
    print(f"      - Non-zero Similarity Ratio: {non_zero_ratio:.2f}% (Expected to be sparse-ish)")

    # 6. 保存
    print("\n   💾 Saving to .pt file...")

    sim_tensor = torch.FloatTensor(sim_matrix)
    disease_degrees_tensor = torch.FloatTensor(disease_degrees)
    disease_indices_tensor = torch.LongTensor(disease_indices)

    output = {
        'sim_matrix': sim_tensor,
        'disease_degrees': disease_degrees_tensor,
        'disease_global_indices': disease_indices_tensor
    }

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(output, save_path)

    end_time = time.time()
    print(f"✅ Done! Saved to {save_path}")
    print(f"⏱ Total time: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    # 根目录下的相对路径
    DATA_DIR = "data/benchmark/PrimeKG"
    SAVE_PATH = "model/TxGNN/txgnn_sim_data.pt"

    # 简单的路径回退逻辑，防止你在不同目录下运行
    if not os.path.exists(DATA_DIR) and os.path.exists("../data/benchmark/PrimeKG"):
        DATA_DIR = "../data/benchmark/PrimeKG"
        SAVE_PATH = "../model/TxGNN/txgnn_sim_data.pt"

    compute_real_jaccard_sim(DATA_DIR, SAVE_PATH)