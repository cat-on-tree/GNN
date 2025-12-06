import torch
import torch.nn as nn
from .rgcn import RGCN


class TxGNNModel(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_rels, device='cuda'):
        super().__init__()
        self.device = device

        # 1. 基础 Encoder (RGCN)
        self.encoder = RGCN(num_nodes, hidden_dim, num_rels)

        # 2. 相似度相关
        self.sim_matrix = None
        self.disease_degrees = None
        self.disease_indices_map = None
        self.global2local = None

    def _safe_register_buffer(self, name, tensor):
        if hasattr(self, name):
            delattr(self, name)
        self.register_buffer(name, tensor)

    def load_similarity(self, sim_path):
        print(f"Loading similarity matrix from {sim_path}...")
        data = torch.load(sim_path, map_location=self.device)

        if 'sim_matrix' in data:
            self._safe_register_buffer('sim_matrix', data['sim_matrix'].to(self.device))
        else:
            raise KeyError("Key 'sim_matrix' not found")

        if 'disease_degrees' in data:
            self._safe_register_buffer('disease_degrees', data['disease_degrees'].to(self.device))
        else:
            self._safe_register_buffer('disease_degrees', torch.sum(self.sim_matrix > 0, dim=1).float())

        if 'disease_global_indices' in data:
            self._safe_register_buffer('disease_indices_map', data['disease_global_indices'].to(self.device))
        else:
            raise KeyError("Key 'disease_global_indices' not found")

        # 构建 Global ID -> Sim Matrix Row Index 的映射表
        max_node_id = int(self.disease_indices_map.max().item())
        lookup_tensor = torch.full((max_node_id + 1,), -1, dtype=torch.long, device=self.device)

        local_indices = torch.arange(len(self.disease_indices_map), device=self.device)
        lookup_tensor[self.disease_indices_map] = local_indices

        self._safe_register_buffer('global2local', lookup_tensor)
        print("✅ Similarity data loaded (Full Matrix Mode).")

    def forward(self, x, edge_index, edge_type, batch_n_id=None):
        """
        Args:
            x: Node features
            edge_index: Local edge index
            edge_type: Edge types
            batch_n_id: [Batch_Nodes] Global IDs of nodes in current batch (Required for Mini-batch)
        """
        # 1. RGCN 原始编码
        h = self.encoder(x, edge_index, edge_type)

        # 如果没有 Sim 矩阵，或者是在 Full Batch 推理但没传 n_id，直接返回
        if self.sim_matrix is None:
            return h

        # 2. 确定当前 Batch 里的疾病节点
        # batch_n_id 包含了当前 batch 节点的 Global ID
        # 如果没有传入 (比如全图推理)，则假设输入的 x 就是全图顺序
        if batch_n_id is None:
            # 尝试假设是全图训练 (慎用)
            # 如果 x 的大小等于 global2local 的大小，可能是在做全图
            # 但为了安全，如果没有 n_id，我们跳过 TxGNN 增强，避免崩溃
            return h

        # 查表：Batch里的节点 -> Sim Matrix 的第几行？
        # 如果不是疾病，值为 -1
        # 注意：batch_n_id 可能包含超出 global2local 范围的节点 (如果 Sim 矩阵只覆盖了部分图)
        # 做一个安全截断或掩码

        valid_mask = batch_n_id < self.global2local.size(0)
        batch_sim_indices = torch.full_like(batch_n_id, -1, dtype=torch.long)
        batch_sim_indices[valid_mask] = self.global2local[batch_n_id[valid_mask]]

        # 找出当前 Batch 中，是疾病的节点
        # is_disease_mask: [Batch_Size] (Bool)
        is_disease_mask = batch_sim_indices != -1

        # 如果当前 Batch 里没有疾病，直接返回
        if not is_disease_mask.any():
            return h

        # 获取这些疾病在 h 中的索引 (Local Batch Index)
        batch_loc_indices = torch.where(is_disease_mask)[0]
        # 获取这些疾病在 Sim Matrix 中的行号
        matrix_row_indices = batch_sim_indices[is_disease_mask]

        # 3. TxGNN 增强逻辑

        # A. 计算 Gating 系数 c (基于 Batch 内的度数)
        deg = torch.zeros(h.size(0), device=self.device)
        ones = torch.ones(edge_index.size(1), device=self.device)
        deg.scatter_add_(0, edge_index[1], ones)  # 此时 edge_index 已经是 local 的，安全

        c = 0.7 * torch.exp(-0.7 * deg).unsqueeze(1) + 0.2

        # B. 计算相似 Embedding (h_sim)
        # 问题：Sim Matrix 是全图的，但我们只拿到 Batch 的 Embedding。
        # 策略：我们只聚合 "当前 Batch 内存在的相似邻居"。这是 Mini-batch GNN 的标准妥协。

        # 取出当前 Batch 内所有疾病的 Embedding
        # h_batch_diseases: [Num_Batch_Diseases, Dim]
        h_batch_diseases = h[batch_loc_indices]

        # 取出 Sim Matrix 的子矩阵
        # Rows: 当前 Batch 的疾病
        # Cols: 当前 Batch 的疾病
        # [Num_Batch_Diseases, Num_Batch_Diseases]
        sim_sub = self.sim_matrix[matrix_row_indices][:, matrix_row_indices]

        # Row Normalize (在子图内部归一化)
        sim_sum = sim_sub.sum(dim=1, keepdim=True) + 1e-9
        sim_norm = sim_sub / sim_sum

        # 聚合: [B_D, B_D] @ [B_D, Dim] -> [B_D, Dim]
        h_sim = torch.mm(sim_norm, h_batch_diseases)

        # 4. 融合
        c_subset = c[batch_loc_indices]
        h_orig_subset = h[batch_loc_indices]

        h_augmented = c_subset * h_sim + (1 - c_subset) * h_orig_subset

        # 写回
        h_final = h.clone()
        h_final[batch_loc_indices] = h_augmented

        return h_final

    def score(self, node_emb, src, tgt, rel):
        return self.encoder.score(node_emb, src, tgt, rel)