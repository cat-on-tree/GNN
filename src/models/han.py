import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class HANLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, dropout):
        super().__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=num_heads, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        return self.gat_conv(x, edge_index)


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # z: [num_nodes, num_metapaths, hidden_dim]
        w = self.project(z).mean(0)  # [num_metapaths, 1]
        beta = torch.softmax(w, dim=0)  # [num_metapaths, 1]
        beta = beta.expand((z.shape[0],) + beta.shape)  # [N, M, 1]
        return (beta * z).sum(1)  # [N, hidden_dim]


class HAN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_rels, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_rels = num_rels
        self.num_layers = num_layers

        # 1. 节点 Embedding
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # 2. 关系 Embedding (用于 DistMult 打分)
        self.relation_embedding = nn.Embedding(num_rels, hidden_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

        # 3. HAN 核心组件
        # 我们把每种 relation 视为一种 meta-path
        # Node-level Attention: 为每种关系建立一个 GAT
        self.gat_layers = nn.ModuleList()
        for _ in range(num_rels):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
            )

        # Semantic-level Attention: 融合不同关系的结果
        self.semantic_attention = SemanticAttention(hidden_dim, hidden_dim)

        self.dropout = dropout
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_type):
        # 1. 初始特征
        h = self.embedding(x)  # [N, D]

        # HAN 通常比较深，但为了兼容性，我们这里实现单层的 HAN 逻辑叠加
        # (因为标准 HAN 是针对特定 meta-path 做一次聚合)
        # 如果需要多层，可以在这里加循环，但显存消耗会巨大。
        # 这里我们实现：Embedding -> Node Level (Multi-relation) -> Semantic Level -> Output

        # 2. Node-level Attention (针对每种关系分别做 GAT)
        semantic_embeddings = []

        # 找出当前 Batch 存在的 unique relations
        unique_rels = torch.unique(edge_type)

        # 这一步如果不优化，对所有 num_rels 循环会很慢
        # 但为了实现 HAN 的语义融合，必须分开处理

        # 创建一个全零的输出张量作为底板
        # 注意：这里有一个工程权衡。标准 HAN 是对所有 meta-path 都算一遍。
        # 但我们的图有 30+ 种关系，如果全算一遍显存会爆。
        # 所以我们只计算当前 batch 存在的那些关系。

        # 为了方便堆叠，我们还是得构建一个 [N, num_active_rels, D] 的张量

        for r_idx in unique_rels:
            r = r_idx.item()
            mask = (edge_type == r)
            sub_edge_index = edge_index[:, mask]

            # 使用对应的 GAT 层处理
            # 注意：这里我们让所有关系共享同一套 GAT 参数，或者每种关系独立参数
            # 为了更像 HAN，每种关系应该有独立的视角 -> 使用 self.gat_layers[r]
            if sub_edge_index.size(1) > 0:
                out = self.gat_layers[r](h, sub_edge_index)
                semantic_embeddings.append(out)
            else:
                # 理论上不会进这里，因为 unique_rels 保证了存在
                pass

        if len(semantic_embeddings) == 0:
            return h  # Fallback

        # 3. Semantic-level Attention
        # Stack: [N, num_active_rels, D]
        # 注意：这里堆叠的顺序不重要，因为 SemanticAttention 会自己学权重
        z = torch.stack(semantic_embeddings, dim=1)

        # 融合
        h_out = self.semantic_attention(z)

        # 可选：残差连接或多层叠加 (这里做简单的一层处理)
        h_out = self.relu(h_out)
        h_out = torch.dropout(h_out, p=self.dropout, train=self.training)

        return h_out

    def score(self, node_emb, src, tgt, rel):
        return (node_emb[src] * self.relation_embedding(rel) * node_emb[tgt]).sum(dim=1)