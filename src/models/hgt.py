import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv


class HGT(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_rels, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_rels = num_rels

        # 1. 节点 Embedding
        # HGT 需要对每种节点类型分别 Embedding，我们这里只有一种节点类型 'entity'
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # 2. 关系 Embedding (用于 DistMult 打分)
        self.relation_embedding = nn.Embedding(num_rels, hidden_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

        # 3. 构建 Metadata
        # HGTConv 需要知道图里有哪些节点类型和边类型
        # 我们将所有关系模拟为 ('entity', 'relation_i', 'entity')
        node_types = ['entity']
        edge_types = [('entity', str(i), 'entity') for i in range(num_rels)]
        metadata = (node_types, edge_types)

        # 4. 定义 HGT 层
        self.convs = nn.ModuleList()
        self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))

        for _ in range(num_layers - 2):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))

        self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))

        self.dropout = dropout
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_type):
        # --- 适配器开始：将 Tensor 转换为 Dict ---
        # 输入 x: [N] (节点 ID)
        x_emb = self.embedding(x)  # [N, hidden_dim]

        # 构造 x_dict
        x_dict = {'entity': x_emb}

        # 构造 edge_index_dict
        # 这一步会在 Batch 内部动态把 unified edge_index 拆分成不同关系的 edge_index
        edge_index_dict = {}

        # 找出当前 Batch 中实际存在的关系类型，避免无效循环
        unique_rels = torch.unique(edge_type)

        for r in unique_rels:
            r_idx = r.item()
            # 筛选出该关系的边
            mask = (edge_type == r)
            # 存入字典，键必须与 metadata 里的定义一致
            edge_index_dict[('entity', str(r_idx), 'entity')] = edge_index[:, mask]

        # --- 适配器结束 ---

        # HGT 传播
        h_dict = x_dict
        for conv in self.convs:
            # HGTConv 返回的是字典 {'entity': new_emb}
            h_dict = conv(h_dict, edge_index_dict)

            # 对字典里的每个 value 做激活和 dropout
            h_dict = {key: self.relu(val) for key, val in h_dict.items()}
            h_dict = {key: torch.dropout(val, p=self.dropout, train=self.training) for key, val in h_dict.items()}

        # 取出最终的节点表示
        return h_dict['entity']

    def score(self, node_emb, src, tgt, rel):
        # 使用与 RGCN 完全一致的 DistMult 打分函数，保证公平对比
        return (node_emb[src] * self.relation_embedding(rel) * node_emb[tgt]).sum(dim=1)