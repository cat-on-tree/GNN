import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_rel, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.relation_embedding = nn.Embedding(num_rel, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_rel))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_rel))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_rel))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_type):
        # 统一接口：不管模型用不用 edge_type，参数都传进来
        h = self.embedding(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, edge_type)  # 如果是 GAT，这里可能不用 edge_type
            h = bn(h)
            h = torch.relu(h)
            h = torch.dropout(h, p=self.dropout, train=self.training)
        return h

    def score(self, node_emb, src, tgt, rel):
        # DistMult 打分函数
        return (node_emb[src] * self.relation_embedding(rel) * node_emb[tgt]).sum(dim=1)