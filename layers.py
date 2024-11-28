import math

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


class CustomSAGEConv(SAGEConv):
    def __init__(self, node_dim, edge_dim, aggr='mean', normalize=True, root_weight=True, bias=True, device='cpu'):
        super(CustomSAGEConv, self).__init__(node_dim, edge_dim)
        self.relu = nn.ReLU()
        self.aggr = aggr
        self.normalize = normalize
        self.root_weight = root_weight
        self.bias = bias
        self.device = device
        if self.root_weight:
            weight_dim = 3*node_dim + edge_dim
        else:
            weight_dim = 2*node_dim + edge_dim
        self.lin = nn.Linear(weight_dim, node_dim, bias=self.bias)
        # nn.init.xavier_normal_(self.lin.weight)


    def forward(self, x, edge_index, edge_features, num_nodes, num_neighbor=None):
        v = edge_index[0, :]
        u = edge_index[1, :]
        v_node_features = x[v]
        u_node_features = x[u]

        message = torch.cat([v_node_features, u_node_features, edge_features], dim=-1).to(self.device)
        message_dim = message.size(1)
        node_messages = torch.zeros((num_nodes, message_dim), device=self.device)

        # 이웃 샘플링 (num_neighbor가 설정되어 있을 때만)
        if num_neighbor is not None:
            # 각 노드에 대해 이웃을 샘플링
            unique_v, counts_v = torch.unique(v, return_counts=True)  # 각 노드의 이웃 수 계산
            # unique_u, counts_u = torch.unique(u, return_counts=True)

            sampled_edges = []
            for node, count in zip(unique_v, counts_v):
                neighbors_idx = (v == node).nonzero(as_tuple=False).squeeze()

                if neighbors_idx.dim() == 0:  # 이웃이 없는 경우에 대비한 처리
                    neighbors_idx = neighbors_idx.unsqueeze(0)

                if count > num_neighbor:  # 이웃이 num_neighbor보다 많은 경우
                    sampled_neighbors = neighbors_idx[torch.randperm(neighbors_idx.size(0))[:num_neighbor]]
                    sampled_edges.append(sampled_neighbors)
                else:
                    sampled_edges.append(neighbors_idx)


            sampled_edges = [edge for edge in sampled_edges if edge.numel() > 0]
            sampled_edges = torch.cat(sampled_edges)

            v = v[sampled_edges]
            u = u[sampled_edges]
            message = message[sampled_edges]

        if self.aggr == 'mean':
            node_messages = node_messages.index_add(0, v, message.float())
            node_messages = node_messages.index_add(0, u, message.float())
            if num_neighbor:
               num_neighbors = torch.bincount(v, minlength=num_nodes).clamp(max=num_neighbor) \
                + torch.bincount(u, minlength=num_nodes).clamp(max=num_neighbor)
            else: 
                num_neighbors = torch.bincount(v, minlength=num_nodes) + torch.bincount(u, minlength=num_nodes)
            node_messages /= (num_neighbors.unsqueeze(1) + 1e-6)

        if self.root_weight:
            out = torch.cat([x, node_messages], dim=-1).to(self.device)
        else:
            out = node_messages

        out = self.lin(out.float())

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

