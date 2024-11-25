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

            # 1차원 이상의 텐서만 연결
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
        # out = self.relu(out)

        # 선택적 정규화 적용
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    # def forward(self, x, edge_index, edge_features, num_nodes, num_neighbor=None):
    #     v = edge_index[0, :]
    #     u = edge_index[1, :]
    #     v_node_features = x[v]
    #     u_node_features = x[u]

    #     message = torch.cat([v_node_features, u_node_features, edge_features], dim=-1).to(self.device)
    #     message_dim = message.size(1)
    #     node_messages = torch.zeros((num_nodes, message_dim), device=self.device)

    #     if self.aggr == 'mean':
    #         node_messages.index_add(0, v, message.float())
    #         node_messages.index_add(0, u, message.float())
    #         num_neighbors = torch.bincount(v, minlength=num_nodes) + torch.bincount(u, minlength=num_nodes)
    #         node_messages /= (num_neighbors.unsqueeze(1) + 1e-6)
    #         # node_messages /= num_nodes
    #     if self.root_weight:
    #         out = torch.cat([x, node_messages], dim=-1).to(self.device)
    #     else:
    #         out = node_messages
    #     out = self.lin(out.float())

    #     # 선택적 정규화 적용
    #     if self.normalize:
    #         out = F.normalize(out, p=2, dim=-1)
        
    #     return out


# class Aggregator(nn.Module):

#     def __init__(self, input_dim=None, output_dim=None, device='cpu'):
#         """
#         Parameters
#         ----------
#         input_dim : int or None.
#             Dimension of input node features. Used for defining fully
#             connected layer in pooling aggregators. Default: None.
#         output_dim : int or None
#             Dimension of output node features. Used for defining fully
#             connected layer in pooling aggregators. Currently only works when
#             input_dim = output_dim. Default: None.
#         """
#         super(Aggregator, self).__init__()

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.device = device

#     def forward(self, features, nodes, mapping, rows, num_samples=25):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             An (n' x input_dim) tensor of input node features.
#         nodes : numpy array
#             nodes is a numpy array of nodes in the current layer of the computation graph.
#         mapping : dict
#             mapping is a dictionary mapping node v (labelled 0 to |V|-1) to
#             its position in the layer of nodes in the computationn graph
#             before nodes. For example, if the layer before nodes is [2,5],
#             then mapping[2] = 0 and mapping[5] = 1.
#         rows : numpy array
#             rows[i] is an array of neighbors of node i which is present in nodes.
#         num_samples : int
#             Number of neighbors to sample while aggregating. Default: 25.

#         Returns
#         -------
#         out : torch.Tensor
#             An (len(nodes) x output_dim) tensor of output node features.
#             Currently only works when output_dim = input_dim.
#         """
#         _choice, _len, _min = np.random.choice, len, min
#         mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
#         if num_samples == -1:
#             sampled_rows = mapped_rows
#         else:
#             sampled_rows = [_choice(row, _min(_len(row), num_samples), _len(row) < num_samples) for row in mapped_rows]

#         n = _len(nodes)
#         if self.__class__.__name__ == 'LSTMAggregator':
#             out = torch.zeros(n, 2*self.output_dim).to(self.device)
#         else:
#             out = torch.zeros(n, self.output_dim).to(self.device)
#         for i in range(n):
#             if _len(sampled_rows[i]) != 0:
#                 out[i, :] = self._aggregate(features[sampled_rows[i], :])

#         return out

#     def _aggregate(self, features):
#         """
#         Parameters
#         ----------

#         Returns
#         -------
#         """
#         raise NotImplementedError

# class MeanAggregator(Aggregator):

#     def _aggregate(self, features):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             Input features.

#         Returns
#         -------
#         Aggregated feature.
#         """
#         return torch.mean(features, dim=0)

# class PoolAggregator(Aggregator):

#     def __init__(self, input_dim, output_dim, device='cpu'):
#         """
#         Parameters
#         ----------
#         input_dim : int
#             Dimension of input node features. Used for defining fully connected layer.
#         output_dim : int
#             Dimension of output node features. Used for defining fully connected layer. Currently only works when output_dim = input_dim.
#         """
#         super(PoolAggregator, self).__init__(input_dim, output_dim, device)

#         self.fc1 = nn.Linear(input_dim, output_dim)
#         self.relu = nn.ReLU()

#     def _aggregate(self, features):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             Input features.

#         Returns
#         -------
#         Aggregated feature.
#         """
#         out = self.relu(self.fc1(features))
#         return self._pool_fn(out)

#     def _pool_fn(self, features):
#         """
#         Parameters
#         ----------

#         Returns
#         -------
#         """
#         raise NotImplementedError

# class MaxPoolAggregator(PoolAggregator):

#     def _pool_fn(self, features):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             Input features.

#         Returns
#         -------
#         Aggregated feature.
#         """
#         return torch.max(features, dim=0)[0]

# class MeanPoolAggregator(PoolAggregator):

#     def _pool_fn(self, features):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             Input features.

#         Returns
#         -------
#         Aggregated feature.
#         """
#         return torch.mean(features, dim=0)

# class LSTMAggregator(Aggregator):

#     def __init__(self, input_dim, output_dim, device='cpu'):
#         """
#         Parameters
#         ----------
#         input_dim : int
#             Dimension of input node features. Used for defining LSTM layer.
#         output_dim : int
#             Dimension of output node features. Used for defining LSTM layer. Currently only works when output_dim = input_dim.

#         """
#         # super(LSTMAggregator, self).__init__(input_dim, output_dim, device)
#         super().__init__(input_dim, output_dim, device)

#         self.lstm = nn.LSTM(input_dim, output_dim, bidirectional=True, batch_first=True)

#     def _aggregate(self, features):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             Input features.

#         Returns
#         -------
#         Aggregated feature.
#         """
#         perm = np.random.permutation(np.arange(features.shape[0]))
#         features = features[perm, :]
#         features = features.unsqueeze(0)

#         out, _ = self.lstm(features)
#         out = out.squeeze(0)
#         out = torch.sum(out, dim=0)

#         return out