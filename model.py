# torch                         1.8.0
# torch-cluster                 1.5.9
# torch-geometric               1.7.0
# torch-scatter                 2.0.6
# torch-sparse                  0.6.9
# torch-spline-conv             1.2.1


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import CustomSAGEConv


class GraphSAGE(nn.Module):

    def __init__(self, node_dim, edge_dim, num_layers,
                 agg_class='mean', dropout=0.5, num_samples=3, output_class = 2,
                 device='cpu'):
        super(GraphSAGE, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.agg_class = agg_class
        self.num_samples = num_samples
        self.device = device
        self.num_layers = num_layers

        self.conv1 = CustomSAGEConv(node_dim, edge_dim, device=self.device)
        self.conv2 = CustomSAGEConv(node_dim, edge_dim, device=self.device)
        self.conv3 = CustomSAGEConv(node_dim, edge_dim, device=self.device)
        # self.mlp = nn.Sequential(nn.Linear(2*node_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_class))
        self.mlp = nn.Sequential(nn.Linear(2 * node_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    # nn.Sigmoid()
                    )
        
        self.bns = nn.BatchNorm1d(node_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr
        num_nodes = data.num_nodes

        for k in range(self.num_layers):
            x = self.conv1(x, edge_index, edge_features, num_nodes, num_neighbor=self.num_samples)
            # x = self.conv2(x, edge_index, edge_features, num_nodes, num_neighbor=self.num_samples)
            x = self.dropout(x)
            x = self.relu(x)
        
        v = edge_index[0, :]
        u = edge_index[1, :]
        v_node_embedding = x[v]  
        u_node_embedding = x[u]  

        pair_embedding = torch.cat([v_node_embedding, u_node_embedding], dim=-1)
        pred = self.mlp(pair_embedding)  # MLP를 통해 예측 (3 클래스 분류: 0, 1, 2)
        # pred = F.log_softmax(pred, dim=1)
        return pred
    


# class GraphSAGE(nn.Module):

#     def __init__(self, input_dim, hidden_dims, output_dim,
#                  agg_class=MaxPoolAggregator, dropout=0.5, num_samples=25,
#                  device='cpu'):
#         """
#         Parameters
#         ----------
#         input_dim : int
#             Dimension of input node features.
#         hidden_dims : list of ints
#             Dimension of hidden layers. Must be non empty.
#         output_dim : int
#             Dimension of output node features.
#         agg_class : An aggregator class.
#             Aggregator. One of the aggregator classes imported at the top of
#             this module. Default: MaxPoolAggregator.
#         dropout : float
#             Dropout rate. Default: 0.5.
#         num_samples : int
#             Number of neighbors to sample while aggregating. Default: 25.
#         device : str
#             'cpu' or 'cuda:0'. Default: 'cpu'.
#         """
#         super(GraphSAGE, self).__init__()

#         self.input_dim = input_dim
#         self.hidden_dims = hidden_dims
#         self.output_dim = output_dim
#         self.agg_class = agg_class
#         self.num_samples = num_samples
#         self.device = device
#         self.num_layers = len(hidden_dims) + 1

#         self.aggregators = nn.ModuleList([agg_class(input_dim, input_dim, device)])
#         self.aggregators.extend([agg_class(dim, dim, device) for dim in hidden_dims])


#         c = 3 if agg_class == LSTMAggregator else 2
#         self.fcs = nn.ModuleList([nn.Linear(c*input_dim, hidden_dims[0])])
#         self.fcs.extend([nn.Linear(c*hidden_dims[i-1], hidden_dims[i]) for i in range(1, len(hidden_dims))])
#         self.fcs.extend([nn.Linear(c*hidden_dims[-1], output_dim)])

#         self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for hidden_dim in hidden_dims])

#         self.dropout = nn.Dropout(dropout)

#         self.relu = nn.ReLU()

#     def forward(self, data, node_layers, mappings, rows):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             An (n' x input_dim) tensor of input node features.
#         node_layers : list of numpy array
#             node_layers[i] is an array of the nodes in the ith layer of the
#             computation graph.
#         mappings : list of dictionary
#             mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
#             in node_layers[i] to its position in node_layers[i]. For example,
#             if node_layers[i] = [2,5], then mappings[i][2] = 0 and
#             mappings[i][5] = 1.
#         rows : numpy array
#             rows[i] is an array of neighbors of node i.

#         Returns
#         -------
#         out : torch.Tensor
#             An (len(node_layers[-1]) x output_dim) tensor of output node features.
#         """
#         data



#         out = features
#         for k in range(self.num_layers):
#             nodes = node_layers[k+1]
#             mapping = mappings[k]
#             init_mapped_nodes = np.array([mappings[0][v] for v in nodes], dtype=np.int64)
#             cur_rows = rows[init_mapped_nodes]
#             aggregate = self.aggregators[k](out, nodes, mapping, cur_rows,
#                                             self.num_samples)
#             cur_mapped_nodes = np.array([mapping[v] for v in nodes], dtype=np.int64)
#             out = torch.cat((out[cur_mapped_nodes, :], aggregate), dim=1)
#             out = self.fcs[k](out)
#             if k+1 < self.num_layers:
#                 out = self.relu(out)
#                 out = self.bns[k](out)
#                 out = self.dropout(out)
#                 out = out.div(out.norm(dim=1, keepdim=True)+1e-6)

#         return out
    





# from torch_geometric.nn import SAGEConv
# conv = SAGEConv(input_dim, output_dim)

# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
#         super().__init__()
#         self.dropout = dropout
#         self.conv1 = SAGEConv(in_dim, hidden_dim)
#         self.conv2 = SAGEConv(hidden_dim, hidden_dim)
#         self.conv3 = SAGEConv(hidden_dim, out_dim)
    
#     def forward(self, data):
#         x = self.conv1(data.x, data.adj_t)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout)
        
#         x = self.conv2(x, data.adj_t)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout)
        
#         x = self.conv3(x, data.adj_t)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout)
#         return torch.log_softmax(x, dim=-1)