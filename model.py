import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import CustomSAGEConv


class customGraphSAGE(nn.Module):

    def __init__(self, node_dim, edge_dim, num_layers,
                 agg_class='mean', dropout=0.3, num_samples=3, output_class = 2,
                 device='cpu'):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.agg_class = agg_class
        self.num_samples = num_samples
        self.device = device
        self.num_layers = num_layers

        self.conv = CustomSAGEConv(node_dim, edge_dim, device=self.device)

        # self.mlp = nn.Sequential(nn.Linear(2*node_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_class))
        self.mlp = nn.Sequential(nn.Linear(2 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    # nn.Sigmoid()
                    )
        
        self.bns = nn.BatchNorm1d(node_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_features, num_nodes):

        for k in range(self.num_layers):
            x = self.conv(x, edge_index, edge_features, num_nodes, num_neighbor=self.num_samples)
            x = self.dropout(x)
            x = self.relu(x)
        
        v = edge_index[0, :]
        u = edge_index[1, :]
        v_node_embedding = x[v]  
        u_node_embedding = x[u]  

        # pair_embedding = torch.cat([v_node_embedding, u_node_embedding], dim=-1)
        pair_embedding = torch.cat([v_node_embedding, u_node_embedding, edge_features], dim=-1).float()
        pred = self.mlp(pair_embedding)  

        return pred
    

class customGraphSAGE2(nn.Module):

    def __init__(self, node_dim, edge_dim, num_layers,
                 agg_class='mean', dropout=0.3, num_samples=3, output_class = 2,
                 device='cpu'):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.agg_class = agg_class
        self.num_samples = num_samples
        self.device = device
        self.num_layers = num_layers

        self.conv = CustomSAGEConv(node_dim, edge_dim, device=self.device)

        # self.mlp = nn.Sequential(nn.Linear(2*node_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_class))
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32),  
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(32, 1),  
        )
        
        self.bns = nn.BatchNorm1d(node_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_features, num_nodes):

        for k in range(self.num_layers):
            x = self.conv(x, edge_index, edge_features, num_nodes, num_neighbor=self.num_samples)
            x = self.dropout(x)
            x = self.relu(x)
        
        v = edge_index[0, :]
        u = edge_index[1, :]
        v_node_embedding = x[v]  
        u_node_embedding = x[u]  

        # pair_embedding = torch.cat([v_node_embedding, u_node_embedding], dim=-1)
        pair_embedding = torch.cat([v_node_embedding, u_node_embedding, edge_features], dim=-1).float()
        pred = self.mlp(pair_embedding)  

        return pred
    


