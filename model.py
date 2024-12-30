import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import CustomSAGEConv
from torch_geometric.nn import SAGEConv, GATConv, GCNConv


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
    

class onlyMLP(nn.Module):

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
        
        v = edge_index[0, :]
        u = edge_index[1, :]
        v_node_embedding = x[v]  
        u_node_embedding = x[u]  

        # pair_embedding = torch.cat([v_node_embedding, u_node_embedding], dim=-1)
        pair_embedding = torch.cat([v_node_embedding, u_node_embedding, edge_features], dim=-1).float()
        pred = self.mlp(pair_embedding)  

        return pred


class MultiClass(nn.Module):

    def __init__(self, node_dim, edge_dim, num_layers,
                 agg_class='mean', dropout=0.5, num_samples=3, output_class = 3,
                 device='cpu'):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.agg_class = agg_class
        self.num_samples = num_samples
        self.device = device
        self.num_layers = num_layers

        self.conv = CustomSAGEConv(node_dim, edge_dim, device=self.device)

        # self.mlp = nn.Sequential(nn.Linear(2*node_dim+edge_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_class))
        self.mlp = nn.Sequential(nn.Linear(2 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_class),  
                    # nn.Sigmoid()
                    )
        
        self.bns = nn.BatchNorm1d(node_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_features, num_nodes):
    
        v = edge_index[0, :]
        u = edge_index[1, :]
        v_node_embedding = x[v]  
        u_node_embedding = x[u]  


        pair_embedding = torch.cat([v_node_embedding, u_node_embedding, edge_features], dim=-1).float()
        pred = self.mlp(pair_embedding)  

        return pred


class customGraphSAGE3(nn.Module):
    def __init__(self, node_dim, edge_dim, num_nodes, device='cpu'):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.device = device

        self.conv = CustomSAGEConv(node_dim, edge_dim, device=self.device)

        self.mlp1 = nn.Sequential(
                        nn.Linear(2 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )

        self.mlp2 = nn.Sequential(
                        nn.Linear(3 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )

        # self.lin = nn.Linear(num_nodes-1, 1)

    def forward(self, x, edge_index, edge_features, num_nodes):

        x = self.conv(x, edge_index, edge_features, num_nodes)

        v = edge_index[0, :]
        u = edge_index[1, :]

        v_embedding = x[v]
        u_embedding = x[u]

        # Pair embedding without neighbors
        pair_embedding_no_neighbor = torch.cat([v_embedding, u_embedding, edge_features], dim=-1).float()
        pred_no_neighbor = self.mlp1(pair_embedding_no_neighbor)  # MLP1 for no-neighbor case
        
        # Pair embedding with neighbors
        pairs = torch.tensor([(v[i].item(), u[i].item()) for i in range(len(v))], dtype=torch.long)
        preds_with_neighbors = []  

        for i, pair in enumerate(pairs):
            z_indices = torch.tensor([z for z in range(num_nodes) if z not in pair], dtype=torch.long)
            pair_embedding = x[pair, :]
            z_embeddings = x[z_indices, :]
            pair_embedding_expanded = pair_embedding.unsqueeze(0).repeat(z_embeddings.size(0), 1, 1)
            z_embeddings_expanded = z_embeddings.unsqueeze(1)
            result = torch.cat([pair_embedding_expanded, z_embeddings_expanded], dim=1)
            result_flattened = result.view(result.size(0), -1)
            edge_f = edge_features[i].unsqueeze(0).repeat(result_flattened.size(0), 1)
            final_result = torch.cat([result_flattened, edge_f], dim=1).float()
            final_output = self.mlp2(final_result).view(1, -1)
            preds_with_neighbors.append(final_output)

        all_outputs_tensor = torch.cat(preds_with_neighbors, dim=0)
        final_preds = torch.cat((pred_no_neighbor, all_outputs_tensor), dim=1)

        # pred = self.lin(final_preds)
        pred = torch.mean(final_preds, dim=1).view(-1, 1)

        return pred


# CIT-GNN
class customGraphSAGE4(nn.Module):
    def __init__(self, node_dim, edge_dim, num_nodes, device='cpu'):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.device = device

        self.conv = CustomSAGEConv(node_dim, edge_dim, device=self.device)

        self.mlp1 = nn.Sequential(
                        nn.Linear(2 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )

        self.mlp2 = nn.Sequential(
                        nn.Linear(3 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )

        self.lin = nn.Linear(num_nodes-1, 1)

    def forward(self, x, edge_index, edge_features, num_nodes_batch, num_nodes):
        x = self.conv(x, edge_index, edge_features, num_nodes_batch)

        v = edge_index[0, :]
        u = edge_index[1, :]

        v_embedding = x[v]
        u_embedding = x[u]

        pair_embedding_no_neighbor = torch.cat([v_embedding, u_embedding, edge_features], dim=-1).float()
        pred_no_neighbor = self.mlp1(pair_embedding_no_neighbor)  # MLP1 for no-neighbor case

        pairs = torch.tensor([(v[i].item(), u[i].item()) for i in range(len(v))], dtype=torch.long)
        preds_with_neighbors = []

        for i, pair in enumerate(pairs):
            graph_id = pair[0] // num_nodes

            node_start = graph_id * num_nodes  
            node_end = (graph_id+1) * num_nodes   
            valid_nodes = torch.arange(node_start, node_end)

            z_indices = valid_nodes[~torch.isin(valid_nodes, pair)]

            pair_embedding = x[pair, :]
            z_embeddings = x[z_indices, :]
            pair_embedding_expanded = pair_embedding.unsqueeze(0).repeat(z_embeddings.size(0), 1, 1)
            z_embeddings_expanded = z_embeddings.unsqueeze(1)
            result = torch.cat([pair_embedding_expanded, z_embeddings_expanded], dim=1)
            result_flattened = result.view(result.size(0), -1)
            edge_f = edge_features[i].unsqueeze(0).repeat(result_flattened.size(0), 1)
            final_result = torch.cat([result_flattened, edge_f], dim=1).float()
            final_output = self.mlp2(final_result).view(1, -1)
            preds_with_neighbors.append(final_output)

        all_outputs_tensor = torch.cat(preds_with_neighbors, dim=0)
        final_preds = torch.cat((pred_no_neighbor, all_outputs_tensor), dim=1)

        # pred = torch.mean(final_preds, dim=1).view(-1, 1)
        pred = self.lin(final_preds)
        return pred


class baseline(nn.Module):

    def __init__(self, node_dim, edge_dim, num_layers,
                 agg_class='mean', dropout=0.5, num_samples=3, output_class = 3,
                 device='cpu'):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.agg_class = agg_class
        self.num_samples = num_samples
        self.device = device
        self.num_layers = num_layers

        self.conv = SAGEConv(node_dim, node_dim)

        # self.mlp = nn.Sequential(nn.Linear(2*node_dim+edge_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_class))
        self.mlp = nn.Sequential(
                        nn.Linear(2 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_class),  
                    )
        self.bns = nn.BatchNorm1d(node_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_features, num_nodes):

        for k in range(self.num_layers):
            x = self.conv(x, edge_index)
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
    

class splitbase(nn.Module):

    def __init__(self, node_dim, edge_dim, num_layers,
                 agg_class='mean', dropout=0.5, num_samples=3, output_class = 1,
                 device='cpu'):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.agg_class = agg_class
        self.num_samples = num_samples
        self.device = device
        self.num_layers = num_layers

        self.conv = SAGEConv(node_dim, node_dim)

        # self.mlp = nn.Sequential(nn.Linear(2*node_dim+edge_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_class))
        self.mlp = nn.Sequential(
                        nn.Linear(2 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )
        self.bns = nn.BatchNorm1d(node_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_features, num_nodes):

        for k in range(self.num_layers):
            x = self.conv(x, edge_index)
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


class customGraphSAGE5(nn.Module):
    def __init__(self, node_dim, edge_dim, num_nodes, device='cpu'):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.device = device

        # self.conv = CustomSAGEConv(node_dim, edge_dim, device=self.device)
        self.conv = SAGEConv(node_dim, node_dim)
        
        self.mlp1 = nn.Sequential(
                        nn.Linear(2 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )

        self.mlp2 = nn.Sequential(
                        nn.Linear(3 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )

        self.lin = nn.Linear(num_nodes-1, 1)

        self.bns = nn.BatchNorm1d(node_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_features, num_nodes_batch, num_nodes):

        # x = self.conv(x, edge_index, edge_features, num_nodes_batch)
        for k in range(3):
            x = self.conv(x, edge_index)
            x = self.dropout(x)
            x = self.relu(x)

        v = edge_index[0, :]
        u = edge_index[1, :]

        v_embedding = x[v]
        u_embedding = x[u]

        pair_embedding_no_neighbor = torch.cat([v_embedding, u_embedding, edge_features], dim=-1).float()
        pred_no_neighbor = self.mlp1(pair_embedding_no_neighbor) 

        pairs = torch.tensor([(v[i].item(), u[i].item()) for i in range(len(v))], dtype=torch.long)
        preds_with_neighbors = []

        for i, pair in enumerate(pairs):
            graph_id = pair[0] // num_nodes

            node_start = graph_id * num_nodes  
            node_end = (graph_id+1) * num_nodes  
            valid_nodes = torch.arange(node_start, node_end)

            z_indices = valid_nodes[~torch.isin(valid_nodes, pair)]

            pair_embedding = x[pair, :]
            z_embeddings = x[z_indices, :]
            pair_embedding_expanded = pair_embedding.unsqueeze(0).repeat(z_embeddings.size(0), 1, 1)
            z_embeddings_expanded = z_embeddings.unsqueeze(1)
            result = torch.cat([pair_embedding_expanded, z_embeddings_expanded], dim=1)
            result_flattened = result.view(result.size(0), -1)
            edge_f = edge_features[i].unsqueeze(0).repeat(result_flattened.size(0), 1)
            final_result = torch.cat([result_flattened, edge_f], dim=1).float()
            final_output = self.mlp2(final_result).view(1, -1)
            preds_with_neighbors.append(final_output)

        all_outputs_tensor = torch.cat(preds_with_neighbors, dim=0)
        final_preds = torch.cat((pred_no_neighbor, all_outputs_tensor), dim=1)

        # pred = torch.mean(final_preds, dim=1).view(-1, 1)
        pred = self.lin(final_preds)
        return pred


class customGAT(nn.Module):
    def __init__(self, node_dim, edge_dim, num_nodes, device='cpu'):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.device = device

        self.conv = GATConv(node_dim, node_dim, heads=2, concat=True)
        self.fc = nn.Linear(node_dim * 2, node_dim)
        
        self.mlp1 = nn.Sequential(
                        nn.Linear(2 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )

        self.mlp2 = nn.Sequential(
                        nn.Linear(3 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )

        # self.lin = nn.Linear(num_nodes-1, 1)
        self.lin = nn.Linear(2, 1)

        self.bns = nn.BatchNorm1d(node_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_features, num_nodes_batch, num_nodes):
        for k in range(3):
            x = self.conv(x, edge_index)
            x = self.fc(x)
            x = self.dropout(x)
            x = self.relu(x)


        v = edge_index[0, :]
        u = edge_index[1, :]

        v_embedding = x[v]
        u_embedding = x[u]


        pair_embedding_no_neighbor = torch.cat([v_embedding, u_embedding, edge_features], dim=-1).float()
        pred_no_neighbor = self.mlp1(pair_embedding_no_neighbor) 

        pairs = torch.tensor([(v[i].item(), u[i].item()) for i in range(len(v))], dtype=torch.long)
        preds_with_neighbors = []

        for i, pair in enumerate(pairs):
            graph_id = pair[0] // num_nodes

            node_start = graph_id * num_nodes  
            node_end = (graph_id+1) * num_nodes 
            valid_nodes = torch.arange(node_start, node_end)

            z_indices = valid_nodes[~torch.isin(valid_nodes, pair)]

            pair_embedding = x[pair, :]
            z_embeddings = x[z_indices, :]
            pair_embedding_expanded = pair_embedding.unsqueeze(0).repeat(z_embeddings.size(0), 1, 1)
            z_embeddings_expanded = z_embeddings.unsqueeze(1)
            result = torch.cat([pair_embedding_expanded, z_embeddings_expanded], dim=1)
            result_flattened = result.view(result.size(0), -1)
            edge_f = edge_features[i].unsqueeze(0).repeat(result_flattened.size(0), 1)
            final_result = torch.cat([result_flattened, edge_f], dim=1).float()
            final_output = self.mlp2(final_result).view(1, -1)
            preds_with_neighbors.append(final_output)

        # all_outputs_tensor = torch.cat(preds_with_neighbors, dim=0)
        all_outputs_tensor = torch.stack(preds_with_neighbors).mean(dim=1)
        final_preds = torch.cat((pred_no_neighbor, all_outputs_tensor), dim=1)

        # pred = torch.mean(final_preds, dim=1).view(-1, 1)
        pred = self.lin(final_preds)
        return pred
    

class customGCN(nn.Module):
    def __init__(self, node_dim, edge_dim, num_nodes, device='cpu'):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.device = device

        self.conv = GCNConv(node_dim, node_dim)

        self.mlp1 = nn.Sequential(
                        nn.Linear(2 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )

        self.mlp2 = nn.Sequential(
                        nn.Linear(3 * node_dim + edge_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),  
                    )

        self.lin = nn.Linear(num_nodes-1, 1)

        self.bns = nn.BatchNorm1d(node_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_features, num_nodes_batch, num_nodes):
        for k in range(3):
            x = self.conv(x, edge_index)
            x = self.dropout(x)
            x = self.relu(x)

        v = edge_index[0, :]
        u = edge_index[1, :]

        v_embedding = x[v]
        u_embedding = x[u]

        pair_embedding_no_neighbor = torch.cat([v_embedding, u_embedding, edge_features], dim=-1).float()
        pred_no_neighbor = self.mlp1(pair_embedding_no_neighbor)  

        pairs = torch.tensor([(v[i].item(), u[i].item()) for i in range(len(v))], dtype=torch.long)
        preds_with_neighbors = []

        for i, pair in enumerate(pairs):
            graph_id = pair[0] // num_nodes

            node_start = graph_id * num_nodes 
            node_end = (graph_id+1) * num_nodes  
            valid_nodes = torch.arange(node_start, node_end)

            z_indices = valid_nodes[~torch.isin(valid_nodes, pair)]

            pair_embedding = x[pair, :]
            z_embeddings = x[z_indices, :]
            pair_embedding_expanded = pair_embedding.unsqueeze(0).repeat(z_embeddings.size(0), 1, 1)
            z_embeddings_expanded = z_embeddings.unsqueeze(1)
            result = torch.cat([pair_embedding_expanded, z_embeddings_expanded], dim=1)
            result_flattened = result.view(result.size(0), -1)
            edge_f = edge_features[i].unsqueeze(0).repeat(result_flattened.size(0), 1)
            final_result = torch.cat([result_flattened, edge_f], dim=1).float()
            final_output = self.mlp2(final_result).view(1, -1)
            preds_with_neighbors.append(final_output)

        all_outputs_tensor = torch.cat(preds_with_neighbors, dim=0)
        final_preds = torch.cat((pred_no_neighbor, all_outputs_tensor), dim=1)

        # pred = torch.mean(final_preds, dim=1).view(-1, 1)
        pred = self.lin(final_preds)
        return pred
