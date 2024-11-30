import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import feature_extract as fe
import features as ft
import os
import torch.multiprocessing as mp
import pickle
import argparse
from tqdm import tqdm


def create_data(x_path, adj_path):
    adj_array = np.loadtxt(adj_path, delimiter=',')
    true_adj = torch.from_numpy(adj_array)
    num_nodes = true_adj.size(0)

    x_pd = pd.read_csv(x_path)
    # node_features = torch.from_numpy(fe.extract_node_features(x_pd))
    # edge_features = torch.from_numpy(fe.extract_edge_features(x_pd))
    node_features = torch.from_numpy(np.nan_to_num(fe.extract_node_features(x_pd), nan=0.0))
    edge_features = torch.from_numpy(np.nan_to_num(fe.extract_edge_features(x_pd), nan=0.0))

    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_exist = torch.nonzero(true_adj, as_tuple=False).t().contiguous()
    edge_label = torch.zeros(edge_index.size(1)) 
    for i, edge in enumerate(edge_index.t()):
        if (edge == edge_exist.t()).all(1).any():
            edge_label[i] = 1
        elif (edge.flip(0) == edge_exist.t()).all(1).any():
            edge_label[i] = 2
    
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=edge_label, num_nodes=num_nodes)
    return data


def create_dataset(dir_path):
    data_list = []
    for folder in tqdm(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.startswith('B_true_'):
                    adj_path = os.path.join(folder_path, file)
                elif file.startswith('X_'):
                    x_path = os.path.join(folder_path, file)
            data = create_data(x_path, adj_path)
            data_list.append(data)

    return data_list
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default = '/home/jina/reprod/report/data/two/ER/mlp/edge1')
    parser.add_argument('--save_dir', type=str, default = '/home/jina/reprod/report/data/pickle')
    parser.add_argument('--pkl_name', type=str)
    
    args = parser.parse_args()

    train_data = create_dataset(args.source)
    pkl_path = os.path.join(args.save_dir, args.pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(train_data, f)


