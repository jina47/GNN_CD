import torch
import numpy as np
import random
import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_model(model, file_name, mode, saved_dir='/home/jina/reprod/models'):
    output_path = os.path.join(saved_dir, mode, file_name)
    print(f"Save model in {output_path}")
    torch.save(model.state_dict(), output_path)


def load_model(model, model_path, device):
    print(f'load checkpoint from {model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def visualize_dag(csv_path):
    adj_matrix = np.loadtxt(csv_path, delimiter=',')
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)  
    pos = nx.spring_layout(G, k=1.5)  
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, arrows=True)
    plt.title("DAG Visualization")
    plt.show()


def visualize_matrix(csv_path):
    adj_matrix = np.loadtxt(csv_path, delimiter=',')
    plt.figure(figsize=(5, 5))
    sns.heatmap(adj_matrix, annot=True, cmap='Blues', cbar=False)
    plt.gca().xaxis.set_label_position('top') 
    plt.gca().xaxis.tick_top()  
    plt.title('Adjacency Matrix Visualization')
    plt.show()


def aggregate_data(data_files, pkl_name, save_dir='/home/jina/reprod/report/data/pickle'):
    data = []
    for file_path in data_files:
        with open(file_path, 'rb') as f:
            temp = pickle.load(f)
            data += temp

    pkl_path = os.path.join(save_dir, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
