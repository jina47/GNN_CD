import os
import pickle
import random
import numpy as np
import torch
import pandas as pd
import networkx as nx 
from causallearn.search.ConstraintBased.PC import pc
import wandb
import time


def get_skeleton_SHD(true_adj, pre_adj):
    true_adj = np.array(true_adj)
    pre_adj = np.array(pre_adj)
    diff = np.abs(true_adj - pre_adj)
    return np.sum(diff) / 2


def calculate_f1_score(oracle_adj_mat, estim_adj_mat):
    indices = np.triu_indices(n=oracle_adj_mat.shape[0], k=1)  
    true_upper = oracle_adj_mat[indices]
    pred_upper = estim_adj_mat[indices]
    oracle_adj = true_upper.flatten()
    estimated_adj = pred_upper.flatten()
    
    tp = np.sum((oracle_adj == 1) & (estimated_adj == 1))
    fp = np.sum((oracle_adj == 0) & (estimated_adj == 1))
    fn = np.sum((oracle_adj == 1) & (estimated_adj == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score


def pc_test(adj_path, x_path, mode='linear'):
    start_time = time.time()
    true_adj = np.loadtxt(adj_path, delimiter=',')
    x_np = np.loadtxt(x_path, delimiter=',')

    true_skeleton = np.maximum(true_adj, true_adj.T)

    # customized parameters
    # fisherz - linear / kci - nonlinear 
    if mode == 'linear':
        indep_test = 'fisherz'
    else:
        indep_test = 'kci'
    cg = pc(data=x_np, alpha=0.05, indep_test=indep_test)
    cg.to_nx_graph()
    pc_array = nx.to_numpy_array(cg.nx_graph)

    pc_skeleton = np.maximum(pc_array, pc_array.T)
    elapsed_time = time.time() - start_time
    return get_skeleton_SHD(true_skeleton, pc_skeleton), elapsed_time


def pc_result(pc_test, data_folder, node_num, edge_num):
    time = 0
    shd_list = []

    folder_path = os.path.join(data_folder, node_num, 'ER')
    for data_type in os.listdir(folder_path):
        # if data_type != 'train' and data_type != 'mlp':
        if data_type == 'uni':
            dataset_path = os.path.join(folder_path, data_type, 'edge'+ edge_num)
            
            for data_folder in os.listdir(dataset_path):
                print(dataset_path, data_folder)
                for csv_path in os.listdir(os.path.join(dataset_path, data_folder)):
                    if csv_path.startswith('B_true'):
                        adj_path = os.path.join(dataset_path, data_folder, csv_path)
                    elif csv_path.startswith('X_ER'):
                        x_path = os.path.join(dataset_path, data_folder, csv_path)
                if data_type == 'mlp':
                    shd, t = pc_test(adj_path, x_path, mode='nonlinear')
                else:
                    shd, t = pc_test(adj_path, x_path)
                time += t
                shd_list.append(shd)
                print(shd, time)
                return

if __name__ == '__main__':
    data_folder = '/home/jina/reprod/report/data'
    node_num = 'five'
    edge_num = '5'

    pc_result(pc_test, data_folder, node_num, edge_num)



