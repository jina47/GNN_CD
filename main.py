import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from model import customGraphSAGE
from sklearn.metrics import accuracy_score, confusion_matrix
from train import *
from inference import *
import pickle
import argparse
import wandb
from utils import seed_everything, load_model
import warnings


def calculate_f1_score(oracle_adj_mat, estim_adj_mat):
    oracle_adj = oracle_adj_mat.flatten()
    estimated_adj = estim_adj_mat.flatten()
    
    tp = np.sum((oracle_adj == 1) & (estimated_adj == 1))
    fp = np.sum((oracle_adj == 0) & (estimated_adj == 1))
    fn = np.sum((oracle_adj == 1) & (estimated_adj == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score


def get_adj_mat(edge_index, y, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for idx, (src, tgt) in enumerate(edge_index.T):
        edge_type = y[idx]
        if edge_type == 1: 
            adj_matrix[src, tgt] = 1
        elif edge_type == 2: 
            adj_matrix[tgt, src] = 1

    return adj_matrix


def get_SHD(oracle_adj_mat, estim_adj_mat):
    diff = np.abs(oracle_adj_mat - estim_adj_mat)
    diff = diff + diff.transpose()
    diff[diff > 1] = 1
    return np.sum(diff) / 2


if __name__ == '__main__':
    warnings.filterwarnings("ignore") 
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default = 'total')
    parser.add_argument('--test_pkl', type=str, default = '/home/jina/reprod/data/pickle/triple_test')
    # parser.add_argument('--test_pkl', type=str, default = '/home/jina/reprod/report/data/pickle/three_ER_mlp_edge2')
    parser.add_argument('--prediction_pth', type=str, default = '/home/jina/reprod/models/prediction/triple3_94717.pth')
    parser.add_argument('--orientation_pth', type=str, default = '/home/jina/reprod/models/orientation/triple3_96985.pth')

    parser.add_argument('--num_layers', type=int, default = 3)
    parser.add_argument('--threshold', type=float, default = 0.5)
    parser.add_argument('--seed', type=int, default = 11)

    parser.add_argument('--device', type=str, default = 'gpu')

    args = parser.parse_args()

    test_name = args.test_pkl.split('/')[-1]

    wandb_name = args.mode + '-' + test_name

    # train, validation, test data
    with open(args.test_pkl, 'rb') as f:
        test_data = pickle.load(f)

    wandb.init(
        project="gnn_cd",
        name = wandb_name,
        config={
            "num_layers": args.num_layers,
            "threshold": args.threshold,
            "seed": args.seed,
            "len(data)": len(test_data),
        }
    )

    num_layers = args.num_layers
    threshold = args.threshold
    mode = args.mode

    seed_everything(args.seed)

    if args.device == 'cpu':
        device = args.device
    elif args.device == '1':
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    elif args.device == '2':
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    elif args.device == '3':
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    node_dim = test_data[0].x.size(1)
    edge_dim = test_data[0].edge_attr.size(1)
    

    if args.mode != 'orientation':
        print('****** Skeleton ******')
        skeleton_model = customGraphSAGE(node_dim, edge_dim, num_layers=num_layers, output_class=2, device=device, num_samples=None).to(device)
        skeleton_model = load_model(skeleton_model, model_path=args.prediction_pth, device=device)
        criterion = nn.BCEWithLogitsLoss()

        wandb.watch(skeleton_model, log="all")
        
        pre_acc_list = []
        pre_f1_list = []
        pre_shd_list = []
        new_test_data_list = []
        ske_labels_list = []
        ske_predictions_list = []

        for data in test_loader:
            skeleton_loss, skeleton_acc, skeleton_f1, skeleton_predictions, skeleton_labels, new_test_data = predict_test(skeleton_model, [data], criterion, device, threshold, mode='test')
            new_test_data_list += new_test_data
            ske_labels_list.extend(skeleton_labels)
            ske_predictions_list.extend(skeleton_predictions)
            
            pre_acc_list.append(skeleton_acc)
            pre_f1_list.append(skeleton_f1)
            pre_shd_list.append(get_SHD(np.array(skeleton_labels), np.array(skeleton_predictions)))
        
        mean_f1 = np.mean(pre_f1_list)
        std_f1 = np.std(pre_f1_list)
        mean_acc = np.mean(pre_acc_list)
        std_acc = np.std(pre_acc_list)
        mean_shd = np.mean(pre_shd_list)
        std_shd = np.std(pre_shd_list)

        print("\nSkeleton Test complete!")
        print(f"Test F1: {mean_f1:.3f}, Test Acc: {mean_acc:.3f}, Test SHD: {mean_shd:.3f}")

        pre_tn, pre_fp, pre_fn, pre_tp = confusion_matrix(ske_labels_list, ske_predictions_list).ravel().tolist()
        print(f"Test tn: {pre_tn} fp: {pre_fp} fn: {pre_fn} tp: {pre_tp}")

        wandb.log({
            "skeleton_acc_mean": mean_acc,
            "skeleton_acc_std": std_acc,
            "skeleton_f1_mean": mean_f1,
            "skeleton_f1_std": std_f1,
            "skeleton_shd_mean": mean_shd,
            "skeleton_shd_std": std_shd,
            "skeleton_tn": pre_tn,
            "skeleton_fp": pre_fp,
            "skeleton_fn": pre_fn,
            "skeleton_tp": pre_tp,
        })
        
    
    if args.mode != 'prediction':
        print('\n****** Orientation ******')
        orientation_model = customGraphSAGE(node_dim, edge_dim, num_layers=num_layers, output_class=2, device=device, num_samples=None).to(device)
        orientation_model = load_model(orientation_model, model_path=args.orientation_pth, device=device)
        criterion = nn.BCEWithLogitsLoss()
        
        wandb.watch(orientation_model, log="all")

        ori_acc_list = []
        ori_f1_list = []
        ori_shd_list = []
        ori_predictions_list = []
        ori_labels_list = []
        
        if args.mode == 'total':
            test_loader = DataLoader(new_test_data_list, batch_size=1, shuffle=False)
        elif args.mode == 'orientation': 
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


        for data in test_loader:
            ori_loss, ori_acc, ori_f1, ori_predictions, ori_labels, test_predictions, test_labels = orient_test(orientation_model, [data], criterion, device, threshold, mode=args.mode)
            ori_labels_list.extend(ori_labels)
            ori_predictions_list.extend(ori_predictions)

            oracle_adj_mat = get_adj_mat(data.edge_index, test_labels)
            estim_adj_mat = get_adj_mat(data.edge_index, test_predictions)
            
            ori_shd = get_SHD(oracle_adj_mat, estim_adj_mat)
            ori_shd_list.append(ori_shd)

            if args.mode == 'total':
                precision, recall, f1 = calculate_f1_score(oracle_adj_mat, estim_adj_mat)
                ori_f1_list.append(f1)
                ori_acc_list.append(accuracy_score(test_labels, test_predictions))
            elif args.mode == 'orientation':
                if ~np.isnan(ori_acc):
                    ori_acc_list.append(ori_acc)
                    ori_f1_list.append(ori_f1)
        
        mean_f1 = np.mean(ori_f1_list)
        std_f1 = np.std(ori_f1_list)
        mean_acc = np.mean(ori_acc_list)
        std_acc = np.std(ori_acc_list)
        mean_shd = np.mean(ori_shd_list)
        std_shd = np.std(ori_shd_list)

        print("\nOrientation Test complete!")
        print(f"Test F1: {mean_f1:.3f}, Test Acc: {mean_acc:.3f}, Test SHD: {mean_shd:.3f}")

        ori_tn, ori_fp, ori_fn, ori_tp = confusion_matrix(ori_labels_list, ori_predictions_list).ravel().tolist()
        print(f"Test tn: {ori_tn} fp: {ori_fp} fn: {ori_fn} tp: {ori_tp}")

        wandb.log({
            "ori_acc_mean": mean_acc,
            "ori_acc_std": std_acc,
            "ori_f1_mean": mean_f1,
            "ori_f1_std": std_f1,
            "ori_shd_mean": mean_shd,
            "ori_shd_std": std_shd,
            "ori_tn": ori_tn,
            "ori_fp": ori_fp,
            "ori_fn": ori_fn,
            "ori_tp": ori_tp,
        })
        

    wandb.finish()
