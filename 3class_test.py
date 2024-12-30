import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from model import MultiClass, baseline
from train import three_train
from inference import three_test
import wandb
from utils import seed_everything, save_model
import argparse
import pickle
from utils import load_model
import numpy as np
from scipy.stats import sem


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
        # print(idx)
        edge_type = y[idx]
        if edge_type == 1: 
            adj_matrix[src, tgt] = 1
        elif edge_type == 2: 
            adj_matrix[tgt, src] = 1

    return adj_matrix


def get_SHD(true_edge_label, pre_edge_label):
    true_edge_label = np.array(true_edge_label)
    pre_edge_label = np.array(pre_edge_label)
    diff = np.array(true_edge_label) != np.array(pre_edge_label)
    # diff[diff == 2] = 1
    return np.sum(diff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--train_pkl', type=str, default = '/home/jina/reprod/report/test/train_dataset/three_ER_exp_mlp_uni')
    # parser.add_argument('--valid_pkl', type=str, default = None)
    parser.add_argument('--test_pkl', type=str, default = '/home/jina/reprod/new_data/test_dataset/three_ER_exp_mlp_2')
    parser.add_argument('--prediction_pth', type=str, default = '/home/jina/reprod/new_data/models/base/three_ER_exp_mlp_93583.pth')
    parser.add_argument('--num_layers', type=int, default = 3)
    parser.add_argument('--threshold', type=float, default = 0.5)
    parser.add_argument('--seed', type=int, default = 11)
    parser.add_argument('--model', type=str, default = 'baseline')

    parser.add_argument('--device', type=str, default = 'gpu')

    args = parser.parse_args()

    test_name = args.test_pkl.split('/')[-1]

    wandb_name = 'test' + '-' + test_name


    if args.test_pkl:
        with open(args.test_pkl, 'rb') as f:
            test_data = pickle.load(f)


    wandb.init(
        project="four_gnn_cd",
        name = wandb_name,
        config={
            "num_layers": args.num_layers,
            "threshold": args.threshold,
            "seed": args.seed,
            "len(data)": len(test_data),
        }
    )

    # hyperparameter
    num_layers = args.num_layers
    threshold = args.threshold

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

    node_dim = test_data[0].x.size(1)
    edge_dim = test_data[0].edge_attr.size(1)

    # model, optimizer, scheduler, loss
    if args.model == 'multi':
        threeclass_model = MultiClass(node_dim, edge_dim, num_layers=num_layers, output_class=3, device=device, num_samples=None).to(device)
    elif args.model == 'baseline':
        threeclass_model = baseline(11, 100, 3).to(device)
    wandb.config.model = threeclass_model.__class__.__name__

    wandb.watch(threeclass_model, log="all")
    print('****** MultiClass ******')
    threeclass_model = load_model(threeclass_model, model_path=args.prediction_pth, device=device)
    criterion = nn.CrossEntropyLoss()

    wandb.watch(threeclass_model, log="all")

    acc_list = []
    f1_list = []
    shd_list = []
    predictions_list = []
    labels_list = []

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    for data in test_loader:
        test_loss, test_acc, test_f1, test_predictions, test_labels = three_test(threeclass_model, [data], criterion, device, threshold, mode='test')
        labels_list.extend(test_labels)
        predictions_list.extend(test_predictions)

        oracle_adj_mat = get_adj_mat(data.edge_index, test_labels)
        estim_adj_mat = get_adj_mat(data.edge_index, test_predictions)
        precision, recall, f1 = calculate_f1_score(oracle_adj_mat, estim_adj_mat)
        f1_list.append(f1)
        acc_list.append(accuracy_score(test_labels, test_predictions))
        shd = get_SHD(test_labels, test_predictions)
        shd_list.append(shd)

    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)
    se_f1 = sem(f1_list)
    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    se_acc = sem(acc_list)
    mean_shd = np.mean(shd_list)
    std_shd = np.std(shd_list)
    se_shd = sem(shd_list)

    print(f"Test F1: {mean_f1:.3f}, Test Acc: {mean_acc:.3f}, Test SHD: {mean_shd:.3f}")

    wandb.log({
        "ori_acc_mean": mean_acc,
        "ori_acc_std": std_acc,
        "ori_acc_se": se_acc,
        "ori_f1_mean": mean_f1,
        "ori_f1_std": std_f1,
        "ori_f1_se": se_f1,
        "ori_shd_mean": mean_shd,
        "ori_shd_std": std_shd,
        "ori_shd_se": se_shd,
    })



    wandb.finish()