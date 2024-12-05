import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
from model import MultiClass
from train import three_train
from inference import three_test
import wandb
from utils import seed_everything, save_model
import argparse
import pickle
import random
import numpy as np


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

    parser.add_argument('--train_pkl', type=str, default = '/home/jina/reprod/report/test/train_dataset/two_ER_exp_mlp_uni')
    parser.add_argument('--valid_pkl', type=str, default = None)
    parser.add_argument('--test_pkl', type=str, default = '/home/jina/reprod/report/test/test_dataset/two_ER_exp_mlp_uni_01')
    # parser.add_argument('--test_pkl', type=str, default = None)

    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--epochs', type=int, default = 50)
    parser.add_argument('--num_layers', type=int, default = 3)
    parser.add_argument('--lr', type=float, default = 0.005)
    parser.add_argument('--threshold', type=float, default = 0.5)
    parser.add_argument('--seed', type=int, default = 11)

    parser.add_argument('--device', type=str, default = 'gpu')

    args = parser.parse_args()

    train_name = args.train_pkl.split('/')[-1]
    if args.test_pkl:
        test_name = args.test_pkl.split('/')[-1]
    else:
        test_name = '' 
    wandb_name = train_name + '&' + test_name

    # train, validation, test data
    with open(args.train_pkl, 'rb') as f:
        data = pickle.load(f)

    if args.valid_pkl:
        with open(args.valid_pkl, 'rb') as f:
            valid_data = pickle.load(f)
            train_data = data
    else:
        train_idx = random.sample(range(len(data)), int(len(data)*0.8))
        valid_idx = [i for i in range(len(data)) if i not in train_idx]
        train_data = [data[i] for i in train_idx]
        valid_data = [data[i] for i in valid_idx]

    if args.test_pkl:
        with open(args.test_pkl, 'rb') as f:
            test_data = pickle.load(f)


    wandb.init(
        project="3class_baseline",
        name = wandb_name,
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "num_layers": args.num_layers,
            "learning_rate": args.lr,
            "threshold": args.threshold,
            "seed": args.seed,
            "len(data)": len(data),
        }
    )

    # hyperparameter
    batch_size = args.batch_size
    epochs = args.epochs
    num_layers = args.num_layers
    lr = args.lr
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

    # dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    node_dim = train_data[0].x.size(1)
    edge_dim = train_data[0].edge_attr.size(1)

    # model, optimizer, scheduler, loss
    threeclass_model = MultiClass(node_dim, edge_dim, num_layers=num_layers, output_class=3, device=device, num_samples=None).to(device)

    optimizer = torch.optim.Adam(threeclass_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    wandb.config.model = threeclass_model.__class__.__name__

    best_valid_f1 = 0
    best_valid_acc = 0
    best_valid_loss = 9999

    wandb.watch(threeclass_model, log="all")
    print('****** MultiClass ******')
    for epoch in range(epochs):

        print(f'Epoch: {epoch + 1}')
        train_loss, train_acc, train_f1 = three_train(threeclass_model, train_loader, optimizer, criterion, device, threshold, mode='3class')
        print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
        valid_loss, valid_acc, valid_f1, valid_predictions, valid_labels = three_test(threeclass_model, valid_loader, criterion, device, threshold, mode='3class')
        print(f'Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}, Valid F1: {valid_f1:.3f}')

        scheduler.step(valid_loss)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "valid_f1": valid_f1,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        if best_valid_f1 < valid_f1:
            best_valid_f1 = valid_f1

        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            best_model = threeclass_model
            best_model = threeclass_model
            best_valid_labels = valid_labels
            best_valid_predictions = valid_predictions

        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss

    print("\nMultiClass Training complete!")
    print(f"Best Valid F1: {best_valid_f1:.3f}, Best Valid Acc: {best_valid_acc:.3f}, Best Valid Loss: {best_valid_loss:.3f}")

    print(best_valid_labels)
    print(best_valid_predictions)
    # tn, fp, fn, tp = confusion_matrix(best_valid_labels, best_valid_predictions).ravel().tolist()
    # print(f"MultiClass Valid tn: {tn} fp: {fp} fn: {fn} tp: {tp}")

    acc_ = f'{best_valid_acc:.5f}'[2:]
    file_name = f'{wandb_name.split("&")[0]}_{acc_}.pth'
    save_model(best_model, mode='', file_name=file_name, saved_dir='/home/jina/reprod/baseline/3class/models')

    wandb.log({
        "best_valid_accuracy": best_valid_acc,
        "best_valid_f1": best_valid_f1,
        # "final_valid_tn": tn,
        # "final_valid_fp": fp,
        # "final_valid_fn": fn,
        # "final_valid_tp": tp,
    })


    # if args.test_pkl:
    #     # test_model = customGraphSAGE(node_dim, edge_dim, num_layers=num_layers, output_class=2, device=device, num_samples=None).to(device)
    #     # test_model = load_model(test_model, model_path=args.pred_model, device=args.device)
    #     acc_list = []
    #     f1_list = []
    #     shd_list = []
    #     predictions_list = []
    #     labels_list = []

    #     best_model.to(device)

    #     test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    #     for data in test_loader:
    #         test_loss, test_acc, test_f1, test_predictions, test_labels = three_test(best_model, [data], criterion, device, threshold, mode='test')
        
    #         shd = get_SHD(test_labels, test_predictions)
    #         shd_list.append(shd)

    #     print("\nSkeleton Test complete!")
    #     print(f"Test F1: {test_f1:.3f}, Test Acc: {test_acc:.3f}, Test Loss: {test_loss:.3f}")

    #     tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel().tolist()
    #     print(f"Skeleton Test tn: {tn} fp: {fp} fn: {fn} tp: {tp}")

    #     wandb.log({
    #     "test_loss": test_loss,
    #     "test_accuracy": test_acc,
    #     "test_f1": test_f1,
    #     "test_tn": tn,
    #     "test_fp": fp,
    #     "test_fn": fn,
    #     "test_tp": tp,
    # })


    wandb.finish()