import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
from model import *
from train import predict_train
from inference import predict_test
import wandb
from utils import seed_everything, save_model,load_model
import argparse
import pickle
import random


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_pkl', type=str, default = '/home/jina/reprod/new_data/train_dataset/six_ER_exp_mlp')
    parser.add_argument('--valid_pkl', type=str, default = '/home/jina/reprod/new_data/valid_dataset/six_ER_exp_mlp')
    parser.add_argument('--test_pkl', type=str, default = '/home/jina/reprod/new_data/test_dataset/six_ER_exp_mlp_7')
    parser.add_argument('--model', type=str, default = 'split')

    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--epochs', type=int, default = 100)
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
        test_data = test_data[50:]

    wandb.init(
        project="final_edge_prediction",
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
    num_node = train_data[0].num_nodes

    # model, optimizer, scheduler, loss
    if args.model == 'model1':
        skeleton_model = customGraphSAGE(node_dim, edge_dim, num_layers=num_layers, output_class=2, device=device, num_samples=None).to(device)
    elif args.model == 'model4':
        skeleton_model = customGraphSAGE4(node_dim=node_dim, edge_dim=edge_dim, num_nodes=num_node, device=device).to(device)
    elif args.model == 'model5':
        skeleton_model = customGraphSAGE5(node_dim=node_dim, edge_dim=edge_dim, num_nodes=num_node, device=device).to(device)
    elif args.model == 'gat':
        skeleton_model = customGAT(node_dim=node_dim, edge_dim=edge_dim, num_nodes=num_node, device=device).to(device)
    elif args.model == 'gcn':
        skeleton_model = customGCN(node_dim=node_dim, edge_dim=edge_dim, num_nodes=num_node, device=device).to(device)
    elif args.model == 'mlp':
        skeleton_model = onlyMLP(node_dim, edge_dim, num_layers=num_layers, output_class=2, device=device, num_samples=None).to(device)
    elif args.model == 'split':
        skeleton_model = splitbase(node_dim, edge_dim, num_layers=num_layers, output_class=2, device=device, num_samples=None).to(device)
    
    # skeleton_model.apply(initialize_weights)
    optimizer = torch.optim.Adam(skeleton_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    wandb.config.model = skeleton_model.__class__.__name__
    if skeleton_model.__class__.__name__ not in ['customGraphSAGE4', 'customGraphSAGE5', 'customGAT', 'customGCN']:
        num_node = 0

    best_valid_f1 = 0
    best_valid_acc = 0
    best_valid_loss = 9999
    best_valid_threshold = 0.5

    wandb.watch(skeleton_model, log="all")
    print('****** Skeleton ******')
    for epoch in range(epochs):

        print(f'Epoch: {epoch + 1}')
        train_loss, train_acc, train_f1 = predict_train(skeleton_model, train_loader, optimizer, criterion, device, best_valid_threshold, num_node=num_node)
        print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
        valid_loss, valid_acc, valid_f1, valid_auroc, valid_threshold, valid_predictions, valid_labels = predict_test(skeleton_model, valid_loader, criterion, device, best_valid_threshold, num_node=num_node)
        print(f'Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}, Valid F1: {valid_f1:.3f}, Valid AUROC: {valid_auroc:.3f}, Valid thres:{valid_threshold:.2f}')

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
            "valid_roc": valid_auroc,
            "valid_threshold": valid_threshold,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        if best_valid_f1 < valid_f1:
            best_valid_f1 = valid_f1

        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            best_model = skeleton_model
            best_valid_threshold = valid_threshold
            best_valid_labels = valid_labels
            best_valid_predictions = valid_predictions

        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss

    print("\nSkeleton Training complete!")
    print(f"Best Valid F1: {best_valid_f1:.3f}, Best Valid Acc: {best_valid_acc:.3f}, Best Valid Loss: {best_valid_loss:.3f}")

    tn, fp, fn, tp = confusion_matrix(best_valid_labels, best_valid_predictions).ravel().tolist()
    print(f"Skeleton Valid tn: {tn} fp: {fp} fn: {fn} tp: {tp}")

    acc_ = f'{best_valid_acc:.5f}'[2:]
    file_name = f'{wandb_name.split("&")[0]}_{acc_}.pth'
    if args.model == 'mlp':
        save_model(best_model, mode='prediction', file_name=file_name, saved_dir='/home/jina/reprod/new_data/threshold/mlp')
    elif args.model == 'model1':
        save_model(best_model, mode='prediction', file_name=file_name, saved_dir='/home/jina/reprod/new_data/threshold/model1')
    elif args.model == 'model4':    
        save_model(best_model, mode='prediction', file_name=file_name, saved_dir='/home/jina/reprod/new_data/threshold/model4')
    elif args.model == 'model5':    
        save_model(best_model, mode='prediction', file_name=file_name, saved_dir='/home/jina/reprod/new_data/threshold/model5')
    elif args.model == 'gat':    
        save_model(best_model, mode='prediction', file_name=file_name, saved_dir='/home/jina/reprod/new_data/threshold/gat')
    elif args.model == 'gcn':    
        save_model(best_model, mode='prediction', file_name=file_name, saved_dir='/home/jina/reprod/new_data/threshold/gcn')
    elif args.model == 'split':    
        save_model(best_model, mode='prediction', file_name=file_name, saved_dir='/home/jina/reprod/new_data/models/split')
    
    wandb.log({
        "best_valid_accuracy": best_valid_acc,
        "best_valid_f1": best_valid_f1,
        "best_valid_threshold": best_valid_threshold,
        "final_valid_tn": tn,
        "final_valid_fp": fp,
        "final_valid_fn": fn,
        "final_valid_tp": tp,
    })


    if args.test_pkl:
        best_model.to(device)

        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        test_loss, test_acc, test_f1, test_probs, test_predictions, test_labels, new_test_data = predict_test(best_model, test_loader, criterion, device, best_valid_threshold, num_node=num_node, mode='test')
        print("\nSkeleton Test complete!")
        print(f"Test F1: {test_f1:.3f}, Test Acc: {test_acc:.3f}, Test Loss: {test_loss:.3f}")

        tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel().tolist()
        print(f"Skeleton Test tn: {tn} fp: {fp} fn: {fn} tp: {tp}")

        wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_tn": tn,
        "test_fp": fp,
        "test_fn": fn,
        "test_tp": tp,
    })


    wandb.finish()