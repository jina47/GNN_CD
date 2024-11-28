import numpy as np
import random
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from model import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from train import *
from inference import *
import pickle
import copy
import argparse
import wandb


# Random seed setting
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_pkl', type=str, default = '/home/jina/reprod/data/pickle/triple3')
    parser.add_argument('--valid_pkl', type=str, default = None)
    parser.add_argument('--test_pkl', type=str, default = '/home/jina/reprod/data/pickle/triple_test')
    # parser.add_argument('--test_pkl', type=str, default = None)

    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--epochs', type=int, default = 50)
    parser.add_argument('--num_layers', type=int, default = 3)
    parser.add_argument('--lr', type=float, default = 0.005)
    parser.add_argument('--threshold', type=float, default = 0.5)
    parser.add_argument('--seed', type=int, default = 11)

    parser.add_argument('--device', type=str, default = 'cpu')

    args = parser.parse_args()

    train_name = args.train_pkl.split('/')[-1]
    if args.test_pkl:
        test_name = args.test_pkl.split('/')[-1]
    else:
        test_name = '' 
    wandb_name = train_name + '_' + test_name

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
        project="gnn_cd",
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
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    node_dim = train_data[0].x.size(1)
    edge_dim = train_data[0].edge_attr.size(1)

    # model, optimizer, scheduler, loss
    skeleton_model = customGraphSAGE(node_dim, edge_dim, num_layers=num_layers, output_class=2, device=device, num_samples=None).to(device)
    optimizer = torch.optim.Adam(skeleton_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    wandb.config.model = skeleton_model.__class__.__name__

    best_valid_f1 = 0
    best_valid_acc = 0
    best_valid_loss = 9999

    wandb.watch(skeleton_model, log="all")
    print('****** Skeleton ******')
    for epoch in range(epochs):

        print(f'Epoch: {epoch + 1}')
        train_loss, train_acc, train_f1 = predict_train(skeleton_model, train_loader, optimizer, criterion, device, threshold)
        print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
        valid_loss, valid_acc, valid_f1, valid_predictions, valid_labels = predict_test(skeleton_model, valid_loader, criterion, device, threshold)
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
            best_model = skeleton_model
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss

    print("\nSkeleton Training complete!")
    print(f"Best Valid F1: {best_valid_f1:.3f}, Best Valid Acc: {best_valid_acc:.3f}, Best Valid Loss: {best_valid_loss:.3f}")


    accuracy = accuracy_score(valid_labels, valid_predictions)
    f1 = f1_score(valid_labels, valid_predictions)
    tn, fp, fn, tp = confusion_matrix(valid_labels, valid_predictions).ravel().tolist()

    print(f"Skeleton Valid Accuracy: {accuracy:}")
    print(f"Skeleton Valid F1 Score: {f1:}")
    print(f"Skeleton Valid tn: {tn} fp: {fp} fn: {fn} tp: {tp}")

    wandb.log({
        "best_valid_accuracy": best_valid_acc,
        "best_valid_f1": best_valid_f1,
        "final_valid_tn": tn,
        "final_valid_fp": fp,
        "final_valid_fn": fn,
        "final_valid_tp": tp,
    })

    if args.test_pkl:
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        test_loss, test_acc, test_f1, test_predictions, test_labels, new_test_data = predict_test(skeleton_model, test_loader, criterion, device, threshold, mode='test')
        print("\nSkeleton Test complete!")
        print(f"Best Test F1: {test_f1:.3f}, Best Test Acc: {test_acc:.3f}, Best Test Loss: {test_loss:.3f}")

        accuracy = accuracy_score(test_labels, test_predictions)
        f1 = f1_score(test_labels, test_predictions)
        tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel().tolist()

        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "test_f1": f1,
            "test_tn": tn,
            "test_fp": fp,
            "test_fn": fn,
            "test_tp": tp,
        })

        print(f"Skeleton Test Accuracy: {accuracy:}")
        print(f"Skeleton Test F1 Score: {f1:}")
        print(f"Skeleton Test tn: {tn} fp: {fp} fn: {fn} tp: {tp}")

    wandb.finish()

    
    # Orientation
    new_test_loader = DataLoader(new_test_data, batch_size=1, shuffle=False)
    orient_model = customGraphSAGE(node_dim, edge_dim, num_layers=num_layers, output_class=2, device=device, num_samples=None).to(device)

    optimizer = torch.optim.Adam(orient_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    # wandb.config.model = orient_model.__class__.__name__

    best_valid_f1 = 0
    best_valid_acc = 0
    best_valid_loss = 9999

    print('****** Orientation ******')
    for epoch in range(epochs):

        print(f'Epoch: {epoch + 1}')
        train_loss, train_acc, train_f1 = orient_train(orient_model, train_loader, optimizer, criterion, device, threshold)
        print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
        valid_loss, valid_acc, valid_f1, valid_predictions, valid_labels = orient_test(orient_model, valid_loader, criterion, device, threshold)
        print(f'Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}, Valid F1: {valid_f1:.3f}')

        scheduler.step(valid_loss)

        if best_valid_f1 < valid_f1:
            best_valid_f1 = valid_f1
        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            best_model = skeleton_model
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss

    print("\nOrientation Training complete!")
    print(f"Best Valid F1: {best_valid_f1:.3f}, Best Valid Acc: {best_valid_acc:.3f}, Best Valid Loss: {best_valid_loss:.3f}")


    accuracy = accuracy_score(valid_labels, valid_predictions)
    f1 = f1_score(valid_labels, valid_predictions)
    tn, fp, fn, tp = confusion_matrix(valid_labels, valid_predictions).ravel().tolist()

    print(f"\nOrientation Valid Accuracy: {accuracy:}")
    print(f"Orientation Valid F1 Score: {f1:}")
    print(f"Orientation Valid tn: {tn} fp: {fp} fn: {fn} tp: {tp}")

    orient_model.eval()
    running_loss = 0
    running_acc = 0
    total_samples = 0
    
    orient_predictions = []
    orient_labels = []
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for i, data in enumerate(new_test_loader):
            data.x = data.x.float()
            data.y = data.y.long()

            original_edge_index = copy.deepcopy(data.edge_index)
            original_y = copy.deepcopy(data.y)
            
            mask = data.pred == 1
            filtered_edge_index = data.edge_index[:, mask]
            filtered_y = original_y[mask]
            filtered_y[filtered_y==2] = 0 # 2 label을 0로 변경 
            
            data = data.to(device)

            logits = orient_model(data.x, filtered_edge_index, data.edge_attr[mask], data.num_nodes)
            preds = (torch.sigmoid(logits) > threshold).float()
            # test_data[i].pred = preds.view(-1)
            loss = criterion(logits, filtered_y.view(-1, 1).float())

            orient_predictions.extend(preds.cpu().numpy())
            orient_labels.extend(filtered_y.cpu().numpy())
            
            preds = preds.view(-1)
            preds[preds==0] = 2 # 다시 되돌리기
            data.pred[mask] = preds
            test_predictions.extend(data.pred.cpu().numpy())
            test_labels.extend(data.y.cpu().numpy())

            running_loss += loss.item() * filtered_y.size(0)
            running_acc += torch.sum(preds.squeeze() == filtered_y).item()
            total_samples += filtered_y.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    epoch_f1 = f1_score(orient_labels, orient_predictions, average="binary")
    
    print(f"\nOrientation Test Accuracy: {epoch_acc:}")
    print(f"Orientation Test F1 Score: {epoch_f1:}")
    print(f"Orientation Test tn: {tn} fp: {fp} fn: {fn} tp: {tp}")

    print('Final')
    # print(test_predictions)
    # print(test_labels)
    # orientation 제대로 예측하는지 & test_label이 바뀌는지 확인해야 

