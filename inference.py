import torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import copy
import numpy as np


def predict_test(model, test_loader, criterion, device, threshold, num_node=0, mode='valid'):
    model.eval()
    running_loss = 0
    total_samples = 0

    test_predictions = []
    test_labels = []
    test_data = []
    test_probs = []

    with torch.no_grad():
        for data in test_loader:
            data.x = data.x.float()
            data = data.to(device)

            original_data = copy.deepcopy(data)
            if num_node != 0:
                logits = model(data.x, data.edge_index, data.edge_attr, data.num_nodes, num_node)
            else:
                logits = model(data.x, data.edge_index, data.edge_attr, data.num_nodes)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            if mode == 'test':
                original_data.pred = preds.view(-1)
                test_data.append(original_data)
            
            data.y = data.y.long()
            data.y[data.y == 2] = 1  # Convert label 2 to 1
            
            loss = criterion(logits, data.y.view(-1, 1).float())

            test_predictions.extend(preds.view(-1).cpu().numpy())
            test_labels.extend(data.y.cpu().numpy())
            test_probs.extend(probs.view(-1).cpu().numpy())  # 확률값 저장

            running_loss += loss.item() * data.y.size(0)
            total_samples += data.y.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(test_labels, test_predictions)
    epoch_f1 = f1_score(test_labels, test_predictions, average="binary")
    thresholds = np.arange(0.1, 1.0, 0.05)
    if mode != 'test':
        auroc = roc_auc_score(test_labels, test_probs)
        best_threshold = 0.0
        best_acc = 0.0
        for t in thresholds:
            t_preds = (np.array(test_probs) > t).astype(int)
            t_acc = accuracy_score(test_labels, t_preds)
            if t_acc > best_acc:
                best_acc = t_acc
                best_threshold = t

    if mode == 'test':
        return epoch_loss, epoch_acc, epoch_f1, test_probs, test_predictions, test_labels, test_data
    else:
        return epoch_loss, epoch_acc, epoch_f1, auroc, best_threshold, test_predictions, test_labels


def orient_test(model, test_loader, criterion, device, threshold, mode='valid'):
    model.eval()
    running_loss = 0
    total_samples = 0
    epoch_loss = 0 
    
    orient_predictions = []
    orient_labels = []
    final_predictions = []
    final_labels = []

    with torch.no_grad():
        for data in test_loader:
            data.x = data.x.float()
            data.y = data.y.long()
            data = data.to(device)

            original_y = copy.deepcopy(data.y)
            
            if mode == 'total':
                mask = data.pred == 1
            else:
                mask = data.y != 0

            filtered_edge_attr = data.edge_attr[mask]
            filtered_edge_index = data.edge_index[:, mask]
            filtered_y = data.y[mask]
            filtered_y[filtered_y==2] = 0 # 2 label을 0로 변경 

            logits = model(data.x, filtered_edge_index, filtered_edge_attr, data.num_nodes)
            preds = (torch.sigmoid(logits) > threshold).float()

            loss = criterion(logits, filtered_y.view(-1, 1).float())

            orient_predictions.extend(np.copy(preds.cpu().numpy()))
            orient_labels.extend(np.copy(filtered_y.cpu().numpy()))
            
            running_loss += loss.item() * filtered_y.size(0)
            total_samples += filtered_y.size(0)

            preds = preds.view(-1)
            preds[preds==0] = 2 # 다시 되돌리기

            if mode == 'total':
                data.pred[mask] = preds.float()
                final_predictions.extend(data.pred.cpu().numpy())
                final_labels.extend(data.y.cpu().numpy())
            else:
                original_y[mask] = preds.long()
                final_predictions.extend(original_y.cpu().numpy())
                final_labels.extend(data.y.cpu().numpy())


    if mode == 'valid':
        epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(orient_labels, orient_predictions)
    epoch_f1 = f1_score(orient_labels, orient_predictions, average="binary")

    if mode == 'orientation':
        
        return epoch_loss, epoch_acc, epoch_f1, original_y.cpu().numpy(), data.y.cpu().numpy(), final_predictions, final_labels
    else:
        return epoch_loss, epoch_acc, epoch_f1, orient_predictions, orient_labels, final_predictions, final_labels



def three_test(model, test_loader, criterion, device, threshold, mode='valid'):
    model.eval()
    running_loss = 0
    total_samples = 0

    test_predictions = []
    test_labels = []
    

    with torch.no_grad():
        for data in test_loader:
            data.x = data.x.float()
            data = data.to(device)

            logits = model(data.x, data.edge_index, data.edge_attr, data.num_nodes)
            
            preds = torch.argmax(logits, dim=1)
            
            data.y = data.y.long()
            
            loss = criterion(logits, data.y)

            test_predictions.extend(preds.view(-1).cpu().numpy())
            test_labels.extend(data.y.cpu().numpy())

            running_loss += loss.item() * data.y.size(0)
            total_samples += data.y.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(test_labels, test_predictions)
    epoch_f1 = f1_score(test_labels, test_predictions, average="macro") 

    return epoch_loss, epoch_acc, epoch_f1, test_predictions, test_labels
