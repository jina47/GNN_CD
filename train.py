import torch
from sklearn.metrics import accuracy_score, f1_score
import copy


def predict_train(model, train_loader, optimizer, criterion, device, threshold):
    model.train()
    running_loss = 0
    total_samples = 0
    train_predictions = []
    train_labels = []

    for data in train_loader:
        data.x = data.x.float()
        data.y = data.y.long()
        data.y[data.y == 2] = 1  
        data = data.to(device)

        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr, data.num_nodes)
        preds = (torch.sigmoid(logits) > threshold).float()
        loss = criterion(logits, data.y.view(-1, 1).float())
        loss.backward()
        optimizer.step()

        train_predictions.extend(preds.cpu().numpy())
        train_labels.extend(data.y.cpu().numpy())

        running_loss += loss.item() * data.y.size(0)
        total_samples += data.y.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(train_labels, train_predictions)
    epoch_f1 = f1_score(train_labels, train_predictions, average="binary")

    return epoch_loss, epoch_acc, epoch_f1


def orient_train(model, train_loader, optimizer, criterion, device, threshold):
    model.train()
    running_loss = 0
    total_samples = 0
    train_predictions = []
    train_labels = []

    for data in train_loader:
        data.x = data.x.float()
        data.y = data.y.long()
        data = data.to(device)
        
        original_y = copy.deepcopy(data.y)

        mask = original_y != 0

        filtered_edge_attr = data.edge_attr[mask]
        filtered_edge_index = data.edge_index[:, mask]
        filtered_y = data.y[mask]
        filtered_y[filtered_y == 2] = 0

        optimizer.zero_grad()
        logits = model(data.x, filtered_edge_index, filtered_edge_attr, data.num_nodes)
        preds = (torch.sigmoid(logits) > threshold).float()
        loss = criterion(logits, filtered_y.view(-1, 1).float())
        loss.backward()
        optimizer.step()

        train_predictions.extend(preds.cpu().numpy())
        train_labels.extend(filtered_y.cpu().numpy())

        running_loss += loss.item() * filtered_y.size(0)
        total_samples += filtered_y.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(train_labels, train_predictions)
    epoch_f1 = f1_score(train_labels, train_predictions, average="binary")

    return epoch_loss, epoch_acc, epoch_f1


def three_train(model, train_loader, optimizer, criterion, device, threshold, mode='our'):
    model.train()
    running_loss = 0
    total_samples = 0
    train_predictions = []
    train_labels = []

    for data in train_loader:
        data.x = data.x.float()
        data.y = data.y.long()
        
        # 다중 클래스 분류이므로 데이터 레이블을 변형하지 않음
        data = data.to(device)

        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr, data.num_nodes)
        
        # CrossEntropyLoss는 로짓을 그대로 사용
        loss = criterion(logits, data.y)
        
        # 각 클래스에 대한 확률을 얻고, 가장 높은 확률의 클래스를 예측으로 선택
        preds = torch.argmax(logits, dim=1)
        
        loss.backward()
        optimizer.step()

        train_predictions.extend(preds.cpu().numpy())
        train_labels.extend(data.y.cpu().numpy())

        running_loss += loss.item() * data.y.size(0)
        total_samples += data.y.size(0)

    # 에포크 손실 및 성능 지표 계산
    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(train_labels, train_predictions)
    epoch_f1 = f1_score(train_labels, train_predictions, average="macro")  # 다중 클래스이므로 macro 평균 사용

    return epoch_loss, epoch_acc, epoch_f1