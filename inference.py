import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import dataset
from model import GraphSAGE
import pickle
import os


valid_path = '/home/jina/reprod/data/valid'
batch_size = 32
epochs = 100
num_layers = 3
lr = 0.001
pk = os.path.join(valid_path, 'valid_1.pkl')

if pk == None:
    valid_data = dataset.create_dataset(valid_path)
    with open('train_pickle.pkl', 'wb') as f:
        pickle.dump(valid_data, f)
else:
    with open(pk, 'rb') as f:
        valid_data = pickle.load(f)

valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

node_dim = valid_data[0].x.size(1)
edge_dim = valid_data[0].edge_attr.size(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GraphSAGE(node_dim, edge_dim, num_layers=num_layers).to(device)
# model load 해오기
model = 

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss() 

model.train()

for epoch in range(epochs):
    running_loss = 0
    running_acc = 0
    total_samples = 0

    for data in valid_loader:
        # node_dim = data.x.size(1)
        # edge_dim = data.edge_attr.size(1)
        # # print(node_dim, edge_dim)
        # model = GraphSAGE(node_dim, edge_dim, num_layers=num_layers).to(device)
        # model.train()
        data = data.to(device)
        
        data.x = data.x.float()
        data.y = data.y.long()

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y)
        _, preds = torch.max(logits, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.y.size(0)
        # print(torch.sum(preds == data.y), data.y.size(0))
        running_acc += torch.sum(preds == data.y).item()
        total_samples += data.y.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    print(f'epoch: {epoch+1}, Train loss: {epoch_loss:.3f}, acc: {epoch_acc:.3f}')


@torch.no_grad()
def evaluate(loader):
    model.eval()
    # evaluator = Evaluator(name='ogbg-molhiv')
    y_true, y_pred = [], []

    for data in loader:
        data = data.to(device)

        # 입력 데이터의 타입을 float으로 변환
        data.x = data.x.float()
        data.y = data.y.float()

        out = model(data.x, data.edge_index, data.batch)
        y_true.append(data.y.view(out.shape).cpu())
        y_pred.append(out.cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)['rocauc']