import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import dataset
from model import GraphSAGE


train_path = '/home/jina/reprod/data/train'
batch_size = 32
epochs = 100
num_layers = 3
lr = 0.001

train_data = dataset.create_dataset(train_path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

node_dim = train_data[0].x.size(1)
edge_dim = train_data[0].edge_attr.size(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(node_dim, edge_dim, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(lr=lr)
criterion = nn.CrossEntropyLoss() 

model.train()

for epoch in range(epochs):
    running_loss = 0
    running_acc = 0

    for data in train_loader:
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

        running_loss += loss.item()
        running_acc += torch.sum(preds == data.y)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    print(f'epoch: {epoch+1}, Train loss: {epoch_loss:.3f}, acc: {epoch_acc:.3f}')
