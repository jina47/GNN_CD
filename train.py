import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import dataset
from model import GraphSAGE
import pickle
import os
import random
import numpy as np


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# train_path = '/home/jina/reprod/data/train'
train_path = '/home/jina/reprod'
batch_size = 32
epochs = 100
num_layers = 3
lr = 0.001
pk = os.path.join(train_path, 'train_pickle.pkl')

if pk == None:
    train_data = dataset.create_dataset(train_path)
    with open('train_pickle.pkl', 'wb') as f:
        pickle.dump(train_data, f)
else:
    with open(pk, 'rb') as f:
        train_data = pickle.load(f)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

node_dim = train_data[0].x.size(1)
edge_dim = train_data[0].edge_attr.size(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(node_dim, edge_dim, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss() 

model.train()

for epoch in range(epochs):
    running_loss = 0
    running_acc = 0
    total_samples = 0

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

        running_loss += loss.item() * data.y.size(0)
        # print(torch.sum(preds == data.y), data.y.size(0))
        running_acc += torch.sum(preds == data.y).item()
        total_samples += data.y.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    print(f'epoch: {epoch+1}, Train loss: {epoch_loss:.3f}, acc: {epoch_acc:.3f}')
