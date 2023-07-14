import torch
from models import PreTrainModel
from datasets import PretrainDataset
import os

import utils

from torch_geometric.loader import DataLoader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


num_epochs = 20
lr = 0.01
T = 1.5
batch_size = 4


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pre_train_data = PretrainDataset('../ec', split='train')

    pre_train_loader = DataLoader(pre_train_data, batch_size, True, num_workers=4)

    net = PreTrainModel(
        geometric_radii=[8.0, 12.0, 16.0, 20.0],
        sequential_kernel_size=5,
        kernel_channels=[24],
        channels=[256, 512, 1024, 2048],
        base_width=64
    )
    net = net.to(device=device)

    loss = utils.sim_loss

    optimizer = torch.optim.SGD(net.parameters(), lr)

    for epoch in range(num_epochs):
        num_batches = tot_loss = 0.0

        for X in pre_train_loader:
            X = X.to(device)

            optimizer.zero_grad()
            res = net(X)
            l = loss(res, T)
            l.backward()
            optimizer.step()
            num_batches += 1
            tot_loss += float(l)

            # print(f'batch {num_batches:.0f}, loss = {float(l):.2f}')
        
        print(f'epoch {epoch+1}, loss = {tot_loss/num_batches:.2f}')

    if not os.path.exists('./para'):
        os.mkdir('./para')

    net.save_parameters()
