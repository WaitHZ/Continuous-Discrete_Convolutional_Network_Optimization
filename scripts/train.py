import numpy as np

import torch
import torch.nn.functional as F

from datasets import FoldDataset
from torch_geometric.loader import DataLoader

from models import Model

from DynamicDrawer import DynamicDrawer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(epoch, dataloader):
    """
        Complete the training of a single epoch
        return the accuracy and loss on the training set within the epoch
    """
    num_batches = tot_loss = correct = 0.0
    net.train()
    for data in dataloader:
        data=data.to(device)
        optimizer.zero_grad()
        tmp = net(data)
        loss = F.cross_entropy(tmp.log_softmax(dim=-1), data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        num_batches += 1
        tot_loss += float(loss)

        with torch.no_grad():
            pred = tmp.max(1)[1]

        correct += pred.eq(data.y).sum().item()

    print(f'epoch {epoch+1}, loss = {tot_loss/num_batches:.4f}')

    return correct/len(dataloader.dataset), tot_loss/num_batches
        
def test(dataloader):
    net.eval()
    correct = 0
    
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = net(data).max(1)[1]

        correct += pred.eq(data.y).sum().item()

    return correct / len(dataloader.dataset)

def parse_args():
    args = {
        'sequential_size': 5,
        'data_dir': '/root/fold',
        'geometric_radius': 4.0,
        'kernel_channels': [24],
        'base_width': 64, 
        'channels': [256, 512, 1024, 2048],
        'num_epochs': 100,
        'batch_size': 8,
        'lr': 0.01,
        'weight_decay': 5e-4, 
        'momentum': 0.9,
        'workers': 8,
        'seed': 0
    }

    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)

    np.random.seed(args['seed'])

    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = FoldDataset(root=args['data_dir'], random_seed=args['seed'], split='training')
    valid_dataset = FoldDataset(root=args['data_dir'], random_seed=args['seed'], split='validation')
    test_fold = FoldDataset(root=args['data_dir'], random_seed=args['seed'], split='test_fold')
    test_family = FoldDataset(root=args['data_dir'], random_seed=args['seed'], split='test_family')
    test_super = FoldDataset(root=args['data_dir'], random_seed=args['seed'], split='test_superfamily')

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])
    fold_loader = DataLoader(test_fold, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])
    family_loader = DataLoader(test_family, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])
    super_loader = DataLoader(test_super, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])
    
    net = Model(
        geometric_radii=[8.0, 12.0, 16.0, 20.0],
        sequential_kernel_size=args['sequential_size'],
        kernel_channels=[24],
        channels=[256, 512, 1024, 2048],
        base_width=args['base_width'],
        num_classes=train_dataset.num_classes
    )
    net = net.to(device=device)
    net.load_para()

    optimizer = torch.optim.SGD(net.parameters(),weight_decay=args['weight_decay'],lr=args['lr'],momentum=args['momentum'])

    # Instantiate your own drawing object
    dd = DynamicDrawer(3, ['loss', 'train_acc', 'valid_acc'], (1, args['num_epochs']), (0, 1.2))

    for epoch in range(args['num_epochs']):
        train_acc, train_loss = train(epoch,train_loader)
        
        valid_acc = test(valid_loader)

        dd.add_points(epoch+1, [train_loss, train_acc, valid_acc])

        print(f'Epoch:{epoch+1:03d}, Train: {train_acc:.4f}, Validation: {valid_acc:.4f}')

    net.save_()  # Save complete model parameters for subsequent prediction and visualization

    # Save the graph of the training process
    dd.save_fig(name='loss')
    
    print(f'Fold: {test(fold_loader):.4f}, Family: {test(family_loader):.4f}, Super: {test(super_loader):.4f}')

