import torch
import torch.nn.functional as F

from datasets import FoldDataset
from torch_geometric.loader import DataLoader

from models import Model

from DynamicDrawer import DynamicDrawer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        'data_dir': '../fold',
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

    train = FoldDataset(root=args['data_dir'], random_seed=args['seed'], split='training')
    family = FoldDataset(root=args['data_dir'], random_seed=args['seed'], split='test_family')
    
    family_loader = DataLoader(family, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])
    
    net = Model(
        geometric_radii=[8.0, 12.0, 16.0, 20.0],
        sequential_kernel_size=args['sequential_size'],
        kernel_channels=[24],
        channels=[256, 512, 1024, 2048],
        base_width=args['base_width'],
        num_classes=train.num_classes
    )
    net = net.to(device=device)
    net.load_()

    family_acc = test(family_loader)

    print(f'family_acc = {family_acc:.4f}')
