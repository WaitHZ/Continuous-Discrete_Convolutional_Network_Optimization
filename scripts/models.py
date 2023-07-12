from typing import List

import torch
import torch.nn as nn

from torch_geometric.nn import MLP, global_mean_pool

from modules import *

import math


class Linear(nn.Module):
    """
        Inherit nn.Module to implement your own fully connected layer
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:

        super().__init__()

        module = []

        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)


class MLP(nn.Module):
    """
        Define a multilayer perceptron module
    """
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super().__init__()

        module = []

        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        # dropout, avoid overfitting
        module.append(nn.Dropout(dropout))

        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))

        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)

class BasicBlock(nn.Module):
    """
        Define the basic blocks that make up the network
    """
    def __init__(self,
                 r: float,
                 l: float,
                 kernel_channels: list[int],
                 in_channels: int,
                 out_channels: int,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:

        super().__init__()


        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                  out_channels=out_channels,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope,
                                  momentum=momentum)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.))

        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope,
                         momentum=momentum)
        self.conv = CDConv(r=r, l=l, kernel_channels=kernel_channels, in_channels=width, out_channels=width)
        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope,
                             momentum=momentum)

    def forward(self, x, pos, seq, ori, batch):
        identity = self.identity(x)

        x = self.input(x)
        # print(x.shape)
        x = self.conv(x, pos, seq, ori, batch)
        out = self.output(x) + identity

        return out
    
class BranchBlock(nn.Module):
    """
        Inspired by GoogLeNet, we try to improve the basic block
    """
    def __init__(self,
                 r: float,
                 kernel_channels: list[int],
                 in_channels: int,
                 out_channels: int,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:
        
        super().__init__()

        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                  out_channels=out_channels,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope,
                                  momentum=momentum)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.))

        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope,
                         momentum=momentum)
        
        self.conv1 = CDConv(r=1.5*r, l=5, kernel_channels=kernel_channels, in_channels=width, out_channels=width//2)
        self.conv2 = CDConv(r=r, l=7, kernel_channels=kernel_channels, in_channels=width, out_channels=width//4)
        self.conv3 = CDConv(r=0.75*r, l=11, kernel_channels=kernel_channels, in_channels=width, out_channels=width//4)
        
        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope,
                             momentum=momentum)

    def forward(self, x, pos, seq, ori, batch):
        identity = self.identity(x)
        x = self.input(x)
        x1 = self.conv1(x, pos, seq, ori, batch)
        x2 = self.conv2(x, pos, seq, ori, batch)
        x3 = self.conv3(x, pos, seq, ori, batch)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.output(x) + identity

        return out


class  Model(nn.Module):
    def __init__(self,
                 geometric_radii: List[float],
                 sequential_kernel_size: float,
                 kernel_channels: List[int],
                 channels: List[int],
                 base_width: float = 16.0,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:

        super().__init__()

        assert (len(geometric_radii) == len(channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()

        layers = []
        in_channels = embedding_dim

        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r = radius,
                                     l=sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)

        self.classifier = MLP(in_channels=channels[-1],
                              mid_channels=max(channels[-1], num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)

    def load_para(self, root='./para/'):
        """
            Load the pretrained parameters, excluding the final classifier.
        """
        self.embedding.load_state_dict(torch.load(root+'embedding'))
        
        for i, layer in enumerate(self.layers):
            layer.load_state_dict(torch.load(root+f'layer {i}'))
            

    def forward(self, data):
        x, pos, seq, ori, batch = (self.embedding(data.x), data.pos, data.seq, data.ori, data.batch)

        for i, layer in enumerate(self.layers):
            # print(layer.__class__.__name__)
            x = layer(x, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                x = global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)

        out = self.classifier(x)

        return out

    def save_(self, root='./vis_para/'):
        """
            Save the full parameters of the model
        """
        torch.save(self.embedding.state_dict(), root+'embedding')

        for i, layer in enumerate(self.layers):
            torch.save(layer.state_dict(), root+f'layer {i}')

        torch.save(self.classifier.state_dict(), root+f'class')

    def load_(self, root='./vis_para/'):
        """
            Load the full parameters of the model
        """
        self.embedding.load_state_dict(torch.load(root+'embedding'))
        
        for i, layer in enumerate(self.layers):
            layer.load_state_dict(torch.load(root+f'layer {i}'))

        self.classifier.load_state_dict(torch.load(root+f'class'))

    

class PreTrainModel(nn.Module):
    """
        Define a pretrained network, including protein encoders and MLP
    """
    def __init__(self,
                 geometric_radii: List[float],
                 sequential_kernel_size: float,
                 kernel_channels: List[int],
                 channels: List[int],
                 base_width: float = 16.0,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False) -> nn.Module:

        super().__init__()

        assert (len(geometric_radii) == len(channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()

        layers = []
        in_channels = embedding_dim

        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)

        self.MLP = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]//2), nn.ReLU(),
            nn.Linear(channels[-1]//2, channels[-1]//4), nn.ReLU(),
            nn.Linear(channels[-1]//4, channels[-1]//8)
        )

        
    def forward(self, data):
        torch.autograd.set_detect_anomaly(True)

        x, pos, seq, ori, batch = (data.x, data.pos, data.seq, data.ori, data.batch)

        x_ = x.reshape((x.shape[0], 1))
        ori_ = ori.reshape((ori.shape[0], 9))
        batch_ = batch.reshape((batch.shape[0], 1))

        X = torch.cat((x_, pos, seq, ori_, batch_), dim=1).type(torch.float32)

        indices = torch.randperm(X.size(0))[:math.floor(0.9*len(X))]
        indices, _ = indices.sort()
        X_new = X[~indices]
        X_new[:, -1] += batch.max() + 1

        X = torch.cat((X, X_new), dim=0)

        x, pos, seq, ori, batch = X[:, 0].clone().type(torch.int32), X[:, 1:4].clone(), X[:, 4].reshape(-1, 1).clone(), X[:, 5:14].reshape((-1, 3, 3)).clone(), X[:, 14].reshape(-1).clone()

        batch = batch.type(torch.int64)

        x = self.embedding(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)

            if i == len(self.layers) - 1:
                x = global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)

        x = self.MLP(x)

        return x


    def save_parameters(self):
        """
            Preserve the pretrained model parameters, except the last added MLP
        """
        torch.save(self.embedding.state_dict(), './para/embedding')
        for i, layer in enumerate(self.layers):
            torch.save(layer.state_dict(), f'./para/layer {i}')
