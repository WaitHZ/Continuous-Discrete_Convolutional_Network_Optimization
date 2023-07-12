import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch_scatter import scatter_max, scatter_mean

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn import radius


def kaiming_uniform_1(tensor, size, leak_neg_k=0.2):
    """
        Kaiming_Uniform for Leakly Relu
    """
    fanin = 1
    for i in range(len(size)):
        fanin *=size[i]

    bound = math.sqrt(6.0 / (1 + leak_neg_k**2) * fanin)

    with torch.no_grad():
        return tensor.uniform_(-bound,bound)

def kaiming_uniform_2(tensor,size):
    """
        Kaiming_Uniform for Linear Layer
    """
    fanin = 1
    for i in range(1,len(size)):
        fanin *=size[i]
    bound = math.sqrt(6.0/((1 + 5)*fanin))

    with torch.no_grad():
        return tensor.uniform_(-bound,bound)

class WeightNet(nn.Module):
    """
        The network in the network is used to generate the weight of the convolution kernel.
    """
    def __init__(self, l: int, kernel_channels: list[int]):
        super(WeightNet, self).__init__()

        self.l = l
        self.kernel_channels = kernel_channels

        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        self.Ws.append(torch.nn.Parameter(torch.empty(l, 7, kernel_channels[0])))
        self.bs.append(torch.nn.Parameter(torch.empty(l, kernel_channels[0])))
        self.relu = nn.LeakyReLU(negative_slope=0.2)  

    def reset_parameters(self):
        kaiming_uniform_1(self.Ws[0].data,size=[self.l,7,self.kernel_channels[0]])
        self.bs[0].data.fill_(0.0)
    
    def forward(self,pos_ori,seq_idx):
        W = torch.index_select(self.Ws[0],0,seq_idx)
        b = torch.index_select(self.bs[0],0,seq_idx)
        #(batch_size, 1, input_dim)-->(batch_size, 1, output_dim)-->(batch_size,output_dim)
        weight =self.relu(torch.bmm(pos_ori.unsqueeze(1),W).squeeze(1)+b)

        return weight


class CDConv(MessagePassing):
    def __init__(self, r: float, l: float, kernel_channels: list[int], in_channels: int, out_channels: int, add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')  # add mean max min
        super().__init__(**kwargs)
        self.r = r
        self.l = l
        self.kernel_channels = kernel_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.WeightNet = WeightNet(l, kernel_channels)
        self.W = torch.nn.Parameter(torch.empty(kernel_channels[-1] * in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.WeightNet.reset_parameters()
        kaiming_uniform_2(self.W.data,size =[self.kernel_channels])

    def forward(self, x: OptTensor, pos: Tensor, seq: Tensor, ori: Tensor, batch: Tensor) -> Tensor:
        # Return the point [row index] [column index] in the range of r, and then define [edge index] edge_index
        row, col = radius(pos, pos, self.r, batch, batch, max_num_neighbors=9999)
        edge_index = torch.stack([col, row], dim=0)

        # pos_batch = torch.cat([pos, batch.unsqueeze(1)], dim=1)
        # dist_matrix = torch.cdist(pos_batch, pos_batch)
        # mask = dist_matrix <= self.r
        # edge_index = mask.nonzero(as_tuple=False).t()

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=pos.size(0))

        out = self.propagate(edge_index, x=(x, None), pos=(pos, pos), seq=(seq, seq), ori=(ori.reshape((-1, 9)), ori.reshape((-1, 9))), size=None)
        # out * self.W 
        out = torch.matmul(out, self.W)
        return out
    
    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor, seq_i: Tensor, seq_j: Tensor, ori_i: Tensor, ori_j: Tensor) -> Tensor:
        # orientation
        # The relative position is obtained by subtraction
        # The distance is calculated by the two norms (calculated on the last dimension + keepdim keeps the dimension unchanged)
        pos = pos_j - pos_i
        distance = torch.norm(input=pos, p=2, dim=-1, keepdim=True)

        pos /= (distance + 1e-9)
        #  pos = ori_i * pos 
        pos = torch.matmul(ori_i.reshape((-1, 3, 3)), pos.unsqueeze(2)).squeeze(2)
        ori = torch.sum(input=ori_i.reshape((-1,3, 3)) * ori_j.reshape((-1,3,3)), dim=2, keepdim=False)

        normed_distance = distance / self.r

        max_seq_diff = self.l // 2
        seq_diff = seq_j - seq_i + max_seq_diff
        seq_diff = torch.clamp(seq_diff, min=0, max=max_seq_diff*2)
        seq_idx = seq_diff.squeeze(1).to(torch.int64)
        normed_length = torch.abs(seq_diff-max_seq_diff) / max_seq_diff


        # generated kernel weight: PointConv or PSTNet
        delta = torch.cat([pos, ori, distance], dim=1)
        kernel_weight = self.WeightNet(delta, seq_idx)
        # smoothing factor
        smooth = 0.5 - torch.tanh(normed_distance*normed_length*16.0 - 14.0)*0.5
        # convolution
        msg = torch.matmul((kernel_weight*smooth).unsqueeze(2), x_j.unsqueeze(1))
        msg = msg.reshape((-1, msg.size(1)*msg.size(2)))
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(r={self.r}, '
                f'l={self.l},'
                f'kernel_channels={self.kernel_channels},'
                f'in_channels={self.in_channels},'
                f'out_channels={self.out_channels})')
    
        
class AvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x,pos,seq,ori,batch):
        # idx = seq/2 and then splice the last element
        idx = torch.div(seq.squeeze(1), 2, rounding_mode='floor')
        idx = torch.cat([idx, idx[-1].view((1,))])
        # When the adjacent ones are the same, it is 0, and when they are different, it is 1
        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        # Each element becomes the previous accumulated value - the original value at this position
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)
        
        x = scatter_mean(src=x, index=idx, dim=0)
        pos = scatter_mean(src=pos, index=idx, dim=0)
        seq = scatter_max(src=torch.div(seq, 2, rounding_mode='floor'), index=idx, dim=0)[0]
        ori = scatter_mean(src=ori, index=idx, dim=0)
        ori = torch.nn.functional.normalize(ori, 2, -1)
        batch = scatter_max(src=batch, index=idx, dim=0)[0]

        return x, pos, seq, ori, batch
