import torch
from torch.nn import functional as F

import numpy as np
from sklearn.preprocessing import normalize


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def orientation(pos):
    """
        Calculate direction information based on coordinates
    """

    u = normalize(X=pos[1:,:] - pos[:-1,:], norm='l2', axis=1) # dim = (n-1, 3)
    u1 = u[1:,:]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)             # The radial direction vector has dimension (n-2, 3)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)    # plane normal direction vector
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)      # Main Chain Direction Vector
    ori = np.stack([b, n, o], axis=1)                       # Three vectors are stacked, with dimensions (n-2, 3, 3)
    return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)# 扩展回维数(n, 3, 3)，直接复制第一行和最后一行


def sim_loss(res: torch.Tensor, T=2.0) -> torch.Tensor: 
    """
        Computing Contrastive Loss for Unsupervised Learning
    """
    res = F.normalize(res, dim=1)
    mat = torch.mm(res, res.T)

    loss = torch.tensor(0, device=device)

    for i in range(mat.shape[0]):
        den = (torch.exp(mat[i]) / T).sum() - torch.exp(mat[i, i]) / T
        loss = loss - torch.log(torch.exp(mat[i][(i+mat.shape[0]//2)%mat.shape[0]])/T/den)

    return loss / mat.shape[0]
