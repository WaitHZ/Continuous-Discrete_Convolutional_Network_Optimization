"""
Our code refers to the code provided in the following article:

@inproceedings{fan2023cdconv,
  title={Continuous-Discrete Convolution for Geometry-Sequence Modeling in Proteins},
  author={Hehe Fan and Zhangyang Wang and Yi Yang and Mohan Kankanhalli},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
"""


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from utils import orientation

aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i in range(0, 21):
    aa_to_id[aa[i]] = i

    
class FoldDataset(Dataset):

    def __init__(self, root='../fold', random_seed=0, split='training'):
        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        npy_dir = os.path.join(root, 'coordinates', split)
        fasta_file = os.path.join(root, split+'.fasta')

        # Load the fasta
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        # Load the class map
        fold_classes = {}
        with open(os.path.join(root, 'class_map.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                fold_classes[arr[0]] = int(arr[1])

        # Load the protein fold
        protein_folds = {}
        with open(os.path.join(root, split+'.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t') 
                protein_folds[arr[0]] = fold_classes[arr[-1]]

        self.data = []
        self.labels = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            
            # According to the first dimension (row) sum and divide by the number of rows to get the average coordinates
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            # Calculate direction information after centering, dimension (n, 3, 3)
            # n is the number of amino acids in a chain
            ori = orientation(pos)

            # Record protein class and information in two dimensions
            self.data.append((pos, ori, amino_ids.astype(int)))
            self.labels.append(protein_folds[protein_name])
        # Record the maximum number of categories
        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        pos, ori, amino = self.data[idx]
        label = self.labels[idx]

        if self.split == "training":
            # Add random noise to ensure model robustness.
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)
        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data
    

class PretrainDataset(Dataset):
    """
        Datasets for pre-train, the data are from scope 1.7 ec.
    """
    def __init__(self, root='../ec/', random_seed=0, split='train'):
        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        npy_dir = os.path.join(root,'coordinates')
        fasta_file = os.path.join(root,split+'.fasta')

        # save protein sequence
        protein_seqs = []
        with open(fasta_file,'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name,np.array(amino_ids)))

        # read in coordinate, and calculate orientation
        self.data = []
        for protein_name,amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos,axis=0,keepdims=True) / pos.shape[0]
            pos = pos -center
            ori =orientation(pos)
            self.data.append((pos,ori,amino_ids.astype(int)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        pos, ori, amino =self.data[idx]
        # add noise when training
        if self.split == "training":
            pos =pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        pos = pos.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]),axis=1).astype(dtype=np.float32)
        data = Data(x=torch.from_numpy(amino),
                    edge_index = None,
                    edge_attr =None,
                    ori =torch.from_numpy(ori),
                    seq = torch.from_numpy(seq),
                    pos = torch.from_numpy(pos))
        return data
