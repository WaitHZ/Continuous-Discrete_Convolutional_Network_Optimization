# Continuous-Discrete Convolutional Network Optimization Based on Pre-Training

这是浙江大学暑期课程第23组的课题的代码实现，主要基于连续离散卷积这篇论文进行改进。

首先，我们使用EC数据集通过对比学习进行蛋白质结构分类任务的预训练，期望改进神经网络在fold数据集上进行蛋白质结构分类的效果。具体而言，实验中我们将原本的EC中的蛋白质随机删除10%的氨基酸后与原本的蛋白质构成正反数据对，使用CDconv进行预训练编码，随后通过MLP提取编码信息，但不执行分类任务，通过最小化正反数据对的编码信息差值来迭代更新网络参数。在EC蛋白质数据集上与训练完的参数用于fold数据集分类任务参数的初始化，实验表明***---------waitting***

其次，我们尝试改进模型的框架，并行运行不同规模的CDconv，以在理解蛋白质氨基酸序列时灵活调整离散卷积核覆盖范围。具体而言，对于原论文代码实现中的Basic Block，我们将其中的l=5的CDconv模块改成了l=5，7，11的三个并行的CDconv模块，整个模块重命名为Branch Block。实验表明这种改进可以在迭代刚开始时取得较大的改进，但随着神经网络训练的进行，最终效果逐渐与原来的网络相似。这说明基于蛋白质序列顺序的离散卷积核覆盖范围的大小与神经网络最终的编码效果关联度不高。

***[配图：branch block 结构图]***

最后，我们通过实验求证论文中的结论——深度学习神经网络对蛋白质结构的理解主要基于中心的氨基酸。我们猜测这个结论的得出有可能跟论文代码实现中的图神经网络中消息传递的聚合函数设置为'sum'相关，'sum'会导致周围点数较多的氨基酸的特征编码值较大，而中心氨基酸周围的氨基酸点数一般比边缘点周围的氨基酸点数多，这就会使得中心点特征编码值相较于边缘点的特征编码值更大，也即激活值更高，造成神经网络理解蛋白质主要基于蛋白质中心氨基酸的现象。我们将这里的聚合函数设置为'mean'后重新训练网络、绘制激活值函数，以此来探索人为设置聚合函数对神经网络理解蛋白质的影响。***--------waiting***

## Installation

The code is tested with Ubuntu 20.04.5 LTS, CUDA v11.7, cuDNN v8.5, PyTorch 1.13.1, PyTorch Geometric (PyG), PyTorch Scatter and PyTorch Sparse.

Install PyTorch 1.13.1:

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install PyG & Pytorch-cluster :

```
conda install pyg -c pyg
conda install pytorch-cluster -c pyg
```

Install PyTorch Scatter and PyTorch Sparse:

```
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

## Datasets

We provide the pre-processed datasets for pretraining , training and evaluating protein representation learning:

1. [Protein Fold](https://drive.google.com/file/d/1vEdezR5L44swsw09WFnaA5zFuA1ZEXHI/view?usp=sharing) &emsp; 2. [Enzyme Commission Number](https://drive.google.com/file/d/1VEIyBSJbRf9x6k_w4Tqy5SC0G6NWWSWl/view?usp=sharing)
