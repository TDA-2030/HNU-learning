#!/usr/bin/env python

import time
from typing import Tuple
from tqdm import tqdm
from utils import torch_plots

import torch
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as transforms

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1, )):

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t().view(-1).numpy()
    print("^^^", pred)
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        print("000", pred)
        print(target.view(1, -1))
        print(target.view(1, -1).expand_as(pred))
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        print("===", correct)


tt = torch.Tensor([[0.1, 0.1, 0.3, 0.11],
                    [0.1, 0.2, 0.3, 0.41],
                    [0.5, 0.1, 0.09, 0.11]])
tar = torch.Tensor([2, 2, 0])
accuracy(tt, tar, topk=(1,))
exit()

dic = ['a', 'b', 'c', 'd', 'e']
pbar = tqdm(dic, ncols=100, unit="img")
for i in pbar:
    pbar.set_description('Processing '+i)
    time.sleep(0.1)



def DatasetGet(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # 定义对数据的预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])

    # 训练集
    trainset = tv.datasets.CIFAR100(root='~/work/data',
                                   train=True,
                                   download=True,
                                   transform=transform)

    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=1)

    # 测试集
    testset = tv.datasets.CIFAR100('~/work/data',
                                  train=False,
                                  download=False,
                                  transform=transform)

    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1)
    return (trainloader, testloader)



trainloader, testloader = DatasetGet(32)

for i, data in enumerate(trainloader, 0):
    # 输入数据
    inputs, labels = data
    print(labels)
    s = str(labels.numpy())
    torch_plots.plot_images(inputs, [], s)
    break