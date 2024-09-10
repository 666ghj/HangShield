import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal
from typing import List, Optional

# 训练MAF模型的一个epoch
def train_one_epoch_maf(model, epoch, optimizer, train_loader):
    model.train()  # 将模型设置为训练模式
    train_loss = 0  # 初始化训练损失

    # 遍历训练数据的每个batch
    for batch in train_loader:
        u, log_det = model.forward(batch.float())  # 前向传播，得到u和log_det

        # 计算负对数似然损失
        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)  # 计算u的平方和的负对数似然项
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)  # 添加高斯分布的常数项
        negloglik_loss -= log_det  # 减去log_det项
        negloglik_loss = torch.mean(negloglik_loss)  # 计算平均损失

        negloglik_loss.backward()  # 反向传播计算梯度
        train_loss += negloglik_loss.item()  # 累积当前batch的损失
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()  # 清空梯度

    # 计算平均损失
    avg_loss = np.sum(train_loss) / len(train_loader)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))  # 打印当前epoch的平均损失
    return avg_loss  # 返回平均损失


# 训练MADE模型的一个epoch
def train_one_epoch_made(model, epoch, optimizer, train_loader, cuda_device: Optional[int] = None):
    if cuda_device is not None:
        torch.cuda.set_device(cuda_device)  # 设置当前的CUDA设备
    model.train()  # 将模型设置为训练模式
    train_loss = []  # 初始化损失列表

    # 遍历训练数据的每个batch
    for batch in train_loader:
        if cuda_device is None:
            out = model.forward(batch.float())  # 前向传播，得到模型输出
            mu, logp = torch.chunk(out, 2, dim=1)  # 将输出拆分为均值(mu)和对数方差(logp)
            u = (batch - mu) * torch.exp(0.5 * logp)  # 标准化数据

            # 计算负对数似然损失
            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

            negloglik_loss = torch.mean(negloglik_loss)  # 计算平均损失
            train_loss.append(negloglik_loss)  # 将损失添加到损失列表中

            negloglik_loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad()  # 清空梯度
        else:
            input = batch.float().cuda()  # 将输入数据转移到GPU
            out = model.forward(input)  # 前向传播，得到模型输出
            mu, logp = torch.chunk(out, 2, dim=1)  # 将输出拆分为均值(mu)和对数方差(logp)
            u = (input - mu) * torch.exp(0.5 * logp).cuda()  # 标准化数据

            # 计算负对数似然损失
            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * input.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

            negloglik_loss = torch.mean(negloglik_loss)  # 计算平均损失
            train_loss.append(negloglik_loss.cpu().detach().numpy())  # 将损失添加到列表中，并转移到CPU

            negloglik_loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad()  # 清空梯度

    N = len(train_loader)  # 获取训练集的batch数量
    avg_loss = np.sum(train_loss) / N  # 计算平均损失

    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))  # 打印当前epoch的平均损失
    return avg_loss  # 返回平均损失
