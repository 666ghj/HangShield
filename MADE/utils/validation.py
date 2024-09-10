import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal
from typing import List, Optional

# 验证MAF模型的性能
def val_maf(model, train, val_loader):
    model.eval()  # 将模型设置为评估模式
    val_loss = []  # 初始化验证损失列表
    _, _ = model.forward(train.float())  # 在训练数据上运行一次前向传播，更新模型状态
    
    # 遍历验证数据集，计算每个批次的负对数似然损失
    for batch in val_loader:
        u, log_det = model.forward(batch.float())  # 前向传播计算u和log_det
        # 计算负对数似然损失
        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        val_loss.extend(negloglik_loss.tolist())  # 将损失添加到列表中

    N = len(val_loader.dataset)  # 获取验证集的总样本数量
    loss = np.sum(val_loss) / N  # 计算平均损失
    print(
        "Validation loss: {:.4f} +/- {:.4f}".format(
            loss, 2 * np.std(val_loss) / np.sqrt(N)  # 打印验证损失的均值和标准误差
        )
    )
    return loss  # 返回验证集的平均损失

# 验证MADE模型的性能
def val_made(model, val_loader, cuda_device: Optional[int] = None):
    if cuda_device is not None:
        torch.cuda.set_device(cuda_device)  # 设置当前的CUDA设备
    model.eval()  # 将模型设置为评估模式
    val_loss = []  # 初始化验证损失列表
    
    # 不计算梯度的情况下进行前向传播和损失计算
    with torch.no_grad():
        for batch in val_loader:
            if cuda_device is None:
                out = model.forward(batch.float())  # 前向传播计算输出
                mu, logp = torch.chunk(out, 2, dim=1)  # 将输出分割为mu和logp
                u = (batch - mu) * torch.exp(0.5 * logp)  # 计算u

                # 计算负对数似然损失
                negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
                negloglik_loss -= 0.5 * torch.sum(logp, dim=1)
                negloglik_loss = torch.mean(negloglik_loss)

                val_loss.append(negloglik_loss.cpu())  # 将损失添加到列表中
            else:
                input = batch.float().cuda()  # 将输入数据移到CUDA设备上
                out = model.forward(input)  # 前向传播计算输出
                mu, logp = torch.chunk(out, 2, dim=1)  # 将输出分割为mu和logp
                u = (input - mu) * torch.exp(0.5 * logp).cuda()  # 计算u

                # 计算负对数似然损失
                negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                negloglik_loss += 0.5 * input.shape[1] * np.log(2 * math.pi)
                negloglik_loss -= 0.5 * torch.sum(logp, dim=1)
                negloglik_loss = torch.mean(negloglik_loss)

                val_loss.append(negloglik_loss.cpu())  # 将损失移回CPU并添加到列表中

    N = len(val_loader)  # 获取验证批次数量
    loss = np.sum(val_loss) / N  # 计算平均损失
    print(
        "Validation loss: {:.4f} +/- {:.4f}".format(
            loss, 2 * np.std(val_loss) / np.sqrt(N)  # 打印验证损失的均值和标准误差
        )
    )
    return loss  # 返回验证集的平均损失
