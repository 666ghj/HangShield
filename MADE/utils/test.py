import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal
from typing import Optional

# 测试MAF（Masked Autoregressive Flow）模型的性能
def test_maf(model, train, test_loader):
    model.eval()  # 将模型设置为评估模式
    test_loss = []  # 用于存储每个测试样本的损失值
    _, _ = model.forward(train)  # 在训练数据上运行一次模型的前向传播
    
    # 不计算梯度的情况下对测试数据进行评估
    with torch.no_grad():
        for batch in test_loader:
            u, log_det = model.forward(batch.float())  # 进行前向传播计算，获取u和log_det

            # 计算负对数似然损失
            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)  # 计算u的平方和
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)  # 加上高斯分布常数项
            negloglik_loss -= log_det  # 减去log_det项

            test_loss.extend(negloglik_loss)  # 将计算出的损失添加到损失列表中
    
    N = len(test_loss)  # 测试样本数量
    print(
        "Test loss: {:.4f} +/- {:.4f}".format(
            np.mean(test_loss), 2 + np.std(test_loss) / np.sqrt(N)  # 打印测试损失的均值和标准误差
        )
    )

# 测试MADE（Masked Autoencoder for Distribution Estimation）模型的性能
def test_made(model, test_loader, cuda_device: Optional[int] = None):
    if cuda_device is not None:
        torch.cuda.set_device(cuda_device)  # 设置当前的CUDA设备
    model.eval()  # 将模型设置为评估模式
    neglogP = []  # 用于存储每个测试样本的负对数似然值

    # 不计算梯度的情况下对测试数据进行评估
    with torch.no_grad():
        for batch in test_loader:
            if cuda_device is None:
                input = batch.float()  # 将输入数据转换为浮点类型
                out = model.forward(input)  # 前向传播计算输出
                mu, logp = torch.chunk(out, 2, dim=1)  # 将输出分为均值mu和logp

                u = (input - mu) * torch.exp(0.5 * logp)  # 计算标准化后的u

                # 计算负对数似然损失
                negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                negloglik_loss += 0.5 * input.shape[1] * np.log(2 * math.pi)
                negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

                neglogP.extend(negloglik_loss)  # 将损失添加到列表中
            else:
                input = batch.float().cuda()  # 将输入数据转换为CUDA浮点类型
                out = model.forward(input)  # 前向传播计算输出
                mu, logp = torch.chunk(out, 2, dim=1)  # 将输出分为均值mu和logp

                u = (input - mu) * torch.exp(0.5 * logp).cuda()  # 计算标准化后的u

                # 计算负对数似然损失
                negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                negloglik_loss += 0.5 * input.shape[1] * np.log(2 * math.pi)
                negloglik_loss -= 0.5 * torch.sum(logp, dim=1)
                neglogP.extend(negloglik_loss.cpu())  # 将损失移回CPU并添加到列表中
    
    print(len(neglogP))  # 打印测试样本数量
    return neglogP  # 返回计算得到的负对数似然损失列表
