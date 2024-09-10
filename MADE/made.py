from typing import List, Optional
import numpy as np
from numpy.random import permutation, randint
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import ReLU

# 这个实现的 MADE 是从 https://github.com/e-hulten/made 复制的。
"""
    NOTE：MADE（用于分布估计的掩码自动编码器）模型主要用于密度估计任务。以下是它的输入和输出的概述：
    输入：
        D维向量：MADE模型的输入是一个D维的向量。这个向量表示数据集中的一个样本，其中每个维度可能对应一个特征或变量。
    输出：
        同样维度的输出向量：
        模型的输出也是一个D维的向量。对于每个维度X_i，模型预测一个条件概率分布P(X_i|X_{<i}),即在前面所有维度的值已知情况下，第i个维度的值的概率。
    MADE模型通过使用掩码机制来保证在计算每个维度的条件概率时，只依赖于先前的维度，从而实现高效的密度估计。
"""


class MaskedLinear(nn.Linear):
    """具有遮掩元素的线性变换。y = x.dot(mask*W.T) + b"""

    def __init__(self, n_in: int, n_out: int, bias: bool = True, cuda_device: Optional[int] = None) -> None:
        """
        参数:
            n_in: 输入样本的大小。
            n_out: 输出样本的大小。
            bias: 是否包括加性偏差项。默认: True。
        """
        super().__init__(n_in, n_out, bias)  # 调用父类的初始化方法
        self.mask = None  # 初始化掩码为None
        self.cuda_device = cuda_device  # 保存CUDA设备的ID

    def initialise_mask(self, mask: Tensor):
        """内部方法，用于初始化掩码。"""
        if self.cuda_device is None:
            self.mask = mask  # 如果没有指定CUDA设备，则直接使用传入的掩码
        else:
            torch.cuda.set_device(self.cuda_device)  # 设置当前CUDA设备
            self.mask = mask.cuda()  # 将掩码移到GPU

    def set_device(self, device):
        torch.cuda.set_device(device)  # 设置当前CUDA设备
        self.cuda_device = device  # 保存当前设备的ID
        self.mask = self.mask.cpu().cuda()  # 将掩码转移到新设置的CUDA设备上

    def forward(self, x: Tensor) -> Tensor:
        """应用掩码后的线性变换。"""
        return F.linear(x, self.mask * self.weight, self.bias)  # 应用掩码，并执行线性变换


class MADE(nn.Module):
    def __init__(
        self,
        n_in: int,
        hidden_dims: List[int],
        gaussian: bool = False,
        random_order: bool = False,
        seed: Optional[int] = None,
        cuda_device: Optional[int] = None,
    ) -> None:
        """初始化MADE模型。

        参数:
            n_in: 输入的大小。
            hidden_dims: 隐藏层的尺寸列表。
            gaussian: 是否使用高斯MADE。默认: False。
            random_order: 是否使用随机顺序。默认: False。
            seed: numpy的随机种子。默认: None。
        """
        super().__init__()
        np.random.seed(seed)  # 设置随机种子
        self.n_in = n_in  # 输入的维度
        self.n_out = 2 * n_in if gaussian else n_in  # 如果是高斯MADE，输出维度是输入维度的两倍
        self.hidden_dims = hidden_dims  # 隐藏层的维度
        self.random_order = random_order  # 是否使用随机顺序
        self.gaussian = gaussian  # 是否使用高斯分布
        self.masks = {}  # 存储每一层的掩码
        self.mask_matrix = []  # 存储掩码矩阵
        self.layers = []  # 存储模型的各个层
        self.cuda_device = cuda_device  # 保存CUDA设备的ID

        if self.cuda_device is not None:
            torch.cuda.set_device(self.cuda_device)  # 设置当前CUDA设备

        # 定义每层的尺寸列表
        dim_list = [self.n_in, *hidden_dims, self.n_out]

        # 创建各个层和激活函数
        for i in range(len(dim_list) - 2):
            if self.cuda_device is None:
                self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]))  # 添加带掩码的线性层
                self.layers.append(ReLU())  # 添加ReLU激活函数
            else:
                self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1], 
                    cuda_device=self.cuda_device).cuda())  # 在CUDA上创建带掩码的线性层
                self.layers.append(ReLU().cuda())  # 在CUDA上创建ReLU激活函数

        # 从最后一层隐藏层到输出层
        if self.cuda_device is None:
            self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))  # 添加最后一层带掩码的线性层
        else:
            self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1], 
                cuda_device=self.cuda_device).cuda())  # 在CUDA上创建最后一层带掩码的线性层

        # 创建模型
        self.model = nn.Sequential(*self.layers)

        # 生成掩码矩阵
        self._create_masks()

    def set_device(self, device):
        torch.cuda.set_device(device)  # 设置当前CUDA设备
        self.cuda_device = device  # 保存当前设备的ID
        for model in self.model:
            if isinstance(model, MaskedLinear):
                model.set_device(device)  # 为每一层带掩码的线性层设置设备
            model = model.cpu().cuda()  # 将模型从CPU转移到CUDA上

    def forward(self, x: Tensor) -> Tensor:
        """前向传播。"""
        if self.gaussian:
            # 如果是高斯分布，直接返回输出的均值和方差
            res = self.model(x)
            return res
        else:
            # 如果是Bernoulli分布，经过sigmoid函数将输出限制在(0,1)之间
            return torch.sigmoid(self.model(x))

    def _create_masks(self) -> None:
        """生成隐藏层的掩码。"""
        L = len(self.hidden_dims)  # 隐藏层的数量
        D = self.n_in  # 输入的维度

        # 根据输入的随机顺序设置掩码
        self.masks[0] = permutation(D) if self.random_order else np.arange(D)

        # 设置隐藏层的连接掩码
        for l in range(L):
            low = self.masks[l].min()  # 获取上一层掩码的最小值
            size = self.hidden_dims[l]  # 当前层的维度
            self.masks[l + 1] = randint(low=low, high=D - 1, size=size)  # 随机生成当前层的掩码

        # 为输出层添加掩码。输出层的顺序与输入层相同。
        self.masks[L + 1] = self.masks[0]

        # 生成输入 -> 隐藏层 -> 输出层的掩码矩阵
        for i in range(len(self.masks) - 1):
            m = self.masks[i]
            m_next = self.masks[i + 1]
            # 初始化掩码矩阵
            M = torch.zeros(len(m_next), len(m))
            for j in range(len(m_next)):
                # 通过广播将m_next[j]与m的每个元素进行比较
                M[j, :] = torch.from_numpy((m_next[j] >= m).astype(int))
            # 将生成的掩码矩阵添加到列表中
            self.mask_matrix.append(M)

        # 如果是高斯分布，将输出单元数量加倍(均值和方差)
        # 成对相同的掩码
        if self.gaussian:
            m = self.mask_matrix.pop(-1)
            self.mask_matrix.append(torch.cat((m, m), dim=0))

        # 用生成的掩码初始化MaskedLinear层
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))  # 初始化掩码
