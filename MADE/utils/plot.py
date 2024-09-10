import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.distributions import MultivariateNormal

# 从MAF（Masked Autoregressive Flow）模型中采样数据
def sample_digits_maf(model, epoch, random_order=False, seed=None, test=False):
    model.eval()  # 将模型设置为评估模式
    n_samples = 80  # 设定采样数量为80

    # 如果提供了种子，则设置随机数生成种子，以保证结果的可重复性
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 如果设置了随机顺序，生成随机排列的顺序，否则按顺序排列
    if random_order is True:
        np.random.seed(seed)
        order = np.random.permutation(784)  # 随机排列
    else:
        order = np.arange(784)  # 按顺序排列

    # 从标准正态分布中采样u向量，u的形状为(n_samples, 784)
    u = torch.zeros(n_samples, 784).normal_(0, 1)
    # 创建一个多元正态分布对象，用于计算对数概率
    mvn = MultivariateNormal(torch.zeros(28 * 28), torch.eye(28 * 28))
    log_prob = mvn.log_prob(u)  # 计算采样数据的对数概率
    samples, log_det = model.backward(u)  # 通过模型反向计算得到样本

    # 对样本进行sigmoid变换，并调整值的范围
    samples = (torch.sigmoid(samples) - 1e-6) / (1 - 2e-6)
    samples = samples.detach().cpu().view(n_samples, 28, 28)  # 将样本重塑为28x28的矩阵

    # 创建一个8行10列的子图，展示采样结果
    fig, axes = plt.subplots(ncols=10, nrows=8)
    ax = axes.ravel()
    for i in range(n_samples):
        ax[i].imshow(np.transpose(samples[i], (0, 1)), cmap="gray", interpolation="none")  # 显示图像
        ax[i].axis("off")  # 隐藏坐标轴
        ax[i].set_xticklabels([])  # 去除x轴标签
        ax[i].set_yticklabels([])  # 去除y轴标签
        ax[i].set_frame_on(False)  # 去除边框

    # 如果结果保存的目录不存在，则创建它
    if not os.path.exists("gif_results"):
        os.makedirs("gif_results")

    # 根据是否处于测试模式，设置保存路径
    if test is False:
        save_path = "gif_results/samples_gaussian_" + str(epoch) + ".png"
    else:
        save_path = "figs/samples_gaussian_" + str(epoch) + ".png"

    # 调整子图间的间距，并保存生成的图像
    fig.subplots_adjust(wspace=-0.35, hspace=0.065)
    plt.gca().set_axis_off()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()  # 关闭绘图窗口

# 绘制训练损失和验证损失随时间变化的曲线
def plot_losses(epochs, train_losses, val_losses, title=None):
    sns.set(style="white")  # 设置绘图风格
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=[10, 5], sharey=True, sharex=True, dpi=400)

    train = pd.Series(train_losses).astype(float)  # 转换训练损失为浮点数序列
    val = pd.Series(val_losses).astype(float)  # 转换验证损失为浮点数序列
    train.index += 1  # 将索引从1开始
    val.index += 1

    axes = sns.lineplot(data=train, color="gray", label="Training loss")  # 绘制训练损失曲线
    axes = sns.lineplot(data=val, color="orange", label="Validation loss")  # 绘制验证损失曲线

    axes.set_ylabel("Negative log-likelihood")  # 设置y轴标签为负对数似然
    axes.legend(frameon=False, prop={"size": 14}, fancybox=False, handletextpad=0.5, handlelength=1)  # 设置图例
    axes.set_ylim(1250, 1600)  # 设置y轴范围
    axes.set_xlim(0, 50)  # 设置x轴范围
    axes.set_title(title) if title is not None else axes.set_title(None)  # 设置标题
    if not os.path.exists("plots"):  # 如果保存目录不存在，则创建
        os.makedirs("plots")
    save_path = "plots/train_plots" + str(epochs[-1]) + ".pdf"  # 设置保存路径
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)  # 保存图像
    plt.close()  # 关闭绘图窗口
