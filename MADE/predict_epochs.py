import torch
import os
import numpy as np
from .made import MADE
from .datasets.data_loaders import get_data, get_data_loaders
from .utils.test import test_made
import sys
import os

def predict_epochs(feat_dir, model_dir, made_dir, TRAIN, TEST, DEVICE, epoch):

    # --------- 设置参数 ----------
    model_name = 'made'  # 模型名称，可以是 'MAF' 或 'MADE'
    dataset_name = 'myData'  # 数据集名称
    train_type = TRAIN  # 训练集类型
    test_type = TEST  # 测试集类型
    batch_size = 1024  # 批处理大小
    n_mades = 5  # 使用的MADE模型数量
    hidden_dims = [512]  # 隐藏层维度
    lr = 1e-4  # 学习率
    random_order = False  # 是否使用随机顺序
    patience = 30  # 用于早停的耐心次数
    seed = 290713  # 随机种子
    cuda_device = int(DEVICE) if DEVICE != 'None' else None  # CUDA设备ID，如果没有指定设备则为None
    # -----------------------------------

    # 获取数据集
    data = get_data(dataset_name, feat_dir, train_type, test_type)
    train = torch.from_numpy(data.train.x)  # 将训练数据转换为Tensor
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
    # 获取模型输入维度
    n_in = data.n_dims
    # 设置模型保存文件的名称格式
    save_name = f"{model_name}_{dataset_name}_{train_type}_{'_'.join(str(d) for d in hidden_dims)}.pt"

    # 加载预训练的模型
    model = torch.load(os.path.join(model_dir, 'epochs_' + save_name))

    # 如果指定了CUDA设备，设置设备并将模型转移到CUDA上
    if cuda_device is not None:
        torch.cuda.set_device(cuda_device)
        model = model.cuda()

    # 使用测试集对模型进行评估，获取负对数似然损失（Negative Log-Likelihood Loss）
    neglogP = test_made(model, test_loader, cuda_device)

    # 将测试结果保存到指定的文件中
    with open(os.path.join(made_dir, '%s_%sMADE_%d' % (test_type, train_type, epoch)), 'w') as fp:
        for neglogp in neglogP:
            fp.write(str(float(neglogp)) + '\n')  # 将每个测试样本的负对数似然损失写入文件
