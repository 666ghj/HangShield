import torch
import numpy as np
from .made import MADE
from .datasets.data_loaders import get_data, get_data_loaders
from .utils.train import train_one_epoch_made
from .utils.validation import val_made
import sys
import os
from .predict_epochs import predict_epochs
import re

# 训练MADE模型并在训练过程中记录损失
def main(feat_dir, model_dir, made_dir, TRAIN, DEVICE, MINLOSS):

    # --------- 设置参数 ----------
    model_name = 'made'  # 模型名称，可以是 'MAF' 或 'MADE'
    dataset_name = 'myData'  # 数据集名称
    train_type = TRAIN  # 训练类型
    batch_size = 128  # 批处理大小
    hidden_dims = [512]  # 隐藏层维度列表
    lr = 1e-4  # 学习率
    random_order = False  # 是否随机输入顺序
    patience = 50  # 用于早停的耐心值
    min_loss = int(MINLOSS)  # 最小损失值
    seed = 290713  # 随机种子
    cuda_device = int(DEVICE) if DEVICE != 'None' else None  # 使用的CUDA设备ID
    plot = True  # 是否绘制图表
    max_epochs = 2000  # 最大训练轮数
    # -----------------------------------

    # 清空模型目录中的所有文件
    for filename in os.listdir(made_dir):
        file_path = os.path.join(made_dir, filename)
        if os.name == 'nt':  # 如果是Windows操作系统
            os.system(f'del /F /Q "{file_path}"')
        else:  # 如果是Linux/Unix操作系统
            os.system(f'rm -f "{file_path}"')
            
    # 获取数据集
    data = get_data(dataset_name, feat_dir, train_type, train_type)
    train = torch.from_numpy(data.train.x)  # 将训练数据转换为PyTorch张量
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
    # 初始化模型
    n_in = data.n_dims  # 输入特征的维度
    model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True, cuda_device=cuda_device)

    # 获取优化器
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # 如果指定了CUDA设备，则将模型转移到该设备
    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
        model = model.cuda()

    # 格式化保存模型文件的名称
    save_name = f"{model_name}_{dataset_name}_{train_type}_{'_'.join(str(d) for d in hidden_dims)}.pt"
    # 初始化列表用于记录每个epoch的损失和绘图
    epochs_list = []
    train_losses = []
    val_losses = []
    # 初始化早停相关的变量
    i = 0
    max_loss = np.inf  # 设置初始最大损失为无穷大
    # 训练循环
    for epoch in range(1, max_epochs):
        # 训练一个epoch并返回训练损失
        train_loss = train_one_epoch_made(model, epoch, optimiser, train_loader, cuda_device)
        # 验证模型并返回验证损失
        val_loss = val_made(model, val_loader, cuda_device)

        # 记录当前epoch的编号、训练损失和验证损失
        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 每10个epoch保存一次模型并进行预测
        if epoch % 10 == 0:
            model = model.cpu()
            torch.save(
                model, os.path.join(model_dir, 'epochs_' + save_name)
            )  # 保存模型
            if cuda_device != None:
                model = model.cuda()

            # 使用保存的模型对当前数据进行预测
            predict_epochs(feat_dir, model_dir, made_dir, TRAIN, 'be', DEVICE, epoch)
            predict_epochs(feat_dir, model_dir, made_dir, TRAIN, 'ma', DEVICE, epoch)

        # 早停策略：如果验证损失有改进且训练损失大于最小损失，则保存模型并重置耐心计数器
        if val_loss < max_loss and train_loss > min_loss:
            i = 0
            max_loss = val_loss
            model = model.cpu()
            torch.save(
                model, os.path.join(model_dir, save_name)
            )  # 保存改进的模型
            if cuda_device != None:
                model = model.cuda()
        else:
            i += 1  # 否则增加耐心计数器

        # 打印当前的耐心计数器，如果达到上限则终止训练
        if i < patience:
            print("Patience counter: {}/{}".format(i, patience))
        else:
            print("Patience counter: {}/{}\n Terminate training!".format(i, patience))
            break  # 终止训练

