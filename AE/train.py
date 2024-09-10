from .model import LSTM_AE_GMM
import numpy as np
import torch
import torch.nn as nn
import sys
import os
from tqdm import tqdm

# 设置批处理大小和最大训练轮数
batch_size = 128
Max_epochs = 1000

def main(data_dir, model_dir, device):
    # 加载训练数据集(BEN、RAT、PST、BDT、SPT、DLT)
    ben = np.load(os.path.join(data_dir, 'BEN.npy'))  # 加载BEN数据
    rat = np.load(os.path.join(data_dir, 'RAT.npy'))
    pst = np.load(os.path.join(data_dir, 'PST.npy'))
    bdt = np.load(os.path.join(data_dir, 'BDT.npy'))
    spt = np.load(os.path.join(data_dir, 'SPT.npy'))
    dlt = np.load(os.path.join(data_dir, 'DLT.npy'))

    # 计算每个数据集的训练集长度（80%用于训练）
    ben_len = int(ben.shape[0] * 0.8)
    rat_len = int(rat.shape[0] * 0.8)
    pst_len = int(pst.shape[0] * 0.8)
    bdt_len = int(bdt.shape[0] * 0.8)
    spt_len = int(spt.shape[0] * 0.8)
    dlt_len = int(dlt.shape[0] * 0.8)
    
    # 保存训练数据（覆盖原数据）
    np.save(os.path.join(data_dir, 'BEN.npy'), ben[:ben_len, :])
    np.save(os.path.join(data_dir, 'RAT.npy'), rat[:rat_len, :])
    np.save(os.path.join(data_dir, 'PST.npy'), pst[:pst_len, :])
    np.save(os.path.join(data_dir, 'BDT.npy'), bdt[:bdt_len, :])
    np.save(os.path.join(data_dir, 'SPT.npy'), spt[:spt_len, :])
    np.save(os.path.join(data_dir, 'DLT.npy'), dlt[:dlt_len, :])
    # 将剩下的20%整合起来保存为test.npy
    np.save(os.path.join(data_dir, 'test.npy'), np.concatenate([ben[ben_len:], rat[rat_len:], pst[pst_len:], bdt[bdt_len:], spt[spt_len:], dlt[dlt_len:]], axis=0))
    print("原始npy文件已划分为训练集和测试集，训练集已覆盖源文件，测试集已整合保存为test.npy")
    
    # 打印每个训练用数据集的长度
    print("BEN、RAT、PST、BDT、SPT、DLT训练用数据集长度依次为:", ben_len, rat_len, pst_len, bdt_len, spt_len, dlt_len)

    # 将各个数据集的训练数据拼接成一个整体训练集, 只取前50个特征，无监督自编码最后一个标签用不到
    train_data = np.concatenate(
        [ben[:ben_len, :50], rat[:rat_len, :50], pst[:pst_len, :50], bdt[:bdt_len, :50], spt[:spt_len, :50], dlt[:dlt_len, :50]], axis=0)

    # 打乱训练数据的顺序
    np.random.shuffle(train_data)
    
    # 获取训练数据的大小和输入特征的维度
    total_size, input_size = train_data.shape
    print("训练数据总样本数：", total_size)  # 打印训练数据的总样本数
    device_id = int(device)  # 获取当前使用的设备ID
    print("当前使用的设备id：", device_id)  # 打印设备ID
    torch.cuda.set_device(device_id)  # 设置当前CUDA设备

    # 计算最大训练轮数
    max_epochs = Max_epochs * 200 // total_size
    print("最大训练轮数：", max_epochs)  # 打印最大训练轮数

    # 初始化LSTM_AE_GMM模型
    dagmm = LSTM_AE_GMM(
        input_size=input_size,  # 输入特征的维度
        max_len=2000,  # 最大序列长度
        emb_dim=32,  # 嵌入维度
        hidden_size=8,  # 隐藏层维度
        dropout=0.2,  # Dropout概率
        est_hidden_size=64,  # 估计模块隐藏层维度
        est_output_size=8,  # 估计模块输出维度
        device=device_id,  # 使用的CUDA设备ID
    ).cuda()  # 将模型加载到CUDA设备

    dagmm.train_mode()  # 将模型设置为训练模式
    optimizer = torch.optim.Adam(dagmm.parameters(), lr=1e-2)  # 使用Adam优化器

    # 训练循环
    for epoch in range(max_epochs):
        for batch in tqdm(range(total_size // batch_size + 1), desc=f"Epoch {epoch}/{max_epochs-1}"):
            if batch * batch_size >= total_size:
                break  # 如果超出总样本数，跳出循环
            optimizer.zero_grad()  # 清空梯度
            input = train_data[batch_size * batch : batch_size * (batch + 1)]  # 获取当前批次的数据
            loss = dagmm.loss(torch.Tensor(input).long().cuda())  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
        print('epoch:', epoch, 'loss:', loss)  # 打印当前轮次的损失
        if (epoch + 1) % 10 == 0:  # 每10轮保存一次模型
            dagmm.to_cpu()  # 将模型转移到CPU
            dagmm = dagmm.cpu()
            torch.save(dagmm, os.path.join(model_dir, 'gru_ae_ma.pkl'))  # 保存模型
            dagmm.to_cuda(device_id)  # 将模型重新转移到CUDA设备
            dagmm = dagmm.cuda()

if __name__ == '__main__':
    data_dir = '../data/data'  # 数据集目录
    model_dir = '../data/model'  # 模型保存目录
    device = 0  # 设备ID（默认为0）
    main(data_dir, model_dir, device)  # 调用主函数，开始训练
