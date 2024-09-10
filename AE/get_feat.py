import sys
from .model import LSTM_AE_GMM
import numpy as np
import torch
import torch.nn as nn
import os

# 设置批处理大小
batch_size = 128

def main(data_dir, model_dir, feat_dir, data_type, device):
    # 将设备号转为整数类型，如果设备号为 'None'，则设置为 None
    device = int(device) if device != 'None' else None

    # 加载测试数据集，文件名由 data_type 决定
    test_data_label = np.load(os.path.join(data_dir, data_type + '.npy'))
    
    # 取出前50列作为测试数据，最后一列作为测试标签
    test_data = test_data_label[:, :50]
    test_label = test_data_label[:, -1]
    
    # 获取测试数据的大小
    total_size, _ = test_data.shape

    # 设置当前 CUDA 设备
    device_id = int(device)
    torch.cuda.set_device(device_id)

    # 加载预训练的 LSTM_AE_GMM 模型
    try:
        dagmm = torch.load(os.path.join(model_dir, 'gru_ae_ma.pkl'))
    except:
        print("模型加载失败，请检查模型是否存在")
        sys.exit(1)

    # 将模型转移到指定的 CUDA 设备
    dagmm.to_cuda(device_id)
    dagmm = dagmm.cuda()

    # 设置模型为测试模式（停止dropout等操作）
    dagmm.test_mode()
    
    feature = []  # 初始化特征存储列表
    # 遍历测试数据集，按批次提取特征
    for batch in range(total_size // batch_size + 1):
        if batch * batch_size >= total_size:
            break  # 如果超过总样本数，跳出循环
        
        # 获取当前批次的数据
        input = test_data[batch_size * batch : batch_size * (batch + 1)]
        
        # 通过模型提取特征，转为 PyTorch 张量并放入 CUDA
        output = dagmm.feature(torch.Tensor(input).long().cuda())
        
        # 将提取的特征从 CUDA 设备中移回 CPU 并添加到特征列表中
        feature.append(output.detach().cpu())

    # 将所有批次的特征拼接在一起，形成完整的特征矩阵
    feature = torch.cat(feature, dim=0).numpy()
    
    # 将特征和对应的标签拼接在一起，形成最终的输出
    feature = np.concatenate([feature, test_label[:, None]], axis=1)
    
    # 保存提取的特征和标签到指定目录，文件名由 data_type 决定
    np.save(os.path.join(feat_dir, data_type + '.npy'), feature)

