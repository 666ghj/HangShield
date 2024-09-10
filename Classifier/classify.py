# -*- coding:utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .model import MLP  # 从model模块导入MLP模型
import sys, os
import numpy as np
from tqdm import tqdm  # 导入进度条工具

from .loss import loss_coteaching  # 从loss模块导入loss_coteaching函数

# 超参数设置
batch_size = 128  # 批处理大小
learning_rate = 1e-3  # 学习率
epochs = 100  # 训练轮数
num_gradual = 10  # 学习率逐渐下降的轮数
forget_rate = 0.1  # 忘记率
exponent = 1  # 指数衰减系数
rate_schedule = np.ones(epochs) * forget_rate  # 生成一个全为forget_rate的数组，长度为epochs
rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)  # 在前num_gradual个周期内生成线性衰减的忘记率

def accuracy(logit, target):
    """计算指定k值的精度"""
    output = F.softmax(logit, dim=1)  # 对logit进行softmax变换
    batch_size = target.size(0)  # 获取批处理大小

    _, pred = output.topk(1, 1, True, True)  # 获取top1预测值
    pred = pred.t()  # 转置预测值
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 计算预测正确的数量

    correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)  # 计算正确数量
    return correct_k.mul_(100.0 / batch_size)  # 返回精度百分比

# 训练模型
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, device):
    
    train_total1=0
    train_correct1=0 
    train_total2=0
    train_correct2=0 

    for i, data_labels in enumerate(train_loader):
        
        feats = data_labels[:, :-1].to(dtype=torch.float32)  # 提取特征并转换为浮点数
        labels = data_labels[:, -1].to(dtype=int)  # 提取标签并转换为整数
        if device != None:
            torch.cuda.set_device(device)  # 设置CUDA设备
            feats = feats.cuda()  # 将特征移动到GPU
            labels = labels.cuda()  # 将标签移动到GPU
    
        logits1 = model1(feats)  # 使用模型1进行前向传播
        prec1 = accuracy(logits1, labels)  # 计算模型1的精度
        train_total1 += 1
        train_correct1 += prec1

        logits2 = model2(feats)  # 使用模型2进行前向传播
        prec2 = accuracy(logits2, labels)  # 计算模型2的精度
        train_total2 += 1
        train_correct2 += prec2
        loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch])  # 计算损失

        optimizer1.zero_grad()  # 清空梯度
        loss_1.backward()  # 反向传播
        optimizer1.step()  # 更新模型1的参数
        optimizer2.zero_grad()  # 清空梯度
        loss_2.backward()  # 反向传播
        optimizer2.step()  # 更新模型2的参数

    train_acc1=float(train_correct1)/float(train_total1)  # 计算模型1的训练精度
    train_acc2=float(train_correct2)/float(train_total2)  # 计算模型2的训练精度
    return train_acc1, train_acc2  # 返回两个模型的训练精度

# 预测未知流量数据的标签
def predict(test_loader, model, device, alpha=0.5):

    preds = []
    for i, data in enumerate(test_loader):
        
        # 前向传播
        feats = data.to(dtype=torch.float32)  # 提取特征并转换为浮点数
        
        if device is not None:
            torch.cuda.set_device(device)  # 设置CUDA设备
            feats = feats.cuda()  # 将特征移动到GPU
        
        logits = model(feats)  # 使用模型进行前向传播
        outputs = F.softmax(logits, dim=1)  # 计算softmax输出
        
        # INFO：这里保存的是概率值，而不是标签，方便多个模型的融合判断概率最大的类别
        # 保存概率而不是标签
        preds.append(outputs[:, 1].detach().cpu().numpy())  # 保存概率值

    return np.concatenate(preds, axis=0)  # 返回所有预测的概率结果

def main(feat_dir, model_dir, result_dir, TRAIN, cuda_device, parallel=5):
    
    cuda_device = int(cuda_device)
    # 获取原始训练集
    be = np.load(os.path.join(feat_dir, 'be_corrected.npy'))[:, :32]  # 加载并截取前32维的be数据
    ma = np.load(os.path.join(feat_dir, 'ma_corrected.npy'))[:, :32]  # 加载并截取前32维的ma数据
    be_shape = be.shape[0]
    ma_shape = ma.shape[0]

    for index in range(parallel):

        # 将合成的流量特征添加到原始训练集中
        be_gen = np.load(os.path.join(feat_dir, 'be_%s_generated_GAN_%d.npy' % (TRAIN, index)))  # 加载be合成数据
        ma_gen1 = np.load(os.path.join(feat_dir, 'ma_%s_generated_GAN_1_%d.npy' % (TRAIN, index)))  # 加载ma合成数据1
        ma_gen2 = np.load(os.path.join(feat_dir, 'ma_%s_generated_GAN_2_%d.npy' % (TRAIN, index)))  # 加载ma合成数据2
        np.random.shuffle(be_gen)  # 随机打乱be合成数据
        np.random.shuffle(ma_gen1)  # 随机打乱ma合成数据1
        np.random.shuffle(ma_gen2)  # 随机打乱ma合成数据2
        be = np.concatenate([
            be, 
            be_gen[:be_shape // (parallel)],  # 取部分be合成数据添加到be中
        ], axis=0)
        
        ma = np.concatenate([
            ma,
            ma_gen1[:ma_shape // (parallel) // 5],  # 取部分ma合成数据1添加到ma中
            ma_gen2[:ma_shape // (parallel) // 5],  # 取部分ma合成数据2添加到ma中
        ], axis=0)

    print(be.shape, ma.shape)

    train_data = np.concatenate([be, ma], axis=0)  # 合并be和ma数据
    train_label = np.concatenate([np.zeros(be.shape[0]), np.ones(ma.shape[0])], axis=0)  # 生成对应的标签
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)  # 合并数据和标签形成训练集

    test_data_label = np.load(os.path.join(feat_dir, 'test.npy'))  # 加载测试集数据
    test_data = test_data_label[:, :32]  # 提取测试集特征
    test_label = test_data_label[:, -1]  # 提取测试集标签

    device = int(cuda_device) if cuda_device != 'None' else None
    # 定义丢弃率计划

    if device != None:
        torch.cuda.set_device(device)
    # 数据加载器（输入管道）
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # 加载训练数据
    # 定义模型
    print('building model...')
    mlp1 = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)  # 初始化MLP模型1
    if device != None:
        mlp1.to_cuda(device)  # 将模型1移至CUDA
        mlp1 = mlp1.cuda()  # 将模型1移至GPU
    optimizer1 = torch.optim.Adam(mlp1.parameters(), lr=learning_rate)  # 为模型1定义优化器
    
    mlp2 = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)  # 初始化MLP模型2
    if device != None:
        mlp2.to_cuda(device)  # 将模型2移至CUDA
        mlp2 = mlp2.cuda()  # 将模型2移至GPU
    optimizer2 = torch.optim.Adam(mlp2.parameters(), lr=learning_rate)  # 为模型2定义优化器

    epoch=0
    mlp1.train()  # 设置模型1为训练模式
    mlp2.train()  # 设置模型2为训练模式
    for epoch in tqdm(range(epochs)):
        train(train_loader, epoch, mlp1, optimizer1, mlp2, optimizer2, device)  # 训练模型
    
    mlp1.eval()  # 设置模型1为评估模式
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)  # 加载测试数据
    preds = predict(test_loader, mlp1, device)  # 预测测试集标签
    np.save(os.path.join(result_dir, 'prediction.npy'), preds)  # 保存预测结果

    scores = np.zeros((2, 2))  # 初始化混淆矩阵
    for label, pred in zip(test_label, preds):
        scores[int(label), int(pred)] += 1  # 更新混淆矩阵
    TP = scores[1, 1]  # 计算真阳性
    FP = scores[0, 1]  # 计算假阳性
    TN = scores[0, 0]  # 计算真阴性
    FN = scores[1, 0]  # 计算假阴性
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)  # 计算准确率
    Recall = TP / (TP + FN)  # 计算召回率
    Precision = TP / (TP + FP)  # 计算精确率
    F1score = 2 * Recall * Precision / (Recall + Precision)  # 计算F1分数
    print(Recall, Precision, F1score)
    
    with open(os.path.join(result_dir, 'detection_result.txt'), 'w') as fp:
        fp.write('测试数据: Benign/Malicious = %d/%d\n'%((TN + FP), (TP + FN)))  # 写入测试数据统计
        fp.write('召回率: %.2f, 精确率: %.2f, F1分数: %.2f\n'%(Recall, Precision, F1score))  # 写入召回率、精确率和F1分数
        fp.write('准确率: %.2f\n'%(Accuracy))  # 写入准确率

    mlp1 = mlp1.cpu()  # 将模型1移至CPU
    mlp1.to_cpu()  # 确保模型在CPU上
    torch.save(mlp1, os.path.join(model_dir, 'Detection_Model.pkl'))  # 保存训练好的模型
    

# XXX：自己写的直接测试函数
# 测试执行函数
def test_model(feat_dir, model_dir, result_dir, feat_name, cuda=None, batch_size=128):
    """
    使用保存的模型对测试数据进行预测，并将预测结果保存为prediction.npy文件。
    
    参数：
    feat_dir: str, 测试数据的路径（如test.npy所在的目录）
    model_dir: str, 保存的模型路径（如Detection_Model.pkl所在的目录）
    result_dir: str, 预测结果保存路径
    cuda: int or None, 使用的GPU设备编号，如果为None，则使用CPU
    batch_size: int, 批处理大小
    """
    # 设置文件路径
    test_data_path = os.path.join(feat_dir, 'test.npy')
    model_path = os.path.join(model_dir, 'Detection_Model.pkl')

    # 加载测试数据
    test_data_label = np.load(test_data_path)
    test_data = test_data_label[:, :32]  # 提取特征
    test_label = test_data_label[:, -1]  # 提取标签
    
    # 加载保存的模型
    model = torch.load(model_path, map_location='cpu')  # 确保模型在CPU上
    model.eval()  # 设置模型为评估模式
    
    if cuda is not None:
        torch.cuda.set_device(cuda)  # 设置CUDA设备
        model = model.cuda()  # 将模型移动到GPU
    
    # 构建测试数据加载器
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    
    # 预测
    preds = []
    with torch.no_grad():  # 在预测时不需要计算梯度
        for data in test_loader:
            feats = data.to(dtype=torch.float32)
            if cuda is not None:
                feats = feats.cuda()  # 将特征移动到GPU
            logits = model(feats)  # 前向传播
            outputs = F.softmax(logits, dim=1)  # 计算softmax输出
            preds.append(outputs[:, 1].detach().cpu().numpy())  # 保存概率值
    
    # 保存预测结果
    preds = np.concatenate(preds, axis=0)
    prediction_path = os.path.join(result_dir, 'prediction_'+feat_name+'.npy')
    np.save(prediction_path, preds)
    print(f"预测结果已保存至: {prediction_path}")
