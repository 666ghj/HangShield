import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# 定义一个基于LSTM和GMM的自动编码器模型
# NOTE：GMM虽然在本模块中没有被用到，但是在标签噪声纠正模块中用到了
class LSTM_AE_GMM(nn.Module):
    def __init__(self, emb_dim, input_size, hidden_size, dropout, max_len,
                 est_hidden_size, est_output_size, device=0, est_dropout=0.5,
                 learning_reat=0.0001, lambda1=0.1, lambda2=0.0001):
        super(LSTM_AE_GMM, self).__init__()

        self.max_len = max_len  # 包长度序列的最大长度（已提前处理）
        self.emb_dim = emb_dim  # 嵌入向量的维度
        self.input_size = input_size  # 输入序列的长度（AE最终的预测长度）
        self.hidden_size = hidden_size  # GRU层的隐藏状态维度
        self.dropout = dropout  # dropout的比例
        self.device = device  # 运行设备的ID（默认为0）

        # 设置当前CUDA设备（GPU）
        torch.cuda.set_device(self.device)
        
        # 词嵌入层：将输入的包长度序列转换为嵌入向量
        self.embedder = nn.Embedding(self.max_len, self.emb_dim)

        # 编码器部分：双向GRU层，堆叠了两层GRU，用于从嵌入向量中提取特征
        self.encoders = nn.ModuleList([
            nn.GRU(
                input_size=self.emb_dim,  # GRU输入的维度是嵌入向量的维度
                hidden_size=self.hidden_size,  # GRU隐藏状态的维度
                batch_first=True,  # 输入的第一个维度是batch size
                bidirectional=True,  # 使用双向GRU
            ).cuda(), 
            nn.GRU(
                input_size=self.hidden_size * 2,  # 双向GRU的输出维度是2倍的隐藏状态维度
                hidden_size=self.hidden_size,  # 第二层GRU的隐藏状态维度
                batch_first=True,
                bidirectional=True,  # 使用双向GRU
            ).cuda()
        ])

        # 解码器部分：双向GRU层，结构与编码器类似
        self.decoders = nn.ModuleList([
            nn.GRU(
                input_size=self.hidden_size * 4,  # 解码器的输入是编码器输出的4倍隐藏状态维度
                hidden_size=self.hidden_size,  # GRU隐藏状态的维度
                batch_first=True,
                bidirectional=True,  # 使用双向GRU
            ).cuda(), 
            nn.GRU(
                input_size=self.hidden_size * 2,  # 解码器第二层GRU的输入是2倍隐藏状态维度
                hidden_size=self.hidden_size,  # GRU隐藏状态的维度
                batch_first=True,
                bidirectional=True,  # 使用双向GRU
            ).cuda()
        ])

        # 重构层：将解码器的输出转换为原始包长度序列
        self.rec_fc1 = nn.Linear(4 * self.hidden_size, self.hidden_size)  # 线性层，将输入维度缩减到隐藏状态维度
        self.rec_fc2 = nn.Linear(self.hidden_size, self.max_len)  # 线性层，将隐藏状态维度转换为包长度序列的维度
        self.rec_softmax = nn.Softmax(dim=2)  # 对包长度序列应用softmax
        self.cross_entropy = nn.CrossEntropyLoss()  # 计算重构损失的交叉熵损失函数

        # 估计模块的参数设置
        self.est_hidden_size = est_hidden_size  # 估计模块隐藏层的维度
        self.est_output_size = est_output_size  # 估计模块输出的维度
        self.est_dropout = est_dropout  # 估计模块中的dropout比例
        self.fc1 = nn.Linear(4 * self.hidden_size, self.est_hidden_size)  # 线性层，将输入维度转换为估计模块的隐藏层维度
        self.fc2 = nn.Linear(self.est_hidden_size, self.est_output_size)  # 线性层，将隐藏层维度转换为输出维度
        self.est_drop = nn.Dropout(p=self.est_dropout)  # 估计模块中的dropout层
        self.softmax = nn.Softmax(dim=1)  # softmax层，生成分类概率分布

        self.training = False  # 标识当前模型是否处于训练模式

        # 损失函数中的正则化参数
        self.lambda1 = lambda1  # 正则化参数lambda1
        self.lambda2 = lambda2  # 正则化参数lambda2

    # 编码器的前向传播过程
    def encode(self, x):
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        embed_x = self.embedder(x.long())  # 将输入转换为嵌入向量
        if self.training is True:
            embed_x = F.dropout(embed_x)  # 在训练模式下应用dropout
        outputs = [embed_x]  # 初始化GRU层的输出列表
        hs = []  # 初始化隐藏状态列表
        for layer in range(2):  # 遍历每一层GRU
            gru = self.encoders[layer]  # 获取第layer层的GRU
            output, h = gru(outputs[-1])  # 计算GRU的输出和隐藏状态
            outputs.append(output)  # 将GRU输出添加到输出列表
            hs.append(torch.transpose(h, 0, 1).reshape(
                -1, 2 * self.hidden_size))  # 将隐藏状态转置并调整形状后添加到隐藏状态列表
        res = torch.cat(outputs[1:], dim=2)  # 将所有GRU层的输出拼接在一起
        res_h = torch.cat(hs, dim=1)  # 将所有隐藏状态拼接在一起
        return res_h  # 返回最终隐藏状态
    
    # 将编码器的输出转换为解码器的输入
    def decode_input(self, x):
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        y = x.reshape(-1, 1, 4 * self.hidden_size)  # 调整输入的形状
        y = y.repeat(1, self.input_size, 1)  # 将输入重复，以匹配解码器输入的序列长度
        return y
    
    # 解码器的前向传播过程
    def decode(self, x):
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        input = x.view(-1, self.input_size, 4 * self.hidden_size)  # 调整输入的形状
        outputs = [input]  # 初始化GRU层的输出列表
        hs = []  # 初始化隐藏状态列表
        for layer in range(2):  # 遍历每一层GRU
            gru = self.decoders[layer]  # 获取第layer层的GRU
            output, h = gru(outputs[-1])  # 计算GRU的输出和隐藏状态
            outputs.append(output)  # 将GRU输出添加到输出列表
            hs.append(torch.transpose(h, 0, 1).reshape(
                -1, 2 * self.hidden_size))  # 将隐藏状态转置并调整形状后添加到隐藏状态列表
        res = torch.cat(outputs[1:], dim=2)  # 将所有GRU层的输出拼接在一起
        res_h = torch.cat(hs, dim=1)  # 将所有隐藏状态拼接在一起
        return res, res_h  # 返回解码器的输出和隐藏状态
    
    # 重构过程：计算重构损失
    def reconstruct(self, x, y):
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        x_rec = self.rec_fc2(F.selu(self.rec_fc1(x)))  # 通过重构层进行前向传播
        loss = torch.stack([
            torch.stack([
                self.cross_entropy(
                    x_sft.unsqueeze(0), 
                    y_label.unsqueeze(0)
                ) for x_sft, y_label in zip(xi, yi)  # 计算交叉熵损失
            ]) for xi, yi in zip(x_rec, y)
        ], dim=0)
        mask = y.bool()  # 计算mask
        loss_ret = torch.sum(loss * mask, dim=1) / torch.sum(mask, dim=1)  # 计算加权平均损失
        return loss_ret  # 返回重构损失
    
    # 估计模块的前向传播过程
    def estimate(self, x):
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        x = x.view(-1, 4 * self.hidden_size)  # 调整输入的形状
        res = self.est_drop(F.tanh(self.fc1(x)))  # 通过估计模块的第一层线性层，并应用tanh激活函数
        res = self.softmax(self.fc2(res))  # 通过估计模块的第二层线性层，并应用softmax激活函数
        return res  # 返回估计模块的输出
    
    # 获取特征表示
    def feature(self, input):
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        res_encode_h = self.encode(input.float())  # 获取编码器的输出
        return res_encode_h  # 返回特征表示

    # 预测过程：获取编码器输出和重构损失
    def predict(self, input):
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        res_encode_h = self.encode(input.float())  # 获取编码器的输出
        decode_input = self.decode_input(res_encode_h)  # 生成解码器的输入
        res_decode, res_decode_h = self.decode(decode_input)  # 获取解码器的输出
        loss_all = self.reconstruct(res_decode, input)  # 计算重构损失
        return res_encode_h, loss_all  # 返回编码器的输出和重构损失

    # 计算总损失
    def loss(self, input):
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        _, loss_all = self.predict(input)  # 获取重构损失
        return torch.mean(loss_all, dim=0)  # 返回平均损失值
    
    # 分类损失：结合重构损失和分类损失
    def classify_loss(self, input, labels):
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        feats, rec_loss = self.predict(input)  # 获取编码器输出和重构损失
        score = self.estimate(feats)  # 通过估计模块获取分类得分
        return F.cross_entropy(score, labels) + torch.mean(rec_loss, dim=0)  # 返回分类损失和重构损失之和
    
    # 分类损失的另一种实现
    def classify_loss_1(self, input, labels):
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        feats, rec_loss = self.predict(input)  # 获取编码器输出和重构损失
        score = self.estimate(feats)  # 通过估计模块获取分类得分
        return F.cross_entropy(score, labels, reduce=False) + rec_loss  # 返回逐样本的分类损失和重构损失之和

    # 切换到训练模式
    def train_mode(self):
        self.training = True  # 将模型设置为训练模式
    
    # 切换到测试模式
    def test_mode(self):
        self.training = False  # 将模型设置为测试模式

    # 将模型转移到CPU
    def to_cpu(self):
        self.device = None  # 清除设备信息
        for encoder in self.encoders:
            encoder = encoder.cpu()  # 将编码器的GRU层转移到CPU
        for decoder in self.decoders:
            decoder = decoder.cpu()  # 将解码器的GRU层转移到CPU
    
    # 将模型转移到CUDA
    def to_cuda(self, device):
        self.device = device  # 设置新的设备ID
        torch.cuda.set_device(self.device)  # 设置当前CUDA设备
        for encoder in self.encoders:
            encoder = encoder.cuda()  # 将编码器的GRU层转移到CUDA
        for decoder in self.decoders:
            decoder = decoder.cuda()  # 将解码器的GRU层转移到CUDA
