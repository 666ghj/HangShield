import torch
import torch.nn as nn
import numpy as np
import os
from . import combine

# 定义一个更复杂的分类器类，增加隐藏层和激活函数
class ComplexClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(ComplexClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # 第一隐藏层
        self.relu1 = nn.ReLU()  # 第一激活函数
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # 第二隐藏层
        self.relu2 = nn.ReLU()  # 第二激活函数
        self.fc3 = nn.Linear(hidden_size2, num_classes)  # 输出层
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# 训练复杂分类器的函数
def train_complex_classifier(prediction_data, labels, num_classes, static, learning_rate=0.003, epochs=5000, model_save_path="../multi_Classifier/model.pth"):
    input_size = prediction_data.shape[1]
    hidden_size1 = 128  # 第一隐藏层的节点数
    hidden_size2 = 64   # 第二隐藏层的节点数

    # 初始化模型
    model = ComplexClassifier(input_size, hidden_size1, hidden_size2, num_classes)
    
    # 如果static等于1且模型路径存在，直接加载模型
    if static == 1 and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"已加载训练好的复杂softmax模型: {model_save_path}")
        return model
    
    # 如果static不等于1，进行训练
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器

    # 将数据和标签转换为Tensor
    prediction_data = torch.tensor(prediction_data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # 训练过程
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        optimizer.zero_grad()  # 梯度清零

        outputs = model(prediction_data)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失

        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        # 每100个epoch打印一次损失
        # if (epoch + 1) % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # 训练完成后保存模型到指定路径
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"复杂softmax模型已保存至: {model_save_path}")

    return model  # 返回训练好的模型


# 预测函数
def predict(model, prediction_data):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 关闭梯度计算
        prediction_data = torch.tensor(prediction_data, dtype=torch.float32)
        outputs = model(prediction_data)  # 进行前向传播
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
    return predicted.numpy()  # 返回numpy数组格式的预测结果

# 主函数
def main(feat_dir, result_dir, static=0):
    # 调用combine模块中的main函数，合并数据
    combine.main(feat_dir, result_dir)
    
    # 加载已经保存的combined_data_with_labels.npy文件
    combined_data_file = os.path.join(result_dir, 'combined_data_with_labels.npy')
    combined_data_with_labels = np.load(combined_data_file)

    # 将数据分为特征数据和标签数据
    prediction_data = combined_data_with_labels[:, :-1]  # 特征数据
    labels = combined_data_with_labels[:, -1].astype(int)  # 标签数据

    # 获取类别数量
    num_classes = len(np.unique(labels))
    
    # 训练复杂的分类器
    model = train_complex_classifier(prediction_data, labels, num_classes, static)

    # 进行预测
    predictions = predict(model, prediction_data)

    # 保存预测结果到文件
    predictions_file = os.path.join(result_dir, 'final_predictions.npy')
    np.save(predictions_file, predictions)
    print(f"最终预测结果已保存至: {predictions_file}")
