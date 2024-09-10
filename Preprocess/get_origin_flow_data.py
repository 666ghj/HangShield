import os
import sys
import numpy as np


def get_feat(file, data_type_id):
    """
    从文件中读取特征数据，并将其转换为固定长度的特征数组。

    参数:
    file (str): 输入文件的路径，文件内容包含序列数据。
    data_type_id (int): 数据类型的标识符，用于标记数据所属的类别。

    返回:
    np.array: 返回包含所有流特征的NumPy数组，每个流的特征是一个长度为50的数组，最后一个元素为数据类型标识符。如果文件无法打开，则返回None。
    """
    try:
        # 尝试打开文件，如果失败则返回None
        fp = open(file, 'r')
    except:
        return None
    
    flows = []  # 用于存储所有流的特征
    for i, line in enumerate(fp):
        # 将每一行的数据拆分为特征序列
        line_s = line.strip().split(';')
        sq = line_s[0].split(',')
        feat = []  # 用于存储单个流的特征

        # 遍历序列，计算差异并生成特征
        for i in range(50):
            if i >= len(sq):
                # 如果序列长度不足50，用0填充
                feat.append(0)
            else:
                # 计算当前元素与前一个元素的差异
                length = abs(int(sq[i]) - (0 if i == 0 else int(sq[i - 1])))
                if length >= 2000:
                    # 如果差异值大于等于2000，将其设置为1999（上限）
                    feat.append(1999)
                else:
                    # 否则，直接添加差异值
                    feat.append(length)
        
        # 在特征数组末尾添加数据类型标识符
        feat.append(data_type_id)
        # 将该流的特征添加到flows列表中
        flows.append(feat)
    
    # 将所有流的特征转换为NumPy数组并返回
    return np.array(flows, dtype=int)


def main(sequence_data_path, save_dir, data_type, data_type_id):
    """
    读取序列数据文件，生成特征数组，并将其保存到指定目录。

    参数:
    sequence_data_path (str): 序列数据文件的路径。
    save_dir (str): 保存特征数据的目录路径。
    data_type (str): 数据类型的名称，用于生成保存文件的名称。
    data_type_id (int): 数据类型的标识符，用于标记数据所属的类别。
    """
    # 调用get_feat函数生成特征数组
    data = get_feat(sequence_data_path, data_type_id)
    # 将特征数组保存为NumPy格式的文件，文件名为数据类型的名称
    np.save(os.path.join(save_dir, data_type), data)


if __name__ == '__main__':
    # 读取types.txt文件，获取所有数据类型的列表
    with open('types.txt', 'r') as file:
        # 按行分割数据类型名称
        type_list = file.read().split()
    
    # 遍历所有数据类型，为每种类型生成并保存特征数据
    for data_type_id, data_type in enumerate(type_list):
        # 构建序列数据文件的路径
        sequence_data_path = "feature/feature_" + data_type
        # 设置保存特征数据的目录
        save_dir = './data'

        # 调用主函数处理当前数据类型的文件
        main(sequence_data_path, save_dir, data_type, data_type_id)

