from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import re
from sklearn.cluster import DBSCAN

def main(feat_dir, made_dir, alpha, TRAIN):
    
    # 根据MADE模型的密度估计，选择真正的良性样本
    alpha = float(alpha)
    be = np.load(os.path.join(feat_dir, 'be.npy'))  # 加载良性样本（benign）
    ma = np.load(os.path.join(feat_dir, 'ma.npy'))  # 加载恶意样本（malicious）
    feats = np.concatenate((be, ma), axis=0)  # 将良性和恶意样本合并
    print(feats.shape)
    be_number, be_shape = be.shape  # 良性样本数量及其特征维度
    ma_number, ma_shape = ma.shape  # 恶意样本数量及其特征维度
    assert(be_shape == ma_shape)  # 确保良性和恶意样本的特征维度相同
    NLogP = [0 for _ in range(be_number + ma_number)]  # 初始化负对数概率列表
    nlogp_lst = [[] for _ in range(be_number + ma_number)]  # 初始化每个样本的负对数概率记录列表

    epochs = 0
    # 计算模型生成的epochs数量
    for filename in os.listdir(made_dir):
        if re.match('be_%s_\d+' % (TRAIN), filename):
            epochs += 1

    # 从训练的中后期epochs提取样本的负对数概率
    for i in range(epochs // 2, epochs): 
        epoch = (i + 1) * 10
        # 读取良性样本的负对数概率
        with open(os.path.join(made_dir, 'be_%sMADE_%d' % (TRAIN, epoch)), 'r') as fp:
            for i, line in enumerate(fp):
                s = float(line.strip())
                if s > 10000:
                    s = 10000  # 对极端大值进行裁剪
                NLogP[i] = NLogP[i] + s  
                nlogp_lst[i].append(s)

        # 读取恶意样本的负对数概率
        with open(os.path.join(made_dir, 'ma_%sMADE_%d' % (TRAIN, epoch)), 'r') as fp:
            for i, line in enumerate(fp):
                s = float(line.strip())
                if s > 10000:
                    s = 10000  # 对极端大值进行裁剪
                NLogP[i + be_number] = NLogP[i + be_number] + s
                nlogp_lst[i + be_number].append(s)

    # 将样本按照NLogP值排序
    seq = list(range(len(NLogP)))
    seq.sort(key=lambda x: NLogP[x]) 

    be_extract = []
    be_extract_lossline = []
    extract_range = int(alpha * (be_number + ma_number))  # 选取前alpha比例的样本
    for i in range(extract_range): 
        be_extract.append(feats[seq[i]])  # 提取前alpha比例的样本
        be_extract_lossline.append(nlogp_lst[seq[i]])  # 记录这些样本的损失曲线

    # 根据样本之间的距离进一步过滤出更可信的良性样本
    be_extract = np.array(be_extract)

    def gaussian(feat, target_set):
        ro = 0
        sigma = 5  # 高斯核的标准差
        # 计算当前样本与目标集合中每个样本的欧式距离
        toBe = np.sort(np.linalg.norm(feat[None, :32].repeat(target_set.shape[0], axis=0) - target_set[:, :32], axis=1))
        num = target_set.shape[0] // 2
        for i in range(num):
            dis = toBe[i]
            ro += np.exp(-(dis ** 2 / 2 / sigma ** 2))  # 计算高斯核
        return ro / num  # 返回平均距离

    toBes = []
    toBesort = []
    for feat in be_extract:
        gauss = gaussian(feat, be_extract)  # 计算高斯核密度
        toBes.append(gauss)
        toBesort.append(gauss)
    toBesort.sort()
    dom = toBesort[int(len(toBesort) * 0.5)]  # 取高斯密度的中位数作为阈值

    be_clean = []
    be_clean_lossline = []
    remain_index = []
    for i, toBe in enumerate(toBes):
        if toBe >= dom:
            be_clean.append(be_extract[i])  # 保留高斯密度大于阈值的样本
            be_clean_lossline.append(be_extract_lossline[i])
        else:
            remain_index.append(seq[i])  # 其余样本保留索引

    remain_index += seq[extract_range:]  # 将未提取的样本索引添加到保留索引中

    # 根据剩余样本与清洁良性样本的距离，进一步过滤出更可信的恶意样本
    remain_index.sort(key=lambda x: -NLogP[x])  
    ma_extract = [feats[index] for index in remain_index]
    ma_extract_lossline = [nlogp_lst[index] for index in remain_index]

    ma_extract = np.array(ma_extract)
    be_clean = np.array(be_clean)
    ma_clean = []
    ma_clean_lossline = []
    
    be_unknown = []
    ma_unknown = []
    unknown_index = []

    be_unknown_lossline = []
    ma_unknown_lossline = []

    toBes = []
    for feat in ma_extract:
        toBe = np.sort(np.linalg.norm(feat[None, :32].repeat(be_clean.shape[0], axis=0) - be_clean[:, :32], axis=1))
        toBes.append(toBe[:].mean())  # 计算恶意样本到清洁良性样本的平均距离
    toMas = []
    for feat in ma_extract:
        toMa = np.sort(np.linalg.norm(feat[None, :32].repeat(ma_extract.shape[0], axis=0) - ma_extract[:, :32], axis=1))
        toMas.append(toMa[1:].mean())  # 计算恶意样本到其他恶意样本的平均距离

    relative_dis = [(toMa - toBe) for toMa, toBe in zip(toMas, toBes)]
    relative_dis.sort()
    dom = relative_dis[int(len(be_clean) * 1)]  # 取相对距离的中位数作为阈值
    
    for i, (toMa, toBe, feat, lossline, index) in \
        enumerate(zip(toMas, toBes, ma_extract, ma_extract_lossline, remain_index)):
        
        if toMas[i] - toBes[i] < dom or np.isnan(dom) or np.isinf(dom):
            ma_clean.append(feat)  # 保留高可信度的恶意样本
            ma_clean_lossline.append(lossline)
        else:
            unknown_index.append(index)
            if index < be_number:
                be_unknown.append(feat)  # 标记为未知的良性样本
                be_unknown_lossline.append(nlogp_lst[index])
            else:
                ma_unknown.append(feat)  # 标记为未知的恶意样本
                ma_unknown_lossline.append(nlogp_lst[index])

    # 打印最终的清洁良性和恶意样本数量
    be_num = 0
    ma_num = 0
    for feat in be_clean:
        if int(feat[-1]) == 0:
            be_num += 1
        else:
            ma_num += 1
    print('be_clean: {} be + {} ma.'.format(be_num, ma_num))

    be_num = 0
    ma_num = 0
    for feat in ma_clean:
        if int(feat[-1]) == 0:
            be_num += 1
        else:
            ma_num += 1
    print('ma_clean: {} be + {} ma.'.format(be_num, ma_num))
    
    # 保存清洁和未知的样本数据
    np.save(os.path.join(feat_dir, 'be_groundtruth.npy'), np.array(be_clean))
    np.save(os.path.join(feat_dir, 'ma_groundtruth.npy'), np.array(ma_clean))
    np.save(os.path.join(feat_dir, 'be_unknown.npy'), np.array(be_unknown))
    np.save(os.path.join(feat_dir, 'ma_unknown.npy'), np.array(ma_unknown))

    print(len(be_clean), len(ma_clean), len(be_unknown), len(ma_unknown))
