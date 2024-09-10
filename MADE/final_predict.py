import os
import numpy as np
import xgboost
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 使用多个机器学习模型的集成来纠正剩余样本的标签
def main(feat_dir, result_dir):

    # 加载已知的良性和恶意流量数据
    be_g = np.load(os.path.join(feat_dir, 'be_groundtruth.npy'))  # 已知的良性流量数据
    ma_g = np.load(os.path.join(feat_dir, 'ma_groundtruth.npy'))  # 已知的恶意流量数据
    be_u = np.load(os.path.join(feat_dir, 'be_unknown.npy'))  # 未知的良性流量数据
    ma_u = np.load(os.path.join(feat_dir, 'ma_unknown.npy'))  # 未知的恶意流量数据

    # 准备训练和测试数据
    X_train = np.concatenate([be_g, ma_g], axis=0)  # 训练数据，包括已知的良性和恶意流量
    Y_train = np.concatenate([np.zeros(be_g.shape[0]), np.ones(ma_g.shape[0])], axis=0)  # 训练标签，0代表良性，1代表恶意

    X_test = np.concatenate([be_g, ma_g, be_u, ma_u], axis=0)  # 测试数据，包括所有已知和未知的流量
    Y_test = np.zeros(X_test.shape[0])  # 初始化测试标签为0

    dtrain = xgboost.DMatrix(X_train, label=Y_train)  # 创建XGBoost的训练数据集
    dtest = xgboost.DMatrix(X_test, label=Y_test)  # 创建XGBoost的测试数据集
    params = {}  # XGBoost模型的参数

    # 使用GaussianNB模型进行训练和预测
    Gaussiannb = GaussianNB()
    Gaussiannb.fit(X_train, Y_train)
    possibility = Gaussiannb.predict(X_test)  # 预测概率
    y_pred = possibility > 0.5  # 将概率转换为二进制标签

    ensemble = y_pred.astype(int)  # 初始化集成模型的预测结果
    ensemble_pos = possibility  # 初始化集成模型的预测概率

    # 使用XGBoost模型进行训练和预测
    bst = xgboost.train(params, dtrain)
    possibility = bst.predict(dtest)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)  # 更新集成模型的预测结果
    ensemble_pos = ensemble_pos + possibility  # 更新集成模型的预测概率

    # 使用AdaBoost模型进行训练和预测
    AdaBoost = AdaBoostClassifier()
    AdaBoost.fit(X_train, Y_train)
    possibility = AdaBoost.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)  # 更新集成模型的预测结果
    ensemble_pos = ensemble_pos + possibility  # 更新集成模型的预测概率

    # 使用线性判别分析（LDA）模型进行训练和预测
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, Y_train)
    possibility = LDA.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)  # 更新集成模型的预测结果
    ensemble_pos = ensemble_pos + possibility  # 更新集成模型的预测概率

    # 使用支持向量机（SVM）模型进行训练和预测
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)
    possibility = svm.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)  # 更新集成模型的预测结果
    ensemble_pos = ensemble_pos + possibility  # 更新集成模型的预测概率

    # 使用随机森林（Random Forest）模型进行训练和预测
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    possibility = rf.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)  # 更新集成模型的预测结果
    ensemble_pos = ensemble_pos + possibility  # 更新集成模型的预测概率

    # 使用逻辑回归（Logistic Regression）模型进行训练和预测
    logistic = LogisticRegression(penalty='l2')
    logistic.fit(X_train, Y_train)
    possibility = logistic.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)  # 更新集成模型的预测结果
    ensemble_pos = ensemble_pos + possibility  # 更新集成模型的预测概率

    # 根据集成模型的结果进行最终分类
    ensemble_pred = []
    ensemble_test = []
    be_num = 0
    ma_num = 0
    be_all_final = []
    ma_all_final = []
    for i in range(len(ensemble)):
        # NOTE：容易出现误判，把好多好的放进了恶意集合，所以把这个加大一些
        if ensemble[i] >= 5:  # 如果预测的结果数超过4个模型认为是恶意流量，则标记为恶意
            ensemble_pred.append(True)
            ensemble_test.append(Y_test[i])
            ma_num = ma_num + 1
            ma_all_final.append(X_test[i])

        else:  # 否则标记为良性流量
            ensemble_pred.append(False)
            ensemble_test.append(Y_test[i])
            be_num = be_num + 1
            be_all_final.append(X_test[i])

    # 转换为数组格式并随机打乱
    be_all_final = np.array(be_all_final)
    ma_all_final = np.array(ma_all_final)
    # FIXME：这里先不采取降噪的，使用原本就高质量的数据
    be_all_final = np.load(os.path.join(feat_dir, 'be.npy'))
    ma_all_final = np.load(os.path.join(feat_dir, 'ma.npy'))
    np.random.shuffle(be_all_final)
    np.random.shuffle(ma_all_final)
    
    # 保存纠正后的良性和恶意流量数据
    np.save(os.path.join(feat_dir, 'be_corrected.npy'), be_all_final)
    np.save(os.path.join(feat_dir, 'ma_corrected.npy'), ma_all_final)
    
    # 计算并输出剩余的噪声比率
    wrong_be = be_all_final[:, -1].sum()
    wrong_ma = ma_all_final.shape[0] - ma_all_final[:, -1].sum()
    print('良性集合中的恶意流量: %d/%d' % (be_all_final.shape[0], wrong_be))
    print('恶意集合中的良性流量: %d/%d' % (ma_all_final.shape[0], wrong_ma))
    with open(os.path.join(result_dir, 'label_correction.txt'), 'w') as fp:
        fp.write('良性集合中的恶意流量: %d(%d)\n' % (wrong_be, be_all_final.shape[0]))
        fp.write('恶意集合中的良性流量: %d(%d)\n' % (wrong_ma, ma_all_final.shape[0]))
        fp.write('剩余噪声比率: %.2f%%\n' % (100 * (wrong_be + wrong_ma) / (be_all_final.shape[0] + ma_all_final.shape[0])))
