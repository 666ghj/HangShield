import numpy as np
import os

def load_prediction_data(prediction_files):
    """
    从多个npy文件加载预测数据，并合并为一个数组。
    每一行代表一个样本的多个模型的预测结果。
    """
    predictions = [np.load(file) for file in prediction_files]
    return np.stack(predictions, axis=1)

def save_combined_data(prediction_data, labels, save_path):
    """
    将预测数据和标签整合并保存为npy文件。
    每一行是一个七维数据：前六维是预测数据，最后一维是标签。
    """
    combined_data = np.concatenate((prediction_data, labels[:, None]), axis=1)
    np.save(save_path, combined_data)
    print(f"整合后的数据已保存至: {save_path}")

def main(feat_dir, result_dir):
    # 设置文件路径
    test_data_path = os.path.join(feat_dir, 'test.npy')
    prediction_files = [
        os.path.join(feat_dir, 'prediction_BEN.npy'),
        os.path.join(feat_dir, 'prediction_RAT.npy'),
        os.path.join(feat_dir, 'prediction_PST.npy'),
        os.path.join(feat_dir, 'prediction_BDT.npy'),
        os.path.join(feat_dir, 'prediction_SPT.npy'),
        os.path.join(feat_dir, 'prediction_DLT.npy')
    ]
    
    # 加载数据
    test_data = np.load(test_data_path)
    labels = test_data[:, -1].astype(int)
    prediction_data = load_prediction_data(prediction_files)

    # 保存整合后的数据
    combined_data_path = os.path.join(result_dir, 'combined_data_with_labels.npy')
    save_combined_data(prediction_data, labels, combined_data_path)
