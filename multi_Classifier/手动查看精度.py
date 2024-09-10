import numpy as np

def calculate_accuracy(labels, predictions):
    # 计算整体正确预测的数量
    correct_predictions = np.sum(labels == predictions)
    # 计算总样本数量
    total_samples = len(labels)
    # 计算总体准确度
    accuracy = correct_predictions / total_samples
    return accuracy

def calculate_classwise_accuracy(labels, predictions, num_classes):
    accuracy_per_class = []
    
    for class_label in range(num_classes):
        # 计算 True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN)
        TP = np.sum((labels == class_label) & (predictions == class_label))  # 真实类别和预测类别都是该类
        FP = np.sum((labels != class_label) & (predictions == class_label))  # 真实类别不是该类，但预测为该类
        FN = np.sum((labels == class_label) & (predictions != class_label))  # 真实类别是该类，但预测不是该类
        TN = np.sum((labels != class_label) & (predictions != class_label))  # 真实类别不是该类，预测也不是该类

        # 计算 Accuracy
        if (TP + FP + FN + TN) > 0:
            accuracy = (TP + TN) / (TP + FP + FN + TN)
        else:
            accuracy = 0.0
        
        accuracy_per_class.append(accuracy)
    
    return accuracy_per_class

def calculate_precision_recall_f1_multi_class(labels, predictions, num_classes):
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for class_label in range(num_classes):
        # 计算 True Positive (TP), False Positive (FP), False Negative (FN)
        TP = np.sum((labels == class_label) & (predictions == class_label))  # 真实类别和预测类别都是该类
        FP = np.sum((labels != class_label) & (predictions == class_label))  # 真实类别不是该类，但预测为该类
        FN = np.sum((labels == class_label) & (predictions != class_label))  # 真实类别是该类，但预测不是该类
        
        # 计算 Precision
        if (TP + FP) > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0.0
        
        # 计算 Recall
        if (TP + FN) > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0.0
        
        # 计算 F1-Score
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1_score)
    
    return precision_per_class, recall_per_class, f1_per_class

def calculate_false_positive_rate_multi_class(labels, predictions, num_classes):
    fpr_per_class = []
    
    for class_label in range(num_classes):
        # 计算 False Positive (FP) 和 True Negative (TN)
        FP = np.sum((labels != class_label) & (predictions == class_label))  # 真实类别不是该类，但预测为该类
        TN = np.sum((labels != class_label) & (predictions != class_label))  # 真实类别不是该类，预测也不是该类
        
        # 计算误报率
        if (FP + TN) > 0:
            fpr = FP / (FP + TN)
        else:
            fpr = 0.0
        
        fpr_per_class.append(fpr)
    
    return fpr_per_class

def main():
    # 加载已经保存的combined_data_with_labels.npy文件
    combined_data_file = 'combined_data_with_labels.npy'
    combined_data_with_labels = np.load(combined_data_file)

    # 加载已经保存的final_predictions.npy文件
    predictions_file = 'final_predictions.npy'
    predictions = np.load(predictions_file)

    # 提取真实标签数据
    labels = combined_data_with_labels[:, -1].astype(int)  # 标签数据

    # 获取类别数量
    num_classes = len(np.unique(labels))

    # 计算整体准确度
    overall_accuracy = calculate_accuracy(labels, predictions)
    print(f"整体预测准确度: {overall_accuracy * 100:.2f}%")

    # 计算每个类别的误报率
    fpr_per_class = calculate_false_positive_rate_multi_class(labels, predictions, num_classes)
    
    # 打印每个类别的误报率
    for i, fpr in enumerate(fpr_per_class):
        print(f"类别 {i} 的误报率: {fpr * 100:.2f}%")

    # 计算平均误报率
    average_fpr = np.mean(fpr_per_class)
    print(f"平均误报率: {average_fpr * 100:.2f}%")

    # 计算每个类别的 Precision, Recall, F1-Score
    precision_per_class, recall_per_class, f1_per_class = calculate_precision_recall_f1_multi_class(labels, predictions, num_classes)
    
    # 打印每个类别的 Precision, Recall, F1-Score
    for i in range(num_classes):
        print(f"类别 {i} 的精确度 (Precision): {precision_per_class[i] * 100:.2f}%")
        print(f"类别 {i} 的召回率 (Recall): {recall_per_class[i] * 100:.2f}%")
        print(f"类别 {i} 的 F1-Score: {f1_per_class[i] * 100:.2f}%")

    # 计算平均 Precision, Recall, F1-Score
    average_precision = np.mean(precision_per_class)
    average_recall = np.mean(recall_per_class)
    average_f1 = np.mean(f1_per_class)
    
    print(f"平均精确度 (Precision): {average_precision * 100:.2f}%")
    print(f"平均召回率 (Recall): {average_recall * 100:.2f}%")
    print(f"平均 F1-Score: {average_f1 * 100:.2f}%")

    # 计算每个类别的 Accuracy
    accuracy_per_class = calculate_classwise_accuracy(labels, predictions, num_classes)
    
    # 打印每个类别的 Accuracy
    for i, accuracy in enumerate(accuracy_per_class):
        print(f"类别 {i} 的准确率 (Accuracy): {accuracy * 100:.2f}%")

    # 计算平均 Accuracy
    average_accuracy = np.mean(accuracy_per_class)
    print(f"平均准确率: {average_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
