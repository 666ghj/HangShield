import os 
import sys 
sys.path.append('..')  # 将上一级目录添加到Python的模块搜索路径中
import MADE  # 导入MADE模块，用于训练和预测
import Classifier  # 导入Classifier模块，用于分类
import multi_Classifier  # 导入multi_Classifier模块，用于多分类
import AE  # 导入AE模块，用于自编码器操作
import numpy as np

# 定义generate函数，用于生成和处理数据
def generate(feat_dir, model_dir, made_dir, index, cuda):
    TRAIN_be = 'be_corrected'  # 设置良性样本的训练标签
    TRAIN_ma = 'ma_corrected'  # 设置恶意样本的训练标签
    TRAIN = 'corrected'  # 通用的训练标签
    
    # 训练MADE模型并预测良性和恶意样本
    MADE.train.main(feat_dir, model_dir, TRAIN_be, cuda, '-30')
    MADE.train.main(feat_dir, model_dir, TRAIN_ma, cuda, '-30')
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_be, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_ma, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_ma, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_be, cuda)

    # 训练生成对抗网络（GAN）并生成新数据
    MADE.train_gen_GAN.main(feat_dir, model_dir, made_dir, TRAIN, cuda)
    MADE.generate_GAN.main(feat_dir, model_dir, TRAIN, index, cuda)

# 定义generate_cpus函数，用于并行生成数据
def generate_cpus(feat_dir, model_dir, made_dir, indices, cuda):
    for index in indices:
        generate(feat_dir, model_dir, made_dir, index, cuda)  # 依次调用generate函数

# 定义main_AE函数，用于获取特征并为后续做准备数据
def main_AE(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, static=0):
    # 如果为训练状态
    if static==0:
        # NOTE：训练自编码器（Auto-Encoder, AE）
        AE.train.main(data_dir, model_dir, cuda)
        
        # NOTE：使用自编码器提取各个数据集的特征
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'BEN', cuda)  # 良性样本
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'RAT', cuda)  # 恶意软件（RAT）
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'PST', cuda)  # 恶意样本（PST）
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'BDT', cuda)  # 恶意软件（BDT）
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'SPT', cuda)  # 恶意样本（SPT）
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'DLT', cuda)  # 恶意样本（DLT）
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)  # 测试数据

        # NOTE：分别保存为多份data文件
        
        # 从data_dir中的types.txt文件中读取所有的恶意软件类型，存储在types列表中
        with open(os.path.join(data_dir, 'types.txt'), 'r') as f:
            types = f.readlines()
            types = [t.strip() for t in types] # 去除换行符
            
        # 依次训练和预测各个恶意软件类型
        for malware_type in types:
            combined_data = []
            
            for other_type in types:
                if other_type != malware_type:
                    # 读取每种恶意软件的特征文件
                    file_path = os.path.join(feat_dir, f'{other_type}.npy')
                    data = np.load(file_path)
                    
                    # 将第32列的标签设置为0
                    data[:, 32] = 0
                    combined_data.append(data)
            
            # 合并所有其他类型的特征数据
            if combined_data:
                combined_data = np.concatenate(combined_data, axis=0)
            
                # 在feat_dir下创建一个名为type_not_in的文件，将数据保存进去
                output_file_path = os.path.join(feat_dir, f'{malware_type}_not_in.npy')
                np.save(output_file_path, combined_data)
                
            # 创建一个名为data_malware_type的文件夹，保存在上一级目录
            data_malware_type = os.path.join('..', 'data_'+malware_type)
            if not os.path.exists(data_malware_type):
                os.makedirs(data_malware_type)
            
            # 将当前恶意软件类型的数据复制到data_malware_type文件夹下，重命名为ma.npy，对应的not_in复制重命名为be.npy，判断系统
            if os.name == 'nt':
                # 将feat_dir与data_malware_type转为适合Windows系统的路径
                feat_dir = feat_dir.replace('/', '\\')
                data_malware_type = data_malware_type.replace('/', '\\')
                os.system(f'copy {feat_dir}\\{malware_type}.npy {data_malware_type}\\ma.npy')
                os.system(f'copy {feat_dir}\\{malware_type}_not_in.npy {data_malware_type}\\be.npy')
                os.system(f'copy {feat_dir}\\test.npy {data_malware_type}\\test.npy')
            else:
                os.system(f'cp {feat_dir}/{malware_type}.npy {data_malware_type}/ma.npy')
                os.system(f'cp {feat_dir}/{malware_type}_not_in.npy {data_malware_type}/be.npy')
                os.system(f'cp {feat_dir}/test.npy {data_malware_type}/test.npy')
                
            # 加载复制后的test.npy文件并处理
            test_data = np.load(os.path.join(data_malware_type, 'test.npy'))
            # 修改最后一位类型位
            # 获取当前malware_type在types中的索引
            index = types.index(malware_type)
            test_data[:, 32] = np.where(test_data[:, 32] != index, 0, 1)
            # 保存处理后的test.npy文件
            np.save(os.path.join(data_malware_type, 'test.npy'), test_data)
            
            # 加载复制后的be.npy文件并处理
            ma_data = np.load(os.path.join(data_malware_type, 'ma.npy'))
            # 修改最后一位类型位
            ma_data[:, 32] = 1
            # 保存处理后的test.npy文件
            np.save(os.path.join(data_malware_type, 'ma.npy'), ma_data)
    # 如果为测试状态      
    else:
        # npy流包序列特征提取
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)
        
        
        
def main_BI(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, static=0, feat_name='test'):
    
    # 如果为训练状态
    if static==0:
    
        TRAIN = 'be'
        MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20')
        MADE.get_clean_epochs.main(feat_dir, made_dir, '0.5', TRAIN)
        MADE.final_predict.main(feat_dir, result_dir)
        
        generate_cpus(feat_dir, model_dir, made_dir, list(range(5)), cuda)
        
        TRAIN = 'corrected'
        Classifier.classify.main(feat_dir, model_dir, result_dir, TRAIN, cuda, parallel=5)
        
    # 如果为测试状态
    else:
        Classifier.classify.test_model(feat_dir, model_dir, result_dir, feat_name, cuda)
    
def mian_MU(data_dir, feat_dir, multi_dir, cuda, static=0):
    # 将data_dir中的test.npy文件复制到multi_dir中
    if os.name == 'nt':
        feat_dir = feat_dir.replace('/', '\\')
        multi_dir = multi_dir.replace('/', '\\')
        os.system(f'copy {feat_dir}\\test.npy {multi_dir}\\test.npy')
    else:
        os.system(f'cp {feat_dir}/test.npy {multi_dir}/test.npy')
    # 读取每个data_malware_type文件夹下的result文件夹下的prediction.npy文件,重命名为prediction_malware_type.npy
    with open(os.path.join(data_dir, 'types.txt'), 'r') as f:
        types = f.readlines()
        types = [t.strip() for t in types]
    for malware_type in types:
        data_malware_type = os.path.join('..', 'data_'+malware_type)
        result_dir = os.path.join(data_malware_type, 'result')
        prediction = np.load(os.path.join(result_dir, 'prediction.npy'))
        np.save(os.path.join(multi_dir, f'prediction_{malware_type}.npy'), prediction)
        
    if static==0:
        # 训练复杂softmax模型
        multi_Classifier.model.main(multi_dir, multi_dir)
    else:
        # 测试复杂softmax模型
        multi_Classifier.model.main(multi_dir, multi_dir, static)
        
def mianTRAIN():
    # XXX：以下代码为训练代码
    
    # NOTE：数据初步处理，自编码后划分数据集
    # 设置各个数据、模型、特征、结果目录路径和CUDA设备
    data_dir = '../data/data'
    feat_dir = '../data/feat'
    model_dir = '../data/model'
    made_dir = '../data/made'
    result_dir = '../data/result'
    cuda = 0  # 设置CUDA设备为0
    main_AE(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda)
    
    # NOTE：在每个分类对应的data文件夹下进行训练和预测
    # 从data_dir中的types.txt文件中读取所有的恶意软件类型，存储在types列表中
    with open(os.path.join(data_dir, 'types.txt'), 'r') as f:
        types = f.readlines()
        types = [t.strip() for t in types] # 去除换行符
    # 调用main_BI函数的文件准备工作
    for malware_type in types:
        # 为每个data_malware_type文件夹下创建下面的目录
        data_malware_type = os.path.join('..', 'data_'+malware_type)
        if not os.path.exists(data_malware_type+'/feat'):
            os.makedirs(data_malware_type+'/feat')
        if not os.path.exists(data_malware_type+'/model'):
            os.makedirs(data_malware_type+'/model')
        if not os.path.exists(data_malware_type+'/made'):
            os.makedirs(data_malware_type+'/made')
        if not os.path.exists(data_malware_type+'/result'):
            os.makedirs(data_malware_type+'/result')
        if not os.path.exists(data_malware_type+'/data'):
            os.makedirs(data_malware_type+'/data')
        # 如果data_malware_type文件夹下直接有ma.npy、be.npy、test.npy文件，则移动到data文件夹下，windos和linux系统不同
        if os.path.exists(data_malware_type+'/ma.npy') and os.path.exists(data_malware_type+'/be.npy') and os.path.exists(data_malware_type+'/test.npy'):
            if os.name == 'nt':
                data_malware_type = data_malware_type.replace('/', '\\')
                os.system(f'move {data_malware_type}\\ma.npy {data_malware_type}\\feat\\ma.npy')
                os.system(f'move {data_malware_type}\\be.npy {data_malware_type}\\feat\\be.npy')
                os.system(f'move {data_malware_type}\\test.npy {data_malware_type}\\feat\\test.npy')
            else:
                os.system(f'mv {data_malware_type}/ma.npy {data_malware_type}/data/ma.npy')
                os.system(f'mv {data_malware_type}/be.npy {data_malware_type}/data/be.npy')
                os.system(f'mv {data_malware_type}/test.npy {data_malware_type}/data/test.npy')
    # 依次调用main_BI函数           
    for malware_type in types:
        data_malware_type = os.path.join('..', 'data_'+malware_type)
        data_dir = os.path.join(data_malware_type, 'data')
        feat_dir = os.path.join(data_malware_type, 'feat')
        model_dir = os.path.join(data_malware_type, 'model')
        made_dir = os.path.join(data_malware_type, 'made')
        result_dir = os.path.join(data_malware_type, 'result')
        cuda = 0  # 设置CUDA设备为0
        main_BI(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda)
        
    # NOTE：多分类，将所有二分类的结果放在一起进行多分类，训练复杂softmax分类器
    data_dir = '../data/data'
    feat_dir = '../data/feat'
    multi_dir = '../multi_Classifier'
    cuda = 0  # 设置CUDA设备为0
    mian_MU(data_dir, feat_dir, multi_dir, cuda)
    
def mainUSE():
    # XXX：以下为测试代码，无需训练模型直接使用
    
    # NOTE：先对测试数据进行特征向量表示
    data_dir = '../data/data'
    model_dir = '../data/model'
    feat_dir = '../data/feat'
    cuda = 0
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)
    # NOTE：输入到多个二分类中
    with open(os.path.join(data_dir, 'types.txt'), 'r') as f:
        types = f.readlines()
        types = [t.strip() for t in types] # 去除换行符
    # 依次调用main_BI函数           
    for malware_type in types:
        data_malware_type = os.path.join('..', 'data_'+malware_type)
        data_dir = os.path.join(data_malware_type, 'data')
        model_dir = os.path.join(data_malware_type, 'model')
        made_dir = os.path.join(data_malware_type, 'made')
        result_dir = os.path.join(data_malware_type, 'result')
        feat_dir = '../data/feat'
        cuda = 0  # 设置CUDA设备为0
        main_BI(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, static=1, feat_name=malware_type)
    # NOTE:输入到复杂softmax模型中作最终预测
    data_dir = '../data/data'
    feat_dir = '../data/feat'
    multi_dir = '../multi_Classifier'
    cuda = 0  # 设置CUDA设备为0
    mian_MU(data_dir, feat_dir, multi_dir, cuda, static=1)
        
# 如果是直接运行本脚本，则执行以下内容
if __name__ == '__main__':
    mainUSE()