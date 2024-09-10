# 写一个程序，读取当前目录下的BEN.npy文件，随机删除其中的一些行，最后剩余500行，将剩余的数据保存为BEN.npy文件
import numpy as np

# 读取当前目录下的BEN.npy文件
data = np.load('BEN.npy')

# 如果数据的行数大于500，随机删除一些行
if data.shape[0] > 500:
    # 随机选择要保留的500行的索引
    indices = np.random.choice(data.shape[0], 500, replace=False)
    data = data[indices]

# 将剩余的数据保存回BEN.npy文件
np.save('BEN.npy', data)

print(f"数据处理完成，最终保留了 {data.shape[0]} 行数据。")
