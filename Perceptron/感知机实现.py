"""
    数据组织形式：
    每个实例点占一行，最后一个为label，其余为输入值，中间用空格隔开
"""
import numpy as np
import matplotlib.pyplot as plt

# 从data.txt中加载数据
data = []
label = []
file = open('Data.txt')
for line in file:
    line = line.split(' ')
    for i in range(len(line)):
        line[i] = float(line[i])
    data.append(line[0: len(line)-1])
    label.append(int(line[-1]))
file.close()
data = np.array(data)
label = np.array(label)
# 初始化alpha, w, b
alpha = 1
w = np.array([0, 0])
b = 0

# 根据y*(w*x+b)判断是否为误分类点
f = (np.dot(data, w.T) + b) * label
idx = np.where(f <= 0)
print(f)
# 对w, b使用SGD进行更新
iter = 1
while f[idx].size != 0:
    point = np.random.randint((f[idx].shape[0]))
    print(f[idx].shape[0])
    x = data[idx[0][point], :]
    y = label[idx[0][point]]
    w = w + alpha * y * x
    b = b + alpha * y
    print("Iter: ", iter, "\tw: ", w, "\tb: ", b)
    iter = iter + 1
    f = (np.dot(data, w.T) + b) * label
    idx = np.where(f <= 0)
    iteration = iter + 1
print(w)


x1 = np.arange(0, 6, 0.1)
# 避免w中某一维度为0造成无法正常除法
if -w[1] == 0:
    x2 = 0
else:
    x2 = (w[0] * x1 + b) / (-w[1])
idx_p = np.where(label == 1)
idx_n = np.where(label != 1)
data_p = data[idx_p]
data_n = data[idx_n]
plt.scatter(data_p[:, 0], data_p[:, 1], color='red')
plt.scatter(data_n[:, 0], data_n[:, 1], color='blue')
plt.plot(x1, x2)
plt.show()