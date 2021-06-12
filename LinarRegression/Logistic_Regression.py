"""
* @author   孟子喻
* @time     2021.4.19
* @file     Logistic_Regression.py
*           make_dataset.py
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
# 此处用优化后的sigmoid，不然会报overflow的warning（其实不要紧）
def sigmoid(x):
    if x>0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def plotRegressionResult(weights):
    dataset, labels = read_data()
    n = np.shape(dataset)[0]
    x1_positive = []
    y1_positive = []
    x2_negative = []
    y2_negative = []
    for i in range(n):
        if labels[i] == 1:
            x1_positive.append(dataset[i][1])
            y1_positive.append(dataset[i][2])
        else:
            x2_negative.append(dataset[i][1])
            y2_negative.append(dataset[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1_positive, y1_positive, s=30, c='red')
    ax.scatter(x2_negative, y2_negative, s=30, c='green')
    x = np.arange(-10, 100, 0.1)
    y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 该函数用于读取x1、x1和label（即正负类）
def read_data():
    dataset = np.loadtxt('data_C.csv')
    # print(data)  # 测试查看data形状
    data = dataset[:, 0:-1]  # 表示读取所有数据的对应行
    labels = dataset[:, -1]  # 读标签
    data = np.insert(data, 0, 1, axis=1)  # 添加1是为了把w*x+b简化为w*x
    print(data)
    return data, labels



def SGD(dataset, labels, epoch=1000):
    data_size, x_dim = np.shape(dataset)
    print(data_size, x_dim)
    weights = np.ones(x_dim)  # 把weight初始化一下
    for j in range(epoch):
        dataIndex = list(range(data_size))
        for i in range(data_size):
            lr = 4 / (1 + i + j) + 1
            # 随机取一个错误实例点
            randIndex = int(np.random.uniform(0, len(dataIndex)))

            h = sigmoid(sum(dataset[i] * weights))
            error = labels[i] - h
            weights = weights + lr * error * dataset[i]
            del(dataIndex[randIndex])
    return weights



if __name__ == '__main__':
    dataset, labels = read_data()
    r = SGD(dataset, labels)
    print(r)
    r = np.mat(r).transpose()
    plotRegressionResult(r)
