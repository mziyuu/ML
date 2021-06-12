"""
* @author   孟子喻
* @time     2021.4.16
* @file     classify_iris.py
*           svm.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from svm_mulity import MulitySVM
from pca import PCA
from load_data import create_iris_data
from load_data import create_wine_data
from load_data import create_mnist_data
from load_data import create_cifar10_data


def calc_acc(y, y_hat):
    """计算准确率"""
    acc = 0
    if type(y_hat) != 'numpy.ndarray':
        y_hat = np.array(y_hat)
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            acc += 1/len(y)
    return acc


def get_sup_vc(data, w, b):
    """获取支持向量"""
    distance = get_dist(data[0], w, b)
    sup_vc   = data[0]
    for x in data:
        if get_dist(x, w, b) < distance:
            sup_vc = x
            distance = get_dist(x, w, b)
    return sup_vc


def get_dist(features, w, b):
    """计算欧氏距离"""
    return abs(np.dot(features, w) + b)


def showpoints(data, w, b):
    """绘图"""
    positive_x = []
    positive_y = []
    negative_x = []
    negative_y = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for item in data:
        if item[-1] == 1:
            positive_x.append(item[0])
            positive_y.append(item[1])
        if item[-1] == -1:
            negative_x.append(item[0])
            negative_y.append(item[1])
        ax.scatter(positive_x, positive_y, color="r")
        ax.scatter(negative_x, negative_y, color="b")
    x = np.arange(3, 7, 0.1)
    y = []
    for item in x:
        y.append(-w[0]/w[1]*item - b/w[1])
    ax.plot(x, y)
    sup_vc = get_sup_vc(data[:, 0:2], w, b)

    ax.scatter(sup_vc[0], sup_vc[1], color="g")
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.show()

def divide_train_test(features, labels):
    """划分测试集与训练集"""
    # 前80%为训练集，后20%为测试集
    data_size = len(labels)
    train_feature = features[0:int(0.8*data_size)]
    train_label = labels[0:int(0.8*data_size)]
    test_feature = features[int(0.8*data_size):-1]
    test_label = labels[int(0.8 * data_size):-1]
    return train_feature, train_label, test_feature, test_label


if __name__ == "__main__":

    dataset = ['iris', 'wine', 'mnist', 'cafir10']
    data = 'cifar10'                       # 选择数据集
    rgb2gray = False                    # 是否将图像转为灰度图

    dim_reduce = True                  # 是否进行pca降维
    threshholds = {'wine': 0.999999,     # pca降维时方差阈值设置
                   'mnist': 0.99,
                   'cifar10': 0.99
                   }
    threshhold = threshholds[data]


    if data == 'iris':
        trainset = create_iris_data()
        features, labels = trainset[:, :4], trainset[:, -1]

    if data == 'wine':
        trainset = create_wine_data()
        features, labels = trainset[:, :13], trainset[:, -1]

    if data == 'mnist':
        features, labels = create_mnist_data()
        features, labels = features[0:10000], labels[0:10000]

    if data == 'cifar10':
        features, labels = create_cifar10_data(rgb2gray)
        features, labels = features[0:5000], labels[0:5000]
        # 仅在样本数量大于特征数量时才可以使用pca降维

    train_feature, train_label, test_feature, test_label = divide_train_test(features, labels)

    """PCA降维"""
    if dim_reduce == True:
        print("开始PCA降维,数据特征维度为", str(train_feature.shape[1]))
        pca = PCA(train_feature, threshhold)
        train_feature, trans_matrix = pca.reduce_train_dimension()  # trans_matrix是降维的转换矩阵
        test_feature = pca.reduce_test_dimension(test_feature)
        print("PCA降维完成,数据特征维度为", str(train_feature.shape[1]))

    """Train"""
    time_start = time.time()
    # 因为有些参数对多分类SVM的结构是有影响的，所以在初始化模型时便传入特征和标签
    model = MulitySVM(train_feature, train_label)
    # 训练模型
    model.train()

    """"""
    # 计算准确率
    y_hat = model.predict(test_feature)
    acc = calc_acc(test_label, y_hat)
    print("Accuracy: ", str(acc*100), "%")
    time_end = time.time()
    print("训练总用时：", str(time_end-time_start))
