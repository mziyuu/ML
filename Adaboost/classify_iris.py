"""
* @author   孟子喻
* @time     2021.5.16
* @file     classify_iris.py
*           adaboost.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from adaboost import AdaBoost
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
  
# 鸢尾花(iris)数据集
# 数据集内包含 3 类共 150 条记录，每类各 50 个数据，
# 每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
# 可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。
# 这里只取前100条记录，两项特征，两个类别。

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data

def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN)/len(y)


if __name__ == "__main__":

    trainset = create_data()

    features, labels = trainset[:, :2], trainset[:, -1]
    x = features
    y = labels

    ada = AdaBoost()
    clf_num = ada.fit(x, y)  # 返回使用的弱分类器数量
    print(clf_num)
    ada.show_para() # 显示一些参数

    # 进行预测
    y_hat = ada.output_final_pred(x, clf_num)

    acc = calc_acc(y, y_hat)    # 计算准确率
    print("Accuracy: ", str(100* acc), "%")

    plt.figure()

    for i in range(clf_num):
        if ada.classifier[i].decision_feature == 0:
            plt.plot([ada.classifier[i].decision_threshold, ada.classifier[i].decision_threshold], [0, 10])
        else:
            plt.plot([0, 10], [ada.classifier[i].decision_threshold, ada.classifier[i].decision_threshold])

    for item in trainset:
        if item[-1] == 1:
            plt.scatter(item[0], item[1], s=30, marker='+', color='blue')
        else:
            plt.scatter(item[0], item[1], s=30, marker='_', color='red')
    plt.xlim(3, 8)
    plt.ylim(1, 6)
    plt.show()
