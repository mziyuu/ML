"""
* @author   孟子喻
* @time     2021.4.16
* @file     classify_iris.py
*           svm.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from svm import SVM
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

def get_sup_vc(data, w, b):
    distance = get_dist(data[0], w, b)
    sup_vc   = data[0]
    for x in data:
        if get_dist(x, w, b) < distance:
            sup_vc = x
            distance = get_dist(x, w, b)
    return sup_vc

def get_dist(features, w, b):
    return abs(np.dot(features, w) + b)

def showpoints(data, w, b):
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


if __name__ == "__main__":
    trainset = create_data()

    # save data as csv when first run
    with open("iris_data.csv", "w") as csvdata:
        writer = csv.writer(csvdata, delimiter="\n")
        writer.writerows(trainset)

    features, labels = trainset[:, :2], trainset[:, -1]

    model = SVM()

    # 训练模型
    model.train(features, labels)

    # 计算准确率
    y_hat = model.predict(features)
    acc = calc_acc(labels, y_hat)

    print("bias:\t\t%.3f" % (model.b))
    print("w:\t\t" + str(model.w))
    print("accuracy:\t%.3f" % (acc))

    showpoints(trainset, model.w, model.b)

