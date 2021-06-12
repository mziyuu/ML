"""
* @author   孟子喻
* @time     2021.4.16
* @file     make_dataset.py
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

def make_dataset_A(data_size,  weight, bias):
    """
    设置一条线划两个区间，直线以上的点为正类，以下的点为负类
    """
    x = 100*np.random.rand(data_size)
    dataset = []
    num = 0
    for item in x:
        # 设一半正类一半负类
        if (num % 2) == 0:
            y = item * weight + bias - 10*np.random.rand() - 5
            dataset.append([item, y, 0])
        else:
            y = item * weight + bias + 5*np.random.rand() + 5
            dataset.append([item, y, 1])
        num += 1
    show_dataset(dataset)
    resize_dataset = []
    for data in dataset:
        resize_dataset.append([[data[0], data[1], 1], data[2]]) # 把bias并入weight

    with open("data_A.csv", "w")as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(dataset)

    return resize_dataset

def make_dataset_B(data_size, weight, bias):
    """
    设置一条线划两个区间，直线以上的点为正类，以下的点为负类
    """
    weight = 1
    bias = 0
    dataset = []
    while data_size > 0:
        data = []
        x1 = 100*np.random.rand()
        x2 = 100*np.random.rand()
        data.append(x1)
        data.append(x2)
        if x1*weight+bias<x2:
            data.append(1)
        else:
            data.append(0)
        dataset.append(list(data))
        data_size -= 1

    show_dataset(dataset)

    with open("data_B.csv", "w")as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(dataset)

    return dataset

def make_dataset_C(data_size, weight, bias):
    """
    设置一条线划两个区间，直线以上的点为正类，以下的点为负类
    """
    weight = 1
    bias = 0
    dataset = []
    while data_size > 0:
        data = []
        x1 = 100*np.random.rand()
        x2 = 100*np.random.rand()
        data.append(x1)
        data.append(x2)
        if x1*weight+bias<x2:
            data.append(1)
        else:
            data.append(0)
        dataset.append(list(data))
        data_size -= 1

    for data in dataset:
        if data[2] == 0:
            data[2] = 1
            break
    for data in dataset:
        if data[2] == 1:
            data[2] = 0
            break

    show_dataset(dataset)


    with open("data_C.csv", "w")as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(dataset)

    return dataset

def show_dataset(dataset):
    positive_x = []
    positive_y = []
    negative_x = []
    negative_y = []
    for item in dataset:
        if item[2] == 1:
            positive_x.append(item[0])
            positive_y.append(item[1])
        if item[2] == 0:
            negative_x.append(item[0])
            negative_y.append(item[1])
        plt.scatter(positive_x, positive_y, color="r")
        plt.scatter(negative_x, negative_y, color="b")
        plt.show()

if __name__ == "__main__":
    data_size = 100
    weight, bias = 1, 0
    dataset = make_dataset_A(data_size, weight, bias)