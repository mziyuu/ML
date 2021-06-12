from __future__ import print_function
import numpy as np
import os
import struct
import platform
import pandas as pd
from six.moves import cPickle as pickle
from PIL import Image


def create_iris_data():
    """加载iris鸢尾花数据集"""
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:150, [0, 1, -1]])
    print("已加载iris鸢尾花数据集")
    return data


def create_wine_data():
    """加载wine红酒数据集"""
    from sklearn.datasets import load_wine
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['label'] = wine.target
    print("已加载wine红酒数据集")
    return df.values


def create_mnist_data():
    """加载mnist手写数字数据集"""
    image_path = "mnist/train-images.idx3-ubyte"
    label_path = "mnist/train-labels.idx1-ubyte"
    feature = decode_idx3_ubyte(image_path)
    # 因为其他数据集都是1*N的，而mnist是28*28的，所以要先拉成1*784的
    image_num = feature.shape[0]
    feature = feature.reshape(image_num, 28*28)
    feature = feature+1
    label   = decode_idx1_ubyte(label_path)
    print("已加载mnist手写数据数据集")
    return feature, label


def create_cifar10_data(gray):
    """加载cifar10数据集"""
    cifar_path = "cifar10"
    images, labels = load_CIFAR10(cifar_path)
    # 转灰度
    image_num = images.shape[0]
    if gray == True:
        images_gray = rgb2gray(images)
        # numpy array使用连续内存块，所以应一开始就定义好大小,然后根据索引对其赋值
        features = np.zeros((len(images_gray), 32, 32))
        for i in range(0, len(images_gray)):
            image_gray = images_gray[i]
            feature = np.array(image_gray)
            features[i] = feature
        features = features.reshape(image_num, 32 * 32)
    else:
        features = images.reshape(image_num, 32 * 32 * 3)
    print("已加载cifar10数据集")
    return features, labels


def rgb2gray(images):
    """RGB图像转为灰度图"""
    image_gray = []
    for image in images:
        image_i = Image.fromarray(np.uint8(image))
        image_gray.append(image_i.convert('L'))
    return image_gray


def load_pickle(f):
    """使用pickle加载文件"""
    version = platform.python_version_tuple()  # 取python版本号
    if version[0] == '2':
        return pickle.load(f)  # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """加载一定batch数量的cifar10数据集"""
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)  # dict类型
        X = datadict['data']  # X, ndarray, 像素值
        Y = datadict['labels']  # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """加载路径下所有cifar10数据集"""
    xs = []  # list
    ys = []

    # 训练集batch 1～5
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
        ys.append(Y)
    Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
    Ytr = np.concatenate(ys)
    del X, Y

    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr


def decode_idx1_ubyte(idx1_ubyte_file):
    """读取idx1二文件"""
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def decode_idx3_ubyte(idx3_ubyte_file):
    """读取idx3文件"""
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，
    # 指针位置（即偏移位置offset）指向0016。
    fmt_image = '>' + str(image_size) + 'B'
    # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据
    # 如果没有则只会读取一个值（即一副图像中的一个像素值）
    images = np.empty((num_images, num_rows, num_cols))

    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
    return images
