"""
* @author   孟子喻
* @time     2021.5.30
* @file     EM.py
* @reference 《统计学习方法》Page183-Page187 especially 算法9.2 on Page187
"""
import copy
import numpy as np


def creat_data(N, u, alpha, sigma):
    """生成两个随机分布的高斯模型
    :return X   生成的数据集X
    """
    X = np.zeros(N)         # 初始化X
    for i in range(N):
        if np.random.random(1) < alpha[0]:
            # 40%的概率用第一个高斯模型生成数据
            X[i] = np.random.normal(u[0], sigma[0], None)
        else:
            # 60%的概率用第二个高斯模型生成数据
            X[i] = np.random.normal(u[1], sigma[1], None)
    return X



def E_Step(sigma, alpha, k, N, E, X):
    """E步：依据当前模型参数计算模型对观测数据的响应度
    :return E 模型对观测数据的响应度
    """
    for i in range(N):
        denominator = 0
        for j in range(0, k):
            # 课本P186算法9.2公式（2）
            denominator += alpha[j] * np.exp(-(X[i] - u[j]) * sigma[j] * np.transpose(X[i] - u[j])) / np.sqrt(
                sigma[j])  # 分母
        for j in range(0, k):
            numerator = np.exp(-(X[i] - u[j]) * sigma[j] * np.transpose(X[i] - u[j])) / np.sqrt(sigma[j])  # 分子
            E[i, j] = alpha[j] * numerator / denominator  # 求期望
    return E


def M_Step(alpha, k, N, E, X):
    """M步：计算新一轮迭代的模型参数u和alpha
    :return alpha   预测的数据分布alpha
    :return u       预测的均值u
    """
    for j in range(0, k):
        denominator = 0  # 分母
        numerator = 0    # 分子
        for i in range(N):
            numerator += E[i, j] * X[i]
            denominator += E[i, j]
        u[j] = numerator / denominator
        alpha[j] = denominator / N     # 课本P186算法9.2公式（3）
    return u, alpha


if __name__ == '__main__':

    """受随机性影响较大，如果预测值与真实值差异较大，可以调小epsilon重复几次"""
    max_iter = 100      # 最大迭代次数
    N = 100             # 样本数目
    k = 2               # 高斯模型数
    u = [0, 20]         # 设定两种模式的均值
    epsilon = 0.00001   # 设定允许误差
    alpha = [0.25, 0.75]  # 设定混合项系数,即两种模型所占比例
    sigma = [np.sqrt(10), np.sqrt(20)]  # 标准差分别为sqrt(10)和sqrt(20)

    # 生成数据
    X = creat_data(N, u, alpha, sigma)

    E = np.zeros((N, 2))        # 初始化E,模型对观测数据的响应度，其值为第i个样本属于第j个模型的概率的期望
    # 迭代计算
    i = 0
    for i in range(max_iter):
        error = 0
        alpha_error = 0
        old_u = copy.deepcopy(u)        # 注意对u和alpha进行深拷贝
        old_alpha = copy.deepcopy(alpha)
        E = E_Step(sigma, alpha, k, N, E, X)  # E步
        u, alpha = M_Step(alpha, k, N, E, X)  # M步
        print("Iter:", i + 1)
        print("u_hat:\t\t", u)
        print("alpha_hat:\t", alpha, '\n')

        break_sign = False
        for j in range(0, k):
            # 判断是否在误差允许范围内
            if abs(u[j]-old_u[j]) < epsilon and \
                    abs(alpha[j]-old_alpha[j]) < epsilon:
                break_sign = True

        if break_sign == True:
            break

    print("Finished in ", str(i + 1), ' iters')
    print("u_hat:\t\t", u)
    print("alpha_hat:\t", alpha)
