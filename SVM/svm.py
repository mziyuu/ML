from __future__ import division, print_function
import os
import numpy as np
import random as rnd
filepath = os.path.dirname(os.path.abspath(__file__))

class SVM():
    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.0001):
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic
        }                                       # 根据不同的核函数实现线性与非线性支持向量机
        self.max_iter = max_iter                # 最大迭代次数，超过将自动退出
        self.kernel_type = kernel_type          # 选择核函数
        self.C = C                              # C为惩罚参数，C越大对误分类的惩罚越大
        self.epsilon = epsilon                  # 设置允许差错的范围
    def train(self, X, y):
        # 初始化
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))                   # 取初值拉格朗日乘子alpha全为0
        kernel = self.kernels[self.kernel_type] # 设置核函数
        count = 0                               # 计算迭代次数
        while True:
            count += 1
            alpha_prev = np.copy(alpha)         # 将alpha深拷贝
            # print(alpha.shape)
            # print(alpha)

            for j in range(0, n):
                i = self.get_rnd_int(0, n-1, j)                     # 随机获取不同的i与j，得到两个优化变量
                x_1, x_2, y_1, y_2 = X[i, :], X[j, :], y[i], y[j]   # 储存实例的特征和标签
                k_ij = kernel(x_1, x_1) + kernel(x_2, x_2) - 2 * kernel(x_1, x_2)
                if k_ij == 0:                                       # k_ij
                    continue                                        # 保证分子不为0
                alpha_prime_2, alpha_prime_1 = alpha[j], alpha[i]
                # 求alpha2所在对角线端点的边界，即alpha2的取值范围
                (L, H) = self.compute_L_H(self.C, alpha_prime_2, alpha_prime_1, y_2, y_1)
                # print(L, H)
                # 计算weight和bias
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # 计算x_i，x_j的预测值与真实值的误差
                E_i = self.calc_E(x_1, y_1, self.w, self.b)
                E_j = self.calc_E(x_2, y_2, self.w, self.b)

                # 求出alpha2未经剪辑的解
                alpha[j] = alpha_prime_2 + float(y_2 * (E_i - E_j))/k_ij
                # 利用求出的L,H对alpha2进行剪辑
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                # 用alpha2反推alpha1
                alpha[i] = alpha_prime_1 + y_1*y_2 * (alpha_prime_2 - alpha[j])

            # 检查是否超出误差允许范围
            diff = np.linalg.norm(alpha - alpha_prev)  # 求范数
            if diff < self.epsilon:
                break

            # 如果超出设定的最大迭代次数仍未求出最优解，则返回
            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return

        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
            # 临输出前计算最终的weight和bias
        return count

    def predict(self, X):
        return self.decision_f(X, self.w, self.b)
    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y))

    def decision_f(self, X, w, b):
        # 决策函数，即对输入进行预测
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    def calc_E(self, x_k, y_k, w, b):
        # 求E，即g(x)对输入x_k的预测值y_k与真实值之差
        return self.decision_f(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        # 求alpha2所在对角线端点的边界，即alpha2的取值范围
        # print(C, alpha_prime_j, alpha_prime_i, y_j, y_i)
        if(y_i != y_j):         # 若非同类
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:                   # 若为同类
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
    def get_rnd_int(self, a,b,z):
        i = z
        cnt = 0
        while i == z and cnt<1000:
            i = rnd.randint(a,b)
            cnt = cnt+1
        return i
    # 定义核函数
    def kernel_linear(self, x1, x2):
        # 线性核函数
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2):
        # 二次核函数
        return (np.dot(x1, x2.T) ** 2)
