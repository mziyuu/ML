import numpy as np
import pandas as pd


class PCA():
    """定义PCA类"""
    def __init__(self, x, threshhold):
        """x的数据结构应为ndarray"""
        self.x = x                   # 输入训练集
        self.x_num = x.shape[0]      # 输入训练集样本数量
        self.dimension = x.shape[1]  # 数据维度
        self.trans_matrix = None                # 转移矩阵（代表降维所用的正交基及其重要程度）
        self.threshhold = threshhold

    def cov(self):
        """求x的协方差矩阵"""
        x_T = np.transpose(self.x)  # 矩阵转置
        x_cov = np.cov(x_T)  # 协方差矩阵
        return x_cov

    def get_feature(self):
        """求协方差矩阵C的特征值和特征向量"""
        x_cov = self.cov()  # 返回协方差矩阵
        cov_eigenvalues, cov_eigenvectors = np.linalg.eig(x_cov)  # eigenvalues为特征值，eigenvectors为特征向量

        # 将特征值在水平方向上平铺构建特征矩阵
        cov_eigen_num = cov_eigenvalues.shape[0]  # 特征值数量
        cov_eigenmatrix = np.hstack((cov_eigenvalues.reshape((cov_eigen_num, 1)), cov_eigenvectors))

        # 使用np.hstack()时注意只允许各矩阵（向量）第一个维度不同
        cov_eigenmatrix_df = pd.DataFrame(cov_eigenmatrix)  # 转为DataFrame类型
        cov_eigenmatrix_df_sort = cov_eigenmatrix_df.sort_values(by=0, ascending=False)  # 按照特征值大小降序排列特征向量
        return cov_eigenmatrix_df_sort  # 返回的是按照特征值排序的协方差矩阵的特征矩阵

    def reduce_train_dimension(self):
        """根据方差贡献度自动降维"""
        cov_eigenmatrix_sort = self.get_feature()  # 获取输入的X的特征矩阵,并按照特征值大小降序排列特征向量

        # 协方差矩阵最大特征值对应的特征向量就是样本方差最大的方向！
        varience = cov_eigenmatrix_sort.values[:, 0]  # 取最大特征值对应的特征向量

        # 计算每个维度的方差在总方差中所占的比例
        varience_sum = sum(varience)
        varience_radio = varience / varience_sum

        varience_contribution = 0
        for R in range(self.dimension):
            varience_contribution += varience_radio[R]    # 当前R个方差已经占到总方差的99%时，舍弃剩下维度的特征
            if varience_contribution >= self.threshhold:  # 当不存在能使方差占比
                break

        self.trans_matrix = cov_eigenmatrix_sort.values[0:R + 1, 1:]  # 取前R个特征向量,获得转移矩阵，此时R就是降维后的维度

        y = np.dot(self.trans_matrix, np.transpose(self.x))  # 矩阵叉乘得到降维后的数据
        return np.transpose(y), self.trans_matrix

    def reduce_test_dimension(self, test):
        """对测试集进行降维"""
        y = np.dot(self.trans_matrix, np.transpose(test))  # 矩阵叉乘得到降维后的数据
        return np.transpose(y)

"""
if __name__ == '__main__':
    pca = PCA(X)
    y = pca.reduce_dimension()
"""
