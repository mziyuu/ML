"""
* @author   孟子喻
* @time     2020.6.2
* @file     HMM.py
"""
import numpy as np

class HMM():
    def __init__(self, A, B, Pi):
        self.A = A      # 状态转移概率矩阵
        self.B = B      # 观测概率矩阵
        self.Pi = Pi    # 初始状态序列

    def forward(self, sequence, t):
        """计算前向概率
        :param t        观测时间
        :param sequence T时刻的观测序列
        :return alpha   前向概率
        @reference Page198 算法10.2
        """
        alpha = self.Pi * self.B[:, sequence[0]]   # alpha初值 Page198 10.15
        for i in range(t):
            alpha = (alpha * self.A.T).T
            alpha = np.sum(alpha, axis=0) * self.B[:, sequence[i+1]]  # 递推 Page109 10.16
        return alpha

    def backward(self, sequence, t):
        """计算后向概率
        :param t        观测时间
        :param sequence T时刻的观测序列
        :return beta    后向概率
        @reference Page201 算法10.3
        """
        beta = np.ones(self.A.shape[0])
        for _t in range(len(sequence)-t-1):
            beta = beta * self.B[:, sequence[len(sequence) - _t - 1]] * self.A
            beta = np.sum(beta, axis=1)
        return beta

    def cal_prob(self, sequence, t, state):
        """
        :param t:        观测时间
        :param sequence: T时刻的观测序列
        :param state:    当前状态
        :return:prob     当前状态概率
        """
        alpha = self.forward(sequence, t)
        beta = self.backward(sequence, t)
        prob = alpha[state] * beta[state] / np.sum(alpha * beta)
        return prob

    def viterbi(self, sequence):
        """维特比(viterbi)路径
        :param sequence: T时刻的观测序列
        :return:path     路径
        """
        T = len(sequence)
        N = A.shape[0]
        path = np.zeros(T)
        delta = np.zeros((T, N))
        Phi = np.zeros((T, N))
        for i in range(N):
            delta[0][i] = self.Pi[i] * self.B[i][sequence[0]]
            Phi[0][i] = 0
        for t in range(1, T):
            for i in range(N):
                a = []
                for j in range(N):
                    a.append(delta[t - 1][j] * self.A[j][i])
                delta[t][i] = np.max(a) * self.B[i][sequence[t]]
                Phi[t][i] = np.argmax(a, axis=0) + 1
        path[T - 1] = np.argmax(delta[T - 1], axis=0) + 1
        for t in range(T-2, -1, -1):
            path[t] = Phi[t + 1][int(path[t + 1] - 1)]
        for i in range(0, len(path)):
            path[i] = int(path[i])
        return path

if __name__ == "__main__":
    sequence = np.array([0, 1, 0, 0, 1, 0, 1, 1])   # {"red": 0, "white": 1}
    A = np.array([[0.5, 0.1, 0.4],  # 状态转移概率矩阵
                  [0.3, 0.5, 0.2],
                  [0.2, 0.2, 0.6]])
    B = np.array([[0.5, 0.5],       # 观测概率矩阵
                  [0.4, 0.6],
                  [0.7, 0.3]])
    Pi = np.array([0.2, 0.3, 0.5])  # 初始状态概率向量
    t = 3
    model = HMM(A, B, Pi)
    """计算概率"""
    prob = model.cal_prob(sequence, t, 2)  # 2:"box_2"
    print("probability:\t", str(prob))

    """viterbi路径"""
    path = model.viterbi(sequence)     # {"box_0": 0, "box_1": 1, "box_2": 2}
    print('viterbi path:\t', path)
