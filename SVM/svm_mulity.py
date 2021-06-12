"""
* @author   孟子喻
* @time     2021.6.4
* @file     svm_mulity.py
*           classify_iris_mulity.py
*           svm.py
"""
from svm import SVM
import numpy as np
from collections import Counter

class MulitySVM:
    def __init__(self, features, labels, C=1.0, max_iter=10000):
        self.features = features
        # 在模型训练前把label转换为[0, N],N为label类型数，用一个字典来保存这个对应关系
        self.dic, self.labels = self.label_transform(labels)
        self.class_type = np.unique(labels)     # 存储所有class类型
        self.class_num = len(self.class_type)   # label数量
        self.clf_num = self.cal_clf_num()       # 单分类器数量

        # SVM单分类器的参数
        self.C = C
        self.max_iter = max_iter

        self.clf = []        # SVM单分类器列表
        self.clf_class = []  # 单分类器与classes的对应关系
        # 生成self.clf_class
        self.assign_clf_class()

    def assign_clf_class(self):
        """将单分类器与他所分类的classes对应"""
        # 每个单分类器负责的classes，clf_class[i][0]代表单分类器编号,clf_class[i][1]和clf_class[i][2]代表对应的class编号
        class_num = self.class_num
        for class_i in range(0, class_num):  # 取clf_class[i][1]
            for class_j in list(range(class_i+1, class_num)):  # 取clf_class[i][2]
                self.clf_class.append([class_i, class_j])

    def cal_clf_num(self):
        """计算分类器数量"""
        # 使用one vs one 方式进行多分类器构建，每两类之间用一个clf进行划分，共需要(k(k-1))/2个
        clf_num = (self.class_num * (self.class_num - 1)) / 2
        return int(clf_num)

    def train(self):
        for i in range(0, self.clf_num):
            print("正在训练第", str(i+1), "个二分类器，共有", str(self.clf_num), "个")
            feature = []    # 当前分类器所分两类的feature
            label   = []    # 当前分类器所分两类的label
            class_1 = self.clf_class[i][0]
            class_2 = self.clf_class[i][1]
            for j in range(0, len(self.labels)):
                # 遍历数据集找出所分的两类，将他们的特征和标签加入feature和label
                if self.labels[j] == class_1:
                    feature.append(self.features[j])
                    label.append(self.labels[j])
                if self.labels[j] == class_2:
                    feature.append(self.features[j])
                    label.append(self.labels[j])
            svm = SVM()

            feature = np.array(feature)
            # 因为我之前写的SVM只支持-1，+1输入，所以还需要再转一下，利用self.clf.class即可，label较小的为-1，较大的为+1
            min_label = min(class_1, class_2)
            for item in range(len(label)):
                if label[item] == min_label:
                    label[item] = -1
                else:
                    label[item] = 1
            svm.train(feature, label)
            # print("W:",str(svm.w))
            # print("B;",str(svm.b))
            self.clf.append(svm)    # self.clf中的svm应该与self.clf_class一一对应

    def predict(self, feature):
        """所有二分类器对结果进行预测，最终所有分类器投票"""
        # 此处输入的feature是待预测的feature
        pred = []   # pred的每个元素是当前分类器对所有实例的预测
        pred_result = []  # 存放最后预测结果（未对应回原label）

        # for model in self.clf:
        for i in range(len(self.clf)):
            model = self.clf[i]
            # 用所有分类器进行预测，结果存入single_temp
            pred_temp = model.predict(feature)
            # 注意svm输出结果只有+1和-1,所以要借助self.clf_class转换一下

            for j in range(len(pred_temp)):
                if pred_temp[j] == 1:
                    pred_temp[j] = max(self.clf_class[i][0], self.clf_class[i][1])
                else:
                    pred_temp[j] = min(self.clf_class[i][0], self.clf_class[i][1])
            pred.append(pred_temp)

        for i in range(0, np.shape(feature)[0]):  # 对每个实例的分类结果,i代表一个实例
            pred_single = []  # 存放每个分类器对当前feature的预测值
            for j in range(0, self.clf_num):  # 当前特征下每个二分类器的结果,j代表一个二分类器
                pred_single.append(pred[j][i])
            pred_result.append(self.find_most_elem(pred_single))

        # 将svm中的新的label转回原问题的label
        pred_final = []
        for label in pred_result:
            for key, value in self.dic.items():
                if label == value:
                    pred_final.append(key)

        return pred_final

    def find_most_elem(self, data):
        """寻找出现最多的元素"""
        result = None  # 存放
        result_dict = Counter(data)  # 获取每个元素出现次数的字典
        for key, value in result_dict.items():
            if value == max(result_dict.values()):
                result = key
        return result

    def label_transform(self, labels):
        """在模型训练前把label转换为[0, N],N为label类型数，用一个字典来保存这个对应关系"""
        label_type = np.unique(labels)  # 获取转换前的label类型
        new_label = 0       # 转换后的label为从0到N的整数
        dic = {}            # label转换前后对应关系字典

        for label in label_type:
            dic[label] = new_label  # key为原label，value为新label
            new_label += 1
        new_labels = []  # 转换后的label
        for label in labels:
            new_labels.append(dic[label])

        return dic, new_labels












