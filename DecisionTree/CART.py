
import numpy as np

class CARTTree:
    def __init__(self, value=None, trueBranch=None, falseBranch=None, results=None, col=-1, summary=None, data=None):
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.col = col
        self.summary = summary
        self.data = data

    def __str__(self):
        print(self.col, self.value)
        print(self.results)
        print(self.summary)
        return ""



def gini(dataset):
    # 计算基尼指数
    data_num = len(dataset)
    # 以下部分用于统计每个类别出现的个数，并组成一个字典，存在result中
    results = {}
    for data in dataset:
        # data[-1] means dataType
        if data[-1] not in results:
            results.setdefault(data[-1], 1)
        else:
            results[data[-1]] += 1

    gini_result = 0
    gini_result = float(gini_result)
    for i in results:
        gini_result += (results[i] / data_num) * (results[i] / data_num)
    return 1 - gini_result

def chooseSplitData(dataset, value, column):
    # 根据条件分离数据集(splitDatas by value, column)
    # return 2 part（list1, list2）

    lefttree = []
    righttree = []

    if isinstance(value, int) or isinstance(value, float):
        for row in dataset:
            if row[column] >= value:
                lefttree.append(row)
            else:
                righttree.append(row)
    else:
        for row in dataset:
            if row[column] == value:
                lefttree.append(row)
            else:
                righttree.append(row)
    return lefttree, righttree

def buildTree(rows):
    # 递归建立决策树， 当gain=0，时停止回归
    # build decision tree bu recursive function
    # stop recursive function when gain = 0
    # return tree
    currentGain = gini(rows)
    column_lenght = len(rows[0])
    rows_length = len(rows)

    best_gain = 0.0
    best_value = None
    best_set = None

    # choose the best gain
    for col in range(column_lenght - 1):
        col_value_set = set([x[col] for x in rows])
        for value in col_value_set:
            list1, list2 = chooseSplitData(rows, value, col)
            p = len(list1) / rows_length
            gain = currentGain - p * gini(list1) - (1 - p) * gini(list2)
            if gain > best_gain:
                best_gain = gain
                best_value = (col, value)
                best_set = (list1, list2)
    dcY = {'impurity': '%.3f' % currentGain, 'sample': '%d' % rows_length}
    #
    # stop or not stop

    if best_gain > 0:
        trueBranch = buildTree(best_set[0])
        falseBranch = buildTree(best_set[1])
        return CARTTree(col=best_value[0], value=best_value[1], trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY)
    else:
        return CARTTree(results=calculateDiffCount(rows), summary=dcY, data=rows)


def calculateDiffCount(datas):
    results = {}
    for data in datas:
        # data[-1] means dataType
        if data[-1] not in results:
            results.setdefault(data[-1], 1)
        else:
            results[data[-1]] += 1
    return results


def pruneTree(tree, miniGain):

    if tree.trueBranch.results == None:
        pruneTree(tree.trueBranch, miniGain)
    if tree.falseBranch.results == None:
        pruneTree(tree.falseBranch, miniGain)

    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        len1 = len(tree.trueBranch.data)
        len2 = len(tree.falseBranch.data)
        len3 = len(tree.trueBranch.data + tree.falseBranch.data)

        p = float(len1) / (len1 + len2)

        gain = gini(tree.trueBranch.data + tree.falseBranch.data) - p * gini(tree.trueBranch.data) - (1 - p) * gini(tree.falseBranch.data)

        if gain < miniGain:
            tree.data = tree.trueBranch.data + tree.falseBranch.data
            tree.results = calculateDiffCount(tree.data)
            tree.trueBranch = None
            tree.falseBranch = None

def classify(data, tree):
    if tree.results != None:
        return tree.results
    else:
        branch = None
        v = data[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        else:
            if v == tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        return classify(data, branch)


def loadCSV():
    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s
    data = np.loadtxt("name_datas.csv", dtype='str', delimiter=',')
    data = data[1:, :]
    dataSet =([[convertTypes(item) for item in row] for row in data])
    return dataSet

# 画树

if __name__ == '__main__':
    dataSet = loadCSV()
    decisionTree = buildTree(dataSet)
    pruneTree(decisionTree, 0.4)
    pre_name = [5.1, 3.5, 1.4, 0.2]

    print(classify(pre_name, decisionTree))
