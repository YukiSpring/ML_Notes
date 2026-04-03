'''
date: 2026/04/1
id: 23375158
description:
        程序生成了一个3D的数据集。一共有1000个数据，被分成了两大类：C0与C1。 
        请利用该数据做训练，
        同时利用程序新生成与训练数据同分布的500个数据（250个为C0类，250个数据为C1类）来做测试。
        比较利用Decision Trees, AdaBoost + DecisionTrees, 与SVM的分类性能。
        并讨论对于这个问题，为什么某些算法表现相对更好。 
        其中SVM至少选用三种不同的Kernel Fucntions. 
'''

# Generating 3D make-moons data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

# Generate the data (1000 datapoints)
# X, labels = make_moons_3d(n_samples=1000, noise=0.2)

# Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
# legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend1)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('3D Make Moons')
# plt.show()

x_train, y_train = make_moons_3d(n_samples=1000, noise=0.2)  # Generate training data
x_test, y_test = make_moons_3d(n_samples=500, noise=0.2)  # Generate test data

y_train = y_train.astype(int)
y_test = y_test.astype(int)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], c=y_test, cmap='viridis', marker='o')
# legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend1)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('3D Make Moons Test Data')
# plt.show()

def plot_3d_error(X_test, y_test, y_pred, title="Model Error Analysis"):

    correct_idx = (y_pred == y_test)
    incorrect_idx = ~correct_idx

    fig = plt.figure(figsize=(10, 8)) 
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_test[correct_idx, 0], X_test[correct_idx, 1], X_test[correct_idx, 2],
               c=y_test[correct_idx], cmap='viridis', marker='o', s=20, alpha=0.3, label='Correct (C0/C1 background)')

    if np.any(incorrect_idx): 
        ax.scatter(X_test[incorrect_idx, 0], X_test[incorrect_idx, 1], X_test[incorrect_idx, 2],
                   c='red', marker='x', s=40, alpha=1.0, label='Incorrect (Errors)')

    ax.set_xlabel('Dimension X', fontsize=12)
    ax.set_ylabel('Dimension Y', fontsize=12)
    ax.set_zlabel('Dimension Z', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10) 

    plt.tight_layout() 
    plt.show()


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 维度
        self.threshold = threshold  # 分割点
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 0/1
    
# 计算Gini指数 1-p0^2-p1^2
def gini(labels):
    if len(labels) == 0:
        return 0 
    p0 = np.sum(labels == 0) / len(labels)
    p1 = np.sum(labels == 1) / len(labels)
    return 1 - p0**2 - p1**2

def best_split(X, y):
    best_split={}
    best_gini = 1.0

    n_samples, n_features = X.shape #(n_samples, dimensions)
    # n_features=3 
    for d_idx in range(n_features):
        # 当前维度 所有可能的分割点 放入thresholds
        thresholds = np.unique(X[:, d_idx])
        # 遍历每个分割点，计算Gini指数
        for threshold in thresholds:
            left_idx=X[:, d_idx] < threshold
            right_idx=~left_idx
            # 边缘排除
            if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                continue
            
            gini_left = gini(y[left_idx])
            gini_right = gini(y[right_idx])

            n=len(y)
            nl=len(y[left_idx])
            nr=len(y[right_idx])
            gini_split = (nl/n)*gini_left + (nr/n)*gini_right

            # 更新best_split
            if gini_split < best_gini:
                best_gini = gini_split
                best_split = {
                    'feature': d_idx,
                    'threshold': threshold,
                    'left_idx': left_idx,
                    'right_idx': right_idx
                }
    return best_split

def build_tree(X, y, max_depth=None, depth=0):
    n_samples=len(y)
    n_labels=len(np.unique(y))

    # 停止条件：所有样本属于同一类/达到最大深度
    if n_labels == 1 or depth>=max_depth or n_samples<2:
        leaf_value = np.bincount(y).argmax()  # 选择出现频率最高的类别作为叶节点的值
        return Node(value=leaf_value)
    
    # 寻找best split
    split=best_split(X, y)

    # 递归
    left_subtree=build_tree(X[split['left_idx']], y[split['left_idx']], max_depth, depth+1)
    right_subtree=build_tree(X[split['right_idx']], y[split['right_idx']], max_depth, depth+1)

    return Node(feature=split['feature'], threshold=split['threshold'], 
                left=left_subtree, right=right_subtree)

def predict_one(node,x):
    if node.value is not None:  #叶节点
        return node.value
    if x[node.feature] < node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

def predict(node, X):
    return np.array([predict_one(node, x) for x in X])


# train
Dtree=build_tree(x_train, y_train, max_depth=5)

# test
y_pred_dt=predict(Dtree, x_test)

# accuracy
acc_dt=np.mean(y_pred_dt==y_test)
print(f"Decision Tree Accuracy: {acc_dt:.4f}")

# plot
plot_3d_error(x_test, y_test, y_pred_dt, title="Decision Tree Error Analysis")