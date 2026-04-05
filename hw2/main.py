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
import os


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

fig=plt.figure(figsize=(14,8))
ax1=fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], c=y_train, cmap='viridis', marker='o', alpha=0.5, label='Training Data')
ax1.set_title('3D Make Moons - Training Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.grid(linestyle='--', alpha=0.7)

ax2=fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], c=y_test, cmap='viridis', marker='o', alpha=0.5, label='Testing Data')
ax2.set_title('3D Make Moons - Testing Data')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.grid(linestyle='--', alpha=0.7)
save_path = 'hw2/asserts'
plt.savefig(os.path.join(save_path, '3d_make_moons.png'))
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
    plt.savefig(os.path.join('hw2/asserts', f'{title}.png'))
    # plt.show()


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
y_pred_dt_test=predict(Dtree, x_test)
y_pred_dt_train=predict(Dtree, x_train)
# accuracy
acc_dt_test=np.mean(y_pred_dt_test==y_test)
acc_dt_train=np.mean(y_pred_dt_train==y_train)
print(f"Decision Tree Accuracy: {acc_dt_test:.4f} (Test), {acc_dt_train:.4f} (Train)")

# plot
plot_3d_error(x_test, y_test, y_pred_dt_test, title="Decision Tree Test Error Analysis")
plot_3d_error(x_train, y_train, y_pred_dt_train, title="Decision Tree Train Error Analysis")

# AdaBoost

y_train_ada=2*y_train-1  # 转换为-1/1标签
y_test_ada=2*y_test-1
num=len(y_train_ada)
# weight
D=np.ones(num)/num  

def gini_ada(labels, weights):
    if len(labels) == 0:
        return 0 
    p0=np.sum(weights[labels==-1])/np.sum(weights)
    p1=np.sum(weights[labels==1])/np.sum(weights)
    return 1 - p0**2 - p1**2

def best_split_ada(X, y, D):
    best_split={}
    best_gini = 1.0

    n_samples, n_features = X.shape 
    for d_idx in range(n_features):
        thresholds = np.unique(X[:, d_idx])
        for threshold in thresholds:
            left_idx=X[:, d_idx] < threshold
            right_idx=~left_idx
            if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                continue
            
            gini_left = gini_ada(y[left_idx], D[left_idx])
            gini_right = gini_ada(y[right_idx], D[right_idx])

            n=np.sum(D)
            nl=np.sum(D[left_idx])
            nr=np.sum(D[right_idx])
            gini_split = (nl/n)*gini_left + (nr/n)*gini_right

            if gini_split < best_gini:
                best_gini = gini_split
                best_split = {
                    'feature': d_idx,
                    'threshold': threshold,
                    'left_idx': left_idx,
                    'right_idx': right_idx
                }
    return best_split

def build_tree_ada(X, y, D, M):
    forest=[]
    for i in range(M):
        split=best_split_ada(X, y, D)
        polarity=1
        y_pred_i=np.where(X[:, split['feature']] < split['threshold'], -1, 1)
        error_i=np.sum(D[y_pred_i != y])
        if error_i == 0:
            error_i=1e-10 

        if error_i > 0.5:
            y_pred_i=-y_pred_i
            error_i=1-error_i
            polarity=-1
        
        a=0.5*np.log((1-error_i)/error_i)
        D*=np.exp(-a*y*y_pred_i)
        D/=np.sum(D) 
        forest.append({
            'feature': split['feature'],
            'threshold': split['threshold'],
            'alpha': a,
            'polarity': polarity
        })
    return forest

def predict_ada(forest, X):
    score=np.zeros(X.shape[0])
    for s in forest:
        y_pred_s=np.where(X[:, s['feature']] < s['threshold'], -1, 1)*s['polarity']
        score+=s['alpha']*y_pred_s

    return np.sign(score)

Forest=build_tree_ada(x_train, y_train_ada, D, M=20)
y_pred_ada_test=predict_ada(Forest, x_test)
y_pred_ada_train=predict_ada(Forest, x_train)
acc_ada_test=np.mean(y_pred_ada_test==y_test_ada)
acc_ada_train=np.mean(y_pred_ada_train==y_train_ada)
print(f"AdaBoost + Decision Tree Accuracy: {acc_ada_test:.4f} (Test), {acc_ada_train:.4f} (Train)")
plot_3d_error(x_test, y_test_ada, y_pred_ada_test, title="AdaBoost + Decision Tree Test Error Analysis")
plot_3d_error(x_train, y_train_ada, y_pred_ada_train, title="AdaBoost + Decision Tree Train Error Analysis")

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score


# results = {}

# # --- 1. 官方版 决策树 ---
# # max_depth 设为你手撕时的深度，方便公平对比
# sk_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
# sk_dt.fit(x_train, y_train)
# y_pred_dt = sk_dt.predict(x_test)
# results['Decision Tree (Sklearn)'] = accuracy_score(y_test, y_pred_dt)

# # --- 2. 官方版 AdaBoost ---
# # n_estimators 就是你代码里的 M (迭代次数)
# sk_ada = AdaBoostClassifier(n_estimators=20, random_state=42)
# sk_ada.fit(x_train, y_train)
# y_pred_ada = sk_ada.predict(x_test)
# results['AdaBoost (Sklearn)'] = accuracy_score(y_test, y_pred_ada)

# # --- 3. 官方版 SVM (三种核函数) ---
# # 线性核 (Linear)
# sk_svm_lin = SVC(kernel='linear')
# sk_svm_lin.fit(x_train, y_train)
# results['SVM (Linear)'] = accuracy_score(y_test, sk_svm_lin.predict(x_test))

# # 多项式核 (Polynomial)
# sk_svm_poly = SVC(kernel='poly', degree=3) # degree=3 表示 3 次方
# sk_svm_poly.fit(x_train, y_train)
# results['SVM (Poly)'] = accuracy_score(y_test, sk_svm_poly.predict(x_test))

# # 径向基核 (RBF/高斯核) —— 处理这种弯曲数据的“大杀器”
# sk_svm_rbf = SVC(kernel='rbf')
# sk_svm_rbf.fit(x_train, y_train)
# results['SVM (RBF)'] = accuracy_score(y_test, sk_svm_rbf.predict(x_test))

# # --- 打印对比结果 ---
# print("\n" + "="*30)
# print("   Model Performance Comparison")
# print("="*30)
# # 把你自己手撕的结果也打印出来做对比（假设你存了这些变量）
# print(f"Your Manual DT:     {acc_dt_test:.4f}") 
# print(f"Your Manual Ada:    {acc_ada_test:.4f}")
# print("-" * 30)
# for model, acc in results.items():
#     print(f"{model:20}: {acc:.4f}")
# print("="*30)