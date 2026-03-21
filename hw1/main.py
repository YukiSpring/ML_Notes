'''
date: 2026/03/20
author: css
id: 23375158
description: 
    给定一组二维数据： Data4Regression， 其中表单一为训练数据，表单二为测试数据。 
    1）分别使用最小二乘法，梯度下降法(GD)和牛顿法来对数据进行线性拟合，观察其训练误差与测试误差。
    2）如果发现线形模型拟合的不是很理想（数据实际是非线性的，所以上一问的实验结果不好是正常的），
       是否可以找到更合适的模型对给定数据进行拟合？请给出你选择该模型原因、具体的实验结果以及结果的分析。 
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

'''load data & plot scatter figure'''

# print(os.getcwd())

# read data
file_path = 'hw1\Data4Regression.xlsx'
df=pd.read_excel(file_path,sheet_name=0)
test_df=pd.read_excel(file_path,sheet_name=1)

# pre see
# print(df.head())
# print(test_df.head())

# save
x_train = df['x'].values
y_train = df['y_complex'].values
x_test = test_df['x_new']
y_test = test_df['y_new_complex']

# plot
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', marker='o', label='Training Data')
plt.scatter(x_test, y_test, color='red', marker='x', label='Testing Data')
plt.title('Training Data Scatter Plot')
plt.xlabel('x')
plt.ylabel('y_complex')

plt.ylim(-3, 3)

plt.legend()
plt.grid(linestyle='--', alpha=0.7)

save_path = 'hw1/asserts'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# plt.savefig(os.path.join(save_path, 'scatter_plot.png'))

# plt.show()

def plot_fit(x_line,y_line,x_line_test,y_line_test,title):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    fig.suptitle(f'{title} Fit', fontsize=16)
    axes[0].set_ylim(-3, 3)
    axes[1].set_ylim(-3, 3)

    axes[0].scatter(x_train,y_train,color='blue',marker='o',label='Training Data')
    axes[0].plot(x_line,y_line,color='green',label=title)
    axes[0].set_title('Training Data')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y_complex')

    axes[1].scatter(x_test,y_test,color='red',marker='x',label='Testing Data')
    axes[1].plot(x_line_test,y_line_test,color='green',label=title)
    axes[1].set_title('Testing Data')
    axes[1].set_xlabel('x_new')
    axes[1].set_ylabel('y_new_complex')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title}_fit.png'))
    plt.show()

'''least square method'''

# 添加一列计算截距b
x_b=np.c_[x_train,np.ones((len(x_train),1))]

# W=(X^T X)^(-1) X^T Y
W=np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)

w_ls=W[0]
b_ls=W[1]
print(f'Least Square Method: w={w_ls:.4f}, b={b_ls:.4f}')

def predict(x,w,b):
    return w * x + b

train_pred = predict(x_train,w_ls,b_ls)
test_pred = predict(x_test,w_ls,b_ls)

# mean squared error
train_mse=np.mean((train_pred - y_train) ** 2)
test_mse=np.mean((test_pred - y_test) ** 2)
print(f'Train: {train_mse:.4f}, Test: {test_mse:.4f}')

# 绘图
x_line_train=np.array([min(x_train),max(x_train)])
y_line_train=predict(x_line_train,w_ls,b_ls)
x_line_test=np.array([min(x_test),max(x_test)])
y_line_test=predict(x_line_test,w_ls,b_ls)

plot_fit(x_line_train,y_line_train,x_line_test,y_line_test,'Least Square Method')

'''gradient descent method'''

learning_rate=0.01
epochs=1000
w_gd=0.0
b_gd=0.0
n=len(x_train)

loss_history=[]

for i in range(epochs):
    # 预测
    y_pred_gd=w_gd*x_train+b_gd
    # 计算误差
    error_gd=y_pred_gd-y_train
    # 计算梯度
    dw=(2/n)*np.sum(error_gd*x_train)
    db=(2/n)*np.sum(error_gd)
    # 更新参数
    w_gd-=learning_rate*dw
    b_gd-=learning_rate*db
    # 计算损失
    loss_gd=np.mean(error_gd**2)
    loss_history.append(loss_gd)

    if i%100==0:
        print(f'Epoch {i}: w={w_gd:.4f}, b={b_gd:.4f}, Loss={loss_gd:.4f}')

print(f'Gradient Descent Method: w={w_gd:.4f}, b={b_gd:.4f}')
# 计算训练和测试误差
train_pred_gd=predict(x_train,w_gd,b_gd)
test_pred_gd=predict(x_test,w_gd,b_gd)

train_mse_gd=np.mean((train_pred_gd - y_train) ** 2)
test_mse_gd=np.mean((test_pred_gd - y_test) ** 2)

print(f'Train: {train_mse_gd:.4f}, Test: {test_mse_gd:.4f}')

# 绘图
x_line_train_gd=np.array([min(x_train),max(x_train)])
y_line_train_gd=predict(x_line_train_gd,w_gd,b_gd)
x_line_test_gd=np.array([min(x_test),max(x_test)])
y_line_test_gd=predict(x_line_test_gd,w_gd,b_gd)
plot_fit(x_line_train_gd,y_line_train_gd,x_line_test_gd,y_line_test_gd,'Gradient Descent Method')
