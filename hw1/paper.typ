// 设置页面和基本字体
#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
)
#set text(
  size: 12pt,
  lang: "zh",
  region: "cn",
  // font:
)

#set par(
  first-line-indent: 2em,
  justify: true,
  leading: 0.8em,
)



// 自定义一级标题样式
#show heading.where(level: 1): it => [
  #set text(size: 16pt, weight: "bold")
  #set align(center)
  #v(1em)
  #it.body
  #v(0.5em)
]

// 自定义二级标题样式
#show heading.where(level: 2): it => [
  #set text(size: 13.5pt, weight: "bold")
  #v(0.5em)
  #align(center)[#it.body]
  #v(0.5em)
]

#show heading.where(level: 3): it => [
  #set text(size: 12pt, weight: "bold")
  #v(0.5em)
  #align(center)[#it.body]
  #v(0.5em)
]
// 表格居中
#show table: set align(center)



// 正文开始

#align(center)[
  #text(size: 20pt, weight: "bold")[Machine Learning Homework 1]
  #v(0.5em)
  蔡燊生 \
  #link("mailto:bronze-age@qq.com")
  // #v(1em)
]

= 1 Abstract

#[
  #set text(size: 11pt)
  针对回归问题，本文首先采用线性回归模型，研究了最小二乘法、梯度下降法以及牛顿法在参数优化中的表现。实验结果表明，
  三种优化路径最终均收敛至相同的全局最优解，但由于原始数据呈现显著的非线性分布，线性模型均出现明显的欠拟合。
  为此，本文采用了高阶多项式回归与 $K$ 近邻回归模型。使得测试集上的均方误差大幅降低。
]




= 2 Problem Statement


给定一组二维数据：`Data4Regression`，其中表单一为训练数据，表单二为测试数据。

1）分别使用最小二乘法，梯度下降法(GD)和牛顿法来对数据进行线性拟合，观察其训练误差与测试误差。

2）如果发现线形模型拟合的不是很理想，是否可以找到更合适的模型对给定数据进行拟合？请给出选择该模型原因、具体的实验结果以及结果的分析。

= 3 Methodology



== 3.1 Linear Regression
针对线性模型 $y = w x + b$，将参数表示为 $theta = [w, b]^T$，特征矩阵表示为 $X$。目标即是最小化损失函数，采用均方误差作为损失函数：

$ J(theta) = 1/n sum_{i=1}^n (y_i - hat(y)_i)^2 $

=== 3.1.1 Least Square Method

最小二乘法直接求解使损失函数最小的参数。其解为：
$ theta = (X^T X)^(-1) X^T Y $
其中 $X$ 是训练数据包含偏置项的特征矩阵，$Y$ 是对应的标签向量。

=== 3.1.2 Gradient Descent

梯度下降法通过迭代循环更新参数：
$ theta_(t+1) = theta_t - alpha nabla J(theta_t) $
其中 $alpha$ 为学习率。在本实验中，$w$ 和 $b$ 的梯度分别为：
$
  (partial J) / (partial w) = 2/n sum (hat(y)_i - y_i)x_i ,
  quad (partial J) / (partial b) = 2/n sum (hat(y)_i - y_i)
$

=== 3.1.3 Newton's Method

牛顿法利用损失函数的一阶导数和二阶导数信息进行优化，损失函数的二阶导数（Hessian矩阵）为：
$ H = 1/n X^T X $
其更新规则为：
$ theta_(t+1) = theta_t - H^(-1) nabla J(theta_t) $

== 3.2 Polynomial Regression
观测原始数据散点图发现，数据分布呈现出明显的波动或曲线特征，
线性模型存在严重的欠拟合。根据泰勒展开原理，任何连续函数都可以用多项式
进行逼近。因此采用多项式回归来拟合数据的非线性关系是一个合理的选择。\

多项式回归虽然在特征空间上是非线性的，但对于待求参数 $theta$ 而言，它依然是线性的。这意味着我们可以沿用线性回归中最小二乘法进行解析求解，即为：
$ theta = (X^T X)^(-1) X^T Y $

本文采用10阶多项式特征，即将原始特征 $X$ 转换为 $[X^10, X^9, ..., X^1, X^0]$

== 3.3 K-Nearest Neighbors Regression
线性回归和多项式回归都假设数据符合某种全局的函数形式，而 KNN 不需要对数据的整体分布做任何先验假设。它直接从数据出发，非常适合处理函数形式未知的回归问题。
所以在数据分布不均匀时，KNN 能够比全局拟合模型更灵敏地捕捉到局部的变化规律。

KNN回归是一种基于实例的非参数方法。对于一个新的输入 $x$，KNN回归通过找到训练集中与 $x$ 最近的 $k$ 个邻居，并返回这些邻居的标签的平均值作为预测结果。
本实验中$k$ 取值为3，即选择最近的三个邻居进行预测，距离采用欧氏距离:
$ "distance"(x_i, x) = ||x_i - x||_2 =|x_i-x| $

\
= 4 Experimental Studies

== 4.1 Data Analysis
首先我们对 `Data4Regression` 进可视化分析。如下图所示，显然训练集与测试集呈现非线性关系。

#figure(
  image("asserts/scatter_plot.png", width: 80%),
  caption: [原始数据散点图（训练集 & 测试集）],
)

== 4.2 Linear Regression
实验可得三种方法的回归曲线以及参数与误差结果如下：
#v(1em)
#figure(
  image("asserts\Least Square Method_fit.png", width: 100%),
  caption: [最小二乘法结果可视化],
)
#v(0.5em)
#figure(
  image("asserts\Gradient Descent Method_fit.png", width: 100%),
  caption: [梯度下降法结果可视化],
)
#v(0.5em)
#figure(
  image("asserts/Newton Method_fit.png", width: 100%),
  caption: [牛顿法线性结果可视化],
)

// #v(6em)
#figure(
  table(
    columns: (3.5cm, 2.5cm, 2.5cm, 2.5cm, 2.5cm),
    inset: 10pt,
    align: center,
    [*方法*], [*w*], [*b*], [*train loss*], [*test loss*],
    [最小二乘法], [0.1089], [-0.6487], [0.6134], [0.5950],
    [梯度下降法], [0.1083], [-0.6443], [0.6134], [0.5949],
    [牛顿法], [0.1089], [-0.6487], [0.6134], [0.5950],
  ),
  caption: [线性回归参数及误差对比],
)
#h(2em)
分析表1的数据知：
最小二乘法和牛顿法在此类小规模数据集上表现出了较高的计算效率，无需迭代即可求出最优解；而梯度下降法虽然也达到了相同的精度，但由于其一阶收敛的特性，需要设置合理的迭代次数和学习率。
此外，观察拟合图像可见，线性模型完全无法捕捉数据的非线性趋势，训练与测试误差均在 0.6 左右，说明线性模型存在严重的欠拟合现象，因此需要采用更合适的模型进行拟合。


== 4.3 Polynomial Regression
为了克服线性模型的局限性，本文采用了 10 阶多项式进行拟合。
#v(1em)
#figure(
  image("asserts\Polynomial Regression_fit.png", width: 100%),
  caption: [10阶多项式回归结果可视化],
)
#v(0.5em)
#figure(
  table(
    columns: (3.5cm, 2.5cm, 2.5cm),
    inset: 10pt,
    align: center,
    [*方法*], [*train loss*], [*test loss*],
    [多项式回归], [0.3497], [0.3837],
  ),
  caption: [多项式回归误差对比],
)
#h(2em)
显然,引入高阶项后，模型的复杂度提升，能够更好地拟合数据的非线性关系，
训练误差和测试误差均有所降低。

== 4.4 K-Nearest Neighbors Regression
针对KNN回归模型，实验结果如下所示：
#v(1em)
#figure(
  image("asserts\KNN Regression (k=3)_fit.png", width: 100%),
  caption: [KNN回归结果可视化（k=3）],
)

#v(0.5em)
#figure(
  table(
    columns: (3.5cm, 2.5cm, 2.5cm),
    inset: 10pt,
    align: center,
    [*方法*], [*train loss*], [*test loss*],
    [KNN], [0.1385], [0.2599],
  ),
  caption: [KNN回归误差对比（k=3）],
)
#h(2em)
本文中预测值由最近的三个邻居决定，可见KNN回归能够较好地捕捉数据的局部结构，
因此，得到了最低的训练误差和测试误差。
但是，注意到此时的训练误差低于测试误差，反映出模型具有一定的过拟合倾向。这是因为在 $k$ 值较小时会过于关注训练数据的局部噪声。

= 5 Conclusions
通过本次实验，我们研究了三个典型的线性回归优化方法和两种非线性回归模型在给定数据集上的表现。
结果表明，线性模型由于数据的非线性分布而出现了明显的欠拟合现象，而多项式回归和KNN回归则能够更好地拟合数据，显著降低了训练和测试误差。

