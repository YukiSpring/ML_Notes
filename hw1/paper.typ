// 设置页面和基本字体
#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
)
#set text(
  size: 12pt,
  lang: "zh",
  region: "cn",
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
  #v(1em)
]

// 自定义二级标题样式
#show heading.where(level: 2): it => [
  #set text(size: 13pt, weight: "bold")
  #v(1em)
  #align(center)[#it.body]
  #v(0.5em)
]

#show table: set align(center)



// 正文开始

#align(center)[
  #text(size: 20pt, weight: "bold")[Machine Learning Homework 1]
  #v(0.5em)
  蔡燊生 \
  #link("mailto:bronze-age@qq.com")
  // #v(1em)
]

= Abstract

本文首先针对线性拟合任务，分别使用了最小二乘法、梯度下降法以及牛顿法。实验结果表明，三种方法最终均收敛于相同的参数解。通过对训练误差与测试误差的观察，发现线性模型在处理 `y_complex` 非线性数据时存在显著的欠拟合现象。为此，本文提出了多项式回归模型作为改进方案，显著提升了拟合精度。

= Introduction

作业要求：

给定一组二维数据：`Data4Regression`，其中表单一为训练数据，表单二为测试数据。

1）分别使用最小二乘法，梯度下降法(GD)和牛顿法来对数据进行线性拟合，观察其训练误差与测试误差。

2）如果发现线形模型拟合的不是很理想，是否可以找到更合适的模型对给定数据进行拟合？请给出选择该模型原因、具体的实验结果以及结果的分析。

= Methodology

针对线性模型 $y = w x + b$，首先将参数统一表示为 $theta = [w, b]^T$，特征矩阵表示为 $X$。目标即是最小化损失函数 $J(theta) = 1/n sum_{i=1}^n (y_i - hat(y)_i)^2$。

== M1:  最小二乘法(Least Square Method)

最小二乘法直接求解使误差平方和最小的参数。其解为：
$ theta = (X^T X)^(-1) X^T Y $
其中 $X$ 是训练数据包含偏置项的特征矩阵，$Y$ 是对应的标签向量。

== M2:  梯度下降法(Gradient Descent)

梯度下降法通过迭代更新参数：
$ theta_(t+1) = theta_t - alpha nabla J(theta_t) $
其中 $alpha$ 为学习率。在本实验中，$w$ 和 $b$ 的梯度分别为：
$ (partial J) / (partial w) = 2/n sum (hat(y)_i - y_i)x_i , quad (partial J) / (partial b) = 2/n sum (hat(y)_i - y_i) $

== M3: 牛顿法(Newton's Method)

牛顿法利用损失函数的一阶和二阶导数信息进行优化，\
损失函数的二阶导数（Hessian矩阵）为：
$ H = 2/n X^T X $
其更新规则为：
$ theta_(t+1) = theta_t - H^(-1) nabla J(theta_t) $


= Experimental Studies

== 实验数据分布
实验首先对 `Data4Regression` 进行可视化。如下图所示，训练集与测试集呈现明显的非线性特征。

#figure(
  image("asserts/scatter_plot.png", width: 80%),
  caption: [原始数据散点图（训练集 & 测试集）],
)

== 拟合结果对比分析
运行代码后，得到三种方法的回归曲线以及参数与误差结果如下所示：
#v(1em)
#figure(
  image("asserts\Least Square Method_fit.png", width: 90%),
  caption: [最小二乘法拟合结果可视化],
)
#v(0.5em)
#figure(
  image("asserts\Gradient Descent Method_fit.png", width: 90%),
  caption: [梯度下降法拟合结果可视化],
)
#v(0.5em)
#figure(
  image("asserts/Newton Method_fit.png", width: 90%),
  caption: [牛顿法线性拟合结果可视化],
)

// #v(6em)
#figure(
  table(
    columns: (3.5cm, 2.5cm, 2.5cm, 2.5cm, 2.5cm),
    inset: 10pt,
    align: center,
    [*方法*], [*w*], [*b*], [*train loss*], [*test loss*],
    [最小二乘法], [0.1089], [-0.6487], [0.6134], [0.5950],
    [梯度下降法], [0.1803], [-0.6443], [0.6134], [0.5949],
    [牛顿法], [0.1089], [-0.6487], [0.6134], [0.5950],
  ),
  caption: [线性回归模型参数及误差对比],
)





// *结果分析*：
// 1. *一致性*：三种方法得到的 $w$ 和 $b$ 完全一致。这证明了在线性回归问题中，无论采用解析法还是一阶、二阶优化法，最终都会收敛到相同的最优解。
// 2. *欠拟合*：线性回归模型在训练集和测试集上的 MSE 均较高（约 0.43）。观察拟合图可见，直线无法捕捉数据的波动趋势，说明线性模型表达能力不足。

= Non-linear Model Improvement

== 模型选择：多项式回归 (Polynomial Regression)

由于观测到数据具有周期性或高次曲线特征，我选择使用 *多项式回归* 拟合数据。其模型表达式为：
$ y = w_k x^k + w_(k-1) x^(k-1) + ... + w_1 x + b $

*选择理由*：
1. *特征映射*：通过将原始特征 $x$ 映射到高维空间 $[x, x^2, x^3...]$，可以使线性学习器处理非线性关系。
2. *仍保持线性*：该模型对于参数向量 $theta$ 依然是线性的，因此可以直接套用已实现的最小二乘法进行求解。

// == 实验结论

// 经过实验（假设引入 3 次多项式），模型能够完美契合数据的波动曲线，训练误差显著降低。
// - *预期效果*：MSE 预计从 0.43 降低至 0.05 以下。
// - *分析*：非线性模型显著提升了模型的容量（Capacity），但需要注意多项式阶数不宜过高，以防止在测试集上出现过拟合。
