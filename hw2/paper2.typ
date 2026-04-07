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

// 自定义标题样式
#show heading.where(level: 1): it => [
  #set text(size: 16pt, weight: "bold")
  #set align(center)
  #v(1em)
  #it.body
  #v(0.5em)
]

#show heading.where(level: 2): it => [
  #set text(size: 13.5pt, weight: "bold")
  #v(1em)
  #align(center)[#it.body]
  #v(1em)
]

#show heading.where(level: 3): it => [
  #set text(size: 12pt, weight: "bold")
  #v(1em)
  #align(center)[#it.body]
  #v(1em)
]

#show table: set align(center)

// 正文开始
#align(center)[
  #text(size: 20pt, weight: "bold")[Machine Learning Homework 2]
  #v(0.5em)
  蔡燊生 \
  #link("mailto:bronze-age@qq.com")
]

= 1 Abstract

#[
  #set text(size: 11pt)
  针对 3D 空间中线性不可分的 Make-Moons 数据集分类问题，本文对比研究了三种分类算法：决策树（Decision Tree）、集成学习算法 AdaBoost 以及支持向量机（SVM）。本文通过手动实现基于基尼系数的递归决策树与 AdaBoost 框架，探讨了不同模型在处理非线性流形数据时的表达能力。实验结果显示，SVM 在采用 RBF 核函数时取得了 98.7% 的最优准确率，而单棵决策树表现优于基于决策桩的 AdaBoost，揭示了模型复杂程度与空间切分方式对 3D 分类性能的关键影响。
]

= 2 Problem Statement

给定一组由程序生成的 3D 数据集，共包含 1000 个训练样本，分为 $C_0$ 与 $C_1$ 两类。要求：
1. 利用训练数据进行模型构建，并生成同分布的 500 个测试样本（每类 250 个）进行性能评估。
2. 比较Decision Tree、AdaBoost+Decision Tree 以及 SVM（至少三种核函数）的分类性能。
3. 讨论不同算法在处理该数据时的优劣原因。


= 3 Methodology

== 3.1 Decision Tree Classification
决策树通过递归地将特征空间划分为若干个正交的超矩形区域来进行分类。在 3D 空间中，这表现为一系列垂直于坐标轴的平面切割。

=== 3.1.1 Gini Impurity
本文采用基尼指数来度量节点 $p$ 的纯度。对于二分类问题，若类别 $k in {0, 1}$ 在节点中的概率为 $p_k$，则基尼不纯度定义为：
$ G(p) = 1 - sum_(k=0)^1 p_k^2 $
#h(2em)当节点内所有样本属于同一类时，$G(p) = 0$。算法的目标是寻找能够最大限度降低基尼指数的切分方案。

#v(2em)
=== 3.1.2 Best Split Search
对于当前节点，算法遍历所有特征维度 $d in {X, Y, Z}$ 及其所有可能的候选阈值 $t$。切分后的加权基尼指数为：
$ G_"split" = n_"left" / n G(p_"left") + n_"right" / n G(p_"right") $

#h(2em)
其中 $n$ 为样本总数，$n_"left"$ 和 $n_"right"$ 分别为左右子节点的样本数。算法选择使 $G_"split"$ 最小的 $(d, t)$ 组合作为当前节点的切分准则。

== 3.2 AdaBoost + Decision Tree
AdaBoost 是一种集成学习方法，其核心思想是通过迭代构建一系列弱分类器，并根据其性能分配权重，最终组合成强分类器。

=== 3.2.1 Decision Stump
本实验使用决策桩作为弱分类器，即深度为 1 的决策树。其数学表达式为：
$ h(x; d, t, p) = cases(1 & "if " p dot x_d < p dot t, -1 & "otherwise") $
其中 $p in {1, -1}$ 为极性系数，用于控制不等式的方向。

=== 3.2.2 Weight Update
设第 $m$ 轮迭代时样本权重分布为 $D_m$。弱分类器 $h_m$ 的加权误差率为：
$ epsilon_m = sum_(i=1)^N D_m (i) dot II (h_m (x_i) != y_i) $
分类器的权重$alpha_m$ 计算公式为：
$ alpha_m = 1/2 ln((1 - epsilon_m) / epsilon_m) $
更新下一轮样本权重 $D_(m+1)$，使模型更关注被误分的样本：
$ D_(m+1) (i) = (D_m (i) dot exp(-alpha_m y_i h_m (x_i))) / Z_m $
其中 $Z_m$ 为规范化因子，确保 $sum D = 1$。
#v(2em)
#v(2em)
=== 3.2.3 Strong Classifier
$M$ 轮迭代后，最终的强分类器通过对弱分类器进行加权求和并取符号函数得到：
$ H(x) = "sign"(sum_(m=1)^M alpha_m h_m (x)) $

== 3.3 Support Vector Machine
支持向量机的目标是在高维空间中寻找一个最优超平面，使得分类间隔（Margin）最大化。

=== 3.3.1 Hard Margin SVM
SVM 寻找满足 $y_i (w^T x_i + b) >= 1$ 的超平面，通过最小化 $1/2 ||w||^2$ 来最大化间隔。引入拉格朗日乘子 $lambda_i$，其对偶形式为：
$ max_(lambda) sum_(i=1)^N lambda_i - 1/2 sum_(i=1)^N sum_(j=1)^N lambda_i lambda_j y_i y_j K(x_i, x_j) $
满足约束：$sum lambda_i y_i = 0$ 且 $0 <= lambda_i <= C$。

=== 3.3.2 Kernel Trick
针对 3D Make-Moons 的非线性特性，通过核函数将输入空间映射到高维特征空间：
1. 线性核 (Linear): 假设数据线性可分。$K(x_i, x_j) = x_i^T x_j$
2. 多项式核 (Polynomial): 构建高次非线性边界。$K(x_i, x_j) = (gamma x_i^T x_j + r)^d$
3. RBF 核 (Radial Basis Function): 利用样本间的欧氏距离构建局部感应边界，理论上可映射至无穷维空间。$K(x_i, x_j) = exp(-gamma ||x_i - x_j||^2)$



= 4 Experimental Studies

== 4.1 Data Analysis
3D Make-Moons 数据呈现出两个交织的月牙形状，且带有 $n=0.2$ 的高斯噪声。这要求分类器必须能构建复杂的非线性边界。

生成的训练数据和测试数据分布如下图：
#figure(
  image("asserts/3d_make_moons.png", width: 80%),
  caption: [3D Make-Moons 训练数据与测试数据分布],
)

== 4.2 Decision Tree Results
设置最大深度为 5，实验结果如下：

#figure(
  table(
    columns: (4cm, 3cm, 3cm),
    inset: 10pt,
    align: center,
    [*Max Depth*], [*训练准确率*], [*测试准确率*],
    [5], [91.25%], [90.00%],
  ),
  caption: [决策树分类性能评估],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("asserts/Decision Tree Train Error Analysis.png", width: 100%),
    caption: [DT 训练集错误分析],
  ),
  figure(
    image("asserts/Decision Tree Test Error Analysis.png", width: 100%),
    caption: [DT 测试集错误分析],
  ),
)

=== 4.2.1 结果分析
决策树取得了 90% 的优异表现。由于模型深度足够（depth=5），其能够在 3D 空间中形成较为复杂的分割边界，成功捕捉了月牙形状的非线性特征。然而，仍有部分样本位于边界附近，导致误分类。

== 4.3 AdaBoost Results
设置迭代轮数 $M = 20$，实验结果如下：

#figure(
  table(
    columns: (4cm, 3cm, 3cm),
    inset: 10pt,
    align: center,
    [*迭代轮数 M*], [*训练准确率*], [*测试准确率*],
    [20], [69.55%], [69.10%],
  ),
  caption: [AdaBoost 分类性能评估],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("asserts/AdaBoost + Decision Tree Train Error Analysis.png", width: 100%),
    caption: [AdaBoost 训练集错误分析],
  ),
  figure(
    image("asserts/AdaBoost + Decision Tree Test Error Analysis.png", width: 100%),
    caption: [AdaBoost 测试集错误分析],
  ),
)

=== 4.3.1 结果分析
由结果可见，AdaBoost 的表现不佳。主要原因在于其基学习器（决策桩）过于简单，仅一次轴对齐切分无法在 3D 空间中形成复杂的包络结构。20 轮叠加后的线性投票仍难以对抗高度弯曲的数据分布。

== 4.4 SVM Results
使用 Scikit-learn 库直接调用SVM，对比三种核函数，结果如下：

#figure(
  table(
    columns: (4cm, 3cm, 3cm),
    inset: 10pt,
    align: center,
    [*核函数*], [*训练准确率*], [*测试准确率*],
    [Linear], [67.80%], [68.50%],
    [Polynomial (d=3)], [86.80%], [87.30%],
    [*RBF (Gaussian)*], [*98.35%*], [*98.70%*],
  ),
  caption: [SVM 不同核函数的性能对比],
)


#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("asserts/SVM (linear kernel) Train Error Analysis.png", width: 100%),
    caption: [SVM Linear Kernel 训练集错误分析],
  ),
  figure(
    image("asserts/SVM (linear kernel) Test Error Analysis.png", width: 100%),
    caption: [SVM Linear Kernel 测试集错误分析],
  ),
)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("asserts/SVM (poly kernel) Train Error Analysis.png", width: 100%),
    caption: [SVM Polynomial Kernel 训练集错误分析],
  ),
  figure(
    image("asserts/SVM (poly kernel) Test Error Analysis.png", width: 100%),
    caption: [SVM Polynomial Kernel 测试集错误分析],
  ),
)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("asserts/SVM (rbf kernel) Train Error Analysis.png", width: 100%),
    caption: [SVM RBF Kernel 训练集错误分析],
  ),
  figure(
    image("asserts/SVM (rbf kernel) Test Error Analysis.png", width: 100%),
    caption: [SVM RBF Kernel 测试集错误分析],
  ),
)

=== 4.4.1 结果分析
1. *RBF Kernel*: 取得了最优结果。这是因为在 3D 空间中，RBF 核能根据数据局部密度生成平滑的曲面，完美捕捉了非线性拓扑结构。
2. *Linear Kernel*: 表现最差，验证了该数据集在原空间是绝对线性不可分的。
3. *Polynomial Kernel*: 能够拟合曲面，但在月牙的尖端处理上不如 RBF 精准。

= 5 Conclusions

1. SVM-RBF 的优点：RBF 核函数表现最优，原因在于其本质是基于“相似度”的分类。3D 月牙数据在几何上是平滑弯曲的流形，RBF 核通过在每个支持向量周围建立径向对称的感应场，能够完美勾勒出非线性的曲面边界，从而规避了线性切分的僵硬性。

2. 决策树的优点：决策树凭借其深度的局部搜索能力，通过多次“轴对齐”切割（阶梯状近似），较好地逼近了月牙的轮廓。然而，由于它无法产生倾斜或弯曲的切面，在两类交织严重的边缘区域仍存在固有的近似误差。

3. AdaBoost 的局限：集成学习在本实验中表现一般。原因在于所选的弱分类器表达能力极其有限，仅能在 3D 空间中横竖切一刀。即便通过 50 轮迭代加权投票，这种“线性分段函数”的叠加在处理三维螺旋交织结构时，效率远不如深层决策树或核 SVM。
