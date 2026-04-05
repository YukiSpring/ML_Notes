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
  针对 3D 空间中线性不可分的 Make-Moons 数据集分类问题，本文对比研究了三种主流分类算法：决策树（Decision Tree）、集成学习算法 AdaBoost 以及支持向量机（SVM）。本文通过手动实现基于基尼系数的递归决策树与 AdaBoost 框架，深入探讨了不同模型在处理非线性流形数据时的表达能力。实验结果显示，SVM 在采用 RBF 核函数时取得了 98.7% 的最优准确率，而单棵决策树表现优于基于决策桩的 AdaBoost，揭示了模型复杂程度与空间切分方式对 3D 分类性能的关键影响。
]

= 2 Problem Statement

给定一组由程序生成的 3D 数据集（make-moons-3d），共包含 1000 个训练样本，分为 $C_0$ 与 $C_1$ 两类。要求：
1. 利用训练数据进行模型构建，并生成同分布的 500 个测试样本（每类 250 个）进行性能评估。
2. 比较决策树、AdaBoost（配合决策树）以及 SVM（至少三种核函数）的分类性能。
3. 深入讨论不同算法在处理该 3D 非线性数据时的优劣原因。

= 3 Methodology

== 3.1 Decision Tree Classification
决策树通过在特征空间中寻找最优切分超平面来递归划分数据。

=== 3.1.1 Gini Impurity
本文采用基尼系数作为衡量标准：
$ G(p) = 1 - sum_(i=1)^K p_i^2 $
其中 $p_i$ 为类别占比。该指标反映了节点内类别的混乱程度。

=== 3.1.2 Recursive Tree Building
算法在 X, Y, Z 三个维度中搜索使加权基尼指数最小的特征与阈值进行切分，直至达到最大深度（本文设为 5）或节点纯净。

== 3.2 AdaBoost (Adaptive Boosting)
AdaBoost 是一种集成学习框架，通过“加权投票”将多个弱分类器组合。

=== 3.2.1 Weak Learner: Decision Stump
本实验使用深度为 1 的决策桩作为基学习器。虽然单棵桩分类能力有限，但通过迭代更新样本权重 $D$，使后续分类器重点关注前序分类器误分的“难点”。

=== 3.2.2 Weight and Alpha Update
每轮计算误差 $epsilon_m$ 与分类器权重 $alpha_m$：
$ alpha_m = 1/2 ln((1 - epsilon_m) / epsilon_m) $
$ D_(m+1) = (D_m dot exp(-alpha_m y hat(y)_m)) / Z_m $

== 3.3 Support Vector Machine (SVM)
SVM 的核心是在高维空间中寻找具有“最大间隔（Maximum Margin）”的决策超平面。

=== 3.3.1 Kernel Trick
针对 3D 非线性数据，SVM 利用核函数 $K(x_i, x_j)$ 将数据投影至高维特征空间，使其线性可分。本文研究三种核函数：
1. *Linear Kernel*: $K(x_i, x_j) = x_i^T x_j$
2. *Polynomial Kernel*: $K(x_i, x_j) = (gamma x_i^T x_j + r)^d$
3. *RBF (Gaussian) Kernel*: $K(x_i, x_j) = exp(-gamma ||x_i - x_j||^2)$

= 4 Experimental Studies

== 4.1 Data Analysis
3D Make-Moons 数据呈现出两个交织的月牙形状，且带有 $n=0.2$ 的高斯噪声。这要求分类器必须能构建复杂的非线性边界。

#figure(
  image("asserts/3d_make_moons.png", width: 80%),
  caption: [3D Make-Moons 原始数据分布],
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
决策树取得了 90% 的优异表现。由于模型深度足够（depth=5），它通过 31 个潜在的节点切分，较好地近似了月牙的边界。错误点主要集中在两类数据高度重叠的边缘区域。

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
AdaBoost 的表现仅略优于线性分类。主要原因在于其基学习器（决策桩）过于简单，仅一次轴对齐切分无法在 3D 空间中形成复杂的包络结构。20 轮叠加后的线性投票仍难以对抗高度弯曲的月牙流形。

== 4.4 SVM Results
使用 Scikit-learn 库对比三种核函数，结果如下：

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
1. *RBF Kernel (98.7%)*: 取得了全场最高分。RBF 核能根据数据局部密度生成平滑的曲面，完美捕捉了 3D 空间中的非线性拓扑。
2. *Linear Kernel*: 表现最差，验证了该数据集在原空间是绝对线性不可分的。
3. *Polynomial Kernel*: 能够拟合曲面，但在月牙的尖端处理上不如 RBF 精准。

= 5 Conclusions
1. *流形拟合能力*: 针对 3D 弯曲数据，基于局部感应的 RBF-SVM 是最优选择，其准确率高达 98.7%。
2. *模型复杂度权衡*: 手撕决策树由于具备一定深度，其空间划分能力在本项目中优于基学习器过弱的 AdaBoost。
3. *算法健壮性*: 各算法训练与测试准确率差距均极小，说明在 1000 样本规模下，模型展现了良好的泛化性。
