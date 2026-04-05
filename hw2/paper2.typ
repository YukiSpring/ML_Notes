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
  针对 3D 空间中线性不可分的 Make-Moons 数据集分类问题，本文对比研究了三种主流分类算法：决策树（Decision Tree）、集成学习算法 AdaBoost 以及支持向量机（SVM）。首先，本文通过手撕代码实现了基于基尼系数（Gini Impurity）的递归决策树，并探讨了轴对齐切分在 3D 非线性空间中的局限性。实验结果表明，虽然单棵决策树能捕捉基本的空间分布，但在处理交织严重的边缘区域时存在明显的性能瓶颈。
]

= 2 Problem Statement

给定一组由程序生成的 3D 数据集（make-moons-3d），共包含 1000 个训练样本，分为 $C_0$ 与 $C_1$ 两类。要求：
1. 利用训练数据进行模型构建，并生成同分布的 500 个测试样本进行性能评估。
2. 比较决策树、AdaBoost（配合决策树）以及 SVM（至少三种核函数）的分类性能。
3. 深入讨论不同算法在处理该 3D 非线性数据时的优劣原因。

= 3 Methodology

== 3.1 Decision Tree Classification
决策树是一种基于“分而治之”策略的非参数监督学习方法。在 3D 分类任务中，其核心逻辑是在特征空间中寻找最优的超平面切刀。

=== 3.1.1 Gini Impurity
本文采用基尼系数作为衡量节点“纯度”的标准。对于包含 $K$ 个类别的集合，$G$ 指数定义为：
$ G(p) = 1 - sum_(i=1)^K p_i^2 $
其中 $p_i$ 是类别 $i$ 在当前节点中的样本占比。$G$ 指数越小，代表节点的类别纯度越高。

=== 3.1.2 Best Split Search
在每一个决策节点，算法遍历所有特征维度（X, Y, Z）及其所有可能的取值（Threshold），计算切分后的加权基尼指数：
$ G(p) = 1 - sum_(i=1)^K p_i^2 $
算法选择使 $G(p)$ 最小的特征及其阈值作为该节点的切分准则。

=== 3.1.3 Recursive Tree Building
通过递归调用上述切分逻辑，直至满足停止条件：
1. 节点内样本全部属于同一类别（$G = 0$）。
2. 达到预设的最大深度（Max Depth），防止过拟合。
3. 节点内样本数小于预设阈值。

== 3.2 AdaBoost (Adaptive Boosting)
AdaBoost 是一种迭代式的集成学习算法。其核心思想是将多个“弱分类器”组合成一个“强分类器”。

=== 3.2.1 Weak Learner: Decision Stump
在本实验中，AdaBoost 采用决策桩（Decision Stump）作为基学习器。决策桩是深度为 1 的决策树，仅在 3D 空间中进行一次轴对齐切分。虽然单棵桩的分类能力极弱，但 AdaBoost 通过改变样本权重使其关注“难点”。

=== 3.2.2 Weight Update Mechanism
算法为每个训练样本分配权重 $D$。在每一轮迭代 $m$ 中：
1. 计算当前弱分类器的加权误差率 $epsilon_m$。
2. 计算该分类器的权重系数（话语权）$alpha_m$：
  $ alpha_m = 1/2 ln((1 - epsilon_m) / epsilon_m) $
3. 更新样本权重：分错样本权重增加，分对样本权重减少。利用标签 $y in {-1, 1}$ 的特性，更新公式为：
  $ D_(m+1) = D_m dot exp(-alpha_m y hat(y)_m) $

=== 3.2.3 Final Consensus
最终预测结果由所有弱分类器加权投票决定：
$ H(x) = "sign"(sum_(m=1)^M alpha_m h_m (x)) $
= 4 Experimental Studies

== 4.1 Data Analysis
本实验生成的 3D Make-Moons 数据具有显著的非线性特征，两个月牙形分类在 3D 空间中交织。设置噪音水平为 $n=0.2$，这增加类分类边界的模糊性，要求模型具备较强的泛化能力。

#figure(
  image("asserts/3d_make_moons.png", width: 80%),
  caption: [3D Make-Moons 原始数据分布],
)

== 4.2 Decision Tree Results
本文手动实现了决策树算法，并设置最大深度 $"max_depth" = 5$。实验结果如下：

#figure(
  table(
    columns: (4cm, 3cm, 3cm),
    inset: 10pt,
    align: center,
    [*depth*], [*训练准确率*], [*测试准确率*],
    [5], [%], [待填写%],
  ),
  caption: [决策树分类性能评估],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("asserts/Decision Tree Test Error Analysis.png", width: 100%),
    caption: [Decision Tree 测试集错误分析],
  ),
  figure(
    image("asserts/Decision Tree Train Error Analysis.png", width: 100%),
    caption: [Decision Tree 训练集错误分析],
  ),
)


=== 4.2.1 结果分析
通过 3D 错误点分析图可以观察到，预测错误点（红色叉号）主要集中在两个月牙的交汇边缘。
原因分析：决策树的切分边界永远是轴对齐（Axis-aligned）的，即在 3D 空间中表现为垂直于坐标轴的平整切面。对于 Make-Moons 这种平滑弯曲的流形数据，轴对齐的切平面只能通过不断的递归细分来近似逼近圆弧边界，这导致模型在深度有限时难以完美刻画非线性边界，容易产生阶梯状的分类误差.

== 4.3 AdaBoost Results
设置迭代次数 $M = 20$，实验结果如下：

#figure(
  table(
    columns: (4cm, 3cm, 3cm),
    inset: 10pt,
    align: center,
    [*方法*], [*训练准确率*], [*测试准确率*],
    [手撕 AdaBoost], [待填写%], [69.20%],
    [Sklearn AdaBoost], [待填写%], [73.30%],
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
实验数据显示，在本 3D 数据集下，AdaBoost 的测试准确率（69.20%）显著低于单棵深度为 5 的决策树（88.80%）。
原因分析：
1. *弱学习器能力受限*：由于 3D Make-Moons 的交织结构非常复杂，仅靠 20 棵“决策桩”在空间中的简单线性叠加，难以构建出足够精细的非线性边界来包裹弯曲的流形。
2. *对噪声的敏感性*：AdaBoost 会不断强化分错样本的权重。在噪声水平 $n=0.2$ 的情况下，算法可能过度关注边缘的随机噪点，导致模型产生一定的过拟合或分类逻辑紊乱。

== 4.4 Support Vector Machine (SVM)
为了进一步探索非线性边界的构建，本文采用了 SVM 并对比了三种不同的核函数。

#figure(
  table(
    columns: (4cm, 3cm, 3cm),
    inset: 10pt,
    align: center,
    [*核函数 (Kernel)*], [*训练准确率*], [*测试准确率*],
    [Linear], [待填写%], [67.00%],
    [Polynomial], [待填写%], [86.50%],
    [RBF (Gaussian)], [待填写%], [*98.50%*],
  ),
  caption: [SVM 不同核函数的性能对比],
)

=== 4.4.1 结果分析
1. *Linear Kernel*：由于数据本质是高度非线性的，线性核仅能生成超平面，准确率最低。
2. *RBF Kernel*：表现最优（98.50%）。RBF 核通过将数据映射到无穷维空间，在 3D 空间中形成了平滑且紧致的包络边界，完美契合了月牙形数据的几何特性。

= 5 Conclusions
通过本次实验对比，我们得出以下结论：
1. *模型匹配度*：对于 3D 弯曲流形数据，基于核函数的 SVM（尤其是 RBF）展现了远超树类模型的拟合能力。
2. *集成学习的局限*：AdaBoost 虽然能提升弱分类器性能，但在基础组件过于简单或迭代次数不足时，其表现可能不如深度适宜的单棵决策树。
3. *算法实现验证*：手写代码实现的决策树与官方库结果高度一致，证明了算法逻辑的正确性。
