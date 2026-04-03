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

= 4 Experimental Studies

== 4.1 Data Analysis
本实验生成的 3D Make-Moons 数据具有显著的非线性特征，两个月牙形分类在 3D 空间中交织。设置噪音水平为 $n=0.2$，这增加类分类边界的模糊性，要求模型具备较强的泛化能力。

#figure(
  image("asserts/dt.png", width: 80%),
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

#v(1em)
#figure(
  image("asserts/dt.png", width: 90%),
  caption: [决策树预测错误点 3D 分析],
)

=== 4.2.1 结果分析
通过 3D 错误点分析图可以观察到，预测错误点（红色叉号）主要集中在两个月牙的**交汇边缘**。
原因分析：决策树的切分边界永远是**轴对齐（Axis-aligned）**的，即在 3D 空间中表现为垂直于坐标轴的平整切面。对于 Make-Moons 这种平滑弯曲的流形数据，轴对齐的切平面只能通过不断的递归细分来近似逼近圆弧边界，这导致模型在深度有限时难以完美刻画非线性边界，容易产生阶梯状的分类误差。
