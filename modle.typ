// 设置页面和基本字体
#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
)
#set text(
  // fallback: true,
  size: 12pt,
  lang: "zh",
  region: "cn",
)

// 居中对齐
// #set align(center)

#set par(
  first-line-indent: 2em,
  justify: true,
  leading: 0.8em,
) // 设置全局首行缩进两个字符

#show heading: set align(center)

// 自定义一级标题样式
#show heading.where(level: 1): it => [
  #set text(size: 16pt, weight: "bold")
  #v(1em)
  #it.body
  #v(1em)
]

// 自定义二级标题样式 (M1, M2, etc.)
#show heading.where(level: 2): it => [
  #set text(size: 13pt, weight: "bold")
  #v(1em)
  #it.body
  #v(0.5em)
]

// --- 正文开始 ---

// 报告大标题
// #v(1em)
#align(center)[
  #text(size: 18pt, weight: "bold")[Machine Learning Homework 1]
  #v(0.5em)

  // 作者信息
  ccc \
  #link("mailto:bronze-age@qq.com")
]


// #v(0em)

= Abstract

摘要

= Introduction
// #h(2em)
作业要求：\
#h(2em)给定一组二维数据： Data4Regression， 其中表单一为训练数据，表单二为测试数据。\
1）分别使用最小二乘法，梯度下降法(GD)和牛顿法来对数据进行线性拟合，
观察其训练误差与测试误差。\
2）如果发现线形模型拟合的不是很理想（数据实际是非线性的，所以上一问的实验结果不好是正常的），是否可以找到更合适的模型对给定数据进行拟合？请给出你选择该模型原因、具体的实验结果以及结果的分析。


= Methodology

方法介绍

== M1: DL Model

The description of the first model.

== M2: NLP Model

The description of the second model.

== M3: DL-NLP Model

The description of the third model.

= Experimental Studies

The figures shows it works so well
