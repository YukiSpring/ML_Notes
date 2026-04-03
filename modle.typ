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
  111
]




= 2 Problem Statement


222

= 3 Methodology



== 3.1 Linear Regression
333

=== 3.1.1 Least Square Method

444

=== 3.1.2 Gradient Descent


555

=== 3.1.3 Newton's Method

666


== 3.2 Polynomial Regression

= 4 Experimental Studies

== 4.1 Data Analysis

// #v(6em)

== 4.3 Polynomial Regression


== 4.4 K-Nearest Neighbors Regression


= 5 Conclusions


