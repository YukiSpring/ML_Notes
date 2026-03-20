// 设置页面和基本字体
#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
)
#set text(
  font: "Times New Roman",
  size: 12pt,
  lang: "zh",
  region: "cn",
)

// 居中对齐
#set align(center)

// 自定义一级标题样式
#show heading.where(level: 1): it => [
  #set text(size: 16pt, weight: "bold")
  #v(2em)
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
#v(1em)
#text(size: 18pt, weight: "bold")[Report of Deep Learning for Natural Langauge Processing]
#v(1em)

// 作者信息
蔡燊生 \
#link("mailto:bronze-age@qq.com")

#v(0em)

= Abstract

摘要

= Introduction

简要介绍
test123
456789

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
