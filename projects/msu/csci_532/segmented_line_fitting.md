---
title: "Project 2: Segmented Line Fitting"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>
<a href = "https://github.com/CategorIAN/CSCI_532_HW2">Code Repository</a>\
[Notes on Segmented Line Fitting](https://categorian.github.io/pdfs/Notes on Segmented Line Fitting.pdf)

<h2>Description</h2>
<p>
Given a set of points \(P = \{(x_i, y_i)\}_{i=1}^n\) such that \(x_1 < x_2 < ... < x_n\) and given a line y = ax + b\), the error is \(\sum_{i=1}^n (y_i - ax_i-b)^2\). The line of best fit is found by
\[
a = \dfrac{n\sum x_i y_i - (\sum x_i) (\sum y_i)}{n \sum x_i^2 - (\sum x_i)^2}
\]
\[
b = \dfrac{\sum y_i - a \sum x_i}{n}
\]
</p>
