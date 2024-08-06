---
title: "M 508 Project 1: Gradient Descent Linear Regression"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>

<a href = "https://github.com/CategorIAN/M508_HW1">Code Repository</a>\
[Notes on Gradient Descent Linear Regression](https://categorian.github.io/pdfs/Notes on Gradient Descent Linear Regression.pdf)

<h2>Description</h2>
<p>
For each example \(x^{(i)}\), let our feature vector be \([x_1^{(i)}, x_2^{(i)}, ..., x_d^{(i)}] \in \mathbb{R}^d\) and let \(x_0^{(i)} = 1\). Then, let \(x^{(i)} = [x_0^{(i)}, x_1^{(i)}, ..., x_d^{(i)}] \in \mathbb{R}^{d+1}\), and let our linear regression weights be \(\theta \in \mathbb{R}^{d+1}\). Then, our hypothesis function is defined to be \(h_{\theta}(x^{(i)}) = \sum_{j=0}^{d+1} \theta_j x_j^{(i)} = \theta^{\intercal} x^{(i)} = \theta \cdot x^{(i)}\). We want to train our regression weights \(\theta\) by minimizing the loss function
</p>
