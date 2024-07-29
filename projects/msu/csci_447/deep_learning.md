---
title: "CSCI 447 Project 3: Deep Learning"
layout: default
---

<h1>{{page.title}}</h1>

<h2>References</h2>
<a href = "https://github.com/EthanSkelton9/csci447_project3">Code Repository</a>\
[Full Report](https://categorian.github.io/pdfs/CSCI_447_Project_3.pdf)

<h2>Description</h2>
<p>
We used a feed forward neural network to predict the classes and target values of our data. For our regression sets, the final layer consisted of a real value \(V\in \mathbb{R}\) that represented our prediction for the target value. For our classification sets, the final layer consisted of \(V \in \mathbb{R}^n\), where \(n\) is the number of classes and \(V_i\) is the value associated with class \(i\). Then, the predicted class was determined to be the softmax \(j = \text{argmax}_{i\in [1..n]} V_i\).
</p>
