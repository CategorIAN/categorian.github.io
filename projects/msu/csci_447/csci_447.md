---
title: CSCI 447 Projects
layout: default
---

<h1>"CSCI 447: Machine Learning" Projects</h1>

[Project 1: Naive Bayes](/naive_bayes.md)\
[Project 2: K Nearest Neighbors](/k_nearest_neighbors.md)\
[Project 3: Deep Learning](/deep_learning.md)

For my machine learning course, my partner Ethan Skelton and I worked on three projects together. Each project involved creating a supervised learning model to make predictions on classes for classification sets and target values for regression sets. 

<h2>The Data Sets</h2>
The data sets were from the <a href = "https://archive.ics.uci.edu/datasets">UC Irvine Machine Learning Repository</a>.
<h3>Classification Data Sets</h3>
* <a href = "https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original">Breast Cancer</a>

* <a href = "https://archive.ics.uci.edu/dataset/42/glass+identification">Glass</a>

* <a href = "https://archive.ics.uci.edu/dataset/53/iris">Iris</a>

* <a href = "https://archive.ics.uci.edu/dataset/91/soybean+small">Soybean (small)</a>

* <a href = "https://archive.ics.uci.edu/dataset/105/congressional+voting+records">Vote</a>

<h3>Regression Data Sets</h3>

* <a href = "https://archive.ics.uci.edu/dataset/1/abalone">Abalone</a>

* <a href = "https://archive.ics.uci.edu/dataset/29/computer+hardware">Computer Hardware</a>

* <a href = "https://archive.ics.uci.edu/dataset/162/forest+fires">Forest Fires</a>

<h2>Experimental Design</h2>
<p>
For each project, we used 10-Fold Cross Validation to evaluate our model. For this method, we partitioned our data set \(S\) into 10 equally sized subsets \(\cup_{i=1}^{10} S_i = S\). Then, for each of the 10 folds, we calculate the error of fold i, \(E_i\), by training our model on \(\cup_{i\neq} S_i\) and calculating the error on the prediction of our model on the test set \(S_i\). Then, the error of our model is calcuated as the mean of \(\{E_i | i \in [1..10]\}\).
</p>
