---
title: "CSCI 447 Project 2: K Nearest Neighbor Classifier"
layout: default
---

<h1>References</h1>
<a href = "https://github.com/EthanSkelton9/csci447_project2">Code Repository</a>\
[Full Report](https://categorian.github.io/pdfs/CSCI_447_Project_2.pdf)

<h1>Description</h1>

<p>
For my second project in my Machine Learning course, my partner, Ethan Skelton, and I created a K Nearest Neighbor supervised learning model. Our algorithm trains a model using a real world data set to predict the class of examples from the same data set. The examples used to train the model make up the training data, and the examples that had their classes predicted from the model make up the test data. The assignment of training data and test data for any given data set was created from 10-fold cross validation. 
</p>

<h2>K-Nearest Neighbor Classifier</h2>
<p>
Hello World
{%highlight python %}
'''
nnEstimator returns a function that predicts the target of an example using k nearest neighbors
@param train_set - the set that we will use for our neighbors
@param k - the number of neighbors we will use to predict an example
@param sigma - the band width only used in regression sets
@param epsilon - the max tolerance used to determine if two regression examples have the same target for editing
@param edit - determines whether to use edited nearest neighbors or not
@param test_set - test set used to determine whether the edited neighbors improves performance
@return function that takes @param example x and returns predicted class or target value
'''
{%end highlight%}
</p>
