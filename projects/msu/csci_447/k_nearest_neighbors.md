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
Given an example to classify, <em>x</em>, we compute the distance of each training example, <em>y</em> from <em>x</em> using the Euclidean norm:
{%highlight python linenos%}
distances = train_set_values.map(lambda y: math.sqrt((x_vec - y.to_numpy()).dot(x_vec-y.to_numpy())))
{%endhighlight%}
</p>

<p>
We then sort the distances and take the <em>k</em> shortest distances.

{%highlight python linenos%}
dist_sorted = distances.sort_values()
dist_sorted = dist_sorted.take(range(k))
{%endhighlight%}
</p>

<p>
If the data is a classification set, we determine the class of <em>x</em> to be the most popular class among the k nearest neighbors of <em>x</em>
{%highlight python linenos%}
w = train_set.filter(items = nn, axis=0).groupby(by = ['Target'])['Target'].agg('count')
count = lambda cl: w.at[cl] if cl in w.index else 0
return rd(lambda cl1, cl2: cl1 if count(cl1) > count(cl2) else cl2, self.classes)
{%endhighlight%}
</p>
