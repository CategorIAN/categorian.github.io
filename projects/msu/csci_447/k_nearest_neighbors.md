---
title: "CSCI 447 Project 2: K Nearest Neighbor Classifier"
layout: default
---
<h1>{{page.title}}</h1>

<h2>References</h2>
<a href = "https://github.com/EthanSkelton9/csci447_project2">Code Repository</a>\
[Full Report](https://categorian.github.io/pdfs/CSCI_447_Project_2.pdf)

<h2>Description</h2>

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
If the data is a classification set, we determine the class of <em>x</em> to be the most popular class among the k nearest neighbors of <em>x</em>:
{%highlight python linenos%}
w = train_set.filter(items = nn, axis=0).groupby(by = ['Target'])['Target'].agg('count')
count = lambda cl: w.at[cl] if cl in w.index else 0
return rd(lambda cl1, cl2: cl1 if count(cl1) > count(cl2) else cl2, self.classes)
{%endhighlight%}
</p>

<p>
However, if the data is a regression set, we determine the target value of the <em>x</em> by a weighted average of the targets of its k nearest neighbors:
{%highlight python linenos%}
def kernel(u):
    return math.exp(-math.pow(u, 2) / sigma)
v = dist_sorted.map(kernel).to_numpy()
r = nn.map(lambda i: train_set.at[i, 'Target'])
return v.dot(r)/v.sum()
{%endhighlight%}

The weights are determined by the Gaussian Kernel Function:
\[K(u) = \exp{[-\frac{u^2}{\sigma}]},\]
where \(\sigma\) is a hyperparameter to be tuned. 
</p>

