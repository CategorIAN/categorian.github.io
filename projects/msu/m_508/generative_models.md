---
title: "Project 2: Generative Models"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>

<a href = "https://github.com/CategorIAN/M508_HW2">Code Repository</a>\
[Notes on Generative Models](https://categorian.github.io/pdfs/Notes on Generative Models.pdf)

<h2>Description</h2>
<h3>Overview</h3>
<p>
Let \(x\) be a possible feature vector and \(y\) be a possible class for a labeled data set. Then, a generative model is one that computes \(P(x|y)\) and \(P(y)\) rather than calculating \(P(y|x)\) directly as done by discriminative models. After calculating \(P(y)\) and \(P(x|y)\), we can calculate \(P(y|x)\) using Bayes' rule:
\[
P(y|x) = \dfrac{P(x|y)P(y)}{P(x)}
\]
</p>

<p>
If we are given training data, we calculate the probability of each class \(P(y) = \frac{\text{# of samples of class y}}{\text{# of samples in the data set}}\). We can create a dataframe that stores the probability of each class of our training data. The following code computes this dataframe:
{%highlight python linenos%}
def getQ(self):
    '''
    :return: Pandas dataframe that shows the probability of each class in the data
    '''
    df, target = self.data.df, self.data.target_name
    Qframe = pd.DataFrame(df.groupby(by=[target])[target].agg("count")).rename(columns={target: "Count"})
    return pd.concat([Qframe, pd.Series(Qframe["Count"] / df.shape[0], name="Q")], axis=1)
{%endhighlight%}
</p>

