---
title: "M 508 Project 2: Generative Models"
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

<p>
How we compute the conditional probability \(P(x|y)\) depends on the specific model we use. The two types of generative models we will focus on are the Gaussian Discriminant Analysis (GDA) model and the Naive Bayes model. Once we calculate \(P(y)\) and \(P(x|y)\), we can get some measure of \(P(y|x)\) by computing \(P(x, y) = P(x|y)P(y)\). Looking at Bayes' rule, this is not quite \(P(y|x)\), since we are not dividing \(P(x)\). However, this probability is not important since we only need to compare probabilities. Specifically, if we are given a feature vector \(x\) from the test set, and we want to know which class among \(y_1, y_2\) is more probable, then we only need to compare \(P(x,y_1)\) and \(P(x,y_2)\) since \(P(x, y_1) < P(x, y_2) \Leftrightarrow P(y_1|x) < P(y_2|x)\). The following code shows how to compute \(P(x, y)\) for feature vector \(x\) and class \(y\):
{%highlight python linenos%}
def class_prob(self, cl, x):
    '''
    :param cl: The class
    :param x: The data features
    :return: The probability of getting the data features and the class
    '''
    cond_prob = self.cond_prob_func(cl, x)
    return cond_prob * self.Q.at[cl, "Q"]
{%endhighlight%}
</p>

<p>
Thus, with a given feature vector \(y\), we predict the class by finding that class \(y\) that yields the largest \(P(y|x)\):
\[
y^* = \text{argmax}_{y\in \text{Classes}} P(y|x) = \text{argmax}_{y\in \text{Classes}} P(x, y)
\]
The following code computes the predicted class \(y^*\) for a given feature vector \(x\):
{%highlight python linenos%}
def predicted_class(self, x):
    '''
    :param x: The data features.
    :return: The class most likely having the given features
    '''
    cl_probs = self.Q.index.map(lambda cl: (cl, self.class_prob(cl, x)))
    return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, cl_probs, (None, None))[0]
{%endhighlight%}
</p>
<h3>Gaussian Discriminant Analysis (GDA)</h3>
<p>
For GDA, we assume a multivariate normal distribution of the data. Let us assume the data has \(d\) features and the training data has \(n\) examples. From the training data, we find the mean vector \(\mu \in \mathbb{R}^d\) such that \(\mu_j = \dfrac{\sum_{i=1}^n x^{(i)}_j}{n}\) for j in [1..d], and we find the covariance matrix \(\Sigma \in \mathbb{R}^{d\times d}\) such that \(\Sigma_{ij} = E[(X_i - \mu_i)^{\intercal}(X_j - \mu_j)]\) for i, j in [1..d]. Code to compute the covariance matrix is shown below:
{%highlight python linenos%}
def covMat_Components(self):
    '''
    :return: The invertible covariance matrix based on nonzero-variance components along with those components
    '''
    def addMatrix(m, i):
        #print("-----------------------------")
        #print("i: {}".format(i))
        v = self.X[i] - self.mu_dict[self.data.target(i)]
        return m + np.outer(v, v)
    M = reduce(addMatrix, range(self.n), np.zeros((self.d, self.d))) / self.n
    alpha = 0.5
    nonzero_comps = list(self.filter(pd.Series(range(self.d)), lambda i: np.linalg.norm(M[i, :]) > (10 ** alpha)))
    return M[np.ix_(nonzero_comps, nonzero_comps)], nonzero_comps
{%endhighlight%}
</p>

<p>
Once we have the mean vector and the covariance matrix, we then calculate the probability of an example with feature vector \(x\) to be 
\[
    P(x;\mu, \Sigma) = \dfrac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp(-\frac{1}{2}(x-\mu)^{\intercal}\Sigma^{-1}(x-\mu))
\]
</p>

<p>
For it to be a generative model, we want to find for a given class \(y\), we want to find the probability of an example having the feature vector \(x\) given class \(y\), \(P(x|y)\). To find this probability, instead of using \(\mu\), we use \(\mu(y)\), which is calculated as, for each j in [1..d] 
\[
\mu(y)_j = \dfrac{\sum_{i\in S_y} x^{(i)}_j}{|S_y|},
\]
where \(S_y\) is the set of \(\{i\in [1..n]| y^{(i)} = y\}\). Thus, for the training data, we need to calculate \(\mu(y)\) for each class \(y\). Code to compute this vector for a given class is shown below:
{%highlight python linenos%}
def mu(self, cl):
    '''
    :param cl: The class
    :return: The average feature value per feature within the class
    '''
    return np.array(self.data.df.loc[lambda df: df[self.data.target_name] == cl][self.data.features].mean())
{%endhighlight%}

Then, for an example in the test set with feature vector \(x\), the conditional probability is calculated as \(P(x|y) = P(x;\mu(y), \Sigma)\). Code to compute the conditional probability with the GDA model is shown below:
{%highlight python linenos%}
def cond_prob_func(self, cl, x):
    '''
    :param cl: The class
    :param x: The data features
    :return: The probability of getting the data features given the class
    '''
    v = np.array(x)[self.components] - self.mu_dict[cl][self.components]
    return np.exp(-0.5 * v @ self.Sigma_inv @ v) / ((2 * np.pi) ** (self.p / 2) * self.Sigma_det ** (1 / 2))
{%endhighlight%}
</p>

<h3>Naive Bayes</h3>
<p>
Let us assume an example has class \(y\). Let \(x=[x_1, x_2, ..., x_d]\) be a feature vector. The product rule of probabilities tells us that 
\[
P(x_1, ..., x_d|y) = P(x_1|y)\cdot P(x_2|y, x_1)\cdot ...\cdot P(x_d|y, x_1, x_2, ..., x_{d-1}).
\]
The Naive Bayes assumption is that for any class \(y\), the features \(x_1, x_2, ..., x_d\) are conditionally independent on \(y\):
\[
P(x_1, ..., x_d|y) = P(x_1|y)\cdot P(x_2|y) \cdot ... \cdot P(x_d | y) = \prod_{j=1}^d P(x_i | y).
\]
</p>


