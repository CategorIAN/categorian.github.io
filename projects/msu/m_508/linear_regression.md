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
For each example \(i\), the components that represent our features are \([x_1^{(i)}, x_2^{(i)}, ..., x_d^{(i)}] \in \mathbb{R}^d\), the intercept component is \(x_0^{(i)} = 1\), and our target value is \(y^{(i)}\). Then, let \(x^{(i)} = [x_0^{(i)}, x_1^{(i)}, ..., x_d^{(i)}] \in \mathbb{R}^{d+1}\), and let our linear regression weights be \(\theta \in \mathbb{R}^{d+1}\). Then, our hypothesis function is defined to be \(h_{\theta}(x^{(i)}) = \sum_{j=0}^{d+1} \theta_j x_j^{(i)} = \theta^{\intercal} x^{(i)} = \theta \cdot x^{(i)}\). We want to train our regression weights \(\theta\) by minimizing the least squares loss function
\[
J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_{\theta}(x^{(i)}) - y^{(i)})^2.
\]
</p>

<p>
Code that calculates this error is shown below:
{%highlight python linenos%}
def J(self, X, Y, theta):
    '''
    :param X: data matrix of predictors
    :param Y: target vector
    :param theta: vector of regression weights
    :return: the error of the regression model
    '''
    v = theta[0] + X @ theta[1:] - Y
    return 0.5 * np.dot(v.T, v)[0][0]
{%endhighlight%}
</p>

<p>
We update our \(\theta\) by moving in the negative direction of the gradient of our loss function:
\[
\theta \leftarrow \theta - \alpha \nabla J(\theta),
\]
where \(\alpha\) is a hyperparameter called the learning rate. 
</p>

<p>
This update on \(\theta\) can be shown to be 
\[
\theta \leftarrow \theta + \alpha \sum_{i=1}^n (y^{(i)} - h_{\theta}(x^{(i)}))x^{(i)}.
\]
This is called batch gradient descent.
</p>

<p>
Unfortunately, batch gradient descent is very computationally intensive. Another way to update is by stochastic gradient descent, where we update for each \(i=1, 2, ..., n,\)
\[
\theta \leftarrow \theta + \alpha(y^{(i)} - h_{\theta}(x^{(i)})x^{(i)},
\]
so we update \(\theta\) one data point at a time.
</p>

<p>
The following code shows how we update \(\theta\) for each data point:
{%highlight python linenos%}
def theta_update(self, alpha):
    '''
    :param alpha: the learning rate
    :return: function that takes theta and index and returns updated theta
    '''
    def f(theta, i):
        '''
        :param theta: vector of regression weights
        :param i: index of dataset to use for updating
        :return: an updated vector of regression weights
        '''
        grad = ((theta[0] + self.X_train_mat[i] @ theta[1:] - self.Y_train_vec[i]) *
                np.concatenate(([1], self.X_train_mat[i].T)))
        return theta - alpha * grad
    return f
{%endhighlight%}
</p>

<p>
Updating \(\theta\) for each example in our training set is considered to be an goint through an epoch. The following code returns the updated \(\theta\) after going through an epoch:
{%highlight python linenos%}
def epoch(self, alpha, start_theta = None):
    '''
    :param alpha: the learning rate
    :param start_theta: starting regression weights
    :return: an updated vector of regression weights after going through dataset
    '''
    @tail_recursive
    def go(theta, i):
        if i == n:
            return theta
        else:
            new_theta = theta_update_func(theta, index[i])
            return go.tail_call(new_theta, i + 1)
    start_theta = np.random.rand(1 + len(self.data.feats_enc)) if start_theta is None else start_theta
    theta_update_func = self.theta_update(alpha)
    n = self.X_train_mat.shape[0]
    index = random.sample(range(n), k=n)
    return go(start_theta, 0)
{%endhighlight%}
</p>
