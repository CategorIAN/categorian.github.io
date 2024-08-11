---
title: "M 508 Project 2: Generative Models"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>

<a href = "https://github.com/CategorIAN/M508_HW2">Code Repository</a>\
[Notes on Generative Models](https://categorian.github.io/pdfs/Notes on Generative Models.pdf)\
<a href = "https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data">MNIST Data Set</a>\

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
P(x_1, ..., x_d|y) = P(x_1|y)\cdot P(x_2|y) \cdot ... \cdot P(x_d | y) = \prod_{j=1}^d P(x_j | y).
\]
With this assumption, we can store from the training data for each class \(y\) and for each feature value \(x_j\) for each j in [1..d] from the training data and compute the conditional probabilities \(P(x_j|y)\) and store them in a dataframe. Calculating \(P(x_j|y)\) is found by 
    \[ 
    P(x_j|y) = \dfrac{|S_{x_j} \cap S_y|}{|S_y|},
    \]
where, as before, \(S_y\) is the set of \(\{i\in [1..n]| y^{(i)} = y\}\), and \(S_{x_j}\) is the set of \(\{i\in [1..n]| x^{(i)}_j = x_j\}\). The following code computes a dataframe of all of the \(P(x_j | y)\) for all feature values \(x_j\) and classes \(y\) for a given feature \(j\):
{%highlight python linenos%}
    def F(self, j):
        '''
        :param j: The index of the feature
        :return: A dictionary that maps class-feature value tuples to probability of feature value given class
        '''
        target = self.data.target_name
        grouped_df = self.data.df.groupby(by=[target, self.data.features[j]])
        Fframe = pd.DataFrame(grouped_df[target].agg("count")).rename(columns={target: "Count"})
        Ffunc = lambda t: (Fframe["Count"][t] + 1) / (self.Q.at[t[0], "Count"] + len(self.data.features))
        Fcol = Fframe.index.to_series().map(Ffunc)
        return pd.concat([Fframe, pd.Series(Fcol, name = "F")], axis = 1).to_dict()["F"]
{%endhighlight%}
We then can create a dictionary that maps a feature \(j\) to its dataframe of \(P(x_j|y)\) values:
{%highlight python linenos%}
self.Fs = dict([(j, self.F(j)) for j in range(len(self.data.features))])
{%endhighlight%}
</p>

<p>
Once we have this dictionary, we can then use our conditional probability function that takes a class \(y\) and feature vector \(x\) and computes \(P(x|y)\) using the Naive Bayes assumption:
{%highlight python linenos%}
def cond_prob_func(self, cl, x):
    '''
    :param cl: The class
    :param x: The data features
    :return: The probability of getting the data features given the class
    '''
    return reduce(lambda r, j: r * self.Fs[j].get((cl, x[j]), 0), range(len(self.data.features)), 1)
{%endhighlight%}
</p>

<h2>Models Tested On MNist Data</h2>
<h3>MNIST</h3>
<p>
We tested our generative models on the MNIST data set, which is a data set of handwritten digits. The objective is to predict the digit from the handwriting image. The only classes we considered were the "0" and "1" classes. The following is code to extract the MNIST and write it to csv file:
{%highlight python linenos%}
import numpy as np  # linear algebra
import struct
from array import array
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
from MLData import MLData

class MnistDataloader(object):
    def __init__(self, createcsvs = False):
        '''
        :param createcsvs: Determines whether to create csv files of 0-1 training and test data.
        '''
        names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
        files = ["\\".join([os.getcwd(), "archive", name, name]) for name in names]
        self.train_images_file, self.train_labels_file, self.test_images_file, self.test_labels_file = files
        if createcsvs:
            self.zeroOneCSV("train")
            self.zeroOneCSV("test")
        features = ["({},{})".format(i // 28, i % 28) for i in range(28 * 28)]
        zero_one_file = lambda type: "\\".join([os.getcwd(), "zero_one_csvs", "zero_one_{}.csv".format(type)])
        self.zero_one_train = MLData("MNist_ZeroOne_Train", zero_one_file("train"), features, "Class")
        self.zero_one_test = MLData("MNist_ZeroOne_Test", zero_one_file("test"), features, "Class")

    def read_images_labels(self, images_file, labels_file):
        '''
        :param images_file: File that contains the images.
        :param labels_file: File that contains the labels.
        :return: A tuple of the images, a list of numpy arrays, and labels, a list of strings.
        '''
        with open(labels_file, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_file, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        image_func = lambda i: np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
        images = [image_func(i) for i in range(size)]
        return images, labels

    def load_data(self, part = None):
        '''
        :param part: Either 'train' or 'test' or None (to get both)
        :return: A tuple of the data features and the data targets
        '''
        if part == "train":
            x_train, y_train = self.read_images_labels(self.train_images_file, self.train_labels_file)
            return x_train, y_train
        elif part == "test":
            x_test, y_test = self.read_images_labels(self.test_images_file, self.test_labels_file)
            return x_test, y_test
        else:
            x_train, y_train = self.read_images_labels(self.train_images_file, self.train_labels_file)
            x_test, y_test = self.read_images_labels(self.test_images_file, self.test_labels_file)
            return (x_train, y_train), (x_test, y_test)

    def show_images(self):
        '''
        :return: Shows random image-label pairs from both the training data and the test data.
        '''
        (x_train, y_train), (x_test, y_test) = self.load_data()
        def index_image_title(i):
            if i < 10:
                r = random.randint(1, 60000)
                return i + 1, x_train[r], "train [{}] = {}".format(r, y_train[r])
            else:
                r = random.randint(1, 10000)
                return i + 1, x_test[r], "test [{}] = {}".format(r, y_test[r])

        size, cols = 15, 5
        rows = int(size / cols) + 1
        plt.figure(figsize=(30, 20))
        for index, image, title in [index_image_title(i) for i in range(15)]:
            plt.subplot(rows, cols, index)
            plt.title(title, fontsize=15)
            plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.show()

    def filter(self, s, predicate):
        return pd.Series(dict(reduce(lambda l, i: l + [(i, s[i])] if predicate(s[i]) else l, s.index, [])))

    def zeroOneCSV(self, part):
        '''
        :param part: Either 'train' of 'test'
        :return: Writes a CSV file of the 0-1 data.
        '''
        x, y = self.load_data(part)
        cols = ["({},{})".format(i // 28, i % 28) for i in range(28 * 28)] + ["Class"]
        zero_one_index = self.filter(pd.Series(y), lambda v: v in {0, 1}).index
        d = dict([(i, np.concatenate((x[i].flatten(), [y[i]]))) for i in zero_one_index])
        df = pd.DataFrame.from_dict(d, "index", columns = cols)
        df.to_csv("\\".join([os.getcwd(), "zero_one_csvs", "zero_one_{}.csv".format(part)]))
{%endhighlight%}
</p>
<h3>Error Calculated In Confusion Matrices</h3>
<p>
We calculated the performance of each model using a confusion matrix.
</p>
<h4>Confusion Matrix of GDA on MNIST</h4>
<img src="/images/GDA CF.png" width = "500" alt="">

<h4>Confusion Matrix of Naive Bayes on MNIST</h4>
<img src="/images/NaiveBayes.png" width = "500" alt="">

