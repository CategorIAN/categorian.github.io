---
title: "Project 2: Segmented Line Fitting"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>
<a href = "https://github.com/CategorIAN/CSCI_532_HW2">Code Repository</a>\
[Notes on Segmented Line Fitting](https://categorian.github.io/pdfs/Notes on Segmented Line Fitting.pdf)

<h2>Description</h2>
<p>
Let \(OPT(j)\) be the optimal fit for points \((x_1, y_1), ..., (x_j, y_j)\). Let us define \(e_{ij}\) to be the error of the best fit for points \((x_i, y_i), ..., (x_j, y_j)\), which is calculated as 
\[
\sum_{k=i}^j (y_k - a_{ij}x_k - b_{ij})^2,
\]
such that 
\[
a_{ij} = \dfrac{n\sum_{k=i}^j x_k y_k - (\sum_{k=i}^j x_k) (\sum_{k=i}^j y_k)}{n \sum_{k=i}^j x_k^2 - (\sum_{k=i}^j x_k)^2},
\]
\[
b_{ij} = \dfrac{\sum_{k=i}^j y_k - a_{ij} \sum_{k=i}^j x_k}{n}.
\]
</p>
<p>
The following is code to compute this line of best fit:
{%highlight python linenos%}
def bestLine(self, i = 0, j = None):
    j = self.n - 1 if j is None else j
    n = j - i + 1
    (x, y) = (np.array(self.df['x'].iloc[i:j + 1]), np.array(self.df['y'].iloc[i:j + 1]))
    a = (n * x @ y - sum(x) * sum(y)) / (n * x @ x - sum(x) * sum(x))
    b = (sum(y) - a * sum(x)) / n
    return Line(a = a, b = b)
{%endhighlight%}
</p>
<p>
The following is code to compute the error of the best fit line from \(x_i\) to \(x_j\)
{%highlight python linenos%}
def leastSquaresError(self, i, j):
    if i >= j:
        return 0
    else:
        L = self.bestLine(i, j)
        fitted = L.fitPoints(self.df['x'].iloc[i:j + 1])
        resids = np.array(self.df['y'].iloc[i:j + 1] - fitted['y'])
        return resids @ resids
{%endhighlight%}
</p>
<p>
With this, we can compute the error matrix \(e\):
{%highlight python linenos%}
def errorMatrix(self):
    matrix = np.zeros((self.n, self.n))
    for i in range(self.n):
        for j in range(i + 1, self.n):
            matrix[i][j] = self.leastSquaresError(i, j)
    return matrix
{%endhighlight%}
<p>
Let \(c\) be the cost of adding a line. Then, \(OPT(j) = \min_{1\leq i \leq j}(e_{ij} + c + OPT(i-1))\). Thus, we can use dynamic programming to compute \(OPT(k)\) for \(k\) from \(0\) to \(n\) in increasing order. The following is code of this dynamic programming algorithm:
{%highlight python linenos%}
def segmentedLeastSquares(self, cost):
    def dynamicUpdate(array, j):
        index_errors = [(i, errorMatrix[i][j] + cost + array[i][1]) for i in range(j + 1)]
        minTuple = lambda t1, t2: t1 if t1[1] < t2[1] else t2
        (index, error) = reduce(minTuple, index_errors)
        return array + [(index, error)]

    errorMatrix = self.errorMatrix()
    tupleArray = reduce(dynamicUpdate, range(self.n), [(-1, 0)])
    (segmentArray, errorArray) = tuple([list(t)[1:] for t in tuple(zip(*tupleArray))])
    return PointPartition(self.recoverSegments(segmentArray), errorArray[self.n - 1])
{%endhighlight%}
</p>
<p>
The following are graphs of segmented line fitting with varying costs per line:</p>
<img src="/images/segmented_line_fitting.png" width = "1000" alt="">
