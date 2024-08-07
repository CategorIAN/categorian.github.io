---
title: "CSCI 447 Project 1: Naive Bayes"
layout: default
---
<h1>{{page.title}}</h1>

<h2>References</h2>
<a href = "https://github.com/CategorIAN/CSCI447_Project_1">Code Repository</a>\
[Full Report](https://categorian.github.io/pdfs/CSCI_447_Project_1.pdf)

<h2>Description</h2>

<p>
For a given training set, for each class, we computed
\[
Q(C=c_i) = \dfrac{\#\{\textbf{x}\in c_i\}}{N},
\]
where \(N\) is the number of examples in the training set. The code to implement our <i>Q</i> values is shown below:
{%highlight python linenos%}
#probability of the class
def getQ(self):
        df = pd.DataFrame(self.train_set.groupby(by = ['Class'])['Class'].agg('count')).rename(columns =
                                                                                               {'Class': 'Count'})
        df['Q'] = df['Count'].apply(lambda x: x / self.train_set.shape[0])
        return df
{%endhighlight%}
  
  
Then, for each attribute and each class, we calculate, 
\[
F(A_j = a_k, C=c_i) = \dfrac{\#\{(\textbf{x}_{A_j} = a_k) \wedge (\textbf{x} \in c_i)\} +1}{N_{c_i} + d},
\]
which was computed with the following code:
{%highlight python linenos%}
#probability of a sigle feature 
    def getF(self, j, m, p, Qtrain = None): 
        if Qtrain is None: Qtrain = self.getQ()
        df = pd.DataFrame(self.train_set.groupby(by = ['Class', self.features[j]])['Class'].agg('count')).rename(
                                                                                        columns = {'Class' : 'Count'})
        y = []
        for ((cl, _), count) in df['Count'].to_dict().items():
            y.append((count + 1 + m*p)/(Qtrain.at[cl, 'Count'] + len(self.features) + m)) 
        df['F'] = y
        return df
{%endhighlight%}

Then, to classify an example from the test set, we compute 
\[
C(\textbf{x}) = Q(C = c_i)\times \prod_{j=1}^{d} F(A_j = a_k, C=c_i),
\]
which was computed with the code:
{%highlight python%}
#Calculate the probabilities of the class based on the features
    def C(self, cl, x, Qtrain = None, Ftrains = None):
        if Qtrain is None: Qtrain = self.getQ()
        if Ftrains is None: Ftrains = self.getFs(Qtrain)
        result = Qtrain.at[cl, 'Q']
        for j in range(len(self.features)):
            F = Ftrains[j]
            if (cl, x[j]) in F.index:
                result = result * F.at[(cl, x[j]), 'F']
            else: return 0
        return result
{%endhighlight%}
Then, we return 
\[
class(\textbf{x}) = argmax_{c_i\in C}C(x),
\]
which returns the class with the highest value for C(<b>x</b>). The function that predicted the class of an example with features <b>x</b> is shown here:
{%highlight python%}
#predict the class value
    def predicted_class(self, x, Qtrain = None, Ftrains = None):
        if Qtrain is None: Qtrain = self.getQ()  #create class if not there
        if Ftrains is None: Ftrains = self.getFs(Qtrain)  #create feature set if not there
        (argmax, max_C) = (None, 0) 
        for cl in Qtrain.index:
            y = self.C(cl, x, Qtrain, Ftrains)
            if y > max_C:
                argmax = cl
                max_C = y
        return argmax
{%endhighlight%}
</p>
