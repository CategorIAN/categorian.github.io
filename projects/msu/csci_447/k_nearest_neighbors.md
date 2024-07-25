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

<h1>K-Nearest Neighbor Classifier</h1>
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
def nnEstimator(self, train_set, k, sigma = None, epsilon = None, edit = False, test_set = None, start_time = None):
if start_time is not None:
    if time.time() - start_time > 60 * 5:
        raise RuntimeError("Time is past 5 minutes.")
train_set_values = train_set.index.to_series().map(lambda i: self.value(train_set, i))
def nn_estimate_by_value(x):
    # print("--------------------")
    #print("New x to estimate")
    x_vec = x.to_numpy()
    #print("Computing Distances:")
    distances = train_set_values.map(lambda y: math.sqrt((x_vec - y.to_numpy()).dot(x_vec-y.to_numpy())))
    #print("Distances Computed")
    # print("Number of distances is {}".format(len(distances)))
    if k < len(distances):
        #print("Distances Number: {}".format(len(distances)))
        dist_sorted = distances.sort_values()
        #print("Sorted Distances Number: {}".format(len(dist_sorted)))
        dist_sorted = dist_sorted.take(range(k))
    else:
        dist_sorted = distances.sort_values()
    #print("Sorted Distances")
    nn = dist_sorted.index
    if self.classification:
        #print("Creating Train Frame")
        w = train_set.filter(items = nn, axis=0).groupby(by = ['Target'])['Target'].agg('count')
        count = lambda cl: w.at[cl] if cl in w.index else 0
        #print("Returning Class")
        return rd(lambda cl1, cl2: cl1 if count(cl1) > count(cl2) else cl2, self.classes)
    else:
        def kernel(u):
            return math.exp(-math.pow(u, 2) / sigma)
        v = dist_sorted.map(kernel).to_numpy()
        r = nn.map(lambda i: train_set.at[i, 'Target'])
        return v.dot(r)/v.sum()

        if edit:
            def correctly_classified(i):
                #print("Testing Neighbors on Themselves")
                target = self.nnEstimator(train_set.drop([i]), k, sigma=sigma, start_time=start_time)(self.value(train_set, i))
                #print("Target is {}".format(target))
                if self.classification:
                    return target == train_set.at[i, 'Target']
                else:
                    return abs(target - train_set.at[i, 'Target']) < epsilon
        
            yes = train_set.index.map(correctly_classified)
            no = yes.map(lambda y: not y)
        
            edited_neighbors = train_set.loc[train_set.index.map(correctly_classified)]
            print("Edited Out: {}".format(train_set.loc[no]))
            #print("Found Edited Neighbors")
            if train_set.shape[0] != edited_neighbors.shape[0]:
                #print("It is a smaller set.")
                pred_func = lambda set: self.comp(self.nnEstimator(set, k, sigma, start_time=start_time), pf(self.value, test_set))
                #print("Old Predictions")
                old_pred = test_set.index.map(pred_func(train_set))
                #print("New Predictions")
                new_pred = test_set.index.map(pred_func(edited_neighbors))
                actual = test_set['Target']
                if self.evaluator(old_pred, actual) >= self.evaluator(new_pred, actual):
                    #print("Recursively Edit Again")
                    return self.nnEstimator(edited_neighbors, k, sigma, epsilon, True, test_set, start_time)
        return nn_estimate_by_value
{%end highlight%}
</p>
