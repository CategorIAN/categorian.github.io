---
title: "CSCI 447 Project 3: Deep Learning"
layout: default
---

<h1>{{page.title}}</h1>

<h2>References</h2>
<a href = "https://github.com/EthanSkelton9/csci447_project3">Code Repository</a>\
[Full Report](https://categorian.github.io/pdfs/CSCI_447_Project_3.pdf)

<h2>Description</h2>
<p>
We used a feed forward neural network to predict the classes and target values of our data. For our regression sets, the final layer consisted of a real value \(V\in \mathbb{R}\) that represented our prediction for the target value. For our classification sets, the final layer consisted of \(V \in \mathbb{R}^n\), where \(n\) is the number of classes and \(V_i\) is the value associated with class \(i\). Then, the predicted class was determined to be the softmax \(j = \text{argmax}_{i\in [1..n]} V_i\).
</p>

<p>
We calcualted our hidden layers using the sigmoidal activation function:
{%highlight python linenos%}
'''
calc_Hidden - calculates the hidden layers on the weights
@param weights[] - weights  between the layers 
@param row: the row of features values that we are using
@param data - the data that we want to read in
@length - number of hidden layers that we have

@return hidden_layers - returns the hidden layers that we created
'''
def calc_Hidden(self, weights, row, num_hidden):
    hidden_layers = []
    layers = []
    
    layers.append(row)
    
    if(num_hidden == 0): #if there are no hidden layers
        return hidden_layers
    else:
        hidden_layers.append(self.sigmoid_v(weights[0]@row)) #find the first hidden layer
        for i in range(num_hidden-1):
            hidden_layers.append(self.sigmoid_v(weights[i+1]@hidden_layers[i])) #all hidden layers after
            
        return hidden_layers #return the hidden layers 
        
    return hidden_layers #return the hidden layers
{%endhighlight%}
</p>

<p>
We trained our neural network using gradient descent and backpropagation. 
{%highlight python linenos%}
error = np.array([r(i) - yi])                                     # return errors at each of the outputs
grads = []
wzs = zip(ws, zs)
previous_z = None
for (w, z) in list(wzs)[::-1]:
    grads = [np.outer(error * self.dsigmoid_v(previous_z), z)] + grads   # create gradient
    error = error @ w                                               # back propagate error
    previous_z = z
new_ws = pd.Series(zip(ws, grads)).map(lambda wg: wg[0] + eta * wg[1])           #calculate new weights
new_ss = None if ss is None else grads                                        #calculate new gradients
return f.tail_call(index_remaining[1:], new_ws, new_ss, y_acc + [(i, yi)])
</p>
