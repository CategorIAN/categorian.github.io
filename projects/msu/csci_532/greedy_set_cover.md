---
title: "CSCI 532 Project 4: Greedy Set Cover"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>

<a href = "https://github.com/CategorIAN/CSCI_532_HW4">Code Repository</a>\
[Notes on Greedy Set Cover](https://categorian.github.io/pdfs/Notes on Greedy Set Cover.pdf)

<h2>Description</h2>
<p>
Let \(\mathcal{C} = \{S_1, S_2, ..., S_m\}\) be a collection of sets covering elements \(\{x_1, x_2, ..., x_n\}\). The set cover problem is to minimize \(|C|\) such that \(C\subseteq \mathcal{C}\) and \(\cup_{S_j \in C} S_j = \{x_1, ..., x_n\}\).
</p>

<p>
The greedy strategy is to choose the next set to add to \(C\) by picking the \(S_j\) that covers the most remaining uncovered elements. This strategy is an approximation algorithm to the set cover problem with an approximation ratio of \(\ln(n)\).
</p>

<p>
The following is code to find a covering in a greedy manner:
{%highlight python linenos%}
def covering(self):
    def go(size, covered, remaining, toRemove):
        #If the size of what is covered is the size of the total contained from the original collection, then stop.
        if size == self.totalContained:
            return covered
        else:
            def filterAndFind(reduced_max, k_s):
                #We want to subtract the last set added to our covering from the remaining sets. This is our "filter".
                mytuple = (k_s[0], k_s[1] - toRemove)
                #We want to keep the largest set from our remaining sets after filtering. This is our "find".
                mymax = mytuple if (reduced_max[1] is None or (mytuple[1] > reduced_max[1][1])) else reduced_max[1]
                #We keep our remaining collection of sets filtered. We keep the max.
                return (reduced_max[0] + mytuple, mymax)
            (reduced, toAdd) = reduce(filterAndFind, remaining.collection.items(), (SetCollection({}, 0), None))
            #We keep track how big our current covered set by adding the size of the added to the size.
            #We add the original set according to the max of the filtered by looking it up from the key of the max.
            #We keep keep looking through the remaining sets that are progressively filtered.
            #The set that was considered the max will be the next set to subtract from the remaining.
            return go(size + len(toAdd[1]), covered + (toAdd[0], self[toAdd[0]]), reduced, toAdd[1])
    return go(0, SetCollection({}, 0), SetCollection(self.collection, self.n), MySet(set()))
{%endhighlight%}
</p>

<p>
For a partial solution \(C\), let \(H(C) = \cup_{S_j \in C} S_j\). Then, we want to find index \(k\) that maximizes \(|S_k\backslash H(C)|\) and add \(S_k\) to our collection, \(C \leftarrow C + S_k\). We can view \(C\) as an ordered list such that \(C_i\) is the \(i^{th}\) subset we add to the collection. From our original subsets \([S_1, S_2, ..., S_m]\), we can keep track of the elements remaining to be added to our collection with reduced subsets \([T_1, T_2, ..., T_m]\) by starting with \(T_j = S_j\) for all \(j\in [1..m]\) and updating our reduced subsets by \(T_j \leftarrow T_j\backslash C_i\) for all \(T_j \notin C\) immediately after we add the \(i^{th}\) subset to our collection \(C\). With this data structure, for a partial solution \(C\), we simply find the index \(k\) that maximizes \(|T_k|\) and add \(S_k\) to our collection \(C \leftarrow C + S_k\). By maintaining these reduced subsets \([T_1, ..., T_m]\), we avoid having to calculate unnecessarily large set differences \(|S_k\backslash H(C)|\) from our original subsets \([S_k: S_k \notin C]\) every time we want to add another subset to a partial solution \(C\).
</p>
