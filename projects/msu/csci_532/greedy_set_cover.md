---
title: "Project 4: Greedy Set Cover"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>

<a href = "https://github.com/CategorIAN/CSCI_532_HW4">Code Repository</a>\
[Notes on Greedy Set Cover](https://categorian.github.io/pdfs/Notes on Greedy Set Cover.pdf)

<h2>Description</h2>
<p>
Let \(\mathcal{C} = \{S_1, S_2, ..., S_m\}\) be a collection of sets covering elements \(\{x_1, x_2, ..., x_n\}\). The set cover problem is to minimize \(|C|\) such that \(C\subseteq \mathcal{C}\) and \(\cup_{S_j \in C} S_j = \{x_1, ..., x_n\}\) = .
</p>

<p>
The greedy strategy is to choose the next set to add to \(C\) by picking the \(S_j\) that covers the most remaining uncovered elements. This strategy is an approximation algorithm to the set cover problem with an approximation ratio of \(\ln(n)\).
</p>
