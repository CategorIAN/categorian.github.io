---
title: "Project 3: Network Flows"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>
<a href = "https://github.com/CategorIAN/CSCI_532_HW3">Code Repository</a>\
[Notes on Network Flows](https://categorian.github.io/pdfs/Notes on Network Flows.pdf)

<h2>Description</h2>
<p>
A flow network consists of a directed graph \(G = (V, E)\) such that there exists \(s, t \in V\), where \(s\) is the source and \(t\) is the sink. Furthermore, there is a capacity function \(c: V\times V \rightarrow \mathbb{Z}^+\) such that \(c(u, v) = 0\) for all \((u, v) \notin E \).
</p>

<p>
Furthermore, there is a flow function \(f: V\times V \rightarrow \mathbb{Z}\) that must satisfy the following constraints:
<ul>
<li> \(f(u, v) \leq c(u, v) \text{ for all }u, v \in V \text{ (capacity constraint)}\) </li>
<li> \(f(u, v) = - f(v, u) \text{ (skew-symmetry) }\) </li>
<li> \(\forall u \in V \backslash \{s, t\}, \sum_{v\in V} f(u, v) = 0 \text{ (conservation of flow)}\) </li>
</ul>
</p>

<p>
For \(X, Y \subseteq V\), let \(f(X, Y) = \sum_{x\in X} \sum_{y \in Y} f(x, y)\). Then, the flow value is defined as \(|f| = f(\{s\}, V)\). Our objective is to maximize the flow value. 
</p>

<p>
For \(f\) be a flow for graph \(G\). Then, for all \((u, v) \in V\times V\), let the residual capacity be defined as \(c_f(u, v) = c(u, v) - f(u, v)\). Let \(G_f\) be the residual network \((V, E_f)\) such that \(E_f = \{(u, v) \in V \times V : c_f(u, v) > 0\}\). Define an augmenting path to be any \(s, t\) path in \(G_f\). We say that \(e \in p\) is critical if \(c_f(e) = \min_{e'\in p} c_f(e').\)
</p>

<h3>The Ford-Fulkerson Algorithm</h3>
<p>
We start with \(f=0\). While there exists an augmenting path \(p\) in \(G_f\), add \(c_f(p)\) units of flow along \(p\). Return flow \(f\). It can be shown that \(f\) is a max flow for graph \(G\).
</p>
