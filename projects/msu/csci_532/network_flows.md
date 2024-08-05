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
Furthermore, there is a flow function \(f: V\times V \rightarrow \mathbb{Z} that must satisfy the following constraints:
\[
\begin{enumerate}
  \item f(u, v) \leq c(u, v) \text{ for all }u, v \in V \text{ (capacity constraint)}
  \item f(u, v) = - f(v, u) \text{ (skew-symmetry) }
  \item \forall u \in V \backslash \{s, t\}, \sum_{v\in V} f(u, v) = 0 \text{ (conservation of flow)}
</p>
