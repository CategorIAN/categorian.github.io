---
title: "Project 3: Network Flows"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>

<a href = "https://github.com/CategorIAN/CSCI_532_HW3">Code Repository</a>\
[Notes on Network Flows](https://categorian.github.io/pdfs/Notes on Network Flows.pdf)

<h2>Description</h2>
<h3>Flow Networks</h3>
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

<p>The following is code for the Ford-Fulkerson Algorithm:
{%highlight python linenos%}
def fordFulkerson(self, EdKarp, count):
  @tail_recursive
  def go(flow, i):
      resNetwork = ResidualNetwork(self, flow)
      resPath = resNetwork.augmentingPathBFS() if EdKarp else resNetwork.augmentingPathDFS()
      print("++++++++++++++++++++++++++++++")
      print("ResPath: {}".format(resPath))
      print("++++++++++++++++++++++++++++++")
      if resPath is None:
          return (flow, i) if count else flow
      else:
          return go.tail_call(resNetwork.augmentFlow(resPath, flow), i + 1)
  return go(self.initFlow(), 0)
{%endhighlight%}
</p>

<p>
As long as there is an augmenting path, we add the path to the flow function. For example, if our current flow is \(f\), and we pick our augmenting path \(p\) in \(G_f\) with \(x=c_f(p)\) units of flow along \(p\), then we update our flow function \(f\) to be \(f(e) \leftarrow f(e) + x\) for all \(e\in p\) and \(f(e) \leftarrow f(e)\) for all \(e \notin p\). The following is code to augment the current flow with the flow from the augmented path:
{%highlight python linenos%}
def augmentFlow(self, resPath, flow = None):
  flow = self.f if flow is None else flow
  if resPath is None:
      return flow
  else:
      signedPath = zip(resPath.path[1:], resPath.sign)
      augment = lambda f, e, s: f|{self.direct(e, bool(s)): f[self.direct(e, bool(s))] + int(pow(-1, s)) * resPath.res}
      augmentThrough = lambda flow_u, v_s: (augment(flow_u[0], (flow_u[1], v_s[0]), v_s[1]), v_s[0])
      return reduce(augmentThrough, signedPath, (flow, 0))[0]
{%endhighlight%}
</p>

<p>
Looking at the "fordFulkerson" function, we see that if "EdKarp" is false, the algorithm performs a depth first search to find an augmenting path. The code to find a path in a depth first search manner is shown below:
{%highlight python linenos%}
def augmentingPathDFS(self):
  @tail
  def go(pathlist, searched):
      #print("%%%%%%%%%%")
      #print("Paths: {}".format(PathSet(pathlist, sort = False)))
      if len(pathlist) == 0:
          return None
      else:
          path = pathlist[0]
          if path.isDone():
              return path
          else:
              newPaths = self.neighbor_paths(path, searched).paths
              return go.tail_call(newPaths + pathlist[1:], searched.union({path.head}))
  return go(self.neighbor_edges[0].paths, {0})
{%endhighlight%}
</p>

<p>
However, the Edmond Karp method is a breadth first search method because the method finds a path that has the fewest number of edges. The code to find a path in a breadth first search manner is shown below:
{%highlight python linenos%}
def augmentingPathBFS(self):
  def go(pathset, searched):
      if len(pathset.paths) == 0:
          return None
      else:
          finished = pathset.finishedPath()
          if finished is not None:
              return finished
          else:
              return go(*self.branch(pathset, searched))
  return go(self.neighbor_edges[0], {0})
{%endhighlight%}
</p>
