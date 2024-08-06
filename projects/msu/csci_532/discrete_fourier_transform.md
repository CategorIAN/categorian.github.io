---
title: "CSCI 532 Project 1: Discrete Fourier Transform"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>
<a href = "https://github.com/CategorIAN/CSCI_532_HW1">Code Repository</a>\
[Notes on Discrete Fourier Transform](https://categorian.github.io/pdfs/Notes on Discrete Fourier Transform.pdf)

<h2>Description</h2>
<p>
Given polynomials \(a(x)\) and \(b(x)\), we want a fast way to multiply the polynomials \(c(x) = a(x)b(x)\). Suppose the degree of polynomial \(p\) is \(n\). Then, \(p\) can be uniquely represented as \(T(p)=(p(\omega_n^0), p(\omega_n^1), p(\omega_n^2), ..., p(\omega_n^{n-1}))\), where \(\omega_n\) is the \(n^{th}\) root of unity. 
</p>

<p>
We want to use a divide-and-conquer method to break down a polynomial. If \(p(x) = p_0x^0 + p_1x_1 + ... + p_{n-1}x^{n-1}\), then \(p(x) = (p_0x^0 + p_2x^2 + p_4x^4 + ... ) + (p_1x^1 + p_3x^3 + p_5x^5 + ... ) = p^{[0]}(x^2) + x\cdot p^{[1]}(x^2)\), where \(p^{[0]}(x) = p_0x^0 + p_2x^1 + p_4x^2 + ...\) and \(p^{[1]}(x) = p_1 + p_3x + p_5x^2 + ...\).
</p>

<p>Using the recursion above and the invertible linear transformation of \(T\), we can calculate the \(T(p)\).
{%highlight python linenos%}
def DFT(self, n = None, inv = False, dec = 2):
  Q = self.padded(n)
  n = Q.deg + 1
  def go(P, k):
      if P.deg == 0:
          return pow(1 / n, int(inv)) * P.coeffs[0] * np.ones(n)
      else:
          (evenDFT, oddDFT) = (go(P.even(), 2 * k), go(P.odd(), 2 * k))
          W = np.vectorize(lambda i: self.rou(n, k * i))(np.array(range(n)))
          return np.vectorize(lambda v: self.cround(v, dec))(evenDFT + W * oddDFT)
  return go(Q, pow(-1, int(inv)))
{%endhighlight%}
</p>

<p>The padded function is to make sure the degree of the polynomial is a power of \(2\) so that we can always perform the recursion. 
{%highlight python linenos%}
def padded(self, limit = None):
  limit = 0 if limit is None else limit
  size = 1
  n = self.deg + 1
  while size < n or size < limit:
      size = size * 2
  return Polynomial(self.coeffs + (size - n) * [0])
{%endhighlight%}
</p>

<p>
The "rou" function is the \(n^{th}\) root of unity, \(\omega_n = e^{i \cdot \frac{2\pi}{n}}\).
{%highlight python linenos%}
def rou(self, n, k = 1):
  return pow(exp(complex(0, 2 * pi / n)), k)
{%endhighlight%}
</p>

<p>Thus, once we can express polynomials \(a(x)\) and \(b(x)\) as \(T(a)\) and \(T(b)\), respectively, we can then multiply their vectors point-wise \(T(a)*T(b)\). This vector represents the product of the original polynomials. Thus, to get the final result, we simply use the inverse transformation of \(T\), \(T^{-1}\). Thus, the product is computed by \(a(x) * b(x) = T^{-1}(T(a)*T(b))(x)\). The following code computes this fast multiplication:
{%highlight python linenos%}
def fast_mult(self, Q, dec = 2):
  n = self.deg + Q.deg + 1
  prod = self.DFT(n, dec = dec) * Q.DFT(n, dec = dec)
  coeffs = Polynomial(list(prod)).DFT(inv = True, dec = dec)
  return Polynomial([round(c.real, 0) for c in coeffs])
{%endhighlight%}
</p>
