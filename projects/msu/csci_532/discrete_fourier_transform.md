---
title: "Project 1: Discrete Fourier Transform"
layout: default
---
<h1>{{page.title}}</h1>

<h2>Reference</h2>
<a href = "https://github.com/CategorIAN/CSCI_532_HW1">Code Repository</a>

<h2>Description</h2>
<p>
Given polynomials \(a(x)\) and \(b(x)\), we want a fast way to multiply the polynomials \(c(x) = a(x)b(x)\). Suppose the degree of polynomial \(p\) is \(n\). Then, \(p\) can be uniquely represented as \(T(p)=(p(\omega_n^0), p(\omega_n^1), p(\omega_n^2), ..., p(\omega_n^{n-1}))\), where \(\omega_n\) is the \(n^{th}\) root of unity. 
</p>

<p>
We want to use a divide-and-conquer method to break down a polynomial. If \(p(x) = p_0x^0 + p_1x_1 + ... + p_{n-1}x^{n-1}\), then \(p(x) = (p_0x^0 + p_2x^2 + p_4x^4 + ... ) + (p_1x^1 + p_3x^3 + p_5x^5 + ... ) = p^{[0]}(x^2) + x\cdot p^{[1]}(x^2)\), where \(p^{[0]}(x) = p_0x^0 + p_2x^1 + p_4x^2 + ...\) and \(p^{[1])(x) = a_1 + a_3x + p_5x^2 + ...\).
</p>
