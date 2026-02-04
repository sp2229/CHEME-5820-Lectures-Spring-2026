# Advanced: Classical and Modified Gram-Schmidt
Before we can implement [QR iteration](https://en.wikipedia.org/wiki/QR_algorithm), we need a QR-decompostion method which returns the $\mathbf{Q}$ and $\mathbf{R}$ matrices. 

The [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) can be computed using a variety of approaches, including a handy technique called [the Gram–Schmidt algorithm](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process). In principle, Gram-Schmidt orthogonalization generates a set of mutually orthogonal vectors $\mathbf{q}_{1},\mathbf{q}_{2},\dots, \mathbf{q}_{n}$ starting from a set of linearly independent vectors $\mathbf{x}_{1},\mathbf{x}_{2},\dots,\mathbf{x}_{n}$ 
by subtracting the projection of each vector onto the previous vectors, i.e.,
$$
\begin{equation*}
\mathbf{q}_{k}=\mathbf{x}_{k}-\sum_{i=1}^{k-1}c_{k,i}\cdot\mathbf{q}_{i},
\qquad{k=1,\dots,n}
\end{equation*}
$$
where the coefficients $c_{k,1},c_{k,2},\dots,c_{k,k-1}$ are chosen to make the vectors $\mathbf{q}_{1},\mathbf{q}_{2},\dots,\mathbf{q}_{k}$ orthogonal.
The $c_{\star}$ coefficients represent the component of the vector $\mathbf{x}_{k}$ that lies in the direction of the vectors $\mathbf{q}_{1},\mathbf{q}_{2},\dots,\mathbf{q}_{k-1}$. 

Classical Gram-Schmidt can sometimes produce _almost_ orthogonal vectors because of roundoff error, which led to the Modified Gram-Schmidt algorithm. 

> __When to use Classical vs. Modified Gram-Schmidt__:  
> * __Classical Gram-Schmidt__: Simpler to understand conceptually and slightly faster computationally, but can suffer from numerical instability in finite-precision arithmetic. Use when: working with well-conditioned matrices; column vectors are far from being linearly dependent; exact arithmetic is available (symbolic computation).  
> * __Modified Gram-Schmidt__: More numerically stable in practice, producing more orthogonal vectors when working with finite-precision arithmetic. Use when: numerical stability is critical; working with ill-conditioned matrices; column vectors are nearly linearly dependent; this is the __recommended approach__ for practical implementations.

Let's explore the pseudo-code implementations for both classical and modified Gram-Schmidt algorithms.

___

## Classical Gram-Schmidt Algorithm

__Initialization__: Given a matrix $\mathbf{A} \in \mathbb{R}^{n\times m}$ with linearly independent columns $\mathbf{a}_{1}, \mathbf{a}_{2}, \ldots, \mathbf{a}_{m}$, we initialize an orthogonal matrix $\mathbf{Q} \in \mathbb{R}^{n\times m}$ and an upper triangular matrix $\mathbf{R} \in \mathbb{R}^{m\times m}$ as zero matrices of the appropriate sizes. Let $n, m$ denote the dimensions of $\mathbf{A}$, and set a tolerance $\epsilon > 0$ for detecting linear dependence (typically $\epsilon = 10^{-10}$).

> __Parameter selection rules of thumb__:  
> * __Tolerance__ $\epsilon$: Used to detect near-linear dependence in the columns. A good starting point is $\epsilon = 10^{-10}$ for double-precision arithmetic. If $r_{j,j} < \epsilon$, the columns are likely linear dependent.  
> * __Matrix dimensions__: The algorithm requires $n \geq m$ (at least as many rows as columns). If $m > n$, the columns cannot all be linearly independent.  
> * __Numerical stability__: Classical Gram-Schmidt can lose orthogonality when: the condition number of $\mathbf{A}$ is large ($\kappa(\mathbf{A}) > 10^{8}$ for double precision); columns are nearly parallel (inner products close to 1); in these cases, prefer Modified Gram-Schmidt or use reorthogonalization

For column $j = 1, 2, \ldots, m$ __do__:
- Get the $j$-th column of $\mathbf{A}$: $\mathbf{v}_{j} \gets \mathbf{a}_{j}$.
- For each previous column $k = 1$ to $j-1$ (skip if $j = 1$) __do__:
  - Compute the projection coefficient: $r_{k,j} \gets \left\langle \mathbf{v}_{j}, \mathbf{q}_{k} \right\rangle = \mathbf{q}_{k}^{\top}\mathbf{v}_{j}$.
  - Remove the projection of $\mathbf{q}_{k}$ from $\mathbf{v}_{j}$: $\mathbf{v}_{j} \gets \mathbf{v}_{j} - r_{k,j} \mathbf{q}_{k}$.
  - Store the coefficient: $\mathbf{R}[k,j] \gets r_{k,j}$.
- Compute the norm of the residual vector: $r_{j,j} \gets \lVert \mathbf{v}_{j} \rVert_{2}$.
- __Check for linear dependence__: 
  * If $r_{j,j} < \epsilon$, then the columns are linearly dependent; terminate with error or warning.
  * Normalize the $j$-th orthogonal vector: $\mathbf{q}_{j} \gets \mathbf{v}_{j} / r_{j,j}$. Store the normalized vector as the $j$-th column of $\mathbf{Q}$ and store the diagonal element: $\mathbf{R}[j,j] \gets r_{j,j}$.

__Output__: Return the orthogonal matrix $\mathbf{Q}$ and the upper triangular matrix $\mathbf{R}$ such that $\mathbf{A} = \mathbf{Q}\mathbf{R}$.

## Modified Gram-Schmidt Algorithm

Classical Gram-Schmidt computes all projection coefficients from the original columns $\mathbf{a}_{j}$ before updating. Modified Gram-Schmidt immediately updates all remaining columns after orthogonalizing each vector, computing subsequent projections from these updated columns. This immediate update strategy reduces the accumulation of roundoff errors and produces more orthogonal vectors in finite-precision arithmetic.

__Initialization__: Given a matrix $\mathbf{A} \in \mathbb{R}^{n\times m}$ with linearly independent columns, we initialize $\mathbf{Q} \in \mathbb{R}^{n\times m}$ and $\mathbf{R} \in \mathbb{R}^{m\times m}$ as zero matrices. Let $n, m$ denote the dimensions of $\mathbf{A}$. Create a working copy $\tilde{\mathbf{A}} \gets \mathbf{A}$ and set a tolerance $\epsilon > 0$ for detecting linear dependence (typically $\epsilon = 10^{-10}$).

For column $j = 1, 2, \ldots, m$ __do__:
- Get the $j$-th working column: $\tilde{\mathbf{a}}_{j} \gets \tilde{\mathbf{a}}_{j}$.
- Compute the norm of the $j$-th working column: $r_{j,j} \gets \lVert \tilde{\mathbf{a}}_{j} \rVert_{2}$.
- __Check for linear dependence__: 
  * If $r_{j,j} < \epsilon$, then the columns are linearly dependent; terminate with error or warning.
  * Normalize to obtain the $j$-th orthogonal vector: $\mathbf{q}_{j} \gets \tilde{\mathbf{a}}_{j} / r_{j,j}$. Store the normalized vector as the $j$-th column of $\mathbf{Q}$ and store the diagonal element: $\mathbf{R}[j,j] \gets r_{j,j}$.
- For each remaining column $k = j+1$ to $m$ __do__:
  - Compute the projection coefficient from the working matrix: $r_{j,k} \gets \left\langle \tilde{\mathbf{a}}_{k}, \mathbf{q}_{j} \right\rangle = \mathbf{q}_{j}^{\top}\tilde{\mathbf{a}}_{k}$.
  - Immediately update the $k$-th working column by removing the projection: $\tilde{\mathbf{a}}_{k} \gets \tilde{\mathbf{a}}_{k} - r_{j,k} \mathbf{q}_{j}$.
  - Store the coefficient: $\mathbf{R}[j,k] \gets r_{j,k}$.

__Output__: Return the orthogonal matrix $\mathbf{Q}$ and the upper triangular matrix $\mathbf{R}$ such that $\mathbf{A} = \mathbf{Q}\mathbf{R}$.

__Additional references__:
* [Prof. Tom Trogdon: UCI MATH 105A: Numerical Analysis (2016), Lecture 21: Orthogonal Matricies](https://faculty.washington.edu/trogdon/105A/html/Lecture21.html)
* [Prof. Tom Trogdon: UCI MATH 105A: Numerical Analysis (2016), Lecture 23: The modified Gram-Schmidt procedure](https://faculty.washington.edu/trogdon/105A/html/Lecture23.html)

___
