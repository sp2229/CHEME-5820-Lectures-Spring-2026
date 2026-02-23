# L3a: Introduction to Singular Value Decomposition (SVD)
In this lecture, we'll introduce the concept of Singular Value Decomposition (SVD), a technique to factor rectangular matrices into three simpler matrices. SVD is widely used in various applications, including data compression, noise reduction, and robust solutions to linear systems.

> __Learning Objectives:__
> 
> By the end of this lecture, you should be able to:
> * __Understand SVD factorization and its components:__ Explain how any matrix can be decomposed into orthogonal matrices and a diagonal matrix of singular values, and describe the properties of each component.
> * __Connect SVD to eigendecomposition:__ Describe the relationship between singular values and eigenvalues, and explain how singular vectors relate to eigenvectors of specific matrices formed from the original matrix.
> * __Apply SVD for dimensionality reduction:__ Use the Eckart-Young theorem to construct optimal low-rank approximations and explain why truncated SVD provides the best rank-k approximation under Frobenius and spectral norms.


Let's get started!
___

## Examples
Today, we will use the following examples to illustrate key concepts:

> [▶ Let's use singular value decomposition on synthetic data](CHEME-5820-L3a-Example-SVD-SyntheticDataDecomposition-Spring-2026.ipynb). In this example, we'll use singular value decomposition (SVD) to analyze a synthetic dataset. We'll verify the properties of SVD and explore what the singular values and vectors reveal about the underlying structure of the data.

We'll also apply SVD to a real-world application:


> [▶ Let's use singular value decomposition on a stoichiometric matrix](CHEME-5820-L3a-Example-SVD-StoichiometricMatrix-Spring-2026.ipynb). In this example, we'll download and construct a stoichiometric matrix for a biochemical network. We'll then apply singular value decomposition (SVD) to analyze the matrix and interpret the results in the context of the biochemical system.
___

## Singular value decomposition (SVD)
One of the central matrix factorizations in linear algebra is the **singular value decomposition** (SVD). The SVD has numerous applications in data science, statistics, and machine learning, including dimensionality reduction, data compression, and noise reduction. Let's introduce the SVD formally.

> __Singular Value Decomposition (SVD)__
> 
> The singular value decomposition _factors any matrix_ into three components. Given a matrix $\mathbf{A}\in\mathbb{R}^{m\times n}$ of rank $r$, the **full** SVD is given by the factorization:
> $$
\boxed{
\mathbf{A} = \mathbf{U}\,\mathbf{S}\,\mathbf{V}^{\top}
}
$$
> where:
> * $\mathbf{U}\in\mathbb{R}^{m\times m}$ is orthogonal ($\mathbf{U}^\top \mathbf{U} = \mathbf{I}_m$),
> * $\mathbf{S}\in\mathbb{R}^{m\times n}$ is rectangular diagonal (with the singular values $\sigma_1\ge\cdots\ge\sigma_r>0$ on the first $r$ diagonal entries, zeros elsewhere). The singular values are the square roots of the non-negative eigenvalues of the positive semi-definite matrix $\mathbf{A}^{\top}\mathbf{A}$, i.e., $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$, 
where $\lambda_i(\mathbf{A}^\top\mathbf{A})\geq 0$ denote the eigenvalues of $\mathbf{A}^\top\mathbf{A}$. The number of non-zero singular values is [the rank](https://en.wikipedia.org/wiki/Rank_(linear_algebra)) of the matrix $\mathbf{A}$, where $r \leq\min\left(n,m\right)$.
> * $\mathbf{V}\in\mathbb{R}^{n\times n}$ is orthogonal ($\mathbf{V}^\top \mathbf{V} = \mathbf{I}_n$).

__The full SVD__ stores **all** $m$ left‐singular vectors and all $n$ right‐singular vectors, even though only the first $r$ correspond to nonzero singular values.

* **Pros:** You have the complete orthonormal bases for both row‐ and column‐spaces.
* **Cons:** Memory requirement is $\mathcal{O}(m^2 + n^2)$, which can be wasteful if $r\ll\min(m,n)$. When you have a low rank matrix, you can save memory by storing only the first $r$ left‐ and right‐singular vectors, which is the idea underlying the **thin SVD**.

__Thin SVD__: The thin factorization is given by $\mathbf{U}_r\in\mathbb{R}^{m\times r},\mathbf{V}_r\in\mathbb{R}^{n\times r},\mathbf{S}_r\in\mathbb{R}^{r\times r}$, where $r$ is the rank of the matrix $\mathbf{A}$. Use the thin SVD whenever you only care about the nonzero singular values/directions (which is most of the time!).

### Connection to eigenvalue decomposition
One natural question arises when studying matrix decompositions: how are the singular values of a matrix related to its eigenvalues? Let's work through this connection. Suppose we have a (rectangular) matrix $\mathbf{A}\in\mathbb{R}^{m\times n}$ (meaning $\mathbf{A}$ is a real-valued matrix with $m$ rows and $n$ columns) that has the singular value decomposition (SVD):
$$
\begin{align*}
\mathbf{A} &=\mathbf{U}\mathbf{S}\mathbf{V}^{\top}\\
\end{align*}
$$
where:
* $\mathbf{U}\in\mathbb{R}^{m\times m}$ is an orthogonal matrix (satisfies $\mathbf{U}^\top\mathbf{U}=\mathbf{I}_m$), whose columns are the **left singular vectors**
* $\mathbf{V}\in\mathbb{R}^{n\times n}$ is an orthogonal matrix (satisfies $\mathbf{V}^\top\mathbf{V}=\mathbf{I}_n$), whose columns are the **right singular vectors**
* $\mathbf{S}\in\mathbb{R}^{m\times n}$ is a rectangular diagonal matrix with nonnegative entries $\sigma_1,\dots,\sigma_p$ on its main diagonal, where $p=\min(m,n)$. These diagonal entries are called the **singular values** of $\mathbf{A}$.

> There are three connections with the eigendecomposition of $\mathbb{R}^{m\times n}$: the singular values $\sigma_i$ are directly related to the eigenvalues $\lambda_i$, and the singular vectors are related to the eigenvectors of two special matrices formed from $\mathbf{A}$:
> * The right singular vectors (columns of $\mathbf{V}$) are exactly the eigenvectors of $\mathbf{A}^\top \mathbf{A}$
> * The left singular vectors (columns of $\mathbf{U}$) are exactly the eigenvectors of $\mathbf{A}\mathbf{A}^\top$
> * The eigenvalues of both matrices satisfy $\lambda_i = \sigma_i^2$ (the singular values are the square roots of these eigenvalues)
> 
> For a deeper understanding of this connection, check out this [derivation of the SVD](lectures/week-3/L3a/docs/CHEME-5820-L3a-Derivation-SingularValues-Spring-2026.ipynb)

Now that we understand the theoretical connection between SVD and eigendecomposition, let's explore how to actually compute the SVD in practice.

### How do we compute the SVD?
The SVD can be computed using various algorithms, however, a naive way to compute the SVD is to use the eigendecomposition of the matrices $\mathbf{A}^\top \mathbf{A}$ and $\mathbf{A}\mathbf{A}^\top$. A [deeper dive on this approach can be found here](docs/CHEME-5820-L3a-Algorithm-SVD-QR-Spring-2026.ipynb). Let's look at the algorithmic steps to compute the SVD using this approach.

__Initialization__: Given a real-valued matrix $\mathbf{A}\in\mathbb{R}^{m\times n}$, form $\mathbf{B}\gets\mathbf{A}^\top\mathbf{A}$ (called the Gram matrix, which captures the inner products between columns of $\mathbf{A}$) and specify a nonnegative tolerance $\tau$ for clipping tiny negative eigenvalues that may arise from numerical roundoff (e.g., $\tau\approx 0$ or a small multiple of machine precision).

> __Numerical note__: In floating-point arithmetic, $\mathbf{B}$ can have small negative eigenvalues due to roundoff error, even though mathematically all eigenvalues should be nonnegative. We use $\max(\lambda_i,0)$ (or $\max(\lambda_i,\tau)$) before taking square roots to avoid numerical issues.

* Compute the eigendecomposition $\mathbf{B} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top$ (e.g., using QR iteration for symmetric matrices).
* Sort the eigenvalues $\lambda_i$ in descending order (largest to smallest) and reorder the columns of $\mathbf{V}$ to match.
* Define the singular values as $\sigma_i \gets \sqrt{\max(\lambda_i,0)}$ and construct the rectangular diagonal matrix $\mathbf{S}$.

For each $\sigma_i>0$ __do__:
- Compute $\mathbf{u}_i \gets \mathbf{A}\mathbf{v}_i/\sigma_i$ (apply $\mathbf{A}$ to the eigenvector and normalize).
- Assemble the left singular vectors into $\mathbf{U}_r=[\mathbf{u}_1,\dots,\mathbf{u}_r]$.

If a full $\mathbf{U}$ is required, extend $\mathbf{U}_r$ with an orthonormal basis for the left nullspace of $\mathbf{A}$ (the set of vectors $\mathbf{w}$ such that $\mathbf{A}^\top\mathbf{w}=\mathbf{0}$).

__Output__: By default, the **reduced (thin) SVD** factorization $\mathbf{A}=\mathbf{U}_r\mathbf{S}_r\mathbf{V}_r^\top$, where $\mathbf{U}_r\in\mathbb{R}^{m\times r}$, $\mathbf{S}_r\in\mathbb{R}^{r\times r}$, and $\mathbf{V}_r\in\mathbb{R}^{n\times r}$ contain only the $r$ nonzero singular values and their corresponding vectors. If the optional extension step is performed, returns the **full SVD** $\mathbf{A}=\mathbf{U}\mathbf{S}\mathbf{V}^\top$ with square $\mathbf{U}\in\mathbb{R}^{m\times m}$ and $\mathbf{V}\in\mathbb{R}^{n\times n}$.

Ok, great! Is this actually how it works in practice? Nope! There are some important practical issues with this approach:

> **Important Practical Observations:**
>
> - **Condition number squared**: Forming $\mathbf{A}^\top\mathbf{A}$ squares the condition number: $\kappa(\mathbf{A}^\top\mathbf{A})=\kappa(\mathbf{A})^2$. This means that small singular values of $\mathbf{A}$ become extremely small eigenvalues of $\mathbf{A}^\top\mathbf{A}$, making them susceptible to roundoff errors.
> - **Negative eigenvalues from roundoff**: Roundoff can produce small negative eigenvalues that must be clipped to zero before taking square roots.
> - **Computational expense**: Forming $\mathbf{A}^\top\mathbf{A}$ requires $O(mn^2)$ operations, which can be expensive for large matrices.
> - **Sparsity destruction**: If $\mathbf{A}$ is sparse (has many zeros), $\mathbf{A}^\top\mathbf{A}$ is typically much denser. For example, if $\mathbf{A}$ has only a few non-zero entries per row, the product $\mathbf{A}^\top\mathbf{A}$ creates interactions between all pairs of columns that share any non-zero entry in the same row. This "fill-in" effect means we lose the computational advantages of sparse matrix storage and operations.
> - **Multiple data passes**: Computing $\mathbf{A}^\top\mathbf{A}$ requires accessing the data in $\mathbf{A}$ multiple times. To compute the $(i,j)$ entry of $\mathbf{A}^\top\mathbf{A}$, we need the inner product of column $i$ and column $j$ of $\mathbf{A}$. Computing all entries means repeatedly accessing columns, which can be inefficient for large datasets stored on disk or distributed across multiple machines.
> 
> For all of these reasons, practical SVD algorithms avoid explicitly forming $\mathbf{A}^\top\mathbf{A}$. Instead, they use more sophisticated techniques like bidiagonalization followed by QR iteration, which are numerically more stable and efficient. For further details, see [Golub and Van Loan, Matrix Computations](https://jhupbooks.press.jhu.edu/title/matrix-computations), 4th edition, Chapter 8.

Let's do an example:


> [▶ Let's use singular value decomposition on synthetic data](CHEME-5820-L3a-Example-SVD-SyntheticDataDecomposition-Spring-2026.ipynb). In this example, we'll use singular value decomposition (SVD) to analyze a synthetic dataset. We'll verify the properties of SVD and explore explore what the singular values and vectors reveal about the underlying structure of the data.

___

## Singular Value Decomposition as a Dimensionality Reduction Tool
The SVD expresses a matrix as a sum of separable rank-1 blocks, revealing the matrix rank through nonzero singular values and enabling optimal low-rank approximations. 

Let $\mathbf{A}\in\mathbb{R}^{m\times{n}}$ have the singular value decomposition $\mathbf{A} = \mathbf{U}\mathbf{S}\mathbf{V}^{\top}$. Then, the matrix $\mathbf{A}$ can be written as:
$$
\mathbf{A} = \sum_{i=1}^{r}\sigma_{i}\,\underbrace{\left(\mathbf{u}_{i}\otimes\mathbf{v}_{i}\right)}_{\mathbf{u}_{i}\mathbf{v}_{i}^{\top}}
$$
where $r$ is the rank of matrix $\mathbf{A}$, the vectors $\mathbf{u}_{i}$ and $\mathbf{v}_{i}$ are the $i$-th left and right singular vectors, and $\sigma_{i}$ are the (ordered) singular values. The [outer-product](https://en.wikipedia.org/wiki/Outer_product) $\mathbf{u}_{i}\mathbf{v}_{i}^{\top}$ is a separable rank-1 block of the matrix $\mathbf{A}$. 

> __Connection to PCA__: 
> 
> The right singular vectors $\mathbf{v}_i$ are the transformation vectors $\mathbf{\phi}_i$ from the principal component analysis (PCA) dimensionality reduction discussion. When $\mathbf{A}$ is the centered data matrix, the first $k$ columns of $\mathbf{V}$ give you the principal component directions!

This connection to PCA explains _which_ vectors to use. But _how many_ should we keep? The Eckart-Young theorem provides the answer:

> __Eckart–Young theorem__
>
> By truncating the sum at $k\ll{r}$, with singular values $\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0$, the SVD yields the best rank-$k$ approximation in both the Frobenius and spectral norms (Eckart–Young theorem):
>$$
\mathbf{A}_k = \sum_{i=1}^k \sigma_i\,\mathbf{u}_i\,\mathbf{v}_i^\top.
$$
> For any other matrix $\mathbf{B}$ of rank at most $k$,
> $$
\|\mathbf{A} - \mathbf{A}_k\|_F \;\le\;\|\mathbf{A} - \mathbf{B}\|_F,\quad\|\mathbf{A} - \mathbf{A}_k\|_2 \;\le\;\|\mathbf{A} - \mathbf{B}\|_2,
$$
> where $\|\cdot\|_F$ is the Frobenius norm and $\|\cdot\|_2$ the operator (spectral) norm.  In other words, $\mathbf{A}_k$ is the best rank-$k$ approximation to $\mathbf{A}$ under both measures of error. 

Let's take a look at an example:

> [▶ Let's use singular value decomposition on a stoichiometric matrix](CHEME-5820-L3a-Example-SVD-StoichiometricMatrix-Spring-2026.ipynb). In this example, we'll download and construct a stoichiometric matrix for a biochemical network. We'll then apply singular value decomposition (SVD) to analyze the matrix and interpret the results in the context of the biochemical system.
___

## Summary
Singular value decomposition factors any matrix into orthogonal matrices and a diagonal matrix of singular values, providing a fundamental tool for understanding matrix structure and enabling optimal low-rank approximations.

> __Key Takeaways:__
> 
> * **SVD factorization applies universally:** Any matrix can be decomposed as $\mathbf{A} = \mathbf{U}\mathbf{S}\mathbf{V}^{\top}$ where $\mathbf{U}$ and $\mathbf{V}$ are orthogonal matrices and $\mathbf{S}$ contains singular values on its diagonal.
> * **SVD connects to eigendecomposition through specific relationships:** The singular values are square roots of eigenvalues of $\mathbf{A}^\top\mathbf{A}$, while the singular vectors are eigenvectors of $\mathbf{A}^\top\mathbf{A}$ and $\mathbf{A}\mathbf{A}^\top$.
> * **The Eckart-Young theorem guarantees optimality:** Truncated SVD produces the best rank-k approximation under both Frobenius and spectral norms, making it the theoretical foundation for dimensionality reduction.


Understanding SVD provides the mathematical foundation for numerous applications in data science, including dimensionality reduction, data compression, and noise reduction in real-world datasets.

___
