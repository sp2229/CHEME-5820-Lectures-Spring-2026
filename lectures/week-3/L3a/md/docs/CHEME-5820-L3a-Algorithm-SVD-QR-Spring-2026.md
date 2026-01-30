# Computing the Singular Value Decomposition (SVD) of a matrix using QR iteration
We saw how to compute the eigendecomposition of a matrix using the QR iteration method. We can use a similar approach to compute the Singular Value Decomposition (SVD) of a matrix. Let's dig into these details.

The singular value decomposition _factors any matrix_ into three components. 
Given a matrix $\mathbf{A}\in\mathbb{R}^{m\times n}$ of rank $r$, the **full** SVD is given by the factorization:
$$
\boxed{
\mathbf{A} = \mathbf{U}\,\mathbf{S}\,\mathbf{V}^\top
}
$$

where:
* $\mathbf{U}\in\mathbb{R}^{m\times m}$ is orthogonal ($\mathbf{U}^\top \mathbf{U} = \mathbf{I}_m$),
* $\mathbf{S}\in\mathbb{R}^{m\times n}$ is rectangular diagonal (with the singular values $\sigma_1\ge\cdots\ge\sigma_r>0$ on the first $r$ diagonal entries, zeros elsewhere). The singular values are the square roots of the non-negative eigenvalues of the positive semi-definite matrix $\mathbf{A}^{\top}\mathbf{A}$, i.e., $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$, 
where $\lambda_i(\mathbf{A}^\top\mathbf{A})\geq 0$ denote the eigenvalues of $\mathbf{A}^\top\mathbf{A}$. The number of non-zero singular values is [the rank](https://en.wikipedia.org/wiki/Rank_(linear_algebra)) of the matrix $\mathbf{A}$, where $r \leq\min\left(n,m\right)$.
* $\mathbf{V}\in\mathbb{R}^{n\times n}$ is orthogonal ($\mathbf{V}^\top \mathbf{V} = \mathbf{I}_n$).

So what is the connection with the eigendecomposition? 

> __Pedagogical note:__ 
> 
> * In this material, we'll explore a **naive approach** to computing the SVD through eigendecomposition of $\mathbf{A}^\top\mathbf{A}$. This approach is conceptually straightforward and helps build foundational understanding of the relationship between eigenvalues and singular values. 
> * However, this method has significant numerical and computational issues that make it impractical for real-world applications. Modern SVD implementations use more sophisticated algorithms (like bidiagonalization followed by QR iteration) that are specifically designed to address these limitations while maintaining numerical stability and efficiency.

Let's see how the naive approach works and understand why practical implementations take a different path.
___

## Naive approach: Eigendecomposition of $\mathbf{A}^\top \mathbf{A}$
The most straightforward way to compute the SVD of a matrix $\mathbf{A}$ is to compute the eigendecomposition of the matrix $\mathbf{A}^\top \mathbf{A}$. For any real matrix $\mathbf{A}\in\mathbb{R}^{m\times n}$, the matrix $\mathbf{A}^\top\mathbf{A}$ is always symmetric and positive semidefinite:

> __Why is $\mathbf{A}^\top\mathbf{A}$ symmetric and positive semidefinite?__
>
> **Symmetry**: For any real matrix $\mathbf{A}$, we have $(\mathbf{A}^\top\mathbf{A})^\top = \mathbf{A}^\top(\mathbf{A}^\top)^\top = \mathbf{A}^\top\mathbf{A}$, so $\mathbf{A}^\top\mathbf{A}$ is symmetric.
>
> **Positive semidefinite**: For any vector $\mathbf{x}\in\mathbb{R}^{n}$, we have $\mathbf{x}^\top(\mathbf{A}^\top\mathbf{A})\mathbf{x} = (\mathbf{A}\mathbf{x})^\top(\mathbf{A}\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|_2^2 \geq 0$. Since the quadratic form is always nonnegative, $\mathbf{A}^\top\mathbf{A}$ is positive semidefinite.
>
> These properties hold for any real matrix $\mathbf{A}$, regardless of its dimensions or rank.

Because $\mathbf{A}^\top\mathbf{A}$ is symmetric and positive semidefinite, it has an orthogonal eigendecomposition:
$$
\mathbf{A}^\top\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top
$$
where $\mathbf{\Lambda}=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$ with $\lambda_i\geq 0$ denotes the diagonal matrix of eigenvalues, and $\mathbf{V}=[\mathbf{v}_1,\dots,\mathbf{v}_n]$ denotes the matrix whose columns are the corresponding eigenvectors. Now, here's the key idea: we can construct the SVD of $\mathbf{A}$ from this eigendecomposition. For any eigenpair $(\lambda_i,\mathbf{v}_i)$ with $\lambda_i>0$, we define:
$$
\sigma_i = \sqrt{\lambda_i},\qquad \mathbf{u}_i = \frac{1}{\sigma_i}\mathbf{A}\mathbf{v}_i
$$
where $\sigma_i$ is a singular value of $\mathbf{A}$ (the square root of the eigenvalue), and $\mathbf{u}_i$ is the corresponding left singular vector (obtained by applying $\mathbf{A}$ to the eigenvector $\mathbf{v}_i$ and normalizing by $\sigma_i$). With these definitions, we can verify that:
$$
\mathbf{A}\mathbf{v}_i = \sigma_i \mathbf{u}_i
$$
which is exactly the relationship we need for the SVD. Furthermore, we can verify that the vectors $\mathbf{u}_i$ are orthonormal. Let's check this carefully by computing the inner product $\mathbf{u}_i^\top\mathbf{u}_j$:
$$
\begin{align*}
\mathbf{u}_i^\top\mathbf{u}_j &= \left(\frac{1}{\sigma_i}\mathbf{A}\mathbf{v}_i\right)^\top\left(\frac{1}{\sigma_j}\mathbf{A}\mathbf{v}_j\right)\quad\text{(substitute definition of $\mathbf{u}_i$ and $\mathbf{u}_j$)}\\
&= \frac{1}{\sigma_i\sigma_j}\left(\mathbf{A}\mathbf{v}_i\right)^\top\left(\mathbf{A}\mathbf{v}_j\right)\quad\text{(pull out scalars)}\\
&= \frac{1}{\sigma_i\sigma_j}\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_j\quad\text{(transpose property: $(\mathbf{A}\mathbf{v}_i)^\top = \mathbf{v}_i^\top\mathbf{A}^\top$)}\\
&= \frac{1}{\sigma_i\sigma_j}\mathbf{v}_i^\top\left(\lambda_j\mathbf{v}_j\right)\quad\text{(use eigenvalue property: $\mathbf{A}^\top\mathbf{A}\mathbf{v}_j = \lambda_j\mathbf{v}_j$)}\\
&= \frac{\lambda_j}{\sigma_i\sigma_j}\mathbf{v}_i^\top\mathbf{v}_j\quad\text{(pull out eigenvalue)}\\
&= \frac{\lambda_j}{\sigma_i\sigma_j}\delta_{ij}\quad\text{(eigenvectors are orthonormal: $\mathbf{v}_i^\top\mathbf{v}_j = \delta_{ij}$)}\\
&= \frac{\sigma_j^2}{\sigma_i\sigma_j}\delta_{ij}\quad\text{(since $\lambda_j = \sigma_j^2$)}\\
&= \left(\frac{\sigma_j}{\sigma_i}\right)\delta_{ij}\quad\text{(when $i=j$: $\sigma_i/\sigma_i=1$; when $i\neq j$: $\delta_{ij}=0$)}\quad\blacksquare\\
\end{align*}
$$
where $\delta_{ij}$ is the Kronecker delta (equals 1 if $i=j$, and 0 otherwise). This confirms that the $\mathbf{u}_i$ vectors form an orthonormal set.

> __What about the zero eigenvalues?__ 
> 
> If $\lambda_i=0$, then $\mathbf{A}\mathbf{v}_i=\mathbf{0}$, which means $\mathbf{v}_i$ lies in the nullspace of $\mathbf{A}$ (the set of vectors that $\mathbf{A}$ maps to zero). These eigenvectors don't contribute to the reconstruction of $\mathbf{A}$. Thus, we ignore them when forming the SVD.

Collecting just the nonzero singular values and their corresponding vectors gives us the reduced (thin) SVD:
$$
\mathbf{A} = \mathbf{U}_r \mathbf{S}_r \mathbf{V}_r^\top,
$$
where $\mathbf{S}_r=\mathrm{diag}(\sigma_1,\dots,\sigma_r)$ is a diagonal matrix containing the $r$ nonzero singular values. If we need the full SVD (with square matrices $\mathbf{U}$ and $\mathbf{V}$), we can extend $\mathbf{U}_r$ by adding an orthonormal basis for the left nullspace of $\mathbf{A}$.

Let's look at the algorithmic steps to compute the SVD using this approach.

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

__Output__: The SVD factorization $\mathbf{A}=\mathbf{U}\mathbf{S}\mathbf{V}^\top$.

Ok, great! Is this actually how it works in practice? Nope! There are some important practical issues with this approach:

> **Important Practical Observations:**
>
> - **Condition number squared**: Forming $\mathbf{A}^\top\mathbf{A}$ squares the condition number: $\kappa(\mathbf{A}^\top\mathbf{A})=\kappa(\mathbf{A})^2$. This means that small singular values of $\mathbf{A}$ become extremely small eigenvalues of $\mathbf{A}^\top\mathbf{A}$, making them susceptible to roundoff errors.
> - **Negative eigenvalues from roundoff**: Roundoff can produce small negative eigenvalues that must be clipped to zero before taking square roots.
> - **Computational expense**: Forming $\mathbf{A}^\top\mathbf{A}$ requires $O(mn^2)$ operations, which can be expensive for large matrices.
> - **Sparsity destruction**: If $\mathbf{A}$ is sparse (has many zeros), $\mathbf{A}^\top\mathbf{A}$ is typically much denser. For example, if $\mathbf{A}$ has only a few non-zero entries per row, the product $\mathbf{A}^\top\mathbf{A}$ creates interactions between all pairs of columns that share any non-zero entry in the same row. This "fill-in" effect means we lose the computational advantages of sparse matrix storage and operations.
> - **Multiple data passes**: Computing $\mathbf{A}^\top\mathbf{A}$ requires accessing the data in $\mathbf{A}$ multiple times. To compute the $(i,j)$ entry of $\mathbf{A}^\top\mathbf{A}$, we need the inner product of column $i$ and column $j$ of $\mathbf{A}$. Computing all entries means repeatedly accessing columns, which can be inefficient for large datasets stored on disk or distributed across multiple machines.
> 
> For all of these reasons, practical SVD algorithms avoid explicitly forming $\mathbf{A}^\top\mathbf{A}$. Instead, they use more sophisticated techniques like bidiagonalization followed by QR iteration, which are numerically more stable and efficient.
___

## Summary
We have established the relationship between the eigendecomposition of $\mathbf{A}^\top\mathbf{A}$ and the SVD of $\mathbf{A}$:

> __Key Results:__
>
> * The eigenvalues of $\mathbf{A}^\top\mathbf{A}$ satisfy $\lambda_i=\sigma_i^2$ (singular values are square roots of eigenvalues)
> * The right singular vectors $\mathbf{V}$ are the eigenvectors of $\mathbf{A}^\top\mathbf{A}$
> * The left singular vectors are computed as $\mathbf{u}_i=\mathbf{A}\mathbf{v}_i/\sigma_i$ for each positive eigenvalue

**Practical Implications:**

While this eigendecomposition approach provides insight into the SVD structure, it squares the condition number and destroys sparsity, making it unsuitable for production use. Modern libraries (LAPACK, NumPy, Julia, MATLAB) use Golub-Kahan bidiagonalization followed by QR iteration, which works directly with $\mathbf{A}$ for better numerical stability. 

For further details, see Golub and Van Loan, [_Matrix Computations_](https://jhupbooks.press.jhu.edu/title/matrix-computations), 4th edition, Chapter 8.
___
