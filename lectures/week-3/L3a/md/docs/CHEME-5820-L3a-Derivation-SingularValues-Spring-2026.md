# Derivation: Relationship between the Singular Values and the Eigenvalues
One natural question arises when studying matrix decompositions: how are the singular values of a matrix related to its eigenvalues? Let's work through this connection carefully. Suppose we have a matrix $\mathbf{A}\in\mathbb{R}^{m\times n}$ (meaning $\mathbf{A}$ is a real-valued matrix with $m$ rows and $n$ columns) that has the singular value decomposition (SVD):
$$
\begin{align*}
\mathbf{A} &=\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}\\
\end{align*}
$$
where:
* $\mathbf{U}\in\mathbb{R}^{m\times m}$ is an orthogonal matrix (satisfies $\mathbf{U}^\top\mathbf{U}=\mathbf{I}_m$), whose columns are the **left singular vectors**
* $\mathbf{V}\in\mathbb{R}^{n\times n}$ is an orthogonal matrix (satisfies $\mathbf{V}^\top\mathbf{V}=\mathbf{I}_n$), whose columns are the **right singular vectors**
* $\mathbf{\Sigma}\in\mathbb{R}^{m\times n}$ is a rectangular diagonal matrix with nonnegative entries $\sigma_1,\dots,\sigma_p$ on its main diagonal, where $p=\min(m,n)$. These diagonal entries are called the **singular values** of $\mathbf{A}$.

Here's the connection we'll prove: the singular values $\sigma_i$ are directly related to the eigenvalues $\lambda_i$ of two special matrices formed from $\mathbf{A}$:
* The right singular vectors (columns of $\mathbf{V}$) are exactly the eigenvectors of $\mathbf{A}^\top \mathbf{A}$
* The left singular vectors (columns of $\mathbf{U}$) are exactly the eigenvectors of $\mathbf{A}\mathbf{A}^\top$
* The eigenvalues of both matrices satisfy $\lambda_i = \sigma_i^2$ (the singular values are the square roots of these eigenvalues)

Where does this relationship come from? Let's derive it step by step.

___

## Right singular vectors and eigenvalues of $\mathbf{A}^\top \mathbf{A}$
Let's start by examining a special matrix formed from $\mathbf{A}$. Consider the product $\mathbf{A}^\top\mathbf{A}\in\mathbb{R}^{n\times n}$ (this is called the **Gram matrix** and is always square, even when $\mathbf{A}$ is not). This matrix has two important properties:

**First, it's always symmetric** (meaning $\mathbf{A}^\top\mathbf{A} = (\mathbf{A}^\top\mathbf{A})^\top$). We can verify this:
$$
(\mathbf{A}^\top\mathbf{A})^\top = \mathbf{A}^\top(\mathbf{A}^\top)^\top = \mathbf{A}^\top\mathbf{A}
$$

**Second, it's positive semidefinite** (meaning all its eigenvalues are nonnegative). To see why, take any vector $\mathbf{x}\in\mathbb{R}^n$ and compute the quadratic form:
$$
\mathbf{x}^\top(\mathbf{A}^\top\mathbf{A})\mathbf{x} = (\mathbf{A}\mathbf{x})^\top(\mathbf{A}\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|^2 \geq 0
$$
Since this is always nonnegative (it's a squared norm!), we know $\mathbf{A}^\top\mathbf{A}$ is positive semidefinite. Because $\mathbf{A}^\top\mathbf{A}$ is symmetric and positive semidefinite, it has an eigendecomposition with real, nonnegative eigenvalues. Let's write this as:
$$
\begin{align*}
\mathbf{A}^\top\mathbf{A} &= \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\top}\\
\end{align*}
$$
where $\mathbf{Q}\in\mathbb{R}^{n\times n}$ is an orthogonal matrix whose columns are the eigenvectors, and $\mathbf{\Lambda}\in\mathbb{R}^{n\times n}$ is a diagonal matrix containing the eigenvalues $\lambda_1,\dots,\lambda_n$ (all $\lambda_i \geq 0$).

Our goal is to show that $\mathbf{Q} = \mathbf{V}$ (meaning the eigenvectors of $\mathbf{A}^\top\mathbf{A}$ are exactly the right singular vectors) and to connect the eigenvalues in $\mathbf{\Lambda}$ to the singular values in $\mathbf{\Sigma}$. Let's see how this works.

Now here's the key insight: let's compute $\mathbf{A}^\top\mathbf{A}$ using the SVD of $\mathbf{A}$ and see what we get:
$$
\begin{align*}
\mathbf{A}^\top\mathbf{A} &= (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top})^\top(\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top})\quad\text{(substitute the SVD of $\mathbf{A}$)}\\
&= \left(\mathbf{V}\mathbf{\Sigma}^{\top}\mathbf{U}^{\top}\right)\left(\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}\right)\quad\text{(apply the transpose: $(\mathbf{ABC})^\top = \mathbf{C}^\top\mathbf{B}^\top\mathbf{A}^\top$)}\\
&= \mathbf{V}\mathbf{\Sigma}^{\top}\underbrace{\left(\mathbf{U}^\top\mathbf{U}\right)}_{\mathbf{I}_{m}}\mathbf{\Sigma}\mathbf{V}^{\top}\quad\text{(since $\mathbf{U}$ is orthogonal, $\mathbf{U}^\top\mathbf{U} = \mathbf{I}_m$)}\\
&= \mathbf{V}\left(\mathbf{\Sigma}^{\top}\mathbf{\Sigma}\right)\mathbf{V}^{\top}\quad\text{(the identity matrices cancel out)}\\
\end{align*}
$$

What is the matrix $\mathbf{\Sigma}^{\top}\mathbf{\Sigma}$? Since $\mathbf{\Sigma}$ is rectangular diagonal (with singular values on the diagonal), the product $\mathbf{\Sigma}^{\top}\mathbf{\Sigma}\in\mathbb{R}^{n\times n}$ is a square diagonal matrix whose entries are the **squares** of the singular values. Specifically:
$$
\begin{align*}
\mathbf{\Sigma}^{\top}\mathbf{\Sigma} = \begin{bmatrix}
\sigma_1^2 & 0 & \cdots & 0\\
0 & \sigma_2^2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & 0
\end{bmatrix}
\end{align*}
$$
where we have $\sigma_1^2,\dots,\sigma_p^2$ on the first $p$ diagonal entries (recall $p=\min(m,n)$), followed by $n-p$ zeros. We can write this compactly as $\mathbf{\Sigma}^{\top}\mathbf{\Sigma} = \mathrm{diag}(\sigma_1^2,\dots,\sigma_p^2,0,\dots,0)$.

Now we have two different expressions for the same matrix $\mathbf{A}^\top\mathbf{A}$:
$$
\begin{align*}
\mathbf{A}^\top\mathbf{A} &= \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\top}\quad\text{(eigendecomposition with eigenvectors $\mathbf{Q}$, eigenvalues $\mathbf{\Lambda}$)}\\
\mathbf{A}^\top\mathbf{A} &= \mathbf{V}\left(\mathbf{\Sigma}^{\top}\mathbf{\Sigma}\right)\mathbf{V}^{\top}\quad\text{(from the SVD of $\mathbf{A}$)}
\end{align*}
$$
Since $\mathbf{A}^\top\mathbf{A}$ is symmetric and positive semidefinite, its eigendecomposition is unique (up to the ordering of eigenvalues and the signs of eigenvectors). Since both expressions describe the eigendecomposition of the same matrix, we can match them up term by term:
$$
\boxed{
\begin{align*}
\mathbf{Q} &= \mathbf{V}\quad\text{(the eigenvectors match the right singular vectors!)}\\
\mathbf{\Lambda} &= \mathbf{\Sigma}^{\top}\mathbf{\Sigma} = \mathrm{diag}(\sigma_1^2,\dots,\sigma_p^2,0,\dots,0)\quad\text{(eigenvalues are squared singular values!)}\\
\end{align*}}
$$
This proves our first claim: the columns of $\mathbf{V}$ (the right singular vectors of $\mathbf{A}$) are exactly the eigenvectors of $\mathbf{A}^\top \mathbf{A}$, and the corresponding eigenvalues are $\lambda_i=\sigma_i^2$ for $i=1,\dots,p$, with any remaining $n-p$ eigenvalues equal to zero.

___

## Left singular vectors and eigenvalues of $\mathbf{A}\mathbf{A}^\top$
Now let's apply the same reasoning to another important matrix. Consider the product $\mathbf{A}\mathbf{A}^\top\in\mathbb{R}^{m\times m}$ (notice we've reversed the order compared to before—this matrix is square with dimension $m\times m$ instead of $n\times n$). Just like $\mathbf{A}^\top\mathbf{A}$, this matrix also has two key properties:

**It's always symmetric:**
$$
(\mathbf{A}\mathbf{A}^\top)^\top = (\mathbf{A}^\top)^\top\mathbf{A}^\top = \mathbf{A}\mathbf{A}^\top
$$

**It's positive semidefinite:** for any vector $\mathbf{y}\in\mathbb{R}^m$,
$$
\mathbf{y}^\top(\mathbf{A}\mathbf{A}^\top)\mathbf{y} = (\mathbf{A}^\top\mathbf{y})^\top(\mathbf{A}^\top\mathbf{y}) = \|\mathbf{A}^\top\mathbf{y}\|^2 \geq 0
$$

Because $\mathbf{A}\mathbf{A}^\top$ is symmetric and positive semidefinite, it also has an eigendecomposition with real, nonnegative eigenvalues:
$$
\begin{align*}
\mathbf{A}\mathbf{A}^\top &= \mathbf{P}\mathbf{\Omega}\mathbf{P}^{\top}\\
\end{align*}
$$
where $\mathbf{P}\in\mathbb{R}^{m\times m}$ is an orthogonal matrix of eigenvectors and $\mathbf{\Omega}\in\mathbb{R}^{m\times m}$ is a diagonal matrix of eigenvalues $\lambda_1,\dots,\lambda_m$ (all $\lambda_i \geq 0$). Our goal now is to show that $\mathbf{P} = \mathbf{U}$ (meaning the eigenvectors of $\mathbf{A}\mathbf{A}^\top$ are exactly the left singular vectors) and to connect the eigenvalues in $\mathbf{\Omega}$ to the singular values in $\mathbf{\Sigma}$. The derivation follows the same pattern as before. Let's compute $\mathbf{A}\mathbf{A}^\top$ using the SVD of $\mathbf{A}$, just as we did before:
$$
\begin{align*}
\mathbf{A}\mathbf{A}^\top &= (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top})(\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top})^\top\quad\text{(substitute the SVD of $\mathbf{A}$)}\\
&= \left(\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}\right)\left(\mathbf{V}\mathbf{\Sigma}^{\top}\mathbf{U}^{\top}\right)\quad\text{(apply the transpose)}\\
&= \mathbf{U}\mathbf{\Sigma}\underbrace{\left(\mathbf{V}^\top\mathbf{V}\right)}_{\mathbf{I}_{n}}\mathbf{\Sigma}^{\top}\mathbf{U}^{\top}\quad\text{(since $\mathbf{V}$ is orthogonal, $\mathbf{V}^\top\mathbf{V} = \mathbf{I}_n$)}\\
&= \mathbf{U}\left(\mathbf{\Sigma}\mathbf{\Sigma}^{\top}\right)\mathbf{U}^{\top}\quad\text{(the identity matrix cancels)}\\
\end{align*}
$$

What about the matrix $\mathbf{\Sigma}\mathbf{\Sigma}^{\top}$? Notice we've switched the order compared to before. This product gives us $\mathbf{\Sigma}\mathbf{\Sigma}^{\top}\in\mathbb{R}^{m\times m}$ (square with dimension $m\times m$), which is also diagonal with the **squares** of the singular values on its diagonal:
$$
\begin{align*}
\mathbf{\Sigma}\mathbf{\Sigma}^{\top} = \begin{bmatrix}
\sigma_1^2 & 0 & \cdots & 0\\
0 & \sigma_2^2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & 0
\end{bmatrix}
\end{align*}
$$
The first $p$ diagonal entries are $\sigma_1^2,\dots,\sigma_p^2$, followed by $m-p$ zeros. In compact notation: $\mathbf{\Sigma}\mathbf{\Sigma}^{\top} = \mathrm{diag}(\sigma_1^2,\dots,\sigma_p^2,0,\dots,0)$.

Once again, we have two different expressions for the same matrix $\mathbf{A}\mathbf{A}^\top$:
$$
\begin{align*}
\mathbf{A}\mathbf{A}^\top &= \mathbf{P}\mathbf{\Omega}\mathbf{P}^{\top}\quad\text{(eigendecomposition with eigenvectors $\mathbf{P}$, eigenvalues $\mathbf{\Omega}$)}\\
\mathbf{A}\mathbf{A}^\top &= \mathbf{U}\left(\mathbf{\Sigma}\mathbf{\Sigma}^{\top}\right)\mathbf{U}^{\top}\quad\text{(from the SVD of $\mathbf{A}$)}
\end{align*}
$$
Using the same uniqueness argument as before (the eigendecomposition of a symmetric positive semidefinite matrix is unique up to ordering and signs), we can match up the terms:
$$
\boxed{
\begin{align*}
\mathbf{P} &= \mathbf{U}\quad\text{(the eigenvectors match the left singular vectors!)}\\
\mathbf{\Omega} &= \mathbf{\Sigma}\mathbf{\Sigma}^{\top} = \mathrm{diag}(\sigma_1^2,\dots,\sigma_p^2,0,\dots,0)\quad\text{(eigenvalues are squared singular values!)}\\
\end{align*}}
$$
This proves our second claim: the columns of $\mathbf{U}$ (the left singular vectors of $\mathbf{A}$) are exactly the eigenvectors of $\mathbf{A}\mathbf{A}^\top$, and the corresponding eigenvalues are $\lambda_i=\sigma_i^2$ for $i=1,\dots,p$, with any remaining $m-p$ eigenvalues equal to zero.

___

## Summary
We have established the relationship between singular values and eigenvalues:

> **Key Takeaways:**
>
> * The eigenvalues of $\mathbf{A}^\top\mathbf{A}$ (and $\mathbf{A}\mathbf{A}^\top$) are $\lambda_i = \sigma_i^2$ for $i=1,\dots,p$, with remaining eigenvalues equal to zero
> * The right singular vectors (columns of $\mathbf{V}$) are the eigenvectors of $\mathbf{A}^\top\mathbf{A}$
> * The left singular vectors (columns of $\mathbf{U}$) are the eigenvectors of $\mathbf{A}\mathbf{A}^\top$
> * Both matrices have the same $p$ nonzero eigenvalues ($\sigma_1^2,\dots,\sigma_p^2$), but $\mathbf{A}^\top\mathbf{A}$ is $n\times n$ while $\mathbf{A}\mathbf{A}^\top$ is $m\times m$, thus they have different numbers of zero eigenvalues when $m\neq n$

**Practical Implications:**
While this connection suggests computing the SVD via eigendecomposition of $\mathbf{A}^\top\mathbf{A}$, this approach squares the condition number and amplifies numerical errors. Modern SVD algorithms work directly with $\mathbf{A}$ using more stable methods like bidiagonalization followed by QR iteration.

___


