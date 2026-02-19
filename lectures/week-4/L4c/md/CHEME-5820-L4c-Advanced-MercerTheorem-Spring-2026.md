# Advanced: Mercer's Theorem and the Kernel Trick
In the lecture, we stated that every positive-definite kernel corresponds to an inner product in some feature space. This is Mercer's theorem. While you don't need the theorem to use kernels effectively, understanding why kernels work provides valuable insight into kernel methods.

> __What does Mercer's theorem tell us?__
>
> Mercer's theorem guarantees that any valid kernel $k(\mathbf{x}, \mathbf{z})$ can be expressed as an inner product in some (possibly infinite-dimensional) feature space. This means we can always interpret kernel evaluations as measuring similarity in a transformed space, even if we never construct the transformation explicitly.

We'll start with the finite-dimensional case, which is easier to understand, then explain how this generalizes to the continuous case covered by Mercer's original theorem.

___

## The Finite-Dimensional Case
Suppose we have a finite dataset $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$ and a symmetric function $k:\mathbb{R}^m \times \mathbb{R}^m \to \mathbb{R}$. We form the kernel matrix (Gram matrix):

$$\mathbf{K} = \begin{bmatrix} k(\mathbf{x}_1, \mathbf{x}_1) & k(\mathbf{x}_1, \mathbf{x}_2) & \cdots & k(\mathbf{x}_1, \mathbf{x}_n) \\ k(\mathbf{x}_2, \mathbf{x}_1) & k(\mathbf{x}_2, \mathbf{x}_2) & \cdots & k(\mathbf{x}_2, \mathbf{x}_n) \\ \vdots & \vdots & \ddots & \vdots \\ k(\mathbf{x}_n, \mathbf{x}_1) & k(\mathbf{x}_n, \mathbf{x}_2) & \cdots & k(\mathbf{x}_n, \mathbf{x}_n) \end{bmatrix}$$

> __Finite-dimensional result__:
>
> If $\mathbf{K}$ is symmetric and positive semi-definite (all eigenvalues $\lambda_i \geq 0$), then $k$ corresponds to an inner product in a feature space of dimension at most $n$.

### Proof sketch

Since $\mathbf{K}$ is symmetric and positive semi-definite, we can perform an eigendecomposition:

$$\mathbf{K} = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^{\top}$$

where $\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_n]$ contains orthonormal eigenvectors and $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ contains non-negative eigenvalues. We can write:

$$\mathbf{K} = \mathbf{U}\mathbf{\Lambda}^{1/2}\mathbf{\Lambda}^{1/2}\mathbf{U}^{\top} = \Phi\Phi^{\top}$$

where $\Phi = \mathbf{U}\mathbf{\Lambda}^{1/2} \in \mathbb{R}^{n \times n}$. The $i$-th row of $\Phi$ gives us the feature map $\phi(\mathbf{x}_i)$.

Now, the kernel between points $i$ and $j$ is:

$$k(\mathbf{x}_i, \mathbf{x}_j) = K_{ij} = [\Phi\Phi^{\top}]_{ij} = \phi(\mathbf{x}_i)^{\top}\phi(\mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$$

This shows that every entry in the kernel matrix is an inner product in the feature space defined by the rows of $\Phi$.

> __Why does this matter?__
>
> This construction proves that whenever we have a positive semi-definite kernel matrix, there exists a feature space where the kernel computes inner products. We never had to guess what $\phi$ should be—the eigendecomposition gave it to us automatically.

___

## Dimension of the Feature Space
The dimension of the feature space equals the number of non-zero eigenvalues of $\mathbf{K}$. If $r$ eigenvalues are positive and $n-r$ are zero, then the feature space is $r$-dimensional.

> __Example__: Linear kernel on finite data
>
> For the linear kernel $k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^{\top}\mathbf{x}_j$ with $\mathbf{X} \in \mathbb{R}^{n \times m}$, we have $\mathbf{K} = \mathbf{X}\mathbf{X}^{\top}$. The rank of $\mathbf{K}$ is at most $\min(n, m)$, so the effective feature space dimension is bounded by both the number of samples and the original feature dimension.

For nonlinear kernels like the polynomial kernel $k(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^{\top}\mathbf{z})^d$ or the RBF kernel, the implicit feature space can be much higher dimensional than the original space. For the RBF kernel $k(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$, the feature space is actually _infinite-dimensional_.

> __The infinite-dimensional case__:
>
> When the kernel's feature space is infinite-dimensional, we cannot perform a finite eigendecomposition. This is where Mercer's theorem in its full generality becomes essential. The theorem extends the finite-dimensional result to continuous functions on compact domains.

___

## Mercer's Theorem (Continuous Case)
The full version of Mercer's theorem applies to continuous kernels on compact domains.

> __Mercer's Theorem (1909)__:
>
> Let $k:\mathcal{X} \times \mathcal{X} \to \mathbb{R}$ be a continuous, symmetric function on a compact domain $\mathcal{X}$. If $k$ is positive semi-definite (i.e., $\int\int k(\mathbf{x}, \mathbf{z})f(\mathbf{x})f(\mathbf{z})d\mathbf{x}d\mathbf{z} \geq 0$ for all square-integrable $f$), then there exists an orthonormal basis $\{\phi_i\}$ and non-negative eigenvalues $\{\lambda_i\}$ such that:
> $$k(\mathbf{x}, \mathbf{z}) = \sum_{i=1}^{\infty} \lambda_i \phi_i(\mathbf{x})\phi_i(\mathbf{z})$$
> where the series converges absolutely and uniformly.

### What this means in practice

The theorem tells us that we can decompose any valid kernel into an (possibly infinite) sum of products of basis functions. Each basis function $\phi_i$ is a coordinate in the feature space, weighted by $\sqrt{\lambda_i}$.

Define the feature map:
$$\Phi(\mathbf{x}) = [\sqrt{\lambda_1}\phi_1(\mathbf{x}), \sqrt{\lambda_2}\phi_2(\mathbf{x}), \sqrt{\lambda_3}\phi_3(\mathbf{x}), \ldots]$$

Then the kernel is the inner product in this (possibly infinite-dimensional) space:
$$k(\mathbf{x}, \mathbf{z}) = \langle \Phi(\mathbf{x}), \Phi(\mathbf{z}) \rangle = \sum_{i=1}^{\infty} \lambda_i \phi_i(\mathbf{x})\phi_i(\mathbf{z})$$

> __Why don't we use this decomposition in practice?__
>
> For kernels like RBF, the feature space is infinite-dimensional (as we'll see below), so we cannot compute or store the full feature map $\Phi(\mathbf{x})$. We could truncate the series, but this sacrifices exactness. The beauty of the kernel trick is that we don't need to—we just evaluate $k(\mathbf{x}, \mathbf{z})$ directly and get the exact inner product in a single computation. Mercer's theorem guarantees the feature space exists, but we never have to visit it.

___

## Example: RBF Kernel Feature Space
The RBF (Gaussian) kernel is:
$$k_{\gamma}(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma\|\mathbf{x} - \mathbf{z}\|^2\right)$$

For simplicity, consider the 1D case with $\gamma = 1/2$. We can expand this using the Taylor series of $e^x$:

$$k(x, z) = \exp\left(-\frac{1}{2}(x-z)^2\right) = \exp\left(-\frac{1}{2}(x^2 + z^2 - 2xz)\right)$$

$$= \exp\left(-\frac{x^2}{2}\right)\exp\left(-\frac{z^2}{2}\right)\exp(xz)$$

$$= \exp\left(-\frac{x^2}{2}\right)\exp\left(-\frac{z^2}{2}\right)\sum_{n=0}^{\infty}\frac{(xz)^n}{n!}$$

$$= \sum_{n=0}^{\infty}\frac{1}{\sqrt{n!}}\left(x^n\exp\left(-\frac{x^2}{2}\right)\right) \cdot \frac{1}{\sqrt{n!}}\left(z^n\exp\left(-\frac{z^2}{2}\right)\right)$$

This shows that the RBF kernel corresponds to the inner product:
$$k(x, z) = \langle \Phi(x), \Phi(z) \rangle$$
where the feature map is:
$$\Phi(x) = \left[\phi_0(x), \phi_1(x), \phi_2(x), \ldots\right] = \left[\exp\left(-\frac{x^2}{2}\right), \frac{x}{\sqrt{1!}}\exp\left(-\frac{x^2}{2}\right), \frac{x^2}{\sqrt{2!}}\exp\left(-\frac{x^2}{2}\right), \ldots\right]$$

This is an _infinite-dimensional_ feature space! Yet we compute $k(x, z)$ with a single exponential evaluation.

> __Takeaway__:
>
> The RBF kernel implicitly maps data into an infinite-dimensional space spanned by weighted polynomials. This is why RBF kernels are so powerful—they can capture arbitrarily complex nonlinear relationships. Mercer's theorem guarantees this feature space exists and has the right structure, even though we never construct $\Phi(x)$ explicitly.

___

## Connection to the Lecture
In the lecture, we saw that the quadratic kernel $k(v, w) = (1 + vw)^2$ corresponds to the feature map $\phi(v) = [1, \sqrt{2}v, v^2]^{\top}$ in a 3-dimensional space. This is a special case of the finite-dimensional result we proved above.

> __When do we use which version?__
>
> * __Finite-dimensional analysis__: When working with a fixed dataset $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$, the kernel matrix $\mathbf{K}$ is finite, and we can use eigendecomposition directly. This is what happens in kernel PCA, kernel ridge regression, and kernel k-means.
> * __Continuous analysis (Mercer's theorem)__: When proving general theoretical results about kernels or understanding infinite-dimensional feature spaces (like RBF), we need the full theorem. This matters for kernel design and theoretical guarantees.

In practice, we almost never need to explicitly construct the feature map $\phi$. The kernel trick lets us work entirely with $k(\mathbf{x}, \mathbf{z})$ evaluations, and Mercer's theorem assures us that the implicit feature space has the right mathematical structure.

___

## Summary
Mercer's theorem is the mathematical foundation that justifies the kernel trick.

> __Key insights__:
>
> * __Finite case__: When $\mathbf{K}$ is positive semi-definite, eigendecomposition gives us explicit feature coordinates via $\Phi = \mathbf{U}\mathbf{\Lambda}^{1/2}$.
> * __Infinite case__: For continuous kernels on compact domains, Mercer's theorem guarantees an infinite-dimensional feature space exists, even if we cannot construct it explicitly.
> * __Practical implication__: We can use kernels confidently knowing there is always a valid feature space where the kernel computes inner products, without ever building that space.

The genius of kernel methods is that we skip the feature construction entirely and compute similarity directly. Mercer's theorem tells us this shortcut is mathematically sound.

__Additional references__:
* [Mercer, J. (1909). "Functions of positive and negative type and their connection with the theory of integral equations."](https://www.jstor.org/stable/91043) _Philosophical Transactions of the Royal Society A_.
* [Smola & Schölkopf (2004). "A tutorial on support vector regression."](https://alex.smola.org/papers/2004/SmoSch04.pdf) _Statistics and Computing_.
* [Shawe-Taylor & Cristianini (2004). "Kernel Methods for Pattern Analysis."](https://www.cambridge.org/core/books/kernel-methods-for-pattern-analysis/DC6BAEF43C9CD1B26A3AFD0A7B72DAC0) Cambridge University Press.

___
