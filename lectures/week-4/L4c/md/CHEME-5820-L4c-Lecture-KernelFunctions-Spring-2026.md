# L4c: Kernel Functions and Kernel Regression
In this lecture, we explore positive-definite kernels and how they power our first kernel machine: kernel regression.

**You have already seen a matrix built using kernel functions.** The empirical covariance $\hat{\mathbf{\Sigma}} = \frac{1}{n-1}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ from L2a is a scaled Gram matrix built from the linear kernel $k(\mathbf{a},\mathbf{b}) = \mathbf{a}^\top\mathbf{b}$ (see the [derivation](CHEME-5820-L4c-Derivation-Covariance-Redux-Spring-2026.ipynb)). What happens if we replace the linear kernel with something more flexible?

> __Learning Objectives:__
>
> By the end of this lecture, you should be able to:
>
> * __Recognize kernel methods in previous work:__ Understand that the empirical covariance matrix from L2a is a Gram matrix built from the linear kernel, connecting PCA to kernel methods.
> * __Kernel Function Definition:__ Understand what kernel functions are, their role as similarity measures, and the mathematical requirements (symmetry, positive semi-definiteness) for valid kernels.
> * __Kernel Ridge Regression:__ Understand how kernel functions enable non-parametric regression through the "kernel trick" and how to derive the dual formulation with α coefficients.


Let's get started!
___

## Examples
Today, we will use the following examples to illustrate key concepts:
 
> [▶ Can we estimate the similarity of different firms?](CHEME-5820-L4c-Example-MeasureFirmSimilarityScores-Spring-2026.ipynb). In this example, let's explore how to measure the similarity between different firms based upon the similarity of their daily growth rates over 10-year periods. Does this similarity correlate with other firm metrics, e.g., business sector, market capitalization, etc.?
___

## Kernel Functions
Kernel functions let algorithms operate in high-dimensional spaces without explicitly building the coordinates.

> __What are kernel functions__?
>
> A kernel function $k:\mathbb{R}^{m}\times\mathbb{R}^{m}\to\mathbb{R}$ maps two vectors to a scalar similarity score. Valid kernels are symmetric and positive semi-definite. Common kernels include the linear kernel $k(\mathbf{z}_i, \mathbf{z}_j) = \mathbf{z}_i^{\top}\mathbf{z}_j$, the polynomial kernel $k_{d}(\mathbf{z}_i, \mathbf{z}_j) = (1+\mathbf{z}_i^{\top}\mathbf{z}_j)^d$, and the RBF kernel $k_{\gamma}(\mathbf{z}_i, \mathbf{z}_j) = \exp\left(-\gamma \left\|\mathbf{z}_i - \mathbf{z}_j\right\|_{2}^{2}\right)$ with $\gamma>0$.
>
> **When to use each:** The linear kernel is computationally efficient and works well when you expect linear relationships. Polynomial kernels capture polynomial interactions and are useful for interpretable nonlinearity. RBF kernels are the default for complex or unknown relationships, adapting flexibly to the data structure.

The linear kernel $k(\mathbf{z}_i, \mathbf{z}_j) = \mathbf{z}_i^\top\mathbf{z}_j$ is just the dot product. Applied to centered features in L2a, it gave us the empirical covariance matrix (up to $\frac{1}{n-1}$ scaling), a *linear* view of how features relate.  Polynomial and RBF kernels extend this to *nonlinear* similarity, implicitly mapping data into a richer feature space. All the machinery we built for covariance and PCA (Gram matrices, eigendecompositions, centering) carries over directly; we just swap the kernel function.

### Kernels correspond to inner products in feature spaces

Every positive-definite kernel corresponds to an inner product in some feature space. By Mercer's theorem, there exists a transformation $\phi$ such that:

$$k(\mathbf{x},\mathbf{z}) = \langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle$$

So kernels compute inner products in a transformed space $\mathcal{H}$ without explicitly constructing that space. In that space, the model is linear:

$$
\boxed{
    \hat y(\mathbf{z})
    = \phi(\mathbf{z})^\top\,\hat\theta_\lambda
    = \sum_{i=1}^n \alpha_i\,\underbrace{\langle \phi(\mathbf{z}),\,\phi(\mathbf{x}_i)\rangle}_{\text{similarity in $\mathcal H$}}
    = \sum_{i=1}^n \alpha_i\,k(\mathbf{z},\mathbf{x}_i).}
$$

> **Quadratic kernel**
> For scalars $v_i, v_j \in \mathbb{R}$, consider:
> $$k(v_i, v_j) = (1 + v_i v_j)^2$$
> Expand:
> $$k(v_i, v_j) = 1 + 2v_i v_j + (v_i v_j)^2$$
> Define $\phi(v) = [1, \sqrt{2}v, v^2]^{\top}$. Then we can compute the inner product in the feature space:
> $$
\begin{align*}
\langle \phi(v_i), \phi(v_j) \rangle &= \langle [1, \sqrt{2}v_i, v_i^2]^{\top}, [1, \sqrt{2}v_j, v_j^2]^{\top} \rangle \\
& = 1 \cdot 1 + (\sqrt{2}v_i)(\sqrt{2}v_j) + (v_i^2)(v_j^2) \\
& = 1 + 2v_i v_j + (v_i v_j)^2 \quad\blacksquare
\end{align*}
$$

**Takeaway:** The quadratic kernel maps a one-dimensional input into a three-dimensional feature space. Nonlinearity in the original space becomes a linear inner product in the transformed space. Let's see how we can use kernels for measuring similarity.

> __Example__
>
> [▶ Can we estimate the similarity of different firms?](CHEME-5820-L4c-Example-MeasureFirmSimilarityScores-Spring-2026.ipynb). In this example, let's explore how to measure the similarity between different firms based upon the similarity of their daily growth rates over 10-year periods. Does this similarity correlate with other firm metrics, e.g., business sector, market capitalization, etc.?
___

### Can any function be a kernel function?

Not every function is a valid kernel. Valid kernels must satisfy strict mathematical constraints.

> __Rules for a valid kernel function__:
>
> A function $k:\mathbb{R}^{m}\times\mathbb{R}^{m}\to\mathbb{R}$ is valid if, for any finite set $\{\mathbf{v}_1, \dots, \mathbf{v}_n\}$, the kernel matrix $\mathbf{K}$ with entries $K_{ij} = k(\mathbf{v}_i, \mathbf{v}_j)$ is symmetric and positive semidefinite. Equivalently, all eigenvalues of $\mathbf{K}$ are non-negative and $\mathbf{x}^{\top}\mathbf{K}\mathbf{x} \geq 0$ for any real vector $\mathbf{x}$.

The covariance matrix $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ from L2a was a Gram matrix of centered features under the linear kernel; $\tilde{\mathbf{X}}\tilde{\mathbf{X}}^\top$ was a Gram matrix of centered samples (see the [derivation](CHEME-5820-L4c-Derivation-Covariance-Redux-Spring-2026.ipynb) for the centering proof). More generally, for any valid kernel:

$$K_{ij} = k(\mathbf{v}_i, \mathbf{v}_j)$$

With the linear kernel and rows of $\mathbf{X} \in \mathbb{R}^{n \times m}$, this gives $\mathbf{K} = \mathbf{X}\mathbf{X}^{\top}$. Replacing the linear kernel with a polynomial or RBF kernel produces a different $\mathbf{K}$ that captures nonlinear relationships between the same samples. In all cases, $\mathbf{K}$ is symmetric and positive semidefinite, so its eigendecomposition exists with non-negative eigenvalues. This spectral structure is why eigendecomposition of Gram matrices is central to kernel PCA and other kernel methods.

Next, we will compute a Gram matrix from our financial data and examine its eigendecomposition.

___

## Kernel regression

Now that we know what makes a valid kernel, we can use kernels for nonlinear regression. Kernel regression is non-parametric: it predicts outputs using weighted combinations of training examples rather than a single global model.

> __How does it work__? Kernel regression assigns weights to training points based on similarity to a query point. The prediction is a weighted sum of those points. 

Suppose we have a dataset $\mathcal{D} = \{(\mathbf{x}_{i},y_{i}) \mid i = 1,2,\dots,n\}$ with features $\mathbf{x}_i \in \mathbb{R}^{m}$ and targets $y_i \in \mathbb{R}$. We can model this as linear regression:
$$
\hat{\mathbf{y}} = \hat{\mathbf{X}}\theta
$$
where $\hat{\mathbf{X}}$ contains augmented feature vectors and $\theta\in\mathbb{R}^{p}$ with $p=m+1$. The ridge solution is:
$$
\hat{\mathbf{\theta}}_{\lambda} = \left(\hat{\mathbf{X}}^{\top}\hat{\mathbf{X}}+\lambda\,\mathbf{I}\right)^{-1}\hat{\mathbf{X}}^{\top}\mathbf{y}
$$

#### Kernel ridge regression
Write the parameter vector as a weighted sum of training examples: $\hat{\theta}_{\lambda} = \sum_{i=1}^{n}\alpha_{i}\hat{\mathbf{x}}_{i}$. This leads to predictions based on inner products only. It lets us replace inner products with kernel evaluations, enabling implicit feature engineering. For a new point $\hat{\mathbf{z}}$:
$$
\begin{align*}
\hat{y} & = \hat{\mathbf{z}}^{\top}\hat{\theta}_{\lambda} = \sum_{i=1}^{n}\alpha_{i}\left\langle\hat{\mathbf{z}},\hat{\mathbf{x}}_{i}\right\rangle\\
        & = \sum_{i=1}^{n}\alpha_{i}\,k(\hat{\mathbf{z}},\hat{\mathbf{x}}_{i})
\end{align*}
$$

To solve for $\alpha_i$, we need the matrix of all pairwise inner products between training points. For the linear kernel in the augmented feature space, this is $\mathbf{K} = \hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}$. More generally, if $\phi(\cdot)$ is the (possibly implicit) feature map, then
$$K_{ij} = k(\mathbf{x}_i,\mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle,$$
so $\mathbf{K} = \Phi\Phi^{\top}$ with rows of $\Phi$ equal to $\phi(\mathbf{x}_i)$. We never need to form $\Phi$ explicitly; we evaluate $k$ directly (e.g., polynomial $k(\mathbf{x},\mathbf{z})=(\mathbf{x}^{\top}\mathbf{z}+c)^d$ or RBF $k(\mathbf{x},\mathbf{z})=\exp(-\|\mathbf{x}-\mathbf{z}\|^2/(2\sigma^2))$). This Gram matrix is symmetric and positive semi-definite by construction.

The two expressions for $\hat{\theta}_{\lambda}$ can be equated. Starting from $\hat{\theta}_{\lambda} = \hat{\mathbf{X}}^{\top}\alpha$, we substitute into the ridge regression optimality condition $(\hat{\mathbf{X}}^{\top}\hat{\mathbf{X}}+\lambda\,\mathbf{I})\hat{\theta}_{\lambda} = \hat{\mathbf{X}}^{\top}\mathbf{y}$:
$$
\begin{align*}
(\hat{\mathbf{X}}^{\top}\hat{\mathbf{X}}+\lambda\,\mathbf{I})\hat{\mathbf{X}}^{\top}\alpha &= \hat{\mathbf{X}}^{\top}\mathbf{y}\\
\hat{\mathbf{X}}(\hat{\mathbf{X}}^{\top}\hat{\mathbf{X}}+\lambda\,\mathbf{I})\hat{\mathbf{X}}^{\top}\alpha & = \hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}\mathbf{y}\quad\mid\text{multiply both sides by $\hat{\mathbf{X}}$ on the left} \\
(\hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}+\lambda\,\mathbf{I})\hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}\alpha & = \hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}\mathbf{y}\quad\mid\text{key step: see explanation below}\\
(\mathbf{K}+\lambda\,\mathbf{I})\mathbf{K}\alpha & = \mathbf{K}\mathbf{y}\quad\mid\text{substitute $\mathbf{K} = \hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}$}\\
\alpha &= (\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{K}\mathbf{y}\quad\mid\text{multiply both sides by $(\mathbf{K}+\lambda\mathbf{I})^{-1}$ on the left}
\end{align*}
$$

> __Understanding the key transformation step__: The identity
> $$\hat{\mathbf{X}}(\hat{\mathbf{X}}^{\top}\hat{\mathbf{X}}+\lambda \mathbf{I})\hat{\mathbf{X}}^{\top} = (\hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}+\lambda \mathbf{I})\hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}$$
> follows by expanding both sides. It moves the computation from feature space ($p\times p$) to sample space ($n\times n$). When $p \gg n$, the sample-space system is far cheaper; when $n \gg p$, the feature-space formulation may be preferable.

where $\mathbf{K}$ is the Gram matrix with entries $K_{ij}=k(\mathbf{x}_i,\mathbf{x}_j)$ (symmetric and positive semi-definite by kernel validity), $\mathbf{I}$ is the identity matrix, $\mathbf{y}$ is the observed output vector, and $\lambda\geq{0}$ is the regularization parameter. For the linear kernel, this reduces to $\mathbf{K}=\hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}$.

> **Regularization intuition:**  As $\lambda \to 0$, the solution emphasizes fitting the training data (high variance, low bias). As $\lambda \to \infty$, the solution shrinks $\alpha$ toward zero (low variance, high bias). The choice of $\lambda$ controls this bias-variance tradeoff and is typically selected via cross-validation.

In practice, $\alpha = (\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{K}\mathbf{y}$ simplifies in implementation since we can factor or directly solve the linear system.

> __Technical Note__: We derived $\alpha$ using the linear kernel Gram matrix $\mathbf{K} = \hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}$. The solution depends only on kernel evaluations, so any valid kernel can replace $\mathbf{K}$ without changing the algorithm.

#### The Kernel Trick

Predictions depend only on inner products between data points. By substituting $k(\hat{\mathbf{z}},\hat{\mathbf{x}}_{i})$ for $\left\langle\hat{\mathbf{z}},\hat{\mathbf{x}}_{i}\right\rangle$, we can work in high-dimensional feature spaces without explicitly constructing them.

> __The Kernel Trick__: Express predictions as inner products, then replace those inner products with kernel evaluations. The algorithm stays the same; only the similarity function changes. We always solve $\alpha = (\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{K}\mathbf{y}$ using a kernel matrix. By choosing different kernels (linear, polynomial, RBF), we adapt to different feature spaces while maintaining the same mathematical framework.

___

## Lab
In the lab, we will apply kernel regression to financial data. We will implement kernel ridge regression to predict stock returns for a ticker using a feature set constructed from a market factor. This gives us a non-linear model to predict returns that could be used instead of the classical single-factor linear model.

## Summary
Kernel functions are similarity measures that enable non-parametric modeling of complex relationships without explicitly constructing high-dimensional feature representations.

> __Key Takeaways:__
>
> * __The covariance matrix was a kernel matrix all along.__ The empirical covariance $\hat{\mathbf{\Sigma}} = \frac{1}{n-1}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ is a scaled Gram matrix under the linear kernel. All of our PCA work from L2a was implicitly a linear kernel method (see the [derivation](CHEME-5820-L4c-Derivation-Covariance-Redux-Spring-2026.ipynb) for the full proof).
> * __Nonlinear kernels generalize this idea.__ By replacing the linear kernel with polynomial or RBF kernels, we capture nonlinear relationships between features or samples using the same Gram matrix and eigendecomposition machinery.
> * __Valid kernels must be symmetric and positive semi-definite__, ensuring the corresponding Gram matrices have the spectral properties needed for all kernel-based algorithms.
> * __The kernel trick__ allows us to work implicitly in high-dimensional spaces by replacing inner products $\langle\mathbf{z}_i, \mathbf{z}_j\rangle$ with kernel evaluations $k(\mathbf{z}_i, \mathbf{z}_j)$, enabling scalable algorithms.
> * __Kernel ridge regression__ reformulates linear regression using a dual representation with coefficients $\alpha$, shifting from a model-centric view (fitting $\theta$) to a data-centric view (weighted training examples).

Kernel methods form the mathematical foundation for support vector machines (SVMs), Gaussian process models, and other modern machine learning algorithms.

___
