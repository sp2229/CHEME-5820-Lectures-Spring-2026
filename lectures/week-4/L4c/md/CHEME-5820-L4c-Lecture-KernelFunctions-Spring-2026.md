# L4c: Kernel Functions and Kernel Regression
In this lecture, we explore positive-definite kernel functions and how they power our first kernel machine: kernel regression.

**You have already seen kernel functions (and didn't even know it).** The empirical covariance $\hat{\mathbf{\Sigma}} = \frac{1}{n-1}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ is a scaled Gram matrix built from the linear kernel $k(\mathbf{a},\mathbf{b}) = \mathbf{a}^\top\mathbf{b}$ (see the [derivation](CHEME-5820-L4c-Derivation-Covariance-Redux-Spring-2026.ipynb)). What happens if we replace the linear kernel with a nonlinear one?

> __Learning Objectives:__
>
> By the end of this lecture, you should be able to:
>
> * __Recognize kernel methods in previous work:__ Understand that the empirical covariance matrix from L2a is a Gram matrix built from the linear kernel, connecting PCA to kernel methods.
> * __Kernel Function Definition:__ Understand what kernel functions are, their role as similarity measures, and the mathematical requirements (symmetry, positive semi-definiteness) for valid kernels.
> * __Kernel Ridge Regression:__ Understand how kernel functions enable memory-based learning through the _kernel trick_ and how to derive the dual formulation with $\mathbf{\alpha}$ coefficients.


Let's get started!
___

## Examples
Today, we will use the following examples to illustrate key concepts:
 
> [▶ Can we estimate the similarity of different firms?](CHEME-5820-L4c-Example-MeasureFirmSimilarityScores-Spring-2026.ipynb). In this example, we measure the similarity between different firms based on the similarity of their daily growth rates over 10-year periods. Does this similarity correlate with other firm metrics, e.g., business sector, market capitalization, etc.?
___

## Kernel Functions
Kernel functions let algorithms operate in high-dimensional spaces without explicitly building the coordinates.

> __What are kernel functions__?
>
> A kernel function $k:\mathbb{R}^{m}\times\mathbb{R}^{m}\to\mathbb{R}$ maps two vectors to a scalar similarity score. Valid kernels are symmetric and positive semidefinite. Common kernels include the linear kernel $k(\mathbf{z}_i, \mathbf{z}_j) = \mathbf{z}_i^{\top}\mathbf{z}_j$, the polynomial kernel $k_{d}(\mathbf{z}_i, \mathbf{z}_j) = (1+\mathbf{z}_i^{\top}\mathbf{z}_j)^d$, and the RBF kernel $k_{\gamma}(\mathbf{z}_i, \mathbf{z}_j) = \exp\left(-\gamma \left\|\mathbf{z}_i - \mathbf{z}_j\right\|_{2}^{2}\right)$ with $\gamma>0$.
>
> **When to use each:** The linear kernel is computationally efficient and works well when you expect linear relationships. Polynomial kernels capture polynomial interactions and are useful for interpretable nonlinearity. RBF kernels are the default for complex or unknown relationships, adapting to the data structure.

The linear kernel $k(\mathbf{z}_i, \mathbf{z}_j) = \mathbf{z}_i^\top\mathbf{z}_j$ is just the dot product. Applied to centered features, it gave us the empirical covariance matrix (up to $\frac{1}{n-1}$ scaling), a *linear* view of how features relate.  

Polynomial and RBF kernels extend this to *nonlinear* similarity, implicitly mapping data into a higher-dimensional feature space. All the machinery we built for covariance and PCA (Gram matrices, eigendecompositions, centering) carries over directly; we just swap the kernel function.

### Kernels correspond to inner products in feature spaces

Every positive-definite kernel corresponds to an inner product in some feature space. There exists a transformation $\phi$ such that:

$$k(\mathbf{x},\mathbf{z}) = \langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle$$

This means kernels compute inner products in a transformed space without explicitly constructing that space. 

> **Mathematical Foundation (optional):** The formal result is Mercer's theorem, which guarantees that valid kernels have this inner product representation. For the curious, see the [advanced derivation](CHEME-5820-L4c-Advanced-MercerTheorem-Spring-2026.ipynb) for details, but you don't need the theorem to use kernels effectively.

A concrete example:

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

**Takeaway:** The quadratic kernel maps a one-dimensional input into a three-dimensional feature space. The kernel $k(v_i, v_j)$ computes the inner product in that space without ever constructing the coordinates $[1, \sqrt{2}v, v^2]$. 

This __kernel trick__ applies to regression and classification: we can work in very high-dimensional spaces while only evaluating the kernel function. 
___

### Can any function be a kernel function?

Not every function is a valid kernel. Valid kernels must satisfy strict mathematical constraints.

> __Rules for a valid kernel function__:
>
> A function $k:\mathbb{R}^{m}\times\mathbb{R}^{m}\to\mathbb{R}$ is valid if, for any finite set $\{\mathbf{v}_1, \dots, \mathbf{v}_n\in\mathbb{R}^m\}$, the kernel matrix $\mathbf{K}$ with entries $K_{ij} = k(\mathbf{v}_i, \mathbf{v}_j)$ is symmetric and positive semidefinite. Equivalently, all eigenvalues of $\mathbf{K}$ are non-negative and $\mathbf{x}^{\top}\mathbf{K}\mathbf{x} \geq 0$ for any real vector $\mathbf{x}$.

The covariance matrix $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ from L2a was a Gram matrix of centered features under the linear kernel; $\tilde{\mathbf{X}}\tilde{\mathbf{X}}^\top$ was a Gram matrix of centered samples (see the [derivation](CHEME-5820-L4c-Derivation-Covariance-Redux-Spring-2026.ipynb) for the centering proof). 

> __Sample vs. Feature Gram Matrices__
>
> More generally, for any valid kernel:
> $$K_{ij} = k(\mathbf{v}_i, \mathbf{v}_j)$$
> With the linear kernel and data $\mathbf{X} \in \mathbb{R}^{n \times m}$ (n samples, m features), > we can form two different Gram matrices:
>
> - **Sample Gram matrix** $\mathbf{K} = \mathbf{X}\mathbf{X}^{\top} \in \mathbb{R}^{n \times n}$: measures similarity between data points. For kernel methods (kernel ridge regression, kernel PCA, SVMs), we use the **sample Gram matrix** $\mathbf{K} = \mathbf{X}\mathbf{X}^{\top}$ (this give us some interesting behavior that we'll see in a minute). 
>
> - **Feature Gram matrix** $\mathbf{C} = \mathbf{X}^{\top}\mathbf{X} \in \mathbb{R}^{m \times m}$: measures similarity between features (covariance structure). When the data is centered, $\mathbf{C}$ is proportional to the empirical covariance matrix. When we use a nonlinear kernel, $\mathbf{C}$ captures nonlinear relationships between features in the implicit feature space.
> 
> Replacing the linear kernel with a polynomial or RBF kernel produces a different $\mathbf{K}$ that captures nonlinear relationships between the same samples.

In all cases, $\mathbf{K}$ is symmetric and positive semidefinite, so its eigendecomposition exists with non-negative eigenvalues. This spectral structure is why eigendecomposition of Gram matrices is central to kernel PCA and other kernel methods that we will explore in future lectures.

Next, we'll use kernels to measure firm similarity based on growth rates.

> __Example: Measuring Firm Similarity__
> 
> [▶ Can we estimate the similarity of different firms?](CHEME-5820-L4c-Example-MeasureFirmSimilarityScores-Spring-2026.ipynb). In this example, we measure the similarity between different firms based on the similarity of their daily growth rates over 10-year periods. Does this similarity correlate with other firm metrics, e.g., business sector, market capitalization, etc.?

___

## Kernel regression

Now that we know what makes a valid kernel, we can use kernels for nonlinear regression. Kernel regression is a memory-based learning approach: predictions depend on weighted combinations of training examples rather than a fixed parametric form.

> __How does it work__? Kernel regression assigns weights to training points based on similarity to a query point. The prediction is a weighted sum of those points. 

Suppose we have a dataset $\mathcal{D} = \{(\mathbf{x}_{i},y_{i}) \mid i = 1,2,\dots,n\}$ with features $\mathbf{x}_i \in \mathbb{R}^{m}$ and targets $y_i \in \mathbb{R}$. We can model this as __linear regression__:
$$
\hat{\mathbf{y}} = \hat{\mathbf{X}}\theta
$$
where $\hat{\mathbf{X}}$ contains augmented feature vectors and $\theta\in\mathbb{R}^{p}$ with $p=m+1$. The ridge solution is:
$$
\hat{\mathbf{\theta}}_{\lambda} = \left(\hat{\mathbf{X}}^{\top}\hat{\mathbf{X}}+\lambda\,\mathbf{I}\right)^{-1}\hat{\mathbf{X}}^{\top}\mathbf{y}
$$

#### Kernel ridge regression
Write the parameter vector as a weighted sum of training examples: $\hat{\theta}_{\lambda} = \sum_{i=1}^{n}\alpha_{i}\hat{\mathbf{x}}_{i}$. Then, for a new point $\hat{\mathbf{z}}$:
$$
\begin{align*}
\hat{y} & = \hat{\mathbf{z}}^{\top}\hat{\theta}_{\lambda} = \sum_{i=1}^{n}\alpha_{i}\left\langle\hat{\mathbf{z}},\hat{\mathbf{x}}_{i}\right\rangle\quad\text{(replace inner products with kernel evaluations)}\\
        & = \sum_{i=1}^{n}\alpha_{i}\,k(\hat{\mathbf{z}},\hat{\mathbf{x}}_{i})
\end{align*}
$$

To solve for $\alpha_i$, we need the matrix of all pairwise inner products between training points. For the linear kernel in the augmented feature space, this is $\mathbf{K} = \hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}$. More generally, if $\phi(\cdot)$ is the (possibly implicit) feature map, then
$$K_{ij} = k(\mathbf{x}_i,\mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle,$$
so $\mathbf{K} = \Phi\Phi^{\top}$ with rows of $\Phi$ equal to $\phi(\mathbf{x}_i)$. We never need to form $\Phi$ explicitly; we evaluate $k$ directly (e.g., polynomial $k(\mathbf{x},\mathbf{z})=(\mathbf{x}^{\top}\mathbf{z}+c)^d$ or RBF $k(\mathbf{x},\mathbf{z})=\exp(-\|\mathbf{x}-\mathbf{z}\|^2/(2\sigma^2))$). This Gram matrix is symmetric and positive semidefinite by construction.

The two expressions for $\hat{\theta}_{\lambda}$ can be equated. Starting from $\hat{\theta}_{\lambda} = \hat{\mathbf{X}}^{\top}\alpha$, we substitute into the ridge regression optimality condition $(\hat{\mathbf{X}}^{\top}\hat{\mathbf{X}}+\lambda\,\mathbf{I})\hat{\theta}_{\lambda} = \hat{\mathbf{X}}^{\top}\mathbf{y}$:
$$
\begin{align*}
(\hat{\mathbf{X}}^{\top}\hat{\mathbf{X}}+\lambda\,\mathbf{I})\hat{\mathbf{X}}^{\top}\alpha &= \hat{\mathbf{X}}^{\top}\mathbf{y}\\
\hat{\mathbf{X}}(\hat{\mathbf{X}}^{\top}\hat{\mathbf{X}}+\lambda\,\mathbf{I})\hat{\mathbf{X}}^{\top}\alpha & = \hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}\mathbf{y}\quad\mid\text{multiply both sides by $\hat{\mathbf{X}}$ on the left} \\
(\hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}+\lambda\,\mathbf{I})\hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}\alpha & = \hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}\mathbf{y}\quad\mid\text{key step: see explanation below}\\
(\mathbf{K}+\lambda\,\mathbf{I})\mathbf{K}\alpha & = \mathbf{K}\mathbf{y}\quad\mid\text{substitute $\mathbf{K} = \hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}$}\\
\mathbf{K}^{-1}(\mathbf{K}+\lambda\,\mathbf{I})\mathbf{K}\alpha & = \mathbf{K}^{-1}\mathbf{K}\mathbf{y}\quad\mid\text{multiply both sides by $\mathbf{K}^{-1}$ on the left}\\
(\mathbf{K}^{-1}\mathbf{K}+\lambda\mathbf{K}^{-1})\mathbf{K}\alpha & = \mathbf{y}\quad\mid\text{distribute $\mathbf{K}^{-1}$ on the left; simplify right side}\\
(\mathbf{I}+\lambda\mathbf{K}^{-1})\mathbf{K}\alpha & = \mathbf{y}\quad\mid\text{simplify $\mathbf{K}^{-1}\mathbf{K} = \mathbf{I}$}\\
\mathbf{K}\alpha+\lambda\mathbf{I}\alpha & = \mathbf{y}\quad\mid\text{distribute $\mathbf{K}$ on the left}\\
(\mathbf{K}+\lambda\,\mathbf{I})\alpha & = \mathbf{y}\quad\mid\text{factor out $\alpha$}\\
\alpha &= (\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{y}\quad\mid\text{multiply both sides by $(\mathbf{K}+\lambda\mathbf{I})^{-1}$ on the left}
\end{align*}
$$

> __Understanding the key transformation step__: The identity
> $$\hat{\mathbf{X}}(\hat{\mathbf{X}}^{\top}\hat{\mathbf{X}}+\lambda \mathbf{I})\hat{\mathbf{X}}^{\top} = (\hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}+\lambda \mathbf{I})\hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}$$
> follows by expanding both sides. It moves the computation from feature space ($p\times p$) to sample space ($n\times n$). When $p \gg n$, the sample-space system is far cheaper; when $n \gg p$, the feature-space formulation may be preferable.

where $\mathbf{K}$ is the Gram matrix with entries $K_{ij}=k(\mathbf{x}_i,\mathbf{x}_j)$ (symmetric and positive semidefinite by kernel validity), $\mathbf{I}$ is the identity matrix, $\mathbf{y}$ is the observed output vector, and $\lambda\geq{0}$ is the regularization parameter. 

> **Regularization intuition:**  As $\lambda \to 0$, the solution emphasizes fitting the training data (high variance, low bias). As $\lambda \to \infty$, the solution shrinks $\alpha$ toward zero (low variance, high bias). The choice of $\lambda$ controls this bias-variance tradeoff and is typically selected via cross-validation.

In practice, $\alpha = (\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{y}$ simplifies in implementation since we can factor or directly solve the linear system.

> __Technical Note__: We derived $\alpha$ using the linear kernel Gram matrix $\mathbf{K} = \hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}$. The solution depends only on kernel evaluations, so any valid kernel can replace $\mathbf{K}$ without changing the algorithm.

#### The Kernel Trick

Predictions depend only on inner products between data points. By substituting $k(\hat{\mathbf{z}},\hat{\mathbf{x}}_{i})$ for $\left\langle\hat{\mathbf{z}},\hat{\mathbf{x}}_{i}\right\rangle$, we can work in high-dimensional feature spaces without explicitly constructing them.

> __The Kernel Trick__: Express predictions as inner products, then replace those inner products with kernel evaluations. The algorithm stays the same; only the similarity function changes. We always solve $\alpha = (\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{y}$ using a kernel matrix. By choosing different kernels (linear, polynomial, RBF), we adapt to different feature spaces while maintaining the same mathematical framework.

___

## Lab
In the lab, we will apply kernel regression to financial data. We will implement kernel ridge regression to predict stock returns for a ticker using a feature set constructed from a market factor. This gives us a nonlinear model to predict returns that could be used in place of the classical single-factor linear model.

## Summary
Kernel functions are similarity measures that enable memory-based learning for modeling complex relationships without explicitly constructing high-dimensional feature representations.

> __Key Takeaways:__
>
> * __Kernels generalize our previous work with covariance and PCA.__ The empirical covariance matrix is a scaled Gram matrix under the linear kernel. By replacing the linear kernel with polynomial or RBF kernels, we capture nonlinear relationships between features or samples using the same Gram matrix and eigendecomposition machinery (see the [derivation](CHEME-5820-L4c-Derivation-Covariance-Redux-Spring-2026.ipynb) for the full proof).
> * __Valid kernels enable the kernel trick.__ Symmetric, positive semidefinite kernels correspond to inner products in possibly infinite-dimensional feature spaces. By replacing inner products with kernel evaluations, we work implicitly in high-dimensional spaces without explicitly constructing feature coordinates.
> * __Kernel ridge regression shifts from model-centric to memory-based learning.__ The dual representation expresses predictions as weighted combinations of training examples retained from the dataset, enabling nonlinear regression through the choice of kernel function.

Kernel methods form the mathematical foundation for support vector machines (SVMs), Gaussian process models, and other modern machine learning algorithms.

___
