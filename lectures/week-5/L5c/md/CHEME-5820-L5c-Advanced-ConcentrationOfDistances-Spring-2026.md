# Advanced Topic: Concentration of Distances in High Dimensions
In high-dimensional spaces, classical distance metrics lose their ability to distinguish between nearby and distant points. This phenomenon, known as the concentration of distances, has direct consequences for distance-based algorithms such as K-Nearest Neighbors. This advanced topic presents the formal mathematical framework underlying distance concentration, following the analysis of [Beyer et al. (1999)](https://link.springer.com/chapter/10.1007/3-540-49257-7_15).

> __Learning Objectives:__
> 
> By the end of this advanced topic, you should be able to:
> 
> * __State the concentration of distances result:__ Formulate the conditions under which the ratio of maximum to minimum pairwise distances converges to unity as dimensionality increases.
> * __Derive the concentration using the law of large numbers:__ Show that squared Euclidean distances between random points concentrate around a common mean as the number of features grows.
> * __Explain implications for KNN classification:__ Describe why distance concentration undermines nearest-neighbor algorithms and how kernel functions provide an alternative.

Let's explore the mathematical foundations!
___

## Setup and Notation
Consider $n$ data points $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n \in \mathbb{R}^m$ and a query point $\mathbf{x}^* \in \mathbb{R}^m$. We measure distances using the Euclidean norm. The key question is: what happens to the relative contrast between nearest and farthest neighbors as the dimension $m$ grows?

> __Definition (Distance contrast).__ For a query point $\mathbf{x}^*$ and a set of reference points $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$, define the minimum and maximum distances:
> $$d_{\min}^{(m)} = \min_{i=1,\ldots,n} \|\mathbf{x}_i - \mathbf{x}^*\|_2 \qquad d_{\max}^{(m)} = \max_{i=1,\ldots,n} \|\mathbf{x}_i - \mathbf{x}^*\|_2$$
> The __relative contrast__ is the ratio:
> $$\rho_m = \frac{d_{\max}^{(m)} - d_{\min}^{(m)}}{d_{\min}^{(m)}}$$
> When $\rho_m \to 0$, all points are approximately equidistant from the query, and nearest-neighbor distinctions become meaningless.

To analyze the behavior of $\rho_m$, we model the feature components as independent random variables with shared statistical properties.
___

## Concentration of Squared Euclidean Distance
The squared Euclidean distance between a reference point $\mathbf{x}_i$ and the query $\mathbf{x}^*$ is a sum of $m$ terms:
$$\|\mathbf{x}_i - \mathbf{x}^*\|_2^2 = \sum_{j=1}^{m} (x_{ij} - x_j^*)^2$$
where $x_{ij}$ is the $j$-th component of $\mathbf{x}_i$ and $x_j^*$ is the $j$-th component of $\mathbf{x}^*$.

> __Assumption (Independent features).__ Assume the feature components of each data point are independent and identically distributed (i.i.d.) random variables with finite mean $\mu$ and finite variance $\sigma^2 > 0$. The query point $\mathbf{x}^*$ is drawn from the same distribution, independently of the reference points.

Under this assumption, each squared difference $Z_j = (x_{ij} - x_j^*)^2$ is itself a random variable. Since $x_{ij}$ and $x_j^*$ are independent with the same distribution, the difference $x_{ij} - x_j^*$ has mean $0$ and variance $2\sigma^2$. Therefore:
$$\mathbb{E}[Z_j] = \mathbb{E}[(x_{ij} - x_j^*)^2] = \text{Var}(x_{ij} - x_j^*) + (\mathbb{E}[x_{ij} - x_j^*])^2 = 2\sigma^2$$

> __Proposition (Concentration of normalized distance).__ Under the independent features assumption, for any fixed reference point $\mathbf{x}_i$:
> $$\frac{1}{m}\|\mathbf{x}_i - \mathbf{x}^*\|_2^2 = \frac{1}{m}\sum_{j=1}^{m} Z_j \xrightarrow{p} 2\sigma^2 \quad \text{as } m \to \infty$$
> by the weak law of large numbers. This holds for every reference point $\mathbf{x}_i$, $i = 1, \ldots, n$.

The law of large numbers applies because $Z_1, Z_2, \ldots, Z_m$ are i.i.d. random variables with finite mean $\mathbb{E}[Z_j] = 2\sigma^2$. As $m$ grows, the sample average $\frac{1}{m}\sum_{j=1}^{m} Z_j$ converges in probability to the population mean $2\sigma^2$.
___

## The Beyer et al. Concentration Result
The proposition above applies to each individual distance. Combining it across all reference points yields the main concentration result.

> __Theorem ([Beyer et al., 1999](https://link.springer.com/chapter/10.1007/3-540-49257-7_15)).__ Let $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$ be reference points and $\mathbf{x}^*$ a query point in $\mathbb{R}^m$, with feature components drawn i.i.d. from a distribution with finite mean $\mu$ and finite variance $\sigma^2 > 0$. If the number of reference points $n$ is fixed, then:
> $$\frac{d_{\max}^{(m)} - d_{\min}^{(m)}}{d_{\min}^{(m)}} \xrightarrow{p} 0 \quad \text{as } m \to \infty$$

The proof follows from the proposition. Since $\frac{1}{m}\|\mathbf{x}_i - \mathbf{x}^*\|_2^2 \xrightarrow{p} 2\sigma^2$ for every $i$, the unnormalized distances satisfy $\|\mathbf{x}_i - \mathbf{x}^*\|_2 / \sqrt{m} \xrightarrow{p} \sqrt{2}\sigma$. Both the maximum and minimum over a fixed, finite set of convergent sequences converge to the same limit:
$$\frac{d_{\max}^{(m)}}{\sqrt{m}} \xrightarrow{p} \sqrt{2}\sigma \qquad \frac{d_{\min}^{(m)}}{\sqrt{m}} \xrightarrow{p} \sqrt{2}\sigma$$

Therefore:
$$\frac{d_{\max}^{(m)} - d_{\min}^{(m)}}{d_{\min}^{(m)}} = \frac{d_{\max}^{(m)}/\sqrt{m} - d_{\min}^{(m)}/\sqrt{m}}{d_{\min}^{(m)}/\sqrt{m}} \xrightarrow{p} \frac{\sqrt{2}\sigma - \sqrt{2}\sigma}{\sqrt{2}\sigma} = 0$$

This completes the argument. The relative contrast vanishes because every pairwise distance grows at the same rate $\sqrt{2m\sigma^2}$, with the spread between distances growing slower than the distances themselves.
___

## Rate of Concentration
The central limit theorem provides additional insight into how quickly the contrast vanishes.

> __Proposition (CLT-based variance bound).__ Under the same assumptions as the theorem above, and further assuming that $Z_j = (x_{ij} - x_j^*)^2$ has finite fourth moment, the central limit theorem gives:
> $$\sqrt{m}\left(\frac{1}{m}\|\mathbf{x}_i - \mathbf{x}^*\|_2^2 - 2\sigma^2\right) \xrightarrow{d} \mathcal{N}(0,\text{Var}(Z_j))$$
> Equivalently, the normalized squared distance concentrates around $2\sigma^2$ and the typical fluctuations shrink like $1/\sqrt{m}$ as $m$ grows.
> More precisely, the variance of the normalized squared distance satisfies:
> $$\text{Var}\!\left(\frac{1}{m}\|\mathbf{x}_i - \mathbf{x}^*\|_2^2\right) = \frac{\text{Var}(Z_j)}{m}$$
> These statements quantify how quickly the squared distances concentrate; translating this into a rate for $\rho_m$ requires additional steps.

This rate quantifies how rapidly the curse of dimensionality degrades distance-based algorithms: even moderately high dimensions can render Euclidean distance nearly non-discriminative.
___

## Implications for KNN and Motivation for Kernels
The concentration of distances has direct consequences for the KNN algorithm and motivates the use of kernel functions.

> __Implications__
> 
> * __KNN becomes unreliable__: When all distances are approximately equal, the set of $K$ nearest neighbors is determined by noise rather than by meaningful structure in the data. The majority vote over these neighbors becomes essentially random.
> * __The $L_p$ norm matters__: [Aggarwal et al. (2001)](https://link.springer.com/chapter/10.1007/3-540-44503-X_27) showed that lower-order $L_p$ norms (e.g., Manhattan distance with $p=1$) concentrate more slowly than Euclidean distance ($p=2$), making them somewhat more robust in high dimensions. However, all fixed $L_p$ norms eventually suffer the same concentration.
> * __Kernels as a remedy__: Kernel functions measure similarity in an implicit feature space $k(\mathbf{x}, \mathbf{z}) = \langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle$ where the mapping $\phi$ is chosen to preserve discriminative structure. A well-chosen kernel can maintain meaningful similarity contrasts even when explicit distance metrics in the original space cannot.

___

## Summary
The concentration of distances is a fundamental limitation of distance-based methods in high-dimensional spaces, providing mathematical justification for replacing explicit distance metrics with kernel functions.

> __Key Takeaways:__
>
> * __Distances concentrate around a common value__: Under the i.i.d. features assumption, all pairwise Euclidean distances grow as $\sqrt{2m\sigma^2}$ with the same leading behavior, causing the relative contrast between nearest and farthest neighbors to vanish.
> * __Concentration rate from CLT__: The central limit theorem shows that normalized squared distances cluster around $2\sigma^2$ and the spread shrinks like $1/\sqrt{m}$ as $m$ grows, so even moderate dimensionality can erode the discriminative power of Euclidean distance.
> * __Kernels preserve similarity structure__: By measuring similarity in an implicit feature space rather than computing explicit distances, kernel functions can maintain meaningful distinctions between points even when the original feature space suffers from distance concentration.

This result motivates the kernelized KNN algorithm introduced in the lecture, where kernel similarity replaces Euclidean distance as the basis for neighbor selection.

___
