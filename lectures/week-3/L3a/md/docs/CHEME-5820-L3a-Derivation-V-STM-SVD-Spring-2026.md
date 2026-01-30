# Derivation: The $\mathbf{V}$ Matrix and Steady-State Flux Patterns in SVD of Stoichiometric Matrices

When we compute the SVD of a stoichiometric matrix $\mathbf{S}$, we obtain three factors: $\mathbf{S} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$. Which columns of $\mathbf{V}$ correspond to steady-state flux patterns (reactions that produce no net metabolite accumulation)? We claimed that the columns of $\mathbf{V}$ associated with zero singular values form an orthonormal basis for all steady-state flux patterns. Where does this come from?

___

## Steady-state flux patterns and the right nullspace of $\mathbf{S}$

The stoichiometric matrix $\mathbf{S}\in\mathbb{R}^{m\times n}$ maps reaction flux vectors to metabolite net-production rates:
$$
\dot{\mathbf{x}} = \mathbf{S}\mathbf{v}
$$
where $\mathbf{x}\in\mathbb{R}^m$ is the vector of metabolite concentrations and $\mathbf{v}\in\mathbb{R}^n$ is the vector of reaction fluxes. A **steady-state flux pattern** is a set of reaction fluxes that produce no net accumulation or depletion of any metabolite. Mathematically, if $\mathbf{v}\in\mathbb{R}^n$ is a steady-state flux pattern, then $\mathbf{S}\mathbf{v} = \mathbf{0}$. At steady state:
$$
\dot{\mathbf{x}} = \mathbf{S}\mathbf{v} = \mathbf{0}
$$
This means the rate of production equals the rate of consumption for every metabolite. Steady-state flux patterns are exactly the vectors in the **right nullspace** of $\mathbf{S}$, denoted $N(\mathbf{S})$.

Biologically, steady-state flux patterns are central to metabolic flux analysis. They represent balanced metabolic states where the network operates without accumulating intermediates. For example, in a linear pathway $A \to B \to C$, a steady-state flux has all reactions proceeding at the same rate. By the rank-nullity theorem, if $\text{rank}(\mathbf{S}) = r$, then:
$$
\dim(N(\mathbf{S})) = n - r
$$
So there are exactly $n-r$ linearly independent steady-state flux patterns. The question is: how do we find them?
___

## How the SVD reveals steady-state flux patterns

The singular value decomposition factorizes the stoichiometric matrix as $\mathbf{S} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ where $\mathbf{U}\in\mathbb{R}^{m\times m}$ is orthogonal, $\mathbf{\Sigma}\in\mathbb{R}^{m\times n}$ is diagonal with nonnegative singular values $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$ followed by zeros, and $\mathbf{V}\in\mathbb{R}^{n\times n}$ is orthogonal with columns called right singular vectors.

Partition the columns of $\mathbf{V}$ based on whether the corresponding singular value is zero or nonzero:
$$
\mathbf{V} = \begin{bmatrix} \mathbf{V}_1 & \mathbf{V}_0 \end{bmatrix}
$$
where $\mathbf{V}_1\in\mathbb{R}^{n\times r}$ contains the first $r$ columns (nonzero singular values) and $\mathbf{V}_0\in\mathbb{R}^{n\times(n-r)}$ contains the remaining $n-r$ columns (corresponding to the zero singular values). We want to show that the columns of $\mathbf{V}_0$ are steady-state flux patterns, i.e., that $\mathbf{S}\mathbf{V}_0 = \mathbf{0}$. Start with the SVD factorization:
$$
\mathbf{S} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top
$$
Note that $\mathbf{\Sigma}\in\mathbb{R}^{m\times n}$ has block structure:
$$
\mathbf{\Sigma} = \begin{bmatrix} \mathbf{\Sigma}_r & \mathbf{0} \\ \mathbf{0} & \mathbf{0} \end{bmatrix}
$$
where $\mathbf{\Sigma}_r\in\mathbb{R}^{r\times r}$ is diagonal with the $r$ nonzero singular values. Now let's compute $\mathbf{S}\mathbf{V}_0$. Since the columns of $\mathbf{V}$ are orthonormal and we have partitioned $\mathbf{V} = [\mathbf{V}_1 \; \mathbf{V}_0]$:
$$
\mathbf{V}^\top\mathbf{V}_0 = \begin{bmatrix} \mathbf{V}_1^\top \\ \mathbf{V}_0^\top \end{bmatrix}\mathbf{V}_0 = \begin{bmatrix} \mathbf{V}_1^\top\mathbf{V}_0 \\ \mathbf{V}_0^\top\mathbf{V}_0 \end{bmatrix} = \begin{bmatrix} \mathbf{0}_{r\times(n-r)} \\ \mathbf{I}_{n-r} \end{bmatrix}
$$
The first block is zero because columns of $\mathbf{V}_1$ and $\mathbf{V}_0$ are orthogonal; the second block is identity because columns of $\mathbf{V}_0$ are orthonormal. Substituting into $\mathbf{S}\mathbf{V}_0$:
$$
\mathbf{S}\mathbf{V}_0 = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top\mathbf{V}_0 = \mathbf{U}\mathbf{\Sigma}\begin{bmatrix} \mathbf{0}_{r\times(n-r)} \\ \mathbf{I}_{n-r} \end{bmatrix}
$$
Multiply $\mathbf{\Sigma}$ by this partitioned matrix:
$$
\mathbf{\Sigma}\begin{bmatrix} \mathbf{0}_{r\times(n-r)} \\ \mathbf{I}_{n-r} \end{bmatrix} = \begin{bmatrix} \mathbf{\Sigma}_r & \mathbf{0} \\ \mathbf{0} & \mathbf{0} \end{bmatrix}\begin{bmatrix} \mathbf{0}_{r\times(n-r)} \\ \mathbf{I}_{n-r} \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ \mathbf{0} \end{bmatrix} = \mathbf{0}
$$
The zero blocks in $\mathbf{\Sigma}$ ensure the product vanishes. Therefore:
$$
\boxed{\mathbf{S}\mathbf{V}_0 = \mathbf{U} \cdot \mathbf{0} = \mathbf{0}}
$$
Thus, each column of $\mathbf{V}_0$ satisfies $\mathbf{S}\mathbf{v} = \mathbf{0}$, so each column is a steady-state flux pattern. Since there are $n-r$ orthonormal columns in $\mathbf{V}_0$ and $\dim(N(\mathbf{S})) = n-r$, the columns of $\mathbf{V}_0$ form an orthonormal basis for all steady-state flux patterns.

___

## Summary: Connecting SVD and Steady-State Flux Patterns

We have established the relationship between the right singular vectors and steady-state flux patterns:

**Key Results:**
1. Steady-state flux patterns are vectors $\mathbf{v}$ satisfying $\mathbf{S}\mathbf{v} = \mathbf{0}$, i.e., vectors in the right nullspace of $\mathbf{S}$
2. The columns of $\mathbf{V}_0$ (those with zero singular values) form an orthonormal basis for the right nullspace
3. Any linear combination $\mathbf{v} = \mathbf{V}_0\mathbf{a}$ is a steady-state flux pattern

**Practical Implications:**
To extract steady-state flux patterns from a stoichiometric matrix, compute the SVD and identify singular values that are numerically zero (below a tolerance $\tau = \epsilon \cdot \max(m,n) \cdot \sigma_1$ where $\epsilon \approx 10^{-15}$). The corresponding columns of $\mathbf{V}$ span all steady-state flux patterns. This approach is numerically stable and provides an orthonormal basis for flux balance analysis.

___
