# Derivation: The $\mathbf{U}$ Matrix and Conservation Relations in SVD of Stoichiometric Matrices

When we compute the SVD of a stoichiometric matrix $\mathbf{S}$, we obtain three factors: $\mathbf{S} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$. Which columns of $\mathbf{U}$ correspond to conservation relations? The columns of $\mathbf{U}$ associated with zero singular values form an orthonormal basis for all conservation relations in the metabolic network. Where does this come from?

___

## Conservation relations and the left nullspace of $\mathbf{S}$

The stoichiometric matrix $\mathbf{S}\in\mathbb{R}^{m\times n}$ maps reaction flux vectors to metabolite net-production rates:
$$
\dot{\mathbf{x}} = \mathbf{S}\mathbf{v}
$$
where $\mathbf{x}\in\mathbb{R}^m$ is the vector of metabolite concentrations and $\mathbf{v}\in\mathbb{R}^n$ is the vector of reaction fluxes. A **conservation relation** is a weighted sum of metabolites whose total amount never changes in time. Mathematically, if $\mathbf{w}\in\mathbb{R}^m$ defines such a weighted sum, then $\mathbf{w}^\top\mathbf{x}(t)$ must be constant for all $t$. Taking the time derivative:
$$
\frac{d}{dt}(\mathbf{w}^\top\mathbf{x}) = \mathbf{w}^\top\dot{\mathbf{x}} = \mathbf{w}^\top(\mathbf{S}\mathbf{v}) = (\mathbf{w}^\top\mathbf{S})\mathbf{v} = (\mathbf{S}^\top\mathbf{w})^\top\mathbf{v}
$$
For this to equal zero for all possible flux vectors $\mathbf{v}$, we need $\mathbf{S}^\top\mathbf{w} = \mathbf{0}$. Conservation relations are exactly the vectors in the **left nullspace** of $\mathbf{S}$, denoted $N(\mathbf{S}^\top)$.Biologically, conservation relations correspond to pools like adenylate ($\text{ATP} + \text{ADP} + \text{AMP}$), phosphate, and redox cofactors ($\text{NAD}^+ + \text{NADH}$). By the rank-nullity theorem, if $\text{rank}(\mathbf{S}) = r$, then:
$$
\dim(N(\mathbf{S}^\top)) = m - r
$$
So there are exactly $m-r$ linearly independent conservation relations. The question is: how do we find them?

___

## How the SVD reveals conservation relations

The singular value decomposition factorizes the stoichiometrix matrix $\mathbf{S} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ where $\mathbf{U}\in\mathbb{R}^{m\times m}$ is orthogonal with columns called left singular vectors, $\mathbf{\Sigma}\in\mathbb{R}^{m\times n}$ is diagonal with nonnegative singular values $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$ followed by zeros, and $\mathbf{V}\in\mathbb{R}^{n\times n}$ is orthogonal. Partition the columns of $\mathbf{U}$ based on whether the corresponding singular value is zero or nonzero gives:
$$
\mathbf{U} = \begin{bmatrix} \mathbf{U}_1 & \mathbf{U}_0 \end{bmatrix}
$$
where $\mathbf{U}_1\in\mathbb{R}^{m\times r}$ contains the first $r$ columns (nonzero singular values) and $\mathbf{U}_0\in\mathbb{R}^{m\times(m-r)}$ contains the remaining $m-r$ columns (zero singular values). We want to show that the columns of $\mathbf{U}_0$ are conservation relations, i.e., that $\mathbf{S}^\top\mathbf{U}_0 = \mathbf{0}$. Start by transposing the SVD:
$$
\mathbf{S}^\top = (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top)^\top = \mathbf{V}\mathbf{\Sigma}^\top\mathbf{U}^\top
$$
Note that $\mathbf{\Sigma}^\top\in\mathbb{R}^{n\times m}$ has block structure:
$$
\mathbf{\Sigma}^\top = \begin{bmatrix} \mathbf{\Sigma}_r & \mathbf{0} \\ \mathbf{0} & \mathbf{0} \end{bmatrix}
$$
where $\mathbf{\Sigma}_r\in\mathbb{R}^{r\times r}$ is diagonal with the $r$ nonzero singular values. Now let's compute $\mathbf{S}^\top\mathbf{U}_0$. Since the columns of $\mathbf{U}$ are orthonormal and we have partitioned $\mathbf{U} = [\mathbf{U}_1 \; \mathbf{U}_0]$:
$$
\mathbf{U}^\top\mathbf{U}_0 = \begin{bmatrix} \mathbf{U}_1^\top \\ \mathbf{U}_0^\top \end{bmatrix}\mathbf{U}_0 = \begin{bmatrix} \mathbf{U}_1^\top\mathbf{U}_0 \\ \mathbf{U}_0^\top\mathbf{U}_0 \end{bmatrix} = \begin{bmatrix} \mathbf{0}_{r\times(m-r)} \\ \mathbf{I}_{m-r} \end{bmatrix}
$$
The first block is zero because columns of $\mathbf{U}_1$ and $\mathbf{U}_0$ are orthogonal; the second block is identity because columns of $\mathbf{U}_0$ are orthonormal. Substituting into $\mathbf{S}^\top\mathbf{U}_0$:
$$
\mathbf{S}^\top\mathbf{U}_0 = \mathbf{V}\mathbf{\Sigma}^\top\mathbf{U}^\top\mathbf{U}_0 = \mathbf{V}\mathbf{\Sigma}^\top\begin{bmatrix} \mathbf{0}_{r\times(m-r)} \\ \mathbf{I}_{m-r} \end{bmatrix}
$$

Multiply $\mathbf{V}\mathbf{\Sigma}^\top$ by this partitioned matrix:
$$
\mathbf{V}\mathbf{\Sigma}^\top\begin{bmatrix} \mathbf{0}_{r\times(m-r)} \\ \mathbf{I}_{m-r} \end{bmatrix} = \mathbf{V}\begin{bmatrix} \mathbf{\Sigma}_r & \mathbf{0} \\ \mathbf{0} & \mathbf{0} \end{bmatrix}\begin{bmatrix} \mathbf{0}_{r\times(m-r)} \\ \mathbf{I}_{m-r} \end{bmatrix} = \mathbf{V}\begin{bmatrix} \mathbf{0} \\ \mathbf{0} \end{bmatrix} = \mathbf{0}
$$
The zero blocks in $\mathbf{\Sigma}^\top$ ensure the product vanishes. Therefore:
$$
\boxed{\mathbf{S}^\top\mathbf{U}_0 = \mathbf{0}\quad\blacksquare}
$$
Each column of $\mathbf{U}_0$ satisfies $\mathbf{S}^\top\mathbf{u} = \mathbf{0}$, so each column is a conservation relation. Since there are $m-r$ orthonormal columns in $\mathbf{U}_0$ and $\dim(N(\mathbf{S}^\top)) = m-r$, the columns of $\mathbf{U}_0$ form an orthonormal basis for all conservation relations.

___

## Summary: Connecting SVD and Conservation Relations

We have established the relationship between the left singular vectors and conservation relations:

> **Key Takeaways:**
>
> * Conservation relations are vectors $\mathbf{w}$ satisfying $\mathbf{S}^\top\mathbf{w} = \mathbf{0}$, i.e., vectors in the left nullspace of $\mathbf{S}$
> * The columns of $\mathbf{U}_0$ (those corresponding to zero singular values) form an orthonormal basis for the left nullspace, i.e., represent all conservation relations
> * Any linear combination $\mathbf{w} = \mathbf{U}_0\mathbf{a}$ is a conservation relation, where $\mathbf{a}\in\mathbb{R}^{(m-r)}$ is an arbitrary coefficient vector (non-zero $\mathbf{a}$ gives a non-zero conservation relation)

**Practical Implications:**
To extract conservation relations from a stoichiometric matrix, compute the SVD and identify singular values that are numerically zero (below a tolerance $\tau = \epsilon \cdot \max(m,n) \cdot \sigma_1$ where $\epsilon \approx 10^{-15}$). The corresponding columns of $\mathbf{U}$ span all conservation relations. This approach is numerically stable and requires no prior knowledge of which metabolite pools are conserved.

___


