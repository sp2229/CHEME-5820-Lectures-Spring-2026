# General Problem
Let’s begin with a general nonlinear optimization problem and the theory needed to establish optimality. Suppose, we want to minimize a nonlinear objective function $f(x)$ subject to equality constraints $h_j(x) = 0$ for $j = 1, \ldots, p$ and inequality constraints $g_i(x) \leq 0$ for $i = 1, \ldots, m$. The problem can be formulated as follows:
$$
\begin{align*}
    \min_{x\in\mathbb{R}^n} \; f(x)
    \quad\text{s.t.}\quad
    \begin{cases}
    g_i(x) \le 0, & i = 1,\dots,m,\\
    h_j(x) = 0, & j = 1,\dots,p,
    \end{cases}
\end{align*}
$$

### Lagrangian
For this problem, we introduce multipliers $\lambda_i\ge{0}$ for each inequality and $\nu_j$ (free) for each equality. The Lagrangian is then given by:
$$
\boxed{
\begin{align*}
\mathcal L(x,\lambda,\nu) & =\;f(x)\;+\;\sum_{i=1}^m \lambda_i\,g_i(x)\;+\;\sum_{j=1}^p \nu_j\,h_j(x) \\
\lambda_i & \ge 0\;(i=1,\dots,m)\quad\text{convention}\\
\end{align*}}
$$

### Karush-Kuhn-Tucker (KKT) Conditions
The Karush–Kuhn–Tucker (KKT) conditions play a central role in the theory and practice of constrained nonlinear optimization by generalizing the method of Lagrange multipliers to handle both equality and inequality constraints. Assuming a suitable constraint qualification (e.g., LICQ or Slater’s condition), the following are necessary for optimality:

1. __Stationarity__: The gradient of the Lagrangian with respect to $x$ must vanish at the optimal point $x^*$:
    $$
    \begin{align*}
    \nabla_x\mathcal L(x^*,\lambda^*,\nu^*) = 0\quad\Longleftrightarrow\quad\nabla f(x^*) + \sum_{i=1}^{m} \lambda_i^* \nabla g_i(x^*) + \sum_{j=1}^{p} \nu_j^* \nabla h_j(x^*) = 0.
    \end{align*}
    $$
2. __Primal feasibility__: The constraints must be satisfied at the optimal point $x^*$:
    $$    \begin{align*}
    & g_i(x^*) \le 0 \quad(i = 1, \ldots, m)\\
    & h_j(x^*) = 0 \quad(j = 1, \ldots, p).
    \end{align*}
    $$
3. **Dual feasibility**: The Lagrange multipliers for the inequality constraints must be non-negative:
    $$\lambda_i^* \ge 0 \quad(i = 1, \ldots, m).$$
4. **Complementary slackness**: For each inequality constraint, either the constraint is active (i.e., $g_i(x^*) = 0$) or the corresponding multiplier is zero ($\lambda_i^* = 0$):
    $$\lambda_i^* \cdot g_i(x^*) = 0 \quad(i = 1, \ldots, m).$$

These conditions provide a powerful framework for analyzing and solving constrained optimization problems. They are necessary for optimality under certain regularity conditions, such as the constraint qualifications (e.g., Slater's condition). If $f$ and each $g_i$ are convex and each $h_j$ is affine, then any point satisfying the KKT conditions is a global minimizer!
___


