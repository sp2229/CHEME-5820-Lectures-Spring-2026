# Gradient Descent
Given the general problem of constrained nonlinear optimization, we can apply _gradient descent_ to find a solution. 

Gradient descent is a first-order optimization algorithm that iteratively updates the solution by moving in the direction of the steepest descent of the objective function. However, by default, gradient descent is designed for unconstrained optimization problems. Thus, we need to adapt it to handle constraints.

Toward this end, we can use a barrier or penalty method to incorporate the constraints into the objective function.

> __Penalty and Barrier Methods__: A penalty method involves adding a penalty term to the objective function for violating the constraints. There are several approaches to this, but one common approach is to use a quadratic penalty function for equality constraints and a __barrier function__ for inequality constraints. When a candidate solution violates the constraints, the penalty term increases, discouraging such violations in future iterations.

Consider the following (augmented) objective function that combines the original objective function with penalty terms:
$$
\begin{align*}
    \min_{x\in\mathbb{R}^n}\;P_{\mu,\rho}(x)\;&=f(x)\;-\;\underbrace{\frac{1}{\mu}\sum_{i=1}^m\ln\bigl(-\,g_i(x)\bigr)}_{\text{barrier term}}\;+\;\underbrace{\frac{1}{2\rho}\sum_{j=1}^p 
    \bigl[h_j(x)\bigr]^2}_{\text{penalty term}},\quad\text{where}\quad\mu>0,\;\rho>0\\
\end{align*}
$$
The smooth barrier terms $\frac{1}{\mu}\sum_{i=1}^m\ln\bigl(-\,g_i(x)\bigr)$ penalize violations of the inequality constraints, while the $\frac{1}{2\rho}\sum_{j=1}^p\bigl[h_j(x)\bigr]^2$ terms penalize violations of the equality constraints. 

> __Parameters__
>
> The parameters $\mu$ and $\rho$ control the strength of these penalties:
> * __Barrier weight__: The $\mu$ parameter is typically _decreased_ over iterations to enforce stricter adherence to the inequality constraints. As $\mu\to0$, the coefficient $\frac{1}{\mu}$ grows, strengthening the barrier effect.
> * __Penalty weight__: The $\rho$ parameter is typically _decreased_ over iterations to enforce stricter adherence to the equality constraints. As $\rho\to0$, the coefficient $\frac{1}{\rho}$ grows, strengthening the penalty effect.
> 
> We'll provide some heuristics for updating these parameters in the algorithm below.

### Algorithm
Let's develop a simple gradient descent algorithm for this problem. The algorithm iteratively updates the solution $x_k$ using the gradient of the augmented objective function $P_{\mu,\rho}(x)$.

__Initialization__: Given an initial guess $x_0$, set $\mu > 0$ and $\rho > 0$. Specify a tolerance $\epsilon > 0$, a maximum number of iterations $K$, and a step size (learning rate) $\alpha > 0$. Set $\texttt{converged} \gets \texttt{false}$, the iteration counter to $k \gets 0$ and specify values for the penalty update parameters $(\tau_{\mu},\tau_{\rho})\in\left(0,1\right)$.

While not $\texttt{converged}$ __do__:
1. Compute the gradient: $\nabla P_{\mu,\rho}(x_k) = \nabla f(x_k) + \frac{1}{\mu} \sum_{i=1}^m \frac{\nabla g_i(x_k)}{-g_i(x_k)} + \frac{1}{\rho} \sum_{j=1}^p h_j(x_k) \nabla h_j(x_k)$ evaluated at the current solution $x_k$.
2. Update the solution: $x_{k+1} = x_k - \alpha \nabla P_{\mu,\rho}(x_k)$. $\texttt{Note}$: $\alpha$ is fixed here, but it can be adapted dynamically based on the convergence behavior.
3. Check convergence: 
     - If $\|x_{k+1} - x_k\|_{2} \leq \epsilon$, set $\texttt{converged} \gets \texttt{true}$. Return $x_{k+1}$ as the approximate solution. $\texttt{Note}$: here we look at the Euclidean norm of the difference between the current and next solution. However, many other criteria can be used, such as the change in the objective function value or the gradient norm.
     - If $k \geq K$, set $\texttt{converged} \gets \texttt{true}$. Warn that the maximum number of iterations has been reached without convergence.
4. Increment the iteration counter: $k \gets k + 1$, update $\mu\gets \tau_\mu\,\mu$ and $\rho\gets \tau_\rho\,\rho$ as needed, and repeat.

As $\mu\to0$, the coefficient $\frac{1}{\mu}$ in the barrier term grows, creating an increasingly strong barrier that keeps the solution away from constraint boundaries (where $g_i(x)\to 0^-$). Similarly, as $\rho\to0$, the coefficient $\frac{1}{\rho}$ in the penalty term grows, enforcing $h_j(x)\to0$ ever more strictly.
___