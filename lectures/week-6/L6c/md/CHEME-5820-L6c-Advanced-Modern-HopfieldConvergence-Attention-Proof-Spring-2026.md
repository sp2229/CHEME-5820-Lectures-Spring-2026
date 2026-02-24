# Advanced: Why Modern Hopfield Networks Converge

This notebook addresses the convergence question for the modern Hopfield model used in L6c: why do continuous-state updates settle, and what does convergence mean in that setting?

This version focuses on derivation only. For computation and simulation, see the example notebook.

> __Learning Objectives:__
>
> By the end of this notebook, you should be able to:
>
> * __Derive the modern descent identity:__ Starting from the continuous-state LSE energy and the softmax update map, derive the gradient relation $\nabla g(\mathbf{s})=\mathbf{T}(\mathbf{s})$ and the local quadratic surrogate $Q(\mathbf{u}\mid\mathbf{v})$. Use that surrogate minimization step to prove the one-step decrease formula $E(\mathbf{T}(\mathbf{v}))\le E(\mathbf{v})-\tfrac12\lVert\mathbf{T}(\mathbf{v})-\mathbf{v}\rVert_2^2$.
> * __Connect descent to convergence:__ Derive an explicit lower bound for the energy using the log-sum-exp upper bound and norm inequalities, yielding $E(\mathbf{s})\ge\tfrac12(\lVert\mathbf{s}\rVert_2-M)^2\ge0$. Combine that lower bound with the telescoping sum of descent inequalities to show finite total squared step length and therefore $\lVert\mathbf{s}^{t+1}-\mathbf{s}^{t}\rVert_2\to0$.
> * __Interpret the endpoint correctly:__ Show that iterates remain in the convex hull of stored memories, so accumulation points exist by compactness. Then use continuity of $\mathbf{T}$ and $\nabla E(\mathbf{s})=\mathbf{s}-\mathbf{T}(\mathbf{s})$ to prove accumulation points are fixed/stationary, while recognizing this is an asymptotic result rather than a universal finite-step guarantee.

Let's get started!

___


## Convergence

Before we write equations, fix the narrative arc: each modern Hopfield update can be interpreted as minimizing a local surrogate of the energy. That local minimization gives one-step descent. Once descent is paired with a lower bound, asymptotic convergence follows.

> __Intuition:__
>
> The update map computes similarities between the current state and stored memories, then replaces the current state by a softmax-weighted average of those memories. The convergence proof formalizes why this replacement is energy-descending.
>
> The key endpoint is asymptotic, not finite-step: energy converges, update increments vanish, and limit points satisfy the fixed-point equation.

With that narrative in place, we state the theorem in a notationally self-contained form.

> __Convergence Theorem (Modern Hopfield, continuous-state LSE energy; self-contained notation):__
>
> Let $N,K\in\mathbb{Z}_{\geq 1}$, where $N$ is the state dimension (number of nodes/features) and $K$ is the number of stored memories. Let $\beta>0$ denote the inverse temperature parameter. Let
> $\mathbf{X}=[\mathbf{m}_1,\ldots,\mathbf{m}_K]\in\mathbb{R}^{N\times K}$ be the memory matrix, where column $\mathbf{m}_i\in\mathbb{R}^{N}$ is stored memory $i$. Define
> $M:=\max_{i=1,\ldots,K}\lVert\mathbf{m}_i\rVert_2$, i.e., $M$ is the largest memory norm.
>
> For any similarity vector $\mathbf{z}\in\mathbb{R}^{K}$ and any state $\mathbf{s}\in\mathbb{R}^{N}$, define
> $$
> \begin{aligned}
> \operatorname{lse}_{\beta}(\mathbf{z})
> &:= \frac{1}{\beta}\log\left(\sum_{i=1}^{K}e^{\beta z_i}\right),\\
> E(\mathbf{s})
> &:= -\operatorname{lse}_{\beta}(\mathbf{X}^{\top}\mathbf{s}) + \frac{1}{2}\lVert\mathbf{s}\rVert_2^2 + \frac{1}{\beta}\log K + \frac{1}{2}M^2,\\
> \mathbf{T}(\mathbf{s})
> &:= \mathbf{X}\,\operatorname{softmax}(\beta\mathbf{X}^{\top}\mathbf{s}).
> \end{aligned}
> $$
> Here $\operatorname{softmax}(\mathbf{u})_i := \frac{e^{u_i}}{\sum_{j=1}^{K}e^{u_j}}$ for $\mathbf{u}\in\mathbb{R}^{K}$ and $i=1,\ldots,K$. Also, $\operatorname{lse}_{\beta}$ is the smooth maximum over similarities, $E$ is the Lyapunov energy, and $\mathbf{T}$ is the retrieval update map.
>
> Let the iteration be $\mathbf{s}^{t+1}=\mathbf{T}(\mathbf{s}^{t})$ for $t=0,1,2,\ldots$. Then, for every $t$,
> $$
> E(\mathbf{s}^{t+1}) \le E(\mathbf{s}^{t}) - \frac{1}{2}\lVert\mathbf{s}^{t+1}-\mathbf{s}^{t}\rVert_2^2.
> $$
> Also, for every $\mathbf{s}\in\mathbb{R}^{N}$,
> $$
> E(\mathbf{s}) \ge \frac{1}{2}\left(\lVert\mathbf{s}\rVert_2 - M\right)^2 \ge 0.
> $$
> Therefore $E(\mathbf{s}^{t})$ is monotone non-increasing and convergent, and
> $$
> \lVert\mathbf{s}^{t+1}-\mathbf{s}^{t}\rVert_2 \to 0.
> $$
> Every accumulation point (that is, every subsequential limit) $\mathbf{s}^{\ast}$ satisfies
> $$
> \mathbf{T}(\mathbf{s}^{\ast})=\mathbf{s}^{\ast},
> $$
> equivalently
> $$
> \nabla E(\mathbf{s}^{\ast})=\mathbf{0}.
> $$
> This theorem is asymptotic; it does not assert universal finite-step convergence.

We begin by exposing the gradient structure behind the update map. Let
$$
g(\mathbf{s}) := \operatorname{lse}_{\beta}(\mathbf{X}^{\top}\mathbf{s}).
$$
Here $g$ is the smooth-similarity term in the energy. Then
$$
\begin{align*}
\nabla g(\mathbf{s})
&= \mathbf{X}\,\operatorname{softmax}(\beta\mathbf{X}^{\top}\mathbf{s}) && \text{differentiate lse and apply chain rule}\\
&= \mathbf{T}(\mathbf{s}) && \text{use update-map definition},\\
\nabla E(\mathbf{s})
&= \mathbf{s}-\nabla g(\mathbf{s}) && \text{differentiate the energy}\\
&= \mathbf{s}-\mathbf{T}(\mathbf{s}) && \text{substitute the previous line}.
\end{align*}
$$
Hence fixed points of $\mathbf{T}$ are exactly stationary points of $E$.

To obtain one-step descent, apply convexity of $g$ for arbitrary $\mathbf{u},\mathbf{v}\in\mathbb{R}^{N}$ and rearrange:
$$
\begin{align*}
g(\mathbf{u})
&\ge g(\mathbf{v}) + \nabla g(\mathbf{v})^{\top}(\mathbf{u}-\mathbf{v}) && \text{use convexity},\\
-g(\mathbf{u})
&\le -g(\mathbf{v}) - \nabla g(\mathbf{v})^{\top}(\mathbf{u}-\mathbf{v}) && \text{multiply both sides by }-1,\\
E(\mathbf{u})
&\le Q(\mathbf{u}\mid\mathbf{v})
:= \frac12\lVert\mathbf{u}\rVert_2^2 - \nabla g(\mathbf{v})^{\top}\mathbf{u} + c(\mathbf{v}) && \text{add the same quadratic and constants on both sides}.
\end{align*}
$$
Here $Q(\cdot\mid\mathbf{v})$ is the local quadratic surrogate at anchor $\mathbf{v}$, and $c(\mathbf{v})$ is independent of $\mathbf{u}$.

The minimizer of $Q(\cdot\mid\mathbf{v})$ and its quadratic gap are
$$
\begin{align*}
\mathbf{u}^{\ast}
&= \nabla g(\mathbf{v}) = \mathbf{T}(\mathbf{v}) && \text{set derivative of }Q\text{ to zero},\\
Q(\mathbf{u}^{\ast}\mid\mathbf{v})
&= Q(\mathbf{v}\mid\mathbf{v}) - \frac12\lVert\mathbf{u}^{\ast}-\mathbf{v}\rVert_2^2 && \text{complete the square}.
\end{align*}
$$
Combining $E(\mathbf{u}^{\ast})\le Q(\mathbf{u}^{\ast}\mid\mathbf{v})$ with $Q(\mathbf{v}\mid\mathbf{v})=E(\mathbf{v})$ gives
$$
\begin{align*}
E(\mathbf{T}(\mathbf{v}))
&\le E(\mathbf{v}) - \frac12\lVert\mathbf{T}(\mathbf{v})-\mathbf{v}\rVert_2^2 && \text{one-step decrease formula}.
\end{align*}
$$
Setting $\mathbf{v}=\mathbf{s}^{t}$ yields the theorem's one-step descent statement.

Next derive the lower bound in one continuous chain. Start with
$$
\operatorname{lse}_{\beta}(\mathbf{z}) \le \max_i z_i + \frac{1}{\beta}\log K,
$$
and substitute $\mathbf{z}=\mathbf{X}^{\top}\mathbf{s}$:
$$
\begin{align*}
E(\mathbf{s})
&= -\operatorname{lse}_{\beta}(\mathbf{X}^{\top}\mathbf{s}) + \frac12\lVert\mathbf{s}\rVert_2^2 + \frac{1}{\beta}\log K + \frac12 M^2 && \text{definition of }E,\\
&\ge -\max_i \mathbf{m}_i^{\top}\mathbf{s} + \frac12\lVert\mathbf{s}\rVert_2^2 + \frac12 M^2 && \text{replace lse with its upper bound},\\
&\ge -M\lVert\mathbf{s}\rVert_2 + \frac12\lVert\mathbf{s}\rVert_2^2 + \frac12 M^2 && \text{bound dot product by product of norms},\\
&= \frac12\left(\lVert\mathbf{s}\rVert_2 - M\right)^2 \ge 0 && \text{complete the square}.
\end{align*}
$$
Hence energy is lower-bounded and, by descent, $E(\mathbf{s}^{t})$ converges.

The same descent inequality also gives vanishing increments through cancellation of middle terms:
$$
\begin{align*}
\frac12\sum_{t=0}^{T-1}\lVert\mathbf{s}^{t+1}-\mathbf{s}^{t}\rVert_2^2
&\le \sum_{t=0}^{T-1}\big(E(\mathbf{s}^{t})-E(\mathbf{s}^{t+1})\big) && \text{sum one-step decrease inequalities},\\
&= E(\mathbf{s}^{0})-E(\mathbf{s}^{T}) && \text{middle terms cancel},\\
&\le E(\mathbf{s}^{0}) && \text{use }E(\mathbf{s}^{T})\ge 0.
\end{align*}
$$
Letting $T\to\infty$, the series $\sum_t\lVert\mathbf{s}^{t+1}-\mathbf{s}^{t}\rVert_2^2$ is finite, so
$\lVert\mathbf{s}^{t+1}-\mathbf{s}^{t}\rVert_2\to0$.

Finally, classify the endpoints. For $t\ge1$, $\mathbf{s}^{t}=\mathbf{X}\mathbf{p}^{t}$ with
$\mathbf{p}^{t}:=\operatorname{softmax}(\beta\mathbf{X}^{\top}\mathbf{s}^{t-1})$, so $\mathbf{p}^{t}$ is a probability vector (nonnegative entries summing to one). Therefore
$\mathbf{s}^{t}\in\mathrm{conv}\{\mathbf{m}_1,\ldots,\mathbf{m}_K\}$, the convex hull of stored memories, which is compact, so accumulation points exist.

Take a convergent subsequence $\mathbf{s}^{t_n}\to\mathbf{s}^{\ast}$. Since increments vanish,
$\mathbf{s}^{t_n+1}-\mathbf{s}^{t_n}\to0$, hence $\mathbf{s}^{t_n+1}\to\mathbf{s}^{\ast}$. Continuity of $\mathbf{T}$ gives
$$
\begin{align*}
\mathbf{T}(\mathbf{s}^{\ast})
&= \lim_{n\to\infty}\mathbf{T}(\mathbf{s}^{t_n}) && \text{use continuity of update map},\\
&= \lim_{n\to\infty}\mathbf{s}^{t_n+1} && \text{replace using iteration rule},\\
&= \mathbf{s}^{\ast} && \text{shifted sequence has same limit}.
\end{align*}
$$
Therefore every accumulation point is fixed, and therefore stationary by
$\nabla E(\mathbf{s})=\mathbf{s}-\mathbf{T}(\mathbf{s})$.

> __Proof conclusion:__
>
> Modern Hopfield retrieval is an energy-descent process for the continuous LSE model. The descent gap controls step size, which forces vanishing increments, and every accumulation point is a fixed/stationary state.

___


## Summary

This notebook derives a proof-only convergence result for the modern Hopfield model used in L6c.

> __Key Takeaways:__
>
> * __Update rule and energy geometry are tightly coupled:__ In this model, the softmax-based update map is exactly the gradient of the smooth similarity term, which lets us rewrite $\nabla E(\mathbf{s})$ as $\mathbf{s}-\mathbf{T}(\mathbf{s})$. That identity is what makes the surrogate-minimization argument precise and produces a quantitative one-step descent inequality.
> * __Asymptotic settling follows from descent plus a lower bound:__ The proof gives both a strict per-step decrease term and an explicit global lower bound on energy, so the sequence cannot decrease forever without control. Summing the descent inequalities yields finite cumulative squared step size, which forces update increments to vanish.
> * __Convergence means fixed/stationary accumulation points, not finite-step completion:__ Because iterates are convex combinations of memories, trajectories stay in a compact region and subsequential limits exist. Every such limit satisfies $\mathbf{T}(\mathbf{s}^{\ast})=\mathbf{s}^{\ast}$ (equivalently $\nabla E(\mathbf{s}^{\ast})=\mathbf{0}$), which is the endpoint guarantee established in this notebook.

___


## References
1. Ramsauer H, Schafl B, Lehner J, et al. (2021), *Hopfield Networks is All You Need*, ICLR 2021.
   - arXiv: https://arxiv.org/abs/2008.02217
   - Local copy verified in this repo: `lectures/week-6/L6c/docs/Ramsauer-HNAYN-2021.pdf`
   - Used here for: continuous-state energy definition (Eq. (2)), update rule (Eq. (3)), and convergence statements (Theorem 1 and Theorem 2).

2. Krotov D, Hopfield JJ (2021), *Large Associative Memory Problem in Neurobiology and Machine Learning*, ICLR 2021.
   - arXiv: https://arxiv.org/abs/2008.06996
   - Local copy verified in this repo: `lectures/week-6/L6c/docs/Krotov-Hopfield-2021.pdf`
   - Used here for: Lyapunov/energy-minimization interpretation of modern associative-memory dynamics.

3. Krotov D, Hopfield JJ (2016), *Dense Associative Memory for Pattern Recognition*, NeurIPS 2016.
   - arXiv: https://arxiv.org/abs/1606.01164
   - Local copy verified in this repo: `lectures/week-6/L6c/docs/Krotov-Hopfield-2016.pdf`
   - Used here for: historical transition from classical quadratic energies to higher-order modern associative-memory energies.

4. Demircigil M, Heusel J, Lowe M, Upgang S, Vermet F (2017), *On a Model of Associative Memory with Huge Storage Capacity*, Journal of Statistical Physics.
   - arXiv: https://arxiv.org/abs/1702.01929
   - Local copy verified in this repo: `lectures/week-6/L6c/docs/Vermet-2017.pdf`
   - Used here for: exponential-capacity context for exponential-interaction associative-memory models.

____
