# Advanced: Why Classical Hopfield Networks Converge and Why Hebbian Learning Works

This notebook answers a central question in associative memory: why does a classical Hopfield network settle, and when does it settle on the memory we intended?

This version focuses on derivation only. For the computational checks, see [Advanced Numerical: Verifying Classical Hopfield Convergence and Hebbian Recall Limits](CHEME-5820-L6a-Advanced-Classical-HopfieldConvergence-NumericalVerification-Spring-2026.ipynb).


> __Learning Objectives:__
> 
> By the end of this notebook, you should be able to:
>
> * __Derive one-step energy change:__ Show exactly where $\Delta E$ comes from, where $\Delta E$ denotes the one-step change in network energy after an asynchronous update. Track each algebraic simplification from the full energy difference to the compact local-field form.
> * __Connect descent to convergence:__ Explain why finite state space plus lower-bounded energy implies convergence in finite time. Distinguish clearly between non-increasing updates and strictly decreasing accepted flips.
> * __Interpret Hebbian recall limits:__ Derive the signal-crosstalk decomposition and use it to explain why higher load $\alpha=K/N$ increases interference and retrieval risk, where $K$ is the number of stored patterns and $N$ is the number of neurons.


Let's get started!

___


## Convergence

Before we write equations, fix the narrative arc: asynchronous updates create descent, and Hebbian learning shapes where that descent ends.

> __Intuition first:__
>
> Before writing formal algebra, it helps to picture the network as a system moving on an energy landscape. Each asynchronous update asks whether changing one bit moves the state downhill.
>
> * __Why convergence should happen__: If every accepted update lowers one global scalar energy, the network can only move downhill or stay flat. Because the number of binary states is finite and the energy is bounded below, descent cannot continue forever and the dynamics must eventually stop.
> * __Why memory retrieval can fail__: Stopping does not guarantee that the endpoint is the memory we intended to retrieve. Interference between stored patterns can create competing local minima, so the same descent process can converge to a spurious or wrong attractor; here, a spurious attractor means a stable state that is not one of the stored memories.
>
> Now convert the energy-landscape intuition into the Convergence Theorem and prove the one-step energy-change result step by step.


Before we formalize the claim, let's fix notation so each symbol has one clear meaning.

> __Notation guide (used throughout):__
>
> Neuron indices $i,j,k\in\{1,\dots,N\}$ refer to positions in the network state vector, where $N$ is the number of neurons. Memory indices $\mu\in\{1,\dots,K\}$ label stored patterns, and $\nu\in\{1,\dots,K\}$ denotes one fixed target pattern being analyzed.
>
> We write $s_i$ for the current state of neuron $i$ and $s_i'$ for its state after one asynchronous update; an __accepted flip__ means $s_i'\neq s_i$. We use a tie-preserving sign update: $\mathrm{sign}(x)=+1$ for $x>0$, $\mathrm{sign}(x)=-1$ for $x<0$, and if $x=0$ we keep the current state unchanged.
>
> The load is $\alpha=K/N$. The crosstalk term $\eta_i^{\nu}$ and stability margin $m_i^{\nu}$ are defined explicitly in the Hebbian theorem below.

__Plain-language mapping for memory symbols:__ The stored-memory set is $\{\boldsymbol{\xi}^{\mu}\}_{\mu=1}^{K}$, i.e., the full collection of patterns used to build the weights. The memory/cue you provide during retrieval is the initial state $\mathbf{s}^{(0)}$ (often a noisy version of one stored pattern). A single stored pattern chosen for stability analysis is written $\boldsymbol{\xi}^{\nu}$; in that test we set $\mathbf{s}=\boldsymbol{\xi}^{\nu}$ and ask whether it is a fixed point.

With notation fixed, we can now state the formal convergence claim.

> __Convergence Theorem (Classical Hopfield, asynchronous updates):__
>
> Let $\mathbf{s}\in\{-1,+1\}^{N}$, assume $w_{ij}=w_{ji}$ and $w_{ii}=0$, and define the local field and energy as:
> $$
> \begin{align*}
> h_i(\mathbf{s}) &= \sum_{j=1}^{N} w_{ij}s_j + b_i\\
> E(\mathbf{s}) &= -\frac{1}{2}\sum_{j=1}^{N}\sum_{k=1}^{N} w_{jk}s_js_k - \sum_{j=1}^{N} b_j s_j.
> \end{align*}
> $$
> Here, $h_i(\mathbf{s})$ is the local field at neuron $i$, i.e., the weighted input plus bias evaluated at state $\mathbf{s}$. $E(\mathbf{s})$ is the global Hopfield energy at state $\mathbf{s}$, where lower values correspond to more stable configurations.
> In this notation, $w_{ij}$ is the connection weight from neuron $j$ to neuron $i$, and $b_i$ is the bias (threshold) term for neuron $i$.
> If one asynchronous update changes only neuron $i$ via $s_i' = \mathrm{sign}(h_i(\mathbf{s}))$ (and $s_j'=s_j$ for $j\neq i$), the one-step energy change is given by:
> $$
> \Delta E = E(\mathbf{s}')-E(\mathbf{s}) = -(s_i' - s_i)h_i(\mathbf{s}).
> $$
> Using the identity $\Delta E = E(\mathbf{s}')-E(\mathbf{s}) = -(s_i' - s_i)h_i(\mathbf{s})$, we obtain:
> * __Descent behavior__: $\Delta E = E(\mathbf{s}')-E(\mathbf{s}) \le 0$ for every asynchronous update, so energy never increases from one step to the next. Moreover, if $s_i'\neq s_i$, then $\Delta E < 0$, so accepted flips produce strict progress.
> * __Finite-time convergence__: repeated asynchronous updates reach a fixed point in finitely many accepted flips. Here, a fixed point means a state vector that no longer changes under subsequent asynchronous sign updates.
> 
> To prove the Convergence Theorem, we now derive the one-step energy change exactly.

Start from the one-step energy-change definition, $\Delta E = E(\mathbf{s}') - E(\mathbf{s})$, and substitute the explicit Hopfield energy $E(\mathbf{s})$ from the theorem before simplifying:
$$
\begin{align*}
\Delta E &= E(\mathbf{s}') - E(\mathbf{s}) && \text{definition of one-step energy change}\\
&= \left[-\frac{1}{2}\sum_{j=1}^{N}\sum_{k=1}^{N} w_{jk}s_j's_k' - \sum_{j=1}^{N} b_j s_j'\right] - \left[-\frac{1}{2}\sum_{j=1}^{N}\sum_{k=1}^{N} w_{jk}s_js_k - \sum_{j=1}^{N} b_j s_j\right] && \text{substitute }E(\mathbf{s}')\text{ and }E(\mathbf{s})\\
&= -\frac{1}{2}\sum_{j=1}^{N}\sum_{k=1}^{N} w_{jk}s_j's_k' + \frac{1}{2}\sum_{j=1}^{N}\sum_{k=1}^{N} w_{jk}s_js_k - \sum_{j=1}^{N} b_j(s_j' - s_j) && \text{distribute signs and group bias terms}\\
&= -\frac{1}{2}\sum_{j=1}^{N}\sum_{k=1}^{N} w_{jk}(s_j's_k' - s_js_k) - \sum_{j=1}^{N} b_j(s_j' - s_j) && \text{combine pairwise energy terms}.
\end{align*}
$$

Because an asynchronous update changes only neuron $i$ (so $s_j'=s_j$ for all $j\neq i$), every term with $j,k\neq i$ cancels in $s_j's_k' - s_js_k$. Keep only terms that contain index $i$:
$$
\begin{align*}
\Delta E &= -\frac{1}{2}\sum_{k\neq i} w_{ik}(s_i's_k - s_is_k) - \frac{1}{2}\sum_{j\neq i} w_{ji}(s_js_i' - s_js_i) - b_i(s_i' - s_i) && \text{only terms containing index }i\text{ survive}\\
&= -\frac{1}{2}(s_i' - s_i)\sum_{k\neq i} w_{ik}s_k - \frac{1}{2}(s_i' - s_i)\sum_{j\neq i} w_{ji}s_j - b_i(s_i' - s_i) && \text{factor }(s_i' - s_i)\text{ from each term}.
\end{align*}
$$

Use symmetry $w_{ji}=w_{ij}$ and combine the two sums:
$$
\begin{align*}
\Delta E &= -(s_i' - s_i)\sum_{j\neq i} w_{ij}s_j - b_i(s_i' - s_i) && \text{use symmetry }w_{ji}=w_{ij}\text{ and merge sums}\\
&= -(s_i' - s_i)\left(\sum_{j=1}^{N} w_{ij}s_j + b_i\right)\qquad (w_{ii}=0) && \text{extend }\sum_{j\neq i}\text{ to }\sum_{j=1}^{N}\\
&= -(s_i' - s_i)h_i(\mathbf{s}) && \text{identify the local field }h_i(\mathbf{s}).
\end{align*}
$$

We now use the closed-form identity $\Delta E = -(s_i' - s_i)h_i(\mathbf{s})$ as the reference expression. To make the cancellation concrete, verify the same result in a fully indexed $N=4$ case.

> __Example:__
>
> Consider neuron $i=2$ in a network with $N=4$ nodes. Starting from $\Delta E := E(\mathbf{s}')-E(\mathbf{s})$ and substituting the explicit Hopfield energy expression, we obtain:
> $$
> \begin{align*}
> \Delta E &= -\frac{1}{2}\sum_{j=1}^{4}\sum_{k=1}^{4} w_{jk}(s_j's_k' - s_js_k) - \sum_{j=1}^{4} b_j(s_j' - s_j).
> \end{align*}
> $$
>
> Only neuron 2 changes (asynchronous update), so $s_1'=s_1$, $s_3'=s_3$, and $s_4'=s_4$. The surviving terms are given by:
> $$
> \begin{align*}
> \Delta E &= -\frac{1}{2}\Big[
> w_{21}(s_2's_1 - s_2s_1) + w_{23}(s_2's_3 - s_2s_3) + w_{24}(s_2's_4 - s_2s_4)\\
> &\qquad\quad + w_{12}(s_1s_2' - s_1s_2) + w_{32}(s_3s_2' - s_3s_2) + w_{42}(s_4s_2' - s_4s_2)\Big] - b_2(s_2' - s_2).
> \end{align*}
> $$
>
> Factor $(s_2' - s_2)$ and use symmetry of the weight matrix, $w_{jk}=w_{kj}$:
> $$
> \begin{align*}
> \Delta E &= -\frac{1}{2}(s_2' - s_2)\left[(w_{21}+w_{12})s_1 + (w_{23}+w_{32})s_3 + (w_{24}+w_{42})s_4\right] - b_2(s_2' - s_2)\\
> &= -(s_2' - s_2)(w_{21}s_1 + w_{23}s_3 + w_{24}s_4 + b_2)\\
> &= -(s_2' - s_2)h_2(\mathbf{s})\quad\blacksquare
> \end{align*}
> $$
>
> The indexed $N=4$ expansion reproduces the general formula exactly.

With the $N=4$ cancellation shown explicitly, return to general $N$ and close the convergence proof.

> __Proof conclusion:__
>
> * __Update-level descent__: Under the asynchronous rule $s_i' = \mathrm{sign}(h_i(\mathbf{s}))$, either $s_i'=s_i$ (so $\Delta E=0$) or $s_i'\neq s_i$ with the same sign as $h_i(\mathbf{s})$, which gives $(s_i'-s_i)h_i(\mathbf{s})>0$ and therefore $\Delta E<0$. This means every asynchronous step is energy non-increasing, and every accepted flip is strictly descending.
> * __Finite-time stopping__: The state space has size $2^N$, and the energy is bounded below. One explicit lower bound is:
> $$
> E(\mathbf{s}) \ge -\frac{1}{2}\sum_{j,k}|w_{jk}| - \sum_j |b_j| > -\infty.
> $$
> Because strict descent cannot continue forever in a finite state space with a lower-bounded energy, the trajectory reaches a fixed point in finitely many accepted flips.


## Retrieving a Stored Memory
Now we ask the second question: why does convergence often land on a stored memory?
In retrieval language, the user-provided input is the cue state $\mathbf{s}^{(0)}$, while $\{\boldsymbol{\xi}^{\mu}\}_{\mu=1}^{K}$ denotes all memories encoded in the weights.
Typically, $\mathbf{s}^{(0)}$ is not perfect: it is a noisy or incomplete cue that overlaps with one stored pattern but does not exactly match it.
A relatable example is the "tip-of-the-tongue" moment for a movie title: you remember fragments (an actor, one scene, maybe the soundtrack), and those fragments act like $\mathbf{s}^{(0)}$; Hopfield retrieval is the iterative process that uses those partial features to settle on a full stored memory pattern.

> __Hebbian Retrieval Theorem (signal-crosstalk form):__
>
> __Notation in this theorem:__ $\{\boldsymbol{\xi}^{\mu}\}_{\mu=1}^{K}$ denotes the full set of stored memories encoded in the weights; $\mu$ is the running memory index in sums; $\nu$ is one fixed stored-memory index whose stability is tested; and $\mathbf{s}^{(0)}$ is the retrieval cue (input state) given before updates.
>
> During retrieval, the network evolves from $\mathbf{s}^{(0)}$ through states $\mathbf{s}(t)$. The quantity $h_i(\boldsymbol{\xi}^{\nu})$ below is not the unknown final answer; it is an offline stability check performed at a candidate stored pattern $\boldsymbol{\xi}^{\nu}$.
>
> Let $\{\boldsymbol{\xi}^{\mu}\}_{\mu=1}^{K}$ be binary memories with $\xi_i^{\mu}\in\{-1,+1\}$, and define Hebbian weights by:
> $$
> w_{ij}=\frac{1}{N}\sum_{\mu=1}^{K}\xi_i^{\mu}\xi_j^{\mu},\quad i\neq j,\qquad w_{ii}=0.
> $$
> For stored pattern index $\nu$, evaluate the local field at the candidate stored state $\mathbf{s}=\boldsymbol{\xi}^{\nu}$. We write this as $h_i := h_i(\boldsymbol{\xi}^{\nu})$ for neuron $i$, and the local field is given by:
> $$
> h_i = \underbrace{\frac{N-1}{N}\xi_i^{\nu}}_{\text{signal}} + \underbrace{\frac{1}{N}\sum_{\mu\neq\nu}\xi_i^{\mu}\sum_{j\neq i}\xi_j^{\mu}\xi_j^{\nu}}_{\text{crosstalk }\eta_i^{\nu}}.
> $$
> Therefore the bit-stability margin (the signed alignment between candidate stored bit $\xi_i^{\nu}$ and its local field) is given by:
> $$
> m_i^{\nu}=\xi_i^{\nu}h_i = \frac{N-1}{N}+\xi_i^{\nu}\eta_i^{\nu}.
> $$
> If $m_i^{\nu}>0$ for all $i$, then $\boldsymbol{\xi}^{\nu}$ is stable under asynchronous sign updates.

To prove the Hebbian Retrieval Theorem, start from the local-field definition evaluated at a candidate stored pattern, $h_i = \sum_{j\neq i} w_{ij}\xi_j^{\nu}$, substitute the Hebbian weights $w_{ij}=\frac{1}{N}\sum_{\mu=1}^{K}\xi_i^{\mu}\xi_j^{\mu}$, and then separate the $\mu=\nu$ signal term from the $\mu\neq\nu$ crosstalk terms:
$$
\begin{align*}
h_i &= \sum_{j\neq i} w_{ij}\xi_j^{\nu} && \text{local field at neuron }i\text{ for the stability-check state }\mathbf{s}=\boldsymbol{\xi}^{\nu}\\
&= \sum_{j\neq i}\left(\frac{1}{N}\sum_{\mu=1}^{K}\xi_i^{\mu}\xi_j^{\mu}\right)\xi_j^{\nu} && \text{substitute Hebbian weights }w_{ij}\\
&= \frac{1}{N}\sum_{\mu=1}^{K}\xi_i^{\mu}\sum_{j\neq i}\xi_j^{\mu}\xi_j^{\nu} && \text{swap order of finite sums}\\
&= \frac{1}{N}\xi_i^{\nu}\sum_{j\neq i}(\xi_j^{\nu})^2 + \frac{1}{N}\sum_{\mu\neq\nu}\xi_i^{\mu}\sum_{j\neq i}\xi_j^{\mu}\xi_j^{\nu} && \text{separate }\mu=\nu\text{ (signal) from }\mu\neq\nu\text{ (crosstalk)}\\
&= \underbrace{\frac{N-1}{N}\xi_i^{\nu}}_{\text{signal}} + \underbrace{\frac{1}{N}\sum_{\mu\neq\nu}\xi_i^{\mu}\sum_{j\neq i}\xi_j^{\mu}\xi_j^{\nu}}_{\text{crosstalk }\eta_i^{\nu}} && \text{use }(\xi_j^{\nu})^2=1\text{ for binary bits}.
\end{align*}
$$

Next, define the stability margin by multiplying the local field by the target bit, $m_i^{\nu}:=\xi_i^{\nu}h_i$:
$$
\begin{align*}
m_i^{\nu} &= \xi_i^{\nu}h_i && \text{definition of bit-stability margin}\\
&= \frac{N-1}{N} + \xi_i^{\nu}\eta_i^{\nu} && \text{substitute the signal-crosstalk form of }h_i.
\end{align*}
$$

> __Proof conclusion:__
>
> The first term is an aligned retrieval signal from the candidate stored pattern $\boldsymbol{\xi}^{\nu}$, while the second term is interference from all other stored patterns. This decomposition explains why low load favors correct recall and high load increases the chance of spurious or wrong attractors.

For a direct simulation check of these results, continue to [Advanced Numerical: Verifying Classical Hopfield Convergence and Hebbian Recall Limits](CHEME-5820-L6a-Advanced-Classical-HopfieldConvergence-NumericalVerification-Spring-2026.ipynb).

___

## Summary


This notebook derives the core convergence and recall-limit results for classical Hopfield networks using algebraic identities from the energy function and the Hebbian weight definition.


> __Key takeaways:__
>
> * __One-step descent__: The derivation shows $\Delta E = -(s_i' - s_i)h_i(\mathbf{s})$, so asynchronous updates never increase energy and accepted flips strictly decrease it.
> * __Recall limits__: The signal-crosstalk decomposition separates the retrieval signal from interference, explaining why higher load $\alpha=K/N$ increases the risk of wrong or spurious attractors.


___


### References
1. Hopfield JJ (1982), *Neural networks and physical systems with emergent collective computational abilities*, PNAS 79(8):2554-2558. DOI: https://doi.org/10.1073/pnas.79.8.2554, PubMed: https://pubmed.ncbi.nlm.nih.gov/6953413/
   - Accessible full-text copy used for equation-level verification: https://redwood.berkeley.edu/bruno/npb261/hopfield.pdf
   - Supports claims used here: asynchronous updates in a symmetric network admit an energy function that decreases monotonically until a local minimum is reached.

2. Amit DJ, Gutfreund H, Sompolinsky H (1985), *Storing infinite numbers of patterns in a spin-glass model of neural networks*, Phys. Rev. Lett. 55:1530-1533. DOI: https://doi.org/10.1103/PhysRevLett.55.1530, APS abstract: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.55.1530
   - Supports claims used here: retrieval behavior depends on load $\alpha=K/N$, and beyond low-load conditions additional stable states / retrieval errors appear.

3. McEliece RJ, Posner EC, Rodemich ER, Venkatesh SS (1987), *The capacity of the Hopfield associative memory*, IEEE Trans. Inf. Theory 33(4):461-482. DOI: https://doi.org/10.1109/TIT.1987.1057328, CaltechAUTHORS: https://authors.library.caltech.edu/records/q92rz-95p89/latest
   - Supports claims used here: classical outer-product storage setting and convergence to stable states under symmetric connectivity.

Note: the signal-crosstalk decomposition in this notebook is derived algebraically from the Hebbian weight definition shown above, without introducing extra modeling assumptions.

