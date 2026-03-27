# Week 3: Stochastic Differential Equations -- Homework

**Estimated time:** 12-15 hours
**Prerequisites:** Probability and stochastic processes (Week 1), score functions (Week 2), basic calculus, PyTorch

---

## Problem 1: Brownian Motion Properties (Theory)

### Part (a): Quadratic Variation

Let $\lbrace t\_0, t\_1, \ldots, t\_n\rbrace $ be an equally spaced partition of $[0, T]$ with $t\_i = iT/n$.

Define the quadratic variation:

$$
Q_n = \sum_{i=0}^{n-1} (W_{t_{i+1}} - W_{t_i})^2
$$

1. Compute $\mathbb{E}[Q\_n]$.
2. Compute $\text{Var}(Q\_n)$. *Hint: the increments are independent, and for $Z \sim \mathcal{N}(0, \sigma^2)$, $\text{Var}(Z^2) = 2\sigma^4$.*
3. Show that $\text{Var}(Q\_n) \to 0$ as $n \to \infty$. What does this imply about $Q\_n$ as a random variable? Relate this to the Ito rule $(dW)^2 = dt$.

### Part (b): Non-Differentiability

Consider the finite-difference approximation to the derivative:

$$
D_h(t) = \frac{W_{t+h} - W_t}{h}
$$

1. Compute $\mathbb{E}[D\_h(t)]$ and $\text{Var}(D\_h(t))$.
2. Show that $\text{Var}(D\_h(t)) \to \infty$ as $h \to 0$, and explain why this means $W\_t$ is not differentiable.
3. Compare this to a smooth function $f(t) = t^2$: compute $\text{Var}(D\_h(t))$ for this deterministic function. Why is the behaviour fundamentally different?

---

## Problem 2: Ito's Lemma Practice (Theory)

### Part (a): Warmup

Apply Ito's lemma to compute $df(W\_t)$ for each of the following functions, where $W\_t$ is a standard Brownian motion ($dW\_t = dW\_t$, i.e., $\mu = 0$, $\sigma = 1$):

1. $f(x) = x^3$
2. $f(x) = e^x$
3. $f(x) = \cos(x)$

For each, write the result in the form $df = A(W\_t) \, dt + B(W\_t) \, dW\_t$ and identify the Ito correction term (the part that would be absent in ordinary calculus).

### Part (b): The Exponential Martingale

Let $X\_t = e^{W\_t - t/2}$.

1. Using Ito's lemma on $f(t, x) = e^{x - t/2}$ with $x = W\_t$, show that $dX\_t = X\_t \, dW\_t$. (Note: there is no $dt$ term!)
2. Conclude that $\mathbb{E}[X\_t] = X\_0 = 1$ for all $t$. A process with this property (no drift) is called a **martingale**.
3. Verify directly: compute $\mathbb{E}[e^{W\_t - t/2}]$ using the moment generating function of the Gaussian. *Hint: if $Z \sim \mathcal{N}(0, \sigma^2)$, then $\mathbb{E}[e^Z] = e^{\sigma^2/2}$.*

### Part (c): Ito vs Stratonovich

In Stratonovich calculus (an alternative to Ito calculus), the standard chain rule holds: $d(W\_t^2) = 2W\_t \circ dW\_t$ (where $\circ$ denotes the Stratonovich integral).

Using the Ito result $d(W\_t^2) = 2W\_t \, dW\_t + dt$, and the conversion formula $X\_t \circ dW\_t = X\_t \, dW\_t + \frac{1}{2}dX\_t \cdot dW\_t$, verify that the Stratonovich form is consistent. Why do diffusion model papers typically use Ito calculus rather than Stratonovich?

---

## Problem 3: Solving the OU Process (Theory)

### Part (a): Derivation

Starting from the OU SDE $dX\_t = -\theta X\_t \, dt + \sigma \, dW\_t$ with initial condition $X\_0 = x\_0$:

1. Define $Y\_t = e^{\theta t} X\_t$. Apply Ito's lemma to show that $dY\_t = \sigma e^{\theta t} \, dW\_t$.
2. Integrate to obtain $X\_t = e^{-\theta t}x\_0 + \sigma \int\_0^t e^{-\theta(t-s)} \, dW\_s$.
3. Use the Ito isometry ($\text{Var}(\int\_0^t h(s) \, dW\_s) = \int\_0^t h(s)^2 \, ds$) to show:

$$
X_t \sim \mathcal{N}\left(e^{-\theta t} x_0, \; \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})\right)
$$

### Part (b): Conditional Distribution

Show that the transition kernel of the OU process -- the conditional distribution of $X\_t$ given $X\_s$ for $s < t$ -- is:

$$
X_t \mid X_s \sim \mathcal{N}\left(e^{-\theta(t-s)} X_s, \; \frac{\sigma^2}{2\theta}(1 - e^{-2\theta(t-s)})\right)
$$

*Hint: The OU process is Markov. Apply the result from Part (a) with "initial time" $s$ instead of $0$.*

### Part (c): Stationary Distribution

1. Take $t \to \infty$ in your result to show that the stationary distribution is $\mathcal{N}(0, \sigma^2/2\theta)$.
2. What choice of $\sigma$ and $\theta$ gives a standard normal $\mathcal{N}(0, 1)$ stationary distribution?
3. For the VP SDE $dX\_t = -\frac{1}{2}\beta X\_t \, dt + \sqrt{\beta} \, dW\_t$ (constant $\beta$), identify $\theta$ and $\sigma$, and verify that the stationary distribution is $\mathcal{N}(0, 1)$.

---

## Problem 4: The Fokker-Planck Equation for the OU Process (Theory)

### Part (a): Verification

The Fokker-Planck equation for the OU process is:

$$
\frac{\partial p}{\partial t} = \theta \frac{\partial}{\partial x}(x \, p) + \frac{\sigma^2}{2}\frac{\partial^2 p}{\partial x^2}
$$

Verify that the Gaussian density $p\_t(x) = \mathcal{N}(m\_t, v\_t)$ with $m\_t = e^{-\theta t} x\_0$ and $v\_t = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$ satisfies this equation.

Steps:
1. Write the Gaussian density explicitly: $p\_t(x) = \frac{1}{\sqrt{2\pi v\_t}} \exp\left(-\frac{(x - m\_t)^2}{2v\_t}\right)$
2. Compute $\frac{\partial p}{\partial t}$, $\frac{\partial}{\partial x}(x p)$, and $\frac{\partial^2 p}{\partial x^2}$ (this is tedious but instructive)
3. Verify the identity

*Hint: You will need $\dot{m}\_t = -\theta m\_t$ and $\dot{v}\_t = -2\theta v\_t + \sigma^2$.*

### Part (b): Stationary Solution

Set $\partial p / \partial t = 0$ in the Fokker-Planck equation. Show that the stationary solution satisfies:

$$
\theta x \, p + \frac{\sigma^2}{2} \frac{\partial p}{\partial x} = 0
$$

Solve this first-order ODE for $p(x)$ and confirm it is the Gaussian $\mathcal{N}(0, \sigma^2/2\theta)$.

### Part (c): Score at Stationarity

At stationarity, compute the score function $\nabla\_x \log p(x)$. Verify that it equals $-2\theta x / \sigma^2$. Plug this into Anderson's reverse SDE formula and show that the reverse OU process at stationarity is identical to the forward OU process (i.e., the process is time-reversible at equilibrium).

---

## Problem 5: Euler-Maruyama SDE Simulation (Implementation)

Implement the Euler-Maruyama method in PyTorch and use it to simulate various SDEs.

### Part (a): Simulate Brownian Motion

Write a function `simulate_brownian(T, num_steps, num_paths, dim)` that simulates `num_paths` independent $d$-dimensional Brownian motions from $t=0$ to $t=T$ using `num_steps` time steps.

```python
def simulate_brownian(T: float, num_steps: int, num_paths: int, dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        t: Time points, shape (num_steps + 1,)
        W: Brownian paths, shape (num_paths, num_steps + 1, dim)
    """
    ...
```

Plot 10 sample paths of 1D Brownian motion for $T = 5$. Verify empirically that:
1. $\mathbb{E}[W\_t] \approx 0$ (estimate from many paths)
2. $\text{Var}(W\_t) \approx t$ (estimate from many paths)
3. The quadratic variation $\sum\_i (W\_{t\_{i+1}} - W\_{t\_i})^2 \approx T$ (for a single path, with fine enough discretization)

### Part (b): Simulate the OU Process

Write a function `simulate_ou(x0, theta, sigma, T, num_steps, num_paths)` that simulates the OU process using Euler-Maruyama:

$$
X_{t+\Delta t} = X_t - \theta X_t \Delta t + \sigma \sqrt{\Delta t} \, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1)
$$

Simulate with $\theta = 1.0$, $\sigma = \sqrt{2}$ (so the stationary distribution is $\mathcal{N}(0,1)$), $x\_0 = 5.0$, $T = 10$, using 1000 paths.

1. Plot 20 sample paths. Observe the mean reversion toward zero.
2. At each time $t$, compute the empirical mean and variance across paths. Plot these against the theoretical values $m\_t = e^{-t} \cdot 5$ and $v\_t = 1 - e^{-2t}$. How well do they agree?
3. At $t = 10$, plot a histogram of $X\_T$ across all paths. Overlay the theoretical stationary density $\mathcal{N}(0, 1)$.

### Part (c): Simulate Geometric Brownian Motion

Simulate the geometric Brownian motion SDE $dS\_t = \mu S\_t \, dt + \sigma\_{\text{gbm}} S\_t \, dW\_t$ with $\mu = 0.1$, $\sigma\_{\text{gbm}} = 0.3$, $S\_0 = 100$, $T = 2$.

1. Implement Euler-Maruyama: $S\_{t+\Delta t} = S\_t + \mu S\_t \Delta t + \sigma\_{\text{gbm}} S\_t \sqrt{\Delta t} \, \epsilon$.
2. Compare 100 simulated paths against the exact solution: $S\_T = S\_0 \exp[(\mu - \sigma\_{\text{gbm}}^2/2)T + \sigma\_{\text{gbm}} W\_T]$.
3. Compute the mean of $S\_T$ across many paths and compare to the theoretical $\mathbb{E}[S\_T] = S\_0 e^{\mu T}$. Also compare the log-normal distribution of $S\_T$.

### Part (d): Convergence of Euler-Maruyama

For the OU process with $\theta = 1$, $\sigma = 1$, $x\_0 = 3$, $T = 2$:

Run Euler-Maruyama with $\Delta t \in \lbrace 0.1, 0.05, 0.01, 0.005, 0.001\rbrace $, using the same Brownian path for each (fix the random seed and interpolate the Brownian motion). Compare the final value $X\_T$ to the "reference" solution computed with $\Delta t = 0.0001$.

Plot the error $|X\_T^{\Delta t} - X\_T^{\text{ref}}|$ vs. $\Delta t$ on a log-log plot. What is the empirical convergence order? (Euler-Maruyama has strong order 0.5 for general SDEs -- does your plot agree?)

---

## Problem 6: The Forward Diffusion Process in 2D (Implementation)

This problem connects the SDE theory to the diffusion model framework.

### Part (a): Define a 2D Data Distribution

Create a synthetic 2D data distribution -- a mixture of Gaussians arranged in an interesting pattern:

```python
def sample_data(n: int) -> torch.Tensor:
    """Sample n points from a mixture of 8 Gaussians arranged in a circle."""
    angles = torch.linspace(0, 2 * torch.pi, 9)[:-1]  # 8 equally spaced angles
    centers = 4.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    # Each point: pick a random center, then add Gaussian noise
    idx = torch.randint(0, 8, (n,))
    x = centers[idx] + 0.3 * torch.randn(n, 2)
    return x
```

Sample 10000 points and plot them. This is your data distribution $p\_0$.

### Part (b): Forward SDE

Implement the VP forward SDE:

$$
dX_t = -\frac{1}{2}\beta(t) X_t \, dt + \sqrt{\beta(t)} \, dW_t
$$

with a linear schedule $\beta(t) = \beta\_{\min} + (\beta\_{\max} - \beta\_{\min}) t / T$ where $\beta\_{\min} = 0.1$, $\beta\_{\max} = 20$, $T = 1$.

Use Euler-Maruyama to simulate the forward process. Start from 5000 data points and evolve them to $t = T$.

1. Plot the point cloud at times $t \in \lbrace 0, 0.1, 0.2, 0.5, 0.8, 1.0\rbrace $ (six subplots). You should see the structured data distribution gradually dissolving into Gaussian noise.
2. At each time, plot a histogram of the marginal distributions along the $x$-axis. How does the distribution change?

### Part (c): The Score at Each Time

For Gaussian transition kernels, we can compute the score analytically. Given $X\_t | X\_0 = x\_0 \sim \mathcal{N}(\alpha\_t x\_0, \sigma\_t^2 I)$, the marginal $p\_t(x) = \int p(x\_t | x\_0) p\_0(x\_0) \, dx\_0$ is a mixture of Gaussians (since $p\_0$ is a mixture of Gaussians).

For each time $t$, compute and plot the score field $\nabla\_x \log p\_t(x)$ as a 2D vector field on a grid. Use the fact that for a mixture of Gaussians, the score is:

$$
\nabla_x \log p_t(x) = \frac{\sum_k w_k \, \mathcal{N}(x; \mu_k(t), \sigma_t^2 I) \cdot \frac{x - \mu_k(t)}{-\sigma_t^2}}{\sum_k w_k \, \mathcal{N}(x; \mu_k(t), \sigma_t^2 I)}
$$

where $\mu\_k(t) = \alpha\_t \mu\_k$ are the time-evolved centers and $w\_k$ are the mixture weights.

Plot the score field at $t = 0.01$ (scores point strongly toward cluster centers), $t = 0.2$ (scores show broader structure), and $t = 0.9$ (scores point toward the origin, like a single Gaussian).

### Part (d): Reverse SDE (Preview)

Using the analytically computed score from Part (c), implement Anderson's reverse SDE:

$$
dX_t = \left[-\frac{1}{2}\beta(t)X_t - \beta(t)\nabla_x \log p_t(X_t)\right] dt + \sqrt{\beta(t)} \, d\bar{W}_t
$$

Start from 5000 points sampled from $\mathcal{N}(0, I)$ at $t = T = 1$ and integrate backward to $t = 0$ using Euler-Maruyama (with $dt < 0$, using 1000 steps).

Plot the generated samples. Do they match the original data distribution? Overlay the generated points on the original data.

This is a diffusion model with a perfect (analytical) score function. In practice, we will replace the analytical score with a neural network estimate (Week 4).

---

## Submission Checklist

- [ ] Problem 1: Brownian motion quadratic variation and non-differentiability derivations
- [ ] Problem 2: Ito's lemma applied to three functions, exponential martingale, Ito vs Stratonovich discussion
- [ ] Problem 3: OU process solution, transition kernel, stationary distribution, VP SDE identification
- [ ] Problem 4: Fokker-Planck verification for OU, stationary solution, reversibility at equilibrium
- [ ] Problem 5: Euler-Maruyama simulations of BM, OU, and GBM with convergence analysis
- [ ] Problem 6: 2D forward diffusion, score field visualization, reverse SDE generation

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs and plots.
