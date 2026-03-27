# Week 7: The SDE Unification -- Homework

**Estimated time:** 12-15 hours
**Prerequisites:** Stochastic differential equations (Week 3), DDPM (Week 5), NCSN (Week 6), PyTorch, basic ODE/SDE numerical methods

---

## Problem 1: Derive the VP-SDE as the Continuous Limit of DDPM (Theory)

### Part (a): The Limiting SDE

Starting from the DDPM forward step:

$$
x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, I)
$$

1. Write $\Delta x = x\_t - x\_{t-1}$ and expand $\sqrt{1-\beta\_t} \approx 1 - \beta\_t/2$ for small $\beta\_t$.

2. Interpret the discrete index $t$ as continuous time by setting $\beta\_t = \beta(t)\, \Delta t$ where $\Delta t = 1/T$. Substitute and take $\Delta t \to 0$ to obtain:

$$
dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dW
$$

3. Verify that this is the VP-SDE from the notes.

### Part (b): The Transition Kernel

For the VP-SDE $dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dW$:

1. This is a linear SDE of the form $dx = a(t)x\,dt + b(t)\,dW$ with $a(t) = -\beta(t)/2$ and $b(t) = \sqrt{\beta(t)}$. Using the integrating factor method (or by direct verification), show that the solution is:

$$
x(t) = e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}\, x(0) + \int_0^t e^{-\frac{1}{2}\int_s^t \beta(r)\,dr}\, \sqrt{\beta(s)}\, dW(s)
$$

2. Since the Ito integral of a deterministic integrand against Brownian motion is Gaussian, compute the mean and variance of $x(t)$ given $x(0)$. Define $\bar{\alpha}(t) = e^{-\int\_0^t \beta(s)\,ds}$ and show:

$$
q(x(t) \mid x(0)) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}(t)}\, x(0),\; (1 - \bar{\alpha}(t))\, I\right)
$$

*Hint: for the variance, you need to compute $\int\_0^t e^{-\int\_s^t \beta(r)\,dr}\, \beta(s)\, ds$. Use the substitution $u = \int\_s^t \beta(r)\,dr$.*

3. Verify that this recovers the DDPM formula $q(x\_t \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t}\, x\_0, (1-\bar{\alpha}\_t)I)$ when you discretize.

---

## Problem 2: The Reverse SDE and Probability Flow ODE (Theory)

### Part (a): Derive the Reverse VP-SDE

Starting from the general reverse-time SDE:

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\, d\bar{W}
$$

Substitute $f(x,t) = -\frac{1}{2}\beta(t)x$ and $g(t) = \sqrt{\beta(t)}$ to obtain the reverse VP-SDE. Write it out explicitly.

### Part (b): Derive the Probability Flow ODE for VP-SDE

Using the formula from the notes:

$$
dx = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right] dt
$$

write the probability flow ODE for the VP-SDE. Show that it can be written as:

$$
dx = -\frac{1}{2}\beta(t)\!\left[x + s_\theta(x, t)\right] dt
$$

### Part (c): Derive the Probability Flow ODE for VE-SDE

Repeat Part (b) for the VE-SDE ($f = 0$, $g(t) = \sqrt{d\sigma^2(t)/dt}$). Show that:

$$
dx = -\frac{1}{2}\frac{d[\sigma^2(t)]}{dt}\, s_\theta(x, t)\, dt
$$

### Part (d): Same Marginals, Different Paths

Consider a 1D Ornstein-Uhlenbeck process: $dx = -x\,dt + \sqrt{2}\,dW$. Its stationary distribution is $\mathcal{N}(0, 1)$.

1. Write the probability flow ODE for this process.
2. Show that both the SDE and the ODE have the same stationary distribution $\mathcal{N}(0,1)$.
3. Simulate 100 trajectories of both the SDE and ODE from $x(0) = 5$ over $t \in [0, 5]$. Plot all trajectories. The SDE trajectories should be noisy and spread out; the ODE trajectories should be smooth and converge to a single path. Yet the distributions of $x(t)$ at any fixed $t$ should be approximately the same.

---

## Problem 3: Implement the Probability Flow ODE (Implementation)

### Part (a): Train a Score Network

Using the 8-Gaussians dataset from previous weeks:

1. Train a time-conditional score network $s\_\theta(x, t)$ using the continuous-time denoising score matching loss. Sample $t \sim \mathcal{U}(0, 1)$ uniformly.

2. For the VP-SDE, use $\beta(t) = \beta\_{\min} + t(\beta\_{\max} - \beta\_{\min})$ with $\beta\_{\min} = 0.1$, $\beta\_{\max} = 20$.

3. The forward transition kernel is $q(x(t) \mid x(0)) = \mathcal{N}(\sqrt{\bar{\alpha}(t)}\, x(0),\; (1-\bar{\alpha}(t))I)$ where $\bar{\alpha}(t) = e^{-\int\_0^t \beta(s)\,ds}$. Precompute $\bar{\alpha}(t)$ analytically: for a linear $\beta(t)$, $\int\_0^t \beta(s)\,ds = \beta\_{\min} t + \frac{1}{2}(\beta\_{\max} - \beta\_{\min})t^2$.

4. Train for 10000 gradient steps.

### Part (b): Reverse SDE Sampling

Implement Euler-Maruyama discretization of the reverse VP-SDE:

```python
def reverse_sde_sample(model, n_samples, T=1.0, N=1000, beta_min=0.1, beta_max=20.0):
    """Sample by discretizing the reverse VP-SDE."""
    dt = T / N
    x = torch.randn(n_samples, 2)  # x(T) ~ N(0, I)

    for i in range(N):
        t = T - i * dt
        beta_t = beta_min + t * (beta_max - beta_min)
        score = model(x, t * torch.ones(n_samples, 1))

        # Reverse SDE: dx = [-beta/2 * x - beta * score] dt + sqrt(beta) dW_bar
        drift = (-0.5 * beta_t * x - beta_t * score) * (-dt)  # negative dt for reverse
        diffusion = (beta_t * dt).sqrt() * torch.randn_like(x)
        x = x + drift + diffusion

    return x
```

Generate 5000 samples and plot them.

### Part (c): Probability Flow ODE Sampling

Implement Euler discretization of the probability flow ODE:

```python
def ode_sample(model, n_samples, T=1.0, N=1000, beta_min=0.1, beta_max=20.0):
    """Sample by discretizing the probability flow ODE."""
    dt = T / N
    x = torch.randn(n_samples, 2)

    for i in range(N):
        t = T - i * dt
        beta_t = beta_min + t * (beta_max - beta_min)
        score = model(x, t * torch.ones(n_samples, 1))

        # ODE: dx = [-beta/2 * x - beta/2 * score] dt
        drift = (-0.5 * beta_t * x - 0.5 * beta_t * score) * (-dt)
        x = x + drift

    return x
```

Generate 5000 samples and plot them alongside the reverse SDE samples.

### Part (d): Adaptive ODE Solver

Use `torchdiffeq` or `scipy.integrate.solve_ivp` to solve the probability flow ODE with an adaptive RK45 solver. Compare the number of function evaluations (NFE) needed for comparable sample quality. How many NFE does the adaptive solver use vs. the fixed 1000-step Euler method?

```python
from scipy.integrate import solve_ivp

def ode_rhs(t_scalar, x_flat, model, beta_min, beta_max, dim=2):
    """Right-hand side of the probability flow ODE (forward in time, t: T -> 0)."""
    # Note: solve_ivp runs forward, so we parameterize as s = T - t
    t = t_scalar  # actual time
    x = torch.tensor(x_flat.reshape(-1, dim), dtype=torch.float32)
    beta_t = beta_min + t * (beta_max - beta_min)

    with torch.no_grad():
        score = model(x, t * torch.ones(x.shape[0], 1))

    drift = -0.5 * beta_t * (x + score)
    return drift.numpy().flatten()
```

---

## Problem 4: VP-SDE and VE-SDE Produce the Same Reverse Formula (Theory)

This problem shows that the reverse SDE has the same structure regardless of whether we use VP or VE.

### Part (a): Write Both Reverse SDEs

1. Write the reverse VP-SDE explicitly (substituting the VP drift and diffusion into Anderson's formula).
2. Write the reverse VE-SDE explicitly (substituting the VE drift and diffusion).
3. In both cases, identify the "score-dependent correction" term and the "noise" term.

### Part (b): Show Structural Equivalence

Both reverse SDEs have the general form:

$$
dx = [\text{linear in } x + \text{score-dependent}]\, dt + \text{noise}\, d\bar{W}
$$

Show that for both VP and VE, the score-dependent term is proportional to $-g(t)^2\, s\_\theta(x, t)$, where $g(t)$ is the forward diffusion coefficient. The difference is only in the linear drift term.

### Part (c): Effective Noise Scale Matching

At time $t$, the VP-SDE has marginal $q(x\_t \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}(t)}\, x\_0,\; (1-\bar{\alpha}(t))I)$ and the VE-SDE has marginal $q(x\_t \mid x\_0) = \mathcal{N}(x\_0,\; \sigma^2(t)I)$.

Define the "effective signal-to-noise ratio" for each:
- VP: $\text{SNR}\_{\text{VP}}(t) = \bar{\alpha}(t) / (1 - \bar{\alpha}(t))$
- VE: $\text{SNR}\_{\text{VE}}(t) = 1 / \sigma^2(t)$ (assuming unit-variance data)

Show that if the noise schedules are chosen so that $\text{SNR}\_{\text{VP}}(t) = \text{SNR}\_{\text{VE}}(t)$ for all $t$, then the two models produce the same noised distributions (up to a global rescaling of $x$).

---

## Problem 5: Exact Log-Likelihoods via the Instantaneous Change of Variables (Theory + Implementation)

### Part (a): Derive the Formula (Theory)

The probability flow ODE $dx/dt = \tilde{f}(x, t)$ defines a flow $\phi\_t: x(0) \mapsto x(t)$.

1. Starting from the change of variables formula for diffeomorphisms:

$$
p_0(x_0) = p_T(\phi_T(x_0)) \cdot |\det J_{\phi_T}(x_0)|
$$

where $J\_{\phi\_T}$ is the Jacobian of $\phi\_T$, take logarithms and show:

$$
\log p_0(x_0) = \log p_T(x_T) - \log |\det J_{\phi_T}(x_0)|
$$

2. The instantaneous change of variables (Chen et al., 2018) converts the Jacobian log-determinant into an integral:

$$
\log |\det J_{\phi_T}(x_0)| = \int_0^T \nabla \cdot \tilde{f}(x(t), t)\, dt
$$

where $\nabla \cdot \tilde{f}$ is the divergence (trace of the Jacobian). State why this integral is generally cheaper to compute than the full Jacobian determinant.

3. Combine to get:

$$
\log p_0(x_0) = \log p_T(x_T) + \int_0^T \nabla \cdot \tilde{f}(x(t), t)\, dt
$$

*Note the sign: the divergence integral has a positive sign here because we integrate forward from 0 to T. Some references write it with a negative sign, depending on whether they define the flow from T to 0.*

### Part (b): The Hutchinson Estimator (Theory)

Computing $\nabla \cdot \tilde{f}(x, t) = \text{tr}\!\left(\frac{\partial \tilde{f}}{\partial x}\right)$ requires the trace of the $d \times d$ Jacobian, which costs $O(d)$ backward passes (one per dimension).

The Hutchinson trace estimator provides an unbiased estimate using a single backward pass:

$$
\text{tr}(A) = \mathbb{E}_{v \sim \mathcal{N}(0,I)}[v^\top A v]
$$

1. Prove that this estimator is unbiased: $\mathbb{E}[v^\top A v] = \text{tr}(A)$.

2. For the probability flow ODE, explain how to compute $v^\top \frac{\partial \tilde{f}}{\partial x} v$ using a single vector-Jacobian product (which can be computed via one backward pass with `torch.autograd`).

### Part (c): Compute Log-Likelihoods (Implementation)

Using your trained VP-SDE score network from Problem 3:

1. Implement the log-likelihood computation:

```python
def log_likelihood(model, x_0, T=1.0, N=500, beta_min=0.1, beta_max=20.0):
    """
    Compute log p_0(x_0) using the instantaneous change of variables.

    Returns:
        log_p: shape (B,)
    """
    dt = T / N
    x = x_0.clone().requires_grad_(True)
    log_det = torch.zeros(x.shape[0])

    for i in range(N):
        t = i * dt  # integrate forward from 0 to T
        beta_t = beta_min + t * (beta_max - beta_min)

        # Compute ODE velocity
        score = model(x, t * torch.ones(x.shape[0], 1))
        velocity = -0.5 * beta_t * (x + score)

        # Hutchinson estimator for divergence
        v = torch.randn_like(x)
        vjp = torch.autograd.grad(velocity, x, v, create_graph=False)[0]
        div = (v * vjp).sum(dim=-1)

        log_det = log_det + div * dt
        x = x + velocity.detach() * dt

    # log p_T(x_T) under the prior N(0, I)
    log_prior = -0.5 * (x.detach()**2).sum(dim=-1) - x.shape[-1] * 0.5 * torch.log(torch.tensor(2 * 3.14159))

    return log_prior - log_det
```

2. Compute the average log-likelihood on 1000 held-out test points from the 8-Gaussians distribution. Compare to the true log-likelihood (which you can compute analytically for a Gaussian mixture).

3. How sensitive is the result to the number of discretization steps $N$? Plot log-likelihood vs. $N$ for $N \in \lbrace 50, 100, 200, 500, 1000\rbrace $.

---

## Problem 6: From DDPM to ODE and Back (Implementation)

### Part (a): DDIM as a Probability Flow ODE

The DDIM sampler (Song et al., 2020) can be viewed as a discretization of the probability flow ODE. Starting from the DDPM $\varepsilon$-prediction model:

1. Convert your trained DDPM noise predictor $\varepsilon\_\theta(x\_t, t)$ from Week 5 Homework Problem 3 (or Problem 4) into a score network: $s\_\theta(x, t) = -\varepsilon\_\theta(x, t)/\sqrt{1-\bar{\alpha}\_t}$.

2. Implement the DDIM update rule:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1 - \bar{\alpha}_{t-1}}\,\varepsilon_\theta(x_t, t)
$$

3. Show (on paper or by numerical comparison) that this is equivalent to one Euler step of the probability flow ODE for the VP-SDE.

### Part (b): Subsampled Timesteps

DDIM allows sampling with a subsequence of timesteps $\tau\_1, \tau\_2, \ldots, \tau\_S$ where $S \ll T$. Implement this:

```python
def ddim_sample(model, alpha_bar, timesteps, shape):
    """
    DDIM sampling with a subset of timesteps.

    Args:
        model: noise prediction network
        alpha_bar: full schedule, shape (T,)
        timesteps: subsequence of timestep indices, e.g. [999, 949, 899, ..., 49, 0]
        shape: sample shape (B, C, H, W)
    """
    x = torch.randn(shape)
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        # TODO: implement DDIM step
        pass
    return x
```

Generate samples using $S \in \lbrace 10, 20, 50, 100, 200, 1000\rbrace $ evenly-spaced timesteps. Compare sample quality (visually and, for the 2D case, via Wasserstein distance). At what $S$ does quality become acceptable?

### Part (c): Deterministic Encoding

Using DDIM with $S = 100$ steps, encode 10 data points into latent space by running the ODE forward (from $t = 0$ to $t = T$). Then decode them by running the ODE backward (from $t = T$ to $t = 0$).

1. Compare the reconstructed $\hat{x}\_0$ to the original $x\_0$. How large is the reconstruction error? (For the 2D case, plot original and reconstructed points; for MNIST, display original and reconstructed images.)

2. Take two data points, encode them to $z\_1, z\_2$, interpolate in latent space ($z\_\alpha = (1-\alpha)z\_1 + \alpha z\_2$), and decode. Plot the interpolation for $\alpha \in \lbrace 0, 0.25, 0.5, 0.75, 1\rbrace $.

---

## Submission Checklist

- [ ] Problem 1: VP-SDE derivation as continuous DDPM limit, transition kernel derivation
- [ ] Problem 2: Reverse SDE and probability flow ODE derivations for VP and VE, OU process comparison
- [ ] Problem 3: Score network training, reverse SDE sampling, ODE sampling, adaptive solver comparison
- [ ] Problem 4: Structural equivalence of VP and VE reverse SDEs, SNR matching
- [ ] Problem 5: Change of variables derivation, Hutchinson estimator proof, log-likelihood computation
- [ ] Problem 6: DDIM as ODE discretization, subsampled timestep sampling, deterministic encoding and interpolation

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs and plots.
