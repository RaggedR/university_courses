---
title: "Week 8: Samplers and Acceleration -- Homework"
---

# Week 8: Samplers and Acceleration -- Homework

**Estimated time:** 12-15 hours
**Prerequisites:** DDPM (Week 5), probability flow ODE (Week 7), PyTorch, basic ODE theory

---

## Problem 1: DDIM Derivation (Theory)

### Part (a): From DDPM to DDIM

The DDPM forward process defines $q(x\_t | x\_{t-1})$ as Markovian. DDIM instead defines a non-Markovian forward process with the same marginals $q(x\_t | x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t}\, x\_0,\, (1-\bar{\alpha}\_t)I)$.

Starting from the DDIM reverse update:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \cdot \epsilon
$$

where $\hat{x}\_0 = (x\_t - \sqrt{1-\bar{\alpha}\_t}\,\epsilon\_\theta(x\_t,t)) / \sqrt{\bar{\alpha}\_t}$ and $\epsilon \sim \mathcal{N}(0, I)$:

1. Show that when $\sigma\_t^2 = \frac{(1-\alpha\_t)(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}$ (the DDPM posterior variance), the DDIM update reduces to the DDPM reverse step.

2. Show that when $\sigma\_t = 0$, the update becomes deterministic.

3. Substitute the expression for $\hat{x}\_0$ back into the update and simplify. Express $x\_{t-1}$ purely in terms of $x\_t$ and $\epsilon\_\theta(x\_t, t)$ (no $\hat{x}\_0$). Verify that the coefficients of $x\_t$ and $\epsilon\_\theta$ sum correctly so that $x\_{t-1}$ has the right marginal distribution $q(x\_{t-1} | x\_0)$ when $\epsilon\_\theta$ is perfect.

### Part (b): DDIM as the Probability Flow ODE

The probability flow ODE for the DDPM forward process (in continuous time) is:

$$
\frac{dx}{dt} = -\frac{1}{2}\beta(t)\left[x + \nabla_x \log p_t(x)\right]
$$

Using the score approximation $\nabla\_x \log p\_t(x) \approx -\epsilon\_\theta(x\_t, t) / \sqrt{1 - \bar{\alpha}\_t}$, show that a first-order Euler discretization of this ODE, with the change of variables $x\_t = \sqrt{\bar{\alpha}\_t}\, x\_0 + \sqrt{1 - \bar{\alpha}\_t}\, \epsilon$, yields the DDIM update with $\sigma\_t = 0$.

*Hint:* Work in the $\bar{\alpha}\_t$ parameterization. Let $x\_t = \sqrt{\bar{\alpha}\_t}\, \hat{x}\_0 + \sqrt{1-\bar{\alpha}\_t}\, \hat{\epsilon}$ and discretize the evolution of $\hat{x}\_0$ and $\hat{\epsilon}$.

---

## Problem 2: Implement DDIM Sampling (Implementation)

### Part (a): DDIM Sampler

Assume you have a pretrained DDPM model (use a simple model trained on MNIST or CIFAR-10, or use a mock network for testing). Implement the DDIM sampler:

```python
def ddim_sample(model, shape, timesteps, eta=0.0):
    """
    DDIM sampling.

    Args:
        model: trained noise prediction network epsilon_theta(x_t, t)
        shape: (batch_size, channels, height, width)
        timesteps: list of timesteps to use, e.g., [999, 949, 899, ..., 49, 0]
                   (a subsequence of [0, ..., T-1])
        eta: stochasticity parameter (0 = deterministic, 1 = DDPM)

    Returns:
        x_0: generated samples
    """
    # Your implementation here
    pass
```

Your implementation should:
1. Start from $x\_T \sim \mathcal{N}(0, I)$
2. Iterate through the timestep subsequence in reverse
3. At each step, compute $\hat{x}\_0$, the predicted direction, and the noise term
4. Handle the $\eta$ parameter correctly

### Part (b): Step Count Experiment

Using your DDIM sampler, generate 64 samples with each of the following step counts: $S \in \lbrace 5, 10, 20, 50, 100, 200, 1000\rbrace$. For each:

1. Display a grid of 16 samples
2. Measure the average time per sample
3. If feasible, compute FID against 10000 real samples (otherwise, report qualitative observations)

Plot a curve of visual quality (FID or subjective rating) vs. NFE.

### Part (c): The $\eta$ Sweep

Fix $S = 20$ steps. Generate 64 samples for each $\eta \in \lbrace 0, 0.2, 0.5, 0.8, 1.0\rbrace$.

1. Display samples for each $\eta$ side by side
2. With $\eta = 0$, generate two batches from the same initial $x\_T$. Verify they are identical. With $\eta = 1$, verify they differ.
3. Describe the visual differences as $\eta$ increases. Does more stochasticity help or hurt at 20 steps?

---

## Problem 3: Euler and Heun ODE Solvers (Implementation)

### Part (a): Euler Solver

Implement the Euler solver for the probability flow ODE:

```python
def euler_sample(model, shape, num_steps, noise_schedule):
    """
    Euler method for the probability flow ODE.

    Args:
        model: trained noise prediction network
        shape: sample shape
        num_steps: number of Euler steps
        noise_schedule: object with alpha_bar(t) for continuous t in [0, 1]

    Returns:
        x_0: generated samples
        trajectory: list of intermediate x_t (for visualization)
    """
    # Your implementation here
    pass
```

Convert the discrete DDPM noise schedule to a continuous one by interpolation, or define $\bar{\alpha}(t)$ analytically (e.g., cosine schedule).

The velocity field for the probability flow ODE is:

$$
v_\theta(x, t) = -\frac{1}{2}\beta(t) x - \frac{1}{2}\beta(t) \frac{\epsilon_\theta(x, t)}{\sqrt{1-\bar{\alpha}(t)}} + \frac{1}{2}\beta(t) x
$$

Simplify this expression. (Note: the two $\frac{1}{2}\beta(t)x$ terms cancel partially depending on the exact SDE formulation. Be careful with the drift term.)

### Part (b): Heun Solver

Implement Heun's method (improved Euler):

```python
def heun_sample(model, shape, num_steps, noise_schedule):
    """
    Heun's method (2nd order) for the probability flow ODE.
    Uses 2 NFEs per step.
    """
    # Your implementation here
    pass
```

### Part (c): Comparison

Using the same pretrained model and initial noise $x\_T$:

1. Generate samples with Euler at $N \in \lbrace 10, 20, 50, 100\rbrace$ steps (10, 20, 50, 100 NFEs)
2. Generate samples with Heun at $N/2 \in \lbrace 5, 10, 25, 50\rbrace$ steps (10, 20, 50, 100 NFEs)
3. For each NFE budget, display the Euler and Heun samples side by side

At which NFE count does Heun start to clearly outperform Euler? Is the theoretical $O(1/N)$ vs. $O(1/N^2)$ advantage visible empirically?

### Part (d): Trajectory Visualization

For a 2D toy problem (e.g., a mixture of Gaussians), visualize the sampling trajectories:

1. Train a simple score network on a 2D distribution (e.g., 8 Gaussians arranged in a circle)
2. Plot the Euler trajectory (connected dots from $x\_T$ to $x\_0$) for 10 steps and 100 steps
3. Plot the Heun trajectory for 5 steps and 50 steps
4. Overlay the true data density as a heatmap

How do the trajectories differ? Where do the Euler trajectories deviate most from the smooth ODE solution?

---

## Problem 4: DPM-Solver (Theory + Implementation)

### Part (a): Derive DPM-Solver-1

Starting from the exact solution of the semi-linear ODE:

$$
x_{\lambda_s} = \frac{\alpha_s}{\alpha_t} x_{\lambda_t} - \alpha_s \int_{\lambda_t}^{\lambda_s} e^{-\lambda} \hat{\epsilon}_\theta(x_\lambda, \lambda)\, d\lambda
$$

Approximate $\hat{\epsilon}\_\theta(x\_\lambda, \lambda) \approx \hat{\epsilon}\_\theta(x\_{\lambda\_t}, \lambda\_t)$ (constant) over the interval $[\lambda\_t, \lambda\_s]$.

1. Evaluate the integral analytically under this approximation.
2. Show that the result is: $x\_{\lambda\_s} = \frac{\alpha\_s}{\alpha\_t} x\_{\lambda\_t} - \sigma\_s(e^h - 1) \hat{\epsilon}\_\theta(x\_{\lambda\_t}, \lambda\_t)$ where $h = \lambda\_s - \lambda\_t$ and $\sigma\_s = \sqrt{1 - \bar{\alpha}\_s}$.
3. Show that this is equivalent to DDIM with $\eta = 0$ (up to the choice of time parameterization).

### Part (b): The $x\_0$ Parameterization

The relationship between $\hat{\epsilon}\_\theta$ and $\hat{x}\_0$ is:

$$
\hat{x}_0 = \frac{x_t - \sigma_t \hat{\epsilon}_\theta}{\alpha_t}
$$

Using this, re-derive DPM-Solver-1 in the $x\_0$ parameterization. Show that:

$$
x_{\lambda_s} = \frac{\sigma_s}{\sigma_t} x_{\lambda_t} - \alpha_s(e^{-h} - 1) \hat{x}_0(x_{\lambda_t}, \lambda_t)
$$

*Hint:* Substitute the relationship between $\hat{\epsilon}\_\theta$ and $\hat{x}\_0$ into the DPM-Solver-1 formula and simplify using $\alpha\_t^2 + \sigma\_t^2 = 1$ and $e^{\lambda\_t} = \alpha\_t / \sigma\_t$.

### Part (c): Implementation

Implement DPM-Solver-1 and DPM-Solver-2 (both in the $x\_0$ parameterization):

```python
def dpm_solver_1_step(model, x_t, t, s, noise_schedule):
    """One step of DPM-Solver-1 (equivalent to DDIM)."""
    pass

def dpm_solver_2_step(model, x_t, t, s, noise_schedule):
    """One step of DPM-Solver-2 (second order, 2 NFEs)."""
    pass

def dpm_solver_sample(model, shape, num_steps, order=2):
    """Full DPM-Solver sampling loop."""
    pass
```

Compare DPM-Solver-1 (10 steps), DPM-Solver-2 (10 steps = 20 NFEs), and DDIM (20 steps = 20 NFEs). At equal NFE budgets, does DPM-Solver-2 produce better samples than DDIM?

---

## Problem 5: Adaptive Step Sizes (Theory + Implementation)

### Part (a): Why Uniform Steps Are Suboptimal

Consider the probability flow ODE for a cosine noise schedule. Compute the velocity field magnitude $\Vert v\_\theta(x\_t, t)\Vert$ at $t = 0.01, 0.1, 0.5, 0.9, 0.99$ (using your trained model on a batch of samples).

Plot $\Vert v\_\theta\Vert$ vs. $t$. At which times is the velocity largest? Argue that these are the times where more steps should be concentrated.

### Part (b): Log-SNR Spacing

Implement a timestep schedule that is uniform in the log signal-to-noise ratio:

$$
\lambda(t) = \log \frac{\bar{\alpha}(t)}{1 - \bar{\alpha}(t)}
$$

1. For a linear noise schedule, compute $\lambda(t)$ analytically and invert to find $t(\lambda)$.
2. Generate $S$ uniformly-spaced points in $[\lambda\_{\min}, \lambda\_{\max}]$ and map them back to timesteps.
3. Compare this schedule to uniform-in-$t$ by plotting both sets of timesteps on a number line. Where are the log-SNR timesteps more concentrated?

### Part (c): Compare Schedules

Generate samples using DDIM with 20 steps under three schedules:
1. Uniform in $t$: $\tau\_i = \lfloor i \cdot T / S \rfloor$
2. Uniform in $\lambda$ (log-SNR)
3. Quadratic in $t$: $\tau\_i = \lfloor (i/S)^2 \cdot T \rfloor$ (concentrates steps near $t=0$)

Display samples and report FID (or visual quality) for each schedule. Which performs best?

---

## Problem 6: Sampler Showdown (Implementation)

This problem brings everything together. Using the same pretrained model and the same set of 16 initial noise vectors $x\_T$:

### Part (a): Generate Comparison Grids

Generate a $6 \times 16$ grid where each row is a different sampler, all starting from the same 16 noise vectors, with a budget of 20 NFEs:

| Row | Sampler | Steps | NFEs |
|-----|---------|-------|------|
| 1 | DDIM ($\eta = 0$) | 20 | 20 |
| 2 | DDIM ($\eta = 0.5$) | 20 | 20 |
| 3 | Euler | 20 | 20 |
| 4 | Heun | 10 | 20 |
| 5 | DPM-Solver-1 | 20 | 20 |
| 6 | DPM-Solver-2 | 10 | 20 |

### Part (b): Scaling with NFEs

For each sampler, generate 256 samples at NFE budgets $\lbrace 5, 10, 20, 50, 100\rbrace$. Compute FID (or, if FID computation is infeasible, use a perceptual quality metric or manual ranking).

Plot FID vs. NFEs for all samplers on the same graph. Identify:
1. Which sampler is best at 10 NFEs?
2. Which sampler is best at 50 NFEs?
3. At what NFE count do all samplers converge to similar quality?

### Part (c): Discussion

Answer the following in 2-3 sentences each:

1. The Karras et al. (2022) paper argues that second-order Heun is the best general-purpose sampler. Based on your experiments, do you agree? Under what conditions might a different sampler be preferable?

2. DDIM with $\eta = 0$ and DPM-Solver-1 are theoretically equivalent (both first-order ODE solvers). Did they produce identical samples in your experiments? If not, what accounts for the difference?

3. If you needed to generate 1 million images as fast as possible with acceptable quality, which sampler and step count would you choose? Justify your answer with reference to the quality-vs-NFE curves.

---

## Submission Checklist

- [ ] Problem 1: DDIM derivation showing DDPM recovery and ODE connection
- [ ] Problem 2: DDIM implementation, step count experiment, $\eta$ sweep
- [ ] Problem 3: Euler and Heun implementations, comparison at equal NFEs, trajectory visualization
- [ ] Problem 4: DPM-Solver derivation in both parameterizations, implementation, comparison
- [ ] Problem 5: Velocity field analysis, log-SNR schedule implementation, schedule comparison
- [ ] Problem 6: Sampler showdown grid, scaling curves, discussion

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs. Include all generated images and plots.
