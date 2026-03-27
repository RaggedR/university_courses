# Week 6: Score-Based Generative Models -- Homework

**Estimated time:** 12-15 hours
**Prerequisites:** Score functions, Langevin dynamics (Week 2), denoising score matching (Week 4), DDPM (Week 5), PyTorch

---

## Problem 1: Score Estimation Fails in Low-Density Regions (Theory + Implementation)

### Part (a): Analytic Score (Theory)

Consider the 1D mixture of Gaussians:

$$
p(x) = 0.5\,\mathcal{N}(x; -5, 0.5^2) + 0.5\,\mathcal{N}(x; 5, 0.5^2)
$$

1. Derive the exact score $\nabla\_x \log p(x)$. *Hint: compute $p'(x)/p(x)$.*

2. Evaluate the score at $x = 0$ (the midpoint between modes). What is its magnitude? What does this tell you about the quality of the gradient signal for Langevin dynamics initialized at $x = 0$?

3. Now consider the noised distribution $p\_\sigma(x) = p * \mathcal{N}(0, \sigma^2)$ (convolution with a Gaussian). For $\sigma = 3$, derive (or numerically compute) $\nabla\_x \log p\_\sigma(x)$ at $x = 0$. Compare to the unperturbed case.

### Part (b): Empirical Demonstration (Implementation)

1. Draw 10000 samples from $p(x)$ above. Train a small MLP $s\_\theta(x)$ to estimate the score via the explicit score matching loss (Hyvarinen, 2005):

$$
\mathcal{L} = \mathbb{E}_{p(x)}\!\left[\frac{1}{2}\|s_\theta(x)\|^2 + \text{tr}(\nabla_x s_\theta(x))\right]
$$

*For a 1D scalar network, $\text{tr}(\nabla\_x s\_\theta(x)) = s\_\theta'(x)$, the derivative with respect to the input. Use `torch.autograd` to compute it.*

2. Plot the learned score $s\_\theta(x)$ vs. the true score $\nabla\_x \log p(x)$ over $x \in [-10, 10]$. Where does the learned score match well? Where does it diverge?

3. Run Langevin dynamics for 5000 steps with $\eta = 0.01$, starting from $x\_0 = 0$, using (i) the true score and (ii) the learned score. Plot histograms of 1000 independent chains. Does the learned-score version recover both modes?

### Part (c): The Noise Fix (Implementation)

Repeat Part (b) but now train $s\_\theta(x, \sigma)$ with denoising score matching at noise level $\sigma = 3$. Show that the learned score at $\sigma = 3$ is accurate even near $x = 0$.

---

## Problem 2: Implement NCSN Training (Implementation)

Implement the full NCSN training pipeline on a 2D dataset.

### Part (a): Noise-Conditional Score Network

Using the same 8-Gaussians dataset from Week 5 Homework Problem 4:

1. Define $L = 10$ geometrically spaced noise levels from $\sigma\_1 = 5.0$ to $\sigma\_L = 0.01$.

2. Implement a noise-conditional score network: an MLP that takes $(x, \sigma)$ as input (concatenate $x \in \mathbb{R}^2$ with $\log \sigma$) and outputs $s\_\theta(x, \sigma) \in \mathbb{R}^2$.

   Architecture: 4 hidden layers, 256 units each, SiLU activations.

3. Implement the NCSN training loss with $\lambda(\sigma\_i) = \sigma\_i^2$ weighting:

```python
def ncsn_loss(model, x, sigmas):
    """
    Args:
        model: score network s_theta(x, sigma)
        x: clean data, shape (B, 2)
        sigmas: noise levels, shape (L,)
    Returns:
        loss: scalar
    """
    # Sample a random noise level for each data point
    i = torch.randint(0, len(sigmas), (x.shape[0],))
    sigma = sigmas[i].unsqueeze(-1)           # (B, 1)

    # Add noise
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise               # (B, 2)

    # Score prediction
    score = model(x_noisy, sigma)              # (B, 2)

    # Target: -noise / sigma
    target = -noise / sigma

    # Weighted loss
    loss = (sigma.squeeze()**2 * ((score - target)**2).sum(dim=-1)).mean()
    return loss
```

4. Train for 10000 gradient steps. Plot the loss curve.

### Part (b): Visualize Learned Scores

For each of the 10 noise levels $\sigma\_i$, plot the learned score field $s\_\theta(x, \sigma\_i)$ as a vector field over $[-3, 3]^2$. At high noise, the arrows should point uniformly toward the center (one broad mode). At low noise, they should point toward the 8 individual modes.

---

## Problem 3: Annealed Langevin Dynamics (Implementation)

### Part (a): Implement Sampling

Using your trained NCSN from Problem 2, implement annealed Langevin dynamics:

```python
def annealed_langevin(model, sigmas, n_samples, K=100, eps=0.01):
    """
    Generate samples using annealed Langevin dynamics.

    Args:
        model: trained score network
        sigmas: noise levels (L,), decreasing
        n_samples: number of samples to generate
        K: Langevin steps per noise level
        eps: base step size
    Returns:
        samples: shape (n_samples, 2)
    """
    x = torch.randn(n_samples, 2) * sigmas[0]  # initialize from broad Gaussian

    for i, sigma in enumerate(sigmas):
        eta = eps * (sigma / sigmas[-1])**2     # step size
        for k in range(K):
            z = torch.randn_like(x)
            score = model(x, sigma * torch.ones(n_samples, 1))
            x = x + (eta / 2) * score + eta.sqrt() * z

    return x
```

Generate 5000 samples and plot them alongside the true data distribution.

### Part (b): Effect of Langevin Steps $K$

Run annealed Langevin dynamics with $K \in \lbrace 1, 10, 50, 100, 500\rbrace $ steps per noise level. For each, plot the generated samples and compute the Wasserstein distance to the true distribution. How many steps per level are needed for good samples?

### Part (c): Visualize the Annealing

Generate one batch of 500 samples and record the intermediate states at each noise level transition. Create a 2x5 grid showing the sample distribution after levels $\lbrace 1, 3, 5, 7, 10\rbrace $ (from highest to lowest noise). You should see the samples progressively organize from a diffuse cloud into 8 tight clusters.

---

## Problem 4: Prove the DDPM-NCSN Equivalence (Theory)

This is the key theoretical result of the week. Prove each part carefully.

### Part (a): Score of the Forward Process

Starting from $q(x\_t \mid x\_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}\_t}\, x\_0,\; (1 - \bar{\alpha}\_t)\, I\right)$:

1. Write the log-density $\log q(x\_t \mid x\_0)$ explicitly.
2. Compute $\nabla\_{x\_t} \log q(x\_t \mid x\_0)$.
3. Using $x\_t = \sqrt{\bar{\alpha}\_t}\, x\_0 + \sqrt{1-\bar{\alpha}\_t}\, \varepsilon$, express the score in terms of $\varepsilon$.

### Part (b): Noise Prediction = Score Estimation

1. In DDPM, the network predicts $\varepsilon\_\theta(x\_t, t)$ and the training loss is $\Vert \varepsilon - \varepsilon\_\theta(x\_t, t)\Vert ^2$. Show that minimizing this loss is equivalent to training a score network $s\_\theta(x\_t, t)$ with the denoising score matching objective, where:

$$
s_\theta(x_t, t) = -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

2. Derive the relationship between the DDPM loss weight (uniform over $t$) and the NCSN loss weight ($\sigma^2$).

### Part (c): Sampling Equivalence

Starting from the DDPM sampling formula:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t)\right) + \sigma_t\, z
$$

Substitute $\varepsilon\_\theta = -\sqrt{1-\bar{\alpha}\_t}\, s\_\theta$ and rewrite the sampling step in terms of the score function $s\_\theta$. Compare the resulting expression to one step of Langevin dynamics:

$$
x \leftarrow x + \frac{\eta}{2}\, s_\theta(x, \sigma) + \sqrt{\eta}\, z
$$

Identify the effective step size $\eta$ in terms of the DDPM noise schedule parameters. Are they exactly the same, or is there an additional correction factor?

### Part (d): The Missing Piece

The DDPM sampling step includes a $1/\sqrt{\alpha\_t}$ factor that pure Langevin dynamics does not have. Explain what this factor does geometrically. *Hint: in the DDPM forward process, the signal is scaled by $\sqrt{\bar{\alpha}\_t}$ at time $t$. The reverse process must undo this scaling.*

---

## Problem 5: NCSN vs. DDPM Head-to-Head (Implementation)

Train both models on the same dataset and compare them directly.

### Part (a): Setup

Using the 8-Gaussians dataset (10000 training points):

1. Train a DDPM (from Week 5 Homework Problem 4) with $T = 100$ steps and a linear schedule.
2. Train an NCSN with $L = 100$ noise levels, geometrically spaced, chosen so that the effective noise levels match the DDPM schedule: $\sigma\_i = \sqrt{(1-\bar{\alpha}\_i)/\bar{\alpha}\_i}$.
3. Use the same MLP architecture for both (3 hidden layers, 256 units, SiLU), with identical capacity.
4. Train both for the same number of gradient steps (10000).

### Part (b): Compare Samples

Generate 5000 samples from each model:
- DDPM: using Algorithm 2 from Ho et al.
- NCSN: using annealed Langevin dynamics with $K = 10$ steps per noise level.

Plot the samples side by side. Compute:
1. Wasserstein-2 distance to the true distribution (use the `scipy.stats.wasserstein_distance` for 1D marginals, or compute on a 2D grid).
2. Mode coverage: what fraction of the 8 modes have at least 100 samples within 3 standard deviations?

### Part (c): Convert Between Parameterizations

Take your trained DDPM noise predictor $\varepsilon\_\theta$ and convert it to a score network: $s\_{\text{converted}}(x, t) = -\varepsilon\_\theta(x, t)/\sqrt{1-\bar{\alpha}\_t}$.

Use $s\_{\text{converted}}$ for annealed Langevin dynamics. Compare the samples to those from the directly-trained NCSN. They should be similar (both are trained with equivalent objectives).

---

## Problem 6: Multi-Scale Score Fields (Theory + Visualization)

### Part (a): Analytic Score at Multiple Scales (Theory)

Consider a 2D mixture: $p(x) = 0.5\,\mathcal{N}(x; \mu\_1, 0.3^2 I) + 0.5\,\mathcal{N}(x; \mu\_2, 0.3^2 I)$ where $\mu\_1 = (-2, 0)$ and $\mu\_2 = (2, 0)$.

1. Compute the exact score $\nabla\_x \log p\_\sigma(x)$ for the noised distribution $p\_\sigma = p * \mathcal{N}(0, \sigma^2 I)$. *Hint: each component becomes $\mathcal{N}(\mu\_k, (0.3^2 + \sigma^2)I)$.*

2. Evaluate the score at $x = (0, 0)$ for $\sigma \in \lbrace 0.1, 0.5, 1, 2, 5\rbrace $. At what $\sigma$ does the score at the origin become negligible? What does this tell you about Langevin mixing between modes?

### Part (b): Visualization (Implementation)

For the same distribution:

1. Plot the score field $\nabla\_x \log p\_\sigma(x)$ as a quiver plot over $[-5, 5]^2$ for $\sigma \in \lbrace 0.1, 0.5, 1.0, 3.0\rbrace $. Use a 2x2 grid.

2. At $\sigma = 0.1$: the score should point toward the nearest mode, with a "watershed" boundary between them.
   At $\sigma = 3.0$: the score should point toward the overall center of mass, with nearly unimodal behavior.

3. Overlay Langevin trajectories (10 chains of 500 steps each) on each score field. At high noise, chains should mix freely between modes. At low noise, chains should be trapped in whichever mode they start near.

---

## Submission Checklist

- [ ] Problem 1: Score estimation failure in low-density regions, analytic + empirical demonstration, noise fix
- [ ] Problem 2: NCSN implementation, training loss curve, score field visualizations at all noise levels
- [ ] Problem 3: Annealed Langevin dynamics, effect of $K$, annealing visualization
- [ ] Problem 4: Full DDPM-NCSN equivalence proof (score, loss, sampling)
- [ ] Problem 5: Head-to-head comparison, parameterization conversion
- [ ] Problem 6: Multi-scale score analysis and visualization

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs and plots.
