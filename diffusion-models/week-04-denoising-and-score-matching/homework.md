---
title: "Week 4: Denoising and Score Matching -- Homework"
---

# Week 4: Denoising and Score Matching -- Homework

**Estimated time:** 12-15 hours
**Prerequisites:** Score functions (Week 2), SDEs and OU process (Week 3), PyTorch, basic probability

---

## Problem 1: Tweedie's Formula for Gaussians (Theory)

### Part (a): Univariate Case

Let $x \sim \mathcal{N}(\mu, \tau^2)$ and $\tilde{x} = x + \sigma\epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$.

1. Compute the marginal distribution $p\_\sigma(\tilde{x})$ by convolving the prior with the noise. Show that $\tilde{x} \sim \mathcal{N}(\mu, \tau^2 + \sigma^2)$.

2. Compute the posterior $p(x | \tilde{x})$ using Bayes' theorem. Show that it is Gaussian with:
   $$
   \mathbb{E}[x | \tilde{x}] = \frac{\tau^2 \tilde{x} + \sigma^2 \mu}{\tau^2 + \sigma^2}
   $$

3. Compute $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$ directly from the marginal.

4. Verify Tweedie's formula: show that $\mathbb{E}[x|\tilde{x}] = \tilde{x} + \sigma^2 \nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$.

### Part (b): Mixture of Gaussians

Now let $x \sim \frac{1}{2}\mathcal{N}(-3, 1) + \frac{1}{2}\mathcal{N}(3, 1)$ (a bimodal distribution) and $\tilde{x} = x + \sigma\epsilon$.

1. Write the marginal $p\_\sigma(\tilde{x})$ as a mixture of Gaussians. Compute $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$ in closed form.

2. Compute $\mathbb{E}[x|\tilde{x}]$ using the posterior. *Hint: the posterior is also a mixture of Gaussians, with weights that depend on $\tilde{x}$.*

3. Verify Tweedie's formula numerically: for $\sigma \in \lbrace 0.1, 0.5, 1.0, 2.0, 5.0\rbrace$ and $\tilde{x} \in \lbrace -5, -3, -1, 0, 1, 3, 5\rbrace$, compute both sides and check they agree.

4. Plot the score function $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$ as a function of $\tilde{x}$ for each $\sigma$. How does the score change as noise increases? At what noise level does the bimodal structure disappear from the score?

### Part (c): General Proof

Prove Tweedie's formula for a general (not necessarily Gaussian) data distribution $p(x)$ in $d$ dimensions.

Starting from $p\_\sigma(\tilde{x}) = \int p(x) \mathcal{N}(\tilde{x}; x, \sigma^2 I) dx$:

1. Compute $\nabla\_{\tilde{x}} p\_\sigma(\tilde{x})$ by differentiating under the integral sign.
2. Show that $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x}) = \frac{1}{\sigma^2}(\mathbb{E}[x|\tilde{x}] - \tilde{x})$.
3. Rearrange to obtain Tweedie's formula.

---

## Problem 2: The Denoising Score Matching Equivalence (Theory)

### Part (a): The Equivalence

Prove that the denoising score matching loss:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{x \sim p, \, \epsilon \sim \mathcal{N}(0,I)}\left[\left\Vert s_\theta(x + \sigma\epsilon) + \frac{\epsilon}{\sigma}\right\Vert ^2\right]
$$

equals the true score matching loss:

$$
\mathcal{L}_{\text{SM}}(\theta) = \frac{1}{2}\mathbb{E}_{\tilde{x} \sim p_\sigma}\left[\Vert s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})\Vert ^2\right]
$$

up to a constant independent of $\theta$.

Follow these steps:
1. Expand $\mathcal{L}\_{\text{SM}}$ into three terms: $\Vert s\_\theta\Vert ^2$, cross term, and $\Vert \nabla \log p\_\sigma\Vert ^2$.
2. Show that the cross term can be rewritten as $-\mathbb{E}\_{x, \tilde{x}}[s\_\theta(\tilde{x})^\top \nabla\_{\tilde{x}} \log p(\tilde{x}|x)]$ using the identity $p\_\sigma(\tilde{x}) = \int p(x) p(\tilde{x}|x) dx$.
3. Expand $\mathcal{L}\_{\text{DSM}}$ and compare term by term.

### Part (b): The Minimizer

Show that the minimizer of $\mathcal{L}\_{\text{DSM}}$ satisfies:

$$
s_\theta^*(\tilde{x}) = \nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) = -\frac{\mathbb{E}[\epsilon | \tilde{x}]}{\sigma}
$$

That is, the optimal score network predicts the conditional expectation of the normalized noise.

### Part (c): Three Parameterizations

Given the equivalences between the score, noise, and denoiser parameterizations:

$$
s_\theta(\tilde{x}) = -\frac{\epsilon_\theta(\tilde{x})}{\sigma} = \frac{D_\theta(\tilde{x}) - \tilde{x}}{\sigma^2}
$$

Write the denoising score matching loss in terms of:
1. The noise predictor $\epsilon\_\theta$: show that $\mathcal{L} = \frac{1}{\sigma^2}\mathbb{E}\Vert \epsilon\_\theta - \epsilon\Vert ^2 + C$.
2. The denoiser $D\_\theta$: show that $\mathcal{L} = \frac{1}{\sigma^4}\mathbb{E}\Vert D\_\theta(x + \sigma\epsilon) - x\Vert ^2 + C'$.

Which parameterization seems most natural from the perspective of training stability? (Argue based on the scale of the prediction targets.)

---

## Problem 3: Implementing a Score Network for 2D Data (Implementation)

### Part (a): Data and Noising

Create a 2D dataset (the same mixture of 8 Gaussians from Week 3, Problem 6, or a different interesting distribution like a Swiss roll or concentric circles).

Write a function that:
1. Samples a batch of data points $x\_0$
2. Samples a noise level $\sigma$ from a predefined set $\lbrace \sigma\_1, \ldots, \sigma\_L\rbrace$ (use $L = 10$ levels geometrically spaced from $\sigma\_1 = 0.01$ to $\sigma\_L = 5.0$)
3. Adds noise: $\tilde{x} = x\_0 + \sigma\epsilon$
4. Returns $\tilde{x}$, $\sigma$, and the target $-\epsilon/\sigma$

### Part (b): Architecture

Implement a simple noise-conditional score network $s\_\theta(\tilde{x}, \sigma)$:

```python
class ScoreNet(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128, num_layers=3):
        ...
        # Network takes (x, sigma_embedding) and outputs score of same dim as x

    def forward(self, x, sigma):
        # Encode sigma using sinusoidal embedding
        # Concatenate with x (or use FiLM conditioning)
        # Pass through MLP
        # Output: predicted score, shape (batch, data_dim)
        ...
```

For the sinusoidal embedding, use:
$$
\gamma(\sigma) = [\sin(\omega_1 \log\sigma), \cos(\omega_1 \log\sigma), \ldots, \sin(\omega_K \log\sigma), \cos(\omega_K \log\sigma)]
$$
with $K = 32$ and $\omega\_k = 2^{k/4}$.

### Part (c): Training

Train the score network using the denoising score matching loss:

$$
\mathcal{L} = \mathbb{E}_{\sigma, x_0, \epsilon}\left[\sigma^2 \left\Vert s_\theta(x_0 + \sigma\epsilon, \sigma) + \frac{\epsilon}{\sigma}\right\Vert ^2\right]
$$

Note the $\sigma^2$ weighting (so the loss is equivalent to $\Vert \epsilon\_\theta - \epsilon\Vert ^2$).

Train for 10000-20000 steps with Adam (lr=1e-3, batch size=256).

Plot the training loss over time. Does it converge?

### Part (d): Visualize the Learned Score

After training, visualize the learned score field $s\_\theta(x, \sigma)$ as a 2D vector field on a grid:
1. At low noise ($\sigma = 0.1$): the score should point toward the nearest cluster center
2. At medium noise ($\sigma = 1.0$): the score should show broader structure
3. At high noise ($\sigma = 5.0$): the score should point toward the origin

Compare with the analytical score from Week 3, Problem 6 (if you used the same distribution).

---

## Problem 4: Sampling with the Learned Score (Implementation)

### Part (a): Annealed Langevin Dynamics

Using the trained score network from Problem 3, implement annealed Langevin dynamics (Song and Ermon 2019):

```
For l = L, L-1, ..., 1:  (from highest to lowest noise)
    For i = 1, ..., N:  (Langevin steps at each noise level)
        x = x + (step_size/2) * s_theta(x, sigma_l) + sqrt(step_size) * z
        where z ~ N(0, I)
```

Use $N = 100$ Langevin steps per noise level, with step size $\alpha\_\ell = c \cdot \sigma\_\ell^2 / \sigma\_L^2$ where $c = 0.01$.

Start from $x \sim \mathcal{N}(0, \sigma\_L^2 I)$. Generate 5000 samples and plot them alongside the true data distribution.

### Part (b): Reverse SDE Sampling

Implement sampling via the discretized reverse SDE (Anderson's theorem):

```
Start with x_T ~ N(0, I * sigma_T^2)
For t = T, T-dt, T-2dt, ..., dt:
    x = x + [f(x,t) - g(t)^2 * s_theta(x, sigma(t))] * (-dt) + g(t) * sqrt(dt) * z
```

Use 1000 reverse steps from $t = 1$ to $t = 0$.

Generate 5000 samples and compare with:
1. The true data distribution
2. The annealed Langevin samples from Part (a)
3. The "oracle" reverse SDE samples from Week 3, Problem 6 (using the analytical score)

### Part (c): Quality Metrics

Quantify the sample quality:
1. Compare the sample mean and covariance to the true data distribution's moments
2. If using the mixture of Gaussians: estimate the weight assigned to each mode (what fraction of samples are nearest to each cluster center). Are all modes equally represented?
3. Plot kernel density estimates (KDE) of the generated and true distributions. How well do they match?

---

## Problem 5: Tweedie's Formula in Action (Implementation)

### Part (a): 1D Denoising

Take $p(x) = \frac{1}{2}\mathcal{N}(-2, 0.5^2) + \frac{1}{2}\mathcal{N}(2, 0.5^2)$.

For $\sigma \in \lbrace 0.1, 0.5, 1.0, 2.0\rbrace$:
1. Sample 10000 points from $p(x)$, add noise to get $\tilde{x}$
2. Compute the analytical optimal denoiser $D^*(\tilde{x}) = \mathbb{E}[x|\tilde{x}]$ (using the mixture of Gaussians posterior)
3. Compute the analytical score $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$
4. Verify that $D^*(\tilde{x}) = \tilde{x} + \sigma^2 \nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$

Plot, for each $\sigma$:
- The true $x$ vs. $\tilde{x}$ (scatter plot)
- The denoised $D^*(\tilde{x})$ vs. $\tilde{x}$ (curve)
- The identity line $\tilde{x}$ vs. $\tilde{x}$ (for comparison)

Observe: at low noise, the denoiser is nearly the identity. At high noise, the denoiser shrinks everything toward the two modes.

### Part (b): The Denoiser Is a Score Estimator

Using the score network from Problem 3 (trained on 2D data), extract the implied denoiser:

$$
D_\theta(\tilde{x}, \sigma) = \tilde{x} + \sigma^2 s_\theta(\tilde{x}, \sigma)
$$

For each noise level $\sigma$:
1. Take 1000 noisy data points $\tilde{x} = x\_0 + \sigma\epsilon$
2. Compute the neural network denoiser $D\_\theta(\tilde{x}, \sigma)$
3. Compute the reconstruction MSE: $\frac{1}{N}\sum\_n \Vert D\_\theta(\tilde{x}\_n) - x\_{0,n}\Vert ^2$
4. Compare to the theoretical minimum MSE (the posterior variance, for Gaussian mixtures this can be computed analytically)

Plot the MSE vs. $\sigma$. At what noise level is the network most effective? At what noise level is the gap between the network and the optimal denoiser largest?

---

## Problem 6: Multi-Scale Score Network from Scratch (Implementation)

This is the culminating problem: build a complete noise-conditional score network and train it on a more challenging distribution.

### Part (a): Swiss Roll Data

Generate 2D Swiss roll data:

```python
def sample_swiss_roll(n: int) -> torch.Tensor:
    t = 1.5 * torch.pi * (1 + 2 * torch.rand(n))
    x = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
    x = x + 0.3 * torch.randn(n, 2)
    x = x / 5.0  # Scale to reasonable range
    return x
```

### Part (b): Continuous Noise Schedule

Instead of a discrete set of noise levels, use a continuous noise schedule. Sample $t \sim \mathcal{U}[0, 1]$ and define:

$$
\sigma(t) = \sigma_{\min}^{1-t} \cdot \sigma_{\max}^t
$$

with $\sigma\_{\min} = 0.01$ and $\sigma\_{\max} = 10.0$ (geometric interpolation in log space).

### Part (c): Training

Train a noise-conditional score network with:
- Architecture: MLP with 4 hidden layers of 256 units, SiLU activations, sinusoidal time embedding
- Loss: $\mathcal{L} = \mathbb{E}\_{t, x\_0, \epsilon}[\sigma(t)^2 \Vert s\_\theta(\alpha(t) x\_0 + \sigma(t)\epsilon, t) + \epsilon/\sigma(t)\Vert ^2]$
- 50000 training steps, batch size 512, Adam with lr=3e-4

Record the loss at each step. Plot the loss curve and confirm convergence.

### Part (d): Generation

Generate 10000 samples using the reverse SDE with 1000 discretization steps. Plot:
1. The generated samples overlaid on the true Swiss roll data
2. Intermediate samples at $t \in \lbrace 1.0, 0.8, 0.5, 0.2, 0.05, 0.0\rbrace$ (six subplots showing the progressive refinement from noise to data)

### Part (e): Analysis

1. **Score field evolution:** Plot the learned score field at $t \in \lbrace 0.01, 0.1, 0.5, 0.9\rbrace$. At high noise, the score should point toward the center of the spiral. At low noise, the score should point along the spiral toward the nearest data points.

2. **Effect of steps:** Generate samples using $\lbrace 10, 50, 100, 500, 1000, 5000\rbrace$ reverse SDE steps. How many steps are needed for reasonable quality? Plot the sample quality vs. number of steps.

3. **Probability flow ODE:** Implement the probability flow ODE from Week 3 notes:
   $$
   \frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 s_\theta(x, t)
   $$
   Generate samples using the ODE (with an Euler integrator, 1000 steps). Compare the samples to those from the reverse SDE. Are they similar? Which gives better samples?

---

## Submission Checklist

- [ ] Problem 1: Tweedie's formula derivations for univariate Gaussian, mixture of Gaussians, and general distributions
- [ ] Problem 2: Denoising score matching equivalence proof, three parameterizations
- [ ] Problem 3: Score network implementation, training, and score field visualization
- [ ] Problem 4: Annealed Langevin and reverse SDE sampling with quality comparison
- [ ] Problem 5: Tweedie's formula verification (analytical and neural network denoiser)
- [ ] Problem 6: Full pipeline on Swiss roll -- continuous noise, training, generation, analysis

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs and plots.
