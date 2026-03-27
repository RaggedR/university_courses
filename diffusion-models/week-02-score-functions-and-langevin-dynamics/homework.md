# Week 2: Score Functions and Langevin Dynamics -- Homework

**Estimated time:** 10-12 hours
**Prerequisites:** Week 1 (Gaussians, forward process), multivariable calculus (gradients, divergence), PyTorch

---

## Problem 1: Computing Score Functions Analytically (Theory)

### Part (a): Single Gaussian

Let $p(x) = \mathcal{N}(x; \mu, \Sigma)$ be a multivariate Gaussian in $\mathbb{R}^d$.

1. Derive the score function $\nabla\_x \log p(x)$. Show that $\nabla\_x \log p(x) = -\Sigma^{-1}(x - \mu)$.

2. For the special case $\mu = 0$, $\Sigma = \sigma^2 I$, sketch the score field in 2D. At $x = (3, 1)^\top$ with $\sigma = 1$, what is the score vector? In which direction does it point?

3. Where is the score zero? What is the geometric significance of this point?

### Part (b): Gaussian Mixture

Consider the mixture of two isotropic Gaussians in $\mathbb{R}^d$:

$$
p(x) = \pi_1 \mathcal{N}(x; \mu_1, \sigma^2 I) + \pi_2 \mathcal{N}(x; \mu_2, \sigma^2 I)
$$

where $\pi\_1 + \pi\_2 = 1$.

1. Derive the score function $\nabla\_x \log p(x)$. Express your answer in terms of the **responsibilities** $r\_1(x) = \frac{\pi\_1 \mathcal{N}(x; \mu\_1, \sigma^2 I)}{p(x)}$ and $r\_2(x) = 1 - r\_1(x)$.

2. Show that the score is a **responsibility-weighted average** of the individual component scores:

$$
\nabla_x \log p(x) = r_1(x) \cdot \nabla_x \log \mathcal{N}(x; \mu_1, \sigma^2 I) + r_2(x) \cdot \nabla_x \log \mathcal{N}(x; \mu_2, \sigma^2 I)
$$

3. Consider the 1D case with $\pi\_1 = \pi\_2 = 0.5$, $\mu\_1 = -5$, $\mu\_2 = 5$, $\sigma = 1$. Compute the score at $x = 0$, $x = 5$, $x = -5$, and $x = 100$. Interpret each value.

### Part (c): The Effect of Noise

Let $p\_\sigma(x) = \int p\_{\text{data}}(x\_0) \mathcal{N}(x; x\_0, \sigma^2 I) dx\_0$ be the data distribution convolved with Gaussian noise.

1. For the mixture in Part (b), what is $p\_\sigma(x)$? (*Hint: convolving a Gaussian with a Gaussian gives a Gaussian with summed variances.*)

2. What happens to the score of $p\_\sigma(x)$ as $\sigma \to \infty$? As $\sigma \to 0$?

3. At what value of $\sigma$ (approximately) does the mixture $p\_\sigma$ become effectively unimodal? (*Hint: the two modes merge when the standard deviation of the convolved components exceeds half the distance between the means.*)

---

## Problem 2: Deriving the Score Matching Identity (Theory)

### Part (a): The 1D Case

Let $p(x)$ be a density on $\mathbb{R}$ and $s\_\theta(x)$ a scalar-valued score model. Prove the score matching identity in 1D:

$$
\mathbb{E}_p\left[(s_\theta(x) - \nabla_x \log p(x))^2\right] = \mathbb{E}_p\left[s_\theta'(x) + \frac{1}{2}s_\theta(x)^2\right] + C
$$

where $C$ is a constant independent of $\theta$ and $s\_\theta'(x) = ds\_\theta/dx$.

Steps:
1. Expand the squared difference on the left side.
2. Isolate the cross-term $\mathbb{E}\_p[s\_\theta(x) \cdot p'(x)/p(x)]$.
3. Rewrite the cross-term as $\int s\_\theta(x) p'(x) dx$.
4. Apply integration by parts, assuming $s\_\theta(x) p(x) \to 0$ as $|x| \to \infty$.
5. Combine terms to obtain the right side.

### Part (b): Verify on a Known Distribution

Let $p(x) = \mathcal{N}(x; 0, 1)$ and $s\_\theta(x) = -\theta x$ (a linear score model parameterized by $\theta$).

1. Compute the true score matching objective $J(\theta) = \frac{1}{2}\mathbb{E}\_p[(s\_\theta(x) - \nabla\_x \log p(x))^2]$ as a function of $\theta$. Find the minimizer.

2. Compute the Hyvarinen score matching objective $J\_{\text{SM}}(\theta) = \mathbb{E}\_p[s\_\theta'(x) + \frac{1}{2}s\_\theta(x)^2]$ as a function of $\theta$. Find the minimizer.

3. Verify that both minimizers agree. (They must, by the identity you proved in Part (a).)

### Part (c): The Multivariate Case

Extend the identity to $\mathbb{R}^d$. State clearly where you use the divergence theorem (the multivariate analog of integration by parts). What boundary conditions are required?

---

## Problem 3: Langevin Dynamics for a 2D Gaussian (Implementation)

### Part (a): Single Gaussian

Implement Langevin dynamics to sample from $p(x) = \mathcal{N}(x; \mu, \Sigma)$ with $\mu = (3, -2)^\top$ and $\Sigma = \begin{pmatrix} 2 & 0.8 \\ 0.8 & 1 \end{pmatrix}$.

```python
def langevin_dynamics(score_fn, x_init, step_size, num_steps):
    """
    Run Langevin dynamics.

    Args:
        score_fn: Function that takes x (shape (batch, d)) and returns the score (same shape)
        x_init: Initial samples, shape (batch, d)
        step_size: Step size eta
        num_steps: Number of Langevin steps

    Returns:
        trajectory: All samples, shape (num_steps+1, batch, d)
    """
    pass
```

1. Implement the score function for the Gaussian analytically (use the formula from Problem 1a).
2. Initialize 500 samples from $\mathcal{N}(0, 10I)$ (far from the target).
3. Run Langevin for 1000 steps with step size $\eta = 0.01$.
4. Plot: (i) a scatter plot of the initial samples, (ii) a scatter plot of the final samples overlaid with the true density contours, (iii) the trajectory of 5 individual samples.

### Part (b): Step Size Sensitivity

Run Langevin dynamics with step sizes $\eta \in \lbrace 0.001, 0.01, 0.1, 0.5, 1.0\rbrace$ for 2000 steps each. For each:

1. Plot the final sample distribution.
2. Compute the sample mean and sample covariance. Compare to the true $\mu$ and $\Sigma$.
3. At what step size does the algorithm diverge? Why?

(*Hint: the stability condition for Langevin dynamics is roughly $\eta < 2 / L$ where $L$ is the Lipschitz constant of the score. For a Gaussian with covariance $\Sigma$, $L = \Vert \Sigma^{-1}\Vert \_{\text{op}}$ -- the largest eigenvalue of $\Sigma^{-1}$.*)

---

## Problem 4: Langevin Dynamics for a 2D Mixture (Implementation)

### Part (a): The Target Distribution

Define a mixture of 4 Gaussians in $\mathbb{R}^2$:

$$
p(x) = \frac{1}{4}\sum_{i=1}^{4} \mathcal{N}(x; \mu_i, 0.5^2 I)
$$

with means $\mu\_1 = (5, 5)$, $\mu\_2 = (5, -5)$, $\mu\_3 = (-5, -5)$, $\mu\_4 = (-5, 5)$ (four corners of a square).

1. Implement the score function for this mixture analytically using the result from Problem 1(b).
2. Visualize the score field: on a grid of points in $[-8, 8]^2$, plot the score vectors as arrows (use `plt.quiver`). Overlay the density contours.

### Part (b): Langevin Sampling

Run Langevin dynamics with 2000 samples initialized from $\mathcal{N}(0, 10I)$, step size $\eta = 0.01$, for 5000 steps.

1. Display scatter plots of the samples at steps $\lbrace 0, 100, 500, 1000, 5000\rbrace$.
2. Are all four modes discovered? If not, run for longer (20000 steps) and check again.
3. Compute the fraction of samples that end up in each mode (define "in mode $i$" as $\Vert x - \mu\_i\Vert < 2$). Is the distribution across modes approximately uniform ($\frac{1}{4}$ each)?

### Part (c): The Mixing Problem

Now increase the separation. Set $\mu\_1 = (15, 15)$, $\mu\_2 = (15, -15)$, $\mu\_3 = (-15, -15)$, $\mu\_4 = (-15, 15)$ (keeping $\sigma = 0.5$).

1. Repeat Part (b). How many modes are discovered after 5000 steps? After 50000 steps?
2. Explain why Langevin dynamics struggles with well-separated modes. What is the energy barrier between modes, approximately?

### Part (d): Annealed Langevin Dynamics

Implement annealed Langevin dynamics for the well-separated mixture from Part (c):

1. Define noise levels $\sigma\_1 = 20, \sigma\_2 = 10, \sigma\_3 = 5, \sigma\_4 = 2, \sigma\_5 = 1, \sigma\_6 = 0.5$.
2. At each noise level $\sigma\_i$, the target is $p\_{\sigma\_i}(x) = \frac{1}{4}\sum\_{j=1}^4 \mathcal{N}(x; \mu\_j, (\sigma^2 + \sigma\_i^2)I)$ (data convolved with $\mathcal{N}(0, \sigma\_i^2 I)$).
3. Run 500 Langevin steps at each noise level, using the final samples from level $i$ as initialization for level $i+1$.
4. Display scatter plots at the end of each noise level. Does the annealed version discover all four modes?

---

## Problem 5: Denoising Score Matching (Theory + Implementation)

### Part (a): Derivation

The denoising score matching objective is:

$$
J_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{x_0 \sim p_{\text{data}}} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\left[\left\Vert s_\theta(x_0 + \sigma\epsilon) - \left(-\frac{\epsilon}{\sigma}\right)\right\Vert ^2\right]
$$

Prove that this is equivalent to the score matching objective (up to a constant) by showing that the optimal $s\_\theta^*$ satisfies $s\_\theta^*(x) = \nabla\_x \log q\_\sigma(x)$, where $q\_\sigma(x) = \int p\_{\text{data}}(x\_0) \mathcal{N}(x; x\_0, \sigma^2 I) dx\_0$.

(*Hint: The key identity is $\nabla\_x \log q\_\sigma(x) = \mathbb{E}\_{q(x\_0 \mid x)}\left[\frac{x\_0 - x}{\sigma^2}\right] = -\frac{1}{\sigma}\mathbb{E}\_{q(x\_0 \mid x)}[\epsilon]$ where $\epsilon = (x - x\_0)/\sigma$.*)

### Part (b): Implementation

Train a small neural network to estimate the score of the 4-Gaussian mixture from Problem 4(a) using denoising score matching.

1. Use a 3-layer MLP with hidden size 128 and ReLU activations. The input is $x \in \mathbb{R}^2$ and the output is $s\_\theta(x) \in \mathbb{R}^2$.

2. Training loop (for a fixed noise level $\sigma = 1.0$):
   ```
   For each batch:
       Sample x_0 from the mixture (use the analytic density to sample)
       Sample epsilon ~ N(0, I)
       x_noisy = x_0 + sigma * epsilon
       loss = ||s_theta(x_noisy) - (-epsilon / sigma)||^2
       Update theta
   ```

3. Train for 5000 steps with batch size 256 and learning rate $10^{-3}$.

4. Visualize: plot the learned score field $s\_\theta(x)$ as arrows on a grid, next to the true score field of $p\_\sigma(x)$. How well does the learned score match the true score?

### Part (c): Score-Based Sampling

Use your learned score network to run Langevin dynamics:

$$
x_{k+1} = x_k + \eta \, s_\theta(x_k) + \sqrt{2\eta} \, z_k
$$

Initialize 1000 samples from $\mathcal{N}(0, 10I)$ and run for 5000 steps with $\eta = 0.01$.

1. Plot the resulting samples. Do they match the target mixture?
2. Compare to the result from Problem 4(b) where you used the true score. How does using a learned score affect sample quality?

---

## Problem 6: Connecting Scores to Diffusion (Theory + Implementation)

### Part (a): The Score-Noise Equivalence

In the diffusion forward process, $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1-\bar{\alpha}\_t} \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$.

1. Compute the score $\nabla\_{x\_t} \log q(x\_t \mid x\_0)$. Express it in terms of $\epsilon$, $x\_t$, $x\_0$, and $\bar{\alpha}\_t$.

2. A network $\epsilon\_\theta(x\_t, t)$ is trained to predict $\epsilon$ from $x\_t$. Show that the corresponding score estimate is:

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

3. A network $x\_{0,\theta}(x\_t, t)$ is trained to predict $x\_0$ from $x\_t$ (the "data prediction" parameterization). Show that the corresponding score estimate is:

$$
s_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_t} \, x_{0,\theta}(x_t, t) - x_t}{1 - \bar{\alpha}_t}
$$

4. Show that all three parameterizations ($\epsilon$-prediction, $x\_0$-prediction, score prediction) are equivalent: given any one, you can recover the other two.

### Part (b): Forward Process as Score Destruction

Using your forward process implementation from Week 1, Problem 4:

1. Take a batch of 1000 MNIST images (or a simple 2D distribution). At each timestep $t$, estimate the score $\nabla\_{x\_t} \log q(x\_t)$ using the empirical distribution of $x\_t$ values.

    For 2D: compute the score on a grid by kernel density estimation.

2. Visualize the score field at $t \in \lbrace 0, 100, 500, 999\rbrace$. How does the score field change as noise is added?

3. At $t = 999$, compare the estimated score field to $-x\_t$ (the score of $\mathcal{N}(0, I)$). They should approximately match, confirming that the forward process has mixed to its stationary distribution.

---

## Submission Checklist

- [ ] Problem 1: Score function derivations for Gaussian, mixture, and noisy distributions
- [ ] Problem 2: Score matching identity proof (1D and multivariate), verification on Gaussian
- [ ] Problem 3: Langevin dynamics implementation, sampling from 2D Gaussian, step size analysis
- [ ] Problem 4: Langevin for mixtures, mixing time analysis, annealed Langevin implementation
- [ ] Problem 5: Denoising score matching derivation, neural score network training, score-based sampling
- [ ] Problem 6: Score-noise equivalence derivation, three parameterizations, score field visualization

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs.
