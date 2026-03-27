# Week 5: Denoising Diffusion Probabilistic Models -- Homework

**Estimated time:** 12-15 hours
**Prerequisites:** Gaussian distributions, KL divergence (Week 1), score matching (Week 4), PyTorch

---

## Problem 1: Derive the Closed-Form Forward Process (Theory)

### Part (a): The Inductive Proof

Prove that $q(x\_t \mid x\_0) = \mathcal{N}\!\left(x\_t;\; \sqrt{\bar{\alpha}\_t}\, x\_0,\; (1 - \bar{\alpha}\_t)\, I\right)$ where $\bar{\alpha}\_t = \prod\_{s=1}^{t}(1 - \beta\_s)$.

Proceed by induction on $t$:

1. **Base case ($t = 1$):** Write $q(x\_1 \mid x\_0)$ from the forward process definition. Verify that it matches the claimed formula with $\bar{\alpha}\_1 = 1 - \beta\_1$.

2. **Inductive step:** Assume the formula holds for $t-1$. Using the reparameterization $x\_{t-1} = \sqrt{\bar{\alpha}\_{t-1}}\, x\_0 + \sqrt{1 - \bar{\alpha}\_{t-1}}\, \varepsilon\_1$ and $x\_t = \sqrt{1 - \beta\_t}\, x\_{t-1} + \sqrt{\beta\_t}\, \varepsilon\_2$ (with $\varepsilon\_1, \varepsilon\_2$ independent standard Gaussians), substitute to express $x\_t$ in terms of $x\_0$ and show that the result is Gaussian with the claimed mean and variance.

*Hint: when you combine two independent Gaussian noise terms $a \varepsilon\_1 + b \varepsilon\_2$, the result is Gaussian with variance $a^2 + b^2$. You will need to verify that $\alpha\_t(1 - \bar{\alpha}\_{t-1}) + \beta\_t = 1 - \bar{\alpha}\_t$.*

### Part (b): Signal-to-Noise Ratio

Define the signal-to-noise ratio at time $t$ as:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

1. Show that $\text{SNR}(t)$ is monotonically decreasing with $t$ (assuming $\beta\_t > 0$ for all $t$).

2. For the linear schedule $\beta\_t = \beta\_{\min} + \frac{t-1}{T-1}(\beta\_{\max} - \beta\_{\min})$ with $\beta\_{\min} = 10^{-4}$, $\beta\_{\max} = 0.02$, $T = 1000$, compute and plot $\text{SNR}(t)$ on a log scale. At which timestep does $\text{SNR}(t) = 1$ (equal parts signal and noise)?

3. Repeat for the cosine schedule: $\bar{\alpha}\_t = \frac{f(t/T)}{f(0)}$ where $f(s) = \cos^2\!\left(\frac{s + 0.008}{1.008} \cdot \frac{\pi}{2}\right)$. How does the SNR profile differ? Why might this produce better results?

---

## Problem 2: Derive the ELBO Decomposition (Theory)

Starting from the evidence lower bound:

$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} \mid x_0)}\!\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right]
$$

### Part (a): Expand and Regroup

1. Write out $\log p\_\theta(x\_{0:T})$ using the definition of the reverse process.
2. Write out $\log q(x\_{1:T} \mid x\_0)$ using the Markov property of the forward process.
3. Substitute both into the ELBO and show that it can be rewritten as:

$$
-L_{\text{VLB}} = \mathbb{E}_q\!\left[\log \frac{p(x_T)}{q(x_T \mid x_0)} + \sum_{t=2}^{T} \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_{t-1} \mid x_t, x_0)} + \log p_\theta(x_0 \mid x_1)\right]
$$

*Hint: use Bayes' rule to rewrite $q(x\_t \mid x\_{t-1}) = \frac{q(x\_{t-1} \mid x\_t, x\_0)\, q(x\_t \mid x\_0)}{q(x\_{t-1} \mid x\_0)}$ and telescope the resulting products.*

### Part (b): Identify the KL Divergences

Show that the expression from Part (a) equals:

$$
-L_{\text{VLB}} = -D_{\text{KL}}(q(x_T \mid x_0) \Vert p(x_T)) - \sum_{t=2}^{T} D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \Vert p_\theta(x_{t-1} \mid x_t)) + \mathbb{E}_q[\log p_\theta(x_0 \mid x_1)]
$$

### Part (c): The Reverse Posterior

Derive $q(x\_{t-1} \mid x\_t, x\_0)$ by computing:

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$

All three distributions on the right are Gaussians. Complete the square in the exponent to show:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\; \tilde{\mu}_t,\; \tilde{\beta}_t I\right)
$$

and derive the expressions for $\tilde{\mu}\_t$ and $\tilde{\beta}\_t$ given in the notes.

### Part (d): From Mean Prediction to Noise Prediction

Starting from the KL term $L\_{t-1} = \frac{1}{2\sigma\_t^2}\Vert \tilde{\mu}\_t - \mu\_\theta(x\_t, t)\Vert ^2$, substitute $x\_0 = \frac{1}{\sqrt{\bar{\alpha}\_t}}(x\_t - \sqrt{1-\bar{\alpha}\_t}\,\varepsilon)$ into $\tilde{\mu}\_t$ and show that the loss can be written as:

$$
L_{t-1} = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1 - \bar{\alpha}_t)} \Vert \varepsilon - \varepsilon_\theta(x_t, t)\Vert ^2
$$

where $\varepsilon\_\theta$ is the noise-prediction network.

---

## Problem 3: Implement DDPM Training from Scratch (Implementation)

Implement the complete DDPM training pipeline in PyTorch.

### Part (a): The Noise Schedule

```python
def linear_beta_schedule(T, beta_min=1e-4, beta_max=0.02):
    """Return betas, alphas, alpha_bars for timesteps 1..T."""
    # TODO: compute beta_t, alpha_t, alpha_bar_t
    # Return as tensors of shape (T,)
    pass
```

Implement this function. Also implement a `cosine_beta_schedule(T)` following Nichol and Dhariwal (2021).

Plot $\bar{\alpha}\_t$ for both schedules. Verify that $\bar{\alpha}\_T \approx 0$ (the signal is destroyed by the end).

### Part (b): The Forward Process

```python
def q_sample(x_0, t, alpha_bar, noise=None):
    """
    Sample x_t from q(x_t | x_0).

    Args:
        x_0: clean data, shape (B, C, H, W)
        t: timesteps, shape (B,), values in {0, ..., T-1}
        alpha_bar: cumulative products, shape (T,)
        noise: optional pre-sampled noise, shape (B, C, H, W)

    Returns:
        x_t: noisy data, shape (B, C, H, W)
    """
    pass
```

Implement this. Visualize the forward process: take one MNIST digit, compute $x\_t$ for $t \in \lbrace 0, 50, 100, 200, 500, 999\rbrace$, and display them side by side.

### Part (c): A Simple U-Net

Implement a small U-Net suitable for 28x28 (MNIST) or 32x32 images:

- 2-3 downsampling levels
- Residual blocks with GroupNorm and SiLU
- Sinusoidal timestep embedding, projected and added to each residual block
- Skip connections via concatenation
- ~2-5M parameters (keep it small for training on CPU/single GPU)

You may use a reference implementation for guidance, but write the code yourself and understand every layer. The key architectural requirement is that the network takes $(x\_t, t)$ as input and outputs a tensor of the same shape as $x\_t$.

### Part (d): The Training Loop

Implement Algorithm 1 from Ho et al.:

```python
def train_ddpm(model, dataloader, optimizer, alpha_bar, T, num_epochs):
    for epoch in range(num_epochs):
        for x_0, _ in dataloader:
            t = torch.randint(0, T, (x_0.shape[0],))
            noise = torch.randn_like(x_0)
            x_t = q_sample(x_0, t, alpha_bar, noise)

            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

Train on MNIST for 20-50 epochs (should take 30-60 minutes on a laptop GPU, or 2-4 hours on CPU).

Plot the training loss curve. It should decrease and plateau.

### Part (e): Sampling

Implement Algorithm 2 from Ho et al.:

```python
@torch.no_grad()
def sample_ddpm(model, shape, alpha, alpha_bar, beta, T):
    """Generate samples by running the reverse process."""
    x = torch.randn(shape)  # x_T ~ N(0, I)
    for t in reversed(range(T)):
        # TODO: implement the reverse step
        pass
    return x
```

Generate a grid of 64 samples. Do they look like MNIST digits?

If the samples are poor, try: (1) training longer, (2) using the cosine schedule, (3) increasing model capacity.

---

## Problem 4: DDPM on a 2D Dataset (Implementation)

Training a full image DDPM takes time. To build intuition, apply DDPM to a 2D distribution where we can visualize everything.

### Part (a): Setup

Create a 2D dataset: a mixture of 8 Gaussians arranged in a circle (the "Swiss roll" or "8 Gaussians" toy dataset). Each component has mean $(\cos(2\pi k/8), \sin(2\pi k/8))$ and standard deviation 0.05 for $k = 0, \ldots, 7$.

Draw 10000 samples for training.

### Part (b): 2D DDPM

Implement a DDPM where $\varepsilon\_\theta$ is a small MLP (3-4 hidden layers, 256 units, SiLU activations) conditioned on $t$ (e.g., concatenate the sinusoidal embedding of $t$ with $x\_t$).

Use $T = 100$ steps and a linear schedule.

Train for 5000-10000 gradient steps.

### Part (c): Visualization

1. **Forward process**: Plot $q(x\_t)$ for $t \in \lbrace 0, 10, 25, 50, 75, 100\rbrace$ by sampling from $q(x\_t \mid x\_0)$ for all training points. You should see the 8 clusters gradually blur into an isotropic Gaussian.

2. **Reverse process**: Starting from $x\_T \sim \mathcal{N}(0, I)$, run the reverse process and plot $x\_t$ at $t \in \lbrace 100, 75, 50, 25, 10, 0\rbrace$. You should see noise gradually organize into 8 clusters.

3. **Learned score field**: At timestep $t = 50$, plot the vector field $-\varepsilon\_\theta(x, t)/\sqrt{1-\bar{\alpha}\_t}$ (the estimated score) over a grid. The arrows should point toward the data modes.

---

## Problem 5: Ablation Studies (Implementation + Analysis)

Using your 2D DDPM from Problem 4, investigate the effect of design choices.

### Part (a): Number of Diffusion Steps

Train and sample with $T \in \lbrace 10, 50, 100, 500, 1000\rbrace$. For each:
1. Generate 5000 samples
2. Plot the samples
3. Compute the Wasserstein-2 distance to the true distribution (or approximate it by computing the distance between empirical histograms on a grid)

How does sample quality scale with $T$? Is there a point of diminishing returns?

### Part (b): The Weighting

Compare three losses:
1. $L\_{\text{simple}}$ (uniform weighting over $t$)
2. $L\_{\text{VLB}}$ (weighted by $w\_t = \frac{\beta\_t^2}{2\sigma\_t^2 \alpha\_t(1-\bar{\alpha}\_t)}$)
3. $L\_{\text{SNR}}$ (weight $w\_t = \text{SNR}(t) - \text{SNR}(t+1)$, the min-SNR weighting from Hang et al. 2023)

Train each for the same number of steps. Compare sample quality. Does the simplified loss outperform the VLB loss, as reported by Ho et al.?

### Part (c): Variance Choice

Sample with $\sigma\_t^2 = \beta\_t$ (upper bound) vs. $\sigma\_t^2 = \tilde{\beta}\_t$ (posterior variance) vs. $\sigma\_t = 0$ (deterministic, DDIM-like). Compare sample quality and diversity for each. What happens when you use $\sigma\_t = 0$?

---

## Problem 6: The Score-Noise Connection (Theory + Implementation)

### Part (a): Prove the Equivalence (Theory)

Starting from the score of the noised distribution:

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\, x_0}{1 - \bar{\alpha}_t}
$$

1. Derive this expression from $q(x\_t \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t}\, x\_0, (1-\bar{\alpha}\_t)\, I)$.

2. Using $x\_t = \sqrt{\bar{\alpha}\_t}\, x\_0 + \sqrt{1 - \bar{\alpha}\_t}\, \varepsilon$, show that:

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\varepsilon}{\sqrt{1 - \bar{\alpha}_t}}
$$

3. Explain why this means that the DDPM loss $\mathbb{E}\Vert \varepsilon - \varepsilon\_\theta(x\_t, t)\Vert ^2$ is equivalent to denoising score matching (up to a timestep-dependent constant).

### Part (b): Empirical Verification (Implementation)

Using your trained 2D DDPM from Problem 4:

1. At timestep $t = 50$, compute the empirical score $\nabla\_{x\_t} \log q\_t(x\_t)$ on a grid using kernel density estimation (KDE) on the noised data.

2. Compute the DDPM-implied score: $s\_\theta(x\_t, t) = -\varepsilon\_\theta(x\_t, t) / \sqrt{1 - \bar{\alpha}\_t}$.

3. Plot both vector fields side by side. How closely do they match?

4. Compute the mean squared error between the two score estimates over the grid. Report the relative error.

---

## Submission Checklist

- [ ] Problem 1: Forward process derivation, SNR analysis and plots
- [ ] Problem 2: Full ELBO decomposition, reverse posterior derivation, noise prediction loss
- [ ] Problem 3: Complete DDPM implementation (schedule, forward process, U-Net, training, sampling), generated MNIST samples
- [ ] Problem 4: 2D DDPM with forward/reverse visualizations, score field plot
- [ ] Problem 5: Ablation results on $T$, loss weighting, and variance choice
- [ ] Problem 6: Score-noise equivalence proof and empirical verification

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs and plots.
