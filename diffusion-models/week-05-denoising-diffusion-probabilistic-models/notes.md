# Week 5: Denoising Diffusion Probabilistic Models

> *"The key insight is that a diffusion process that destroys data can be reversed, and learning the reverse process gives us a generative model."*
> -- Jascha Sohl-Dickstein et al. (2015)

---

## Overview

This is the central week of the course. Everything we have built -- probability theory, stochastic processes, score functions, Langevin dynamics, denoising, and score matching -- converges here on a single, powerful generative modelling framework: **denoising diffusion probabilistic models (DDPMs)**.

The idea is audaciously simple. Take your data. Gradually add Gaussian noise until nothing remains but static. Then train a neural network to reverse this destruction, step by step. To generate new data, start from pure noise and run the learned reverse process. The result is a generative model that produces samples rivalling -- and now surpassing -- GANs in image quality, without the training instability, mode collapse, or adversarial dynamics.

The framework was proposed by Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguly in 2015, drawing on ideas from non-equilibrium statistical mechanics. Their paper was prescient but ahead of its time -- the samples were blurry and uncompetitive. Five years later, Ho, Jain, and Abbeel made it work, producing stunning image samples with a collection of simplifications that reduced the variational bound to a denoising objective. Their 2020 paper, "Denoising Diffusion Probabilistic Models," is the focal point of this week.

We will derive every equation from first principles, understand *why* predicting noise is equivalent to estimating the score, and implement the full training and sampling algorithms.

### Prerequisites
- Week 1: Probability, Gaussian distributions, KL divergence
- Week 2: Score functions, Langevin dynamics
- Week 3: Stochastic differential equations (conceptual)
- Week 4: Denoising score matching

---

## 1. Historical Context

### 1.1 The Thermodynamic Inspiration

Sohl-Dickstein et al. (2015) drew their inspiration from non-equilibrium thermodynamics. In physics, a system in a complex, structured state (say, a crystal) can be driven toward thermal equilibrium by heating -- the structured state is gradually destroyed. The reverse process, cooling a disordered system into an ordered one, is generally intractable to compute from first principles. But if you can *learn* the reverse transitions from data, you have a generative model.

The analogy to generative modelling is precise:
- **Data distribution** $q(x\_0)$: the structured state (images, text, molecules)
- **Forward process**: gradual noise injection, destroying structure
- **Equilibrium**: an isotropic Gaussian $\mathcal{N}(0, I)$ -- pure noise, no structure
- **Reverse process**: learned denoising, reconstructing structure from noise

The 2015 paper established the mathematical framework -- the variational bound, the Markov chain structure, the connection to annealed importance sampling -- but the generated samples were poor by modern standards. The architecture was simple, the noise schedules were not well tuned, and the field's attention was elsewhere (GANs were ascendant).

### 1.2 The 2020 Breakthrough

Ho, Jain, and Abbeel (2020) made three key contributions that transformed DDPMs from a theoretical curiosity into a practical generative model:

1. **The $\varepsilon$-prediction reparameterization.** Instead of having the neural network predict the mean of the reverse distribution, predict the *noise* that was added. This is mathematically equivalent but empirically far more stable.

2. **The simplified loss.** The full variational bound decomposes into a sum of KL divergences. Ho et al. showed that dropping the weighting coefficients and using a simple mean-squared-error loss on the noise prediction works better in practice.

3. **A U-Net architecture with proper conditioning.** The neural network receives the noisy image and the timestep as input, using sinusoidal position embeddings for the timestep (borrowed from transformers) and a U-Net backbone with residual blocks and attention layers.

The result: FID scores competitive with GANs on CIFAR-10 and LSUN, with none of the training pathologies.

---

## 2. The Forward Process

### 2.1 Definition

The forward process (also called the *diffusion process* or *noising process*) is a Markov chain that gradually adds Gaussian noise to the data over $T$ steps:

$$
q(x_1, x_2, \ldots, x_T \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})
$$

Each transition is a Gaussian:

$$
\boxed{q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\right)}
$$

where $\beta\_1, \beta\_2, \ldots, \beta\_T$ is a **variance schedule** with $0 < \beta\_t < 1$.

In words: at each step, we scale down the signal by $\sqrt{1 - \beta\_t}$ and add Gaussian noise with variance $\beta\_t$. The signal shrinks; the noise accumulates.

### 2.2 The Variance Schedule

The original DDPM uses a **linear schedule**: $\beta\_t$ increases linearly from $\beta\_1 = 10^{-4}$ to $\beta\_T = 0.02$, with $T = 1000$. The noise starts small (barely perceptible perturbations) and grows (aggressive corruption).

Why so many steps? Because the reverse process -- which we will learn -- is also Gaussian *only when each forward step is small*. For small $\beta\_t$, the reverse conditional $q(x\_{t-1} \mid x\_t)$ is approximately Gaussian. If we took large steps, the reverse would be complex and multimodal, far harder to learn.

Other schedules have been explored:
- **Cosine schedule** (Nichol and Dhariwal, 2021): $\bar{\alpha}\_t = \frac{f(t)}{f(0)}$ where $f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$. This keeps the signal-to-noise ratio more uniform across timesteps.
- **Learned schedules**: make $\beta\_t$ learnable parameters.

### 2.3 The Closed-Form Jump: $q(x\_t \mid x\_0)$

A crucial property: we do not need to run the forward chain step by step. We can jump directly from $x\_0$ to any $x\_t$ in closed form.

Define:
$$
\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t}(1 - \beta_s)
$$

Then:

$$
\boxed{q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t)\, I\right)}
$$

**Derivation.** We prove this by induction. The base case $t = 1$ is immediate from the definition: $q(x\_1 \mid x\_0) = \mathcal{N}(x\_1; \sqrt{\alpha\_1}\, x\_0, \beta\_1 I)$, and $\bar{\alpha}\_1 = \alpha\_1$, $1 - \bar{\alpha}\_1 = \beta\_1$.

For the inductive step, assume $q(x\_{t-1} \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_{t-1}}\, x\_0,\; (1 - \bar{\alpha}\_{t-1})\, I)$. We can write:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\, \varepsilon_{t-1}, \qquad \varepsilon_{t-1} \sim \mathcal{N}(0, I)
$$

The forward step gives:

$$
x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{\beta_t}\, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, I)
$$

Substituting:

$$
x_t = \sqrt{\alpha_t}\!\left(\sqrt{\bar{\alpha}_{t-1}}\, x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\, \varepsilon_{t-1}\right) + \sqrt{\beta_t}\, \varepsilon_t
$$

$$
= \sqrt{\alpha_t \bar{\alpha}_{t-1}}\, x_0 + \sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\, \varepsilon_{t-1} + \sqrt{\beta_t}\, \varepsilon_t
$$

Since $\varepsilon\_{t-1}$ and $\varepsilon\_t$ are independent standard Gaussians, the sum of the two noise terms is Gaussian with variance:

$$
\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t = \alpha_t - \alpha_t \bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \bar{\alpha}_t
$$

using $\beta\_t = 1 - \alpha\_t$ and $\alpha\_t \bar{\alpha}\_{t-1} = \bar{\alpha}\_t$. Therefore:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

which gives $q(x\_t \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t}\, x\_0,\; (1 - \bar{\alpha}\_t)\, I)$. $\square$

**Interpretation.** At time $t$, the noisy sample $x\_t$ is a weighted mixture of the original data $x\_0$ (scaled by $\sqrt{\bar{\alpha}\_t}$) and pure noise (scaled by $\sqrt{1 - \bar{\alpha}\_t}$). As $t$ increases, $\bar{\alpha}\_t \to 0$, and the signal is completely drowned in noise. At $t = T$, $x\_T \approx \mathcal{N}(0, I)$.

This closed-form expression is essential for training: we can sample any $(x\_0, t, \varepsilon)$ triple, compute $x\_t = \sqrt{\bar{\alpha}\_t}\, x\_0 + \sqrt{1 - \bar{\alpha}\_t}\, \varepsilon$, and train without running the full chain.

---

## 3. The Reverse Process

### 3.1 The True Reverse

If we knew $q(x\_{t-1} \mid x\_t)$, we could run the Markov chain backward to generate data. But $q(x\_{t-1} \mid x\_t)$ depends on the entire data distribution and is intractable.

However, $q(x\_{t-1} \mid x\_t, x\_0)$ -- the reverse transition *conditioned on the original data* -- is tractable and Gaussian. By Bayes' rule:

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0)\, q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$

Since the forward process is Markov, $q(x\_t \mid x\_{t-1}, x\_0) = q(x\_t \mid x\_{t-1})$. All three terms on the right-hand side are Gaussians (we derived the latter two in the previous section). The product and ratio of Gaussians is Gaussian, and we can read off the mean and variance by completing the square.

The result:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\; \tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t I\right)
$$

where:

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t
$$

$$
\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})\,\beta_t}{1 - \bar{\alpha}_t}
$$

### 3.2 The Learned Reverse Process

We parameterize the reverse process as:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\; \mu_\theta(x_t, t),\; \sigma_t^2 I\right)
$$

The neural network $\mu\_\theta$ predicts the mean of the reverse step. The variance $\sigma\_t^2$ can be fixed (Ho et al. use $\sigma\_t^2 = \beta\_t$ or $\sigma\_t^2 = \tilde{\beta}\_t$) or learned (Nichol and Dhariwal, 2021).

The full generative model is:

$$
p_\theta(x_0, x_1, \ldots, x_T) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)
$$

where $p(x\_T) = \mathcal{N}(0, I)$.

---

## 4. The Variational Bound

### 4.1 The Evidence Lower Bound (ELBO)

We want to maximize $\log p\_\theta(x\_0)$. As in variational inference, we derive a lower bound by introducing the forward process $q$ as an approximate posterior.

$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} \mid x_0)}\!\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right] = -L_{\text{VLB}}
$$

### 4.2 Decomposition into Per-Timestep KL Divergences

The variational lower bound can be decomposed as:

$$
L_{\text{VLB}} = \underbrace{D_{\text{KL}}\!\left(q(x_T \mid x_0) \,\|\, p(x_T)\right)}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{\text{KL}}\!\left(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\right)}_{L_{t-1}} - \underbrace{\log p_\theta(x_0 \mid x_1)}_{L_0}
$$

Let us understand each term:

- **$L\_T$**: the KL divergence between the endpoint of the forward process and the prior $\mathcal{N}(0, I)$. This is a constant (no learnable parameters) -- it just ensures the forward process reaches the prior. With a good schedule and large $T$, $L\_T \approx 0$.

- **$L\_{t-1}$ for $t = 2, \ldots, T$**: the KL divergence between the true reverse step $q(x\_{t-1} \mid x\_t, x\_0)$ and our learned reverse step $p\_\theta(x\_{t-1} \mid x\_t)$. Both are Gaussians, so the KL divergence has a closed form. These are the terms we optimize.

- **$L\_0$**: the reconstruction term -- how well the model predicts $x\_0$ from $x\_1$. In practice, treated as a discretized Gaussian log-likelihood or handled separately.

### 4.3 KL Between Gaussians

Since both $q(x\_{t-1} \mid x\_t, x\_0)$ and $p\_\theta(x\_{t-1} \mid x\_t)$ are Gaussians with the same (fixed) covariance $\sigma\_t^2 I$, the KL divergence simplifies to:

$$
L_{t-1} = \frac{1}{2\sigma_t^2} \left\|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\right\|^2 + C
$$

where $C$ is a constant independent of $\theta$. Minimizing $L\_{t-1}$ amounts to making $\mu\_\theta(x\_t, t)$ match the true posterior mean $\tilde{\mu}\_t(x\_t, x\_0)$.

---

## 5. The $\varepsilon$-Prediction Reparameterization

### 5.1 From Mean Prediction to Noise Prediction

Recall the true posterior mean:

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t
$$

Since $x\_t = \sqrt{\bar{\alpha}\_t}\, x\_0 + \sqrt{1 - \bar{\alpha}\_t}\, \varepsilon$, we can solve for $x\_0$:

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\!\left(x_t - \sqrt{1 - \bar{\alpha}_t}\, \varepsilon\right)
$$

Substituting into the expression for $\tilde{\mu}\_t$:

$$
\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \varepsilon\right)
$$

This is a crucial formula. It says: the true posterior mean is a function of $x\_t$ and the noise $\varepsilon$ that was used to create $x\_t$ from $x\_0$.

### 5.2 The $\varepsilon$-Prediction Parameterization

Instead of training $\mu\_\theta$ to directly predict $\tilde{\mu}\_t$, we train a network $\varepsilon\_\theta(x\_t, t)$ to predict the noise $\varepsilon$, and define:

$$
\boxed{\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \varepsilon_\theta(x_t, t)\right)}
$$

The loss becomes:

$$
L_{t-1} = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \left\|\varepsilon - \varepsilon_\theta(x_t, t)\right\|^2
$$

The neural network's job is to look at a noisy image $x\_t$ and predict the noise $\varepsilon$ that was added. This is a **denoising** task -- exactly the setting we studied in Week 4.

### 5.3 The Connection to Score Matching

Here is where the threads come together. Recall from Week 4 that the score function of the noised distribution is:

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\, x_0}{1 - \bar{\alpha}_t} = -\frac{\varepsilon}{\sqrt{1 - \bar{\alpha}_t}}
$$

Therefore:

$$
\varepsilon = -\sqrt{1 - \bar{\alpha}_t}\; \nabla_{x_t} \log q(x_t \mid x_0)
$$

A network trained to predict $\varepsilon$ is implicitly estimating the score, up to a known scaling factor:

$$
\boxed{\varepsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t}\; \nabla_{x_t} \log q_t(x_t)}
$$

Or equivalently:

$$
s_\theta(x_t, t) = \nabla_{x_t} \log q_t(x_t) \approx -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

Noise prediction *is* score estimation. This connection, which we will explore in depth in Week 6, is one of the deepest insights in diffusion models.

---

## 6. The Simplified Loss

### 6.1 Dropping the Weights

The full variational bound loss is:

$$
L_{\text{VLB}} = \sum_{t=1}^{T} w_t \left\|\varepsilon - \varepsilon_\theta(x_t, t)\right\|^2
$$

where $w\_t = \frac{\beta\_t^2}{2\sigma\_t^2 \alpha\_t (1 - \bar{\alpha}\_t)}$ are timestep-dependent weights that arise from the KL divergence computation.

Ho et al. found that *ignoring* these weights and using a simple unweighted loss works better in practice:

$$
\boxed{L_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\!\left[\left\|\varepsilon - \varepsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon,\; t\right)\right\|^2\right]}
$$

where $t \sim \text{Uniform}\lbrace 1, \ldots, T\rbrace $, $x\_0 \sim q(x\_0)$, and $\varepsilon \sim \mathcal{N}(0, I)$.

### 6.2 Why Does Dropping the Weights Help?

The VLB weights $w\_t$ are large at small $t$ (low noise) and small at large $t$ (high noise). This means the VLB emphasizes the fine details (low-noise steps) over the global structure (high-noise steps). But getting the global structure right at high noise levels is crucial for sample quality -- if you get the large-scale structure wrong, no amount of detail refinement will save you.

The uniform weighting of $L\_{\text{simple}}$ gives equal importance to all noise levels, which empirically produces better samples (lower FID) even though it gives a looser variational bound.

This is a recurring theme in diffusion models: the variational bound is mathematically elegant but not the best training objective for sample quality.

---

## 7. Training and Sampling Algorithms

### 7.1 Training (Algorithm 1 from Ho et al.)

```
repeat
    x_0 ~ q(x_0)                                          # sample data
    t ~ Uniform({1, ..., T})                               # sample timestep
    ε ~ N(0, I)                                            # sample noise
    x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε                   # create noisy sample
    Take gradient step on ||ε - ε_θ(x_t, t)||²            # predict noise, backprop
until converged
```

This is remarkably simple. Each training step: pick a data point, pick a random noise level, add that much noise, and train the network to predict the noise you added. There is no adversary, no posterior collapse, no mode-seeking vs. mode-covering trade-off. Just denoising.

### 7.2 Sampling (Algorithm 2 from Ho et al.)

```
x_T ~ N(0, I)                                             # start from noise
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1, else z = 0
    x_{t-1} = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z
return x_0
```

Starting from pure Gaussian noise, we iteratively denoise: the network predicts the noise, we subtract a fraction of it (computing the posterior mean), and add a controlled amount of fresh noise (the stochastic term $\sigma\_t z$). At the final step ($t = 1$), we do not add noise.

The choice of $\sigma\_t$ matters:
- $\sigma\_t^2 = \beta\_t$: the "upper bound" choice (DDPM default)
- $\sigma\_t^2 = \tilde{\beta}\_t = \frac{(1 - \bar{\alpha}\_{t-1})\beta\_t}{1 - \bar{\alpha}\_t}$: the "lower bound" choice (posterior variance)
- $\sigma\_t = 0$: deterministic sampling (DDIM, which we will cover in Week 8)

### 7.3 Computational Considerations

**Training** is efficient: each step requires one forward and backward pass through $\varepsilon\_\theta$, plus computing $x\_t$ (cheap). Training converges in hundreds of thousands to millions of gradient steps.

**Sampling** is expensive: generating one sample requires $T = 1000$ sequential forward passes through $\varepsilon\_\theta$. This is much slower than GANs (one forward pass) or VAEs (one forward + one decode). Accelerating sampling is a major research direction (Week 8).

---

## 8. The U-Net Architecture

### 8.1 Why U-Net?

The noise prediction network $\varepsilon\_\theta(x\_t, t)$ takes a noisy image and a timestep, and outputs a noise prediction of the same spatial dimensions. This is a dense prediction task (every pixel gets a prediction), similar to image segmentation -- which is exactly what U-Nets were designed for.

The U-Net has an encoder path (downsampling, increasing channels) and a decoder path (upsampling, decreasing channels), connected by skip connections at each resolution. The skip connections allow the network to combine high-level semantics (from the bottleneck) with fine-grained spatial detail (from early layers).

### 8.2 Timestep Conditioning

The network must know *how much noise* was added (i.e., the timestep $t$). Ho et al. use sinusoidal position embeddings:

$$
\text{emb}(t)_{2i} = \sin(t / 10000^{2i/d}), \qquad \text{emb}(t)_{2i+1} = \cos(t / 10000^{2i/d})
$$

This embedding is projected through a small MLP and added to (or used to scale) the intermediate features at each residual block. The same idea from transformers, now conditioning a CNN on continuous time.

### 8.3 Architecture Details

The DDPM U-Net uses:
- Residual blocks with GroupNorm and SiLU activations
- Self-attention at low resolutions (e.g., 16x16)
- Downsampling via strided convolution, upsampling via transposed convolution
- Skip connections via concatenation

For a 32x32 image (CIFAR-10), a typical architecture has ~35M parameters. For 256x256 images, the model scales to ~550M parameters.

---

## 9. Putting It All Together

### 9.1 The Generative Story

1. **Forward process** (not learned): data $x\_0$ is progressively noised over $T$ steps, ending at $x\_T \sim \mathcal{N}(0, I)$.

2. **Training**: for random $(x\_0, t, \varepsilon)$ triples, we compute $x\_t$ and train $\varepsilon\_\theta$ to predict $\varepsilon$ from $x\_t$ and $t$.

3. **Sampling**: starting from $x\_T \sim \mathcal{N}(0, I)$, we iteratively apply the learned reverse process $p\_\theta(x\_{t-1} \mid x\_t)$ for $t = T, T-1, \ldots, 1$.

The model never sees a "real" image during sampling. It starts from noise and denoises, step by step, guided only by its learned estimate of the score function.

### 9.2 What the Network Learns

At high noise levels (large $t$, small $\bar{\alpha}\_t$), the noisy image $x\_t$ is mostly noise. The network must estimate the *direction* toward the data manifold -- the large-scale structure. This is a coarse task: is this an image of a face or a landscape?

At low noise levels (small $t$, $\bar{\alpha}\_t \approx 1$), the noisy image is almost clean. The network refines fine details: the exact placement of edges, the texture of hair, the shading of skin. This is a fine-grained denoising task.

This coarse-to-fine hierarchy -- global structure first, then progressive refinement -- is a natural consequence of the multi-scale noise levels. It is reminiscent of how humans draw: rough sketch first, then details.

### 9.3 DDPM Results

Ho et al. (2020) achieved:
- **CIFAR-10** (32x32): FID 3.17, competitive with the best GANs at the time
- **LSUN Bedrooms** (256x256): FID 4.90, high-quality diverse samples
- **LSUN Churches** (256x256): visually compelling samples

These results were a wake-up call. A model based on maximizing a variational bound -- the "boring" approach compared to adversarial training -- was producing samples as good as or better than GANs, without training instability.

---

## 10. Beyond the Basics

### 10.1 Improved DDPM

Nichol and Dhariwal (2021) improved upon the original DDPM in several ways:

- **Learned variance**: instead of fixing $\sigma\_t^2$, parameterize it as an interpolation between $\beta\_t$ and $\tilde{\beta}\_t$ in log-space. This improves log-likelihood without hurting FID.
- **Cosine schedule**: the linear schedule wastes steps at the endpoints (too little noise at the start, signal already destroyed before the end). The cosine schedule distributes the noise more uniformly.
- **Hybrid objective**: train on a weighted combination of $L\_{\text{simple}}$ (for sample quality) and $L\_{\text{VLB}}$ (for the variance parameters).

### 10.2 $v$-Prediction

Salimans and Ho (2022) proposed an alternative parameterization called $v$-prediction:

$$
v = \sqrt{\bar{\alpha}_t}\, \varepsilon - \sqrt{1 - \bar{\alpha}_t}\, x_0
$$

The network predicts $v$ instead of $\varepsilon$ or $x\_0$. This parameterization has better numerical properties at extreme noise levels and enables progressive distillation (Week 12).

### 10.3 $x\_0$-Prediction

Instead of predicting noise, predict the clean data directly:

$$
\hat{x}_0 = f_\theta(x_t, t)
$$

Then compute $\mu\_\theta$ from $\hat{x}\_0$ using the posterior mean formula. This is equivalent to $\varepsilon$-prediction (one can be converted to the other) but may have different gradient dynamics during training.

---

## Summary

1. **DDPMs** define a forward process that adds Gaussian noise over $T$ steps, and learn a reverse process that removes it.

2. **The forward process** admits a closed form: $q(x\_t \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t}\, x\_0, (1 - \bar{\alpha}\_t)\, I)$, where $\bar{\alpha}\_t = \prod\_{s=1}^{t}(1 - \beta\_s)$.

3. **The variational bound** decomposes into per-timestep KL divergences between the true reverse posterior and the learned reverse.

4. **$\varepsilon$-prediction**: instead of predicting the reverse mean directly, predict the noise $\varepsilon$. The posterior mean becomes $\mu\_\theta = \frac{1}{\sqrt{\alpha\_t}}(x\_t - \frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}} \varepsilon\_\theta(x\_t, t))$.

5. **The simplified loss** $L\_{\text{simple}} = \mathbb{E}\Vert \varepsilon - \varepsilon\_\theta(x\_t, t)\Vert ^2$ is just noise prediction -- denoising score matching in disguise.

6. **Training** is simple: sample $(x\_0, t, \varepsilon)$, compute $x\_t$, predict $\varepsilon$, backprop. **Sampling** runs the reverse chain from $x\_T \sim \mathcal{N}(0, I)$ for $T$ steps.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| Forward step | $q(x\_t \mid x\_{t-1}) = \mathcal{N}(\sqrt{1-\beta\_t}\, x\_{t-1}, \beta\_t I)$ |
| Closed-form jump | $q(x\_t \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t}\, x\_0, (1-\bar{\alpha}\_t)I)$ |
| True reverse mean | $\tilde{\mu}\_t = \frac{\sqrt{\bar{\alpha}\_{t-1}}\beta\_t}{1-\bar{\alpha}\_t} x\_0 + \frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t} x\_t$ |
| $\varepsilon$-parameterized mean | $\mu\_\theta = \frac{1}{\sqrt{\alpha\_t}}\!\left(x\_t - \frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}} \varepsilon\_\theta(x\_t, t)\right)$ |
| Simplified loss | $L\_{\text{simple}} = \mathbb{E}\_{t,x\_0,\varepsilon}\!\left[\Vert \varepsilon - \varepsilon\_\theta(x\_t, t)\Vert ^2\right]$ |
| Score-noise relation | $\varepsilon\_\theta \approx -\sqrt{1-\bar{\alpha}\_t}\;\nabla\_{x\_t}\log q\_t(x\_t)$ |

---

## Suggested Reading

- **Ho, Jain, Abbeel** (2020), "Denoising Diffusion Probabilistic Models" -- the paper that launched a thousand papers. Read it in full.
- **Sohl-Dickstein, Weiss, Maheswaranathan, Ganguly** (2015), "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" -- the original framework. The ideas are all here; the execution was before its time.
- **Nichol and Dhariwal** (2021), "Improved Denoising Diffusion Probabilistic Models" -- practical improvements that became standard.
- **Luo** (2022), "Understanding Diffusion Models: A Unified Perspective" -- an excellent tutorial that derives everything we covered today (and more) from a unified variational perspective.
- **Turner** (2024), "An Introduction to Flow Matching" -- Section 2 gives a clean derivation of the DDPM forward process and its connection to SDEs (a preview of Week 7).
