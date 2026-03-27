# Week 6: Score-Based Generative Models

> *"We can learn to generate data by estimating the gradient of the log-density, without ever estimating the density itself."*
> -- Yang Song and Stefano Ermon (2019)

---

## Overview

While Ho et al. were developing DDPMs from the variational inference tradition, Yang Song and Stefano Ermon were building an independent path to the same destination from a different starting point: **score matching and Langevin dynamics**.

The core idea: if you can estimate the score function $\nabla\_x \log p(x)$ -- the gradient of the log-density -- you can generate samples using Langevin dynamics, no likelihood computation required. The problem is that score estimation fails in low-density regions (where you have no data to learn from). The solution: add noise at multiple scales, estimate the score at each scale, and generate by gradually denoising from high to low noise using *annealed* Langevin dynamics.

This is the **Noise Conditional Score Network (NCSN)** framework. It was developed in parallel with DDPM and, as we will show, is mathematically equivalent to it. The two communities -- variational inference and score matching -- discovered the same model through different lenses. This week, we study the score-based perspective and make the equivalence explicit.

### Prerequisites
- Week 2: Score functions, Langevin dynamics
- Week 4: Score matching, denoising score matching
- Week 5: DDPM (especially the $\varepsilon$-prediction reparameterization)

---

## 1. The Score Function Perspective

### 1.1 Recap: Score Functions and Langevin Dynamics

Recall from Week 2 that the **score function** of a distribution $p(x)$ is:

$$
s(x) = \nabla_x \log p(x)
$$

The score points in the direction of steepest increase of the log-density -- toward the modes of the distribution. It has a remarkable property: it does not depend on the normalizing constant of $p(x)$. If $p(x) = \frac{1}{Z} \tilde{p}(x)$, then $\nabla\_x \log p(x) = \nabla\_x \log \tilde{p}(x)$.

Given the score, **Langevin dynamics** generates samples from $p(x)$ by iterating:

$$
x_{k+1} = x_k + \frac{\eta}{2} \nabla_x \log p(x_k) + \sqrt{\eta}\, z_k, \qquad z_k \sim \mathcal{N}(0, I)
$$

For small enough step size $\eta$ and enough iterations, $x\_k$ converges in distribution to $p(x)$. The gradient term pushes $x$ toward high-density regions; the noise term ensures exploration and prevents collapse to a single mode.

### 1.2 The Problem: Score Estimation Fails in the Tails

In Week 4, we learned how to estimate the score from data using **score matching** (Hyvarinen, 2005) or **denoising score matching** (Vincent, 2011). We train a network $s\_\theta(x)$ to approximate $\nabla\_x \log p(x)$ by minimizing:

$$
\mathcal{L}_{\text{SM}} = \mathbb{E}_{p(x)}\!\left[\frac{1}{2}\Vert s_\theta(x) - \nabla_x \log p(x)\Vert ^2\right]
$$

or its tractable denoising variant.

But there is a fundamental problem. The score matching objective weights the error by $p(x)$ -- the expectation is taken under the data distribution. This means:

- In **high-density regions** (near the data), we get many training samples and learn the score well.
- In **low-density regions** (far from the data), we get few or no training samples. The score estimate is unreliable.

Why does this matter for Langevin dynamics? Because sampling starts from random initialization -- which is almost certainly in a low-density region. If the score estimate is garbage in low-density regions, Langevin dynamics will not find its way to the data. It will wander aimlessly in regions where the score is poorly estimated.

### 1.3 A Concrete Example

Consider a mixture of two Gaussians in 1D: $p(x) = 0.5\,\mathcal{N}(x; -5, 0.5^2) + 0.5\,\mathcal{N}(x; 5, 0.5^2)$. The two modes are well-separated, with a vast low-density region between them.

If we train a score network on samples from $p(x)$, the network sees data near $x = -5$ and $x = 5$, but almost nothing near $x = 0$. The score estimate between the modes is unreliable. Langevin dynamics initialized at $x\_0 = 0$ might go either way, or get stuck, or oscillate -- the poor score estimate gives no useful guidance.

Even worse: in high dimensions, *almost everywhere* is low-density. Real data (images, text) lives on a thin manifold in a vast ambient space. The score is well-estimated only on or very near this manifold.

---

## 2. The Multi-Scale Solution: NCSN

### 2.1 The Key Idea: Add Noise

Song and Ermon's insight: **perturbing the data with noise fills in the low-density regions**.

If we convolve the data distribution $p(x)$ with Gaussian noise of standard deviation $\sigma$:

$$
p_\sigma(x) = \int p(y)\, \mathcal{N}(x; y, \sigma^2 I)\, dy
$$

then $p\_\sigma(x)$ has support everywhere (no more zero-density regions). The score $\nabla\_x \log p\_\sigma(x)$ is well-defined and estimable everywhere.

At high $\sigma$, $p\_\sigma$ is a heavily smoothed version of $p$ -- the modes are blurred together, and the score provides useful gradient signal even far from the data. At low $\sigma$, $p\_\sigma \approx p$ -- the score captures fine detail.

### 2.2 Geometric Noise Levels

NCSN uses $L$ noise levels, geometrically spaced:

$$
\sigma_1 > \sigma_2 > \cdots > \sigma_L
$$

with $\sigma\_1$ large enough that $p\_{\sigma\_1}$ is nearly a single Gaussian (modes merged), and $\sigma\_L$ small enough that $p\_{\sigma\_L} \approx p$ (data approximately unperturbed).

Song and Ermon (2019) recommend:
- $\sigma\_1$ should be approximately the maximum pairwise distance between data points
- $\sigma\_L$ should be small enough to be imperceptible (e.g., 0.01)
- Geometric spacing: $\sigma\_i = \sigma\_1 \cdot (\sigma\_L / \sigma\_1)^{(i-1)/(L-1)}$

A typical choice is $L = 10$ to $L = 1000$, with $\sigma\_1 = 50$ and $\sigma\_L = 0.01$.

### 2.3 Why Multiple Scales Work: The Connectivity Argument

Consider again the bimodal distribution $p(x) = 0.5\,\mathcal{N}(-5, 0.5^2) + 0.5\,\mathcal{N}(5, 0.5^2)$.

At $\sigma = 0.1$ (low noise): the two modes are well-separated. The score field between them is weak and unreliable. Langevin dynamics mixes poorly between modes.

At $\sigma = 5$ (high noise): the two modes blur into a single broad distribution. The score field is smooth and well-estimated everywhere. Langevin dynamics mixes freely.

At $\sigma = 2$ (moderate noise): the modes are partially merged. Mixing is easier than at low noise but the distribution is still somewhat bimodal.

The strategy: start with high noise (easy mixing, poor detail), gradually reduce the noise level (harder mixing, better detail). At each level, use the previous level's samples as warm starts. This is annealed Langevin dynamics.

---

## 3. The Score Matching Objective

### 3.1 Noise-Conditional Score Network

We train a single network $s\_\theta(x, \sigma)$ to estimate the score at all noise levels simultaneously:

$$
s_\theta(x, \sigma) \approx \nabla_x \log p_\sigma(x)
$$

The network takes both $x$ and $\sigma$ (or an index $i$ corresponding to $\sigma\_i$) as input.

### 3.2 The Denoising Score Matching Loss

From Week 4, we know that denoising score matching provides a tractable loss for score estimation. For a single noise level $\sigma$:

$$
\mathcal{L}_{\text{DSM}}(\sigma) = \mathbb{E}_{p(x)}\, \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma^2 I)}\!\left[\frac{1}{2}\left\Vert s_\theta(\tilde{x}, \sigma) - \nabla_{\tilde{x}} \log p(\tilde{x} \mid x)\right\Vert ^2\right]
$$

Since $p(\tilde{x} \mid x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$, the target score is:

$$
\nabla_{\tilde{x}} \log p(\tilde{x} \mid x) = -\frac{\tilde{x} - x}{\sigma^2} = -\frac{\varepsilon}{\sigma}
$$

where $\tilde{x} = x + \sigma \varepsilon$ and $\varepsilon \sim \mathcal{N}(0, I)$.

So the per-noise-level loss is:

$$
\mathcal{L}_{\text{DSM}}(\sigma) = \mathbb{E}_{p(x)}\, \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}\!\left[\frac{1}{2}\left\Vert s_\theta(x + \sigma\varepsilon, \sigma) + \frac{\varepsilon}{\sigma}\right\Vert ^2\right]
$$

### 3.3 The Combined Objective

The full NCSN objective averages over all noise levels with weights $\lambda(\sigma)$:

$$
\mathcal{L}_{\text{NCSN}} = \sum_{i=1}^{L} \lambda(\sigma_i)\, \mathcal{L}_{\text{DSM}}(\sigma_i)
$$

Song and Ermon (2019) use $\lambda(\sigma) = \sigma^2$, which ensures that the loss magnitude is comparable across noise levels (since $\Vert \nabla\_x \log p\_\sigma(x)\Vert \sim 1/\sigma$, multiplying by $\sigma^2$ normalizes the expected loss).

With $\lambda(\sigma) = \sigma^2$, the loss for a single noise level becomes:

$$
\lambda(\sigma)\,\mathcal{L}_{\text{DSM}}(\sigma) = \frac{1}{2}\,\mathbb{E}\!\left[\Vert \sigma\, s_\theta(x + \sigma\varepsilon, \sigma) + \varepsilon\Vert ^2\right]
$$

This is strikingly similar to the DDPM loss. We will make this precise in Section 6.

---

## 4. Annealed Langevin Dynamics

### 4.1 The Sampling Algorithm

Given a trained score network $s\_\theta(x, \sigma)$, we generate samples as follows:

```
Initialize x_0 ~ N(0, σ_1² I)      (or some broad distribution)

for i = 1, 2, ..., L:               (from highest noise to lowest)
    η_i = ε · (σ_i / σ_L)²         (step size proportional to σ²)
    for k = 1, 2, ..., K:           (Langevin steps at this noise level)
        z ~ N(0, I)
        x ← x + (η_i / 2) · s_θ(x, σ_i) + √η_i · z

return x
```

### 4.2 Interpretation

At each noise level $\sigma\_i$, we run $K$ steps of Langevin dynamics targeting $p\_{\sigma\_i}(x)$. As $i$ increases (noise decreases):

- The target distribution $p\_{\sigma\_i}$ becomes sharper and more detailed
- The step size $\eta\_i$ shrinks (finer adjustments)
- The samples from the previous level provide warm starts (they are already in roughly the right region)

This is analogous to **simulated annealing**: start at high "temperature" (large noise, easy mixing) and gradually cool to the target distribution.

### 4.3 How Many Steps?

In theory, we need $K \to \infty$ at each noise level for exact convergence. In practice, Song and Ermon use $K = 100$ to $K = 1000$ steps per noise level with $L = 10$ levels, for a total of 1000-10000 score function evaluations per sample.

Compare to DDPM, which uses $T = 1000$ evaluations. The two methods require comparable computation for sampling.

---

## 5. Improved Techniques

### 5.1 Challenges with NCSN v1

The original NCSN (Song and Ermon, 2019) produced good results but had several practical issues:

1. **Sensitivity to noise level selection**: the choice of $\sigma\_1, \sigma\_L, L$ required careful tuning.
2. **Training instability**: the score magnitudes vary enormously across noise levels, leading to gradient imbalance.
3. **Sample quality gap**: NCSN v1 was competitive with but did not surpass GANs.

### 5.2 NCSN v2: Improved Techniques

Song and Ermon (2020) addressed these issues in their follow-up paper, "Improved Techniques for Training Score-Based Generative Models":

**Architecture.** Use a U-Net architecture (like DDPM) instead of a RefineNet. Condition on noise level using instance normalization parameters.

**Noise level conditioning.** Instead of concatenating $\sigma$ to the input, use it to modulate the network activations via conditional instance normalization or FiLM (Feature-wise Linear Modulation):

$$
h \leftarrow \gamma(\sigma) \cdot \frac{h - \mu_h}{\sigma_h} + \beta(\sigma)
$$

where $\gamma(\sigma)$ and $\beta(\sigma)$ are learned functions of $\sigma$.

**Exponential moving average (EMA).** Maintain an EMA of the model weights during training and use the EMA weights for sampling. This stabilizes sample quality significantly.

**Many noise levels.** Increase $L$ from 10 to thousands (approaching the continuous limit). With many levels, the jump between consecutive noise scales is small, and annealed Langevin dynamics works better.

### 5.3 The Architecture: RefineNet to U-Net

The original NCSN used a dilated ResNet (called RefineNet) architecture. The improved version switched to a U-Net, the same architecture used in DDPM. This convergence is not coincidental -- both models are doing the same computation (estimating the score at multiple noise levels), and the U-Net is well-suited to this task.

---

## 6. The DDPM-NCSN Equivalence

This is the intellectual climax of this week. We now show explicitly that DDPM and NCSN are the same model with different notation.

### 6.1 Matching the Noise Levels

In DDPM, the noise level at timestep $t$ is controlled by $\bar{\alpha}\_t$:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t)\, I\right)
$$

In NCSN, the noise level is controlled by $\sigma\_i$:

$$
p_{\sigma_i}(\tilde{x}) = \mathbb{E}_{p(x)}\!\left[\mathcal{N}(\tilde{x}; x, \sigma_i^2 I)\right]
$$

To match them, note that the DDPM forward process can be rewritten as:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon
$$

If we define $\hat{x} = x\_t / \sqrt{\bar{\alpha}\_t}$, then:

$$
\hat{x} = x_0 + \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}}\, \varepsilon
$$

This is exactly data plus Gaussian noise with standard deviation $\sigma = \sqrt{(1 - \bar{\alpha}\_t)/\bar{\alpha}\_t}$. The DDPM forward process is NCSN's noise perturbation with a specific noise-to-signal mapping plus a global rescaling.

More directly: the DDPM noise variance at step $t$ is $(1 - \bar{\alpha}\_t)$, and the signal coefficient is $\sqrt{\bar{\alpha}\_t}$. The NCSN noise variance is $\sigma\_i^2$ with signal coefficient 1. The two formulations are related by the change of variables $\sigma\_i^2 = (1 - \bar{\alpha}\_t)/\bar{\alpha}\_t$.

### 6.2 Matching the Networks

In DDPM, the network $\varepsilon\_\theta(x\_t, t)$ predicts the noise $\varepsilon$ added to create $x\_t$.

In NCSN, the network $s\_\theta(\tilde{x}, \sigma)$ estimates $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$.

From Section 5.3 of the Week 5 notes:

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\varepsilon}{\sqrt{1 - \bar{\alpha}_t}}
$$

Therefore, a DDPM noise predictor and an NCSN score estimator are related by:

$$
\boxed{s_\theta(x_t, t) = -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}}
$$

Or equivalently:

$$
\boxed{\varepsilon_\theta(x_t, t) = -\sqrt{1 - \bar{\alpha}_t}\; s_\theta(x_t, t)}
$$

They are the *same function* up to a known, timestep-dependent scaling factor. A network trained to predict noise is a network trained to estimate the score. The DDPM loss and the NCSN loss are the same loss with different constants.

### 6.3 Matching the Losses

The DDPM simplified loss:

$$
L_{\text{DDPM}} = \mathbb{E}_{t, x_0, \varepsilon}\!\left[\Vert \varepsilon - \varepsilon_\theta(x_t, t)\Vert ^2\right]
$$

Substituting $\varepsilon\_\theta = -\sqrt{1 - \bar{\alpha}\_t}\, s\_\theta$:

$$
L_{\text{DDPM}} = \mathbb{E}\!\left[\left\Vert \varepsilon + \sqrt{1 - \bar{\alpha}_t}\, s_\theta(x_t, t)\right\Vert ^2\right]
$$

$$
= (1 - \bar{\alpha}_t)\,\mathbb{E}\!\left[\left\Vert \frac{\varepsilon}{\sqrt{1 - \bar{\alpha}_t}} + s_\theta(x_t, t)\right\Vert ^2\right]
$$

$$
= (1 - \bar{\alpha}_t)\,\mathbb{E}\!\left[\left\Vert s_\theta(x_t, t) - \nabla_{x_t}\log q(x_t \mid x_0)\right\Vert ^2\right]
$$

This is exactly the NCSN denoising score matching loss with weight $\lambda(\sigma) = 1 - \bar{\alpha}\_t$ (which is the noise variance, playing the role of $\sigma^2$ in the NCSN weighting).

### 6.4 Matching the Sampling

DDPM sampling:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t)\right) + \sigma_t z
$$

Substituting $\varepsilon\_\theta = -\sqrt{1 - \bar{\alpha}\_t}\, s\_\theta$:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t + \beta_t\, s_\theta(x_t, t)\right) + \sigma_t z
$$

This is a discretization of Langevin dynamics with the score $s\_\theta$, plus a scaling by $1/\sqrt{\alpha\_t}$ that accounts for the signal shrinkage in the DDPM forward process.

NCSN's annealed Langevin dynamics:

$$
x \leftarrow x + \frac{\eta}{2}\, s_\theta(x, \sigma_i) + \sqrt{\eta}\, z
$$

The structure is the same: move in the score direction, add noise. The coefficients differ because of the different forward process conventions, but the underlying dynamics are equivalent.

### 6.5 The Equivalence, Summarized

| Aspect | DDPM | NCSN |
|--------|------|------|
| Intellectual tradition | Variational inference | Score matching |
| Network output | Noise $\varepsilon\_\theta$ | Score $s\_\theta$ |
| Relationship | $\varepsilon\_\theta = -\sqrt{1-\bar{\alpha}\_t}\, s\_\theta$ | $s\_\theta = -\varepsilon\_\theta / \sqrt{1-\bar{\alpha}\_t}$ |
| Loss | $\Vert \varepsilon - \varepsilon\_\theta\Vert ^2$ | $\Vert s\_\theta - \nabla \log p\_\sigma\Vert ^2$ |
| Sampling | Reverse Markov chain | Annealed Langevin dynamics |
| Noise parameterization | $\bar{\alpha}\_t$ schedule | $\sigma\_i$ levels |

The two methods converge to the same algorithm from different starting points. This convergence is the strongest evidence that the underlying mathematical structure is fundamental, not an artifact of a particular derivation.

---

## 7. Practical Differences and Trade-Offs

Despite their mathematical equivalence, DDPM and NCSN have some practical differences:

### 7.1 Discrete vs. Continuous Noise Levels

DDPM uses a fixed number of discrete timesteps $t \in \lbrace 1, \ldots, T\rbrace$. NCSN uses discrete noise levels $\sigma\_1, \ldots, \sigma\_L$ but the framework naturally invites the continuous limit $\sigma \in [\sigma\_{\min}, \sigma\_{\max}]$.

The continuous perspective, which we will develop in Week 7, leads to the SDE framework.

### 7.2 Sampling Procedure

DDPM's reverse chain uses the *exact* reverse posterior (conditioned on the noise schedule), while NCSN's annealed Langevin dynamics is an *approximate* sampler that runs multiple steps per noise level. In practice:

- DDPM sampling is simpler (one step per noise level, derived from the generative model)
- NCSN sampling is more flexible (can vary the number of Langevin steps per level)
- Both require many sequential forward passes ($T = 1000$ is standard)

### 7.3 Training

Both train with a denoising objective. The differences are:
- DDPM samples $t$ uniformly; NCSN samples $\sigma\_i$ uniformly (or with weights)
- DDPM predicts $\varepsilon$; NCSN predicts $s$ (equivalent up to rescaling)
- The weighting of the loss across noise levels differs and can affect convergence

### 7.4 Likelihood

DDPM provides a variational lower bound on $\log p(x)$ via the ELBO. NCSN does not directly provide a likelihood bound, though likelihood can be computed via the probability flow ODE (Week 7).

---

## 8. Score-Based Models Beyond Images

### 8.1 Advantages of the Score Perspective

The score-based viewpoint generalizes naturally beyond images:

- **No assumption on data topology.** The score is defined for any continuous distribution, whether the data lives in $\mathbb{R}^d$, on a manifold, or on a graph.
- **No decoder needed.** Unlike VAEs, score-based models do not require a decoder architecture. The score network operates in the data space directly.
- **Principled handling of constraints.** The score can be modified to incorporate constraints (e.g., inpainting, conditional generation) by adjusting the score during sampling.

### 8.2 Applications

Score-based generative models have been applied to:
- **Audio synthesis** (WaveGrad, DiffWave): generating raw waveforms
- **Molecular generation**: designing drug candidates by denoising 3D molecular coordinates
- **3D shape generation**: denoising point clouds
- **Inverse problems**: using the score as a prior for CT reconstruction, MRI, super-resolution

---

## Summary

1. **Score estimation** from data fails in low-density regions because the score matching objective is weighted by the data distribution. Langevin dynamics initialized far from the data gets lost.

2. **Noise perturbation** solves this problem: by convolving the data with Gaussian noise, we fill in low-density regions and make the score estimable everywhere.

3. **NCSN** uses multiple noise levels $\sigma\_1 > \cdots > \sigma\_L$ and trains a single network $s\_\theta(x, \sigma)$ to estimate the score at all levels. At high noise, the score captures global structure; at low noise, fine details.

4. **Annealed Langevin dynamics** generates samples by running Langevin dynamics at decreasing noise levels, starting from pure noise and gradually refining.

5. **DDPM and NCSN are mathematically equivalent**: $\varepsilon\_\theta = -\sqrt{1-\bar{\alpha}\_t}\, s\_\theta$. Noise prediction is score estimation. The DDPM loss is the NCSN denoising score matching loss with specific weights.

6. This equivalence points to a deeper mathematical structure -- the **stochastic differential equation** formulation -- which we will develop next week.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| Score function | $s(x) = \nabla\_x \log p(x)$ |
| Langevin dynamics | $x\_{k+1} = x\_k + \frac{\eta}{2}\nabla\_x \log p(x\_k) + \sqrt{\eta}\, z\_k$ |
| Noised distribution | $p\_\sigma(x) = \int p(y)\,\mathcal{N}(x; y, \sigma^2 I)\,dy$ |
| DSM target | $\nabla\_{\tilde{x}} \log p(\tilde{x} \mid x) = -(\tilde{x} - x)/\sigma^2$ |
| NCSN loss | $\sum\_i \sigma\_i^2\,\mathbb{E}\left[\Vert s\_\theta(x+\sigma\_i\varepsilon, \sigma\_i) + \varepsilon/\sigma\_i\Vert ^2\right]$ |
| DDPM-NCSN equivalence | $\varepsilon\_\theta(x\_t, t) = -\sqrt{1-\bar{\alpha}\_t}\; s\_\theta(x\_t, t)$ |

---

## Suggested Reading

- **Song and Ermon** (2019), "Generative Modeling by Estimating Gradients of the Data Distribution" -- the original NCSN paper. Clean writing, elegant ideas.
- **Song and Ermon** (2020), "Improved Techniques for Training Score-Based Generative Models" -- practical improvements, EMA, better architectures.
- **Hyvarinen** (2005), "Estimation of Non-Normalized Statistical Models by Score Matching" -- the foundational score matching paper (revisit from Week 4).
- **Vincent** (2011), "A Connection Between Score Matching and Denoising Autoencoders" -- the denoising score matching result that underpins both NCSN and DDPM.
- **Song et al.** (2021), "Score-Based Generative Modeling through Stochastic Differential Equations" -- the unified perspective (preview of Week 7).
