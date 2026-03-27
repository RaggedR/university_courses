# Week 2: Score Functions and Langevin Dynamics

> *"The score function tells you which way is uphill. Langevin dynamics tells you to go uphill while stumbling randomly. Remarkably, the stumbling is what makes it work."*

---

## Overview

Last week we built the forward process of a diffusion model: a Markov chain that converts data into noise. This week we ask the reverse question: **how do we convert noise back into data?**

The answer involves two beautiful ideas that predate diffusion models by decades:

1. **The score function** $\nabla\_x \log p(x)$ -- a vector field that points toward regions of high probability density. Unlike the density itself, the score does not require knowledge of the normalizing constant, making it far easier to estimate.

2. **Langevin dynamics** -- an MCMC method that uses the score function to generate samples from a distribution. It is gradient ascent on the log-density, plus noise to ensure proper exploration.

These ideas, which come from statistical physics and computational statistics, turn out to be exactly what we need to build the reverse process of a diffusion model. The neural network in a diffusion model is learning a score function, and the sampling process is doing Langevin dynamics.

This week establishes the conceptual foundation. In Week 3, we will see how DDPM implements these ideas in practice; in Week 4, we will derive the training objective that makes everything work.

### Prerequisites
- Week 1: Multivariate Gaussians, Markov chains, the forward process
- Multivariable calculus (gradients, divergence)
- Basic familiarity with MCMC (helpful but not required)

---

## 1. The Score Function

### 1.1 Definition

Given a probability density $p(x)$ on $\mathbb{R}^d$, the **score function** is the gradient of the log-density:

$$
s(x) = \nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}
$$

The score is a vector field: at every point $x \in \mathbb{R}^d$, it assigns a vector $s(x) \in \mathbb{R}^d$.

### 1.2 Geometric Interpretation

The score function points in the direction of steepest ascent of $\log p(x)$. Equivalently, it points toward regions of higher density. Its magnitude tells you how steeply the log-density is increasing.

At a mode of the distribution ($\nabla\_x p(x) = 0$), the score is zero. Moving away from a mode, the score points back toward it. The score field is like a "gravitational field" that pulls everything toward the high-density regions of the distribution.

For an isotropic Gaussian $p(x) = \mathcal{N}(x; \mu, \sigma^2 I)$:

$$
\log p(x) = -\frac{d}{2}\log(2\pi\sigma^2) - \frac{\Vert x - \mu\Vert ^2}{2\sigma^2}
$$

$$
\nabla_x \log p(x) = -\frac{x - \mu}{\sigma^2}
$$

The score points from $x$ directly toward the mean $\mu$, with magnitude $\Vert x - \mu\Vert  / \sigma^2$. The farther you are from the mean, the stronger the pull. The smaller the variance, the stronger the pull. This makes intuitive sense: a tight distribution exerts stronger "gravity" than a diffuse one.

### 1.3 Why Scores Instead of Densities?

The fundamental problem with probability densities is the **normalizing constant**. For an energy-based model:

$$
p(x) = \frac{1}{Z} \exp(-E(x)), \quad Z = \int \exp(-E(x)) \, dx
$$

The partition function $Z$ is an integral over all of $\mathbb{R}^d$ -- typically intractable to compute. This means we can compute $E(x)$ for any $x$, and therefore the unnormalized density $\exp(-E(x))$, but not the normalized density $p(x)$.

The score sidesteps this entirely:

$$
\nabla_x \log p(x) = \nabla_x \left[\log \exp(-E(x)) - \log Z\right] = -\nabla_x E(x)
$$

The $\log Z$ term vanishes because $Z$ does not depend on $x$. **The score function does not require the normalizing constant.** This is why score-based methods are so powerful: they work with unnormalized models, which are far more expressive and easier to specify.

### 1.4 Score of a Gaussian Mixture

To build intuition, consider a mixture of two Gaussians in 1D:

$$
p(x) = \frac{1}{2}\mathcal{N}(x; -3, 1) + \frac{1}{2}\mathcal{N}(x; 3, 1)
$$

The score is:

$$
\nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)} = \frac{\frac{1}{2}(3 - x - x)\text{...}}{p(x)}
$$

More carefully:

$$
\nabla_x p(x) = \frac{1}{2}\left[-( x + 3)\mathcal{N}(x; -3, 1) - (x - 3)\mathcal{N}(x; 3, 1)\right]
$$

$$
s(x) = \frac{-(x+3) \cdot w_1(x) - (x-3) \cdot w_2(x)}{w_1(x) + w_2(x)}
$$

where $w\_1(x) = \mathcal{N}(x; -3, 1)$ and $w\_2(x) = \mathcal{N}(x; 3, 1)$ are the (unnormalized) component densities.

The score field has three qualitative regimes:
- **Near $x = -3$:** $w\_1 \gg w\_2$, so $s(x) \approx -(x+3)$. The score points toward $-3$.
- **Near $x = 3$:** $w\_2 \gg w\_1$, so $s(x) \approx -(x-3)$. The score points toward $3$.
- **Near $x = 0$:** both components contribute. The score is approximately zero (a saddle point of $\log p$).

### 1.5 Scores for Conditional Distributions

A powerful property of scores: for a conditional distribution $p(x \mid y)$, the score with respect to $x$ is:

$$
\nabla_x \log p(x \mid y) = \nabla_x \log p(y \mid x) + \nabla_x \log p(x) - \underbrace{\nabla_x \log p(y)}_{= 0}
$$

The last term vanishes because $p(y)$ does not depend on $x$. This gives us:

$$
\nabla_x \log p(x \mid y) = \nabla_x \log p(x) + \nabla_x \log p(y \mid x)
$$

The conditional score is the unconditional score plus a "guidance" term. This is the foundation of **classifier guidance** and **classifier-free guidance**, which we will study in Weeks 8-9.

---

## 2. Score Matching

### 2.1 The Problem

Suppose we want to learn the score function of an unknown data distribution $p\_{\text{data}}(x)$. We have samples $\lbrace x\_1, \ldots, x\_N\rbrace  \sim p\_{\text{data}}$ but no access to $p\_{\text{data}}$ itself.

We parameterize a score model $s\_\theta(x) \approx \nabla\_x \log p\_{\text{data}}(x)$ and want to minimize:

$$
J(\theta) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}}\left[\Vert s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\Vert ^2\right]
$$

But we cannot compute this objective! It requires the true score $\nabla\_x \log p\_{\text{data}}(x)$, which we do not know.

### 2.2 The Score Matching Identity (Hyvarinen, 2005)

Hyvarinen's remarkable result: the objective $J(\theta)$ can be rewritten, up to a constant independent of $\theta$, as:

$$
\boxed{J_{\text{SM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\left[\text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2}\Vert s_\theta(x)\Vert ^2\right]}
$$

where $\nabla\_x s\_\theta(x) = \frac{\partial s\_\theta(x)\_i}{\partial x\_j}$ is the Jacobian of the score model, and $\text{tr}(\nabla\_x s\_\theta(x)) = \sum\_i \frac{\partial s\_\theta(x)\_i}{\partial x\_i}$ is its trace (the divergence of $s\_\theta$).

This objective involves only the model $s\_\theta$ and the data samples -- no knowledge of $p\_{\text{data}}$ is required beyond the samples.

### 2.3 Derivation

The derivation uses integration by parts. We start by expanding $J(\theta)$:

$$
J(\theta) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}}\left[\Vert s_\theta(x)\Vert ^2 - 2 s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x) + \Vert \nabla_x \log p_{\text{data}}(x)\Vert ^2\right]
$$

The last term is a constant (independent of $\theta$), so minimizing $J(\theta)$ is equivalent to minimizing:

$$
J'(\theta) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}}\left[\Vert s_\theta(x)\Vert ^2\right] - \mathbb{E}_{p_{\text{data}}}\left[s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)\right]
$$

The key step is handling the cross-term. Using $\nabla\_x \log p(x) = \nabla\_x p(x) / p(x)$:

$$
\mathbb{E}_{p_{\text{data}}}\left[s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)\right] = \int p_{\text{data}}(x) \sum_i s_{\theta,i}(x) \frac{\partial \log p_{\text{data}}(x)}{\partial x_i} dx
$$

$$
= \int \sum_i s_{\theta,i}(x) \frac{\partial p_{\text{data}}(x)}{\partial x_i} dx
$$

Now apply integration by parts to each term (assuming boundary terms vanish, i.e., $p\_{\text{data}}(x) s\_\theta(x) \to 0$ as $\Vert x\Vert  \to \infty$):

$$
\int s_{\theta,i}(x) \frac{\partial p_{\text{data}}(x)}{\partial x_i} dx = -\int p_{\text{data}}(x) \frac{\partial s_{\theta,i}(x)}{\partial x_i} dx
$$

Summing over $i$:

$$
\mathbb{E}_{p_{\text{data}}}\left[s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)\right] = -\mathbb{E}_{p_{\text{data}}}\left[\text{tr}(\nabla_x s_\theta(x))\right]
$$

Substituting back:

$$
J'(\theta) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}}\left[\Vert s_\theta(x)\Vert ^2\right] + \mathbb{E}_{p_{\text{data}}}\left[\text{tr}(\nabla_x s_\theta(x))\right] = J_{\text{SM}}(\theta)
$$

This completes the derivation. $\square$

### 2.4 The Computational Cost Problem

The score matching objective $J\_{\text{SM}}$ requires computing $\text{tr}(\nabla\_x s\_\theta(x))$, the trace of the Jacobian of the score model. For a model with $d$-dimensional input, this requires $d$ backward passes (one per diagonal element of the Jacobian), making it $O(d)$ times more expensive than a single forward pass.

For high-dimensional data (images with $d = 784$ or $d = 3072$ or more), this is prohibitively expensive.

**Sliced score matching** (Song et al., 2020) addresses this by projecting onto random directions:

$$
J_{\text{SSM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\mathbb{E}_{v \sim p_v}\left[v^\top \nabla_x s_\theta(x) v + \frac{1}{2}(v^\top s_\theta(x))^2\right]
$$

where $v$ is a random vector (e.g., $v \sim \mathcal{N}(0, I)$ or uniform on the unit sphere). The term $v^\top \nabla\_x s\_\theta(x) v$ requires only one backward pass (a Jacobian-vector product), regardless of $d$.

### 2.5 Denoising Score Matching

There is an even simpler approach: **denoising score matching** (Vincent, 2011). Instead of matching the score of $p\_{\text{data}}$, we match the score of a noise-perturbed distribution.

Define the noisy distribution:

$$
q_\sigma(x) = \int p_{\text{data}}(x_0) \mathcal{N}(x; x_0, \sigma^2 I) \, dx_0
$$

This is the data distribution convolved with Gaussian noise of standard deviation $\sigma$. Its score has a remarkable closed-form expression:

$$
\nabla_x \log q_\sigma(x) = \mathbb{E}_{q(x_0 \mid x)}\left[\frac{x_0 - x}{\sigma^2}\right]
$$

The denoising score matching objective is:

$$
J_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}(x_0)}\mathbb{E}_{\mathcal{N}(\epsilon; 0, I)}\left[\left\Vert s_\theta(x_0 + \sigma\epsilon) + \frac{\epsilon}{\sigma}\right\Vert ^2\right]
$$

This is equivalent to the original score matching objective up to a constant, but it is trivial to compute: sample a data point $x\_0$, add noise $\sigma\epsilon$ to get $x = x\_0 + \sigma\epsilon$, and train $s\_\theta(x)$ to predict $-\epsilon/\sigma$.

**The connection to diffusion:** In a diffusion model, the neural network is trained to predict the noise $\epsilon$ that was added to $x\_0$ to produce $x\_t$. This is exactly denoising score matching! The score at noise level $\sigma\_t = \sqrt{1-\bar{\alpha}\_t}/\sqrt{\bar{\alpha}\_t}$ is:

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

Training a network to predict $\epsilon$ is equivalent to training it to estimate the score.

---

## 3. Langevin Dynamics

### 3.1 The Algorithm

**Langevin dynamics** (named after the physicist Paul Langevin) is an MCMC method that generates samples from a distribution using only its score function. The update rule is:

$$
\boxed{x_{k+1} = x_k + \eta \nabla_x \log p(x_k) + \sqrt{2\eta} \, z_k, \quad z_k \sim \mathcal{N}(0, I)}
$$

where $\eta > 0$ is the step size.

The update has two components:
1. **Gradient ascent on $\log p$:** The term $\eta \nabla\_x \log p(x\_k)$ moves $x$ toward regions of higher probability density. This is deterministic and follows the score field.
2. **Gaussian noise:** The term $\sqrt{2\eta} z\_k$ adds random perturbation. This is essential -- without it, the algorithm would simply converge to a mode of $p$ and stay there.

The noise serves two purposes:
- **Exploration:** It prevents the algorithm from getting stuck at a single mode
- **Correct stationary distribution:** The specific scaling $\sqrt{2\eta}$ (relative to the gradient step $\eta$) is exactly what is needed for the chain to converge to $p(x)$

### 3.2 Why Does It Work?

The Langevin dynamics update is the discretization of a continuous-time stochastic differential equation:

$$
dx = \nabla_x \log p(x) \, dt + \sqrt{2} \, dW
$$

This is an **overdamped Langevin diffusion** from statistical physics. Under mild conditions on $p$ (e.g., smoothness and sufficient decay at infinity), this SDE has $p(x)$ as its unique stationary distribution.

**Intuition from physics.** Think of a particle moving in a potential landscape $V(x) = -\log p(x)$, subject to friction and thermal noise. The gradient term $\nabla\_x \log p(x) = -\nabla\_x V(x)$ is the force pulling the particle downhill (toward low energy = high probability). The noise term models thermal fluctuations. The Boltzmann distribution $p(x) \propto \exp(-V(x))$ is the thermal equilibrium -- the distribution the particle reaches after running long enough.

The critical ratio of noise to gradient ($\sqrt{2\eta}$ vs. $\eta$) corresponds to the Einstein relation in statistical physics: the fluctuation-dissipation theorem requires that the noise strength be proportional to the square root of the friction coefficient (here, the step size).

### 3.3 Convergence Under Log-Concavity

A distribution $p(x)$ is **log-concave** if $\log p(x)$ is a concave function, i.e., $\nabla^2 \log p(x) \preceq 0$ everywhere. This is equivalent to saying the negative Hessian of the log-density is positive semi-definite. Examples include Gaussians and log-concave exponential families.

**Strongly log-concave** means $\nabla^2 \log p(x) \preceq -mI$ for some $m > 0$. The function curves at least as sharply as a Gaussian with variance $1/m$ in every direction.

For strongly log-concave distributions with parameter $m$ and Lipschitz score (with constant $L$), Langevin dynamics with step size $\eta \leq 2/(m + L)$ converges exponentially fast:

$$
W_2(p_k, p) \leq (1 - m\eta)^k W_2(p_0, p) + O(\sqrt{\eta d})
$$

where $W\_2$ is the 2-Wasserstein distance and $d$ is the dimension. The first term is geometric convergence, and the second is the bias introduced by discretization (which vanishes as $\eta \to 0$).

This is good news: for "nice" distributions, Langevin dynamics works provably well. But most distributions we care about -- image distributions, for instance -- are emphatically not log-concave.

### 3.4 The Challenge of Multimodal Distributions

Langevin dynamics struggles with **multimodal distributions**. Consider a mixture of two well-separated Gaussians:

$$
p(x) = \frac{1}{2}\mathcal{N}(x; -10, 1) + \frac{1}{2}\mathcal{N}(x; 10, 1)
$$

Starting near one mode, the gradient points back toward that mode. To reach the other mode, the chain must traverse a region of very low density where $\log p(x)$ is extremely negative. The gradient in this region is nearly zero (both modes exert approximately equal and opposite pull), so the chain must rely entirely on the noise term to cross. For well-separated modes, this crossing takes exponentially long.

This is a fundamental problem: **Langevin dynamics mixes slowly between modes**. The mixing time grows exponentially with the distance between modes (relative to the noise level).

### 3.5 The Challenge of Low-Density Regions

A related problem: **score estimation is poor in low-density regions**. If we are estimating the score from data using score matching, the objective:

$$
J_{\text{SM}} = \mathbb{E}_{p_{\text{data}}}[\ldots]
$$

is an expectation under $p\_{\text{data}}$. The loss receives no training signal in regions where $p\_{\text{data}}(x) \approx 0$ -- there are simply no data points there. But Langevin dynamics initialized from a random starting point will pass through these low-density regions!

The estimated score in low-density regions is essentially random, so the Langevin chain receives no useful guidance when it needs it most.

---

## 4. Annealed Langevin Dynamics

### 4.1 The Key Insight

Song and Ermon (2019) proposed a solution to both problems: **use multiple noise levels**. The idea is elegantly simple.

Instead of estimating the score of $p\_{\text{data}}$ directly, estimate the scores of a sequence of noise-perturbed distributions:

$$
p_{\sigma_i}(x) = \int p_{\text{data}}(x_0) \mathcal{N}(x; x_0, \sigma_i^2 I) \, dx_0
$$

for a decreasing sequence of noise levels $\sigma\_1 > \sigma\_2 > \cdots > \sigma\_L$.

- At high noise levels ($\sigma\_1$ large): $p\_{\sigma\_1}$ is nearly Gaussian, unimodal, and fills the space. There are no low-density regions. Score estimation is easy. Langevin dynamics mixes quickly between modes.

- At low noise levels ($\sigma\_L$ small): $p\_{\sigma\_L} \approx p\_{\text{data}}$, which is what we want. But score estimation and mixing are hard.

The strategy: **start with Langevin dynamics at the highest noise level, then gradually reduce the noise**. At each level, run Langevin for a while, then use the resulting samples as initialization for the next (lower noise) level.

### 4.2 The Algorithm

**Annealed Langevin Dynamics:**

1. Train a noise-conditional score network $s\_\theta(x, \sigma)$ to estimate $\nabla\_x \log p\_\sigma(x)$ at all noise levels simultaneously
2. To sample:

$$
\textbf{for } i = 1, \ldots, L \textbf{ do:}
$$
$$
\quad \textbf{for } k = 1, \ldots, K \textbf{ do:}
$$
$$
\quad\quad x_{k} = x_{k-1} + \eta_i \, s_\theta(x_{k-1}, \sigma_i) + \sqrt{2\eta_i} \, z_k
$$
$$
\quad \textbf{end for}
$$
$$
\textbf{end for}
$$

Starting from $x\_0 \sim \mathcal{N}(0, \sigma\_1^2 I)$, the algorithm runs Langevin dynamics at each noise level in sequence, from highest to lowest. Each level refines the sample produced by the previous level.

### 4.3 Why It Works

At the highest noise level, $p\_{\sigma\_1}$ is approximately $\mathcal{N}(0, (\sigma\_1^2 + \text{data variance})I)$ -- nearly a single Gaussian. Langevin dynamics can easily sample from this. As the noise level decreases, the distribution becomes more complex (modes separate, fine structure appears), but the chain is already in a good region of space from the previous level.

The analogy is **simulated annealing**: start at high temperature (where the landscape is smooth) and gradually cool (revealing the true structure). But unlike simulated annealing (which converges to a single mode), annealed Langevin dynamics maintains a population of samples that track the distribution as it transforms from smooth to complex.

### 4.4 Connection to Diffusion Models

The connection is now almost obvious.

- The noise levels $\sigma\_1 > \cdots > \sigma\_L$ in annealed Langevin dynamics correspond to the timesteps $T, T-1, \ldots, 0$ in a diffusion model
- The noise-conditional score network $s\_\theta(x, \sigma)$ corresponds to the diffusion model's denoising network
- The forward process (adding noise at decreasing SNR) corresponds to defining the sequence of perturbed distributions
- The reverse process (denoising from $t = T$ to $t = 0$) corresponds to running annealed Langevin dynamics from high noise to low noise

**The reverse process in a diffusion model is doing annealed Langevin dynamics guided by a learned score function.**

This is the conceptual core of diffusion models. Everything else -- the specific parameterizations, the training objectives, the sampling algorithms -- is implementation detail (important implementation detail, but detail nonetheless).

### 4.5 The Score-Noise Equivalence

There is a precise mathematical relationship between the score function and the noise prediction that DDPM networks learn.

Given $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1-\bar{\alpha}\_t} \epsilon$, the score of the noisy distribution $q(x\_t)$ at a specific $x\_t$ that came from $x\_0$ is:

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}
$$

So if a network $\epsilon\_\theta(x\_t, t)$ learns to predict the noise $\epsilon$, the corresponding score estimate is:

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

Noise prediction and score estimation are the same thing, up to a scaling factor. This equivalence is why the DDPM framework (which predicts noise) and the score-based framework (which predicts scores) produce the same models.

---

## 5. Score Estimation in Practice

### 5.1 Score Networks

In practice, we parameterize the score function using a neural network $s\_\theta(x, t): \mathbb{R}^d \times \mathbb{R} \to \mathbb{R}^d$ that takes a noisy input $x$ and a noise level indicator $t$ and outputs a $d$-dimensional vector (the estimated score, which has the same dimensionality as $x$).

For images, $s\_\theta$ is typically a U-Net architecture (we will study this in Week 6). The key requirement is that the output has the same spatial dimensions as the input -- after all, the score at each pixel tells you which direction that pixel should be adjusted.

### 5.2 Time Conditioning

The score function depends on the noise level: $\nabla\_x \log p\_t(x)$ is a different function for each $t$. The network must know which noise level it is operating at.

Common approaches:
- **Sinusoidal embeddings:** Encode $t$ as $(\sin(\omega\_1 t), \cos(\omega\_1 t), \sin(\omega\_2 t), \cos(\omega\_2 t), \ldots)$, similar to positional encodings in Transformers
- **Learned embeddings:** Pass $t$ through a small MLP to produce a conditioning vector
- **FiLM conditioning:** Use the time embedding to modulate the intermediate features of the score network via $\gamma(t) \odot h + \beta(t)$

### 5.3 The Training Objective

Combining denoising score matching with the diffusion forward process, the training objective is:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}\lbrace 1,T\rbrace } \mathbb{E}_{x_0 \sim p_{\text{data}}} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \left[\Vert \epsilon_\theta(x_t, t) - \epsilon\Vert ^2\right]
$$

where $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1-\bar{\alpha}\_t} \epsilon$.

In words: sample a random timestep, sample a data point, add noise, and train the network to predict the noise that was added. This is the training algorithm of DDPM.

We will derive this objective rigorously in Week 4 from the variational bound on the log-likelihood. For now, note that it is equivalent to denoising score matching averaged over all noise levels.

---

## 6. Putting It Together: From Score to Sampler

### 6.1 The Sampling Algorithm

Given a trained score network $s\_\theta(x\_t, t)$ (or equivalently, a noise predictor $\epsilon\_\theta(x\_t, t)$), we can generate samples by running annealed Langevin dynamics. The DDPM sampling algorithm is:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

for $t = T, T-1, \ldots, 1$, starting from $x\_T \sim \mathcal{N}(0, I)$.

This is the reverse process. Each step:
1. Uses the score estimate (via $\epsilon\_\theta$) to predict where the cleaner version $x\_{t-1}$ likely lies
2. Adds noise ($\sigma\_t z$) to maintain the correct stochastic dynamics

### 6.2 The Reverse Process Is Approximate Langevin

The DDPM reverse step can be rewritten as:

$$
x_{t-1} = x_t + \frac{1-\alpha_t}{\sqrt{\alpha_t}} \left(\frac{1}{\sqrt{1-\bar{\alpha}_t}} \cdot \frac{x_t}{\sqrt{\alpha_t}} - \frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z
$$

After some algebra (which we will do in Week 3), this takes the form:

$$
x_{t-1} \approx x_t + \eta_t \, s_\theta(x_t, t) + \sigma_t z
$$

This is Langevin dynamics with the learned score, with step size $\eta\_t$ and noise scale $\sigma\_t$ determined by the noise schedule. The reverse process is literally doing what we described in Sections 3-4: using the score to move toward high density, with noise to maintain the correct distribution.

---

## Summary

1. **The score function** $\nabla\_x \log p(x)$ is a vector field pointing toward high density. It does not require the normalizing constant, making it easier to work with than the density itself.

2. **Score matching** (Hyvarinen 2005) allows estimation of the score from data samples alone, using an integration-by-parts identity that eliminates the unknown true score from the training objective.

3. **Denoising score matching** (Vincent 2011) simplifies this further: train a network to denoise corrupted data. The optimal denoiser implicitly estimates the score.

4. **Langevin dynamics** generates samples using only the score: $x\_{k+1} = x\_k + \eta \nabla\_x \log p(x\_k) + \sqrt{2\eta} z\_k$. It converges to the target distribution under log-concavity but struggles with multimodal distributions.

5. **Annealed Langevin dynamics** (Song and Ermon, 2019) solves the multimodality problem by running Langevin at multiple noise levels, from high (smooth, easy) to low (complex, accurate).

6. **The connection to diffusion:** The reverse process of a diffusion model is annealed Langevin dynamics with a learned score. Noise prediction ($\epsilon\_\theta$) and score estimation ($s\_\theta$) are related by $s\_\theta = -\epsilon\_\theta / \sqrt{1-\bar{\alpha}\_t}$.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| Score function | $s(x) = \nabla\_x \log p(x)$ |
| Gaussian score | $\nabla\_x \log \mathcal{N}(x; \mu, \sigma^2 I) = -(x-\mu)/\sigma^2$ |
| Score matching (Hyvarinen) | $J\_{\text{SM}} = \mathbb{E}\_p[\text{tr}(\nabla\_x s\_\theta) + \frac{1}{2}\Vert s\_\theta\Vert ^2]$ |
| Denoising score matching | $J\_{\text{DSM}} = \mathbb{E}\_{x\_0, \epsilon}[\Vert s\_\theta(x\_0 + \sigma\epsilon) + \epsilon/\sigma\Vert ^2]$ |
| Langevin dynamics | $x\_{k+1} = x\_k + \eta \nabla\_x \log p(x\_k) + \sqrt{2\eta}\, z\_k$ |
| Score-noise equivalence | $s\_\theta(x\_t, t) = -\epsilon\_\theta(x\_t, t) / \sqrt{1-\bar{\alpha}\_t}$ |

---

## Suggested Reading

- **Hyvarinen** (2005), "Estimation of Non-Normalized Statistical Models by Score Matching" -- the original score matching paper. Short and elegant.
- **Vincent** (2011), "A Connection Between Score Matching and Denoising Autoencoders" -- establishes the denoising score matching equivalence.
- **Song and Ermon** (2019), "Generative Modeling by Estimating Gradients of the Data Distribution" -- Noise Conditional Score Networks (NCSN) and annealed Langevin dynamics. The paper that connected score matching to generative modeling.
- **Song and Ermon** (2020), "Improved Techniques for Training Score-Based Generative Models" -- practical improvements to NCSN.
- **Roberts and Tweedie** (1996), "Exponential Convergence of Langevin Distributions and Their Discrete Approximations" -- convergence theory for Langevin dynamics.
- **Welling and Teh** (2011), "Bayesian Learning via Stochastic Gradient Langevin Dynamics" -- using Langevin dynamics for Bayesian inference, connecting SGLD to SGD.
