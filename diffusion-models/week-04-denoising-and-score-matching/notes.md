# Week 4: Denoising and Score Matching

> *"Tell me what noise you added, and I will tell you what you are."*
> -- the voice of Tweedie's formula, if it could speak

---

## Overview

Last week, we established the mathematical foundations: SDEs, Ito calculus, the Fokker-Planck equation, and Anderson's reverse-time SDE. The punchline was that reversing a diffusion process requires knowledge of the score function $\nabla\_x \log p\_t(x)$ at each noise level.

This week, we answer the obvious follow-up: **how do you actually estimate the score function?**

The answer is one of the most beautiful results in the intersection of statistics and machine learning. It connects two problems that seem entirely unrelated:

1. **Denoising:** Given a noisy observation $\tilde{x} = x + \sigma\epsilon$, estimate the clean signal $x$.
2. **Score estimation:** Estimate $\nabla\_x \log p(x)$, the gradient of the log-density.

These are the **same problem**. The optimal denoiser directly gives the score, and training a neural network to denoise is equivalent to training it to estimate the score. The bridge between them is **Tweedie's formula**, and the formal connection is the **denoising score matching** identity of Vincent (2011).

This is not just an elegant theoretical observation -- it is the practical recipe that makes diffusion models work. Every diffusion model you have ever used (DALL-E, Stable Diffusion, Midjourney) is, at its core, a denoiser trained across multiple noise levels.

### Prerequisites
- Week 2: Score functions, basic score matching
- Week 3: SDEs, Fokker-Planck equation, Anderson's reverse SDE, the OU process transition kernel
- Probability: Conditional expectation, Bayes' theorem, Gaussian identities

---

## 1. The Denoising Problem

### 1.1 Setup

Consider a clean signal $x \sim p(x)$ corrupted by additive Gaussian noise:

$$
\tilde{x} = x + \sigma \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

The noisy observation $\tilde{x}$ is drawn from the convolved distribution:

$$
p_\sigma(\tilde{x}) = \int p(x) \, \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, dx = (p * \mathcal{N}_{\sigma^2})(\tilde{x})
$$

where $*$ denotes convolution. The noise blurs the data distribution by convolving it with a Gaussian kernel.

**The denoising problem:** Given $\tilde{x}$, estimate $x$.

### 1.2 The Optimal Denoiser

The optimal denoiser under mean squared error is the **posterior mean** (also called the Bayes estimator):

$$
\hat{x}(\tilde{x}) = \mathbb{E}[x \mid \tilde{x}]
$$

This minimizes $\mathbb{E}[\Vert x - \hat{x}(\tilde{x})\Vert ^2]$ over all possible estimators $\hat{x}$. The proof is a standard exercise in estimation theory: any other estimator adds variance without reducing bias.

The posterior mean is determined by Bayes' theorem:

$$
p(x \mid \tilde{x}) = \frac{p(\tilde{x} \mid x) p(x)}{p(\tilde{x})} = \frac{\mathcal{N}(\tilde{x}; x, \sigma^2 I) \, p(x)}{p_\sigma(\tilde{x})}
$$

For general $p(x)$, the posterior $p(x|\tilde{x})$ and hence the posterior mean $\mathbb{E}[x|\tilde{x}]$ are intractable. But there is a remarkable shortcut.

### 1.3 A Simple Example

Before the general theory, consider a concrete case. Suppose $x \sim \mathcal{N}(\mu, \tau^2)$ and $\tilde{x} = x + \sigma\epsilon$. Then:

$$
p(x \mid \tilde{x}) \propto \exp\left(-\frac{(\tilde{x} - x)^2}{2\sigma^2}\right) \exp\left(-\frac{(x - \mu)^2}{2\tau^2}\right)
$$

Completing the square, $p(x|\tilde{x})$ is Gaussian with mean:

$$
\mathbb{E}[x \mid \tilde{x}] = \frac{\sigma^2 \mu + \tau^2 \tilde{x}}{\sigma^2 + \tau^2} = \tilde{x} - \frac{\sigma^2(\tilde{x} - \mu)}{\sigma^2 + \tau^2}
$$

The optimal denoiser shrinks $\tilde{x}$ toward the prior mean $\mu$. The amount of shrinkage depends on the signal-to-noise ratio $\tau^2/\sigma^2$: high noise means more shrinkage.

We can rewrite this as:

$$
\mathbb{E}[x \mid \tilde{x}] = \tilde{x} + \sigma^2 \underbrace{\left(-\frac{\tilde{x} - \mu}{\sigma^2 + \tau^2}\right)}_{\nabla_{\tilde{x}} \log p_\sigma(\tilde{x})}
$$

Since $p\_\sigma(\tilde{x}) = \mathcal{N}(\tilde{x}; \mu, \sigma^2 + \tau^2)$, we have $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x}) = -\frac{\tilde{x} - \mu}{\sigma^2 + \tau^2}$, confirming the formula. This is Tweedie's formula for the Gaussian case.

---

## 2. Tweedie's Formula

### 2.1 Statement

For the noise model $\tilde{x} = x + \sigma\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$:

$$
\boxed{\mathbb{E}[x \mid \tilde{x}] = \tilde{x} + \sigma^2 \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})}
$$

where $p\_\sigma(\tilde{x}) = \int p(x) \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, dx$ is the marginal density of the noisy observation.

This is **Tweedie's formula** (Robbins 1956, Efron 2011). It holds for **any** data distribution $p(x)$ -- not just Gaussians.

The formula says: the optimal denoiser equals the noisy observation plus $\sigma^2$ times the score of the noisy distribution. In other words:

$$
\text{Optimal denoiser} = \text{Identity} + \sigma^2 \times \text{Score}
$$

Or equivalently:

$$
\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) = \frac{\mathbb{E}[x \mid \tilde{x}] - \tilde{x}}{\sigma^2} = -\frac{\mathbb{E}[\epsilon \mid \tilde{x}]}{\sigma}
$$

The score points from the noisy observation toward the clean signal (in expectation).

### 2.2 Proof

The proof is a direct computation using Bayes' theorem and integration by parts.

**Step 1.** Write the score:

$$
\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) = \frac{\nabla_{\tilde{x}} p_\sigma(\tilde{x})}{p_\sigma(\tilde{x})}
$$

**Step 2.** Compute $\nabla\_{\tilde{x}} p\_\sigma(\tilde{x})$:

$$
\nabla_{\tilde{x}} p_\sigma(\tilde{x}) = \nabla_{\tilde{x}} \int p(x) \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, dx = \int p(x) \nabla_{\tilde{x}} \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, dx
$$

The gradient of the Gaussian likelihood:

$$
\nabla_{\tilde{x}} \mathcal{N}(\tilde{x}; x, \sigma^2 I) = \mathcal{N}(\tilde{x}; x, \sigma^2 I) \cdot \frac{x - \tilde{x}}{\sigma^2}
$$

So:

$$
\nabla_{\tilde{x}} p_\sigma(\tilde{x}) = \int p(x) \mathcal{N}(\tilde{x}; x, \sigma^2 I) \frac{x - \tilde{x}}{\sigma^2} \, dx = \frac{1}{\sigma^2} \int (x - \tilde{x}) \, p(x) \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, dx
$$

**Step 3.** Divide by $p\_\sigma(\tilde{x})$:

$$
\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) = \frac{1}{\sigma^2} \frac{\int (x - \tilde{x}) p(x) \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, dx}{\int p(x) \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, dx}
$$

The numerator is $\mathbb{E}[x - \tilde{x} \mid \tilde{x}] \cdot p\_\sigma(\tilde{x})$ (by definition of conditional expectation). So:

$$
\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) = \frac{\mathbb{E}[x \mid \tilde{x}] - \tilde{x}}{\sigma^2}
$$

Rearranging gives Tweedie's formula. $\square$

### 2.3 The Deepest Insight

Let us state this again, because it is the single most important idea in the course:

> **Denoising and score estimation are the same problem.**

If you have a perfect denoiser $D\_\sigma(\tilde{x}) = \mathbb{E}[x|\tilde{x}]$, you can extract the score:

$$
\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) = \frac{D_\sigma(\tilde{x}) - \tilde{x}}{\sigma^2}
$$

If you have the score $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$, you can construct the optimal denoiser:

$$
D_\sigma(\tilde{x}) = \tilde{x} + \sigma^2 \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})
$$

They are two views of the same mathematical object. This is why diffusion models can be trained as denoisers -- they are simultaneously learning the score function needed for Anderson's reverse SDE.

### 2.4 Three Equivalent Parameterizations

Given the noise model $\tilde{x} = x + \sigma\epsilon$, a neural network can be trained to predict any of:

1. **The clean signal $x$:** The network $D\_\theta(\tilde{x}, \sigma) \approx \mathbb{E}[x|\tilde{x}]$ directly outputs the denoised signal.
2. **The noise $\epsilon$:** The network $\epsilon\_\theta(\tilde{x}, \sigma) \approx \mathbb{E}[\epsilon|\tilde{x}]$ predicts what noise was added.
3. **The score $\nabla \log p\_\sigma$:** The network $s\_\theta(\tilde{x}, \sigma) \approx \nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$ directly estimates the score.

These are all equivalent, related by:

$$
s_\theta = \frac{D_\theta - \tilde{x}}{\sigma^2} = -\frac{\epsilon_\theta}{\sigma}
$$

In practice:
- DDPM (Ho et al. 2020) predicts the noise $\epsilon$
- Score-based models (Song and Ermon 2019) predict the score $s$
- Some recent work (Karras et al. 2022) predicts the clean signal $x$ (with appropriate preconditioning)

All three parameterizations are mathematically identical. The choice affects numerical stability and training dynamics but not the underlying theory.

---

## 3. Score Matching

### 3.1 The Basic Problem

We want to train a neural network $s\_\theta(x)$ to approximate $\nabla\_x \log p(x)$. The natural loss function is:

$$
\mathcal{L}_{\text{SM}}(\theta) = \frac{1}{2}\mathbb{E}_{x \sim p}\left[\Vert s_\theta(x) - \nabla_x \log p(x)\Vert ^2\right]
$$

But we do not know $\nabla\_x \log p(x)$ -- that is what we are trying to estimate! This loss is not computable.

### 3.2 Hyvarinen's Trick (2005)

Hyvarinen (2005) showed that the loss above can be rewritten as:

$$
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{x \sim p}\left[\frac{1}{2}\Vert s_\theta(x)\Vert ^2 + \nabla_x \cdot s_\theta(x)\right] + \text{const}
$$

where $\nabla\_x \cdot s\_\theta = \sum\_i \frac{\partial [s\_\theta]\_i}{\partial x\_i}$ is the divergence of the score model, and the constant does not depend on $\theta$.

The proof uses integration by parts (see Week 2). This is remarkable: the true score $\nabla\_x \log p(x)$ has disappeared from the loss. We only need samples from $p$ and the ability to compute the divergence of $s\_\theta$.

**The problem:** Computing $\nabla\_x \cdot s\_\theta(x)$ requires computing a diagonal of the Jacobian of $s\_\theta$, which is expensive for high-dimensional $x$ (it requires $d$ backward passes). This makes explicit score matching impractical for images.

### 3.3 The Key Question

Is there a way to train a score network that:
1. Does not require knowing the true score $\nabla\_x \log p(x)$
2. Does not require computing the expensive divergence term
3. Only requires samples from $p(x)$

The answer is **denoising score matching**.

---

## 4. Denoising Score Matching

### 4.1 The Idea (Vincent, 2011)

Instead of matching the score of the data distribution $p(x)$, match the score of the **noised** distribution $p\_\sigma(\tilde{x})$.

The noised distribution is:

$$
p_\sigma(\tilde{x}) = \int p(x) \, \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, dx
$$

Its score $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$ is well-defined (unlike $\nabla\_x \log p(x)$, which may be problematic at low-density regions). More importantly, we can compute a tractable training objective.

### 4.2 The Denoising Score Matching Loss

Train $s\_\theta(\tilde{x})$ to minimize:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{x \sim p, \, \epsilon \sim \mathcal{N}(0, I)}\left[\left\Vert s_\theta(x + \sigma\epsilon) - \nabla_{\tilde{x}} \log p(\tilde{x} \mid x)\right\Vert ^2\right]
$$

The key: we replaced the intractable $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$ with the **tractable** $\nabla\_{\tilde{x}} \log p(\tilde{x} \mid x)$.

Since $p(\tilde{x} \mid x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$:

$$
\nabla_{\tilde{x}} \log p(\tilde{x} \mid x) = \frac{x - \tilde{x}}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

So the loss becomes:

$$
\boxed{\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{x \sim p, \, \epsilon \sim \mathcal{N}(0,I)}\left[\left\Vert s_\theta(x + \sigma\epsilon) + \frac{\epsilon}{\sigma}\right\Vert ^2\right]}
$$

This is stunning in its simplicity:
1. Sample a data point $x$
2. Add noise: $\tilde{x} = x + \sigma\epsilon$
3. Train the network to predict $-\epsilon/\sigma$ from $\tilde{x}$

That is it. No knowledge of $p(x)$. No divergence computation. Just noise prediction.

### 4.3 The Equivalence Theorem

**Theorem (Vincent, 2011).** The denoising score matching loss $\mathcal{L}\_{\text{DSM}}(\theta)$ differs from the score matching loss $\mathcal{L}\_{\text{SM}}(\theta)$ by a constant independent of $\theta$:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{\tilde{x} \sim p_\sigma}\left[\Vert s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})\Vert ^2\right] + C
$$

where $C$ does not depend on $\theta$.

In other words: **minimizing the denoising score matching loss finds the score of the noised distribution.** The network $s\_\theta$ that minimizes $\mathcal{L}\_{\text{DSM}}$ satisfies $s\_\theta(\tilde{x}) = \nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$.

### 4.4 Proof of the Equivalence

Expand the score matching loss for the noised distribution:

$$
\mathcal{L}_{\text{SM}}(\theta) = \frac{1}{2}\mathbb{E}_{\tilde{x} \sim p_\sigma}\left[\Vert s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})\Vert ^2\right]
$$

$$
= \frac{1}{2}\mathbb{E}_{\tilde{x}}\left[\Vert s_\theta(\tilde{x})\Vert ^2 - 2 s_\theta(\tilde{x})^\top \nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) + \Vert \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})\Vert ^2\right]
$$

The last term is a constant (does not depend on $\theta$). Focus on the cross term:

$$
\mathbb{E}_{\tilde{x} \sim p_\sigma}\left[s_\theta(\tilde{x})^\top \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})\right]
$$

Now, $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x}) = \frac{\nabla\_{\tilde{x}} p\_\sigma(\tilde{x})}{p\_\sigma(\tilde{x})}$, so:

$$
= \int s_\theta(\tilde{x})^\top \frac{\nabla_{\tilde{x}} p_\sigma(\tilde{x})}{p_\sigma(\tilde{x})} \, p_\sigma(\tilde{x}) \, d\tilde{x} = \int s_\theta(\tilde{x})^\top \nabla_{\tilde{x}} p_\sigma(\tilde{x}) \, d\tilde{x}
$$

Substituting $p\_\sigma(\tilde{x}) = \int p(x) \mathcal{N}(\tilde{x}; x, \sigma^2 I) dx$:

$$
= \int \int s_\theta(\tilde{x})^\top \nabla_{\tilde{x}} \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, p(x) \, dx \, d\tilde{x}
$$

$$
= \int \int s_\theta(\tilde{x})^\top \frac{x - \tilde{x}}{\sigma^2} \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, p(x) \, dx \, d\tilde{x}
$$

$$
= \mathbb{E}_{x \sim p, \, \tilde{x} \sim \mathcal{N}(x, \sigma^2 I)}\left[s_\theta(\tilde{x})^\top \frac{x - \tilde{x}}{\sigma^2}\right]
$$

$$
= \mathbb{E}_{x, \tilde{x}}\left[s_\theta(\tilde{x})^\top \nabla_{\tilde{x}} \log p(\tilde{x} | x)\right]
$$

Substituting back into $\mathcal{L}\_{\text{SM}}$:

$$
\mathcal{L}_{\text{SM}}(\theta) = \frac{1}{2}\mathbb{E}_{\tilde{x}}\Vert s_\theta(\tilde{x})\Vert ^2 - \mathbb{E}_{x, \tilde{x}}\left[s_\theta(\tilde{x})^\top \nabla_{\tilde{x}} \log p(\tilde{x}|x)\right] + C
$$

Now expand the denoising score matching loss:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{x, \tilde{x}}\left[\Vert s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p(\tilde{x}|x)\Vert ^2\right]
$$

$$
= \frac{1}{2}\mathbb{E}\Vert s_\theta(\tilde{x})\Vert ^2 - \mathbb{E}\left[s_\theta(\tilde{x})^\top \nabla_{\tilde{x}} \log p(\tilde{x}|x)\right] + \frac{1}{2}\mathbb{E}\Vert \nabla_{\tilde{x}} \log p(\tilde{x}|x)\Vert ^2
$$

The first two terms are identical to $\mathcal{L}\_{\text{SM}}$ (after noting that $\mathbb{E}\_{\tilde{x}}\Vert s\_\theta\Vert ^2 = \mathbb{E}\_{x, \tilde{x}}\Vert s\_\theta\Vert ^2$ since $s\_\theta$ depends only on $\tilde{x}$). The last term is a constant. Therefore:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathcal{L}_{\text{SM}}(\theta) + C'
$$

where $C' = \frac{1}{2}\mathbb{E}\Vert \nabla\_{\tilde{x}} \log p(\tilde{x}|x)\Vert ^2 - C$ is independent of $\theta$. $\square$

### 4.5 The Loss in Practice

Substituting $\nabla\_{\tilde{x}} \log p(\tilde{x}|x) = -\epsilon/\sigma$, the DSM loss is:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{x, \epsilon}\left[\left\Vert s_\theta(x + \sigma\epsilon) + \frac{\epsilon}{\sigma}\right\Vert ^2\right]
$$

If we reparameterize the network as $s\_\theta(\tilde{x}, \sigma) = -\epsilon\_\theta(\tilde{x}, \sigma)/\sigma$, the loss becomes:

$$
\mathcal{L}(\theta) = \frac{1}{2\sigma^2}\mathbb{E}_{x, \epsilon}\left[\Vert \epsilon_\theta(x + \sigma\epsilon, \sigma) - \epsilon\Vert ^2\right]
$$

This is the **noise prediction loss**: train the network to predict the noise that was added. DDPM (Ho et al. 2020) uses exactly this loss (without the $1/\sigma^2$ weighting, as we discuss below).

---

## 5. Noise-Conditional Score Networks

### 5.1 The Single-Scale Problem

So far, we have a score network $s\_\theta(\tilde{x})$ trained at a single noise level $\sigma$. But for the reverse SDE (Anderson's theorem), we need the score at every noise level $t$ along the diffusion trajectory.

If $\sigma$ is too small, the noised distribution $p\_\sigma$ is nearly identical to $p$ -- the score is accurate but the landscape is complex (many modes), making Langevin dynamics/reverse SDE simulation difficult.

If $\sigma$ is too large, $p\_\sigma$ is nearly Gaussian -- the landscape is simple but the score tells us little about $p$.

### 5.2 The Multi-Scale Solution

The solution: train a **single network** $s\_\theta(\tilde{x}, \sigma)$ that takes the noise level $\sigma$ as an additional input and estimates the score at that noise level.

This is a **noise-conditional score network** (NCSN), introduced by Song and Ermon (2019). The network learns to denoise at all scales simultaneously.

### 5.3 Multi-Scale Denoising Loss

Choose a set of noise levels $\sigma\_1 < \sigma\_2 < \cdots < \sigma\_L$ (or sample them continuously from some distribution). The multi-scale denoising score matching loss is:

$$
\boxed{\mathcal{L}(\theta) = \sum_{\ell=1}^{L} \lambda(\sigma_\ell) \, \mathbb{E}_{x \sim p, \, \epsilon \sim \mathcal{N}(0,I)}\left[\left\Vert s_\theta(x + \sigma_\ell \epsilon, \sigma_\ell) + \frac{\epsilon}{\sigma_\ell}\right\Vert ^2\right]}
$$

where $\lambda(\sigma\_\ell)$ are **loss weights** that balance the contributions from different noise levels.

In the continuous-time formulation (Song et al. 2021), the sum over $\ell$ becomes an integral over $t$, and the noise level $\sigma(t)$ is determined by the forward SDE:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,T]} \left[\lambda(t) \, \mathbb{E}_{x_0, x_t}\left[\Vert s_\theta(x_t, t) - \nabla_{x_t} \log p(x_t | x_0)\Vert ^2\right]\right]
$$

### 5.4 Choosing the Loss Weights

The choice of $\lambda(\sigma)$ significantly affects training. Several schemes have been proposed:

**Uniform weighting:** $\lambda(\sigma) = 1$. Simple but problematic -- large $\sigma$ terms dominate because $\Vert \epsilon/\sigma\Vert ^2 \sim d/\sigma^2$ is small for large $\sigma$, so the loss is dominated by the small-$\sigma$ terms where the score is large.

**$\sigma^2$ weighting:** $\lambda(\sigma) = \sigma^2$. This gives:

$$
\lambda(\sigma) \cdot \Vert s_\theta + \epsilon/\sigma\Vert ^2 = \Vert \sigma s_\theta + \epsilon\Vert ^2 = \Vert \epsilon_\theta - \epsilon\Vert ^2
$$

which is the noise prediction loss without any $\sigma$-dependent weighting. This is the DDPM weighting (Ho et al. 2020), and it works surprisingly well in practice.

**Likelihood weighting:** $\lambda(t) = g(t)^2$, where $g(t)$ is the diffusion coefficient from the SDE. Song et al. (2021) showed that this weighting gives a loss that upper-bounds the negative log-likelihood.

### 5.5 Noise Level Conditioning

How does the network "know" what noise level it is operating at? Common approaches:

**Sinusoidal embeddings:** Encode $\sigma$ (or $t$, or $\log \sigma$) using sinusoidal position embeddings (borrowed from the Transformer):

$$
\gamma(\sigma) = \left[\sin(\omega_1 \sigma), \cos(\omega_1 \sigma), \sin(\omega_2 \sigma), \cos(\omega_2 \sigma), \ldots\right]
$$

with geometrically spaced frequencies $\omega\_k$. This embedding is then injected into the network (e.g., via FiLM conditioning: scale and shift the activations of each layer).

**Log-SNR conditioning:** Instead of $\sigma$, condition on $\log(\text{SNR}) = \log(\alpha^2/\sigma^2)$ where $\alpha$ and $\sigma$ parameterize the noise schedule. This is more natural because the network's task changes most rapidly in log-SNR space.

---

## 6. Connection to the Diffusion Model Forward Process

### 6.1 The Forward Process Revisited

Recall from Week 3 the VP forward SDE:

$$
dX_t = -\frac{1}{2}\beta(t) X_t \, dt + \sqrt{\beta(t)} \, dW_t
$$

The transition kernel (conditional distribution of $X\_t$ given $X\_0$) is:

$$
X_t \mid X_0 \sim \mathcal{N}\left(\alpha_t X_0, \; (1 - \alpha_t^2) I\right)
$$

where $\alpha\_t = \exp\left(-\frac{1}{2}\int\_0^t \beta(s) \, ds\right)$.

We can write this as:

$$
X_t = \alpha_t X_0 + \sqrt{1 - \alpha_t^2} \, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

This is a noising operation with signal coefficient $\alpha\_t$ and noise standard deviation $\sigma\_t = \sqrt{1 - \alpha\_t^2}$.

### 6.2 The Score of the Forward Process

The conditional score is:

$$
\nabla_{X_t} \log p(X_t \mid X_0) = -\frac{X_t - \alpha_t X_0}{1 - \alpha_t^2} = -\frac{\epsilon}{\sigma_t}
$$

where we used $X\_t - \alpha\_t X\_0 = \sigma\_t \epsilon$.

The denoising score matching loss for the forward process is therefore:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\lambda(t) \left\Vert s_\theta(\alpha_t x_0 + \sigma_t \epsilon, \; t) + \frac{\epsilon}{\sigma_t}\right\Vert ^2\right]
$$

With $\lambda(t) = \sigma\_t^2$, this simplifies to:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\Vert \epsilon_\theta(\alpha_t x_0 + \sigma_t \epsilon, \; t) - \epsilon\Vert ^2\right]
$$

This is the DDPM training objective. The training algorithm is:

```
Repeat:
    Sample x_0 from the dataset
    Sample t uniformly from [0, T]
    Sample epsilon from N(0, I)
    Compute x_t = alpha_t * x_0 + sigma_t * epsilon
    Take gradient step on ||epsilon_theta(x_t, t) - epsilon||^2
```

Five lines. That is the entire training procedure for a diffusion model.

### 6.3 Score for the Reverse Process

Once trained, the score network is plugged into Anderson's reverse SDE:

$$
dX_t = \left[-\frac{1}{2}\beta(t) X_t - \beta(t) s_\theta(X_t, t)\right] dt + \sqrt{\beta(t)} \, d\bar{W}_t
$$

Starting from $X\_T \sim \mathcal{N}(0, I)$ and integrating backward produces samples from (approximately) $p\_{\text{data}}$.

---

## 7. Why Does This Work? Building Intuition

### 7.1 The Denoising Interpretation

At each noise level, the network learns: "given this noisy image, which direction leads toward cleaner images?" At high noise levels, the answer is broad (move toward the center of the data distribution). At low noise levels, the answer is precise (add this specific detail, sharpen this edge).

The reverse process starts from pure noise and progressively reduces the noise level, asking the network "what should this look like if it were slightly less noisy?" at each step. The coarse-to-fine structure is automatic -- no curriculum is needed.

### 7.2 The Score Field Interpretation

The score $\nabla\_x \log p\_t(x)$ is a vector field that points "uphill" in the density landscape. At high noise levels, the density is nearly Gaussian and the score field is simple (pointing toward the origin). At low noise levels, the density has complex structure (multiple modes, sharp features) and the score field is correspondingly complex.

The reverse SDE follows this score field, guided from the simple structure at high noise through the complex structure at low noise.

### 7.3 The Information-Theoretic View

The forward process destroys information about the data at a controlled rate. The score function, at each time, encodes exactly the information needed to undo the last increment of destruction. Training the score network builds a "memory" of what the data distribution looks like at every scale of noise.

---

## 8. Practical Considerations

### 8.1 The Architecture

The standard architecture for the score/noise-prediction network is a **U-Net** (Ronneberger et al. 2015): an encoder-decoder CNN with skip connections. The time/noise level is injected via sinusoidal embeddings and FiLM conditioning.

We will study the architecture in detail in Week 9. For now, the important point is that the architecture must:
1. Take an image (or data point) and a noise level as input
2. Output a tensor of the same shape as the input (the predicted noise or score)
3. Have enough capacity to capture the score at all noise levels

### 8.2 Signal-to-Noise Ratio

The quality of the score estimate depends on the signal-to-noise ratio. At very high noise ($\sigma \gg$ data scale), the noised distribution is nearly Gaussian and the score is easy to estimate. At very low noise ($\sigma \to 0$), the noised distribution approaches the data distribution, which may have complex structure.

In practice, the hardest noise levels to model are the intermediate ones where the data structure is partially but not fully obscured. Getting these right is crucial for sample quality.

### 8.3 Weighting and Loss Design

The choice of loss weighting $\lambda(t)$ affects which noise levels the network focuses on. Ho et al. (2020) found that the simple "noise prediction" weighting $\lambda(t) = \sigma\_t^2$ (which drops the $1/\sigma^2$ factor) works better than the theoretically motivated likelihood weighting for sample quality.

Karras et al. (2022) conducted a thorough empirical study and proposed a preconditioning scheme that makes training more stable across noise levels. We will revisit this in Week 9.

---

## Summary

1. **The denoising problem:** Given $\tilde{x} = x + \sigma\epsilon$, the optimal denoiser is the posterior mean $\mathbb{E}[x|\tilde{x}]$.

2. **Tweedie's formula:** $\mathbb{E}[x|\tilde{x}] = \tilde{x} + \sigma^2 \nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$. The optimal denoiser is the identity plus $\sigma^2$ times the score. **Denoising and score estimation are the same problem.**

3. **Denoising score matching (Vincent 2011):** Training a network to predict the noise $\epsilon$ that was added is equivalent to training it to estimate the score $\nabla \log p\_\sigma$. The loss $\Vert s\_\theta(\tilde{x}) + \epsilon/\sigma\Vert ^2$ is a tractable proxy for the true score matching loss.

4. **Noise-conditional score networks:** A single network $s\_\theta(x, \sigma)$ estimates the score at all noise levels simultaneously. Multi-scale training uses a weighted sum of denoising losses across noise levels.

5. **The DDPM training objective** reduces to: sample data, add noise at a random level, predict the noise. Five lines of pseudocode.

6. **Three equivalent parameterizations:** Predicting the clean signal $x$, the noise $\epsilon$, or the score $\nabla \log p\_\sigma$ are all equivalent formulations of the same problem.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| Tweedie's formula | $\mathbb{E}[x\Vert \tilde{x}] = \tilde{x} + \sigma^2 \nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x})$ |
| Score from noise | $\nabla\_{\tilde{x}} \log p\_\sigma(\tilde{x}) = -\mathbb{E}[\epsilon\Vert \tilde{x}]/\sigma$ |
| DSM loss (score) | $\mathcal{L} = \mathbb{E}\[\Vert s\_\theta(x+\sigma\epsilon) + \epsilon/\sigma\Vert ^2\]$ |
| DSM loss (noise) | $\mathcal{L} = \mathbb{E}\[\Vert \epsilon\_\theta(x+\sigma\epsilon) - \epsilon\Vert ^2\]$ |
| Forward process | $x\_t = \alpha\_t x\_0 + \sigma\_t \epsilon$ |
| Parameterization | $s\_\theta = -\epsilon\_\theta/\sigma = (D\_\theta - \tilde{x})/\sigma^2$ |
| DDPM training | Sample $x\_0, t, \epsilon$; minimize $\Vert \epsilon\_\theta(\alpha\_t x\_0 + \sigma\_t\epsilon, t) - \epsilon\Vert ^2$ |

---

## Suggested Reading

- **Vincent, P.** (2011), "A Connection Between Score Matching and Denoising Autoencoders" -- the foundational paper establishing the denoising-score matching equivalence. Clear and concise.
- **Hyvarinen, A.** (2005), "Estimation of Non-Normalized Statistical Models by Score Matching" -- the original score matching paper.
- **Song, Y. and Ermon, S.** (2019), "Generative Modeling by Estimating Gradients of the Data Distribution" -- noise-conditional score networks (NCSN).
- **Ho, J., Jain, A., and Abbeel, P.** (2020), "Denoising Diffusion Probabilistic Models" -- DDPM, the noise-prediction formulation.
- **Efron, B.** (2011), "Tweedie's Formula and Selection Bias" -- a beautiful exposition of Tweedie's formula and its consequences.
- **Karras, T., Aittala, M., Aila, T., and Laine, S.** (2022), "Elucidating the Design Space of Diffusion-Based Generative Models" -- thorough analysis of parameterization and preconditioning choices.
- **Luo, C.** (2022), "Understanding Diffusion Models: A Unified Perspective" -- Sections 7-9 cover denoising score matching clearly.
