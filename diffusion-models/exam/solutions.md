# CS 372: Diffusion Models — Final Exam Solutions

---

## Section 1: Foundations — Probability and Stochastic Processes (20 marks)

### Question 1.1

**(a)** [5 marks]

We prove $q(\mathbf{x}\_t | \mathbf{x}\_0) = \mathcal{N}(\mathbf{x}\_t; \sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0, (1 - \bar{\alpha}\_t)\mathbf{I})$ by induction on $t$.

**Base case ($t = 1$).** By definition, $q(\mathbf{x}\_1 | \mathbf{x}\_0) = \mathcal{N}(\mathbf{x}\_1; \sqrt{1 - \beta\_1}\,\mathbf{x}\_0, \beta\_1 \mathbf{I})$. Since $\alpha\_1 = 1 - \beta\_1$ and $\bar{\alpha}\_1 = \alpha\_1$, the mean is $\sqrt{\bar{\alpha}\_1}\,\mathbf{x}\_0$ and the variance is $1 - \bar{\alpha}\_1 = \beta\_1$. The formula holds.

**Inductive step.** Assume $q(\mathbf{x}\_{t-1} | \mathbf{x}\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_{t-1}}\,\mathbf{x}\_0, (1 - \bar{\alpha}\_{t-1})\mathbf{I})$. Using the reparameterization trick:

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\,\boldsymbol{\epsilon}_{t-1}, \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

The forward step gives:

$$
\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Substituting the expression for $\mathbf{x}\_{t-1}$:

$$
\mathbf{x}_t = \sqrt{\alpha_t}\left(\sqrt{\bar{\alpha}_{t-1}}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\,\boldsymbol{\epsilon}_{t-1}\right) + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_t
$$

$$
= \sqrt{\alpha_t \bar{\alpha}_{t-1}}\,\mathbf{x}_0 + \sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\,\boldsymbol{\epsilon}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_t
$$

Since $\boldsymbol{\epsilon}\_{t-1}$ and $\boldsymbol{\epsilon}\_t$ are independent standard Gaussians, the sum of the two noise terms is Gaussian with mean zero and variance:

$$
\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t = \alpha_t - \alpha_t \bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \alpha_t \bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t
$$

where we used $\beta\_t = 1 - \alpha\_t$ and $\alpha\_t \bar{\alpha}\_{t-1} = \bar{\alpha}\_t$. We can therefore combine the noise into a single Gaussian:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

This gives $q(\mathbf{x}\_t | \mathbf{x}\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0, (1 - \bar{\alpha}\_t)\mathbf{I})$. $\square$

> **Key insight:** The reparameterization trick lets us combine independent Gaussians at each step. The product structure of $\bar{\alpha}\_t$ arises because the signal is multiplicatively attenuated at each step, while the noise variances add.

**(b)** [3 marks]

As $T \to \infty$ with $\bar{\alpha}\_T \to 0$, we have:

$$
q(\mathbf{x}_T | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_T}\,\mathbf{x}_0, (1 - \bar{\alpha}_T)\mathbf{I}) \to \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

The mean $\sqrt{\bar{\alpha}\_T}\,\mathbf{x}\_0 \to \mathbf{0}$ and the variance $(1 - \bar{\alpha}\_T) \to 1$. The endpoint distribution becomes a standard isotropic Gaussian, independent of $\mathbf{x}\_0$.

**Why this is essential:** The generative model starts from $p(\mathbf{x}\_T) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and runs the reverse process. For this to produce valid samples, the forward process must actually reach this prior -- otherwise we are sampling from the wrong starting distribution. If $\bar{\alpha}\_T$ remained bounded away from zero, the endpoint $q(\mathbf{x}\_T | \mathbf{x}\_0)$ would retain information about $\mathbf{x}\_0$ (a nonzero mean depending on $\mathbf{x}\_0$). The prior $p(\mathbf{x}\_T) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ would then be a poor approximation to the true marginal $q(\mathbf{x}\_T)$, and the $L\_T$ term in the variational bound ($D\_{\text{KL}}(q(\mathbf{x}\_T|\mathbf{x}\_0) \Vert p(\mathbf{x}\_T))$) would be large, causing a mismatch between the forward and reverse processes and degraded sample quality.

---

### Question 1.2

**(a)** [2 marks]

The score function is $\nabla\_{\mathbf{x}} \log p(\mathbf{x})$, the gradient of the log-probability density with respect to $\mathbf{x}$. It is a vector field $\mathbb{R}^d \to \mathbb{R}^d$.

**Geometric interpretation:** At any point $\mathbf{x}$, the score vector points in the direction of steepest ascent of the log-density. It points toward nearby regions of higher probability -- intuitively, toward the nearest mode or high-density region of $p(\mathbf{x})$. The magnitude of the score indicates how rapidly the log-density is increasing in that direction: large near the boundaries of modes, zero at local maxima.

**(b)** [4 marks]

The Langevin dynamics iteration:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\eta}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_k) + \sqrt{\eta}\,\boldsymbol{\epsilon}_k
$$

has three terms with distinct roles:

1. **$\mathbf{x}\_k$ (current position):** The starting point for the update. The process builds upon the previous state, making this a Markov chain.

2. **$\frac{\eta}{2} \nabla\_{\mathbf{x}} \log p(\mathbf{x}\_k)$ (score-driven drift):** This deterministic term moves $\mathbf{x}$ toward regions of higher probability. It is a gradient ascent step on $\log p(\mathbf{x})$ with step size $\eta/2$. Without the other terms, this would be gradient ascent on the log-density, converging to a local mode.

3. **$\sqrt{\eta}\,\boldsymbol{\epsilon}\_k$ (stochastic noise):** Brownian noise injection with variance $\eta$ per dimension. This is the term that distinguishes sampling from optimization.

**Without the noise term,** the iteration becomes $\mathbf{x}\_{k+1} = \mathbf{x}\_k + \frac{\eta}{2} \nabla\_{\mathbf{x}} \log p(\mathbf{x}\_k)$, which is gradient ascent on $\log p$. This converges to a local mode (a point of maximum density) rather than sampling from the full distribution. It would always produce the same output for a given initialization.

**Why noise is necessary for sampling:** Sampling requires exploring the full support of $p(\mathbf{x})$, including regions between and around modes. The noise serves two purposes: (i) it allows the chain to escape local modes and explore the multimodal landscape, and (ii) it ensures the stationary distribution of the Markov chain is $p(\mathbf{x})$ rather than a point mass at a mode. The balance between drift (pulling toward high density) and noise (random exploration) is what makes the chain converge to sampling from $p(\mathbf{x})$ rather than optimizing it.

---

### Question 1.3

**(a)** [3 marks]

For the OU process $d\mathbf{x} = -\frac{1}{2}\beta(t)\,\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{w}$, comparing with the general SDE $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}$:

- **Drift coefficient:** $\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\,\mathbf{x}$
- **Diffusion coefficient:** $g(t) = \sqrt{\beta(t)}$

**Intuition for convergence to $\mathcal{N}(\mathbf{0}, \mathbf{I})$:** The drift $-\frac{1}{2}\beta(t)\,\mathbf{x}$ is a mean-reverting force that pulls $\mathbf{x}$ toward the origin. The farther $\mathbf{x}$ is from zero, the stronger the pull. Simultaneously, the diffusion term $\sqrt{\beta(t)}\,d\mathbf{w}$ injects isotropic noise. These two effects balance: the drift shrinks the signal while the noise adds variance. Over time, regardless of the initial distribution, the process converges to the equilibrium where the contraction and noise injection balance perfectly -- this equilibrium is $\mathcal{N}(\mathbf{0}, \mathbf{I})$. This is the VP-SDE (variance-preserving SDE) that corresponds to the DDPM forward process in continuous time.

**(b)** [3 marks]

Anderson's reverse-time SDE (1982) states that the reverse of $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}$ is:

$$
d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + g(t)\,d\bar{\mathbf{w}}
$$

where $d\bar{\mathbf{w}}$ is a reverse-time Wiener process and time runs backward from $T$ to $0$.

Substituting $\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\,\mathbf{x}$ and $g(t) = \sqrt{\beta(t)}$:

$$
\boxed{d\mathbf{x} = \left[-\frac{1}{2}\beta(t)\,\mathbf{x} - \beta(t)\,\nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + \sqrt{\beta(t)}\,d\bar{\mathbf{w}}}
$$

The reverse drift consists of two parts: the original drift $-\frac{1}{2}\beta(t)\mathbf{x}$ and a score-dependent correction $-\beta(t)\nabla\_{\mathbf{x}} \log p\_t(\mathbf{x})$ that steers the process toward high-density regions of the time-$t$ marginal.

---

## Section 2: Denoising, Score Matching, and DDPM (25 marks)

### Question 2.1

**(a)** [4 marks] — Proof of Tweedie's formula.

We have $\mathbf{x} = \boldsymbol{\mu} + \sigma\boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $\boldsymbol{\mu} \sim p(\boldsymbol{\mu})$. The marginal density is:

$$
p_\sigma(\mathbf{x}) = \int \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I})\, p(\boldsymbol{\mu})\, d\boldsymbol{\mu}
$$

**Step 1.** Compute the score $\nabla\_{\mathbf{x}} \log p\_\sigma(\mathbf{x}) = \frac{\nabla\_{\mathbf{x}} p\_\sigma(\mathbf{x})}{p\_\sigma(\mathbf{x})}$.

**Step 2.** Differentiate under the integral:

$$
\nabla_{\mathbf{x}} p_\sigma(\mathbf{x}) = \int p(\boldsymbol{\mu})\, \nabla_{\mathbf{x}} \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I})\, d\boldsymbol{\mu}
$$

The gradient of the Gaussian with respect to $\mathbf{x}$ is:

$$
\nabla_{\mathbf{x}} \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I}) \cdot \frac{\boldsymbol{\mu} - \mathbf{x}}{\sigma^2}
$$

Substituting:

$$
\nabla_{\mathbf{x}} p_\sigma(\mathbf{x}) = \frac{1}{\sigma^2} \int (\boldsymbol{\mu} - \mathbf{x})\, p(\boldsymbol{\mu})\, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I})\, d\boldsymbol{\mu}
$$

**Step 3.** Divide by $p\_\sigma(\mathbf{x})$:

$$
\nabla_{\mathbf{x}} \log p_\sigma(\mathbf{x}) = \frac{1}{\sigma^2} \cdot \frac{\int (\boldsymbol{\mu} - \mathbf{x})\, p(\boldsymbol{\mu})\, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I})\, d\boldsymbol{\mu}}{\int p(\boldsymbol{\mu})\, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I})\, d\boldsymbol{\mu}}
$$

The ratio is exactly $\mathbb{E}[\boldsymbol{\mu} - \mathbf{x} | \mathbf{x}] = \mathbb{E}[\boldsymbol{\mu} | \mathbf{x}] - \mathbf{x}$, since the posterior density of $\boldsymbol{\mu}$ given $\mathbf{x}$ is $p(\boldsymbol{\mu}|\mathbf{x}) = \frac{p(\boldsymbol{\mu})\,\mathcal{N}(\mathbf{x};\boldsymbol{\mu},\sigma^2\mathbf{I})}{p\_\sigma(\mathbf{x})}$.

Therefore:

$$
\nabla_{\mathbf{x}} \log p_\sigma(\mathbf{x}) = \frac{\mathbb{E}[\boldsymbol{\mu} | \mathbf{x}] - \mathbf{x}}{\sigma^2}
$$

Rearranging gives Tweedie's formula:

$$
\boxed{\mathbb{E}[\boldsymbol{\mu} | \mathbf{x}] = \mathbf{x} + \sigma^2 \nabla_{\mathbf{x}} \log p_\sigma(\mathbf{x})} \quad \square
$$

> **Key insight:** The score of the noisy distribution points from the noisy observation toward the posterior mean of the clean signal. This is the bridge between denoising and score estimation.

**(b)** [4 marks]

In the DDPM setting, $\mathbf{x}\_t = \sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_t}\,\boldsymbol{\epsilon}$. This is exactly the Tweedie setup with $\boldsymbol{\mu} = \sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0$ and $\sigma = \sqrt{1 - \bar{\alpha}\_t}$.

Applying Tweedie's formula:

$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) = \frac{\mathbb{E}[\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 | \mathbf{x}_t] - \mathbf{x}_t}{1 - \bar{\alpha}_t}
$$

We can also compute the score directly from the forward process. Since $\mathbf{x}\_t = \sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_t}\,\boldsymbol{\epsilon}$, rearranging gives $\boldsymbol{\epsilon} = \frac{\mathbf{x}\_t - \sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0}{\sqrt{1 - \bar{\alpha}\_t}}$. Taking the conditional score:

$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0}{1 - \bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}
$$

The marginal score satisfies:

$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) = -\frac{\mathbb{E}[\boldsymbol{\epsilon} | \mathbf{x}_t]}{\sqrt{1 - \bar{\alpha}_t}}
$$

This means a network $\boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t)$ trained to predict $\boldsymbol{\epsilon}$ is implicitly estimating the score:

$$
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \approx -\sqrt{1 - \bar{\alpha}_t}\,\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)
$$

Or equivalently:

$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) \approx -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

**Why $\boldsymbol{\epsilon}$-prediction equals score estimation:** The score points from $\mathbf{x}\_t$ toward the clean data $\mathbf{x}\_0$. The noise $\boldsymbol{\epsilon}$ points from the clean data toward $\mathbf{x}\_t$. They are opposite directions, related by a known scaling factor. Learning one is equivalent to learning the other.

---

### Question 2.2

**(a)** [3 marks]

We derive the forward process posterior $q(\mathbf{x}\_{t-1} | \mathbf{x}\_t, \mathbf{x}\_0)$ using Bayes' rule:

$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0)\, q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)}
$$

Since the forward process is Markov, $q(\mathbf{x}\_t | \mathbf{x}\_{t-1}, \mathbf{x}\_0) = q(\mathbf{x}\_t | \mathbf{x}\_{t-1})$. All three densities on the right are Gaussian:

- $q(\mathbf{x}\_t | \mathbf{x}\_{t-1}) = \mathcal{N}(\sqrt{\alpha\_t}\,\mathbf{x}\_{t-1}, \beta\_t \mathbf{I})$
- $q(\mathbf{x}\_{t-1} | \mathbf{x}\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_{t-1}}\,\mathbf{x}\_0, (1 - \bar{\alpha}\_{t-1})\mathbf{I})$
- $q(\mathbf{x}\_t | \mathbf{x}\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0, (1 - \bar{\alpha}\_t)\mathbf{I})$

The product and ratio of Gaussians is Gaussian. Collecting terms in the exponent that depend on $\mathbf{x}\_{t-1}$ and completing the square:

$$
\propto \exp\!\left(-\frac{1}{2}\left[\frac{(\mathbf{x}_t - \sqrt{\alpha_t}\,\mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\,\mathbf{x}_0)^2}{1 - \bar{\alpha}_{t-1}}\right]\right)
$$

Combining the quadratic terms in $\mathbf{x}\_{t-1}$, the precision (inverse variance) is:

$$
\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} = \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})} = \frac{1 - \bar{\alpha}_t}{\beta_t(1 - \bar{\alpha}_{t-1})}
$$

So the **posterior variance** is:

$$
\boxed{\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})\,\beta_t}{1 - \bar{\alpha}_t}}
$$

Reading off the mean from the linear terms:

$$
\boxed{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\,\mathbf{x}_t}
$$

This is a weighted combination of $\mathbf{x}\_0$ and $\mathbf{x}\_t$, with weights that depend on the noise schedule.

**(b)** [4 marks]

Both $q(\mathbf{x}\_{t-1} | \mathbf{x}\_t, \mathbf{x}\_0)$ and $p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t)$ are Gaussians with the same variance $\sigma\_t^2 = \tilde{\beta}\_t$. The KL divergence between two Gaussians with equal covariance simplifies to:

$$
D_{\text{KL}}(q \Vert p_\theta) = \frac{1}{2\sigma_t^2} \left\Vert \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\right\Vert ^2
$$

Now substitute $\mathbf{x}\_0 = \frac{1}{\sqrt{\bar{\alpha}\_t}}(\mathbf{x}\_t - \sqrt{1 - \bar{\alpha}\_t}\,\boldsymbol{\epsilon})$ into $\tilde{\boldsymbol{\mu}}\_t$:

$$
\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} \cdot \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}} + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\mathbf{x}_t
$$

Collecting the $\mathbf{x}\_t$ terms and simplifying (using $\bar{\alpha}\_t = \alpha\_t \bar{\alpha}\_{t-1}$):

$$
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}\right)
$$

The Ho et al. parameterization sets:

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)
$$

Substituting both into the KL:

$$
D_{\text{KL}} = \frac{1}{2\sigma_t^2} \left\Vert \frac{1}{\sqrt{\alpha_t}} \cdot \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\left(\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)\right\Vert ^2
$$

$$
= \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \left\Vert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right\Vert ^2
$$

The prefactor $w\_t = \frac{\beta\_t^2}{2\sigma\_t^2 \alpha\_t (1 - \bar{\alpha}\_t)}$ is a time-dependent weighting. Up to this constant, minimizing the KL reduces to minimizing $\Vert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t)\Vert ^2$. $\square$

> **Key insight:** The $\boldsymbol{\epsilon}$-prediction parameterization absorbs the complex structure of the posterior mean into a simple noise-prediction task. The network does not need to learn the noise schedule -- it only needs to identify what noise was added.

**(c)** [3 marks]

The VLB weights $w\_t = \frac{\beta\_t^2}{2\sigma\_t^2 \alpha\_t(1-\bar{\alpha}\_t)}$ are large at small $t$ (low noise levels) and small at large $t$ (high noise levels). This means the VLB objective heavily emphasizes getting fine details right at low noise, while giving little weight to the coarse global structure at high noise.

**Why equal weighting helps:** The global structure of an image (object layout, overall composition) is determined at high noise levels, where the model decides what to generate. Fine details (textures, edges) are resolved at low noise levels. If the model gets the global structure wrong at high $t$, no amount of detail refinement at low $t$ will recover -- the sample is fundamentally flawed.

By giving equal weight to all timesteps, $\mathcal{L}\_{\text{simple}}$ forces the model to allocate capacity across all noise scales, ensuring that both coarse structure (high $t$) and fine detail (low $t$) are learned well. This produces better sample quality (lower FID) even though it loosens the variational bound, because perceptual quality depends more on getting the coarse-to-fine hierarchy right than on optimizing a mathematical bound.

---

### Question 2.3

**(a)** [3 marks]

The quantity $-\frac{\tilde{\mathbf{x}} - \mathbf{x}\_0}{\sigma\_i^2}$ is the **score of the conditional (noise) distribution** $\nabla\_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}} | \mathbf{x}\_0)$, evaluated at noise level $\sigma\_i$.

Since $\tilde{\mathbf{x}} = \mathbf{x}\_0 + \sigma\_i \boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, the conditional distribution is $q(\tilde{\mathbf{x}} | \mathbf{x}\_0) = \mathcal{N}(\mathbf{x}\_0, \sigma\_i^2 \mathbf{I})$. Its score is:

$$
\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}} | \mathbf{x}_0) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}_0}{\sigma_i^2}
$$

By the denoising score matching theorem (Vincent, 2011), this conditional score is a valid target for learning the marginal score $\nabla\_{\tilde{\mathbf{x}}} \log q\_{\sigma\_i}(\tilde{\mathbf{x}})$. The two losses have the same gradients with respect to $\theta$. So the network $\mathbf{s}\_\theta$ is being trained to estimate the score of the noised data distribution at each noise level $\sigma\_i$.

**(b)** [4 marks]

In DDPM, the forward process gives $\mathbf{x}\_t = \sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_t}\,\boldsymbol{\epsilon}$. Setting $\sigma\_t = \sqrt{1 - \bar{\alpha}\_t}$, the conditional score is:

$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0}{1 - \bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}} = -\frac{\boldsymbol{\epsilon}}{\sigma_t}
$$

The DDPM noise predictor satisfies $\boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t) \approx \boldsymbol{\epsilon}$, and the NCSN score estimator satisfies $\mathbf{s}\_\theta(\mathbf{x}\_t, t) \approx -\boldsymbol{\epsilon}/\sigma\_t$. Therefore:

$$
\boxed{\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}}
$$

Or equivalently:

$$
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) = -\sqrt{1 - \bar{\alpha}_t}\,\mathbf{s}_\theta(\mathbf{x}_t, t)
$$

The two parameterizations differ by a known, time-dependent scaling factor $\sigma\_t = \sqrt{1 - \bar{\alpha}\_t}$. Predicting noise ($\boldsymbol{\epsilon}$-prediction in DDPM) and estimating the score ($\mathbf{s}$-prediction in NCSN) are the same task expressed in different units.

---

## Section 3: The SDE Framework and Samplers (20 marks)

### Question 3.1

**(a)** [3 marks]

The reverse-time SDE (Anderson, 1982) for the forward SDE $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}$ is:

$$
\boxed{d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + g(t)\,d\bar{\mathbf{w}}}
$$

where $d\bar{\mathbf{w}}$ is a reverse-time Wiener process and time runs backward from $T$ to $0$.

**Why the score suffices:** The reverse drift has two components: the original drift $\mathbf{f}(\mathbf{x}, t)$ (which is known by construction) and the score correction $-g(t)^2 \nabla\_{\mathbf{x}} \log p\_t(\mathbf{x})$. The diffusion coefficient $g(t)$ is also known. The only unknown quantity is the score $\nabla\_{\mathbf{x}} \log p\_t(\mathbf{x})$, which encodes the structure of the time-dependent marginal distribution. Once we have the score at all times (via a trained score network $\mathbf{s}\_\theta$), the entire reverse process is fully determined -- we can solve it numerically to generate samples.

**(b)** [5 marks]

**Why the ODE has the same marginals:** The marginal densities $p\_t(\mathbf{x})$ of the forward SDE evolve according to the Fokker-Planck equation:

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot \left[\mathbf{f}(\mathbf{x},t)\, p_t\right] + \frac{1}{2}g(t)^2 \Delta p_t
$$

The probability flow ODE has velocity field $\tilde{\mathbf{f}}(\mathbf{x},t) = \mathbf{f}(\mathbf{x},t) - \frac{1}{2}g(t)^2 \nabla\_{\mathbf{x}} \log p\_t(\mathbf{x})$. For a deterministic ODE $d\mathbf{x} = \tilde{\mathbf{f}}\,dt$ (no diffusion), the density evolves by the continuity equation:

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot \left[\tilde{\mathbf{f}}\, p_t\right]
$$

Using the identity $\frac{1}{2}g^2 \Delta p\_t = -\nabla \cdot \left[-\frac{1}{2}g^2 (\nabla \log p\_t) \cdot p\_t\right]$ (which follows from $\nabla p\_t = p\_t \nabla \log p\_t$), we can rewrite the Fokker-Planck equation as:

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot \left[\left(\mathbf{f} - \frac{1}{2}g^2 \nabla \log p_t\right) p_t\right] = -\nabla \cdot \left[\tilde{\mathbf{f}}\, p_t\right]
$$

This is identical to the continuity equation for the ODE. Since both equations describe the evolution of the same density $p\_t$, the ODE and SDE share the same marginal distributions at all times.

**Practical advantage of the ODE:**
- **Deterministic sampling:** Given $\mathbf{x}\_T$, the trajectory is unique. This enables exact inversion (encoding data to latent space), latent space interpolation, and reproducible generation.
- **Exact likelihood computation:** The ODE is a continuous normalizing flow, so the instantaneous change of variables formula gives exact log-likelihoods.
- **Adaptive ODE solvers:** Standard adaptive-step solvers (RK45, Dormand-Prince) can concentrate computation where the dynamics change most, typically requiring 20-50 function evaluations for high-quality samples.

**Potential disadvantage:**
- **Sample diversity and quality at few steps.** The stochastic reverse SDE injects noise at each step, which can help correct errors from imperfect score estimates and encourage the trajectory to explore modes of the distribution. The deterministic ODE has no such correction mechanism -- errors accumulate along the trajectory without any stochastic reset. In practice, the reverse SDE with predictor-corrector methods can achieve slightly better FID scores than the ODE, at the cost of more function evaluations.

---

### Question 3.2

**(a)** [4 marks]

The DDIM update from $\mathbf{x}\_t$ to $\mathbf{x}\_{t-1}$ is:

$$
\mathbf{x}_{t-1} = \underbrace{\sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{Term 1: predicted } \mathbf{x}_0 \text{, rescaled}} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}_{\text{Term 2: direction toward } \mathbf{x}_t} + \underbrace{\sigma_t \boldsymbol{\epsilon}}_{\text{Term 3: random noise}}
$$

**The predicted $\mathbf{x}\_0$** is:

$$
\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

This is obtained by rearranging the forward process equation $\mathbf{x}\_t = \sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0 + \sqrt{1-\bar{\alpha}\_t}\,\boldsymbol{\epsilon}$, substituting the network's noise prediction for $\boldsymbol{\epsilon}$.

**Interpretation of each term:**

1. **$\sqrt{\bar{\alpha}\_{t-1}}\,\hat{\mathbf{x}}\_0$:** The predicted clean image, rescaled to the signal level appropriate for time $t-1$. This is the "data component" of $\mathbf{x}\_{t-1}$.

2. **$\sqrt{1-\bar{\alpha}\_{t-1} - \sigma\_t^2}\,\boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t)$:** The "noise direction" component. It re-introduces a deterministic amount of noise in the direction predicted by the model, ensuring $\mathbf{x}\_{t-1}$ has the correct noise structure for time $t-1$. The coefficient is chosen so that the total variance budget is correct.

3. **$\sigma\_t \boldsymbol{\epsilon}$:** Fresh random noise ($\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$). This controls the stochasticity of the process. When $\sigma\_t = 0$, the update is deterministic; when $\sigma\_t$ matches the DDPM posterior variance, the update recovers standard DDPM sampling.

**(b)** [3 marks]

When $\sigma\_t = 0$, the DDIM update is fully deterministic: $\mathbf{x}\_{t-1}$ is a deterministic function of $\mathbf{x}\_t$ and the predicted $\hat{\mathbf{x}}\_0$. The update depends only on the marginals $q(\mathbf{x}\_t | \mathbf{x}\_0)$ (through $\bar{\alpha}\_t$ and $\bar{\alpha}\_{t-1}$), not on the step-by-step Markov transitions.

**Why DDIM can skip timesteps:** The DDIM formula uses $\bar{\alpha}\_t$ and $\bar{\alpha}\_{t-1}$ directly, jumping between any two noise levels regardless of the intermediate steps. If we want to go from $t = 1000$ to $t = 900$, we simply plug in $\bar{\alpha}\_{1000}$ and $\bar{\alpha}\_{900}$. The predicted $\hat{\mathbf{x}}\_0$ provides a "global" estimate of the clean image from any noise level, and the formula reconstructs the correct $\mathbf{x}\_{900}$ from this estimate. The quality of the $\hat{\mathbf{x}}\_0$ prediction is what limits accuracy, not the step size.

**Why DDPM cannot skip steps:** DDPM's reverse process is a Markov chain: $p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t)$ is derived from $q(\mathbf{x}\_{t-1} | \mathbf{x}\_t, \mathbf{x}\_0)$, which depends on $q(\mathbf{x}\_t | \mathbf{x}\_{t-1})$ being a small Gaussian perturbation. The Gaussian approximation to the true reverse transition is only accurate when the forward step is small (small $\beta\_t$). If we try to skip from $t = 1000$ to $t = 900$ in one DDPM step, the forward transition $q(\mathbf{x}\_{900} | \mathbf{x}\_{1000})$ involves a large step, and the reverse transition is no longer well-approximated by a Gaussian -- it becomes multimodal. The Markov property means each step can only use local information from the immediately adjacent timestep, not the global $\hat{\mathbf{x}}\_0$ estimate that DDIM exploits.

---

### Question 3.3

**(a)** [2 marks]

| Criterion | Euler-Maruyama (Reverse SDE) | Probability Flow ODE |
|-----------|------------------------------|---------------------|
| **(i) Stochasticity** | Stochastic: injects fresh noise at each step, so different random seeds yield different samples from the same $\mathbf{x}\_T$. | Deterministic: given $\mathbf{x}\_T$, the trajectory and output are uniquely determined. |
| **(ii) NFE for good quality** | Typically requires more NFE (50-1000) because the stochastic noise introduces variance that must be averaged out over many small steps. However, predictor-corrector variants can improve efficiency. | Requires fewer NFE (10-50) because adaptive ODE solvers and higher-order methods (Heun, DPM-Solver) can take larger, more accurate steps. The deterministic trajectory is smoother and more amenable to acceleration. |

**(b)** [3 marks]

DPM-Solver exploits the observation that the probability flow ODE, when reparameterized in log-SNR time $\lambda = \log(\bar{\alpha}\_t / (1 - \bar{\alpha}\_t))$, has a **semi-linear** structure:

$$
\frac{d\mathbf{x}}{d\lambda} = \underbrace{\frac{1}{2}\mathbf{x}}_{\text{linear part}} - \underbrace{\frac{1}{2}e^{-\lambda}\,\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, \lambda)}_{\text{nonlinear part}}
$$

**The linear part** ($\frac{1}{2}\mathbf{x}$) is a simple scaling that has a known exact solution: $\mathbf{x}(\lambda\_s) = e^{(\lambda\_s - \lambda\_t)/2}\,\mathbf{x}(\lambda\_t)$. DPM-Solver handles this analytically using the variation of constants formula.

**The nonlinear part** involves the neural network $\hat{\boldsymbol{\epsilon}}\_\theta$, which is the only source of approximation error. DPM-Solver approximates only the integral of this nonlinear term using polynomial interpolation (constant for 1st order, linear for 2nd order, etc.), while the exponential linear part is computed exactly.

The practical payoff: because the dominant linear dynamics are handled analytically, DPM-Solver achieves much better accuracy per step than generic ODE solvers (Euler, Heun) that must approximate both the linear and nonlinear parts numerically. This is why DPM-Solver can produce high-quality samples in 10-20 NFE.

---

## Section 4: Architecture, Conditioning, and Guidance (15 marks)

### Question 4.1

**(a)** [3 marks]

Two concrete advantages of running diffusion in latent space rather than pixel space:

1. **Computational efficiency.** A 512x512x3 image has 786,432 dimensions. The VAE compresses this to a 64x64x4 latent (16,384 dimensions) -- a 48x reduction. Since the denoising U-Net processes the latent rather than the image, both training and inference are dramatically cheaper: fewer FLOPs per forward pass, less memory, and faster wall-clock time. This is what made Stable Diffusion runnable on consumer GPUs.

2. **Removal of perceptually irrelevant detail.** Natural images contain high-frequency pixel-level details (sensor noise, imperceptible textures, exact subpixel positioning) that carry little semantic information. The VAE's encoder learns to discard this information, so the diffusion model operates on a space that captures only perceptually meaningful variation. The model does not waste capacity learning to generate details that humans cannot distinguish, focusing instead on semantically important structure.

**(b)** [3 marks]

The cross-attention operation is:

$$
\text{CrossAttn}(\mathbf{h}, \mathbf{c}) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

where:

- $Q = W\_Q \mathbf{h}$ -- **Queries** come from the noisy image features (the U-Net's intermediate spatial feature map $\mathbf{h} \in \mathbb{R}^{n \times d\_h}$, where $n$ is the number of spatial positions)
- $K = W\_K \mathbf{c}$ -- **Keys** come from the conditioning signal (e.g., text embeddings $\mathbf{c} \in \mathbb{R}^{L \times d\_c}$, where $L$ is the sequence length)
- $V = W\_V \mathbf{c}$ -- **Values** also come from the conditioning signal

Each spatial position in the image features independently attends to all conditioning tokens. The attention weights $\text{softmax}(QK^\top/\sqrt{d\_k})$ determine which conditioning tokens are relevant to each spatial location, and the output is a conditioning-weighted combination of the value vectors. This allows different parts of the image to "read" different parts of the text prompt.

---

### Question 4.2

**(a)** [3 marks]

Starting from Bayes' rule:

$$
p(\mathbf{x}_t | y) = \frac{p(y | \mathbf{x}_t)\, p(\mathbf{x}_t)}{p(y)}
$$

Taking the log and the gradient with respect to $\mathbf{x}\_t$:

$$
\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | y) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t)
$$

(The $\log p(y)$ term vanishes because it does not depend on $\mathbf{x}\_t$.) This gives the $\gamma = 1$ case:

$$
\tilde{\nabla}_{\mathbf{x}_t} \log p(\mathbf{x}_t | y) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p_\phi(y | \mathbf{x}_t)
$$

**Why $\gamma > 1$ is used:** Setting $\gamma = 1$ gives the "correct" conditional score, but in practice the classifier $p\_\phi(y|\mathbf{x}\_t)$ is imperfect and the conditioning effect is weak. Using $\gamma > 1$ amplifies the classifier gradient, effectively sampling from a sharpened distribution $\tilde{p}(\mathbf{x}\_t | y) \propto p(\mathbf{x}\_t) \cdot p(y|\mathbf{x}\_t)^\gamma$. This concentrates samples on images that the classifier confidently identifies as class $y$, improving class-conditional fidelity (IS, classification accuracy) at the cost of sample diversity. Dhariwal and Nichol found that $\gamma \approx 2$-$4$ gives the best quality-diversity tradeoff.

**(b)(i)** [3 marks]

Using the relationship $\boldsymbol{\epsilon}\_\theta \approx -\sigma\_t \nabla\_{\mathbf{x}\_t} \log p\_t(\mathbf{x}\_t)$ (from Section 2), we can convert between noise predictions and scores:

$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \approx -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)}{\sigma_t}
$$

$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{c}) \approx -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c})}{\sigma_t}
$$

The implicit classifier gradient is the difference:

$$
\nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{c}) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)
$$

$$
\approx -\frac{1}{\sigma_t}\left[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)\right]
$$

So the difference $\boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, \mathbf{c}) - \boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, \varnothing)$ is proportional to $-\sigma\_t \nabla\_{\mathbf{x}\_t} \log p(y|\mathbf{x}\_t)$ -- the implicit classifier gradient.

Now substitute into the classifier guidance formula. With guidance scale $s$:

$$
\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing) - s \cdot \sigma_t \cdot \nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t)
$$

$$
= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing) + s \left[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)\right]
$$

This is exactly the classifier-free guidance formula with $w = s + 1$ (or equivalently $w = 1$ maps to $s = 0$, no guidance). Classifier-free guidance with weight $w$ is equivalent to classifier guidance with scale $s = w - 1$, where the classifier is implicitly defined by the difference between conditional and unconditional noise predictions.

**(b)(ii)** [3 marks]

As the guidance scale $w$ increases:

**Sample quality (FID):** Initially improves, then degrades. At moderate $w$ (e.g., $w = 5$-$8$), FID decreases because samples become more coherent and condition-adherent. At very high $w$ (e.g., $w > 15$), FID increases because samples become oversaturated, stereotypical, and exhibit artifacts -- they no longer look like natural images.

**Sample diversity:** Monotonically decreases. Higher $w$ concentrates the sampling distribution on a narrower set of high-likelihood images. The model converges on a few "canonical" representations of the condition, reducing variety.

**Mechanism:** Classifier-free guidance with scale $w$ effectively samples from $\tilde{p}(\mathbf{x}|\mathbf{c}) \propto p(\mathbf{x}) \cdot p(\mathbf{c}|\mathbf{x})^{w}$. Raising the implicit likelihood to a power $w > 1$ sharpens the conditional distribution -- it amplifies the modes while suppressing the tails. This is analogous to lowering the temperature in a Boltzmann distribution. The result is higher probability of condition-typical images (better fidelity) but reduced coverage of the full conditional distribution (less diversity). At extreme values, the guided noise prediction can move outside the expected range, causing color oversaturation and geometric distortion.

---

## Section 5: Modern Directions (20 marks)

### Question 5.1

**(a)** [3 marks]

The conditional probability path is $\mathbf{x}\_t = (1 - t)\boldsymbol{\epsilon} + t\,\mathbf{x}\_1$. The conditional velocity field is the time derivative of this path:

$$
\mathbf{v}_t(\mathbf{x}_t | \mathbf{x}_1) = \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \boldsymbol{\epsilon}
$$

This is remarkably simple: the velocity is constant in time, pointing from the noise sample $\boldsymbol{\epsilon}$ to the data point $\mathbf{x}\_1$.

**Comparison to DDPM's paths:** DDPM's forward process defines $\mathbf{x}\_t = \sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_t}\,\boldsymbol{\epsilon}$, where the signal coefficient $\sqrt{\bar{\alpha}\_t}$ and noise coefficient $\sqrt{1 - \bar{\alpha}\_t}$ change nonlinearly with $t$. These paths are **curved** in the data-noise space because the signal-to-noise ratio changes nonlinearly. In contrast, the flow matching path $\mathbf{x}\_t = (1-t)\boldsymbol{\epsilon} + t\,\mathbf{x}\_1$ is a **straight line** -- the signal and noise mix linearly. Straighter paths are easier for ODE solvers to follow accurately, requiring fewer discretization steps.

**(b)** [5 marks]

Comparison along three dimensions:

**(i) What is being predicted:**
- **DDPM** predicts the noise $\boldsymbol{\epsilon}$ that was added to the data. The target is $\boldsymbol{\epsilon}$, the specific noise realization used to corrupt $\mathbf{x}\_0$.
- **Flow matching** predicts the velocity $\mathbf{v}\_t = \mathbf{x}\_1 - \boldsymbol{\epsilon}$ -- the direction of motion from noise to data. The target is the displacement vector between the paired noise and data samples.

These are related: if $\mathbf{x}\_t = (1-t)\boldsymbol{\epsilon} + t\mathbf{x}\_1$, then $\mathbf{v}\_t = \mathbf{x}\_1 - \boldsymbol{\epsilon}$ can be rewritten in terms of $\mathbf{x}\_1$ and $\boldsymbol{\epsilon}$. DDPM predicts one component of this displacement (the noise); flow matching predicts the full displacement.

**(ii) Geometry of the transport paths:**
- **DDPM** uses curved paths determined by the noise schedule: $\mathbf{x}\_t = \sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0 + \sqrt{1-\bar{\alpha}\_t}\,\boldsymbol{\epsilon}$. The coefficients $\sqrt{\bar{\alpha}\_t}$ and $\sqrt{1-\bar{\alpha}\_t}$ trace a quarter-circle in signal-noise space. Navigating these curved paths accurately requires more ODE solver steps.
- **Flow matching** uses straight-line paths: $\mathbf{x}\_t = (1-t)\boldsymbol{\epsilon} + t\mathbf{x}\_1$. The interpolation is linear, giving the most direct route from noise to data. Even a crude Euler solver performs well on straight paths.

**(iii) One practical advantage of flow matching:**
- **No noise schedule to tune.** DDPM requires specifying $\beta\_1, \ldots, \beta\_T$ (or equivalently $\bar{\alpha}\_t$), which significantly affects training stability and sample quality. The literature on noise schedule design (linear, cosine, learned, shifted-cosine) is substantial. Flow matching with linear interpolation eliminates this design choice entirely -- the interpolation $\mathbf{x}\_t = (1-t)\boldsymbol{\epsilon} + t\mathbf{x}\_1$ is fully specified. This makes flow matching simpler to implement and more robust to hyperparameter choices.

---

### Question 5.2

**(a)** [3 marks]

**Geometric meaning of self-consistency:** Consider the probability flow ODE trajectory that starts from some noisy point $\mathbf{x}\_T$ and evolves to a clean data point $\mathbf{x}\_{\epsilon}$ (at some small $\epsilon > 0$). Every point $(\mathbf{x}\_t, t)$ along this trajectory maps to the *same* endpoint $\mathbf{x}\_\epsilon$ when the ODE is solved to completion. The consistency function $f\_\theta(\mathbf{x}\_t, t)$ directly outputs this endpoint without actually solving the ODE.

Self-consistency means: if you pick any two points on the same ODE trajectory -- say $(\mathbf{x}\_t, t)$ and $(\mathbf{x}\_{t'}, t')$ -- they must map to the same output: $f\_\theta(\mathbf{x}\_t, t) = f\_\theta(\mathbf{x}\_{t'}, t')$. Geometrically, $f\_\theta$ collapses each ODE trajectory to a single point (its origin).

**Why this enables single-step generation:** Once $f\_\theta$ is learned, generating a sample requires only: (1) sample $\mathbf{x}\_T \sim \mathcal{N}(\mathbf{0}, T^2\mathbf{I})$, and (2) compute $f\_\theta(\mathbf{x}\_T, T)$. This single function evaluation jumps directly to the trajectory's endpoint -- the clean data point -- bypassing the iterative ODE solve entirely. The multi-step process is "compiled" into a single forward pass.

**(b)** [4 marks]

**(i) Why the EMA target $\theta^-$ is needed:**

If we used $\theta$ directly in both sides of the loss $d(f\_\theta(\mathbf{x}\_{t+\Delta t}, t+\Delta t), f\_\theta(\hat{\mathbf{x}}\_t, t))$, the training would be unstable -- both the prediction and the target move simultaneously as $\theta$ updates, creating a "moving target" problem (similar to the instability in naive Q-learning). The network could satisfy the self-consistency constraint trivially by mapping everything to a constant, collapsing to a degenerate solution.

Using an EMA target $\theta^- = \text{stopgrad}(\text{EMA}(\theta))$ stabilizes training: the target $f\_{\theta^-}$ changes slowly (it is a smoothed, delayed version of $\theta$), giving the online network $f\_\theta$ a nearly fixed target to match at each step. This is the same stabilization technique used in BYOL, DQN, and momentum-based self-supervised learning. The gradients only flow through $f\_\theta$, not through $f\_{\theta^-}$.

**(ii) The role of $\Delta t$ and what happens as $\Delta t \to 0$:**

$\Delta t$ controls how far apart the two points $\mathbf{x}\_{t+\Delta t}$ and $\hat{\mathbf{x}}\_t$ are along the ODE trajectory. The pretrained model provides one ODE step: $\hat{\mathbf{x}}\_t = \text{ODE-step}(\mathbf{x}\_{t+\Delta t}, t+\Delta t \to t)$ using the teacher's score network.

- **Large $\Delta t$:** The two points are far apart on the trajectory. The ODE step is less accurate (larger discretization error), so $\hat{\mathbf{x}}\_t$ is a noisy estimate of the true trajectory point. The self-consistency constraint is enforced over large intervals, which is a coarser but noisier training signal.

- **Small $\Delta t$:** The two points are close together. The ODE step is highly accurate, so we are enforcing self-consistency on nearly adjacent trajectory points. The training signal is more precise but provides less information per pair (the two inputs are almost identical).

- **As $\Delta t \to 0$:** The consistency distillation loss converges to enforcing an infinitesimal self-consistency condition -- that the output of $f\_\theta$ does not change along the ODE's velocity field. In the limit, this becomes a differential constraint: $\frac{\partial f\_\theta}{\partial t} + \nabla\_{\mathbf{x}} f\_\theta \cdot v(x, t) = 0$, where $v$ is the ODE velocity. This exactly characterizes the consistency function. In practice, a curriculum is used: start with larger $\Delta t$ (easier, coarser constraint) and gradually decrease it (harder, finer constraint) during training.

---

### Question 5.3

**(a)** [3 marks]

Two modifications for temporal consistency in video diffusion:

**1. Temporal attention layers.** Video diffusion models insert temporal attention layers that attend across frames at the same spatial position. Given features $\mathbf{h} \in \mathbb{R}^{T \times H \times W \times C}$ (frames x height x width x channels), temporal attention computes attention over the $T$ dimension for each spatial location $(h, w)$. This allows the model to learn frame-to-frame correspondences and enforce consistency of objects, colors, and motion across time.

Why naive frame-by-frame fails: An image diffusion model applied independently to each frame has no mechanism to share information across frames. Each frame is generated from independent noise, so objects may appear, disappear, or change appearance between frames, and there is no coherent motion. Temporal attention provides the cross-frame communication needed for consistent object persistence and smooth motion.

**2. Temporal convolutions (3D convolutions or causal convolutions).** In addition to the spatial convolutions within each frame, video U-Nets add convolution kernels that span the temporal dimension, so each output depends on features from neighboring frames. This directly encodes the prior that adjacent frames should be similar, providing low-level temporal smoothness.

Why naive frame-by-frame fails: Without temporal convolutions, the model's convolutional filters only see within a single frame. There is no inductive bias for temporal smoothness -- high-frequency flickering between frames is not penalized. Temporal convolutions enforce local temporal coherence at the feature level, ensuring that the low-level texture and structure vary smoothly across frames.

**(b)** [2 marks]

One approach to discrete diffusion is the **absorbing state (mask-based) forward process** (D3PM, Austin et al. 2021). Each token independently transitions to a special [MASK] token with probability $\beta\_t$ at each step. The fully corrupted state is a sequence of all [MASK] tokens. The reverse process learns to "fill in" the masks, predicting the original token from context.

The forward process is defined by transition matrices $Q\_t$ over the discrete vocabulary:

$$
q(x_t = j | x_{t-1} = i) = [Q_t]_{ij}
$$

where $[Q\_t]\_{ij} = 1 - \beta\_t$ if $i = j \neq \text{[MASK]}$, $\beta\_t$ if $j = \text{[MASK]}$ and $i \neq \text{[MASK]}$, and $1$ if $i = j = \text{[MASK]}$.

**What plays the role of the score function:** The discrete analogue of the score is the **concrete score** or **ratio of transition probabilities**: $\frac{q(x\_{t-1} | x\_t, x\_0)}{q(x\_{t-1} | x\_t)}$, which can be expressed in terms of the model's predicted token probabilities $p\_\theta(x\_0 | x\_t, t)$. In practice, the model outputs a probability distribution over the vocabulary at each masked position -- essentially predicting "which token was here before it was masked?" This is structurally identical to a masked language model (BERT), and indeed MDLM (Sahoo et al., 2024) showed that masked language model training is equivalent to a specific parameterization of absorbing-state discrete diffusion.

---

## Section 6: Code Reading and Synthesis (20 marks)

### Question 6.1

**(a)** [4 marks] — Two bugs.

**Bug 1: Shape/broadcasting issue with `sqrt_alpha_bar` and `sqrt_one_minus_alpha_bar`.**

The lines:

```python
sqrt_alpha_bar = alphas_cumprod[t].sqrt()
sqrt_one_minus_alpha_bar = (1 - alphas_cumprod[t]).sqrt()
```

Here `t` has shape `(B,)`, so `alphas_cumprod[t]` has shape `(B,)`. When multiplied with `x_0` (shape `(B, C, H, W)`), broadcasting will be incorrect -- a 1D tensor of shape `(B,)` cannot properly broadcast with a 4D tensor. The scalars need to be reshaped to `(B, 1, 1, 1)` for correct element-wise multiplication across spatial and channel dimensions.

**Fix:**

```python
sqrt_alpha_bar = alphas_cumprod[t].sqrt().view(B, 1, 1, 1)
sqrt_one_minus_alpha_bar = (1 - alphas_cumprod[t]).sqrt().view(B, 1, 1, 1)
```

Without this fix, PyTorch will either raise an error or broadcast incorrectly depending on the batch size and spatial dimensions, applying the wrong noise level to different spatial positions.

**Bug 2: Loss function reduces over all dimensions including batch, giving an incorrect gradient scale.**

The line:

```python
loss = nn.MSELoss()(epsilon_pred, epsilon)
```

`nn.MSELoss()` uses `reduction='mean'` by default, which computes the mean over *all* elements (batch, channels, height, width). This means the gradient magnitude depends on the spatial dimensions $C \times H \times W$ -- specifically, it divides by the total number of elements rather than just the batch size. The correct behavior for DDPM training is to compute the mean over the batch but the **sum** (or mean) over spatial dimensions within each sample, so that the loss per sample has consistent magnitude regardless of image resolution.

**Fix:** Replace with:

```python
loss = nn.functional.mse_loss(epsilon_pred, epsilon, reduction='none')
loss = loss.mean(dim=[1, 2, 3]).mean()  # mean over spatial dims per sample, then mean over batch
```

Alternatively, `loss = ((epsilon_pred - epsilon) ** 2).mean()` technically works and is used by many implementations (since the default `mean` over all dimensions is equivalent to a rescaled version of the correct objective). However, the more principled approach is to sum over spatial dimensions per sample, which makes the loss magnitude independent of resolution. The hint says "subtle problem with the loss function," suggesting this reduction behavior.

> **Note:** A reasonable alternative answer for Bug 2 is that `optimizer.zero_grad()` should be called *before* the forward pass rather than before `loss.backward()`. However, in the given code it is called between `loss.backward()` and `optimizer.step()`, meaning gradients from the previous iteration's `loss.backward()` are zeroed *after* already being used by `optimizer.step()`. Actually, re-reading the code: `zero_grad()` is called, then `backward()`, then `step()` -- this ordering is correct. The loss function issue is the intended subtle bug.

**(b)** [3 marks]

For $\mathbf{x}\_0$-prediction, the model directly predicts the clean image:

```python
# Compute x_t (unchanged)
x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon

# Predict x_0 directly (was: epsilon_pred = model(x_t, t))
x_0_pred = model(x_t, t)

# Loss is between predicted x_0 and true x_0 (was: loss on epsilon)
loss = nn.functional.mse_loss(x_0_pred, x_0, reduction='none').mean(dim=[1,2,3]).mean()
```

**One advantage of $\mathbf{x}\_0$-prediction:** The predicted $\hat{\mathbf{x}}\_0$ is directly interpretable as an image estimate, which is useful for DDIM-style sampling (the "predicted $\mathbf{x}\_0$" appears explicitly in the update rule) and for visualization during training. DPM-Solver++ also benefits from the $\mathbf{x}\_0$ parameterization, which is more numerically stable at high noise levels.

**One disadvantage of $\mathbf{x}\_0$-prediction:** At high noise levels (large $t$, $\bar{\alpha}\_t \approx 0$), the input $\mathbf{x}\_t$ is almost pure noise and contains very little information about $\mathbf{x}\_0$. The target $\mathbf{x}\_0$ is essentially unpredictable from $\mathbf{x}\_t$, so the loss has very high variance and the gradients are noisy. In contrast, the noise $\boldsymbol{\epsilon}$ is always a standard Gaussian regardless of $t$, making $\boldsymbol{\epsilon}$-prediction a better-conditioned regression target. This is why Ho et al. found $\boldsymbol{\epsilon}$-prediction trains more stably.

**(c)** [3 marks]

Two methods to reduce generation time:

**1. DDIM (Denoising Diffusion Implicit Models, Song et al. 2020).** DDIM defines a non-Markovian reverse process that shares the same marginals as DDPM but allows skipping timesteps. With $\sigma\_t = 0$, the sampling becomes deterministic and equivalent to solving the probability flow ODE. The mechanism is the "predicted $\mathbf{x}\_0$" reparameterization: at each step, the model estimates the clean image $\hat{\mathbf{x}}\_0$ and uses it to jump directly to any desired noise level, bypassing intermediate steps. This reduces 1000 steps to 20-50 steps with minimal quality loss.

**2. DPM-Solver (Lu et al. 2022).** DPM-Solver exploits the semi-linear structure of the probability flow ODE. It solves the linear part (signal scaling) analytically and approximates only the nonlinear part (the neural network prediction) with higher-order polynomial methods. DPM-Solver-2 achieves 2nd-order accuracy with 2 NFE per step, allowing high-quality generation in 10-20 total NFE -- about 1-2 seconds instead of 45 seconds.

---

### Question 6.2

**(a)** [2 marks] **DDPM $\to$ Score SDE framework**

**Limitation:** The discrete Markov chain formulation of DDPM treats time as discrete ($t = 1, \ldots, T$) with a fixed step count $T$ and a specific noise schedule $\beta\_1, \ldots, \beta\_T$. This obscures the continuous-time structure: the forward process, reverse process, and their relationship to score functions are entangled with the discretization details. Changing $T$ or the schedule requires re-deriving the reverse process.

**Key insight:** Song et al. (2021) showed that DDPM and NCSN are both Euler-Maruyama discretizations of continuous-time SDEs (VP-SDE and VE-SDE respectively). The SDE perspective reveals that: (i) the two models are special cases of a general framework, (ii) the reverse process is determined by Anderson's reverse-time SDE, which depends only on the score function $\nabla\_{\mathbf{x}} \log p\_t(\mathbf{x})$, and (iii) there exists a deterministic counterpart -- the probability flow ODE -- sharing the same marginals. The continuous-time view separates the forward process design from the reverse process solver, enabling new sampler designs.

**(b)** [2 marks] **Score SDE $\to$ Probability flow ODE**

**Limitation:** The reverse SDE is stochastic: it requires injecting noise at every step, which introduces variance and demands many small steps for accuracy. Different random seeds produce different samples from the same starting point, precluding deterministic encoding and exact likelihood computation.

**Key insight:** The probability flow ODE $d\mathbf{x} = [\mathbf{f} - \frac{1}{2}g^2 \nabla \log p\_t]\,dt$ has the same marginal distributions as the SDE but is deterministic. This unlocks: (1) exact log-likelihood computation via the instantaneous change of variables formula (a continuous normalizing flow), (2) deterministic encoding -- a bijective map from data to latent space, enabling interpolation and inversion, and (3) the use of adaptive ODE solvers, which can take larger steps where the dynamics are smooth, reducing NFE.

**(c)** [2 marks] **Probability flow ODE $\to$ DDIM**

**Limitation:** While the probability flow ODE enables efficient sampling in principle, generic ODE solvers (Euler, RK45) treat the velocity field as a black box and do not exploit the specific structure of diffusion models. The VP-SDE's ODE still uses the DDPM noise schedule, which creates curved trajectories in signal-noise space that require many steps to follow.

**Key insight:** DDIM realizes that sampling depends only on the marginals $q(\mathbf{x}\_t | \mathbf{x}\_0)$ (parameterized by $\bar{\alpha}\_t$), not on the Markov transitions. It introduces the "predicted $\mathbf{x}\_0$" reparameterization, which enables arbitrary timestep skipping -- the model estimates $\hat{\mathbf{x}}\_0$ at each step and reconstructs $\mathbf{x}\_{t-1}$ directly. The practical payoff is reducing 1000 steps to 50 with $\eta = 0$ (deterministic), or to 20 with higher-order solvers. DDIM can be understood as a specific discretization of the probability flow ODE with the predicted-$\mathbf{x}\_0$ substitution.

**(d)** [2 marks] **DDIM/ODE solvers $\to$ Flow matching**

**Limitation:** All previous methods inherit the diffusion model's forward process: a stochastic noise injection with a specific schedule $\beta(t)$ that creates curved paths in signal-noise space. The noise schedule is a critical hyperparameter that must be carefully tuned (linear, cosine, shifted-cosine, etc.), and the curved paths require multi-step solvers even with the best ODE methods. The framework is conceptually heavy -- it requires SDE theory, score functions, and careful derivations.

**Key insight:** Flow matching (Lipman et al., 2023) asks a simpler question: learn a velocity field that transports noise to data. The conditional flow matching loss trains $\mathbf{v}\_\theta(\mathbf{x}\_t, t)$ to predict $\mathbf{x}\_1 - \mathbf{x}\_0$ given the interpolation $\mathbf{x}\_t = (1-t)\mathbf{x}\_0 + t\mathbf{x}\_1$ -- a pure regression loss with no SDE machinery. The straight-line paths are geometrically the most direct route from noise to data, requiring fewer ODE solver steps. No noise schedule is needed: the linear interpolation is fully determined. This brings a dramatic simplification in both theory and practice.

**(e)** [2 marks] **Flow matching / Diffusion ODE $\to$ Consistency models**

**Limitation:** Even with straight paths and high-order solvers, ODE-based generation still requires 10-20 sequential neural network evaluations. Each evaluation is a full forward pass through a large model (hundreds of millions of parameters), so even 10 steps takes seconds. For real-time applications (interactive editing, video generation, gaming), this iterative process is the final bottleneck.

**Key insight:** Consistency models observe that every point on an ODE trajectory maps to the same endpoint. If a network can learn this mapping $f\_\theta(\mathbf{x}\_t, t) \to \mathbf{x}\_0$ directly -- collapsing the entire trajectory into a single function evaluation -- then generation becomes a single forward pass. The self-consistency property $f\_\theta(\mathbf{x}\_t, t) = f\_\theta(\mathbf{x}\_{t'}, t')$ is enforced during training by requiring that pairs of points on the same trajectory (identified by the teacher model or by shared noise) produce the same output. This bypasses the sequential ODE solve entirely, achieving one-step generation while optionally allowing multi-step refinement for higher quality. The iterative process is "compiled" into a single network.

---

**END OF SOLUTIONS**
