# Week 1: Probability, Stochastic Processes, and Markov Chains

> *"Noise is not the opposite of signal. Noise is the medium through which signal travels."*
> -- loosely after Claude Shannon

---

## Overview

This is a foundations week. Before we can understand diffusion models -- which are, at their core, a marriage of deep learning and stochastic processes -- we need fluency in the mathematical language they speak. That language is probability, and its grammar is the Markov chain.

The central claim of this course: **a diffusion model is a pair of Markov chains**. The forward chain gradually destroys data by adding noise until nothing recognizable remains. The reverse chain learns to undo this destruction, step by step, recovering data from pure noise. Everything in weeks 2-13 builds on this idea.

This week, we lay the groundwork. We will develop the multivariate Gaussian -- the distribution that dominates diffusion models -- in enough depth to compute with it fluently. We will then develop the theory of Markov chains: what they are, when they converge, and why they converge. Finally, we will construct the **Gaussian channel**, the specific Markov transition that forms the building block of every diffusion model, and study the noise schedules that control how quickly signal is destroyed.

By the end of this week, you should be able to take any image, noise it to any level using a diffusion forward process, and understand exactly what you are doing mathematically.

### Prerequisites
- Multivariable calculus (partial derivatives, integrals over $\mathbb{R}^n$)
- Linear algebra (eigenvalues, positive definite matrices, matrix inverses)
- Basic probability (random variables, expectation, variance, Bayes' theorem)
- Python and PyTorch fundamentals

---

## 1. Multivariate Gaussians

### 1.1 The Density

A random vector $x \in \mathbb{R}^d$ follows a multivariate Gaussian distribution $x \sim \mathcal{N}(\mu, \Sigma)$ if its probability density function is:

$$
p(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)
$$

where $\mu \in \mathbb{R}^d$ is the mean vector and $\Sigma \in \mathbb{R}^{d \times d}$ is the covariance matrix, which must be symmetric and positive semi-definite.

The term $(x - \mu)^\top \Sigma^{-1} (x - \mu)$ is the **Mahalanobis distance** squared -- it measures how far $x$ is from the mean, accounting for the shape of the distribution. Contours of constant density are ellipsoids whose axes are the eigenvectors of $\Sigma$ and whose radii are proportional to the square roots of the eigenvalues.

### 1.2 Key Properties

**Marginalization.** If $(x\_1, x\_2)^\top \sim \mathcal{N}(\mu, \Sigma)$ with

$$
\mu = \begin{pmatrix} \mu_1 \\\\ \mu_2 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\\\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}
$$

then the marginal distributions are also Gaussian:

$$
x_1 \sim \mathcal{N}(\mu_1, \Sigma_{11}), \quad x_2 \sim \mathcal{N}(\mu_2, \Sigma_{22})
$$

This is remarkable: to marginalize, you simply read off the relevant block of the mean and covariance. No integration required.

**Conditioning.** The conditional distribution is also Gaussian:

$$
x_1 \mid x_2 \sim \mathcal{N}\left(\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2), \quad \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\right)
$$

The conditional mean is a linear function of $x\_2$, and the conditional covariance does not depend on $x\_2$ at all. This linearity is why Gaussians are so computationally tractable.

**Affine transformations.** If $x \sim \mathcal{N}(\mu, \Sigma)$ and $y = Ax + b$ for a matrix $A$ and vector $b$, then:

$$
y \sim \mathcal{N}(A\mu + b, \; A\Sigma A^\top)
$$

This is the result we will use most heavily. Every step of the diffusion forward process is an affine transformation of a Gaussian, so the result is again Gaussian.

**Sum of independent Gaussians.** If $x \sim \mathcal{N}(\mu\_x, \Sigma\_x)$ and $y \sim \mathcal{N}(\mu\_y, \Sigma\_y)$ are independent, then:

$$
x + y \sim \mathcal{N}(\mu_x + \mu_y, \; \Sigma_x + \Sigma_y)
$$

### 1.3 The Reparameterization Trick

Suppose we want to sample $x \sim \mathcal{N}(\mu, \Sigma)$. The naive approach -- directly sampling from a $d$-dimensional Gaussian with arbitrary mean and covariance -- is not differentiable with respect to $\mu$ and $\Sigma$. This is a problem when we want to backpropagate through sampling operations.

The **reparameterization trick** (Kingma and Welling, 2014) rewrites the sampling as a deterministic function of the parameters and a noise source:

$$
x = \mu + L\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

where $L$ is any matrix satisfying $LL^\top = \Sigma$ (for example, the Cholesky decomposition of $\Sigma$).

In the isotropic case $\Sigma = \sigma^2 I$, this simplifies to:

$$
x = \mu + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Why does this work? By the affine transformation property, $L\epsilon \sim \mathcal{N}(0, L I L^\top) = \mathcal{N}(0, \Sigma)$, so $\mu + L\epsilon \sim \mathcal{N}(\mu, \Sigma)$.

The trick separates the randomness (in $\epsilon$) from the parameters (in $\mu$ and $L$). Since $x$ is a differentiable function of $\mu$ and $L$, we can compute $\partial x / \partial \mu = I$ and $\partial x / \partial L = \epsilon$. This enables gradient-based optimization through stochastic layers, which is essential for training variational autoencoders and, as we will see, diffusion models.

---

## 2. Markov Chains

### 2.1 Definition

A **Markov chain** is a sequence of random variables $x\_0, x\_1, x\_2, \ldots$ satisfying the **Markov property**: the future depends on the past only through the present.

$$
p(x_{t+1} \mid x_0, x_1, \ldots, x_t) = p(x_{t+1} \mid x_t)
$$

In words: given the current state, the next state is independent of the entire history. This is not an approximation -- it is a structural assumption about the process.

A Markov chain is specified by:
1. An **initial distribution** $p(x\_0)$
2. A **transition kernel** $T(x\_{t+1} \mid x\_t)$ -- the conditional distribution of the next state given the current state

If the transition kernel does not depend on $t$, the chain is **time-homogeneous**. Most of our chains will be time-inhomogeneous: the amount of noise added at each step changes according to a schedule.

### 2.2 Transition Kernels

For finite state spaces, the transition kernel is a **transition matrix** $P$ where $P\_{ij} = p(x\_{t+1} = j \mid x\_t = i)$. Each row sums to 1. The distribution after one step is:

$$
p(x_1) = p(x_0) P
$$

After $n$ steps:

$$
p(x_n) = p(x_0) P^n
$$

For continuous state spaces (which is our setting), the transition kernel is a conditional density $T(x' \mid x)$. The marginal distribution evolves as:

$$
p(x_{t+1}) = \int T(x_{t+1} \mid x_t) \, p(x_t) \, dx_t
$$

This is the continuous analog of matrix multiplication.

### 2.3 The Chapman-Kolmogorov Equation

The Chapman-Kolmogorov equation says that the $n$-step transition probability can be decomposed through any intermediate time:

$$
p(x_{t+n} \mid x_t) = \int p(x_{t+n} \mid x_{t+m}) \, p(x_{t+m} \mid x_t) \, dx_{t+m}
$$

for any $0 < m < n$. In the finite case, this is simply matrix multiplication: $P^n = P^m P^{n-m}$.

This equation will be important when we derive the **closed-form expression** for the forward process: instead of applying the noise one step at a time, we can jump directly from $x\_0$ to $x\_t$ in one step.

### 2.4 Stationary Distributions

A distribution $\pi$ is a **stationary distribution** (or invariant distribution) of a Markov chain with transition kernel $T$ if:

$$
\pi(x') = \int T(x' \mid x) \, \pi(x) \, dx
$$

In words: if the chain is currently distributed according to $\pi$, it stays distributed according to $\pi$ after one step. The distribution is a fixed point of the dynamics.

For the diffusion forward process, the stationary distribution is $\mathcal{N}(0, I)$ -- pure isotropic Gaussian noise. No matter what data distribution we start from, the forward process converges to this fixed point. This is exactly what we want: the forward process transforms data into noise.

### 2.5 Detailed Balance

A transition kernel $T$ satisfies **detailed balance** with respect to a distribution $\pi$ if:

$$
\pi(x) T(x' \mid x) = \pi(x') T(x \mid x')
$$

This says that the "flow" of probability from $x$ to $x'$ equals the flow from $x'$ to $x$. It is a stronger condition than stationarity: detailed balance implies stationarity (integrate both sides over $x$), but not vice versa. A chain satisfying detailed balance is called **reversible**.

Detailed balance is the key property exploited by MCMC methods. If we design a transition kernel that satisfies detailed balance with respect to a target distribution $\pi$, we are guaranteed that the chain's stationary distribution is $\pi$.

**Verifying stationarity from detailed balance.** Suppose $T$ satisfies detailed balance with respect to $\pi$. Then:

$$
\int T(x' \mid x) \pi(x) \, dx = \int \pi(x') T(x \mid x') \, dx = \pi(x') \int T(x \mid x') \, dx = \pi(x') \cdot 1 = \pi(x')
$$

So $\pi$ is indeed stationary. $\square$

### 2.6 Ergodicity and Mixing

Having a stationary distribution is necessary but not sufficient. We also need the chain to **converge** to $\pi$ regardless of where it starts. A chain that does this is called **ergodic**.

Sufficient conditions for ergodicity (in the finite state case):
1. **Irreducibility:** Every state can be reached from every other state (the chain is "connected")
2. **Aperiodicity:** The chain does not cycle deterministically (the GCD of return times to any state is 1)

For continuous state spaces, the analogous conditions are more technical, but the intuition is the same: the chain must be able to explore the entire space and must not get trapped in cycles.

**The mixing time** is how long it takes for the chain to get close to $\pi$. Formally, for any starting distribution $p\_0$:

$$
t_{\text{mix}}(\epsilon) = \min\left\lbrace t : \Vert p_t - \pi\Vert _{\text{TV}} \leq \epsilon\right\rbrace 
$$

where $\Vert \cdot\Vert \_{\text{TV}}$ is the total variation distance. In diffusion models, the number of forward steps $T$ must be large enough that the forward process has approximately mixed -- that $p(x\_T) \approx \mathcal{N}(0, I)$.

**Convergence rate.** For finite-state reversible chains, the convergence rate is governed by the **spectral gap** -- the difference between the largest and second-largest eigenvalues of the transition matrix. If $P$ has eigenvalues $1 = \lambda\_1 > \lambda\_2 \geq \cdots \geq \lambda\_n > -1$, then:

$$
\Vert p_t - \pi\Vert _{\text{TV}} \leq \frac{1}{2}\sqrt{\frac{1-\pi_{\min}}{\pi_{\min}}} \cdot |\lambda_2|^t
$$

The convergence is exponential in $t$, with rate determined by $|\lambda\_2|$. The closer $\lambda\_2$ is to 1, the slower the mixing.

---

## 3. The Gaussian Channel

### 3.1 The Forward Process: One Step

We now arrive at the central construction of diffusion models. The **forward process** takes a data point $x\_0$ and gradually adds Gaussian noise over $T$ steps:

$$
x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

where $\beta\_t \in (0, 1)$ is the **noise schedule** at step $t$.

Written as a transition kernel:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \; \beta_t I\right)
$$

Let us unpack what this does:
- The signal $x\_{t-1}$ is **scaled down** by $\sqrt{1 - \beta\_t}$, which is slightly less than 1
- Independent Gaussian noise with variance $\beta\_t$ is **added**

Each step destroys a little bit of the signal and replaces it with noise. After many steps, the signal is gone.

### 3.2 The Closed-Form: Jumping to Any Time

A key insight: because each step is a linear-Gaussian operation, we can derive a closed-form expression for $x\_t$ directly in terms of $x\_0$, bypassing all intermediate steps.

Define $\alpha\_t = 1 - \beta\_t$ (the "signal retention" at step $t$) and the cumulative product:

$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t} (1 - \beta_s)
$$

Then, by repeatedly applying the one-step formula and using the fact that the sum of independent Gaussians is Gaussian:

$$
\boxed{x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)}
$$

Or equivalently:

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, \; (1 - \bar{\alpha}_t) I\right)
$$

**Derivation.** We prove this by induction. The base case $t = 1$ is immediate: $x\_1 = \sqrt{\alpha\_1} x\_0 + \sqrt{1 - \alpha\_1} \epsilon\_1$, and $\bar{\alpha}\_1 = \alpha\_1$.

For the inductive step, assume $x\_{t-1} = \sqrt{\bar{\alpha}\_{t-1}} x\_0 + \sqrt{1 - \bar{\alpha}\_{t-1}} \bar{\epsilon}\_{t-1}$ where $\bar{\epsilon}\_{t-1} \sim \mathcal{N}(0, I)$. Then:

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t
$$
$$
= \sqrt{\alpha_t}\left(\sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \bar{\epsilon}_{t-1}\right) + \sqrt{1 - \alpha_t} \epsilon_t
$$
$$
= \sqrt{\alpha_t \bar{\alpha}_{t-1}} x_0 + \sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})} \bar{\epsilon}_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t
$$

The last two terms are independent Gaussians with variances $\alpha\_t(1 - \bar{\alpha}\_{t-1})$ and $1 - \alpha\_t$ respectively. Their sum is Gaussian with variance:

$$
\alpha_t(1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t) = \alpha_t - \alpha_t \bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \alpha_t \bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t
$$

So $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1 - \bar{\alpha}\_t} \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$. $\square$

This closed-form is computationally essential: during training, we can sample any $x\_t$ directly from $x\_0$ without running the entire forward chain. We just sample $\epsilon \sim \mathcal{N}(0, I)$ and compute $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1 - \bar{\alpha}\_t} \epsilon$.

### 3.3 Interpreting the Gaussian Channel

The formula $x\_t = \sqrt{\bar{\alpha}\_t} \cdot x\_0 + \sqrt{1 - \bar{\alpha}\_t} \cdot \epsilon$ is a **Gaussian channel** from information theory. It is a weighted sum of signal and noise.

Notice that the coefficients satisfy:

$$
(\sqrt{\bar{\alpha}_t})^2 + (\sqrt{1 - \bar{\alpha}_t})^2 = \bar{\alpha}_t + (1 - \bar{\alpha}_t) = 1
$$

This means $x\_t$ has the same total variance as $x\_0$ (assuming $x\_0$ is normalized). The channel does not inflate or deflate the signal -- it replaces signal energy with noise energy, preserving the total.

At the extremes:
- When $t = 0$: $\bar{\alpha}\_0 = 1$, so $x\_0 = x\_0$. Pure signal, no noise.
- When $t \to \infty$: $\bar{\alpha}\_t \to 0$, so $x\_t \approx \epsilon$. Pure noise, no signal.

The forward process is a smooth interpolation between data and noise.

---

## 4. Signal-to-Noise Ratio and Noise Schedules

### 4.1 Signal-to-Noise Ratio

The **signal-to-noise ratio (SNR)** at time $t$ is the ratio of signal power to noise power:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

At $t = 0$: $\text{SNR} = \infty$ (pure signal). As $t$ grows, $\text{SNR}$ decreases monotonically toward 0 (pure noise).

Often we work with the log-SNR:

$$
\log \text{SNR}(t) = \log \bar{\alpha}_t - \log(1 - \bar{\alpha}_t)
$$

The log-SNR is the natural quantity for analyzing diffusion models. It decreases from $+\infty$ to $-\infty$ over the course of the forward process, and the noise schedule controls the rate of this decrease.

### 4.2 The Linear Schedule

The original DDPM paper (Ho, Sohl-Dickstein, and Abbeel, 2020) used a **linear schedule** for $\beta\_t$:

$$
\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})
$$

with typical values $\beta\_{\min} = 10^{-4}$ and $\beta\_{\max} = 0.02$, and $T = 1000$ steps.

The resulting $\bar{\alpha}\_t$ decreases roughly exponentially (since it is a product of terms close to 1), and the log-SNR decreases approximately linearly in $t$.

The linear schedule has a practical problem: it destroys information too quickly in the early steps and too slowly in the late steps. Most of the "interesting" noise levels are concentrated in a narrow range of $t$.

### 4.3 The Cosine Schedule

Nichol and Dhariwal (2021) proposed a **cosine schedule** that spreads information destruction more evenly:

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)
$$

where $s = 0.008$ is a small offset to prevent $\beta\_t$ from being too small near $t = 0$.

The cosine schedule produces a nearly linear decrease in log-SNR, which means the model spends roughly equal effort at each noise level. This leads to better image quality in practice.

**Why does the schedule matter?** The noise schedule determines where the model "allocates its capacity." If the schedule destroys too much information in the first few steps, the model never learns to denoise at low noise levels (where fine details live). If the schedule is too gentle, training steps are wasted on near-noiseless inputs where denoising is trivial.

The ideal schedule produces a uniform distribution of log-SNR values across the training steps, so that the model practices denoising at all difficulty levels equally.

### 4.4 Choosing $T$

How many steps do we need? Enough that $\bar{\alpha}\_T \approx 0$, so $x\_T \approx \mathcal{N}(0, I)$. With the linear schedule and $\beta\_{\max} = 0.02$:

$$
\bar{\alpha}_T = \prod_{t=1}^{T} (1 - \beta_t) \approx \exp\left(-\sum_{t=1}^{T} \beta_t\right) \approx \exp\left(-T \cdot \bar{\beta}\right)
$$

where $\bar{\beta} = (\beta\_{\min} + \beta\_{\max})/2 \approx 0.01$. For $\bar{\alpha}\_T < 10^{-4}$, we need $T \bar{\beta} > 9.2$, so $T > 920$. Hence the choice of $T = 1000$.

With the cosine schedule, the same quality can be achieved with fewer steps because the schedule is more efficient.

---

## 5. The Forward Process as a Markov Chain

### 5.1 The Full Picture

Let us now connect all the pieces. The forward process of a diffusion model is a Markov chain with:

- **State space:** $\mathbb{R}^d$ (the space of images, or whatever data we are modeling)
- **Initial distribution:** $q(x\_0)$ = the data distribution
- **Transition kernels:** $q(x\_t \mid x\_{t-1}) = \mathcal{N}(x\_t; \sqrt{1-\beta\_t} \cdot x\_{t-1}, \beta\_t I)$, which are time-inhomogeneous (the noise level $\beta\_t$ changes at each step)
- **Stationary distribution:** $\mathcal{N}(0, I)$

The chain satisfies detailed balance with respect to $\mathcal{N}(0, I)$ only in the limit of constant $\beta\_t$ (and even then, the detailed balance is with respect to the Ornstein-Uhlenbeck equilibrium). But stationarity holds: if $x\_{t-1} \sim \mathcal{N}(0, I)$, then $x\_t \sim \mathcal{N}(0, I)$ as well. You can verify this using the affine transformation property:

$$
\mathbb{E}[x_t] = \sqrt{1-\beta_t} \cdot \mathbb{E}[x_{t-1}] = 0
$$
$$
\text{Var}[x_t] = (1-\beta_t) \text{Var}[x_{t-1}] + \beta_t I = (1-\beta_t)I + \beta_t I = I
$$

### 5.2 The Reverse Process: A Preview

The reverse process runs the Markov chain backwards. By Bayes' rule:

$$
q(x_{t-1} \mid x_t) = \frac{q(x_t \mid x_{t-1}) q(x_{t-1})}{q(x_t)}
$$

This requires knowing $q(x\_{t-1})$ and $q(x\_t)$ -- the marginal distributions at each time -- which involve the intractable data distribution. We cannot compute the reverse process exactly.

However, a remarkable fact: **when $\beta\_t$ is small, the reverse transition $q(x\_{t-1} \mid x\_t)$ is approximately Gaussian**. This means we can learn a neural network to approximate it:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
$$

This is the core idea of denoising diffusion probabilistic models (DDPMs). We will develop it fully in Week 3.

### 5.3 The Tractable Posterior

While $q(x\_{t-1} \mid x\_t)$ is intractable, the posterior **conditioned on $x\_0$** is tractable:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

where:

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$

$$
\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1}) \beta_t}{1 - \bar{\alpha}_t}
$$

This follows from applying the Gaussian conditioning formula to the joint distribution $q(x\_{t-1}, x\_t \mid x\_0)$. We will derive this in detail in Week 3 when we develop the DDPM training objective.

---

## 6. Connection to Continuous-Time Processes

### 6.1 The Ornstein-Uhlenbeck Process

As the step size $\beta\_t \to 0$ and $T \to \infty$, the discrete forward process converges to a continuous-time stochastic differential equation (SDE):

$$
dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dW
$$

where $W$ is a standard Wiener process (Brownian motion). This is an **Ornstein-Uhlenbeck process** with time-dependent coefficients -- it drifts toward the origin (the $-\frac{1}{2}\beta(t) x$ term) while being buffeted by random noise (the $\sqrt{\beta(t)} \, dW$ term).

The continuous-time perspective, developed by Song et al. (2021) in their "Score SDE" framework, provides a unified view of diffusion models and enables more flexible sampling strategies. We will return to this in Week 5.

### 6.2 Why Continuous Time Matters (Preview)

The discrete Markov chain and the continuous SDE describe the same process at different resolutions. But the continuous formulation has two advantages:

1. **Exact likelihood computation** via the instantaneous change-of-variables formula
2. **Flexible sampling** via numerical SDE solvers with adaptive step sizes

For now, the discrete formulation is sufficient and more intuitive. We will develop the continuous perspective when we need it.

---

## Summary

1. **Multivariate Gaussians** are closed under marginalization, conditioning, and affine transformations. These closure properties make them the ideal building block for diffusion models.

2. **The reparameterization trick** $x = \mu + \sigma\epsilon$ separates randomness from parameters, enabling gradient-based optimization through stochastic operations.

3. **Markov chains** are sequences of random variables where the future depends on the past only through the present. They are specified by an initial distribution and a transition kernel.

4. **Stationary distributions** are fixed points of the transition dynamics. **Detailed balance** is a sufficient condition for stationarity. **Ergodicity** guarantees convergence to the stationary distribution from any starting point.

5. **The Gaussian channel** $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1 - \bar{\alpha}\_t} \epsilon$ gives a closed-form expression for the noised data at any time $t$, bypassing the need to run the chain step by step.

6. **Noise schedules** (linear, cosine) control the rate of information destruction. The log-SNR $\log(\bar{\alpha}\_t / (1 - \bar{\alpha}\_t))$ should decrease smoothly from $+\infty$ to $-\infty$.

7. **The forward process of a diffusion model is a Markov chain** that gradually converts any data distribution into $\mathcal{N}(0, I)$. The reverse process -- learned by a neural network -- will convert noise back into data.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| Gaussian density | $p(x) = (2\pi)^{-d/2}|\Sigma|^{-1/2}\exp(-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu))$ |
| Reparameterization | $x = \mu + \sigma\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$ |
| Forward step | $q(x\_t \mid x\_{t-1}) = \mathcal{N}(\sqrt{1-\beta\_t}\, x\_{t-1},\; \beta\_t I)$ |
| Closed-form | $x\_t = \sqrt{\bar{\alpha}\_t}\, x\_0 + \sqrt{1-\bar{\alpha}\_t}\, \epsilon$ |
| Cumulative product | $\bar{\alpha}\_t = \prod\_{s=1}^t (1 - \beta\_s)$ |
| Signal-to-noise ratio | $\text{SNR}(t) = \bar{\alpha}\_t / (1 - \bar{\alpha}\_t)$ |

---

## Suggested Reading

- **Ho, Jain, and Abbeel** (2020), "Denoising Diffusion Probabilistic Models" -- the DDPM paper. Read Section 2 (Background) for the forward process. We will study the full paper in Week 3.
- **Kingma and Welling** (2014), "Auto-Encoding Variational Bayes" -- the VAE paper that popularized the reparameterization trick. Appendix B.
- **Nichol and Dhariwal** (2021), "Improved Denoising Diffusion Probabilistic Models" -- the cosine schedule and other improvements.
- **Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole** (2021), "Score-Based Generative Modeling through Stochastic Differential Equations" -- the continuous-time perspective. We will study this in Week 5.
- **Bishop** (2006), *Pattern Recognition and Machine Learning*, Chapter 2 -- an excellent treatment of multivariate Gaussians with all the identities you will need.
- **Levin and Peres** (2017), *Markov Chains and Mixing Times* -- the standard reference for Markov chain convergence theory.
