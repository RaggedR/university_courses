# Week 3: Stochastic Differential Equations

> *"God does not play dice with the universe -- He plays a continuous-time stochastic process."*
> -- apologies to Einstein

---

## Overview

This is the hardest foundations week of the course. It is also the most rewarding.

In Week 1, we studied probability distributions and stochastic processes at a high level. In Week 2, we introduced the score function $\nabla_x \log p(x)$ and saw how Langevin dynamics uses it to draw samples. But Langevin dynamics, as we presented it, was a discrete-time algorithm -- a sequence of steps. The real theoretical backbone of diffusion models lives in continuous time, and the language of continuous-time stochastic processes is the **stochastic differential equation** (SDE).

This week, we develop the mathematical machinery of SDEs from the ground up. The payoff comes at the end: **Anderson's theorem (1982)**, which tells us that any forward diffusion process can be reversed in time, and that the reverse process depends on exactly one thing -- the score function $\nabla_x \log p_t(x)$ at each time $t$. This is the theoretical result that makes diffusion models possible.

The path is: Brownian motion $\to$ Ito calculus $\to$ SDEs $\to$ the Ornstein-Uhlenbeck process $\to$ Fokker-Planck equation $\to$ Anderson's reverse-time SDE.

### Prerequisites
- Week 1: Probability distributions, expectation, Gaussian properties, random walks
- Week 2: Score functions, Langevin dynamics
- Basic multivariable calculus: partial derivatives, chain rule, integration by parts

---

## 1. Brownian Motion (Wiener Process)

### 1.1 Definition

A **Brownian motion** (or **Wiener process**) $W_t$ is a continuous-time stochastic process with the following properties:

1. $W_0 = 0$
2. **Independent increments:** For any $0 \leq s < t \leq u < v$, the increments $W_t - W_s$ and $W_v - W_u$ are independent
3. **Gaussian increments:** $W_t - W_s \sim \mathcal{N}(0, t - s)$ for any $0 \leq s < t$
4. **Continuous paths:** $t \mapsto W_t$ is continuous (with probability 1)

That is all. From these four axioms, an enormous body of theory follows.

The increments are Gaussian with variance equal to the time elapsed. Over a time interval $\Delta t$, the process moves by a random amount with standard deviation $\sqrt{\Delta t}$. This square-root scaling is the signature of Brownian motion and will appear everywhere.

### 1.2 Properties

**Mean and variance.** Since $W_t = W_t - W_0 \sim \mathcal{N}(0, t)$:

$$\mathbb{E}[W_t] = 0, \qquad \text{Var}(W_t) = t$$

The process has no drift -- its expected position is always zero -- but its variance grows linearly with time. Brownian motion wanders further and further from the origin.

**Covariance.** For $s \leq t$:

$$\text{Cov}(W_s, W_t) = \mathbb{E}[W_s W_t] = \mathbb{E}[W_s(W_s + (W_t - W_s))] = \mathbb{E}[W_s^2] + \mathbb{E}[W_s]\mathbb{E}[W_t - W_s] = s$$

More generally, $\text{Cov}(W_s, W_t) = \min(s, t)$.

**Self-similarity.** For any $c > 0$, the rescaled process $\tilde{W}_t = \frac{1}{\sqrt{c}} W_{ct}$ is also a Brownian motion. This is because $\tilde{W}_t - \tilde{W}_s = \frac{1}{\sqrt{c}}(W_{ct} - W_{cs}) \sim \mathcal{N}(0, t-s)$. Brownian motion looks the same at every scale.

### 1.3 Non-Differentiability

Here is the essential difficulty that drives everything in this week: **Brownian motion is continuous but nowhere differentiable.**

Consider the "derivative" $\frac{W_{t+h} - W_t}{h}$. The numerator has standard deviation $\sqrt{h}$, so:

$$\frac{W_{t+h} - W_t}{h} \sim \mathcal{N}\left(0, \frac{1}{h}\right)$$

As $h \to 0$, the variance $1/h \to \infty$. The ratio does not converge -- it explodes. The path is so jagged, so rough, that no tangent line exists at any point.

This is not a technicality. It means we cannot write $\frac{dW}{dt}$ -- the derivative of Brownian motion with respect to time does not exist. Yet we need to do calculus with $W_t$. This is why we need Ito calculus.

### 1.4 The Discrete Approximation

The connection to Week 1: a random walk with steps $\pm 1$ of size $\sqrt{\Delta t}$, taken at intervals $\Delta t$, converges to Brownian motion as $\Delta t \to 0$ (Donsker's theorem). The discrete forward diffusion process from Week 1 -- repeatedly adding small Gaussian noise -- is a discrete approximation to a continuous-time process driven by Brownian motion.

### 1.5 Multidimensional Brownian Motion

For diffusion models, we work in $\mathbb{R}^d$ (e.g., $d = 786432$ for a $512 \times 512$ color image). A $d$-dimensional Brownian motion is a vector $\mathbf{W}_t = (W_t^{(1)}, \ldots, W_t^{(d)})$ where each component is an independent scalar Brownian motion. Its increments satisfy:

$$\mathbf{W}_t - \mathbf{W}_s \sim \mathcal{N}(\mathbf{0}, (t-s) \mathbf{I}_d)$$

Each pixel gets its own independent noise.

---

## 2. Ito Calculus

### 2.1 The Problem

Suppose we have a function $f(W_t)$ and we want to compute $df$. In ordinary calculus:

$$df = f'(W_t) \, dW_t$$

But this is wrong for Brownian motion, and the reason is subtle but fundamental.

Consider the Taylor expansion to second order:

$$f(W_{t+dt}) = f(W_t) + f'(W_t)(W_{t+dt} - W_t) + \frac{1}{2}f''(W_t)(W_{t+dt} - W_t)^2 + \cdots$$

In ordinary calculus, $dx^2$ is negligible compared to $dx$ (it is "second order"), so we drop it. But for Brownian motion, the increment $dW_t = W_{t+dt} - W_t$ has magnitude $\sim \sqrt{dt}$, so:

$$(dW_t)^2 \sim dt$$

This is **first order**, not second order! The second-order term in the Taylor expansion survives. This is the key insight behind Ito calculus.

### 2.2 The Ito Rule

The precise statement: as $dt \to 0$,

$$(dW_t)^2 \to dt \quad \text{(in mean square)}$$

More precisely, for a partition $0 = t_0 < t_1 < \cdots < t_n = T$ with $\max_i (t_{i+1} - t_i) \to 0$:

$$\sum_{i=0}^{n-1} (W_{t_{i+1}} - W_{t_i})^2 \xrightarrow{L^2} T$$

Each term $(W_{t_{i+1}} - W_{t_i})^2$ has expectation $t_{i+1} - t_i$, and the variance of the sum goes to zero. The "quadratic variation" of Brownian motion over $[0, T]$ is exactly $T$.

We also have the rules:

$$dt \cdot dt = 0, \qquad dW_t \cdot dt = 0, \qquad dW_t \cdot dW_t = dt$$

These are the **Ito multiplication rules**. The first two are standard (higher-order infinitesimals vanish), but the third is new and specific to stochastic calculus.

### 2.3 Ito's Lemma

Let $X_t$ be a stochastic process satisfying $dX_t = \mu_t \, dt + \sigma_t \, dW_t$, and let $f(t, x)$ be a twice-differentiable function. Then:

$$\boxed{df(t, X_t) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2}$$

Expanding $(dX_t)^2 = (\mu_t \, dt + \sigma_t \, dW_t)^2 = \sigma_t^2 \, dt$ (using the Ito multiplication rules), this becomes:

$$\boxed{df(t, X_t) = \left(\frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial x} + \frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma_t \frac{\partial f}{\partial x} \, dW_t}$$

This is **Ito's lemma** -- the chain rule for stochastic calculus. The extra term $\frac{1}{2}\sigma_t^2 f''$ is the "Ito correction" that has no analogue in ordinary calculus. It arises because $(dW)^2 = dt$ rather than zero.

### 2.4 Example: $f(W_t) = W_t^2$

Let us verify Ito's lemma on a simple case. Take $f(x) = x^2$ and $X_t = W_t$ (so $\mu_t = 0$, $\sigma_t = 1$).

**Ordinary calculus would give:** $d(W_t^2) = 2W_t \, dW_t$.

**Ito's lemma gives:**

$$d(W_t^2) = 2W_t \, dW_t + \frac{1}{2}(2)(1)^2 \, dt = 2W_t \, dW_t + dt$$

The extra $dt$ term is the Ito correction. Integrating both sides from $0$ to $T$:

$$W_T^2 = 2 \int_0^T W_t \, dW_t + T$$

Rearranging:

$$\int_0^T W_t \, dW_t = \frac{1}{2}(W_T^2 - T)$$

In ordinary calculus, $\int_0^T x \, dx = \frac{1}{2}x^2\big|_0^T = \frac{1}{2}T^2$. The Ito integral differs by the $-T/2$ correction term. Taking expectations (and using $\mathbb{E}[W_T^2] = T$), we get $\mathbb{E}[\int_0^T W_t \, dW_t] = 0$, which is a general property of Ito integrals.

### 2.5 Example: Geometric Brownian Motion

The SDE $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$ models stock prices (the Black-Scholes model). To solve it, apply Ito's lemma to $f(S) = \ln S$:

$$d(\ln S_t) = \frac{1}{S_t} dS_t + \frac{1}{2}\left(-\frac{1}{S_t^2}\right)(S_t)^2 \sigma^2 \, dt = \frac{1}{S_t}(\mu S_t \, dt + \sigma S_t \, dW_t) - \frac{1}{2}\sigma^2 \, dt$$

$$= \left(\mu - \frac{1}{2}\sigma^2\right) dt + \sigma \, dW_t$$

Integrating: $\ln S_T = \ln S_0 + (\mu - \frac{1}{2}\sigma^2)T + \sigma W_T$, so:

$$S_T = S_0 \exp\left[\left(\mu - \frac{1}{2}\sigma^2\right)T + \sigma W_T\right]$$

Notice the $-\frac{1}{2}\sigma^2$ correction to the drift. In ordinary calculus, we would get $S_T = S_0 e^{\mu T + \sigma W_T}$, which is wrong -- it would give $\mathbb{E}[S_T] = S_0 e^{(\mu + \frac{1}{2}\sigma^2)T}$ instead of the correct $\mathbb{E}[S_T] = S_0 e^{\mu T}$.

---

## 3. Stochastic Differential Equations

### 3.1 The General Form

A stochastic differential equation (SDE) is written:

$$\boxed{dX_t = f(X_t, t) \, dt + g(t) \, dW_t}$$

where:
- $X_t \in \mathbb{R}^d$ is the state at time $t$
- $f(X_t, t)$ is the **drift coefficient** -- the deterministic force
- $g(t)$ is the **diffusion coefficient** -- the noise intensity
- $W_t$ is a standard Brownian motion

The drift pulls the process in a specific direction; the diffusion adds randomness. The balance between them determines the character of the process.

The rigorous meaning of this equation is the integral form:

$$X_t = X_0 + \int_0^t f(X_s, s) \, ds + \int_0^t g(s) \, dW_s$$

where the first integral is an ordinary (Lebesgue) integral and the second is an Ito stochastic integral.

### 3.2 Existence and Uniqueness

Under mild conditions on $f$ and $g$ (Lipschitz continuity and linear growth bounds), the SDE has a unique strong solution. We will not prove this, but the conditions are satisfied for all the SDEs we use in this course.

### 3.3 Euler-Maruyama Discretization

To simulate an SDE numerically, we discretize time into steps of size $\Delta t$:

$$X_{t+\Delta t} = X_t + f(X_t, t) \Delta t + g(t) \sqrt{\Delta t} \, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

This is the **Euler-Maruyama method**, the stochastic analogue of the Euler method for ODEs. The key point: the noise scales as $\sqrt{\Delta t}$, not $\Delta t$, because $dW_t \sim \sqrt{dt}$.

This is exactly what we were doing in Weeks 1-2 when we added Gaussian noise at each step of the forward process. Euler-Maruyama is the workhorse numerical method for SDEs, and it is how diffusion models operate in practice.

### 3.4 The SDEs of Diffusion Models

In diffusion models, the forward process that corrupts data into noise is described by an SDE. The two most common choices are:

**Variance Preserving (VP) SDE:**

$$dX_t = -\frac{1}{2}\beta(t) X_t \, dt + \sqrt{\beta(t)} \, dW_t$$

This is the continuous-time limit of the DDPM forward process (Week 5). The drift pulls $X_t$ toward zero while the diffusion adds noise. The schedule $\beta(t)$ controls the speed.

**Variance Exploding (VE) SDE:**

$$dX_t = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dW_t$$

This has no drift -- it only adds noise with an increasing variance $\sigma^2(t)$. It is the continuous-time limit of the SMLD/NCSN approach (Week 6).

We will study both in detail in Week 7 when we unify them under the SDE framework of Song et al. (2021). For now, we focus on the simplest and most fundamental case.

---

## 4. The Ornstein-Uhlenbeck Process

### 4.1 Definition

The **Ornstein-Uhlenbeck (OU) process** is defined by the SDE:

$$\boxed{dX_t = -\theta X_t \, dt + \sigma \, dW_t}$$

where $\theta > 0$ is the mean-reversion rate and $\sigma > 0$ is the noise intensity.

This is the simplest non-trivial SDE and the continuous-time version of the forward diffusion process. It has two competing forces:

1. **Drift $-\theta X_t \, dt$:** Pulls $X_t$ back toward zero. The further $X_t$ is from zero, the stronger the pull. This is "mean reversion."
2. **Diffusion $\sigma \, dW_t$:** Adds random perturbations, pushing $X_t$ away from wherever it currently is.

### 4.2 Solving the OU SDE

This is a linear SDE and can be solved exactly. The technique is the **integrating factor** method, adapted for stochastic calculus.

Define $Y_t = e^{\theta t} X_t$. By Ito's lemma (with $f(t,x) = e^{\theta t} x$):

$$dY_t = \theta e^{\theta t} X_t \, dt + e^{\theta t} \, dX_t$$

Substituting $dX_t = -\theta X_t \, dt + \sigma \, dW_t$:

$$dY_t = \theta e^{\theta t} X_t \, dt + e^{\theta t}(-\theta X_t \, dt + \sigma \, dW_t)$$
$$= \theta e^{\theta t} X_t \, dt - \theta e^{\theta t} X_t \, dt + \sigma e^{\theta t} \, dW_t$$
$$= \sigma e^{\theta t} \, dW_t$$

The drift terms cancel! (This is why we chose the integrating factor $e^{\theta t}$.) Integrating from $0$ to $t$:

$$Y_t = Y_0 + \sigma \int_0^t e^{\theta s} \, dW_s$$

$$e^{\theta t} X_t = X_0 + \sigma \int_0^t e^{\theta s} \, dW_s$$

$$\boxed{X_t = e^{-\theta t} X_0 + \sigma \int_0^t e^{-\theta(t-s)} \, dW_s}$$

The first term is the initial condition decaying exponentially. The second term is accumulated noise, with recent noise weighted more heavily than old noise.

### 4.3 Distribution of the OU Process

Since $X_t$ is a linear function of Gaussian random variables ($X_0$ and the Ito integral), it is Gaussian. We need its mean and variance.

**Mean** (assuming deterministic $X_0 = x_0$):

$$\mathbb{E}[X_t] = e^{-\theta t} x_0$$

The mean decays exponentially toward zero.

**Variance:** The Ito isometry gives $\text{Var}\left(\int_0^t h(s) \, dW_s\right) = \int_0^t h(s)^2 \, ds$:

$$\text{Var}(X_t) = \sigma^2 \int_0^t e^{-2\theta(t-s)} \, ds = \sigma^2 \left[\frac{1 - e^{-2\theta t}}{2\theta}\right]$$

Therefore:

$$\boxed{X_t \mid X_0 = x_0 \sim \mathcal{N}\left(e^{-\theta t} x_0, \; \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})\right)}$$

This is a Gaussian transition kernel -- the **conditional distribution** of $X_t$ given $X_0$. It tells us everything about the forward process.

### 4.4 The Stationary Distribution

As $t \to \infty$, the mean $\to 0$ and the variance $\to \frac{\sigma^2}{2\theta}$:

$$X_\infty \sim \mathcal{N}\left(0, \frac{\sigma^2}{2\theta}\right)$$

This is the **stationary (equilibrium) distribution**. No matter where the process starts, it converges to this Gaussian. The OU process "forgets" its initial condition.

For the standard choice $\sigma^2 = 2\theta$, the stationary distribution is $\mathcal{N}(0, 1)$ -- standard Gaussian noise. This is exactly what we want for diffusion models: start from data, run the OU process, end at pure noise.

### 4.5 Connection to the Forward Diffusion Process

In the discrete-time diffusion process from Weeks 1-2, we added noise at each step:

$$x_t = \sqrt{1 - \beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon_t$$

As $\beta_t \to 0$ and the number of steps $\to \infty$ (continuous-time limit), this becomes the OU process:

$$dX_t = -\frac{1}{2}\beta(t) X_t \, dt + \sqrt{\beta(t)} \, dW_t$$

which is the VP SDE with $\theta = \frac{1}{2}\beta(t)$ and $\sigma = \sqrt{\beta(t)}$. The OU process is the continuous-time backbone of DDPM.

---

## 5. The Fokker-Planck Equation

### 5.1 From Sample Paths to Densities

An SDE describes how individual sample paths $X_t$ evolve. But we often care about how the **probability density** $p_t(x) = p(x, t)$ evolves. Given an initial density $p_0(x)$ (the data distribution) and an SDE that describes how samples move, what is $p_t(x)$ at later times?

The answer is the **Fokker-Planck equation** (also called the Kolmogorov forward equation).

### 5.2 Derivation (Sketch)

Consider the SDE $dX_t = f(X_t, t) \, dt + g(t) \, dW_t$. For any smooth test function $\phi(x)$, the expected value $\mathbb{E}[\phi(X_t)]$ evolves according to:

$$\frac{d}{dt} \mathbb{E}[\phi(X_t)] = \mathbb{E}\left[f(X_t, t) \phi'(X_t) + \frac{1}{2}g(t)^2 \phi''(X_t)\right]$$

This follows from applying Ito's lemma to $\phi(X_t)$ and taking expectations (the $dW_t$ term has zero expectation).

Now rewrite the left side as $\frac{d}{dt}\int \phi(x) p_t(x) \, dx = \int \phi(x) \frac{\partial p_t}{\partial t} dx$, and the right side as $\int [f \phi' + \frac{1}{2}g^2 \phi''] p_t \, dx$. Integrating by parts (moving derivatives from $\phi$ to $p_t$), we get:

$$\int \phi(x) \frac{\partial p_t}{\partial t} dx = \int \phi(x) \left[-\frac{\partial}{\partial x}(f \cdot p_t) + \frac{1}{2}\frac{\partial^2}{\partial x^2}(g^2 \cdot p_t)\right] dx$$

Since this holds for all test functions $\phi$, we conclude:

### 5.3 The Fokker-Planck Equation

$$\boxed{\frac{\partial p_t(x)}{\partial t} = -\frac{\partial}{\partial x}\big[f(x,t) \, p_t(x)\big] + \frac{1}{2}\frac{\partial^2}{\partial x^2}\big[g(t)^2 \, p_t(x)\big]}$$

In multiple dimensions:

$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla \cdot \big[f(\mathbf{x},t) \, p_t(\mathbf{x})\big] + \frac{1}{2}\nabla^2\big[g(t)^2 \, p_t(\mathbf{x})\big]$$

where $\nabla \cdot$ is the divergence and $\nabla^2$ is the Laplacian (sum of second partial derivatives).

The two terms have clear physical interpretations:

1. **$-\nabla \cdot (f \, p_t)$:** The **transport** term. The drift $f$ carries probability density along with it, like a fluid flow. This is a continuity equation -- probability is conserved.
2. **$\frac{1}{2}\nabla^2(g^2 p_t)$:** The **diffusion** term. Noise spreads the density out, smoothing sharp peaks and filling in valleys. This is the heat equation applied to probability.

### 5.4 Fokker-Planck for the OU Process

For the OU process $dX_t = -\theta X_t \, dt + \sigma \, dW_t$, we have $f(x,t) = -\theta x$ and $g(t) = \sigma$. The Fokker-Planck equation becomes:

$$\frac{\partial p_t}{\partial t} = \frac{\partial}{\partial x}(\theta x \, p_t) + \frac{\sigma^2}{2}\frac{\partial^2 p_t}{\partial x^2}$$

$$= \theta \, p_t + \theta x \frac{\partial p_t}{\partial x} + \frac{\sigma^2}{2}\frac{\partial^2 p_t}{\partial x^2}$$

**Verification:** We showed above that $X_t \sim \mathcal{N}(m_t, v_t)$ with $m_t = e^{-\theta t} x_0$ and $v_t = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$. One can verify (a good exercise) that the Gaussian density:

$$p_t(x) = \frac{1}{\sqrt{2\pi v_t}} \exp\left(-\frac{(x - m_t)^2}{2v_t}\right)$$

satisfies this Fokker-Planck equation. The mean drifts toward zero while the variance grows toward $\sigma^2 / 2\theta$.

### 5.5 The Fokker-Planck as a Conservation Law

The Fokker-Planck equation can be written as:

$$\frac{\partial p_t}{\partial t} = -\frac{\partial J}{\partial x}$$

where the **probability current** is:

$$J = f(x,t) \, p_t(x) - \frac{1}{2}\frac{\partial}{\partial x}[g(t)^2 \, p_t(x)]$$

This is a conservation law: probability is neither created nor destroyed, only moved around. At the stationary distribution ($\partial p_t / \partial t = 0$), the current $J$ is constant (and zero, for the OU process with natural boundary conditions).

---

## 6. Anderson's Reverse-Time SDE

### 6.1 The Question

Here is the central question: if we know the forward SDE that turns data into noise, can we find a reverse SDE that turns noise back into data?

Starting from data $X_0 \sim p_{\text{data}}$, the forward SDE $dX_t = f(X_t, t) \, dt + g(t) \, dW_t$ evolves $X_t$ toward the noise distribution $X_T \sim p_T \approx \mathcal{N}(0, I)$ (for large $T$).

We want to start from $X_T \sim p_T$ and run the process backwards in time to recover $X_0 \sim p_{\text{data}}$. Is this possible?

### 6.2 Anderson's Theorem (1982)

The answer is yes, and the result is due to Anderson (1982). The reverse-time SDE is:

$$\boxed{dX_t = \left[f(X_t, t) - g(t)^2 \nabla_{X_t} \log p_t(X_t)\right] dt + g(t) \, d\bar{W}_t}$$

where $\bar{W}_t$ is a Brownian motion running backwards in time, and $dt$ is a **negative** time increment (we are going from $T$ to $0$).

Let us unpack this. The reverse SDE has:
- **The same diffusion coefficient** $g(t)$ as the forward process
- **A modified drift** that is the forward drift $f$ minus a correction term $g^2 \nabla_x \log p_t(x)$

The correction term is $g(t)^2$ times the **score function** $\nabla_x \log p_t(x)$ -- the gradient of the log-density at time $t$.

### 6.3 Intuition

Why does the score function appear? Consider what the reverse process must do. Going forward, the drift $f$ carries probability in one direction while the diffusion $g \, dW$ spreads it out. Going backward, we must:

1. Reverse the drift (easy -- just negate $f$)
2. **Un-spread** the diffusion (hard -- this requires knowing where the probability came from)

The score $\nabla_x \log p_t(x)$ points from low-density regions toward high-density regions. It tells us "where the probability is." Adding $g^2 \nabla_x \log p_t(x)$ to the drift steers the reverse process from spread-out noise back toward the concentrated data distribution.

Another way to see it: the forward diffusion erases information about the data distribution. The score function is exactly the information needed to undo this erasure. Without it, many different initial distributions could lead to the same noise distribution -- the score tells us which one we started from.

### 6.4 Derivation Sketch

The derivation relies on the Fokker-Planck equation. The forward SDE induces the forward Fokker-Planck:

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (f \, p_t) + \frac{1}{2}\nabla^2(g^2 \, p_t)$$

The reverse-time process must satisfy the same marginals $p_t(x)$ at each time $t$, but evolving backward. The **reverse Fokker-Planck** is:

$$-\frac{\partial p_t}{\partial t} = -\nabla \cdot (\tilde{f} \, p_t) + \frac{1}{2}\nabla^2(g^2 \, p_t)$$

where $\tilde{f}$ is the reverse drift. (The left side has a minus sign because time runs backward.)

Adding the forward and reverse Fokker-Planck equations:

$$0 = -\nabla \cdot [(f + \tilde{f}) \, p_t] + \nabla^2(g^2 \, p_t)$$

Expanding $\nabla^2(g^2 p_t) = g^2 \nabla^2 p_t = g^2 \nabla \cdot (\nabla p_t) = g^2 \nabla \cdot (p_t \nabla \log p_t)$ (when $g$ depends only on $t$, not $x$):

$$0 = -\nabla \cdot [(f + \tilde{f}) \, p_t] + \nabla \cdot [g^2 \, p_t \nabla \log p_t]$$

This is satisfied when:

$$\tilde{f} = -f + g^2 \nabla \log p_t$$

So the reverse drift is $\tilde{f} = -f + g^2 \nabla_x \log p_t(x)$. Writing the reverse SDE:

$$dX_t = \tilde{f} \, dt + g \, d\bar{W}_t = [-f + g^2 \nabla_x \log p_t(x)] \, dt + g \, d\bar{W}_t$$

Since $dt < 0$ (backward in time), and flipping the sign convention to write the equation with $|dt|$, we get Anderson's result.

### 6.5 The Reverse OU Process

For the OU process $dX_t = -\theta X_t \, dt + \sigma \, dW_t$, we have $f(x,t) = -\theta x$ and $g = \sigma$.

The reverse SDE is:

$$dX_t = \left[-\theta X_t - \sigma^2 \nabla_x \log p_t(x)\right] dt + \sigma \, d\bar{W}_t$$

Wait -- that minus sign is wrong. Let's be careful. The reverse drift formula gives $\tilde{f} = -f + g^2 \nabla_x \log p_t$. With $f = -\theta x$:

$$\tilde{f} = \theta x + \sigma^2 \nabla_x \log p_t(x)$$

At the stationary distribution $p_\infty(x) = \mathcal{N}(0, \sigma^2/2\theta)$, the score is:

$$\nabla_x \log p_\infty(x) = -\frac{2\theta}{\sigma^2} x$$

So the reverse drift becomes $\tilde{f} = \theta x + \sigma^2 \cdot (-\frac{2\theta}{\sigma^2} x) = \theta x - 2\theta x = -\theta x$, which is the same as the forward drift. This makes sense: the OU process at stationarity is reversible.

### 6.6 The Practical Consequence

Anderson's theorem tells us: **if you can estimate the score function $\nabla_x \log p_t(x)$ for all $t$, you can reverse any diffusion process.**

This is the foundation of score-based generative models:

1. **Forward process:** Run an SDE that turns data into noise (known analytically)
2. **Score estimation:** Train a neural network $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ to estimate the score at each noise level (Week 4)
3. **Reverse process:** Plug the estimated score into Anderson's reverse SDE and simulate it backward from noise to data (Weeks 5-6)

The entire diffusion model framework reduces to one problem: **learn the score function**.

---

## 7. The Probability Flow ODE

### 7.1 A Deterministic Alternative

Anderson's reverse SDE involves randomness ($d\bar{W}_t$). Song et al. (2021) showed that there is also a deterministic ODE that has the same marginal densities $p_t(x)$ as the SDE:

$$\boxed{\frac{dX_t}{dt} = f(X_t, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(X_t)}$$

This is called the **probability flow ODE**. It is an ordinary differential equation -- no randomness. Yet if you start from $X_T \sim p_T$ and integrate this ODE backward in time, you get samples from $p_0 = p_{\text{data}}$.

The probability flow ODE has half the score correction of the reverse SDE (factor of $\frac{1}{2}$ instead of $1$). The missing half is "compensated" by the absence of the noise term $g \, d\bar{W}_t$.

### 7.2 Why This Matters

The probability flow ODE enables:
- **Exact likelihood computation** (via the instantaneous change of variables formula)
- **Deterministic sampling** (same noise $\to$ same image, useful for interpolation and editing)
- **Fast ODE solvers** (adaptive step-size methods like RK45 can be much faster than SDE discretization)

We will return to the probability flow ODE in Week 7.

---

## Summary

1. **Brownian motion** is a continuous, nowhere-differentiable random process with Gaussian increments. It is the source of randomness in SDEs.

2. **Ito calculus** extends ordinary calculus to handle Brownian motion. The key difference: $(dW)^2 = dt$, which gives rise to the Ito correction term in Ito's lemma.

3. **Stochastic differential equations** $dX_t = f(X_t, t) \, dt + g(t) \, dW_t$ describe continuous-time random processes. They are simulated via Euler-Maruyama discretization.

4. **The Ornstein-Uhlenbeck process** $dX_t = -\theta X_t \, dt + \sigma \, dW_t$ is the continuous-time forward diffusion process. It has a known Gaussian transition kernel and converges to a Gaussian stationary distribution.

5. **The Fokker-Planck equation** describes how probability densities evolve under an SDE. It has a transport term (drift) and a diffusion term (noise spreading).

6. **Anderson's theorem (1982):** Every forward SDE has a corresponding reverse-time SDE. The reverse drift depends on the score function $\nabla_x \log p_t(x)$. This is the theoretical foundation of diffusion models.

7. **The probability flow ODE** provides a deterministic alternative to the reverse SDE, enabling exact likelihood computation and deterministic sampling.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| Brownian motion increment | $W_t - W_s \sim \mathcal{N}(0, t-s)$ |
| Ito's lemma | $df = (\partial_t f + \mu \partial_x f + \frac{1}{2}\sigma^2 \partial_{xx} f) \, dt + \sigma \partial_x f \, dW$ |
| General SDE | $dX_t = f(X_t, t) \, dt + g(t) \, dW_t$ |
| Euler-Maruyama | $X_{t+\Delta t} = X_t + f \Delta t + g\sqrt{\Delta t} \, \epsilon$ |
| OU process | $dX_t = -\theta X_t \, dt + \sigma \, dW_t$ |
| OU transition kernel | $X_t \mid X_0 \sim \mathcal{N}(e^{-\theta t}x_0, \frac{\sigma^2}{2\theta}(1-e^{-2\theta t}))$ |
| Fokker-Planck | $\partial_t p = -\nabla \cdot (fp) + \frac{1}{2}\nabla^2(g^2 p)$ |
| Reverse SDE (Anderson) | $dX_t = [f - g^2 \nabla_x \log p_t] \, dt + g \, d\bar{W}_t$ |
| Probability flow ODE | $dX_t/dt = f - \frac{1}{2}g^2 \nabla_x \log p_t$ |

---

## Suggested Reading

- **Anderson, B.D.O.** (1982), "Reverse-time diffusion equation models" -- the foundational result. Short, elegant paper.
- **Oksendal, B.** (2003), *Stochastic Differential Equations*, Chapters 3-5 -- the standard textbook on SDEs. Rigorous but readable.
- **Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., and Poole, B.** (2021), "Score-Based Generative Modeling through Stochastic Differential Equations" -- the paper that unified diffusion models via SDEs.
- **Särkkä, S. and Solin, A.** (2019), *Applied Stochastic Differential Equations*, Chapters 4-6 -- accessible introduction with a focus on applications.
- **Luo, C.** (2022), "Understanding Diffusion Models: A Unified Perspective" -- excellent tutorial that covers SDEs in the diffusion model context.
