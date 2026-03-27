---
title: "Week 7: The SDE Unification"
---

# Week 7: The SDE Unification

> *"We propose a unified framework based on stochastic differential equations that encompasses previous approaches and enables new sampling and likelihood computation methods."*
> -- Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole (2021)

---

## Overview

In Week 5, we built DDPMs from the variational inference tradition. In Week 6, we built NCSNs from the score matching tradition. We showed they are equivalent -- the same model in different notation. This week, we take the next logical step: unify both into a single, continuous-time framework based on **stochastic differential equations (SDEs)**.

The key paper is Song et al. (2021), "Score-Based Generative Modeling through Stochastic Differential Equations." It is one of the most important papers in generative modelling, not because it introduces a radically new algorithm, but because it reveals the mathematical structure underlying all diffusion models. DDPMs and NCSNs are shown to be discrete-time approximations of continuous-time SDEs. The forward noising process is an SDE. The reverse denoising process is the *reverse-time SDE*, which depends only on the score function. And there is a deterministic counterpart -- the **probability flow ODE** -- that has the same marginal distributions as the SDE but uses no randomness, enabling exact likelihood computation and deterministic encoding.

This is the grand synthesis. After this week, you will have a unified language for thinking about all diffusion models, and the tools to design new ones.

### Prerequisites
- Week 3: Stochastic differential equations, Ito calculus, Fokker-Planck equation
- Week 5: DDPM (forward/reverse processes, noise schedules)
- Week 6: NCSN (score estimation, Langevin dynamics, the DDPM-NCSN equivalence)

---

## 1. From Discrete to Continuous Time

### 1.1 The Continuous Limit Intuition

In DDPM, we have $T$ discrete timesteps. The forward process applies a small perturbation at each step. As $T \to \infty$ with each step becoming infinitesimally small, the discrete Markov chain converges to a continuous-time stochastic process -- an SDE.

Consider a generic forward step:

$$
x_{t+1} = \sqrt{1 - \beta_t}\, x_t + \sqrt{\beta_t}\, \varepsilon_t
$$

For small $\beta\_t$, $\sqrt{1-\beta\_t} \approx 1 - \beta\_t/2$, so:

$$
x_{t+1} - x_t \approx -\frac{\beta_t}{2}\, x_t + \sqrt{\beta_t}\, \varepsilon_t
$$

If we think of the index $t$ as continuous time and $\beta\_t$ as a rate $\beta(t)\,dt$:

$$
dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dW
$$

where $dW$ is a Wiener process increment ($dW \sim \mathcal{N}(0, dt\, I)$). This is a linear SDE -- specifically, an Ornstein-Uhlenbeck process.

### 1.2 The General SDE Framework

Song et al. (2021) formalize this by defining the forward process as a general SDE on $t \in [0, T]$:

$$
\boxed{dx = f(x, t)\, dt + g(t)\, dW}
$$

where:
- $f(x, t)$ is the **drift coefficient** -- the deterministic force acting on $x$
- $g(t)$ is the **diffusion coefficient** -- the noise intensity
- $W$ is a standard Wiener process (Brownian motion) in $\mathbb{R}^d$

The solution $x(t)$ is a stochastic process whose marginal distribution at time $t$ we denote $p\_t(x)$. At $t = 0$, $p\_0 = p\_{\text{data}}$. At $t = T$, $p\_T \approx \pi$ for some known prior distribution $\pi$ (typically $\mathcal{N}(0, \sigma\_T^2 I)$).

The drift and diffusion coefficients define the forward process. Different choices give different diffusion models.

---

## 2. Two Canonical SDEs

### 2.1 VP-SDE (Variance Preserving)

The **Variance Preserving SDE** is the continuous-time limit of DDPM:

$$
\boxed{dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dW}
$$

where $\beta(t)$ is a continuous noise schedule (e.g., linearly increasing from $\beta\_{\min}$ to $\beta\_{\max}$).

**Why "variance preserving"?** The drift $-\frac{1}{2}\beta(t)x$ shrinks the signal, while the noise $\sqrt{\beta(t)}\, dW$ adds variance. These are balanced so that if $x(0) \sim \mathcal{N}(0, I)$, then $x(t) \sim \mathcal{N}(0, I)$ for all $t$ -- the variance is preserved at 1 for unit-variance data. (For data that does not have unit variance, the process converges to a known Gaussian.)

**Transition kernel.** Because the VP-SDE is linear, the transition from $x(0)$ to $x(t)$ is Gaussian:

$$
q(x(t) \mid x(0)) = \mathcal{N}\!\left(x(t);\; e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}\, x(0),\; \left(1 - e^{-\int_0^t \beta(s)\,ds}\right) I\right)
$$

Define $\bar{\alpha}(t) = e^{-\int\_0^t \beta(s)\,ds}$. Then:

$$
q(x(t) \mid x(0)) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}(t)}\, x(0),\; (1 - \bar{\alpha}(t))\, I\right)
$$

This is exactly the DDPM closed-form formula from Week 5, now in continuous time. The discrete $\bar{\alpha}\_t = \prod\_{s=1}^t (1-\beta\_s)$ becomes $\bar{\alpha}(t) = e^{-\int\_0^t \beta(s)\,ds}$, and the product becomes an integral in the exponent (since $\log\prod(1-\beta\_s) = \sum\log(1-\beta\_s) \approx -\sum \beta\_s \to -\int \beta(s)\,ds$ for small $\beta\_s$).

### 2.2 VE-SDE (Variance Exploding)

The **Variance Exploding SDE** is the continuous-time limit of NCSN:

$$
\boxed{dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\; dW}
$$

There is no drift -- only diffusion. The noise intensity is chosen so that the marginal variance at time $t$ equals $\sigma^2(t)$.

**Why "variance exploding"?** Unlike the VP-SDE, there is no signal-shrinking drift. The noise accumulates without bound: $\text{Var}(x(t)) = \text{Var}(x(0)) + \sigma^2(t)$. The variance "explodes" to infinity as $t \to T$.

**Transition kernel.** The VE-SDE is also linear, so:

$$
q(x(t) \mid x(0)) = \mathcal{N}\!\left(x(0),\; \sigma^2(t)\, I\right)
$$

The data is not rescaled -- just noise is added. This matches the NCSN convention from Week 6.

### 2.3 Connection to Discrete Models

| Discrete Model | Continuous SDE | Drift $f(x,t)$ | Diffusion $g(t)$ |
|---------------|---------------|-----------------|-------------------|
| DDPM | VP-SDE | $-\frac{1}{2}\beta(t)\, x$ | $\sqrt{\beta(t)}$ |
| NCSN | VE-SDE | $0$ | $\sqrt{d\sigma^2(t)/dt}$ |

The discrete models are Euler-Maruyama discretizations of their respective SDEs. Going to continuous time reveals that these are special cases of a general family, parameterized by the choice of $f$ and $g$.

### 2.4 Sub-VP SDE

Song et al. also define a **sub-VP SDE**, which interpolates between VP and VE:

$$
dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)\,ds})}\; dW
$$

This preserves the mean-shrinkage of VP-SDE but uses a smaller diffusion coefficient, resulting in tighter variational bounds. The marginal variance at time $t$ is $(1 - e^{-\int\_0^t \beta(s)\,ds})^2$ instead of $1 - e^{-\int\_0^t \beta(s)\,ds}$.

---

## 3. The Reverse-Time SDE

### 3.1 Anderson's Result

The remarkable fact that makes all of this work: any SDE has a corresponding reverse-time SDE. Anderson (1982) showed that if the forward SDE is:

$$
dx = f(x, t)\, dt + g(t)\, dW
$$

then the reverse-time SDE (running backward from $t = T$ to $t = 0$) is:

$$
\boxed{dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\, d\bar{W}}
$$

where $d\bar{W}$ is a reverse-time Wiener process and $\nabla\_x \log p\_t(x)$ is the score of the marginal distribution at time $t$.

Note the structure: the reverse drift has two terms:
1. $f(x,t)$: the same drift as the forward process
2. $-g(t)^2 \nabla\_x \log p\_t(x)$: a correction that depends on the **score function**

The reverse SDE is fully determined by the forward SDE coefficients $f, g$ and the time-dependent score $\nabla\_x \log p\_t(x)$. If we can estimate the score at all times, we can reverse the diffusion and generate data.

### 3.2 Reverse VP-SDE

For the VP-SDE ($f = -\frac{1}{2}\beta(t)x$, $g = \sqrt{\beta(t)}$):

$$
dx = \left[-\frac{1}{2}\beta(t)\, x - \beta(t)\, \nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)}\, d\bar{W}
$$

### 3.3 Reverse VE-SDE

For the VE-SDE ($f = 0$, $g = \sqrt{d\sigma^2/dt}$):

$$
dx = -\frac{d[\sigma^2(t)]}{dt}\, \nabla_x \log p_t(x)\, dt + \sqrt{\frac{d[\sigma^2(t)]}{dt}}\; d\bar{W}
$$

### 3.4 Score Approximation

In practice, we replace $\nabla\_x \log p\_t(x)$ with a learned score network $s\_\theta(x, t)$, giving the approximate reverse SDE:

$$
dx = \left[f(x, t) - g(t)^2\, s_\theta(x, t)\right] dt + g(t)\, d\bar{W}
$$

This can be solved numerically using any SDE solver (Euler-Maruyama, Milstein, adaptive methods). The choice of solver trades off speed and accuracy.

---

## 4. The Probability Flow ODE

### 4.1 The Deterministic Counterpart

Here is one of the most beautiful results in the paper. For any SDE of the form $dx = f(x,t)\,dt + g(t)\,dW$, there exists a deterministic ODE whose solution has the *same marginal distributions* $p\_t(x)$ at all times:

$$
\boxed{dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right] dt}
$$

This is the **probability flow ODE**. Compare to the reverse SDE:

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\, d\bar{W}
$$

The ODE is obtained by removing the noise term $g(t)\,d\bar{W}$ and halving the score coefficient from $g(t)^2$ to $\frac{1}{2}g(t)^2$. The two processes have identical marginal distributions but different path-wise behavior:

- The **SDE** traces stochastic paths -- different random seeds give different trajectories, even from the same starting point.
- The **ODE** traces deterministic paths -- given $x(T)$, the trajectory $x(t)$ for $t \in [0, T]$ is unique.

### 4.2 Why This Holds: The Fokker-Planck Connection

The marginal distributions $p\_t(x)$ of the SDE evolve according to the **Fokker-Planck equation** (from Week 3):

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot \left[f(x,t)\, p_t\right] + \frac{1}{2}g(t)^2 \Delta p_t
$$

where $\Delta$ is the Laplacian. The probability flow ODE is derived by finding a deterministic velocity field $\tilde{f}(x, t)$ that produces the same Fokker-Planck equation. For the ODE $dx = \tilde{f}(x,t)\,dt$, the Fokker-Planck equation (which reduces to the continuity equation since there is no diffusion term) is:

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot \left[\tilde{f}(x,t)\, p_t\right]
$$

Setting the two equal and using the identity $\frac{1}{2}g^2 \Delta p\_t = -\nabla \cdot \left[-\frac{1}{2}g^2 \nabla\_x \log p\_t \cdot p\_t\right]$ (which follows from $\nabla p\_t = p\_t \nabla \log p\_t$), we get:

$$
\tilde{f}(x,t) = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)
$$

This is exactly the probability flow ODE drift. $\square$

### 4.3 The Probability Flow ODE for VP-SDE

Substituting $f = -\frac{1}{2}\beta(t)x$ and $g = \sqrt{\beta(t)}$:

$$
dx = \left[-\frac{1}{2}\beta(t)\, x - \frac{1}{2}\beta(t)\, \nabla_x \log p_t(x)\right] dt
$$

$$
= -\frac{1}{2}\beta(t)\left[x + \nabla_x \log p_t(x)\right] dt
$$

With a learned score:

$$
dx = -\frac{1}{2}\beta(t)\left[x + s_\theta(x, t)\right] dt
$$

This is a neural ODE that can be solved with standard ODE solvers (Euler, RK45, Dormand-Prince, etc.).

### 4.4 The Probability Flow ODE for VE-SDE

Substituting $f = 0$ and $g = \sqrt{d\sigma^2/dt}$:

$$
dx = -\frac{1}{2}\frac{d[\sigma^2(t)]}{dt}\, \nabla_x \log p_t(x)\, dt
$$

The ODE simply follows the score, scaled by the rate of noise change.

---

## 5. Why the Probability Flow ODE Matters

### 5.1 Exact Likelihood Computation

The probability flow ODE is a **continuous normalizing flow** (CNF). It defines a smooth, invertible mapping between the data space ($t = 0$) and the latent space ($t = T$). By the instantaneous change of variables formula (Chen et al., 2018):

$$
\log p_0(x(0)) = \log p_T(x(T)) + \int_0^T \nabla \cdot \tilde{f}(x(t), t)\, dt
$$

where $\tilde{f}(x, t) = f(x,t) - \frac{1}{2}g(t)^2 s\_\theta(x, t)$ is the ODE velocity field and $\nabla \cdot \tilde{f}$ is its divergence.

This gives us **exact log-likelihoods** for diffusion models -- something the ELBO only bounds. Computing the integral requires solving the ODE forward (from $t=0$ to $t=T$) while accumulating the divergence, using the Hutchinson trace estimator for efficiency:

$$
\nabla \cdot \tilde{f}(x, t) = \mathbb{E}_{v \sim \mathcal{N}(0,I)}\!\left[v^\top \frac{\partial \tilde{f}}{\partial x} v\right]
$$

Song et al. (2021) achieved state-of-the-art log-likelihoods on CIFAR-10 using the probability flow ODE, surpassing even autoregressive models.

### 5.2 Deterministic Encoding

Running the ODE forward ($t: 0 \to T$) maps data to latent space deterministically:

$$
x_0 \mapsto x_T = \text{ODESolve}(x_0, 0 \to T)
$$

This is a learned encoder with no stochasticity. Two useful properties:

**Uniqueness.** Each data point maps to exactly one latent code, and vice versa. This is unlike the SDE, where the same data point maps to different latent codes depending on the noise realization.

**Invertibility.** The encoding is exactly invertible: running the ODE backward recovers $x\_0$ from $x\_T$ (up to numerical precision).

### 5.3 Latent Space Interpolation

Because the ODE defines a smooth, bijective map between data and latent space, we can interpolate in latent space:

1. Encode two images: $z\_1 = \text{encode}(x\_1)$, $z\_2 = \text{encode}(x\_2)$
2. Interpolate: $z\_\alpha = (1-\alpha)\, z\_1 + \alpha\, z\_2$ for $\alpha \in [0, 1]$
3. Decode: $\hat{x}\_\alpha = \text{decode}(z\_\alpha)$

Because the latent space is approximately Gaussian, linear interpolation produces plausible intermediate images -- smooth transitions between faces, between digits, between scenes.

### 5.4 Sampling Speed

ODE solvers can use adaptive step sizes, taking large steps when the dynamics are smooth and small steps when they change rapidly. In practice, the probability flow ODE can generate samples with 20-100 function evaluations (using RK45 or similar), compared to 1000 for the DDPM reverse chain. We will explore this in depth in Week 8.

---

## 6. The Score Matching Objective in Continuous Time

### 6.1 The Continuous-Time Loss

The training objective generalizes naturally to continuous time:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,T)}\, \mathbb{E}_{x(0) \sim p_0}\, \mathbb{E}_{x(t) \sim q(x(t) \mid x(0))}\!\left[\lambda(t)\, \left\Vert s_\theta(x(t), t) - \nabla_{x(t)} \log q(x(t) \mid x(0))\right\Vert ^2\right]
$$

where $\lambda(t)$ is a positive weighting function and $q(x(t) \mid x(0))$ is the transition kernel of the forward SDE (which is Gaussian for both VP and VE).

For the VP-SDE:

$$
\nabla_{x(t)} \log q(x(t) \mid x(0)) = -\frac{x(t) - \sqrt{\bar{\alpha}(t)}\, x(0)}{1 - \bar{\alpha}(t)} = -\frac{\varepsilon}{\sqrt{1 - \bar{\alpha}(t)}}
$$

This is continuous-time denoising score matching. The discrete-time DDPM and NCSN losses are obtained by discretizing the integral over $t$.

### 6.2 The Weighting Function $\lambda(t)$

The choice of $\lambda(t)$ affects training dynamics:

- **Likelihood weighting**: $\lambda(t) = g(t)^2$. This gives a loss that, when minimized, tightens the continuous-time ELBO. Proved by Song et al. (2021).
- **Uniform weighting** (DDPM-style): $\lambda(t) = 1$. Empirically gives better sample quality (lower FID).
- **SNR weighting**: $\lambda(t) = -d\text{SNR}(t)/dt$. Related to the min-SNR strategy.

The tension between likelihood and sample quality weighting is a fundamental trade-off. Models trained with likelihood weighting achieve better bits-per-dimension; models trained with uniform weighting produce prettier pictures.

---

## 7. The Hierarchy of Diffusion Models

### 7.1 The Unified View

We can now see the full hierarchy:

```
                        General SDE Framework
                      dx = f(x,t)dt + g(t)dW
                       /                   \
                      /                     \
              VP-SDE                     VE-SDE
    f = -β(t)x/2, g = √β(t)     f = 0, g = √(dσ²/dt)
           |                            |
    (discretize)                 (discretize)
           |                            |
         DDPM                         NCSN
    (Ho et al. 2020)         (Song & Ermon 2019)
```

The SDE framework contains both DDPM and NCSN as special cases. But it also enables:

- **New SDEs**: you can design novel forward processes with different drift and diffusion coefficients, potentially suited to specific data domains.
- **New solvers**: instead of the hand-derived DDPM reverse step or NCSN Langevin dynamics, you can use *any* SDE or ODE solver.
- **The probability flow ODE**: a deterministic counterpart that enables exact likelihoods, deterministic encoding, and fast adaptive-step sampling.

### 7.2 Reverse Process Options

Given a trained score network $s\_\theta(x, t)$, you have multiple ways to generate samples:

1. **Reverse SDE** (Euler-Maruyama): discretize the reverse SDE with fixed step sizes. This is what DDPM and NCSN do.

2. **Reverse SDE** (higher-order): use Milstein, Heun, or other higher-order SDE solvers for better accuracy per step.

3. **Probability flow ODE** (fixed step): solve the ODE with Euler or Heun method. Deterministic.

4. **Probability flow ODE** (adaptive step): solve with RK45 or Dormand-Prince. Adaptive step sizes give the same accuracy with fewer function evaluations.

5. **Predictor-corrector methods**: alternate between one step of an SDE/ODE solver ("predictor") and several Langevin correction steps ("corrector"). Song et al. (2021) show this improves sample quality.

All of these use the *same* trained score network. The training is decoupled from the sampling algorithm -- a powerful separation of concerns.

### 7.3 Predictor-Corrector Sampling

The predictor-corrector framework combines the best of both worlds:

**Predictor step**: Take one step of a numerical solver (e.g., one reverse Euler-Maruyama step) to advance from $x(t)$ to $x(t - \Delta t)$.

**Corrector step**: Run $M$ steps of Langevin dynamics at the new noise level $t - \Delta t$ to refine the sample:

$$
x \leftarrow x + \frac{\eta}{2}\, s_\theta(x, t - \Delta t) + \sqrt{\eta}\, z
$$

The predictor moves through noise levels; the corrector improves the sample quality at each level. This is a principled generalization of annealed Langevin dynamics that can be combined with any SDE or ODE solver.

---

## 8. Practical Implications

### 8.1 Choosing Between VP and VE

**VP-SDE (DDPM-style):**
- The forward process rescales data toward zero
- The prior is $\mathcal{N}(0, I)$ regardless of data scale
- Tends to produce slightly better FID scores
- More commonly used in practice

**VE-SDE (NCSN-style):**
- The forward process only adds noise (no rescaling)
- The prior has very large variance ($\sigma\_T^2 \gg 1$)
- Conceptually simpler
- Can have numerical issues due to large variance range

In practice, most modern diffusion models use VP-SDE or close variants.

### 8.2 The Power of ODE Solvers

The probability flow ODE opened the door to fast sampling:

- **DDIM** (Song, Meng, Ermon 2020): can be viewed as a discretization of the probability flow ODE for the VP-SDE. Reduces sampling from 1000 to ~50 steps.
- **DPM-Solver** (Lu et al. 2022): custom high-order ODE solver exploiting the semi-linear structure of the probability flow ODE. 10-20 steps for high-quality samples.
- **Adaptive methods**: off-the-shelf ODE solvers with error control, automatically choosing step sizes.

We will cover these in detail in Week 8.

### 8.3 Controllable Generation

The SDE framework makes it easy to modify the generation process for conditional generation, inpainting, and other controlled tasks. The key idea: modify the score function during sampling without retraining. For example:

- **Classifier guidance**: $\tilde{s}(x, t) = s\_\theta(x, t) + w\, \nabla\_x \log p\_\phi(y \mid x)$
- **Inpainting**: replace known pixels at each step and denoise only the unknown region
- **Inverse problems**: add a likelihood term to the score

These all operate at the level of the score function within the SDE/ODE framework. We will develop this in Week 10.

---

## 9. Continuous-Time ELBO

### 9.1 The Variational Bound

Song et al. (2021) show that the continuous-time ELBO for the VP-SDE is:

$$
\log p_0(x) \geq \mathbb{E}\!\left[-\frac{1}{2}\int_0^T \beta(t)\left[\Vert s_\theta(x(t), t)\Vert ^2 + 2\,\nabla \cdot s_\theta(x(t), t) + \Vert \nabla_{x(t)} \log q(x(t) \mid x(0))\Vert ^2\right] dt\right] + C
$$

where $C$ depends on the prior and entropy terms.

This simplifies (after applying denoising score matching and dropping constants) to:

$$
-\text{ELBO} \propto \int_0^T g(t)^2\, \mathbb{E}\!\left[\Vert s_\theta(x(t), t) - \nabla_{x(t)} \log q(x(t) \mid x(0))\Vert ^2\right] dt
$$

This is exactly the continuous-time score matching loss with $\lambda(t) = g(t)^2$. Thus, the likelihood-weighted score matching objective *is* the continuous-time ELBO.

---

## Summary

1. **The SDE framework** describes the forward noising process as $dx = f(x,t)\,dt + g(t)\,dW$ on $t \in [0, T]$. DDPM corresponds to VP-SDE ($f = -\frac{1}{2}\beta(t)x$, $g = \sqrt{\beta(t)}$). NCSN corresponds to VE-SDE ($f = 0$, $g = \sqrt{d\sigma^2/dt}$).

2. **The reverse-time SDE** is $dx = [f(x,t) - g(t)^2\nabla\_x \log p\_t(x)]\,dt + g(t)\,d\bar{W}$. It depends on the score function $\nabla\_x \log p\_t(x)$, which we approximate with a neural network.

3. **The probability flow ODE** is the deterministic counterpart: $dx = [f(x,t) - \frac{1}{2}g(t)^2\nabla\_x \log p\_t(x)]\,dt$. It has the same marginal distributions as the SDE but enables exact likelihoods, deterministic encoding, and fast adaptive-step sampling.

4. **The continuous-time score matching objective** unifies the DDPM and NCSN losses. With likelihood weighting ($\lambda = g^2$), it equals the continuous-time ELBO.

5. **The practical payoff**: training is decoupled from sampling. Train one score network; sample with any SDE or ODE solver. This separation enables rapid innovation in sampling algorithms.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| General forward SDE | $dx = f(x,t)\,dt + g(t)\,dW$ |
| VP-SDE | $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dW$ |
| VE-SDE | $dx = \sqrt{d\sigma^2(t)/dt}\;dW$ |
| Reverse SDE | $dx = [f - g^2\nabla\_x\log p\_t]\,dt + g\,d\bar{W}$ |
| Probability flow ODE | $dx = [f - \frac{1}{2}g^2\nabla\_x\log p\_t]\,dt$ |
| Instantaneous change of variables | $\log p\_0(x\_0) = \log p\_T(x\_T) + \int\_0^T \nabla\cdot\tilde{f}\,dt$ |
| Score matching loss | $\mathcal{L} = \mathbb{E}\_{t,x\_0,x\_t}[\lambda(t)\Vert s\_\theta - \nabla\_{x\_t}\log q(x\_t\mid x\_0)\Vert ^2]$ |

---

## Suggested Reading

- **Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole** (2021), "Score-Based Generative Modeling through Stochastic Differential Equations" -- the paper of this week. Read at least Sections 1-4 and the appendix on VP/VE derivations.
- **Anderson** (1982), "Reverse-time diffusion equation models" -- the mathematical result that makes it all work. Short and elegant.
- **Chen, Rubanova, Bettencourt, Duvenaud** (2018), "Neural Ordinary Differential Equations" -- the neural ODE framework that connects to the probability flow ODE.
- **Song, Meng, Ermon** (2020), "Denoising Diffusion Implicit Models (DDIM)" -- an early (pre-SDE) derivation of the probability flow ODE for DDPM, disguised as a non-Markovian forward process.
- **Karras, Aittala, Aila, Laine** (2022), "Elucidating the Design Space of Diffusion-Based Generative Models" -- a clarifying paper that re-derives everything in a clean notation and systematically explores design choices. Highly recommended.
