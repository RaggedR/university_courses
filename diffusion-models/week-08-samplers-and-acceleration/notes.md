# Week 8: Samplers and Acceleration

> *"The purpose of computation is insight, not numbers."*
> -- Richard Hamming

---

## Overview

You now understand the theory of diffusion models from three perspectives: the DDPM forward/reverse process (Week 5), the score-based formulation (Week 6), and the SDE unification (Week 7). All of them share a common practical problem: **sampling is painfully slow**.

A DDPM requires 1000 forward passes through the denoising network to generate a single image. At 50ms per pass on a modern GPU, that is 50 seconds per image. This is not a theoretical concern -- it is the bottleneck that prevented diffusion models from practical deployment until the methods in this week's notes were developed.

The key insight is that sampling from a diffusion model is equivalent to solving an ordinary differential equation (the probability flow ODE from Week 7). Once we see it this way, we can bring the entire arsenal of numerical ODE solvers to bear: higher-order methods, adaptive step sizes, and semi-analytical integration. The result is a reduction from 1000 steps to 10-50 steps with minimal quality loss.

We will develop this progression: DDPM sampling (slow, stochastic) $\to$ DDIM (fewer steps, deterministic option) $\to$ probability flow ODE solvers (Euler, Heun) $\to$ DPM-Solver (exponential integrators, 10-20 steps) $\to$ predictor-corrector methods (best of both worlds).

### Prerequisites
- Week 5: DDPM forward/reverse process, the noise schedule $\bar{\alpha}_t$
- Week 6: Score functions, Langevin dynamics
- Week 7: The SDE framework, probability flow ODE

---

## 1. DDPM Sampling: The Baseline

### 1.1 Recap

Recall from Week 5 that DDPM generates samples by running the reverse process:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

where $z \sim \mathcal{N}(0, I)$ and $\sigma_t^2 = \frac{(1-\alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$ (the posterior variance).

This is a Markov chain: $x_T \to x_{T-1} \to \cdots \to x_1 \to x_0$. Each step requires:
1. A forward pass through $\epsilon_\theta$ (the neural network)
2. Scaling and adding noise

With $T = 1000$ steps, this means 1000 neural network evaluations per sample.

### 1.2 Why 1000 Steps?

The number of steps is not arbitrary. The reverse process is derived as the time-reversal of the forward process, which adds a small amount of noise at each step. The approximation that the reverse transition $q(x_{t-1} | x_t, x_0)$ is Gaussian is only accurate when the step size $\beta_t$ is small. Large steps violate this Gaussian assumption, causing the reverse chain to diverge from the true reverse process.

In other words: DDPM is a first-order Euler-Maruyama discretization of the reverse-time SDE (Week 7). First-order methods need small step sizes to be accurate. The path to acceleration is clear: use better numerical methods.

### 1.3 The Predicted $x_0$ Reparameterization

Before we proceed, note a useful reparameterization. Given $x_t$ and the network's noise prediction $\epsilon_\theta(x_t, t)$, we can estimate the clean image:

$$\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

This follows directly from the forward process $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$. The "predicted $x_0$" view will be central to everything that follows.

---

## 2. DDIM: Denoising Diffusion Implicit Models

### 2.1 The Key Insight

Song, Meng, and Ermon (2021) made a remarkable observation: the DDPM training objective (the $\epsilon$-prediction loss) does **not** depend on the specific forward process. It only depends on the marginals $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) I)$.

This means we are free to choose a *different* forward process -- one that has the same marginals but is non-Markovian -- and the same trained network $\epsilon_\theta$ will still be valid. By choosing the forward process cleverly, we can derive a reverse process that works with far fewer steps.

### 2.2 The DDIM Forward Process

DDIM defines a family of non-Markovian forward processes indexed by a parameter $\sigma_t$. The reverse (generative) update is:

$$\boxed{x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \underbrace{\hat{x}_0(x_t, t)}_{\text{predicted } x_0} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \underbrace{\epsilon_\theta(x_t, t)}_{\text{predicted direction}} + \sigma_t \cdot \underbrace{\epsilon_t}_{\text{random noise}}}$$

where $\epsilon_t \sim \mathcal{N}(0, I)$ and:

$$\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

Let us parse this formula carefully. The update constructs $x_{t-1}$ from three components:

1. **Predicted $x_0$**, scaled by $\sqrt{\bar{\alpha}_{t-1}}$: This is our best guess of the clean image, scaled to the noise level of time $t-1$.

2. **Direction pointing toward $x_t$**, scaled by $\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}$: This is the "predicted noise direction" -- it ensures $x_{t-1}$ has the correct amount of structured noise.

3. **Fresh random noise**, scaled by $\sigma_t$: This injects stochasticity.

### 2.3 The $\eta$ Parameter

The noise level $\sigma_t$ is parameterized by $\eta \in [0, 1]$:

$$\sigma_t(\eta) = \eta \cdot \sqrt{\frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)}} \cdot \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$$

Two important special cases:

- **$\eta = 1$**: Recovers the DDPM reverse process exactly. The noise level matches the posterior variance from Week 5.

- **$\eta = 0$**: All randomness vanishes. The sampling process becomes **fully deterministic** -- given $x_T$, the generated image $x_0$ is uniquely determined. This is the "pure DDIM" sampler.

The deterministic case ($\eta = 0$) is remarkable: it means the mapping from noise to images is a deterministic function, and the same latent $x_T$ always produces the same image. This enables interpolation in latent space, inversion (finding the latent for a given image), and consistent editing.

### 2.4 Subsampling the Timesteps

The critical practical advantage: DDIM can skip timesteps. Instead of running through all $T = 1000$ steps, we choose a subsequence $\tau_1 < \tau_2 < \cdots < \tau_S$ of $S \ll T$ timesteps and apply the DDIM update only at these times.

For example, with $S = 50$ and uniform spacing: $\tau = \{1, 21, 41, \ldots, 981\}$. The DDIM formula is applied with $(t, t-1)$ replaced by $(\tau_{i+1}, \tau_i)$, using the $\bar{\alpha}$ values at the subsequence times.

This works because the DDIM update does not assume small step sizes in the same way DDPM does. The "predicted $x_0$" reparameterization allows the model to make a global prediction of the clean image at each step, rather than taking a small local step. Even with 50 steps instead of 1000, DDIM produces high-quality samples.

### 2.5 Why DDIM Works: Connection to the Probability Flow ODE

The deterministic DDIM ($\eta = 0$) is not merely a heuristic -- it is a discretization of the **probability flow ODE** from Week 7.

Recall that the probability flow ODE is:

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$

For the variance-preserving SDE (which corresponds to DDPM), this becomes:

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t) \left[ x + \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x, t) \right] + \frac{1}{2}\beta(t) x$$

After simplification and discretization, this yields exactly the DDIM update with $\eta = 0$.

The connection is profound: DDIM is not an approximation to DDPM. It is an exact discretization of the same ODE that DDPM samples from (approximately, via its SDE). The ODE and the SDE define the same marginal distributions $p_t(x)$, so they generate samples from the same distribution -- the ODE just does it deterministically.

---

## 3. ODE Solvers for Diffusion Sampling

### 3.1 The Probability Flow ODE, Restated

Once we recognize sampling as ODE solving, we can write:

$$\frac{dx}{dt} = v_\theta(x, t)$$

where $v_\theta$ is the velocity field derived from the score network. The goal: integrate from $t = T$ (noise) to $t = 0$ (data).

### 3.2 Euler's Method

The simplest ODE solver. Given current state $x_t$ at time $t$, step to time $s < t$:

$$x_s = x_t + (s - t) \cdot v_\theta(x_t, t)$$

This is a first-order method: the error per step is $O(h^2)$ where $h = |s - t|$, and the global error after $N$ steps is $O(h) = O(1/N)$.

DDIM with $\eta = 0$ is essentially Euler's method applied to the probability flow ODE (with a specific change of variables). So when we use 50-step DDIM, we are doing 50-step Euler integration.

### 3.3 Heun's Method (Improved Euler)

Heun's method is a second-order method that makes two function evaluations per step:

$$\begin{aligned}
\tilde{x}_s &= x_t + (s - t) \cdot v_\theta(x_t, t) & \text{(predictor: Euler step)} \\
x_s &= x_t + \frac{s - t}{2} \left[ v_\theta(x_t, t) + v_\theta(\tilde{x}_s, s) \right] & \text{(corrector: average slopes)}
\end{aligned}$$

The first evaluation gives a rough estimate $\tilde{x}_s$; the second evaluates the velocity at that estimate and averages the two slopes. The error per step is $O(h^3)$, and the global error is $O(h^2) = O(1/N^2)$.

Each step costs 2 network evaluations (NFEs), so with a budget of $N$ NFEs, Heun gives $N/2$ steps with $O(4/N^2)$ error, while Euler gives $N$ steps with $O(1/N)$ error. For $N \geq 5$, Heun wins.

### 3.4 Higher-Order Runge-Kutta

The classical RK4 method uses 4 evaluations per step:

$$\begin{aligned}
k_1 &= v_\theta(x_t, t) \\
k_2 &= v_\theta\!\left(x_t + \tfrac{h}{2} k_1,\; t + \tfrac{h}{2}\right) \\
k_3 &= v_\theta\!\left(x_t + \tfrac{h}{2} k_2,\; t + \tfrac{h}{2}\right) \\
k_4 &= v_\theta(x_t + h \cdot k_3,\; t + h) \\
x_{t+h} &= x_t + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}$$

This achieves $O(h^4)$ global error. However, 4 NFEs per step is expensive when each evaluation is a full U-Net forward pass. In practice, second-order methods (Heun, midpoint) often give the best quality-per-NFE for diffusion models.

---

## 4. DPM-Solver: Exploiting the Semi-Linear Structure

### 4.1 The Semi-Linear ODE

Lu et al. (2022) observed that the probability flow ODE has a special structure. Introducing $\lambda_t = \log(\bar{\alpha}_t / (1 - \bar{\alpha}_t))$ (the log signal-to-noise ratio) as the time variable, the ODE can be written as:

$$\frac{dx_\lambda}{d\lambda} = \frac{1}{2} x_\lambda - \frac{1}{2} e^{-\lambda} \hat{\epsilon}_\theta(x_\lambda, \lambda)$$

This is a **semi-linear ODE**: it has a linear part ($\frac{1}{2} x_\lambda$) and a nonlinear part (involving the neural network). The linear part has a known exact solution. Only the nonlinear part needs numerical approximation.

### 4.2 Exact Solution of the Linear Part

The linear ODE $\frac{dx}{d\lambda} = \frac{1}{2} x$ has the exact solution $x(\lambda_s) = e^{(\lambda_s - \lambda_t)/2} x(\lambda_t)$. Using the variation of constants formula, the full solution is:

$$x_{\lambda_s} = \frac{\alpha_s}{\alpha_t} x_{\lambda_t} - \alpha_s \int_{\lambda_t}^{\lambda_s} e^{-\lambda} \hat{\epsilon}_\theta(x_\lambda, \lambda) \, d\lambda$$

where $\alpha_t = \sqrt{\bar{\alpha}_t}$. The key observation: the exponential factor $\alpha_s / \alpha_t$ handles the linear part exactly. We only need to approximate the integral of the nonlinear part.

### 4.3 DPM-Solver-1 (First Order)

Approximate $\hat{\epsilon}_\theta(x_\lambda, \lambda) \approx \hat{\epsilon}_\theta(x_{\lambda_t}, \lambda_t)$ (constant over the interval):

$$x_{\lambda_s} = \frac{\alpha_s}{\alpha_t} x_{\lambda_t} - \sigma_s (e^{h} - 1) \hat{\epsilon}_\theta(x_{\lambda_t}, \lambda_t)$$

where $h = \lambda_s - \lambda_t$ and $\sigma_t = \sqrt{1 - \bar{\alpha}_t}$. This is equivalent to DDIM -- but the derivation via exponential integrators makes higher-order extensions natural.

### 4.4 DPM-Solver-2 (Second Order)

Use a linear approximation of $\hat{\epsilon}_\theta$ over the interval. This requires one additional evaluation at the midpoint:

$$\begin{aligned}
u &= \frac{\alpha_{s_{1/2}}}{\alpha_t} x_{\lambda_t} - \sigma_{s_{1/2}} (e^{h/2} - 1) \hat{\epsilon}_\theta(x_{\lambda_t}, \lambda_t) \\
x_{\lambda_s} &= \frac{\alpha_s}{\alpha_t} x_{\lambda_t} - \sigma_s (e^h - 1) \hat{\epsilon}_\theta(x_{\lambda_t}, \lambda_t) - \frac{\sigma_s}{2h}(e^h - 1) \left[\hat{\epsilon}_\theta(u, \lambda_{s_{1/2}}) - \hat{\epsilon}_\theta(x_{\lambda_t}, \lambda_t)\right]
\end{aligned}$$

This achieves second-order accuracy with 2 NFEs per step, but with better constants than Heun because the linear part is handled exactly.

### 4.5 DPM-Solver++: Predicting $x_0$ Instead of $\epsilon$

DPM-Solver++ (Lu et al., 2022) reformulates the semi-linear ODE in terms of the predicted $x_0$ (the "data prediction" parameterization) rather than $\epsilon$ (the "noise prediction"):

$$x_{\lambda_s} = \frac{\sigma_s}{\sigma_t} x_{\lambda_t} - \alpha_s (e^{-h} - 1) \hat{x}_0(x_{\lambda_t}, \lambda_t)$$

where $\hat{x}_0(x_t, t) = (x_t - \sigma_t \epsilon_\theta(x_t, t)) / \alpha_t$.

Why is this better? At high noise levels (early in sampling), $\epsilon_\theta$ predicts large, volatile noise vectors. The predicted $x_0$, by contrast, is a direct estimate of the clean image and tends to be smoother and more stable. DPM-Solver++ with the $x_0$ parameterization consistently outperforms the $\epsilon$ parameterization, especially with very few steps (10-20).

### 4.6 Multistep Variants

DPM-Solver can also be extended using **multistep** methods: reuse $\hat{\epsilon}_\theta$ evaluations from previous steps to construct higher-order approximations without additional NFEs. For example, DPM-Solver-2 (multistep) uses the current and previous $\hat{\epsilon}_\theta$ values to construct a linear interpolation, achieving second-order accuracy with only 1 NFE per step.

This is analogous to Adams-Bashforth methods in classical ODE theory. The trade-off: multistep methods require a startup phase (the first step must use a single-step method) and can be less stable than single-step methods.

---

## 5. Predictor-Corrector Methods

### 5.1 The Idea

The SDE and ODE formulations suggest a natural hybrid strategy:

1. **Predict** using the ODE (deterministic, efficient)
2. **Correct** using Langevin dynamics (stochastic, refines the sample)

The ODE quickly moves toward the data distribution, but may not explore it fully. A few Langevin correction steps add stochastic exploration, improving sample quality.

### 5.2 The Algorithm

Starting from $x_T \sim \mathcal{N}(0, I)$:

```
For t = T, T-1, ..., 1:
    # Predictor: one ODE step (e.g., DDIM or DPM-Solver)
    x̃_{t-1} = ODE_step(x_t, t, t-1)

    # Corrector: M steps of Langevin MCMC
    for m = 1, ..., M:
        x̃_{t-1} = x̃_{t-1} + δ · ∇_x log p_{t-1}(x̃_{t-1}) + √(2δ) · z
        where z ~ N(0, I) and δ is the Langevin step size
```

The score $\nabla_x \log p_{t-1}(x)$ is approximated by $-\epsilon_\theta(x, t-1) / \sqrt{1 - \bar{\alpha}_{t-1}}$ (from Week 6).

### 5.3 When to Use Predictor-Corrector

Predictor-corrector methods are most useful when:
- You have a computational budget that allows some extra NFEs beyond the minimum
- Sample quality matters more than speed
- You are generating at high resolution where small artifacts are visible

Song et al. (2021) showed that predictor-corrector methods achieve the best FID scores among SDE-based samplers. However, the corrector steps add NFEs (typically 1-5 per predictor step), so the total cost is higher than pure ODE solvers.

---

## 6. Adaptive Step Sizes

### 6.1 The Problem with Uniform Steps

In all the methods above, we implicitly assumed uniformly spaced timesteps. But the dynamics of the diffusion process are not uniform: at high noise levels ($t$ near $T$), the signal changes slowly (it is mostly noise); at low noise levels ($t$ near 0), fine details are being resolved and the signal changes rapidly.

### 6.2 The Log-SNR Schedule

A natural alternative is to space steps uniformly in the log signal-to-noise ratio $\lambda_t = \log(\bar{\alpha}_t / (1 - \bar{\alpha}_t))$. This concentrates steps at the transitions where the SNR changes most rapidly -- typically in the middle of the schedule and near the end (where the image is nearly clean).

DPM-Solver uses this by default, which is one reason for its effectiveness.

### 6.3 Error-Based Adaptation

More sophisticated approaches estimate the local truncation error and adjust the step size to keep it below a tolerance. Given an estimate $\hat{x}_s$ from a $p$-th order method and $\hat{x}_s'$ from a $(p+1)$-th order method, the local error estimate is:

$$\text{err} = \|\hat{x}_s - \hat{x}_s'\|$$

If $\text{err} > \text{tol}$, reject the step and retry with a smaller $h$. If $\text{err} \ll \text{tol}$, accept and increase $h$ for the next step. The optimal step size update is:

$$h_{\text{new}} = h \cdot \left(\frac{\text{tol}}{\text{err}}\right)^{1/(p+1)}$$

This is the principle behind adaptive ODE solvers like Dormand-Prince (dopri5), which have been applied to diffusion sampling with good results.

---

## 7. Comparison: Quality vs. Speed

### 7.1 The NFE Budget

The key metric is **number of function evaluations (NFEs)** -- how many times we evaluate the denoising network $\epsilon_\theta$. This directly determines the wall-clock time.

| Method | NFEs | Order | Quality (FID) |
|--------|------|-------|---------------|
| DDPM | 1000 | 1 | Baseline |
| DDIM (uniform) | 50 | 1 | +2-5 FID |
| DDIM (uniform) | 10 | 1 | +10-20 FID |
| Heun (2nd order) | 50 (25 steps) | 2 | +1-3 FID |
| DPM-Solver-2 | 20 | 2 | +1-2 FID |
| DPM-Solver++ | 10-15 | 2-3 | +1-3 FID |
| PC (Euler + Langevin) | 100-200 | 1 | -0.5-1 FID (improves) |

*FID numbers are approximate and dataset-dependent. Lower FID is better. Differences shown relative to DDPM baseline.*

### 7.2 The Quality Cliff

All fast samplers exhibit a **quality cliff**: below some minimum NFE count, quality degrades catastrophically. For DDIM, this cliff is around 10-20 steps. For DPM-Solver++, it is around 5-10 steps.

The cliff occurs because with too few steps, the ODE solver cannot track the rapidly changing velocity field. The result is blurry, distorted, or mode-collapsed samples.

### 7.3 The $\eta$ Spectrum

For DDIM specifically, the choice of $\eta$ (stochasticity) interacts with the step count:
- With many steps (50+), $\eta \approx 1$ (DDPM-like) and $\eta = 0$ (deterministic) give similar quality
- With few steps (10-20), $\eta = 0$ (deterministic) is markedly better -- the deterministic trajectory is smoother and more tolerant of discretization error
- For some models, a small $\eta \approx 0.2$ provides the best trade-off: a touch of stochasticity to improve diversity without destabilizing the trajectory

### 7.4 Practical Recommendations

For typical use cases:
- **Interactive applications (real-time):** DPM-Solver++ with 10-20 steps
- **High-quality generation:** DPM-Solver++ with 20-50 steps, or predictor-corrector with 50+ NFEs
- **Exact reproduction:** DDIM ($\eta = 0$) for deterministic generation
- **Maximum quality, no time constraint:** DDPM with full 1000 steps, or predictor-corrector with many corrector steps

---

## 8. Beyond These Methods: A Brief Look Ahead

### 8.1 Distillation (Week 12)

The samplers in this week accelerate inference without retraining the model. An orthogonal approach is **distillation**: train a student model that produces the same output in fewer steps. Progressive distillation (Salimans & Ho, 2022) halves the step count repeatedly, achieving 4-step generation. Consistency models (Song et al., 2023) push this to a single step. We will study these in Week 12.

### 8.2 Flow Matching (Week 11)

Flow matching (Lipman et al., 2023) learns a velocity field directly via a regression loss, bypassing the score function entirely. The resulting ODE tends to have straighter trajectories, making it easier to solve in few steps. This is the foundation of recent models like Stable Diffusion 3 and Flux.

---

## Summary

1. **DDPM sampling** requires 1000 network evaluations, each computing $\epsilon_\theta(x_t, t)$. The bottleneck is that DDPM is a first-order Euler-Maruyama discretization of the reverse SDE.

2. **DDIM** reinterprets the sampling process as a non-Markovian reverse process. With $\eta = 0$, it is deterministic and equivalent to Euler integration of the probability flow ODE. It works well with 50 steps and tolerably with 20.

3. **Higher-order ODE solvers** (Heun, RK4) reduce the discretization error per step. Second-order methods typically give the best quality-per-NFE.

4. **DPM-Solver** exploits the semi-linear structure of the probability flow ODE. By solving the linear part exactly and approximating only the nonlinear part, it achieves better accuracy than generic ODE solvers. DPM-Solver++ uses the $x_0$ parameterization for improved stability.

5. **Predictor-corrector methods** combine ODE prediction with Langevin correction steps. They achieve the best sample quality but at higher computational cost.

6. **Adaptive step sizes** concentrate computational effort where the dynamics change fastest, improving efficiency.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| DDPM reverse step | $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t,t)\right) + \sigma_t z$ |
| Predicted $x_0$ | $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}$ |
| DDIM update | $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t,t) + \sigma_t \epsilon$ |
| Euler step | $x_s = x_t + (s-t) \cdot v_\theta(x_t, t)$ |
| Heun step | $x_s = x_t + \frac{s-t}{2}[v_\theta(x_t,t) + v_\theta(\tilde{x}_s, s)]$ |
| DPM-Solver-1 | $x_s = \frac{\alpha_s}{\alpha_t} x_t - \sigma_s(e^h - 1)\hat{\epsilon}_\theta(x_t, t)$ |
| DPM-Solver++ (1st order) | $x_s = \frac{\sigma_s}{\sigma_t} x_t - \alpha_s(e^{-h}-1)\hat{x}_0(x_t, t)$ |

---

## Suggested Reading

- **Song, Meng, and Ermon** (2021), "Denoising Diffusion Implicit Models" -- the DDIM paper. Read Sections 3-4 carefully for the derivation of the non-Markovian forward process.
- **Lu et al.** (2022), "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps" -- the DPM-Solver paper. The key ideas are in Sections 3-4.
- **Lu et al.** (2022), "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models" -- the shift to $x_0$ prediction.
- **Song et al.** (2021), "Score-Based Generative Modeling through Stochastic Differential Equations" -- Section 4 on predictor-corrector methods.
- **Karras et al.** (2022), "Elucidating the Design Space of Diffusion-Based Generative Models" -- an excellent analysis of sampler design choices, including the Heun solver. Highly recommended.
