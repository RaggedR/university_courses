---
title: "Week 12: Distillation and Fast Generation"
---

# Week 12: Distillation and Fast Generation

> *"Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away."*
> -- Antoine de Saint-Exupery

---

## Overview

We have now built diffusion models (Weeks 5-7), learned to sample them efficiently (Week 8), and studied flow matching as a path to straighter, faster-to-sample generative processes (Week 11). But even with the best ODE solvers and the straightest paths, practical generation still requires 10-20 neural network evaluations. For real-time applications -- interactive image editing, video generation, game asset creation -- even 10 steps at 100ms each is too slow.

This week, we study the methods that push generation down to 1-4 steps while maintaining (most of) the quality. The core idea is **distillation**: training a fast *student* model to imitate a slow *teacher* model. The teacher's knowledge, accumulated over hundreds of sampling steps, is compressed into a student that reproduces the teacher's outputs in a fraction of the time.

We will cover three families of distillation: **progressive distillation** (iteratively halving the step count), **consistency models** (exploiting the ODE structure directly), and **adversarial distillation** (using a discriminator to maintain perceptual quality). We close with **score distillation sampling**, which uses diffusion models not for generation but as a loss function for optimizing other representations.

### Prerequisites
- Week 5: DDPM training and sampling
- Week 7: Probability flow ODE
- Week 8: ODE solvers (DDIM, DPM-Solver)
- Week 11: Flow matching and velocity prediction

---

## 1. The Speed Problem

### 1.1 Why Sampling Is Slow

A diffusion model (or flow matching model) generates a sample by solving an ODE or SDE from noise to data. Each step requires one evaluation of the neural network $v\_\theta(x\_t, t)$ -- a forward pass through a U-Net or transformer with hundreds of millions of parameters.

For a high-resolution image ($1024 \times 1024$), one forward pass through a DiT-XL model takes approximately 100-500ms on a modern GPU. With 20 sampling steps, that is 2-10 seconds per image. For video generation (e.g., 60 frames at $1024 \times 1024$), the situation is far worse.

**The quality-speed tradeoff**: Reducing the number of steps directly degrades quality. With too few steps, the ODE solver cannot follow the curved trajectory accurately, producing blurry or distorted outputs. The solver error grows as $O(\Delta t^p)$ for a $p$-th order solver (Euler is $p = 1$), and with only $N$ steps, $\Delta t = 1/N$.

### 1.2 The Approaches

There are four main strategies to reduce the step count:

1. **Better ODE solvers** (Week 8): Use higher-order solvers (DDIM, DPM-Solver) to get more accuracy per step. Diminishing returns below ~10 steps.

2. **Straighter paths** (Week 11): Flow matching and rectified flows give straighter trajectories, requiring fewer solver steps. Helps but does not solve the problem completely.

3. **Distillation**: Train a student model that produces in $N/2$ steps what the teacher produces in $N$ steps. This is the focus of this week.

4. **Architecture optimization**: Smaller models, quantization, caching. Orthogonal to distillation.

---

## 2. Progressive Distillation

### 2.1 The Key Idea

Progressive distillation (Salimans and Ho, 2022) is conceptually beautiful: train a student model to take one step where the teacher takes two steps. Then use the student as the new teacher and repeat. Each round halves the number of steps:

$$
N \to N/2 \to N/4 \to \cdots \to 1
$$

Starting from a teacher that uses 1024 steps, after 10 rounds of distillation we reach a 1-step student.

### 2.2 The Distillation Step

Let the teacher model be $v\_T(x, t)$ and the student model be $v\_S(x, t)$, initialized as a copy of the teacher. Consider a time interval $[t, t + 2\Delta t]$ where the teacher takes two steps.

**Teacher's two steps:**
$$
x_{t+\Delta t} = x_t + \Delta t \cdot v_T(x_t, t)
$$
$$
x_{t+2\Delta t} = x_{t+\Delta t} + \Delta t \cdot v_T(x_{t+\Delta t}, t + \Delta t)
$$

The teacher maps $x\_t$ to $x\_{t+2\Delta t}$ using two network evaluations.

**Student's one step:**
$$
\hat{x}_{t+2\Delta t} = x_t + 2\Delta t \cdot v_S(x_t, t)
$$

The student maps $x\_t$ to $\hat{x}\_{t+2\Delta t}$ using one network evaluation.

**Distillation loss:**
$$
\mathcal{L}_{\text{PD}} = \mathbb{E}_{t, x_t}\left[\Vert x_{t+2\Delta t}^{\text{teacher}} - x_{t+2\Delta t}^{\text{student}}\Vert ^2\right]
$$

The student learns to match the teacher's two-step output in a single step.

### 2.3 The Full Algorithm

```
Input: Teacher model v_T with N sampling steps
Initialize: v_S = copy(v_T), current_steps = N

while current_steps > target_steps:
    dt = 1.0 / current_steps
    for each training iteration:
        1. Sample t uniformly from {0, dt, 2*dt, ..., 1 - 2*dt}
        2. Sample x_0 ~ N(0, I), x_1 ~ p_data
        3. Compute x_t on the ODE trajectory (using the teacher)
        4. Teacher: two steps from x_t → x_{t+2dt}
        5. Student: one step from x_t → x_{t+2dt}  (with step size 2*dt)
        6. Loss = ||teacher_output - student_output||^2
        7. Update v_S

    v_T = v_S  (student becomes the new teacher)
    current_steps = current_steps / 2
```

### 2.4 What the Student Learns

At each round, the student must learn a velocity field that "looks ahead" twice as far as the teacher. After $k$ rounds:

- Round 0 (teacher): $v\_T$ is accurate over intervals of length $\Delta t = 1/N$
- Round 1: $v\_S^{(1)}$ is accurate over intervals of length $2/N$
- Round 2: $v\_S^{(2)}$ is accurate over intervals of length $4/N$
- Round $k$: $v\_S^{(k)}$ is accurate over intervals of length $2^k/N$

As $k$ increases, the student must predict the *curvature* of the trajectory over larger intervals. This is a harder prediction problem, which is why quality degrades with each round. The remarkable finding of Salimans and Ho is that the degradation is mild for the first several rounds, with noticeable quality loss only at very low step counts (1-2 steps).

### 2.5 The $v$-Prediction Parameterization

Salimans and Ho found that the **velocity prediction** ($v$-prediction) parameterization works much better for progressive distillation than noise prediction ($\epsilon$-prediction).

For a diffusion process $x\_t = \alpha\_t x\_1 + \sigma\_t \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$:

$$
v_t = \alpha_t \epsilon - \sigma_t x_1 = \frac{d x_t}{d \log \text{SNR}_t}
$$

The network predicts $v\_\theta(x\_t, t) \approx v\_t$, from which we can recover both $\epsilon$ and $x\_1$:

$$
\hat{x}_1 = \alpha_t x_t - \sigma_t v_\theta, \quad \hat{\epsilon} = \sigma_t x_t + \alpha_t v_\theta
$$

This is the same velocity prediction used in flow matching (Week 11), and it is no coincidence that it works well for distillation -- the velocity parameterization is better conditioned for predicting the direction of the trajectory.

---

## 3. Consistency Models

### 3.1 The Self-Consistency Property

Consistency models (Song et al., 2023) take a different approach, exploiting a beautiful property of ODE trajectories.

Consider the probability flow ODE trajectory $\lbrace x\_t\rbrace \_{t \in [0, T]}$ of a diffusion model. Every point on this trajectory maps to the same data point when we solve the ODE from $t$ to $t = 0$ (or, more precisely, to a small $t\_{\min} > 0$ for numerical stability):

$$
x_{t_{\min}} = \text{ODE-Solve}(x_t, t \to t_{\min})
$$

The **consistency function** $f : (x\_t, t) \to x\_{t\_{\min}}$ maps any noisy point to the trajectory's origin. The defining property is **self-consistency**:

$$
f(x_t, t) = f(x_{t'}, t') \quad \text{for all } t, t' \text{ on the same trajectory}
$$

If we can learn $f\_\theta$ such that this property holds, we get a one-step generator: sample $x\_T \sim \mathcal{N}(0, T^2 I)$, then compute $f\_\theta(x\_T, T) \approx x\_0$.

### 3.2 The Boundary Condition

At $t = t\_{\min}$, the consistency function must be the identity (a point at the origin of the trajectory maps to itself):

$$
f(x_{t_{\min}}, t_{\min}) = x_{t_{\min}}
$$

Song et al. enforce this by parameterizing $f\_\theta$ as:

$$
f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta(x, t)
$$

where $c\_{\text{skip}}(t\_{\min}) = 1$ and $c\_{\text{out}}(t\_{\min}) = 0$, so that $f\_\theta(x, t\_{\min}) = x$ regardless of $F\_\theta$. The functions $c\_{\text{skip}}$ and $c\_{\text{out}}$ interpolate smoothly from this boundary condition.

### 3.3 Consistency Distillation (CD)

Given a pre-trained diffusion model with score function $s\_\phi(x, t)$, we can train a consistency model by enforcing self-consistency along the teacher's ODE trajectories.

**Training procedure:**

1. Sample $x\_0 \sim p\_{\text{data}}$, $t \sim U[t\_{\min}, T]$
2. Add noise: $x\_t = \alpha\_t x\_0 + \sigma\_t \epsilon$
3. Take one ODE step backward: $x\_{t-\Delta t} = \text{ODE-step}(x\_t, t \to t - \Delta t)$ using the teacher $s\_\phi$
4. Consistency loss: $\mathcal{L}\_{\text{CD}} = d(f\_\theta(x\_t, t), f\_{\theta^-}(x\_{t-\Delta t}, t - \Delta t))$

Here $\theta^-$ is an exponential moving average (EMA) of $\theta$ (a target network, as in DQN or BYOL), and $d(\cdot, \cdot)$ is a distance metric (e.g., L2, LPIPS, or pseudo-Huber loss).

The idea: two points on the same ODE trajectory should map to the same origin. The teacher tells us which points are on the same trajectory (by providing the ODE step), and we train $f\_\theta$ to agree on these pairs.

### 3.4 Consistency Training (CT)

A remarkable feature of consistency models is that they can be trained **from scratch** without a pre-trained diffusion model.

The key observation: instead of using the teacher's ODE to find pairs of points on the same trajectory, use the *consistency model's own* ODE (implicitly defined by $f\_\theta$). The loss becomes:

$$
\mathcal{L}_{\text{CT}} = \mathbb{E}\left[d\left(f_\theta(x_{t_{n+1}}, t_{n+1}), f_{\theta^-}(x_{t_n}, t_n)\right)\right]
$$

where $(x\_{t\_n}, x\_{t\_{n+1}})$ are adjacent noisy versions of the same data point:

$$
x_{t_n} = \alpha_{t_n} x_0 + \sigma_{t_n} \epsilon, \quad x_{t_{n+1}} = \alpha_{t_{n+1}} x_0 + \sigma_{t_{n+1}} \epsilon
$$

using the *same* noise $\epsilon$. This ensures they lie on (approximately) the same trajectory, even without a teacher model.

The schedule $\lbrace t\_n\rbrace$ starts coarse (few time steps) and becomes finer during training. Song et al. showed that as $\Delta t \to 0$, the CT objective converges to enforcing exact self-consistency.

### 3.5 Multi-Step Sampling with Consistency Models

While consistency models enable one-step generation, they can also do multi-step generation for higher quality. The procedure alternates between denoising and re-noising:

1. Start with $x\_T \sim \mathcal{N}(0, T^2 I)$
2. Denoise: $\hat{x}\_0 = f\_\theta(x\_T, T)$
3. Re-noise to an intermediate level: $x\_{t\_1} = \alpha\_{t\_1} \hat{x}\_0 + \sigma\_{t\_1} \epsilon$
4. Denoise again: $\hat{x}\_0 = f\_\theta(x\_{t\_1}, t\_1)$
5. Repeat for desired number of steps

Each denoise-renoise cycle refines the sample. With $k$ steps, the quality approaches the teacher's quality. This gives a smooth quality-speed tradeoff.

---

## 4. Latent Consistency Models

### 4.1 Applying Consistency to Latent Diffusion

Latent consistency models (LCMs, Luo et al., 2023) apply the consistency model framework to **latent diffusion models** (Week 9). The key modifications:

1. **Work in latent space**: The consistency function operates on the VAE latent $z\_t$, not the pixel-space image $x\_t$. This is computationally cheaper (the latent space is $64 \times 64 \times 4$ vs. $512 \times 512 \times 3$).

2. **Augmented probability flow ODE**: To incorporate classifier-free guidance during distillation, Luo et al. define a guided ODE:

$$
\frac{dz}{dt} = (1 + w) \cdot \epsilon_\theta(z_t, t, c) - w \cdot \epsilon_\theta(z_t, t, \varnothing)
$$

where $w$ is the guidance scale. The consistency model is trained to be self-consistent along trajectories of this *guided* ODE.

3. **Skipping steps in the teacher**: Instead of single ODE steps, use the teacher to skip multiple steps ($k$-step DDIM), which provides more accurate trajectory information.

### 4.2 LCM-LoRA

A particularly practical variant: **LCM-LoRA** (Luo et al., 2023). Instead of distilling the full model, train a LoRA adapter (low-rank adaptation) that converts any fine-tuned Stable Diffusion model into a fast, few-step generator.

The LoRA adapters are small ($\sim$70MB vs. the base model's $\sim$3GB) and can be combined with other LoRA adapters for style, subject, etc. This makes LCM distillation a plug-and-play acceleration method.

### 4.3 Practical Results

LCM results on Stable Diffusion v1.5 and SDXL:

| Steps | FID (SD v1.5) | Inference Time |
|-------|---------------|----------------|
| 50 (teacher) | 8.7 | 5.2s |
| 4 (LCM) | 9.2 | 0.4s |
| 2 (LCM) | 12.1 | 0.2s |
| 1 (LCM) | 18.5 | 0.1s |

The quality at 4 steps is nearly indistinguishable from the 50-step teacher, at 13x the speed.

---

## 5. Adversarial Distillation

### 5.1 The Blurriness Problem

Both progressive distillation and consistency models minimize a regression loss (L2 or similar). A well-known problem with regression losses: they produce blurry outputs when the target is multimodal.

Consider a point $x\_t$ at an intermediate noise level that could plausibly resolve to either of two data points $x\_1^{(a)}$ or $x\_1^{(b)}$. The L2-optimal prediction is the mean $\frac{1}{2}(x\_1^{(a)} + x\_1^{(b)})$, which looks like neither. This is the origin of the blurriness seen in 1-2 step distilled models.

### 5.2 Adding a Discriminator

Adversarial distillation (Sauer et al., 2023, 2025) addresses this by adding a GAN-style discriminator loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{distill}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}
$$

The distillation loss $\mathcal{L}\_{\text{distill}}$ ensures the student follows the teacher's trajectory. The adversarial loss $\mathcal{L}\_{\text{adv}}$ ensures the student's outputs are sharp and realistic, even if they deviate slightly from the teacher.

The discriminator $D\_\psi$ is trained to distinguish between:
- Real images from the dataset
- Fake images generated by the student in 1-4 steps

The student is trained to fool the discriminator (standard GAN min-max game) while also matching the teacher's outputs.

### 5.3 SDXL-Turbo and SDXL-Lightning

**SDXL-Turbo** (Sauer et al., 2023) combines adversarial distillation with score distillation:

$$
\mathcal{L} = \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{recon}} \mathcal{L}_{\text{recon}}
$$

where $\mathcal{L}\_{\text{recon}}$ is a pixel-space reconstruction loss computed using the teacher model. The result: high-quality 1-step generation from SDXL.

**SDXL-Lightning** (Lin et al., 2024) uses progressive distillation combined with adversarial loss, achieving state-of-the-art quality at 1-4 steps. The key insight is applying the adversarial loss at every stage of progressive distillation, not just the final stage.

### 5.4 The Quality-Speed Frontier

As of early 2025, the quality-speed frontier looks roughly like this:

| Steps | Method | Quality (FID) | Speed (A100) |
|-------|--------|---------------|--------------|
| 1 | Adversarial distillation | Good (FID ~15-20) | ~100ms |
| 2 | Adversarial distillation | Very good (FID ~10-12) | ~200ms |
| 4 | Consistency/LCM | Near-teacher (FID ~9-10) | ~400ms |
| 8 | Flow matching | Close to teacher | ~800ms |
| 20 | Teacher (flow matching) | Best (FID ~8-9) | ~2s |
| 50 | Teacher (DDPM) | Reference | ~5s |

The practical sweet spot depends on the application: 4 steps for interactive editing, 1 step for real-time applications, 20+ steps for maximum quality.

---

## 6. Score Distillation Sampling (SDS)

### 6.1 A Different Use of Diffusion Models

Score distillation sampling (Poole et al., 2023) uses a diffusion model in a fundamentally different way: not to generate images, but as a **loss function** for optimizing other representations.

The motivating application is text-to-3D generation. We want to create a 3D scene (parameterized by a NeRF or mesh) such that renderings from any viewpoint look like they match a text prompt. We have a text-to-image diffusion model that knows what "a corgi wearing a crown" looks like, but we need to transfer that knowledge to 3D.

### 6.2 The SDS Loss

Given a differentiable renderer $g(\theta, c)$ that renders the 3D representation $\theta$ from camera viewpoint $c$, the SDS loss is:

$$
\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon, c}\left[w(t) \left(\epsilon_\phi(x_t, t, y) - \epsilon\right) \frac{\partial g(\theta, c)}{\partial \theta}\right]
$$

where:
- $x\_t = \alpha\_t g(\theta, c) + \sigma\_t \epsilon$ is the rendered image with added noise
- $\epsilon\_\phi$ is the pre-trained diffusion model's noise prediction
- $y$ is the text prompt
- $w(t)$ is a weighting function

The intuition: render the current 3D scene, add noise, ask the diffusion model "what should this look like?", and update the 3D parameters to make the rendering look more like the diffusion model's prediction.

### 6.3 Deriving SDS

The SDS gradient can be derived from the KL divergence between the distribution of noisy rendered images and the diffusion model's learned distribution.

Consider the rendered image $x = g(\theta, c)$ as a delta distribution $q\_\theta(x) = \delta(x - g(\theta, c))$. We want to minimize:

$$
\mathcal{L}(\theta) = D_{\text{KL}}(q_t(x_t) \Vert p_t(x_t | y))
$$

where $q\_t$ is the distribution of noisy rendered images and $p\_t$ is the diffusion model's marginal at noise level $t$.

Taking the gradient and dropping terms that do not depend on $\theta$:

$$
\nabla_\theta \mathcal{L} \approx \mathbb{E}_{t, \epsilon}\left[w(t)(\epsilon_\phi(x_t, t, y) - \epsilon) \frac{\partial x}{\partial \theta}\right]
$$

This is the SDS gradient. The term $\epsilon\_\phi(x\_t, t, y) - \epsilon$ is the "denoising direction" -- the direction the diffusion model thinks the image should move to look more realistic and text-consistent.

### 6.4 Problems with SDS

SDS tends to produce over-saturated, over-smoothed results (the "Janus problem" and the "mean-seeking" problem). Several improvements have been proposed:

- **VSD (Variational Score Distillation)** (Wang et al., 2024): Uses a learned variational distribution instead of a delta distribution, producing sharper results.
- **SDS with negative prompts**: Adding classifier-free guidance improves diversity.
- **ISM (Interval Score Matching)** (Liang et al., 2024): Matches interval denoising steps rather than single-step scores.

### 6.5 SDS Beyond 3D

The SDS idea is more general than 3D generation. Any differentiable parameterization can be optimized using a diffusion model as the loss:

- **Image editing**: Optimize pixel values to match a text prompt while staying close to an original image
- **Texture synthesis**: Optimize texture maps for 3D meshes
- **Motion synthesis**: Optimize motion parameters to produce natural-looking animations
- **Molecular design**: Optimize molecular structures using a diffusion model trained on molecular data

The pattern is: "I have a diffusion model that knows what good outputs look like, and a differentiable generator. Use the diffusion model to guide the generator."

---

## 7. The Distillation Landscape

### 7.1 Comparing the Methods

| Method | Requires Teacher? | Steps | Quality at 1 Step | Training Cost |
|--------|-------------------|-------|-------------------|---------------|
| Progressive distillation | Yes | $2^k \to 1$ | Moderate | $k$ rounds |
| Consistency distillation | Yes | Any | Good | Single round |
| Consistency training | No | Any | Good (w/ more training) | Single round |
| Adversarial distillation | Yes | 1-4 | Very good | Expensive (GAN) |
| LCM | Yes | 2-8 | Good | Single round |

### 7.2 The Open Question: Is Distillation Necessary?

An intriguing question: is it possible to train a 1-step generator *directly*, without first training a multi-step model and then distilling?

Consistency training (CT) is a partial answer: it trains a consistency model from scratch. But in practice, CT requires significantly more training compute than consistency distillation (CD) to reach the same quality.

Flow matching with reflow (Week 11) is another approach: the reflow procedure progressively straightens paths, enabling few-step generation without explicit distillation. But reflow requires generating paired data, which is itself expensive.

The fundamental tension is that generative models need to learn a complex, multi-modal mapping from noise to data. Multi-step processes decompose this into many small, simple steps. Distillation compresses these small steps into fewer, larger steps. Whether this compression can be done without the intermediate multi-step model remains an open question.

---

## Summary

1. **Progressive distillation** (Salimans and Ho, 2022) trains a student to match the teacher's two-step output in one step, then iterates. Each round halves the step count.

2. **Consistency models** (Song et al., 2023) exploit the self-consistency of ODE trajectories: every point on a trajectory maps to the same origin. Can be trained by distillation (CD) or from scratch (CT).

3. **Latent consistency models** (Luo et al., 2023) apply consistency to latent diffusion, enabling 2-4 step generation from Stable Diffusion. LCM-LoRA provides a plug-and-play acceleration adapter.

4. **Adversarial distillation** adds a GAN discriminator to prevent the blurriness caused by L2 regression losses, enabling high-quality 1-step generation (SDXL-Turbo, SDXL-Lightning).

5. **Score distillation sampling** uses a diffusion model as a loss function for optimizing other representations (3D scenes, textures, etc.), transferring the diffusion model's knowledge to new domains.

6. The **quality-speed frontier** in early 2025: 1-step adversarial distillation for real-time, 4-step consistency/LCM for interactive, 20+ steps for maximum quality.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| Progressive distillation loss | $\mathcal{L} = \Vert x\_{t+2\Delta t}^{\text{teacher}} - x\_{t+2\Delta t}^{\text{student}}\Vert ^2$ |
| Self-consistency | $f(x\_t, t) = f(x\_{t'}, t')$ for $(x\_t, t), (x\_{t'}, t')$ on same trajectory |
| Consistency distillation | $\mathcal{L} = d(f\_\theta(x\_t, t), f\_{\theta^-}(x\_{t-\Delta t}, t - \Delta t))$ |
| Consistency training | $\mathcal{L} = d(f\_\theta(x\_{t\_{n+1}}, t\_{n+1}), f\_{\theta^-}(x\_{t\_n}, t\_n))$ |
| $v$-prediction | $v\_t = \alpha\_t \epsilon - \sigma\_t x\_1$ |
| SDS gradient | $\nabla\_\theta \mathcal{L} = \mathbb{E}[w(t)(\epsilon\_\phi(x\_t, t, y) - \epsilon)\frac{\partial g}{\partial \theta}]$ |

---

## Suggested Reading

- **Salimans and Ho** (2022), "Progressive Distillation for Fast Sampling of Diffusion Models" -- the foundational distillation paper. Clean exposition.
- **Song et al.** (2023), "Consistency Models" -- introduces both consistency distillation and consistency training. Elegant mathematical framework.
- **Luo et al.** (2023), "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference" -- practical application of consistency to Stable Diffusion.
- **Sauer et al.** (2023), "Adversarial Diffusion Distillation" -- the SDXL-Turbo paper. Combines adversarial training with distillation.
- **Lin et al.** (2024), "SDXL-Lightning: Progressive Adversarial Diffusion Distillation" -- state-of-the-art few-step generation.
- **Poole et al.** (2023), "DreamFusion: Text-to-3D using 2D Diffusion" -- introduces SDS for text-to-3D generation.
- **Song and Dhariwal** (2024), "Improved Techniques for Training Consistency Models" -- significantly improved consistency training, closing the gap with distillation.
