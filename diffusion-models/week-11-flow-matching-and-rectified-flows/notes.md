# Week 11: Flow Matching and Rectified Flows

> *"The shortest distance between two points is a straight line."*
> -- Archimedes (attributed)

---

## Overview

For ten weeks we have been building diffusion models from the ground up -- starting from probability and stochastic processes, through score matching and SDEs, to the practical machinery of latent diffusion and guidance. All of these approaches share a common DNA: they define a *stochastic* forward process that gradually destroys data with noise, then learn to reverse it.

This week, we study a fundamentally different perspective that has rapidly become the dominant paradigm in practice: **flow matching**. Instead of learning to reverse a noisy diffusion, we learn a *deterministic velocity field* that transports noise to data along smooth paths. The key insight, due to Lipman et al. (2023), is that this velocity field can be trained with a remarkably simple loss function -- no SDE simulation, no adjoint method, no noise schedule to tune.

The result is a framework that is simultaneously simpler to understand, easier to train, and faster to sample from. Stable Diffusion 3, Flux, and most state-of-the-art image generators released since mid-2024 use flow matching rather than classical diffusion. Understanding why requires us to trace the idea from continuous normalizing flows through the conditional flow matching breakthrough to rectified flows.

### Prerequisites
- Week 3: Stochastic differential equations, ODEs
- Week 6: Score-based generative models, probability flow ODE
- Week 7: The SDE/ODE unification
- Week 8: ODE solvers, Euler method
- Week 10: Classifier-free guidance

---

## 1. Continuous Normalizing Flows

### 1.1 The Idea: ODEs as Generative Models

A **continuous normalizing flow (CNF)** defines a generative model through an ordinary differential equation:

$$\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]$$

Starting from noise $x_0 \sim p_0 = \mathcal{N}(0, I)$ at time $t = 0$, we integrate the velocity field $v_\theta$ forward to time $t = 1$ to obtain a sample $x_1 \sim p_1 \approx p_{\text{data}}$.

The velocity field $v_\theta : \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d$ is a neural network. It tells each point $x$ where to move at each time $t$. The collection of all trajectories forms a **flow** $\phi_t : \mathbb{R}^d \to \mathbb{R}^d$ that maps $p_0$ to $p_1$.

This is a clean, elegant formulation. But it has a serious problem.

### 1.2 The Log-Likelihood and the Instantaneous Change of Variables

To train a CNF by maximum likelihood, we need to evaluate $\log p_1(x_1)$ for data points $x_1$. The density of the pushforward distribution is given by the **instantaneous change of variables** formula (Chen et al., 2018):

$$\log p_1(x_1) = \log p_0(x_0) - \int_0^1 \nabla \cdot v_\theta(\phi_t(x_0), t) \, dt$$

where $\nabla \cdot v_\theta = \text{tr}\left(\frac{\partial v_\theta}{\partial x}\right)$ is the divergence of the velocity field.

This is beautiful mathematics, but computationally painful:

1. **Forward ODE solve**: We need to integrate $dx/dt = v_\theta(x, t)$ from $t = 0$ to $t = 1$ to find the trajectory $\phi_t(x_0)$.
2. **Divergence computation**: At each integration step, we need $\nabla \cdot v_\theta$, which requires computing or estimating the trace of the Jacobian $\partial v_\theta / \partial x$ -- an $O(d)$ operation even with the Hutchinson trace estimator.
3. **Backward ODE solve (adjoint method)**: To compute gradients for training, we solve another ODE backward in time.

The result: training a CNF requires solving ODEs at every gradient step, which is slow and numerically unstable. This is why CNFs, despite their theoretical elegance, were impractical for high-resolution generation.

### 1.3 The Challenge Stated Precisely

We want to learn $v_\theta$ such that the flow induced by $dx/dt = v_\theta(x, t)$ pushes $p_0$ to $p_{\text{data}}$. The naive approach requires simulating the ODE during training, which is expensive. Is there a way to train $v_\theta$ *without* simulating the ODE?

This is the question that flow matching answers.

---

## 2. Conditional Flow Matching: The Breakthrough

### 2.1 The Flow Matching Objective

Suppose there exists a *target* velocity field $u_t(x)$ that generates the correct time-dependent density $p_t(x)$, interpolating from $p_0 = \mathcal{N}(0, I)$ to $p_1 = p_{\text{data}}$. Then the continuity equation holds:

$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t u_t) = 0$$

The **flow matching objective** trains $v_\theta$ to match this target velocity field:

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim U[0,1]} \mathbb{E}_{x \sim p_t(x)} \left[\|v_\theta(x, t) - u_t(x)\|^2\right]$$

This is a simple regression loss -- no ODE simulation needed! The problem is that we generally cannot compute $u_t(x)$ or sample from the marginal $p_t(x)$. We know $p_0$ and $p_1$ but not the intermediate densities.

### 2.2 The Conditional Flow Matching Trick

Here is where Lipman et al. (2023) had their key insight. Instead of working with the *marginal* flow, work with *conditional* flows.

**Step 1: Define conditional probability paths.** For each data point $x_1$, define a conditional probability path $p_t(x | x_1)$ that interpolates from noise to that specific data point. The simplest choice is a Gaussian path:

$$p_t(x | x_1) = \mathcal{N}(x \mid \mu_t(x_1), \sigma_t(x_1)^2 I)$$

where $\mu_t$ and $\sigma_t$ are chosen so that:
- At $t = 0$: $p_0(x | x_1) \approx \mathcal{N}(0, I)$ (noise)
- At $t = 1$: $p_1(x | x_1) \approx \delta(x - x_1)$ (concentrated at the data point)

**Step 2: Compute the conditional velocity field.** Each conditional path $p_t(x | x_1)$ has an associated velocity field $u_t(x | x_1)$ that generates it. Because $p_t(x | x_1)$ is Gaussian with known mean and variance, this velocity field can be computed in closed form.

**Step 3: The marginalization theorem.** The marginal velocity field and density are:

$$p_t(x) = \int p_t(x | x_1) p_{\text{data}}(x_1) \, dx_1$$

$$u_t(x) = \int u_t(x | x_1) \frac{p_t(x | x_1) p_{\text{data}}(x_1)}{p_t(x)} \, dx_1$$

**Step 4: The loss equivalence.** Lipman et al. proved that the **conditional flow matching (CFM) loss**:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1]} \mathbb{E}_{x_1 \sim p_{\text{data}}} \mathbb{E}_{x \sim p_t(x|x_1)} \left[\|v_\theta(x, t) - u_t(x | x_1)\|^2\right]$$

has the same gradients as the intractable flow matching loss $\mathcal{L}_{\text{FM}}(\theta)$.

This is the central result. Let us unpack why it is so powerful:

- We can sample $x_1$ from the dataset (just draw a training example)
- We can sample $x$ from $p_t(x | x_1)$ (it is a Gaussian with known parameters)
- We can evaluate $u_t(x | x_1)$ in closed form (it depends on the choice of path)
- We do *not* need to simulate any ODE
- We do *not* need to compute any divergence or Jacobian

### 2.3 Proof Sketch of the Loss Equivalence

The proof relies on expanding the squared norm and showing that the cross terms match.

The FM loss is:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_t \mathbb{E}_{x \sim p_t} \|v_\theta(x,t) - u_t(x)\|^2$$

Expanding:

$$= \mathbb{E}_t \mathbb{E}_{x \sim p_t} \left[\|v_\theta(x,t)\|^2 - 2 v_\theta(x,t)^\top u_t(x) + \|u_t(x)\|^2\right]$$

The CFM loss is:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_t \mathbb{E}_{x_1} \mathbb{E}_{x \sim p_t(\cdot|x_1)} \|v_\theta(x,t) - u_t(x|x_1)\|^2$$

Expanding similarly and using the marginalization $p_t(x) = \int p_t(x|x_1) p_{\text{data}}(x_1) dx_1$, one can show that:

$$\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$$

The key step is that the cross term $\mathbb{E}_{x \sim p_t} [v_\theta(x,t)^\top u_t(x)]$ equals $\mathbb{E}_{x_1} \mathbb{E}_{x \sim p_t(\cdot|x_1)} [v_\theta(x,t)^\top u_t(x|x_1)]$ by the law of total expectation and the definition of $u_t(x)$ as the conditional expectation of $u_t(x|x_1)$. The two losses differ only by a constant (the $\|u_t\|^2$ term), so their gradients are identical.

---

## 3. Linear Interpolation Paths: The Simplest Case

### 3.1 The Optimal Transport Path

The simplest and most commonly used conditional path is **linear interpolation**:

$$x_t = (1 - t) x_0 + t x_1$$

where $x_0 \sim \mathcal{N}(0, I)$ is a noise sample and $x_1 \sim p_{\text{data}}$ is a data sample. At $t = 0$ we have pure noise; at $t = 1$ we have the data point. The intermediate points lie on the straight line connecting them.

The conditional probability path is:

$$p_t(x | x_1) = \mathcal{N}(x \mid t x_1, (1 - t)^2 I)$$

The conditional velocity field is obtained by differentiating $x_t$ with respect to $t$:

$$u_t(x | x_1) = \frac{dx_t}{dt} = x_1 - x_0$$

This is constant in time! The velocity at every point along the path is simply the direction from noise to data.

### 3.2 The Flow Matching Training Loss

Substituting into the CFM loss:

$$\boxed{\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1]} \mathbb{E}_{x_0 \sim \mathcal{N}(0,I)} \mathbb{E}_{x_1 \sim p_{\text{data}}} \left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]}$$

where $x_t = (1 - t)x_0 + tx_1$.

This is stunningly simple. The training procedure is:

1. Sample $t \sim U[0, 1]$
2. Sample noise $x_0 \sim \mathcal{N}(0, I)$
3. Sample data $x_1$ from the training set
4. Compute $x_t = (1 - t) x_0 + t x_1$
5. Predict $v_\theta(x_t, t)$
6. Loss = $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$

Compare this to the diffusion model training procedure (Week 5): sample noise level, add noise to data, predict the noise (or score). The flow matching procedure is structurally identical but conceptually cleaner -- we predict a *velocity* (direction of motion) rather than a *noise* or *score*.

### 3.3 Why Linear Paths Are (Approximately) Optimal Transport

A natural question: among all possible paths from $p_0$ to $p_1$, which is the best? The optimal transport (OT) perspective gives an answer: the best paths are those that minimize the total transport cost.

For Gaussian source $p_0 = \mathcal{N}(0, I)$ and an arbitrary target $p_1$, the Monge optimal transport map is:

$$T(x_0) = \mu_1 + \Sigma_1^{1/2} x_0$$

when $p_1 = \mathcal{N}(\mu_1, \Sigma_1)$. The optimal transport path is the linear interpolation $x_t = (1-t)x_0 + t \cdot T(x_0)$.

For general (non-Gaussian) $p_1$, the conditional linear paths $x_t = (1-t)x_0 + tx_1$ are not exactly optimal transport (because each $x_0$ is paired with a random $x_1$, not the OT-optimal partner). However, they are a good approximation, and the resulting paths are much straighter than diffusion paths. This is the key practical advantage.

### 3.4 Diffusion Paths vs. Flow Matching Paths

Classical diffusion models define a forward process:

$$x_t = \alpha_t x_1 + \sigma_t x_0$$

where $\alpha_t$ and $\sigma_t$ are determined by the noise schedule (e.g., $\alpha_t = \sqrt{\bar{\alpha}_t}$, $\sigma_t = \sqrt{1 - \bar{\alpha}_t}$ in the DDPM notation from Week 5). These paths are **curved** in the space of distributions -- the signal-to-noise ratio changes nonlinearly with $t$.

Flow matching uses **straight** paths: $x_t = (1-t)x_0 + tx_1$. The signal and noise mix linearly.

Why does this matter? Straighter paths require fewer discretization steps when solving the ODE at sampling time. If the path from noise to data is nearly a straight line, even a 1-step Euler method gives a reasonable approximation. Curved paths require more steps to follow accurately.

This is the fundamental reason flow matching models can generate high-quality images in fewer steps than classical diffusion models.

---

## 4. Connections to Diffusion Models

### 4.1 Diffusion as a Special Case of Flow Matching

Recall from Week 7 that every diffusion SDE has an equivalent **probability flow ODE**:

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

This is a CNF! The velocity field is $v(x, t) = f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$.

So diffusion models are, in a precise sense, a special case of continuous normalizing flows -- one where the velocity field is derived from a particular choice of noise process. Flow matching generalizes this by allowing arbitrary path designs, not just those induced by SDEs.

### 4.2 Score Prediction vs. Velocity Prediction

In diffusion models, we train a network $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ (the score). In flow matching, we train $v_\theta(x, t) \approx u_t(x)$ (the velocity).

These are related by:

$$v_t(x) = f(x, t) - \frac{1}{2}g(t)^2 s_t(x)$$

For the variance-preserving (VP) SDE with linear paths, the score and velocity predictions are related by a time-dependent rescaling. In practice, many implementations allow switching between noise prediction ($\epsilon$-prediction), score prediction, and velocity prediction ($v$-prediction) as different parameterizations of the same underlying model.

### 4.3 The Noise Schedule Disappears

One of the most practically significant differences: flow matching with linear interpolation has no noise schedule to tune. The interpolation $x_t = (1-t)x_0 + tx_1$ is fully specified. There is no $\beta_t$, no $\bar{\alpha}_t$, no cosine schedule, no shifted-cosine schedule.

In diffusion models, the noise schedule is a critical hyperparameter. Too aggressive and the model struggles at high noise levels; too gentle and sampling is slow. The literature on noise schedule design is substantial (Weeks 5, 7, 8). Flow matching sidesteps this entire design space.

---

## 5. Rectified Flows

### 5.1 The Problem: Crossing Paths

While linear interpolation paths are straighter than diffusion paths, they are not perfectly straight in the *marginal* sense. Here is the issue.

Consider two data points $x_1^{(a)}$ and $x_1^{(b)}$, and their paired noise samples $x_0^{(a)}$ and $x_0^{(b)}$. The conditional paths are:

$$x_t^{(a)} = (1-t)x_0^{(a)} + t x_1^{(a)}$$
$$x_t^{(b)} = (1-t)x_0^{(b)} + t x_1^{(b)}$$

These straight lines may **cross** at some intermediate time $t$. When paths cross, the learned velocity field must be multi-valued at the crossing point (it must point toward $x_1^{(a)}$ for trajectory $a$ and toward $x_1^{(b)}$ for trajectory $b$). Since $v_\theta$ is a function, it can only output one value, so it must compromise -- leading to errors and curved effective trajectories.

### 5.2 The Reflow Procedure

**Rectified flows** (Liu et al., 2023) address this by iteratively straightening the flow. The procedure is:

**Step 1: Train an initial flow matching model.** Train $v_\theta^{(1)}$ using the standard CFM loss with linear interpolation paths.

**Step 2: Generate new (noise, data) pairs using the trained model.** For each noise sample $x_0 \sim \mathcal{N}(0, I)$, solve the ODE $dx/dt = v_\theta^{(1)}(x, t)$ to obtain $\hat{x}_1$. This gives a coupled pair $(x_0, \hat{x}_1)$.

**Step 3: Retrain with the new pairs (reflow).** Train a new model $v_\theta^{(2)}$ using linear interpolation paths between the coupled pairs:

$$x_t = (1-t)x_0 + t \hat{x}_1$$

The key insight: because $x_0$ and $\hat{x}_1$ are already connected by the flow $v_\theta^{(1)}$, the straight-line paths between them are closer to the actual flow trajectories. This means fewer crossings, which means straighter effective paths.

**Step 4: Iterate.** Repeat steps 2-3 to get $v_\theta^{(3)}, v_\theta^{(4)}, \ldots$. Each iteration produces straighter paths.

### 5.3 Why Reflow Works

After reflow, the noise-data pairs $(x_0, \hat{x}_1)$ are no longer independent -- they are causally linked through the flow. If the flow is good, points that start near each other in noise space end up near each other in data space (the flow is approximately continuous and injective). This means the straight-line paths between paired points are nearly parallel and rarely cross.

Mathematically, the reflow procedure converges to **straight-line trajectories** in the limit: the velocity field becomes time-independent along each trajectory, meaning $v_\theta(x_t, t) \approx x_1 - x_0$ for all $t$. When this holds, a single Euler step suffices for exact generation:

$$x_1 = x_0 + v_\theta(x_0, 0) \cdot \Delta t = x_0 + v_\theta(x_0, 0)$$

This is the dream: **one-step generation** from a flow matching model, achieved by making the paths perfectly straight.

### 5.4 Reflow in Practice

In practice, one round of reflow is usually sufficient to significantly improve few-step generation quality. The procedure is:

1. Train the base flow matching model (standard CFM training)
2. Generate 1-5 million (noise, data) pairs using the trained model
3. Retrain on the paired data
4. Optionally, distill to 1-4 steps (see Week 12)

The computational cost of reflow is significant (requiring ODE solves to generate the paired data), but it is a one-time cost that yields a model capable of much faster sampling.

---

## 6. Flow Matching in Practice

### 6.1 Architecture

Flow matching models use the same architectures as diffusion models -- typically a U-Net or Diffusion Transformer (DiT). The network takes $(x_t, t)$ as input and outputs a velocity vector of the same dimension as $x_t$. Time conditioning is handled identically (sinusoidal embeddings, adaptive layer norm, etc.).

The only architectural difference is the output interpretation: the network predicts a velocity $v_\theta(x_t, t)$ rather than a noise $\epsilon_\theta(x_t, t)$ or score $s_\theta(x_t, t)$.

### 6.2 Sampling

Sampling from a flow matching model requires solving the ODE:

$$\frac{dx}{dt} = v_\theta(x, t), \quad x_0 \sim \mathcal{N}(0, I)$$

from $t = 0$ to $t = 1$. Any ODE solver works:

**Euler method** (simplest):
$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$$

**Midpoint method** (better accuracy per step):
$$k_1 = v_\theta(x_t, t), \quad k_2 = v_\theta(x_t + \frac{\Delta t}{2} k_1, t + \frac{\Delta t}{2})$$
$$x_{t+\Delta t} = x_t + \Delta t \cdot k_2$$

Because the paths are straighter than diffusion paths, fewer steps are needed. Typical step counts:
- Diffusion (DDPM): 50-1000 steps
- Diffusion (DDIM/DPM-Solver): 10-25 steps
- Flow matching: 10-20 steps
- Flow matching + reflow: 1-4 steps

### 6.3 Conditioning and Guidance

Classifier-free guidance (Week 10) works identically with flow matching. Given conditional and unconditional velocity predictions:

$$\tilde{v}_\theta(x_t, t, c) = v_\theta(x_t, t, \varnothing) + w \cdot (v_\theta(x_t, t, c) - v_\theta(x_t, t, \varnothing))$$

where $w > 1$ is the guidance scale and $c$ is the conditioning signal (text prompt, class label, etc.).

### 6.4 State of the Art

As of early 2025, the major flow matching systems include:

- **Stable Diffusion 3** (Stability AI, 2024): Uses a rectified flow formulation with a multimodal DiT (MM-DiT) architecture. The first major consumer model to use flow matching.
- **Flux** (Black Forest Labs, 2024): Built by ex-Stability researchers. Uses flow matching with a transformer architecture. Currently among the highest-quality open-weight image models.
- **Sora** (OpenAI, 2024): Video generation model. Uses flow matching (details limited, but confirmed in the technical report).

The trend is clear: flow matching has become the default for new generative models.

---

## 7. The Bigger Picture: Why Flow Matching Is Winning

Let us collect the advantages of flow matching over classical diffusion:

| Aspect | Diffusion Models | Flow Matching |
|--------|-----------------|---------------|
| Forward process | Stochastic (noise injection) | Deterministic (interpolation) |
| Training target | Noise $\epsilon$ or score $\nabla \log p_t$ | Velocity $u_t = x_1 - x_0$ |
| Noise schedule | Must be designed carefully | None needed |
| Path geometry | Curved (nonlinear SNR) | Straight (linear interpolation) |
| Sampling | ODE/SDE, 10-50 steps | ODE, 10-20 steps (1-4 with reflow) |
| Theory | Requires SDE machinery | Simpler (just ODEs and regression) |
| Conditioning | CFG works | CFG works identically |

The training loss is arguably the simplest in all of generative modeling:

$$\mathcal{L} = \mathbb{E}\left[\|v_\theta((1-t)x_0 + tx_1, t) - (x_1 - x_0)\|^2\right]$$

Sample noise, sample data, interpolate, regress velocity. That is the entire algorithm.

There is a pedagogical irony here. Diffusion models required ten weeks of mathematical machinery (stochastic calculus, SDEs, score functions, the reverse-time SDE, the probability flow ODE, noise schedules, DDIM, DPM-Solver) to arrive at a practical training procedure. Flow matching arrives at an equally effective (arguably superior) procedure through a much shorter path. But the ten weeks were not wasted -- the diffusion perspective provides deep insights into *why* these methods work, and the mathematical tools transfer to problems where flow matching's simplifications do not apply.

---

## Summary

1. **Continuous normalizing flows** define generative models via ODEs $dx/dt = v_\theta(x,t)$, but training requires expensive ODE simulation.

2. **Conditional flow matching** (Lipman et al., 2023) makes CNFs trainable by showing that the intractable marginal FM loss can be replaced by a tractable conditional loss that has identical gradients.

3. **Linear interpolation paths** $x_t = (1-t)x_0 + tx_1$ give the simplest form: the target velocity is $u_t(x|x_1) = x_1 - x_0$, and the loss is $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$.

4. **Rectified flows** (Liu et al., 2023) iteratively straighten flow paths by coupling noise-data pairs through the learned flow, then retraining on the coupled pairs.

5. **Flow matching vs. diffusion**: Flow matching uses straight paths (fewer sampling steps), has no noise schedule, and has a simpler training procedure. Diffusion is a special case where the paths are determined by an SDE.

6. **State of the art**: Stable Diffusion 3, Flux, and Sora all use flow matching, marking a paradigm shift in generative modeling.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| CNF | $dx/dt = v_\theta(x, t)$ |
| Instantaneous change of variables | $\log p_1(x_1) = \log p_0(x_0) - \int_0^1 \nabla \cdot v_\theta \, dt$ |
| Linear interpolation | $x_t = (1-t)x_0 + tx_1$ |
| Target velocity (linear) | $u_t(x \mid x_1) = x_1 - x_0$ |
| CFM loss | $\mathbb{E}_{t, x_0, x_1}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$ |
| Guided velocity | $\tilde{v} = v(x_t, t, \varnothing) + w(v(x_t, t, c) - v(x_t, t, \varnothing))$ |

---

## Suggested Reading

- **Lipman et al.** (2023), "Flow Matching for Generative Modeling" -- the foundational paper. Clear writing, excellent figures.
- **Liu et al.** (2023), "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" -- rectified flows and the reflow procedure.
- **Albergo and Vanden-Eijnden** (2023), "Building Normalizing Flows with Stochastic Interpolants" -- an independent derivation of similar ideas from a physics perspective.
- **Tong et al.** (2024), "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport" -- OT-CFM, connecting flow matching to optimal transport more tightly.
- **Esser et al.** (2024), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" -- the Stable Diffusion 3 paper, showing flow matching at scale.
- **Chen et al.** (2018), "Neural Ordinary Differential Equations" -- the original Neural ODE paper that started it all.
