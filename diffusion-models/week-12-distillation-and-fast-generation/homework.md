# Week 12: Distillation and Fast Generation -- Homework

**Estimated time:** 12-15 hours
**Prerequisites:** Diffusion model training (Week 5), ODE sampling (Week 8), flow matching (Week 11), PyTorch

---

## Problem 1: Progressive Distillation on 2D Data (Implementation)

Implement progressive distillation starting from a pre-trained flow matching model.

### Part (a): Teacher Model

Use the flow matching model you trained on the 8-Gaussians dataset in Week 11 (or retrain one). Verify it generates good samples with 128 Euler steps. This is your teacher.

### Part (b): One Round of Distillation

Implement one round of progressive distillation:

1. Initialize the student as a copy of the teacher.
2. For each training iteration:
   - Sample $x\_0 \sim \mathcal{N}(0, I)$
   - Sample a time step $t$ uniformly from $\lbrace 0, \Delta t, 2\Delta t, \ldots, 1 - 2\Delta t\rbrace $ where $\Delta t = 1/128$
   - Compute $x\_t$ by running the teacher ODE from $x\_0$ to time $t$ (use 128-step Euler; or, for efficiency, precompute trajectories)
   - **Teacher two-step output:** Run the teacher for two steps from $x\_t$:
     $$
     x_{t+\Delta t} = x_t + \Delta t \cdot v_T(x_t, t)
     $$
     $$
     x_{t+2\Delta t} = x_{t+\Delta t} + \Delta t \cdot v_T(x_{t+\Delta t}, t + \Delta t)
     $$
   - **Student one-step output:** Run the student for one step with doubled step size:
     $$
     \hat{x}_{t+2\Delta t} = x_t + 2\Delta t \cdot v_S(x_t, t)
     $$
   - Loss: $\Vert \hat{x}\_{t+2\Delta t} - x\_{t+2\Delta t}\Vert ^2$

Train the student for 5000-10000 iterations. The student should now work with 64 steps.

### Part (c): Iterate

Repeat the distillation process:
- Round 2: 64 → 32 steps
- Round 3: 32 → 16 steps
- Round 4: 16 → 8 steps
- Round 5: 8 → 4 steps
- Round 6: 4 → 2 steps
- Round 7: 2 → 1 step

At each round, generate 2000 samples from the distilled student. Plot them alongside the true distribution.

### Part (d): Quality vs. Steps

Plot the Wasserstein-2 distance (or another quality metric) vs. number of sampling steps, showing:
- Teacher at each step count (128, 64, 32, 16, 8, 4, 2, 1)
- Distilled student at each step count

At what step count does the distilled model start to noticeably degrade? How does it compare to the teacher at the same step count?

---

## Problem 2: Implement the Consistency Model Training Objective (Implementation)

Implement consistency training (CT) on the 2D Gaussian mixture dataset.

### Part (a): Model Architecture

Design a consistency model $f\_\theta(x, t)$ with the boundary condition $f\_\theta(x, t\_{\min}) = x$.

Use the skip-connection parameterization:

```python
class ConsistencyModel(nn.Module):
    def __init__(self, dim=2, hidden=256, t_min=0.002):
        super().__init__()
        self.t_min = t_min
        self.net = ...  # MLP with time conditioning (same as velocity network)

    def forward(self, x, t):
        # c_skip(t) = t_min / t, c_out(t) = (t - t_min) / t
        # (These ensure f(x, t_min) = x)
        c_skip = self.t_min / t
        c_out = (t - self.t_min) / t
        return c_skip * x + c_out * self.net(x, t)
```

### Part (b): Consistency Training Loss

Implement the CT loss:

1. Sample $x\_0 \sim p\_{\text{data}}$, $\epsilon \sim \mathcal{N}(0, I)$
2. Choose adjacent time steps $t\_n < t\_{n+1}$ from the schedule
3. Compute $x\_{t\_n} = x\_0 + t\_n \epsilon$ and $x\_{t\_{n+1}} = x\_0 + t\_{n+1} \epsilon$ (same $\epsilon$!)
4. Loss: $\Vert f\_\theta(x\_{t\_{n+1}}, t\_{n+1}) - f\_{\theta^-}(x\_{t\_n}, t\_n)\Vert ^2$

where $\theta^-$ is the EMA of $\theta$ (update rate $\mu = 0.999$).

Implement a schedule that starts with $N = 2$ time steps and gradually increases to $N = 150$ over training. Use a time grid $t\_n = t\_{\min}^{1-n/(N-1)} \cdot T^{n/(N-1)}$ with $T = 80$, $t\_{\min} = 0.002$.

Train for 20000-50000 iterations.

### Part (c): One-Step and Multi-Step Sampling

Implement sampling:

1. **One-step:** Sample $x\_T \sim \mathcal{N}(0, T^2 I)$, output $f\_\theta(x\_T, T)$.
2. **Multi-step ($k$ steps):** Alternate between denoising ($\hat{x}\_0 = f\_\theta(x\_t, t)$) and re-noising ($x\_{t'} = \hat{x}\_0 + t' \epsilon$) at decreasing noise levels $T > t\_1 > t\_2 > \cdots > t\_{\min}$.

Generate 2000 samples using 1, 2, 4, and 8 steps. Plot each alongside the true distribution.

### Part (d): Comparison

Compare the consistency model (1-step and 4-step) to the flow matching model (from Week 11) at 1 and 4 Euler steps. Which produces better samples at each step count?

---

## Problem 3: Derive the Consistency Condition (Theory)

### Part (a): ODE Trajectories

Consider the probability flow ODE for a variance-exploding (VE) diffusion:

$$
\frac{dx}{dt} = -t \cdot s_\theta(x, t)
$$

where $s\_\theta(x, t) \approx \nabla\_x \log p\_t(x)$ is the score function and $p\_t(x) = \int p\_0(x') \mathcal{N}(x; x', t^2 I) dx'$.

Let $\phi(x\_t, t, s)$ denote the ODE solution at time $s$ starting from $(x\_t, t)$. Show that the consistency function $f(x, t) = \phi(x, t, t\_{\min})$ satisfies:

$$
\frac{\partial f}{\partial t}(x, t) + \frac{dx}{dt} \cdot \nabla_x f(x, t) = 0
$$

*Hint: Use the chain rule. The total derivative of $f$ along the trajectory is zero because $f$ is constant along trajectories.*

### Part (b): Why the EMA Target?

In consistency training, we use $f\_{\theta^-}$ (with EMA weights) for the target rather than $f\_\theta$ (with current weights).

1. What happens if we use $f\_\theta$ for both the prediction and the target (i.e., minimize $\Vert f\_\theta(x\_{t\_{n+1}}, t\_{n+1}) - f\_\theta(x\_{t\_n}, t\_n)\Vert ^2$)? Why is there a trivial solution?

2. How does the EMA target prevent this collapse? Draw an analogy to target networks in deep reinforcement learning (DQN).

### Part (c): The Schedule Matters

The number of time discretization steps $N$ increases during training (e.g., from $N = 2$ to $N = 150$).

1. When $N = 2$, there are only two time steps: $t\_{\min}$ and $T$. What does the consistency loss reduce to in this case? Is this a meaningful training signal?

2. Why start with small $N$ and increase it? What would happen if we started with large $N$ from the beginning? *Hint: Think about the bias-variance tradeoff in the consistency target.*

---

## Problem 4: The Quality-Speed Frontier (Implementation + Analysis)

### Part (a): Build the Frontier

Using the 2D 8-Gaussians dataset, train the following models:

1. A flow matching model (Week 11) -- sample at $N = 1, 2, 4, 8, 16, 32, 64, 128$ steps
2. A progressively distilled model -- evaluate at its target step count after each distillation round
3. A consistency model -- sample at $N = 1, 2, 4, 8$ steps

For each (model, step-count) pair, generate 5000 samples and compute the Wasserstein-2 distance to the true distribution (generate 50000 true samples as reference).

### Part (b): Plot the Frontier

Create a plot with:
- X-axis: Number of sampling steps (log scale)
- Y-axis: Wasserstein-2 distance (quality metric, lower is better)
- Three curves: flow matching, progressive distillation, consistency model

Which method gives the best quality at each step budget?

### Part (c): Compute Efficiency

Each method also has a *training* cost. Estimate (in terms of number of gradient steps and network evaluations per step):
- Flow matching: standard training cost
- Progressive distillation: training cost + cost of each distillation round (note: each round requires teacher inference)
- Consistency model: training cost (note: requires EMA updates and potentially longer training)

Is the cheapest method to train also the best performer? Discuss the tradeoff between training cost and inference speed.

---

## Problem 5: Score Distillation Sampling in 2D (Implementation)

Implement SDS in a simplified 2D setting to build intuition before applying it to 3D.

### Part (a): Setup

Use your trained flow matching model from Week 11 as the "diffusion prior" (the model that knows what good samples look like).

Define a simple "renderer": a parameterized 2D point $\theta = (\theta\_1, \theta\_2) \in \mathbb{R}^2$ that we want to optimize to lie on the data distribution. (This is a degenerate "scene" that renders to a single 2D point.)

### Part (b): SDS Gradient

Implement the SDS gradient update:

```python
def sds_step(theta, model, t, lr=0.01):
    """One step of SDS optimization."""
    epsilon = torch.randn_like(theta)
    x_t = (1 - t) * epsilon_source + t * theta  # noisy version
    # For flow matching: the "score" direction is v_theta(x_t, t)
    # SDS gradient: (v_model(x_t, t) - (theta - epsilon_source)) * d(theta)/d(theta)
    v_pred = model(x_t.unsqueeze(0), t * torch.ones(1, 1))
    target_velocity = theta - epsilon_source
    grad = (v_pred.squeeze() - target_velocity)
    theta = theta - lr * grad
    return theta
```

*Note: Adapt the SDS gradient formula to the flow matching setting. In the noise-prediction formulation, the SDS gradient is $w(t)(\epsilon\_\phi - \epsilon)$. In the velocity formulation, derive the analogous expression.*

### Part (c): Optimize and Visualize

Starting from $\theta\_0 = (0, 0)$, run SDS optimization for 1000 steps, sampling $t \sim U[0, 1]$ at each step. Record the trajectory of $\theta$.

1. Plot the trajectory on top of the true data distribution. Where does $\theta$ converge to?
2. Run 20 independent optimizations from $\theta\_0 = (0, 0)$ with different random seeds. Plot all final $\theta$ values. Do they always converge to the same point, or do different runs find different modes?
3. What happens if you change the weighting $w(t)$? Try $w(t) = 1$ (uniform) vs. $w(t) = 1 - t$ (emphasize clean images) vs. $w(t) = t$ (emphasize noisy images).

### Part (d): The Mode-Seeking Problem

SDS is known to be "mode-seeking" -- it tends to find high-density regions rather than covering the full distribution.

1. Generate 100 SDS-optimized points from random initializations $\theta\_0 \sim \mathcal{N}(0, I)$. Plot them alongside the true distribution. Does SDS cover all 8 Gaussian modes?

2. Compare to direct sampling from the flow matching model (100 samples). Which gives better coverage of the distribution?

3. In 2-3 sentences, explain why SDS is mode-seeking. *Hint: SDS minimizes $D\_{\text{KL}}(q \Vert  p)$ where $q$ is a delta distribution. Which direction of KL divergence is mode-seeking?*

---

## Problem 6: Analyzing Distillation Theoretically (Theory)

### Part (a): Progressive Distillation Error Accumulation

Suppose the student at round $k$ introduces a per-step velocity error $\delta\_k$ (so $\Vert v\_S^{(k)} - v\_T^{(k)}\Vert  \leq \delta\_k$ at each step). After $k$ rounds of distillation starting from $N$ teacher steps:

1. The student at round $k$ uses $N/2^k$ steps. Write the total trajectory error (distance between student-generated and teacher-generated endpoints) in terms of $\delta\_1, \ldots, \delta\_k$ and the number of steps at each round.

2. Under what conditions on $\delta\_k$ does the total error remain bounded as $k \to \infty$ (i.e., as the step count goes to 1)?

3. In practice, each round of distillation reduces $\delta\_k$ (the student gets better at matching the teacher). But $\delta\_k$ need not decrease fast enough to compensate for the halving of steps. What determines whether progressive distillation "works" at very low step counts?

### Part (b): The Regression-Blurriness Tradeoff

Consider a 1D example where the true conditional distribution $p(x\_0 | x\_t)$ is bimodal:

$$
p(x_0 | x_t) = \frac{1}{2}\mathcal{N}(x_0; -1, 0.01) + \frac{1}{2}\mathcal{N}(x_0; +1, 0.01)
$$

1. What is the L2-optimal prediction $\hat{x}\_0 = \mathbb{E}[x\_0 | x\_t]$?

2. What is the adversarial-optimal prediction (i.e., the prediction that maximizes the likelihood under $p(x\_0 | x\_t)$)?

3. Explain in 2-3 sentences why adversarial distillation produces sharper results than L2 distillation for few-step generation.

---

## Submission Checklist

- [ ] Problem 1: Progressive distillation on 2D data, quality vs. steps plot
- [ ] Problem 2: Consistency model implementation, 1-step and multi-step sampling
- [ ] Problem 3: Consistency condition derivation, EMA analysis (pen and paper)
- [ ] Problem 4: Quality-speed frontier comparison plot
- [ ] Problem 5: SDS implementation in 2D, mode-seeking analysis
- [ ] Problem 6: Distillation error analysis, regression-blurriness tradeoff (theory)

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs.
