# Week 11: Flow Matching and Rectified Flows -- Homework

**Estimated time:** 12-15 hours
**Prerequisites:** ODEs and Euler method (Week 3), score matching (Week 4), diffusion training (Week 5), PyTorch

---

## Problem 1: Conditional Flow Matching on 2D Data (Implementation)

Implement conditional flow matching from scratch on a 2D toy dataset. This is the core algorithm of the week.

### Part (a): Data and Model Setup

Create a 2D target distribution by sampling from a mixture of 8 Gaussians arranged in a circle:

```python
import torch
import torch.nn as nn

def sample_moons_or_gaussians(n, dataset='8gaussians'):
    """Sample from a 2D target distribution."""
    if dataset == '8gaussians':
        scale = 4.0
        centers = [(scale * np.cos(2*np.pi*i/8), scale * np.sin(2*np.pi*i/8))
                    for i in range(8)]
        # Sample cluster indices, then add Gaussian noise around centers
        ...
    return x  # shape (n, 2)
```

Build a simple MLP velocity network $v\_\theta(x\_t, t) : \mathbb{R}^2 \times [0,1] \to \mathbb{R}^2$:

```python
class VelocityNetwork(nn.Module):
    def __init__(self, dim=2, hidden=256):
        super().__init__()
        # Input: x_t (dim) concatenated with t (1)
        # Output: velocity (dim)
        # Use 3-4 hidden layers with SiLU activations
        ...
```

### Part (b): Training Loop

Implement the CFM training loop:

1. Sample $t \sim U[0, 1]$, $x\_0 \sim \mathcal{N}(0, I)$, $x\_1 \sim p\_{\text{data}}$
2. Compute $x\_t = (1-t)x\_0 + tx\_1$
3. Target velocity: $u\_t = x\_1 - x\_0$
4. Loss: $\Vert v\_\theta(x\_t, t) - u\_t\Vert ^2$

Train for 10000-20000 steps with batch size 256. Plot the loss curve.

### Part (c): Sampling

Implement Euler sampling:

```python
def sample_flow(model, n_samples, n_steps=100):
    """Generate samples by solving dx/dt = v_theta(x, t) from t=0 to t=1."""
    x = torch.randn(n_samples, 2)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = i * dt
        v = model(x, t * torch.ones(n_samples, 1))
        x = x + dt * v
    return x
```

Generate 2000 samples and plot them alongside the true data distribution. How many Euler steps are needed for the samples to look good? Try $N = 1, 2, 5, 10, 50, 100$.

### Part (d): Visualize the Velocity Field

At several time steps ($t = 0, 0.25, 0.5, 0.75, 1.0$), create a grid of points in $[-6, 6]^2$ and plot the predicted velocity field as arrows (using `matplotlib.pyplot.quiver`). Overlay the density of $p\_t$ (estimated by histogramming interpolated samples $x\_t = (1-t)x\_0 + tx\_1$).

Describe qualitatively how the velocity field evolves over time.

---

## Problem 2: Diffusion Paths vs. Flow Matching Paths (Theory + Implementation)

This problem makes the comparison between diffusion and flow matching concrete.

### Part (a): Path Comparison (Theory)

Consider two interpolation schemes from noise $x\_0$ to data $x\_1$:

**Flow matching (linear):** $x\_t^{\text{FM}} = (1-t)x\_0 + tx\_1$

**Diffusion (VP-SDE):** $x\_t^{\text{diff}} = \sqrt{\bar{\alpha}\_t} \, x\_1 + \sqrt{1 - \bar{\alpha}\_t} \, x\_0$ where $\bar{\alpha}\_t = e^{-\frac{1}{2}\int\_0^t \beta(s)ds}$ with linear schedule $\beta(t) = \beta\_{\min} + (\beta\_{\max} - \beta\_{\min})t$.

1. For a fixed pair $(x\_0, x\_1)$ in 2D with $x\_0 = (1, -1)$ and $x\_1 = (3, 2)$, plot both paths $x\_t^{\text{FM}}$ and $x\_t^{\text{diff}}$ for $t \in [0, 1]$. Use $\beta\_{\min} = 0.1$, $\beta\_{\max} = 20$.

2. Which path is straighter? Compute the **path length** for each:
$$
L = \int_0^1 \left\Vert \frac{dx_t}{dt}\right\Vert dt
$$

3. Compute the straight-line distance $\Vert x\_1 - x\_0\Vert$ and the **straightness ratio** $\Vert x\_1 - x\_0\Vert / L$ for each path. A ratio of 1 means perfectly straight.

### Part (b): Velocity Field Comparison (Implementation)

Train two models on the same 2D dataset from Problem 1:

1. A **flow matching model** (from Problem 1)
2. A **diffusion model** using the VP noise schedule (predict $\epsilon$, then convert to velocity)

For each model, generate 2000 samples using 10 Euler steps. Compare the sample quality visually. Which model produces better samples at this low step count?

### Part (c): Step Count Sweep

For each model, generate 2000 samples using $N \in \lbrace 1, 2, 4, 8, 16, 32, 64\rbrace$ steps. For each, compute the Wasserstein-2 distance to the true distribution (using `scipy.stats.wasserstein_distance` on each coordinate, or using the `pot` library for 2D OT distance).

Plot W2 distance vs. number of steps for both models. At what step count does each model converge to the true distribution?

---

## Problem 3: The Loss Equivalence (Theory)

This problem walks through the proof that the conditional and marginal flow matching losses have the same gradients.

### Part (a): Setup

Consider the marginal FM loss:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_t \int p_t(x) \Vert v_\theta(x,t) - u_t(x)\Vert ^2 dx
$$

and the conditional FM loss:

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_t \int \int p_t(x|x_1) q(x_1) \Vert v_\theta(x,t) - u_t(x|x_1)\Vert ^2 dx \, dx_1
$$

where $q = p\_{\text{data}}$.

Expand both losses by writing $\Vert a - b\Vert ^2 = \Vert a\Vert ^2 - 2a \cdot b + \Vert b\Vert ^2$. Show that $\mathcal{L}\_{\text{FM}}$ and $\mathcal{L}\_{\text{CFM}}$ can each be written as a sum of three terms.

### Part (b): The First Term

Show that the first term ($\mathbb{E}[\Vert v\_\theta\Vert ^2]$) is identical in both losses. *Hint: use the fact that $p\_t(x) = \int p\_t(x|x\_1) q(x\_1) dx\_1$.*

### Part (c): The Cross Term

Show that the cross terms are equal:

$$
\int p_t(x) v_\theta(x,t)^\top u_t(x) \, dx = \int \int p_t(x|x_1) q(x_1) v_\theta(x,t)^\top u_t(x|x_1) \, dx \, dx_1
$$

*Hint: use the definition $u\_t(x) = \frac{\int u\_t(x|x\_1) p\_t(x|x\_1) q(x\_1) dx\_1}{p\_t(x)}$.*

### Part (d): Conclusion

Conclude that $\nabla\_\theta \mathcal{L}\_{\text{FM}} = \nabla\_\theta \mathcal{L}\_{\text{CFM}}$, noting that the third terms ($\Vert u\_t\Vert ^2$ and $\Vert u\_t(\cdot|x\_1)\Vert ^2$) differ but do not depend on $\theta$.

---

## Problem 4: Implement Reflow (Implementation)

Implement the rectified flow "reflow" procedure and demonstrate that it straightens trajectories.

### Part (a): Train a Base Model

Using your flow matching model from Problem 1 (trained on the 8-Gaussians dataset), generate the ODE trajectories for 5000 noise samples. For each $x\_0 \sim \mathcal{N}(0, I)$, solve $dx/dt = v\_\theta(x, t)$ from $t = 0$ to $t = 1$ using 100 Euler steps. Record the final point $\hat{x}\_1$.

You now have 5000 coupled pairs $(x\_0, \hat{x}\_1)$.

### Part (b): Reflow

Train a new velocity network $v\_\theta^{(2)}$ using CFM on the coupled pairs. The training loop is identical to Problem 1, except that instead of sampling $x\_1$ from the data distribution, you sample a coupled pair $(x\_0, \hat{x}\_1)$ and use:

$$
x_t = (1-t)x_0 + t\hat{x}_1, \quad u_t = \hat{x}_1 - x_0
$$

Train for the same number of steps as the base model.

### Part (c): Measure Straightness

For each model (base and reflowed), generate 500 trajectories and compute the **straightness** of each trajectory:

$$
S = \frac{\Vert x_1 - x_0\Vert }{\int_0^1 \Vert v_\theta(x_t, t)\Vert dt}
$$

where $S = 1$ means perfectly straight. Approximate the integral using the Euler steps.

Report the mean and standard deviation of $S$ for both models. Plot histograms of $S$ for both models on the same axes.

### Part (d): Few-Step Generation

Compare the sample quality of the base model and the reflowed model at $N = 1, 2, 4$ Euler steps. Generate 2000 samples from each and plot them alongside the true distribution.

Does reflow improve few-step generation? By how much?

---

## Problem 5: Flow Matching on MNIST (Implementation)

Scale up to a real dataset.

### Part (a): Model and Training

Train a flow matching model on MNIST digits ($28 \times 28 = 784$ dimensions).

Architecture: Use a small U-Net or a simple MLP with time conditioning:

```python
class FlowMatchingMLP(nn.Module):
    def __init__(self, dim=784, hidden=1024, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))
```

Train using the standard CFM loss for 50-100 epochs on the MNIST training set.

### Part (b): Sample Quality vs. Step Count

Generate 64 samples at each of $N = 1, 5, 10, 20, 50, 100$ Euler steps. Display them as $8 \times 8$ grids.

At what step count do the digits become recognizable? At what step count do they look sharp?

### Part (c): Comparison with DDPM

If you have a DDPM model from Week 5 (or train one quickly), compare the sample quality at matched step counts. Specifically, generate 64 samples from each model using 20 steps. Which produces better-looking digits?

### Part (d): Class-Conditional Generation (Optional)

Extend your model to accept a class label $c \in \lbrace 0, 1, \ldots, 9\rbrace$ as conditioning. Train with 10% unconditional dropout (replacing $c$ with a null token). Sample using classifier-free guidance with guidance scale $w = 2.0$.

Generate 10 samples per class and display them.

---

## Problem 6: Understanding Optimal Transport Paths (Theory)

### Part (a): 1D Optimal Transport

Consider one-dimensional distributions $p\_0 = \mathcal{N}(0, 1)$ and $p\_1 = \mathcal{N}(\mu, \sigma^2)$.

1. The optimal transport map from $p\_0$ to $p\_1$ is $T(x\_0) = \sigma x\_0 + \mu$. Verify this by checking that if $x\_0 \sim \mathcal{N}(0,1)$, then $T(x\_0) \sim \mathcal{N}(\mu, \sigma^2)$.

2. The OT interpolation path is $x\_t = (1-t)x\_0 + t \cdot T(x\_0) = (1-t)x\_0 + t(\sigma x\_0 + \mu) = ((1-t) + t\sigma)x\_0 + t\mu$. What is the distribution $p\_t$ of $x\_t$? Show it is Gaussian and compute its mean and variance.

3. The CFM interpolation path (with random coupling) is $x\_t = (1-t)x\_0 + tx\_1$ where $x\_0 \sim p\_0$ and $x\_1 \sim p\_1$ are independent. Compute the distribution of $x\_t$. How does it differ from the OT path?

### Part (b): When Does the Difference Matter?

The OT path and the CFM path give the same marginal $p\_t$ when the distributions are Gaussian. But for non-Gaussian distributions they can differ significantly.

Consider $p\_0 = \mathcal{N}(0, 1)$ and $p\_1 = \frac{1}{2}\delta(x - 3) + \frac{1}{2}\delta(x + 3)$ (a mixture of two point masses).

1. Sketch (by hand or plot) the CFM interpolation $x\_t = (1-t)x\_0 + tx\_1$ for 20 random pairs. You should see crossing paths.

2. Now consider the OT coupling: $x\_0 > 0 \Rightarrow x\_1 = 3$, $x\_0 < 0 \Rightarrow x\_1 = -3$. Sketch these paths. Are there crossings?

3. Explain in 2-3 sentences why the OT coupling produces straighter paths.

---

## Submission Checklist

- [ ] Problem 1: Flow matching on 2D data, velocity field visualization, step count comparison
- [ ] Problem 2: Diffusion vs. flow matching path comparison (theory + implementation)
- [ ] Problem 3: Proof of loss equivalence (pen and paper)
- [ ] Problem 4: Reflow implementation, straightness measurement, few-step comparison
- [ ] Problem 5: Flow matching on MNIST, sample grids at various step counts
- [ ] Problem 6: Optimal transport paths (theory)

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs.
