# Week 4: Training Neural Networks

## Overview

Last week we built neural networks — multilayer perceptrons with nonlinear activations, capable in principle of approximating any continuous function. But "capable in principle" and "actually works" are separated by a chasm called *training*. This week we cross that chasm.

The central question: given a loss function $\mathcal{L}(\boldsymbol{\theta})$ that measures how poorly our network performs, how do we efficiently find parameters $\boldsymbol{\theta}$ that make the loss small? The answer is **backpropagation** — an algorithm so fundamental that understanding it deeply is non-negotiable for everything that follows in this course.

We will also study the optimizer zoo (SGD, momentum, Adam), regularization techniques that prevent overfitting, and the practical dark arts of training neural networks. By the end of this week, you should be able to take a network from random initialization to strong performance on a real dataset, and understand every step of what is happening under the hood.

---

## 1. Backpropagation

### 1.1 The Problem

Consider a neural network as a function $f(\mathbf{x}; \boldsymbol{\theta})$ parameterized by weights $\boldsymbol{\theta} = \lbrace W^{(1)}, \mathbf{b}^{(1)}, W^{(2)}, \mathbf{b}^{(2)}, \ldots\rbrace $. Given a training example $(\mathbf{x}, y)$, we compute a loss:

$$
\mathcal{L} = \ell(f(\mathbf{x}; \boldsymbol{\theta}), y)
$$

To perform gradient descent, we need $\frac{\partial \mathcal{L}}{\partial \theta\_i}$ for every parameter $\theta\_i$ in the network. A network with $d$ parameters requires $d$ partial derivatives. How do we compute them efficiently?

The naive approach — perturb each parameter by $\epsilon$ and measure the change in loss — requires $d$ forward passes. For a network with $10^6$ parameters, that is $10^6$ forward passes per gradient. Utterly impractical.

### 1.2 Computational Graphs

The key insight is that any neural network computation can be represented as a **directed acyclic graph** (DAG), where:
- Each node represents an intermediate computation (a variable)
- Each edge represents a direct dependency

For example, the computation $\mathcal{L} = (wx + b - y)^2$ can be decomposed into intermediate variables:
- $z\_1 = wx$
- $z\_2 = z\_1 + b$
- $z\_3 = z\_2 - y$
- $\mathcal{L} = z\_3^2$

```
    w   x        b       y
    |   |        |       |
    v   v        |       |
   [  *  ]       |       |
      |          |       |
      v          v       |
     [   +   ]           |
         |               |
         v               v
        [      -      ]
               |
               v
            [ ^2 ]
               |
               v
               L
```

The chain rule lets us propagate derivatives backward through this graph.

### 1.3 The Chain Rule, Carefully

Recall from calculus: if $y = f(g(x))$, then $\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$. In Leibniz notation:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

where $u = g(x)$.

For the multivariate case, if $\mathcal{L}$ depends on $\theta$ through multiple intermediate variables $z\_1, \ldots, z\_k$:

$$
\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{i=1}^{k} \frac{\partial \mathcal{L}}{\partial z_i} \cdot \frac{\partial z_i}{\partial \theta}
$$

This is the **multivariate chain rule**, and it is the entire mathematical content of backpropagation. The algorithm is simply the chain rule applied systematically in the right order.

### 1.4 Forward Pass and Backward Pass

**Backpropagation** is the algorithm that applies the chain rule systematically to a computational graph, proceeding from the output back to the inputs.

**Forward pass:** Compute the output of each node in topological order (inputs first, output last). Store all intermediate values — we will need them during the backward pass.

**Backward pass:** Starting from the loss node (where $\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1$), compute $\frac{\partial \mathcal{L}}{\partial z\_i}$ for each node $z\_i$ in reverse topological order. At each node, use the chain rule to propagate gradients backward to its inputs.

The quantity $\frac{\partial \mathcal{L}}{\partial z\_i}$ is sometimes called the **adjoint** of node $z\_i$, or informally, the "gradient flowing back to $z\_i$."

### 1.5 Complete Derivation for a 2-Layer Network

Let us derive backpropagation in full for a 2-layer network with one hidden layer. This is the most important derivation in the course so far — work through every line.

**Architecture:**

- Input: $\mathbf{x} \in \mathbb{R}^{d\_0}$
- Hidden layer: $\mathbf{h} = \sigma(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)})$, where $W^{(1)} \in \mathbb{R}^{d\_1 \times d\_0}$, $\mathbf{b}^{(1)} \in \mathbb{R}^{d\_1}$
- Output layer: $\hat{\mathbf{y}} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)}$, where $W^{(2)} \in \mathbb{R}^{d\_2 \times d\_1}$, $\mathbf{b}^{(2)} \in \mathbb{R}^{d\_2}$
- Loss: $\mathcal{L} = \frac{1}{2} \Vert \hat{\mathbf{y}} - \mathbf{y}\Vert ^2$ (MSE, with the $\frac{1}{2}$ for convenience)

We use $\sigma$ for the activation function (e.g., ReLU or sigmoid). Let us define intermediate variables:

$$
\mathbf{a}^{(1)} = W^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \quad \text{(pre-activation)}
$$
$$
\mathbf{h} = \sigma(\mathbf{a}^{(1)}) \quad \text{(post-activation)}
$$
$$
\mathbf{a}^{(2)} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)} \quad \text{(output pre-activation)}
$$
$$
\hat{\mathbf{y}} = \mathbf{a}^{(2)} \quad \text{(no activation on output for regression)}
$$

**Forward pass:** Compute $\mathbf{a}^{(1)} \to \mathbf{h} \to \mathbf{a}^{(2)} \to \mathcal{L}$ in order, storing each intermediate result.

**Backward pass:** We compute gradients layer by layer, working backward from the loss.

**Step 1: Gradient of loss w.r.t. output.**

$$
\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} = \hat{\mathbf{y}} - \mathbf{y}
$$

This is a vector in $\mathbb{R}^{d\_2}$. Call this $\boldsymbol{\delta}^{(2)}$ — the error signal at the output layer.

**Step 2: Gradients for the output layer parameters.**

Since $\hat{\mathbf{y}} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)}$, applying the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \boldsymbol{\delta}^{(2)} \mathbf{h}^\top
$$

This is an outer product, yielding a $d\_2 \times d\_1$ matrix — exactly matching the shape of $W^{(2)}$. Each entry $(i, j)$ tells us how much the loss changes if we perturb $W^{(2)}\_{ij}$.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)}
$$

**Step 3: Propagate the gradient backward through the output layer.**

The hidden activations $\mathbf{h}$ influenced the loss through the linear transformation $W^{(2)}\mathbf{h}$. By the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}} = (W^{(2)})^\top \boldsymbol{\delta}^{(2)}
$$

This is a vector in $\mathbb{R}^{d\_1}$ — the gradient "flowing back" to the hidden layer activations. Note the transpose: forward propagation multiplies by $W^{(2)}$, backward propagation multiplies by $W^{(2)\top}$.

**Step 4: Propagate through the activation function.**

The activation $\sigma$ is applied element-wise, so its Jacobian is diagonal:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \odot \sigma'(\mathbf{a}^{(1)})
$$

where $\odot$ denotes element-wise multiplication. Call this $\boldsymbol{\delta}^{(1)}$.

For ReLU, $\sigma'(a) = \mathbf{1}[a > 0]$ (1 if positive, 0 if negative). For sigmoid, $\sigma'(a) = \sigma(a)(1 - \sigma(a))$.

**Step 5: Gradients for the hidden layer parameters.**

By the same logic as Step 2, but one layer earlier:

$$
\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^\top
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}
$$

**The general pattern** is now clear and extends to any depth. For layer $\ell$:

1. Compute the error signal: $\boldsymbol{\delta}^{(\ell)} = \left[(W^{(\ell+1)})^\top \boldsymbol{\delta}^{(\ell+1)}\right] \odot \sigma'(\mathbf{a}^{(\ell)})$
2. Weight gradient: $\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} (\mathbf{h}^{(\ell-1)})^\top$
3. Bias gradient: $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}$

That is the entire algorithm. The beauty of backpropagation is that it is *just the chain rule*, applied in the right order for efficiency.

### 1.6 A Concrete Numerical Example

Let us work through a tiny network with actual numbers. Consider a 1-hidden-layer network with $d\_0 = 2$, $d\_1 = 2$, $d\_2 = 1$, using sigmoid activation.

**Parameters:**

$$
W^{(1)} = \begin{pmatrix} 0.1 & 0.2 \\\\ 0.3 & 0.4 \end{pmatrix}, \quad \mathbf{b}^{(1)} = \begin{pmatrix} 0.1 \\\\ 0.1 \end{pmatrix}
$$

$$
W^{(2)} = \begin{pmatrix} 0.5 & 0.6 \end{pmatrix}, \quad b^{(2)} = 0.1
$$

Input $\mathbf{x} = (1, 2)^\top$, target $y = 1$.

**Forward pass:**

$$
\mathbf{a}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} = \begin{pmatrix} 0.1 \cdot 1 + 0.2 \cdot 2 + 0.1 \\\\ 0.3 \cdot 1 + 0.4 \cdot 2 + 0.1 \end{pmatrix} = \begin{pmatrix} 0.6 \\\\ 1.2 \end{pmatrix}
$$

$$
\mathbf{h} = \sigma(\mathbf{a}^{(1)}) = \begin{pmatrix} \sigma(0.6) \\\\ \sigma(1.2) \end{pmatrix} = \begin{pmatrix} 0.6457 \\\\ 0.7685 \end{pmatrix}
$$

$$
a^{(2)} = W^{(2)}\mathbf{h} + b^{(2)} = 0.5 \times 0.6457 + 0.6 \times 0.7685 + 0.1 = 0.884
$$

$$
\mathcal{L} = \frac{1}{2}(0.884 - 1)^2 = \frac{1}{2}(-0.116)^2 \approx 0.00673
$$

**Backward pass:**

$$
\delta^{(2)} = \hat{y} - y = 0.884 - 1 = -0.116
$$

$$
\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \delta^{(2)} \cdot \mathbf{h}^\top = -0.116 \times \begin{pmatrix} 0.6457 & 0.7685 \end{pmatrix} = \begin{pmatrix} -0.0749 & -0.0891 \end{pmatrix}
$$

$$
\frac{\partial \mathcal{L}}{\partial b^{(2)}} = -0.116
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}} = (W^{(2)})^\top \delta^{(2)} = \begin{pmatrix} 0.5 \\\\ 0.6 \end{pmatrix} \times (-0.116) = \begin{pmatrix} -0.058 \\\\ -0.0696 \end{pmatrix}
$$

$$
\sigma'(\mathbf{a}^{(1)}) = \mathbf{h} \odot (1 - \mathbf{h}) = \begin{pmatrix} 0.6457 \times 0.3543 \\\\ 0.7685 \times 0.2315 \end{pmatrix} = \begin{pmatrix} 0.2288 \\\\ 0.1779 \end{pmatrix}
$$

$$
\boldsymbol{\delta}^{(1)} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \odot \sigma'(\mathbf{a}^{(1)}) = \begin{pmatrix} -0.058 \times 0.2288 \\\\ -0.0696 \times 0.1779 \end{pmatrix} = \begin{pmatrix} -0.01327 \\\\ -0.01238 \end{pmatrix}
$$

$$
\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^\top = \begin{pmatrix} -0.01327 \\\\ -0.01238 \end{pmatrix} \begin{pmatrix} 1 & 2 \end{pmatrix} = \begin{pmatrix} -0.01327 & -0.02654 \\\\ -0.01238 & -0.02476 \end{pmatrix}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \begin{pmatrix} -0.01327 \\\\ -0.01238 \end{pmatrix}
$$

You now have gradients for every parameter in the network, computed with exactly one forward pass and one backward pass.

**Verification:** The negative gradients mean "increasing these parameters decreases the loss" — which makes sense, since the output (0.884) is below the target (1.0).

### 1.7 Forward Mode vs. Reverse Mode Automatic Differentiation

There are two ways to apply the chain rule through a computational graph:

**Forward mode:** Start from the inputs, propagate $\frac{\partial z\_i}{\partial \theta}$ forward through the graph. For one specific parameter $\theta\_j$, one forward pass gives you $\frac{\partial \mathcal{L}}{\partial \theta\_j}$. But for $d$ parameters, you need $d$ forward passes.

**Reverse mode (backpropagation):** Start from the output, propagate $\frac{\partial \mathcal{L}}{\partial z\_i}$ backward through the graph. One backward pass computes $\frac{\partial \mathcal{L}}{\partial \theta\_j}$ for *all* parameters simultaneously.

**Why reverse mode wins for neural networks:** A typical network has millions of parameters but a single scalar loss. Reverse mode gives all $d$ gradients in one backward pass (cost: roughly 2-3x one forward pass). Forward mode would require $d$ passes. The asymmetry is dramatic: for a network with $10^6$ parameters, reverse mode is $\sim 10^6$ times faster.

This is why PyTorch (and all deep learning frameworks) implement reverse-mode automatic differentiation. When you call `loss.backward()`, PyTorch walks the computational graph in reverse, computing and accumulating gradients for every parameter that has `requires_grad=True`.

**When would forward mode be better?** If you had a function with 1 input and $10^6$ outputs, forward mode would compute all partial derivatives in one pass. This is relevant in some scientific computing contexts but rare in machine learning.

### 1.8 Backpropagation with Mini-Batches

In practice, we compute the loss over a mini-batch of $B$ examples:

$$
\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \ell(f(\mathbf{x}_i; \boldsymbol{\theta}), y_i)
$$

Since differentiation is linear, the gradient of the batch loss is the average of the individual gradients:

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \nabla_{\boldsymbol{\theta}} \ell_i
$$

In implementation, the forward and backward passes operate on tensors with an extra batch dimension. The matrix multiplication $W\mathbf{x}$ becomes $WX$ where $X \in \mathbb{R}^{d\_0 \times B}$, and the outer product $\boldsymbol{\delta}\mathbf{h}^\top$ becomes $\Delta H^\top / B$ where the averaging produces the correct batch gradient. PyTorch handles this automatically.

---

## 2. Gradient Descent Variants

### 2.1 Batch Gradient Descent

The simplest approach: compute the gradient over the *entire* training set, then take one step.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{full}}
$$

where $\mathcal{L}\_{\text{full}} = \frac{1}{N} \sum\_{i=1}^{N} \ell\_i$.

**Pros:** The gradient is exact — no noise, stable convergence toward a minimum.

**Cons:** For large $N$, computing the full gradient is prohibitively expensive. You process the entire dataset just to take one step. Also, the lack of noise can trap you in sharp minima that generalize poorly.

### 2.2 Stochastic Gradient Descent (SGD)

Use a single randomly sampled example to estimate the gradient:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \ell(\mathbf{x}_i, y_i)
$$

The gradient of one example is a noisy but *unbiased* estimate of the full gradient:

$$
\mathbb{E}_i\left[\nabla \ell_i\right] = \nabla \mathcal{L}_{\text{full}}
$$

**Pros:** Very fast updates — each step requires only one example. The noise can help escape local minima and saddle points.

**Cons:** High variance in the gradient estimates means the parameter trajectory is erratic. Very small batch sizes underutilize GPU parallelism (GPUs are designed for large matrix operations).

### 2.3 Mini-Batch SGD

The practical compromise: use a mini-batch of $B$ examples (typically $B \in \lbrace 32, 64, 128, 256\rbrace $).

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla \ell_i
$$

The variance of the gradient estimate decreases as $O(1/B)$, but the computational cost per step increases linearly with $B$. The sweet spot balances gradient quality against computation.

Mini-batch SGD also exploits GPU parallelism: matrix multiplications over a batch are far more efficient than sequential single-example operations. A batch of 128 examples might take only 1.5x the time of a single example, not 128x.

When people say "SGD" in the deep learning literature, they almost always mean mini-batch SGD. We will follow this convention.

### 2.4 Momentum

Vanilla SGD oscillates in narrow valleys of the loss landscape — making rapid progress along the valley floor while zigzagging back and forth between the walls. Momentum fixes this with a beautifully simple idea.

**Intuition:** Imagine a ball rolling downhill on a bumpy surface. It does not just follow the steepest direction at each instant — it has *velocity* that accumulates. On a consistent downward slope, the ball speeds up. When the slope briefly reverses (a bump), the accumulated velocity carries the ball through.

**Formulation:**

$$
\mathbf{v}_{t+1} = \mu \mathbf{v}_t - \eta \nabla \mathcal{L}(\boldsymbol{\theta}_t)
$$
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \mathbf{v}_{t+1}
$$

where $\mu \in [0, 1)$ is the momentum coefficient (typically $\mu = 0.9$) and $\mathbf{v}\_0 = \mathbf{0}$.

The velocity vector $\mathbf{v}$ is an exponential moving average of past gradients. Gradients that consistently point in the same direction accumulate, while oscillating components cancel out.

**A concrete example:** Suppose the loss landscape is an elongated bowl — steep in the $x\_1$ direction but gentle in $x\_2$. Without momentum, SGD takes large steps in $x\_1$ (oscillating) and small steps in $x\_2$ (slow progress). With momentum, the $x\_1$ oscillations cancel in the velocity average, while $x\_2$ gradients accumulate. The effective trajectory is smoother and faster.

In the steady state (constant gradient), the effective step size is:

$$
\frac{\eta}{1 - \mu}
$$

With $\mu = 0.9$, that is a 10x amplification in consistent-gradient directions.

### 2.5 Nesterov Momentum

A clever variant due to Yurii Nesterov: instead of computing the gradient at the current position, compute it at the *anticipated* future position — where momentum would take you if the gradient were zero:

$$
\mathbf{v}_{t+1} = \mu \mathbf{v}_t - \eta \nabla \mathcal{L}(\boldsymbol{\theta}_t + \mu \mathbf{v}_t)
$$
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \mathbf{v}_{t+1}
$$

**Intuition:** "Look ahead to where momentum is taking you, then correct." If momentum is about to carry you past a minimum, the gradient at the look-ahead point already points back — giving a natural braking effect. Standard momentum only realizes the overshoot after it happens.

Nesterov momentum has provably better convergence rates for convex functions. In practice, it gives a modest but consistent improvement over standard momentum.

### 2.6 Why SGD Generalizes Better

A surprising and important empirical finding: SGD with momentum often finds solutions that *generalize* better to new data than solutions found by more sophisticated optimizers. Several explanations have been proposed:

**Implicit regularization through noise.** The stochastic gradient is a noisy estimate of the true gradient. This noise acts as an implicit regularizer, preventing the optimizer from precisely fitting every detail of the training data.

**Preference for flat minima.** Sharp minima — deep but narrow valleys — are unstable under gradient noise. A small perturbation can kick the parameters out. Flat minima — broad basins where the loss is low over a large region — are stable. SGD's noise biases it toward flat minima, which tend to generalize better because the test loss is close to the training loss throughout the basin.

**Escaping sharp minima.** Even if SGD momentarily enters a sharp minimum, the gradient noise can push it out. A flat minimum is "sticky" — the noise is not large enough to escape its broad basin.

This has practical consequences: for training large models from scratch (where generalization matters most), SGD with momentum is often preferred. For fine-tuning pre-trained models or in settings where fast convergence matters more than final generalization, Adam is preferred.

---

## 3. Adaptive Learning Rate Methods

The fundamental problem with vanilla SGD: the same learning rate $\eta$ applies to all parameters. But different parameters may need very different learning rates — a parameter that receives sparse, infrequent gradient signals needs larger steps, while a parameter with frequent, large gradients needs smaller steps.

### 3.1 AdaGrad

**Idea:** Accumulate the sum of squared gradients for each parameter, and scale the learning rate inversely by the square root of this sum.

$$
G_{t+1,j} = G_{t,j} + g_{t,j}^2
$$
$$
\theta_{t+1,j} = \theta_{t,j} - \frac{\eta}{\sqrt{G_{t+1,j}} + \epsilon} \cdot g_{t,j}
$$

where $g\_{t,j} = \frac{\partial \mathcal{L}}{\partial \theta\_j}\big|\_t$ and $\epsilon \approx 10^{-8}$ prevents division by zero.

**Effect:** Parameters with large accumulated gradients get smaller learning rates. Parameters with small accumulated gradients get larger learning rates. This is exactly the per-parameter adaptation we wanted.

**The fatal flaw:** $G\_t$ only grows — it never shrinks. Over time, the effective learning rate decays to zero for all parameters, and learning stops completely. This is fine for convex problems (where you want to converge to a fixed point) but catastrophic for non-convex deep learning, where you need to keep exploring.

### 3.2 RMSProp

Geoffrey Hinton's fix (proposed in a Coursera lecture slide, never formally published — a charming piece of deep learning folklore): replace the cumulative sum with an exponential moving average.

$$
v_{t+1,j} = \beta \, v_{t,j} + (1 - \beta) \, g_{t,j}^2
$$
$$
\theta_{t+1,j} = \theta_{t,j} - \frac{\eta}{\sqrt{v_{t+1,j}} + \epsilon} \cdot g_{t,j}
$$

where $\beta \approx 0.9$ or $0.99$. Now $v\_t$ is a running estimate of the recent second moment $\mathbb{E}[g^2]$, and it can both grow and shrink as the gradient statistics change. The effective learning rate adapts without collapsing to zero.

### 3.3 Adam (Adaptive Moment Estimation)

Adam combines the best of momentum and RMSProp, plus a crucial bias correction term. It is the default optimizer for most deep learning tasks. Let us build it piece by piece.

**Step 1: First moment estimate** (like momentum — track the exponential moving average of gradients):

$$
m_{t+1} = \beta_1 m_t + (1 - \beta_1) g_t
$$

This estimates $\mathbb{E}[g\_t]$ — the direction of the gradient.

**Step 2: Second moment estimate** (like RMSProp — track the exponential moving average of squared gradients):

$$
v_{t+1} = \beta_2 v_t + (1 - \beta_2) g_t^2
$$

This estimates $\mathbb{E}[g\_t^2]$ — the magnitude of the gradient.

**Step 3: Bias correction.** Both $m$ and $v$ are initialized to zero. In the early steps, they are biased toward zero. To see why, expand the recurrence at $t = 1$:

$$
m_1 = (1 - \beta_1) g_1
$$

With $\beta\_1 = 0.9$, this is just $0.1 \cdot g\_1$ — an order of magnitude too small. More generally, unrolling the recurrence shows:

$$
\mathbb{E}[m_t] = \mathbb{E}[g_t] \cdot (1 - \beta_1^t)
$$

So we divide by the bias factor:

$$
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}, \qquad \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}
$$

As $t \to \infty$, the correction factors approach 1 and become negligible.

**Step 4: Update.**

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \cdot \hat{m}_{t+1}
$$

The numerator $\hat{m}$ provides momentum (direction). The denominator $\sqrt{\hat{v}} + \epsilon$ provides per-parameter learning rate scaling (adapting to gradient magnitudes). The ratio $\hat{m}/\sqrt{\hat{v}}$ is approximately the "signal-to-noise ratio" of the gradient — Adam takes large steps when the gradient is consistent and small steps when it is noisy.

**Default hyperparameters** (from the original paper by Kingma & Ba, 2015):
- $\beta\_1 = 0.9$ (momentum decay)
- $\beta\_2 = 0.999$ (second moment decay)
- $\eta = 0.001$ (learning rate)
- $\epsilon = 10^{-8}$

These defaults work well for the vast majority of problems. Resist the urge to tune them unless you have specific evidence that they are suboptimal.

### 3.4 AdamW: Decoupled Weight Decay

A subtle but important distinction: L2 regularization and weight decay are *not the same thing* when using adaptive optimizers like Adam.

In SGD, adding $\frac{\lambda}{2}\Vert \boldsymbol{\theta}\Vert ^2$ to the loss produces a gradient contribution of $\lambda\boldsymbol{\theta}$, making the update:

$$
\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta g_t
$$

The weight decay factor $(1 - \eta\lambda)$ is uniform across all parameters. But in Adam, the gradient $g\_t + \lambda\theta\_t$ gets divided by $\sqrt{\hat{v}}$ — so the effective weight decay varies per parameter. Parameters with large gradient magnitudes get *less* weight decay, which is not what we want.

**AdamW** (Loshchilov & Hutter, 2019) decouples weight decay from the adaptive gradient step:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon} + \lambda \theta_t \right)
$$

The weight decay $\lambda\theta\_t$ is applied uniformly, independent of the gradient statistics. This is the version of Adam that should be used in practice. In PyTorch: `torch.optim.AdamW`.

### 3.5 When to Use What

| Optimizer | Best for | Notes |
|-----------|----------|-------|
| SGD + momentum | Training from scratch, when tuning LR schedule | Often best final accuracy |
| Adam / AdamW | Quick convergence, fine-tuning, NLP/transformers | Good default choice |
| RMSProp | RNNs (historical preference) | Largely superseded by Adam |

A reasonable default strategy: start with AdamW at $\eta = 10^{-3}$. If you need to squeeze out the last fraction of accuracy and are willing to tune a learning rate schedule, switch to SGD + momentum.

---

## 4. Regularization

Neural networks have enormous capacity — a typical MNIST classifier has 100x more parameters than training examples. Without regularization, the network memorizes the training data rather than learning general patterns. Zhang et al. (2017) showed that standard architectures can fit *random labels* with zero training error — a dramatic demonstration that memorization is the default mode.

**Regularization** is any technique that constrains the model to prefer simpler, more generalizable solutions.

### 4.1 L2 Regularization (Weight Decay)

Add a penalty on the squared magnitude of the weights:

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \sum_j \theta_j^2
$$

The gradient of the regularization term is $\lambda\theta\_j$, so the update becomes:

$$
\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta \frac{\partial \mathcal{L}_{\text{data}}}{\partial \theta_t}
$$

The factor $(1 - \eta\lambda)$ shrinks the weights toward zero at each step — hence "weight decay." Typical values: $\lambda \in [10^{-4}, 10^{-2}]$.

**Intuition:** Large weights mean the network has found extreme, specific relationships in the training data — relationships that are likely to be noise rather than signal. Keeping weights small forces the network to rely on broad, moderate patterns that generalize.

**Geometric perspective:** L2 regularization constrains the parameters to lie near the origin. The optimal solution balances data fit (pulling parameters toward the loss minimum) against the penalty (pulling parameters toward zero). From Week 1, recall that the L2 ball is smooth — all directions are penalized equally.

### 4.2 L1 Regularization

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \lambda \sum_j |\theta_j|
$$

L1 encourages *sparsity* — many weights become exactly zero. As we discussed in Week 1, the L1 ball has corners aligned with the axes, and the constrained optimum tends to land on these corners (exactly zero in some coordinates).

L1 is less common for neural network weights than L2 (it makes optimization harder and networks are not typically sparse). But L1 sparsity is *central* to the sparse autoencoders we will study in Weeks 9-10. When we add an L1 penalty to autoencoder activations (not weights), we get a sparse representation — exactly the sparse coding we are building toward.

### 4.3 Dropout

**The idea:** During training, randomly zero out each hidden unit with probability $p$ (typically $p = 0.5$ for hidden layers). At test time, use all units but scale their outputs by $(1 - p)$.

**Implementation (inverted dropout — the standard approach):**

During training, for each mini-batch, sample a binary mask $\mathbf{m} \sim \text{Bernoulli}(1 - p)$ and compute:

$$
\tilde{\mathbf{h}} = \frac{1}{1-p} \cdot \mathbf{m} \odot \mathbf{h}
$$

The $\frac{1}{1-p}$ scaling ensures $\mathbb{E}[\tilde{\mathbf{h}}] = \mathbf{h}$, so no adjustment is needed at test time — just use $\mathbf{h}$ directly.

```python
# Dropout in PyTorch (during training)
mask = (torch.rand_like(h) > p).float()
h_dropped = h * mask / (1 - p)

# Or simply:
h_dropped = F.dropout(h, p=p, training=self.training)
```

**Why dropout works — three perspectives:**

1. **Ensemble of sub-networks.** Each dropout mask selects a different subnetwork. With $d$ hidden units and dropout rate 0.5, there are $2^d$ possible subnetworks. Training with dropout approximately trains all $2^d$ subnetworks simultaneously (they share weights). At test time, using all units with scaling approximates the geometric mean of these subnetwork predictions. This is a massively powerful ensemble, for free.

2. **Preventing co-adaptation.** Without dropout, hidden units can develop intricate, co-dependent representations: "unit 7 fires only when units 3 and 12 also fire." These complex relationships are likely to be training-set-specific. Dropout breaks co-adaptation by randomly removing units, forcing each unit to learn features that are useful *on their own* — robust features that transfer to new data.

3. **Noise injection as regularization.** Dropout injects multiplicative noise into the hidden layers. This noise prevents the network from relying too precisely on any particular activation pattern, smoothing the learned function.

**When to use dropout:** It is most effective in fully-connected layers with many parameters. It is less commonly used in convolutional layers (where batch normalization is preferred) or in very small networks (where the capacity is already limited).

### 4.4 Batch Normalization

**The problem it solves:** During training, the distribution of each layer's inputs changes continuously because the preceding layers' parameters are being updated. This "internal covariate shift" makes optimization difficult — each layer is trying to learn on a moving target.

**Batch normalization** (Ioffe & Szegedy, 2015) normalizes activations within each mini-batch:

Given a mini-batch of pre-activations $\lbrace z\_i\rbrace \_{i=1}^B$ at some layer:

$$
\mu_B = \frac{1}{B}\sum_{i=1}^B z_i, \qquad \sigma_B^2 = \frac{1}{B}\sum_{i=1}^B (z_i - \mu_B)^2
$$

$$
\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

Then apply learned scale and shift parameters:

$$
\tilde{z}_i = \gamma \hat{z}_i + \beta
$$

where $\gamma$ and $\beta$ are learnable. The normalization enforces zero mean and unit variance, but the learned $\gamma$ and $\beta$ allow the network to undo the normalization if that is optimal — so BatchNorm never hurts expressiveness.

**Benefits:**
- **Allows higher learning rates.** Normalization keeps activation magnitudes stable, preventing gradients from exploding or vanishing.
- **Reduces sensitivity to initialization.** The normalization acts as a reset at each layer.
- **Regularization effect.** The batch statistics are noisy estimates of the true mean and variance, injecting noise similar to dropout.
- **Faster convergence.** Empirically, networks with BatchNorm train significantly faster.

**At test time:** Replace $\mu\_B$ and $\sigma\_B^2$ with running averages accumulated during training (via exponential moving average). This makes the model deterministic at test time.

**Where to place it:** Typically after the linear transformation and before the activation: $\sigma(\text{BN}(W\mathbf{x}))$. Note that the bias term in $W\mathbf{x} + \mathbf{b}$ is redundant with the learned shift $\beta$, so it is typically omitted.

### 4.5 Early Stopping

The simplest and most effective regularization technique: monitor the loss on a held-out validation set during training. When validation loss stops improving (or starts increasing), stop training.

**Implementation:**
1. After each epoch, evaluate the model on the validation set
2. If validation loss improves, save the model checkpoint
3. If validation loss has not improved for $k$ epochs ("patience"), stop training
4. Return the saved checkpoint with the best validation loss

**Why it works:** In the early phase of training, the network learns general patterns — patterns that appear in both training and validation data. In the later phase, it begins to memorize training-specific details that do not transfer. Early stopping halts training at this transition point.

**Relation to L2 regularization:** There is a deep connection. For linear models trained with gradient descent, early stopping is mathematically equivalent to L2 regularization, where the number of training steps plays the role of the inverse regularization strength $1/\lambda$. For nonlinear networks, the relationship is approximate but the intuition holds.

---

## 5. Loss Landscapes and Optimization Challenges

### 5.1 Visualizing the Loss Landscape

The loss function $\mathcal{L}(\boldsymbol{\theta})$ defines a surface over the high-dimensional parameter space. We cannot visualize this surface directly (it lives in millions of dimensions), but we can take 2D slices.

Li et al. (2018) developed "filter normalization" to produce meaningful 2D cross-sections. Their visualizations reveal:
- ResNets have remarkably smooth loss landscapes (thanks to skip connections)
- Networks without skip connections have chaotic, rough landscapes
- Wider networks have smoother landscapes than narrow ones

These visualizations, while not rigorous proofs, provide powerful intuition.

### 5.2 Local Minima vs. Saddle Points in High Dimensions

A common misconception: "neural network optimization is hard because of local minima." The reality is more interesting.

At any critical point ($\nabla \mathcal{L} = 0$), the Hessian matrix $H$ tells us about the local curvature:
- If all eigenvalues of $H$ are positive: **local minimum** (the loss curves upward in every direction)
- If all eigenvalues are negative: **local maximum**
- If eigenvalues have mixed signs: **saddle point** (curves up in some directions, down in others)

**The key insight:** In $d$-dimensional space, each eigenvalue is independently positive or negative (roughly). The probability that all $d$ eigenvalues are positive is $\sim (1/2)^d$. For a network with $d = 10^6$ parameters, this is $(1/2)^{10^6}$ — incomprehensibly small.

The overwhelming majority of critical points are saddle points. True local minima are exceedingly rare, and when they do exist, they tend to have loss values close to the global minimum (Choromanska et al., 2015).

**The practical implication:** We do not need to worry about getting trapped in bad local minima. The real challenge is navigating through the saddle points efficiently. Fortunately, SGD with momentum does this naturally — the noise helps escape saddle points, and momentum carries the parameters through.

### 5.3 Sharp vs. Flat Minima

Not all minima are equally good. A **sharp minimum** sits in a narrow valley: the loss is low at the minimum but rises steeply in nearby directions. A **flat minimum** sits in a broad basin: the loss is low over a wide region of parameter space.

**Why flat minima generalize better:** The training loss and test loss differ by a small perturbation (different data). If the minimum is flat, the test loss is similar to the training loss throughout the basin. If the minimum is sharp, even a small perturbation (switching from training to test data) can increase the loss dramatically.

SGD with large learning rates and small batch sizes tends to find flat minima. This is another reason SGD generalizes well despite being noisier than batch gradient descent.

### 5.4 Vanishing and Exploding Gradients

In a deep network with $L$ layers, the gradient of the loss with respect to the first layer's parameters involves a product of $L$ Jacobian matrices:

$$
\frac{\partial \mathcal{L}}{\partial W^{(1)}} \propto \prod_{\ell=2}^{L} D^{(\ell)} W^{(\ell)}
$$

where $D^{(\ell)}$ is the diagonal matrix of activation derivatives at layer $\ell$.

If the spectral norm of each factor $\Vert D^{(\ell)} W^{(\ell)}\Vert $ is consistently $< 1$: the product vanishes exponentially. The first layer's gradients are negligibly small — **vanishing gradients**.

If the spectral norm is consistently $> 1$: the product explodes exponentially — **exploding gradients**.

**Solutions to vanishing gradients:**

- **ReLU activation:** For $x > 0$, $\text{ReLU}'(x) = 1$, so the activation derivative does not attenuate the gradient. (But for $x < 0$, $\text{ReLU}'(x) = 0$ — the "dead ReLU" problem. Variants like Leaky ReLU ($\text{ReLU}'(x) = 0.01$ for $x < 0$) address this.)

- **Careful initialization:** He initialization (for ReLU layers): $W\_{ij} \sim \mathcal{N}(0, 2/d\_{\text{in}})$. Xavier/Glorot initialization (for sigmoid/tanh): $W\_{ij} \sim \mathcal{N}(0, 1/d\_{\text{in}})$. These are designed so that the variance of activations (forward pass) and gradients (backward pass) remains approximately constant across layers.

- **Skip connections (preview):** ResNets add the input to the output: $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$. The gradient through the skip connection is simply the identity — it flows unattenuated regardless of depth. This is the key innovation that enabled training networks with hundreds of layers.

- **Batch normalization:** By renormalizing activations at each layer, BatchNorm prevents the compounding of small or large values that leads to vanishing/exploding gradients.

**Solutions to exploding gradients:**

- **Gradient clipping:** Cap the gradient norm before applying the update (see Section 6.2).

### 5.5 The Lottery Ticket Hypothesis

A provocative conjecture by Frankle & Carlin (2019): within a randomly initialized dense network, there exists a sparse subnetwork (the "winning ticket") that, when trained from its original initialization, can match the full network's accuracy.

The procedure: (1) train the full network, (2) prune small-magnitude weights (keep, say, 10%), (3) reset the surviving weights to their *original* random values, (4) retrain. The pruned network matches or exceeds the dense network's performance.

**Why this matters for our course:** It suggests that neural networks are highly over-parameterized, and that the "real" computation is happening in a sparse subset. This connects directly to the themes of sparsity and sparse representations that are central to the second half of the course.

---

## 6. Practical Training Tips

### 6.1 Learning Rate Schedules

The learning rate is the single most important hyperparameter. A learning rate that is too large causes oscillation or divergence; too small means painfully slow convergence. And the optimal learning rate changes during training — you want larger steps early (to explore) and smaller steps later (to converge precisely).

**Step decay:** Multiply $\eta$ by a factor (e.g., 0.1) at predetermined epochs. Simple and effective. Common in computer vision (e.g., reduce LR at epochs 30, 60, 90 for a 100-epoch run).

**Cosine annealing:** Smoothly decrease $\eta$ following a cosine curve:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

Starts at $\eta\_{\max}$ and smoothly decays to $\eta\_{\min}$ over $T$ steps. No sharp jumps, no hyperparameter for when to decay.

**Linear warmup:** Start with $\eta = 0$ and linearly increase to the target learning rate over the first $W$ steps:

$$
\eta_t = \eta_{\text{target}} \cdot \frac{t}{W} \quad \text{for } t \leq W
$$

This is especially important when using large batch sizes (the gradient estimates are unreliable for the first few steps from a random initialization) and with Adam (the second moment estimates need time to stabilize).

**One-cycle policy** (Smith, 2018): Warmup from small $\eta$ to large $\eta$, then anneal back down, all within one training run. Often achieves the best results in the shortest time. Available in PyTorch as `torch.optim.lr_scheduler.OneCycleLR`.

### 6.2 Gradient Clipping

A simple defense against exploding gradients: if the gradient norm exceeds a threshold $c$, rescale the entire gradient vector to have norm $c$:

$$
\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \Vert \mathbf{g}\Vert  \leq c \\\\ \frac{c}{\Vert \mathbf{g}\Vert } \mathbf{g} & \text{if } \Vert \mathbf{g}\Vert  > c \end{cases}
$$

This preserves the gradient *direction* while capping its magnitude. Typical values: $c \in [1, 5]$.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Gradient clipping is essential for training RNNs and often helpful for transformers. For standard feedforward networks with BatchNorm, it is less critical but never harmful.

### 6.3 The Debugging Playbook

When your training run is not working, proceed systematically:

1. **Verify the data pipeline.** Visualize a batch of inputs and labels. Are they correct? Is normalization applied properly?

2. **Overfit one batch first.** Take a single mini-batch (8-16 examples) and train until the loss approaches zero. This should happen within 50-100 iterations. If it does not, you have a bug — not an optimization or generalization problem. Check:
   - Is the loss function correct? (Common mistake: using `nn.CrossEntropyLoss` with one-hot labels instead of integer labels)
   - Is `optimizer.zero_grad()` called before each backward pass?
   - Is the model in training mode? (`model.train()`)
   - Are gradients flowing? (Print `[p.grad.abs().mean() for p in model.parameters()]`)

3. **Scale to full data.** Training loss should decrease steadily. If it plateaus early, try a higher learning rate. If it oscillates or diverges, try a lower one.

4. **Monitor the train/val gap.** If training loss is low but validation loss is high: overfit. Add regularization (dropout, weight decay, data augmentation, or just reduce model size). If both are high: underfit. Increase model capacity or train longer.

5. **Check gradient health per layer.** Plot mean and std of gradients. Vanishing gradients in early layers? Try skip connections, better initialization, or batch norm.

### 6.4 Hyperparameter Tuning Priority

When you have limited time, tune hyperparameters in this order (most impactful first):

1. **Learning rate.** Try values on a log scale: $10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}$. The optimal value is usually clear from the training curves.
2. **Batch size.** Use the largest batch size that fits in memory. Then adjust the learning rate (the "linear scaling rule": double the batch size, double the learning rate).
3. **Architecture.** Number of layers, hidden dimension. Start simple and increase until validation loss stops improving.
4. **Regularization strength.** Dropout rate ($p \in \lbrace 0.1, 0.3, 0.5\rbrace $), weight decay ($\lambda \in \lbrace 10^{-4}, 10^{-3}, 10^{-2}\rbrace $). Only add when overfitting is observed.
5. **Optimizer hyperparameters.** Adam's $\beta\_1, \beta\_2$ rarely need tuning. Momentum coefficient $\mu$ is almost always 0.9.

---

## 7. Summary and Looking Ahead

We now have the complete toolkit to train neural networks:

- **Backpropagation** computes gradients efficiently via reverse-mode automatic differentiation — one backward pass gives gradients for all parameters
- **SGD with momentum** or **Adam** uses those gradients to update parameters, with momentum smoothing the trajectory and adaptive methods scaling per-parameter learning rates
- **Regularization** (dropout, weight decay, batch normalization, early stopping) prevents the network from memorizing the training data
- **Practical heuristics** (overfit one batch, learning rate schedules, gradient clipping, systematic debugging) bridge the gap between theory and working code

A recurring theme: the interplay between exploration (noise, large learning rates) and exploitation (convergence, small learning rates). Good training requires both — exploring the loss landscape broadly before converging precisely.

Next week, we shift from "how to train a network" to "what does a network learn?" We will study **representation learning** — the idea that intermediate layers discover useful features of the data. This will lead us to ask: can we learn useful representations without labels? The answer is the autoencoder, and it is where our course begins its main arc.

---

## References

- Rumelhart, Hinton, Williams, "Learning Representations by Back-propagating Errors" (1986)
- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015)
- Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training" (2015)
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014)
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
- Li et al., "Visualizing the Loss Landscape of Neural Nets" (2018)
- Frankle & Carlin, "The Lottery Ticket Hypothesis" (2019)
- Smith, "A Disciplined Approach to Neural Network Hyper-Parameters" (2018)
- Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization" (2017)
- Goodfellow, Bengio, Courville, *Deep Learning*, Chapters 6-8 (2016)
