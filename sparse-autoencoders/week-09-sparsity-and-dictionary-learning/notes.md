# Week 9: Sparsity and Dictionary Learning

> *"Everything should be made as simple as possible, but not simpler."*
> -- Albert Einstein (attributed)

---

## Overview

This is the week where the course pivots. Everything we have built so far -- linear algebra, optimization, autoencoders, regularization -- converges here on a single, powerful idea: **sparse representation**.

The claim is bold: given a rich enough dictionary of elementary patterns, any signal can be represented as a sparse combination of those patterns. "Sparse" means that only a small fraction of the dictionary elements are used for any given signal. This idea, which emerged independently in neuroscience, signal processing, and statistics, turns out to be the intellectual foundation of the sparse autoencoders we will build in Week 10 and apply to interpretability in Weeks 11-12.

We will develop the theory from three perspectives -- biological, computational, and statistical -- then derive the core algorithms (ISTA for sparse coding, alternating optimization for dictionary learning), and finally connect everything to the sparse autoencoders that are the heart of this course.

### Prerequisites
- Week 1: L1 and L2 norms, optimization, gradient descent
- Week 2: MLE, MAP estimation, Laplace distribution
- Week 4: Backpropagation, regularization
- Week 7: Regularized autoencoders (especially the sparsity preview)

---

## 1. Why Sparsity? Three Arguments

### 1.1 The Biological Argument

In the mammalian visual cortex, neurons are remarkably sparse in their firing patterns. At any given moment, only about 1-4% of neurons in the primary visual cortex (V1) are active. This is not a design flaw -- it appears to be a fundamental feature.

**Why would the brain use sparse codes?**

**Metabolic efficiency.** Neurons are expensive to operate. Each action potential requires ion pumps to restore the electrochemical gradient. A coding scheme where most neurons are silent most of the time is energetically cheap.

**Storage capacity.** Consider $n$ neurons, each either on or off. If all neurons fire independently with probability 0.5, there are $2^n$ possible patterns, but distinguishing them requires precise measurement of the entire population. If only $k \ll n$ neurons fire at a time, there are $\binom{n}{k}$ patterns, which is still exponentially large for moderate $k$, but each pattern is easily identifiable (just check which $k$ neurons are active).

For example, with $n = 1000$ and $k = 10$: $\binom{1000}{10} \approx 2.63 \times 10^{23}$. That is 23 orders of magnitude of distinct patterns, using only 1% activity.

**Associative memory.** Sparse representations are easier to store and retrieve in associative memory networks (Kanerva 1988). Two sparse patterns are unlikely to overlap much, making them easy to distinguish. Two dense patterns share many active units, making them confusable.

Olshausen and Field (1996) made a remarkable discovery: if you take natural images, decompose them into small patches, and find the sparse code that best represents each patch, the dictionary elements you learn look like **Gabor filters** -- oriented edge detectors at various scales and positions. These are strikingly similar to the receptive fields of simple cells in V1.

The brain, it seems, discovered sparse coding hundreds of millions of years before we did.

### 1.2 The Computational Argument

Sparse representations are computationally efficient in several ways:

**Storage.** A sparse vector $z \in \mathbb{R}^{d_z}$ with only $k$ nonzero entries can be stored using $O(k)$ memory instead of $O(d_z)$. When $k \ll d_z$, this is a dramatic compression.

**Computation.** Matrix-vector products $Dz$ (where $D$ is a dictionary) reduce from $O(d_x \cdot d_z)$ to $O(d_x \cdot k)$ when $z$ is $k$-sparse, because only $k$ columns of $D$ contribute.

**Communication.** Transmitting a sparse code requires only the positions and values of the nonzero entries: $O(k \log d_z + k)$ bits instead of $O(d_z)$.

**Compressed sensing.** The remarkable theory of compressed sensing (Candes and Tao, 2005; Donoho, 2006) shows that if a signal is sparse in some basis, it can be recovered from far fewer measurements than its ambient dimension. This has revolutionized MRI, astronomy, and seismology.

### 1.3 The Statistical Argument

Sparsity is a powerful inductive bias for learning.

**Regularization.** In regression, adding an L1 penalty (LASSO) produces sparse coefficient vectors, which perform feature selection automatically. Models with fewer active features are simpler, more interpretable, and less prone to overfitting.

**Interpretability.** If a representation is sparse, you can ask "which features are active?" for any input and get a short, meaningful answer. If 5 out of 10000 features are active, you can examine those 5 and understand the representation. If all 10000 are active, interpretation is hopeless.

**Disentanglement.** Sparse features tend to be disentangled -- each feature captures one factor of variation. This is because the sparsity constraint forces features to be "turned off" for most inputs, meaning each feature must be specifically relevant to the inputs that activate it.

**Superposition.** Recent work on mechanistic interpretability (Elhage et al., 2022, which we will study in Week 11) shows that neural networks pack more features than they have dimensions by using **superposition** -- relying on the sparsity of feature activations to avoid interference. Sparse autoencoders can "unpack" these superimposed features into a larger, sparser representation where each feature is interpretable.

---

## 2. L1 Regularization and the LASSO

### 2.1 Recall: L1 vs L2 Geometry

In Week 1, we studied the geometry of L1 and L2 norms. Let us now connect this geometry to sparsity.

The **L2 ball** $\{x : \|x\|_2 \leq r\}$ is a sphere (circle in 2D). The **L1 ball** $\{x : \|x\|_1 \leq r\}$ is a diamond (a rotated square in 2D, a cross-polytope in higher dimensions).

The crucial difference: the L1 ball has **corners on the coordinate axes**. The L2 ball is smooth everywhere.

### 2.2 The LASSO

The LASSO (Least Absolute Shrinkage and Selection Operator, Tibshirani 1996) adds an L1 penalty to the least squares objective:

$$\min_{\beta} \frac{1}{2} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

where $y \in \mathbb{R}^n$ is the response, $X \in \mathbb{R}^{n \times p}$ is the design matrix, and $\beta \in \mathbb{R}^p$ are the coefficients.

Compare to **Ridge regression** (L2 penalty):

$$\min_{\beta} \frac{1}{2} \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$$

### 2.3 Why L1 Produces Exact Zeros: The Geometric Argument

Consider the constrained form of the LASSO:

$$\min_{\beta} \frac{1}{2}\|y - X\beta\|_2^2 \quad \text{subject to} \quad \|\beta\|_1 \leq t$$

Geometrically, we are finding the point in the L1 ball that is closest to the unconstrained optimum $\hat{\beta}_{\text{OLS}}$. Equivalently, we are shrinking the elliptical contours of the quadratic loss until they touch the constraint set.

**For L2 (Ridge):** The constraint set is a circle. The contour ellipses will typically first touch the circle at a point that has no coordinates exactly zero. The contact point is a smooth tangency.

**For L1 (LASSO):** The constraint set is a diamond with corners on the axes. The contour ellipses will typically first touch the diamond at a corner or edge. A corner lies on a coordinate axis, meaning some coordinates are exactly zero. This is sparsity.

In higher dimensions, the L1 ball has increasingly many corners (and faces, edges, etc.) lying on coordinate subspaces. The probability that the solution sits on a face where some coordinates are exactly zero grows with dimension. This is why L1 regularization produces sparse solutions reliably.

### 2.4 The Bayesian Perspective: Laplace Prior

Recall from Week 2 that regularization can be interpreted as MAP estimation with a prior.

**L2 regularization** = MAP with Gaussian prior: $p(\beta) \propto e^{-\lambda \|\beta\|_2^2}$
**L1 regularization** = MAP with Laplace prior: $p(\beta) \propto e^{-\lambda \|\beta\|_1}$

The Laplace distribution:

$$p(\beta_j) = \frac{\lambda}{2} e^{-\lambda |\beta_j|}$$

This distribution has a sharp peak at zero (unlike the Gaussian, which is smooth at zero). The sharp peak means the prior strongly favours coefficients that are exactly zero -- it "believes" in sparsity.

### 2.5 A Numerical Example

Consider a simple problem: $p = 5$ features, $n = 20$ observations. The true model is $y = 3x_1 + 2x_4 + \text{noise}$, so only 2 of 5 features are relevant.

**OLS:** $\hat{\beta} = (2.8, 0.3, -0.2, 1.9, 0.1)$ -- all nonzero due to noise.

**Ridge ($\lambda = 1$):** $\hat{\beta} = (2.1, 0.2, -0.15, 1.5, 0.08)$ -- all nonzero but shrunk toward zero.

**LASSO ($\lambda = 1$):** $\hat{\beta} = (2.5, 0, 0, 1.7, 0)$ -- exactly three coefficients are zero! The LASSO has performed feature selection.

This is the power of L1: not just shrinkage, but *selection*.

---

## 3. The Soft-Thresholding Operator

### 3.1 The Proximal Operator of L1

The LASSO objective is not differentiable at zero (because $|\beta_j|$ has a kink at zero). We cannot simply set the gradient to zero. Instead, we use the **proximal operator** framework from convex optimization.

For the L1 norm, the proximal operator is the **soft-thresholding** function:

$$\text{prox}_{\lambda \|\cdot\|_1}(v) = S_\lambda(v) = \text{sign}(v) \cdot \max(|v| - \lambda, 0)$$

Applied element-wise, this means:

$$[S_\lambda(v)]_j = \begin{cases} v_j - \lambda & \text{if } v_j > \lambda \\ 0 & \text{if } |v_j| \leq \lambda \\ v_j + \lambda & \text{if } v_j < -\lambda \end{cases}$$

The soft-thresholding operator does two things:
1. **Thresholds:** Values within $[-\lambda, \lambda]$ are set exactly to zero
2. **Shrinks:** Values outside this range are pulled toward zero by $\lambda$

Compare to **hard thresholding**: $H_\lambda(v)_j = v_j \cdot \mathbf{1}[|v_j| > \lambda]$, which sets small values to zero but does not shrink large ones. Soft thresholding is smoother and has better theoretical properties.

### 3.2 Derivation of Soft Thresholding

The proximal operator of $h(x) = \lambda \|x\|_1$ is defined as:

$$\text{prox}_h(v) = \arg\min_x \left\{ \frac{1}{2}\|x - v\|^2 + \lambda \|x\|_1 \right\}$$

Since $\|x\|_1 = \sum_j |x_j|$, this separates across dimensions:

$$[\text{prox}_h(v)]_j = \arg\min_{x_j} \left\{ \frac{1}{2}(x_j - v_j)^2 + \lambda |x_j| \right\}$$

For a single dimension, define $g(x) = \frac{1}{2}(x - v)^2 + \lambda |x|$.

**Case 1: $x > 0$.** Then $g(x) = \frac{1}{2}(x-v)^2 + \lambda x$. Setting $g'(x) = x - v + \lambda = 0$ gives $x = v - \lambda$. This is valid (positive) when $v > \lambda$.

**Case 2: $x < 0$.** Then $g(x) = \frac{1}{2}(x-v)^2 - \lambda x$. Setting $g'(x) = x - v - \lambda = 0$ gives $x = v + \lambda$. This is valid (negative) when $v < -\lambda$.

**Case 3: $|v| \leq \lambda$.** Neither case above gives a valid minimum in its region. The minimum is at $x = 0$ (the boundary). You can verify: $g(0) = \frac{1}{2}v^2$, while $g(v - \lambda) = \frac{1}{2}\lambda^2 + \lambda(v - \lambda) = \lambda v - \frac{1}{2}\lambda^2 \geq \frac{1}{2}v^2$ when $|v| \leq \lambda$.

Combining all three cases gives the soft-thresholding formula. $\square$

### 3.3 Visualizing Soft Thresholding

For $\lambda = 1$:

```
Input v:   -3   -2   -1   -0.5   0   0.5   1   2   3
Output:    -2   -1    0    0     0   0     0   1   2
```

The function is a "dead zone" around zero: everything within $[-\lambda, \lambda]$ collapses to zero, and everything outside is shifted toward zero by $\lambda$. This is the mechanism by which L1 regularization produces exact sparsity.

---

## 4. Sparse Coding

### 4.1 The Classic Formulation

Sparse coding, introduced by Olshausen and Field (1996), is the problem of representing a signal $x \in \mathbb{R}^{d_x}$ as a sparse linear combination of dictionary atoms:

$$x \approx Dz = \sum_{j=1}^{d_z} z_j d_j$$

where $D = [d_1 | d_2 | \cdots | d_{d_z}] \in \mathbb{R}^{d_x \times d_z}$ is the **dictionary** (each column is an atom) and $z \in \mathbb{R}^{d_z}$ is the **sparse code** with most entries zero.

Typically $d_z > d_x$ -- the dictionary is **overcomplete**. There are more dictionary atoms than the dimensionality of the signal. This means the representation $x = Dz$ is underdetermined (there are infinitely many $z$ that give the same $x$), and the sparsity constraint selects among these the one that uses the fewest atoms.

The optimization problem:

$$\min_{D, z_1, \ldots, z_N} \sum_{n=1}^{N} \left[ \frac{1}{2}\|x_n - D z_n\|_2^2 + \lambda \|z_n\|_1 \right]$$

subject to $\|d_j\|_2 \leq 1$ for all $j$ (the dictionary atoms are normalized to prevent a trivial solution where $D$ grows large and $z$ shrinks).

### 4.2 The Two Sub-Problems

This is a joint optimization over both $D$ and all $z_n$. It is not jointly convex, but it is convex in each variable separately:

**Sparse coding step (fix $D$, optimize $z_n$ for each input):**

$$\min_{z_n} \frac{1}{2}\|x_n - Dz_n\|_2^2 + \lambda \|z_n\|_1$$

This is a LASSO problem! It is convex and can be solved efficiently.

**Dictionary update step (fix all $z_n$, optimize $D$):**

$$\min_{D} \sum_{n=1}^{N} \frac{1}{2}\|x_n - Dz_n\|_2^2 \quad \text{s.t.} \quad \|d_j\|_2 \leq 1$$

This is a constrained least squares problem, also convex.

We alternate between these two steps. This is an instance of **block coordinate descent**, and while it does not guarantee convergence to the global optimum, it reliably finds good local optima in practice.

### 4.3 A Concrete Example

Suppose $d_x = 2$ (signals in the plane) and $d_z = 4$ (overcomplete dictionary with 4 atoms). The dictionary is:

$$D = \begin{pmatrix} 1 & 0 & 0.707 & -0.707 \\ 0 & 1 & 0.707 & 0.707 \end{pmatrix}$$

These are the horizontal, vertical, and two diagonal directions.

Given signal $x = (3, 4)^\top$, the dense representation might be $z = (3, 4, 0, 0)^\top$ or $z = (0, 0, 4.95, 0.71)^\top$ or many others.

With sparsity, we seek the representation using the fewest atoms. The best 1-sparse representation uses the atom closest to $x$ in direction: $d_3 = (0.707, 0.707)^\top$, giving $z_3 = x^\top d_3 / \|d_3\|^2 \approx 4.95$, with approximation $Dz = (3.5, 3.5)^\top$.

The best 2-sparse representation uses the standard basis: $z = (3, 4, 0, 0)^\top$, giving exact reconstruction $Dz = (3, 4)^\top$.

The sparsity-reconstruction trade-off, controlled by $\lambda$, determines how many atoms are used.

---

## 5. ISTA: Solving the Sparse Coding Problem

### 5.1 Proximal Gradient Descent

The sparse coding sub-problem for a single input:

$$\min_z F(z) = \underbrace{\frac{1}{2}\|x - Dz\|_2^2}_{f(z)} + \underbrace{\lambda \|z\|_1}_{h(z)}$$

This is a composite optimization problem: $f(z)$ is smooth and differentiable, $h(z)$ is convex but non-differentiable.

**Proximal gradient descent** handles this by alternating between a gradient step on $f$ and a proximal step for $h$:

$$z^{(t+1)} = \text{prox}_{\alpha h}\left(z^{(t)} - \alpha \nabla f(z^{(t)})\right)$$

where $\alpha$ is the step size.

### 5.2 Deriving ISTA

The gradient of $f(z) = \frac{1}{2}\|x - Dz\|^2$ is:

$$\nabla f(z) = D^\top(Dz - x) = D^\top Dz - D^\top x$$

The proximal operator for $h(z) = \lambda \|z\|_1$ is soft-thresholding $S_{\alpha\lambda}$.

Putting it together, the **Iterative Shrinkage-Thresholding Algorithm (ISTA)** is:

$$\boxed{z^{(t+1)} = S_{\alpha\lambda}\left(z^{(t)} - \alpha D^\top(Dz^{(t)} - x)\right) = S_{\alpha\lambda}\left(z^{(t)} + \alpha D^\top(x - Dz^{(t)})\right)}$$

Or equivalently:

$$z^{(t+1)} = S_{\alpha\lambda}\left((I - \alpha D^\top D)z^{(t)} + \alpha D^\top x\right)$$

Let us unpack what each part does:
1. **$\alpha D^\top(x - Dz^{(t)})$:** The gradient step. $x - Dz^{(t)}$ is the reconstruction error. $D^\top$ maps this error back to the code space. We update $z$ in the direction that reduces the reconstruction error.
2. **$S_{\alpha\lambda}(\cdot)$:** The soft-thresholding step. After the gradient update, we shrink small values toward zero and set sufficiently small values to exactly zero. This enforces sparsity.

### 5.3 ISTA in Code

```python
def ista(x, D, lambda_, alpha, num_iters=100):
    """
    Solve min_z 0.5 * ||x - Dz||^2 + lambda * ||z||_1
    using ISTA.

    Args:
        x: Signal, shape (d_x,) or (batch, d_x)
        D: Dictionary, shape (d_x, d_z)
        lambda_: Sparsity penalty
        alpha: Step size (should be < 1 / ||D^T D||_op)
        num_iters: Number of iterations

    Returns:
        z: Sparse code, shape (d_z,) or (batch, d_z)
    """
    d_z = D.shape[1]
    z = torch.zeros(d_z)  # Initialize at zero

    for t in range(num_iters):
        # Gradient step
        residual = x - D @ z
        z = z + alpha * (D.T @ residual)

        # Soft-thresholding step
        z = torch.sign(z) * torch.clamp(torch.abs(z) - alpha * lambda_, min=0)

    return z
```

### 5.4 Convergence

ISTA converges at rate $O(1/t)$ -- after $t$ iterations, the suboptimality is proportional to $1/t$. The step size $\alpha$ must satisfy $\alpha < 1/L$ where $L = \|D^\top D\|_{\text{op}}$ is the largest eigenvalue of $D^\top D$ (the Lipschitz constant of $\nabla f$).

### 5.5 FISTA: The Accelerated Version

FISTA (Fast ISTA, Beck and Teboulle 2009) achieves an $O(1/t^2)$ convergence rate by adding a momentum term:

$$y^{(t+1)} = z^{(t)} + \frac{t-1}{t+2}(z^{(t)} - z^{(t-1)})$$
$$z^{(t+1)} = S_{\alpha\lambda}\left(y^{(t+1)} + \alpha D^\top(x - Dy^{(t+1)})\right)$$

The only difference from ISTA is the extrapolation step that computes $y^{(t+1)}$ -- a form of momentum (cf. Nesterov's accelerated gradient from Week 4). This simple modification provides a quadratic speedup in convergence.

---

## 6. Dictionary Learning

### 6.1 Learning the Dictionary

So far we assumed the dictionary $D$ is given. In practice, we learn it from data. This is **dictionary learning**: given a collection of signals $\{x_1, \ldots, x_N\}$, find the dictionary $D$ and sparse codes $\{z_1, \ldots, z_N\}$ that best represent them:

$$\min_{D, \{z_n\}} \sum_{n=1}^{N} \left[\frac{1}{2}\|x_n - Dz_n\|_2^2 + \lambda \|z_n\|_1 \right] \quad \text{s.t.} \quad \|d_j\|_2 \leq 1 \; \forall j$$

### 6.2 Alternating Optimization

The standard approach alternates between two steps:

**Step 1: Sparse Coding.** Fix $D$, solve for each $z_n$ using ISTA (or FISTA, or any LASSO solver):

$$z_n^* = \arg\min_{z} \frac{1}{2}\|x_n - Dz\|_2^2 + \lambda \|z\|_1$$

**Step 2: Dictionary Update.** Fix all $z_n$, update $D$. This is a constrained least squares problem:

$$D^* = \arg\min_{D} \sum_n \frac{1}{2}\|x_n - Dz_n\|_2^2 \quad \text{s.t.} \quad \|d_j\|_2 \leq 1$$

Without the constraint, the solution is $D = XZ^\top(ZZ^\top)^{-1}$ where $X = [x_1, \ldots, x_N]$ and $Z = [z_1, \ldots, z_N]$. With the constraint, we project each column of $D$ onto the unit ball: $d_j \leftarrow d_j / \max(1, \|d_j\|_2)$.

### 6.3 K-SVD (Brief)

The K-SVD algorithm (Aharon, Elad, and Bruckstein, 2006) is an alternative dictionary learning method that updates one dictionary atom at a time using a rank-1 SVD approximation. It is more efficient per iteration but conceptually similar to alternating optimization.

The idea: for each atom $d_j$, collect all the signals that use it (i.e., where $z_{nj} \neq 0$), compute the residual without $d_j$'s contribution, and update $d_j$ and the corresponding coefficients simultaneously using the rank-1 SVD of the residual matrix.

### 6.4 Online Dictionary Learning

For large datasets, Mairal et al. (2009) proposed an online dictionary learning algorithm that processes one sample (or mini-batch) at a time:

1. Draw a sample $x_n$
2. Sparse code: $z_n = \text{ISTA}(x_n, D)$
3. Update $D$ using a gradient step: $D \leftarrow D + \eta (x_n - Dz_n) z_n^\top$
4. Normalize columns: $d_j \leftarrow d_j / \max(1, \|d_j\|)$

This scales to datasets too large to fit in memory and has good convergence properties.

### 6.5 What Do Learned Dictionaries Look Like?

The key empirical result: when you learn a dictionary from natural image patches using sparse coding, the dictionary atoms are **Gabor-like filters** -- localized, oriented, band-pass filters at various positions, orientations, and scales.

This is Olshausen and Field's famous 1996 result. They took 12x12 patches from natural images, learned a dictionary with ~200 atoms, and found that the atoms closely resemble the receptive fields of simple cells in the primary visual cortex (V1).

The implication is profound: the structure of V1 can be explained as an optimal sparse code for natural images. The brain is not just doing sparse coding -- it is doing sparse coding of the *right kind*, matched to the statistics of the natural world.

For MNIST digits, learned dictionary atoms tend to be parts of digits: short strokes, curves, line segments, and corners. Each digit is reconstructed by combining a small number of these parts.

---

## 7. From Sparse Coding to Sparse Autoencoders

### 7.1 The Inference Problem

There is a fundamental practical limitation of sparse coding: at test time, given a new signal $x$, you must solve an optimization problem to find its sparse code:

$$z^* = \arg\min_z \frac{1}{2}\|x - Dz\|_2^2 + \lambda \|z\|_1$$

Even with FISTA, this requires many iterations (typically 50-500). This is slow. Every new input requires a fresh optimization.

### 7.2 Amortized Inference

**Key idea:** Instead of solving the optimization problem from scratch for each input, train a neural network to predict the solution directly.

$$z \approx f_\phi(x)$$

where $f_\phi$ is an encoder network. This is called **amortized inference** -- the cost of learning the encoder is amortized over all future inferences. One forward pass through the encoder replaces hundreds of ISTA iterations.

### 7.3 The Sparse Autoencoder

A sparse autoencoder is an autoencoder where:
- The **encoder** $f_\phi(x)$ predicts the sparse code $z$ (amortizing the ISTA optimization)
- The **decoder** $g_\theta(z) = Dz$ (linear, with $D$ playing the role of the dictionary)
- A **sparsity penalty** on $z$ replaces the LASSO constraint

The objective:

$$\min_{\phi, \theta} \mathbb{E}_x \left[ \|x - g_\theta(f_\phi(x))\|^2 + \lambda \|f_\phi(x)\|_1 \right]$$

Compare this to the sparse coding objective:

$$\min_{D, \{z_n\}} \sum_n \left[ \|x_n - Dz_n\|^2 + \lambda \|z_n\|_1 \right]$$

The structure is identical! The only difference is that instead of optimizing each $z_n$ independently, we learn a function $f_\phi$ that produces $z_n$ from $x_n$ in a single forward pass.

### 7.4 ISTA as an Architecture: LISTA

There is an elegant way to see the connection. Unroll $T$ iterations of ISTA into a computational graph:

$$z^{(t+1)} = S_{\alpha\lambda}\left(Wz^{(t)} + Sx\right)$$

where $W = I - \alpha D^\top D$ and $S = \alpha D^\top$.

This looks like a recurrent neural network with:
- Weight matrix $W$ applied to the hidden state $z^{(t)}$
- Input projection $S$ applied to $x$
- Soft-thresholding as the activation function

The **Learned ISTA (LISTA)** algorithm (Gregor and LeCun, 2010) makes $W$ and $S$ learnable parameters, untied across iterations:

$$z^{(t+1)} = S_{\theta_t}\left(W_t z^{(t)} + S_t x\right)$$

This is a deep network with $T$ layers, where each layer is structurally similar to one ISTA iteration but with learned parameters. LISTA converges much faster than ISTA (often 5-10 layers instead of 100+ iterations), because the learned matrices can "shortcut" the optimization.

A sparse autoencoder can be seen as a one-step version of this: a single feedforward layer with a sparsity-inducing activation function. It is the maximally amortized version of ISTA.

### 7.5 The Connection, Summarized

| Aspect | Sparse Coding | Sparse Autoencoder |
|--------|--------------|-------------------|
| Inference | Iterative optimization (ISTA) | Single forward pass |
| Encoder | ISTA algorithm | Learned neural network $f_\phi$ |
| Dictionary | Learned matrix $D$ | Decoder weights $W_{\text{dec}}$ |
| Objective | $\|x - Dz\|^2 + \lambda\|z\|_1$ | $\|x - g(f(x))\|^2 + \lambda\|f(x)\|_1$ |
| Speed at test time | Slow ($O(T \cdot d_x \cdot d_z)$) | Fast ($O(d_x \cdot d_z)$) |
| Quality | Optimal (solves to convergence) | Approximate (amortization gap) |

The sparse autoencoder trades optimality for speed. The encoder's prediction may not be the exact minimizer of the sparse coding problem, but it is a good approximation obtained in a single forward pass. This trade-off is overwhelmingly worthwhile in practice.

---

## 8. Sparsity Measures Beyond L1

### 8.1 The L0 "Norm"

The most natural measure of sparsity is the L0 "norm": the number of nonzero entries.

$$\|z\|_0 = |\{j : z_j \neq 0\}|$$

Strictly speaking, this is not a norm (it violates homogeneity). Minimizing $\|z\|_0$ directly leads to an NP-hard combinatorial problem. L1 is the tightest convex relaxation of L0 -- this is why we use L1 as a proxy for sparsity.

### 8.2 KL Divergence Penalty

Andrew Ng's influential course notes popularized a different sparsity penalty for autoencoders: the KL divergence between the average activation and a target sparsity level.

Let $\hat{\rho}_j = \frac{1}{N}\sum_{n=1}^{N} f_j(x_n)$ be the average activation of unit $j$ over the dataset, and let $\rho$ be the target (e.g., $\rho = 0.05$, meaning we want each unit active about 5% of the time). The penalty is:

$$\Omega = \sum_j D_{\text{KL}}(\rho \| \hat{\rho}_j) = \sum_j \left[\rho \log \frac{\rho}{\hat{\rho}_j} + (1-\rho) \log \frac{1-\rho}{1-\hat{\rho}_j}\right]$$

This is minimized when $\hat{\rho}_j = \rho$ for all $j$, which means each unit is active for exactly a $\rho$ fraction of inputs.

### 8.3 Top-k Sparsity

Instead of a soft penalty, enforce exact $k$-sparsity: keep only the top $k$ activations and zero out the rest. This is used in $k$-sparse autoencoders (Makhzani and Frey, 2013) and TopK SAEs (Gao et al., 2024), which we will study in Week 13.

---

## 9. The Big Picture

Let us step back and see where we are in the course.

We started with linear algebra and optimization (Weeks 1-2), built neural networks (Weeks 3-4), learned about representation learning and autoencoders (Weeks 5-6), and explored regularization strategies (Week 7) and probabilistic models (Week 8).

This week, we arrived at the core mathematical concept: **sparse representation**. We saw that:

1. Sparsity is a natural principle, motivated by biology, computation, and statistics
2. L1 regularization is the principled way to enforce sparsity, with deep connections to geometry (the diamond argument), optimization (soft-thresholding), and Bayesian inference (Laplace prior)
3. Sparse coding finds a dictionary and sparse codes jointly, using alternating optimization with ISTA
4. Dictionary learning on natural images discovers biologically plausible features
5. **Sparse autoencoders amortize the sparse coding problem**, replacing iterative optimization with a single forward pass through a learned encoder

Next week, we will build on this foundation to implement sparse autoencoders in full detail -- the specific architectures, training tricks, and evaluation metrics. Then in Weeks 11-12, we will apply them to the fascinating problem of understanding what neural networks have learned.

The intellectual thread from Olshausen and Field (1996) to Anthropic's mechanistic interpretability work (2023-2024) is remarkably direct: sparse coding of natural images to sparse coding of neural network activations. The tools are the same. The dictionary is the same. Only the data has changed -- from pixel patches to neural network residual streams.

---

## Summary

1. **Sparsity** is motivated by biology (efficient neural coding), computation (efficient storage and processing), and statistics (regularization, interpretability, disentanglement).

2. **L1 regularization** produces sparse solutions because the L1 ball has corners on coordinate axes. It is equivalent to MAP estimation with a Laplace prior.

3. **Soft-thresholding** $S_\lambda(v) = \text{sign}(v)\max(|v|-\lambda, 0)$ is the proximal operator for L1 and the mechanism by which L1 produces exact zeros.

4. **Sparse coding** (Olshausen & Field 1996) finds a dictionary $D$ and sparse codes $z$ such that $x \approx Dz$ with $z$ sparse. On natural images, the learned dictionary atoms resemble V1 simple cells.

5. **ISTA** solves the sparse coding problem via proximal gradient descent: alternate between a gradient step and soft-thresholding.

6. **Sparse autoencoders amortize sparse coding**: the encoder learns to predict the sparse code in a single forward pass, replacing iterative optimization. The decoder is the dictionary.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| LASSO | $\min_\beta \frac{1}{2}\|y - X\beta\|^2 + \lambda\|\beta\|_1$ |
| Soft-thresholding | $S_\lambda(v) = \text{sign}(v)\max(\|v\| - \lambda, 0)$ |
| Sparse coding | $\min_{D,z} \frac{1}{2}\|x - Dz\|^2 + \lambda\|z\|_1$ |
| ISTA update | $z^{(t+1)} = S_{\alpha\lambda}(z^{(t)} + \alpha D^\top(x - Dz^{(t)}))$ |
| SAE objective | $\min_{\phi,\theta} \mathbb{E}_x[\|x - g_\theta(f_\phi(x))\|^2 + \lambda\|f_\phi(x)\|_1]$ |

---

## Suggested Reading

- **Olshausen and Field** (1996), "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images" -- the foundational paper. Beautifully written.
- **Tibshirani** (1996), "Regression Shrinkage and Selection via the Lasso" -- the original LASSO paper.
- **Beck and Teboulle** (2009), "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems" -- FISTA.
- **Gregor and LeCun** (2010), "Learning Fast Approximations of Sparse Coding" -- LISTA, the bridge between ISTA and neural networks.
- **Mairal et al.** (2009), "Online Dictionary Learning for Sparse Coding" -- scalable dictionary learning.
- **Goodfellow et al.** (2016), *Deep Learning*, Chapter 13 -- sparse coding in the deep learning context.
- **Andrew Ng**, "Sparse Autoencoder" (CS294A lecture notes) -- the original SAE notes, which we will study in detail next week.
