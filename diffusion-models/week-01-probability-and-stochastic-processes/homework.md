# Week 1: Probability, Stochastic Processes, and Markov Chains -- Homework

**Estimated time:** 10-12 hours
**Prerequisites:** Multivariable calculus, linear algebra, basic probability, PyTorch

---

## Problem 1: Gaussian Marginalization and Conditioning (Theory)

### Part (a): Marginalization

Consider a joint Gaussian over $(x\_1, x\_2) \in \mathbb{R}^2$:

$$
\begin{pmatrix} x_1 \\\\ x_2 \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} 1 \\\\ 3 \end{pmatrix}, \begin{pmatrix} 4 & 2 \\\\ 2 & 5 \end{pmatrix}\right)
$$

1. Write down the marginal distributions $p(x\_1)$ and $p(x\_2)$ by reading off the relevant entries. State the rule you are using.

2. Now derive the marginal $p(x\_1)$ from first principles by integrating out $x\_2$:

$$
p(x_1) = \int_{-\infty}^{\infty} p(x_1, x_2) \, dx_2
$$

Show all steps. You should arrive at the same answer as in part 1. (*Hint: complete the square in $x\_2$, then recognize the remaining integral as a Gaussian normalization constant.*)

### Part (b): Conditioning

Using the same joint distribution:

1. Derive the conditional distribution $p(x\_1 \mid x\_2 = 5)$ using the formula:

$$
x_1 \mid x_2 \sim \mathcal{N}\left(\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2), \; \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\right)
$$

2. Compute the conditional mean and variance numerically. Interpret: does observing $x\_2 = 5$ (which is above its mean of 3) shift our expectation of $x\_1$ upward or downward? Why does this make sense given the positive covariance?

3. Show that the conditional variance $\Sigma\_{11} - \Sigma\_{12}\Sigma\_{22}^{-1}\Sigma\_{21}$ is always less than or equal to the marginal variance $\Sigma\_{11}$. What does this mean intuitively?

### Part (c): Affine Transformations

Let $x \sim \mathcal{N}(\mu, \Sigma)$ with $\mu = (1, 2)^\top$ and $\Sigma = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$.

Define $y = Ax + b$ where $A = \begin{pmatrix} 1 & 1 \\ 2 & -1 \end{pmatrix}$ and $b = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$.

1. Compute the distribution of $y$. State the mean and covariance.

2. The forward process applies $x\_t = \sqrt{\alpha\_t} \cdot x\_{t-1} + \sqrt{1-\alpha\_t} \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$ is independent of $x\_{t-1}$. This is an affine transformation -- but of what? Express $x\_t$ in the form $x\_t = Ax\_{t-1} + b + C\epsilon$ and identify $A$, $b$, $C$. Then derive the distribution $p(x\_t \mid x\_{t-1})$ using the affine transformation result.

---

## Problem 2: The Reparameterization Trick (Theory + Implementation)

### Part (a): Gradient Through Sampling

Suppose we want to minimize $L(\mu) = \mathbb{E}\_{x \sim \mathcal{N}(\mu, 1)}[x^2]$ with respect to $\mu$.

1. Compute $L(\mu)$ analytically. (*Hint: $\mathbb{E}[x^2] = \text{Var}(x) + (\mathbb{E}[x])^2$*.)

2. Compute $\nabla\_\mu L(\mu)$ analytically from your closed-form expression.

3. Now use the reparameterization trick: write $x = \mu + \epsilon$ where $\epsilon \sim \mathcal{N}(0,1)$, so $L(\mu) = \mathbb{E}\_\epsilon[(\mu + \epsilon)^2]$. Compute $\nabla\_\mu L(\mu) = \mathbb{E}\_\epsilon[2(\mu + \epsilon)]$ and verify it matches your answer from part 2.

### Part (b): Implementation

In PyTorch, implement a function that:
1. Takes a learnable parameter $\mu \in \mathbb{R}^2$ (initialized to $(5, -3)$)
2. Samples $x = \mu + \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$
3. Computes the loss $L = \Vert x\Vert ^2$
4. Backpropagates and updates $\mu$ using gradient descent

Run for 200 steps with learning rate 0.05 and batch size 64 (average the loss over 64 samples per step). Plot $\mu$ over the course of optimization. Where does it converge? Does this match your analytical result from Part (a)?

---

## Problem 3: Detailed Balance (Theory)

### Part (a): A Simple Chain

Consider a Markov chain on three states $\lbrace 1, 2, 3\rbrace$ with transition matrix:

$$
P = \begin{pmatrix} 0.5 & 0.3 & 0.2 \\\\ 0.15 & 0.7 & 0.15 \\\\ 0.2 & 0.3 & 0.5 \end{pmatrix}
$$

1. Verify that $\pi = (0.2, 0.5, 0.3)$ is a stationary distribution by showing $\pi P = \pi$.

2. Check whether detailed balance holds: does $\pi\_i P\_{ij} = \pi\_j P\_{ji}$ for all pairs $(i,j)$? Check all six off-diagonal pairs.

3. If detailed balance fails for some pair, does that mean $\pi$ is not stationary? Explain.

### Part (b): Designing a Reversible Chain

Suppose we want a Markov chain on states $\lbrace 1, 2, 3\rbrace$ with stationary distribution $\pi = (0.5, 0.3, 0.2)$ that satisfies detailed balance.

1. Construct a transition matrix $P$ that satisfies detailed balance with respect to $\pi$. (*Hint: start by choosing $P\_{12}$, then use $\pi\_1 P\_{12} = \pi\_2 P\_{21}$ to find $P\_{21}$. Repeat for all off-diagonal pairs. Then set diagonal entries to make rows sum to 1.*)

2. Verify your construction: check that $\pi P = \pi$ and that detailed balance holds for all pairs.

### Part (c): Connection to Diffusion

The Gaussian transition kernel $q(x\_t \mid x\_{t-1}) = \mathcal{N}(x\_t; \sqrt{1-\beta}\, x\_{t-1},\; \beta I)$ with constant $\beta$ has $\mathcal{N}(0, I)$ as its stationary distribution (shown in the notes).

Does this kernel satisfy detailed balance with respect to $\mathcal{N}(0, I)$? That is, does:

$$
\mathcal{N}(x; 0, I) \cdot \mathcal{N}(x'; \sqrt{1-\beta}\, x,\; \beta I) = \mathcal{N}(x'; 0, I) \cdot \mathcal{N}(x; \sqrt{1-\beta}\, x',\; \beta I)
$$

hold for all $x, x'$?

Check this by expanding both sides. (*Hint: write out the exponents, expanding all the quadratic forms. If the exponents are equal, the normalizing constants must also be equal.*)

---

## Problem 4: The Forward Noising Process (Implementation)

This is the key implementation problem. You will build the forward process of a diffusion model from scratch.

### Part (a): Noise Schedules

Implement two noise schedule functions that return the $\beta\_t$ values for $t = 1, \ldots, T$:

```python
def linear_schedule(T, beta_min=1e-4, beta_max=0.02):
    """Return beta_1, ..., beta_T for the linear schedule."""
    pass

def cosine_schedule(T, s=0.008):
    """Return beta_1, ..., beta_T for the cosine schedule."""
    pass
```

For $T = 1000$, plot the following four quantities for both schedules on the same figure (2x2 subplot grid):
1. $\beta\_t$ vs. $t$
2. $\bar{\alpha}\_t$ vs. $t$
3. $\text{SNR}(t) = \bar{\alpha}\_t / (1 - \bar{\alpha}\_t)$ vs. $t$ (use log scale for the $y$-axis)
4. $\log \text{SNR}(t)$ vs. $t$

Which schedule produces a more uniform decrease in log-SNR?

### Part (b): Forward Process

Implement the forward noising function:

```python
def forward_process(x_0, t, alpha_bar):
    """
    Apply the forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.

    Args:
        x_0: Clean data, shape (batch, channels, height, width) or (batch, dim)
        t: Timesteps, shape (batch,) with integer values in [0, T-1]
        alpha_bar: Cumulative product schedule, shape (T,)

    Returns:
        x_t: Noised data, same shape as x_0
        epsilon: The noise that was added, same shape as x_0
    """
    pass
```

### Part (c): Visualize the Forward Process

Load or generate a simple test image (e.g., a digit from MNIST, or a 64x64 image from CIFAR-10 using `torchvision.datasets`, or even a simple synthetic image like a white circle on a black background).

Apply the forward process and display the noised image at timesteps $t \in \lbrace 0, 50, 100, 200, 500, 750, 999\rbrace$ for both the linear and cosine schedules.

Arrange the results as a 2x7 grid (one row per schedule). At what timestep does each schedule make the image unrecognizable?

### Part (d): Verify the Markov Property

The closed-form $q(x\_t \mid x\_0)$ should give the same distribution as running the chain step by step. Verify this empirically:

1. Fix $x\_0$ as a single image. Set $t = 200$.
2. **Method 1 (step-by-step):** Run the forward process one step at a time from $x\_0$ to $x\_{200}$, applying $x\_s = \sqrt{1-\beta\_s} x\_{s-1} + \sqrt{\beta\_s} \epsilon\_s$ at each step. Repeat 1000 times to get 1000 samples of $x\_{200}$.
3. **Method 2 (closed-form):** Sample $x\_{200} = \sqrt{\bar{\alpha}\_{200}} x\_0 + \sqrt{1-\bar{\alpha}\_{200}} \epsilon$ directly. Repeat 1000 times.
4. For both methods, compute the sample mean and sample variance of $x\_{200}$ (average over all pixels). Report both. They should match to within sampling error.

---

## Problem 5: Signal Destruction Analysis (Theory + Implementation)

### Part (a): Variance Preservation

Prove that if $x\_0$ has zero mean and unit variance in each coordinate (i.e., $\mathbb{E}[x\_0] = 0$ and $\text{Cov}(x\_0) = I$), then $x\_t$ also has zero mean and unit variance for all $t$.

(*Hint: use $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1-\bar{\alpha}\_t} \epsilon$ and the fact that $x\_0$ and $\epsilon$ are independent.*)

What happens if $x\_0$ does not have unit variance? Compute $\mathbb{E}[x\_t]$ and $\text{Cov}(x\_t)$ when $\mathbb{E}[x\_0] = \mu$ and $\text{Cov}(x\_0) = \Sigma$.

### Part (b): Mutual Information Decay

The **mutual information** $I(x\_0; x\_t)$ measures how much information $x\_t$ retains about $x\_0$. For Gaussians, there is a simple formula.

If $x\_0 \sim \mathcal{N}(0, I\_d)$ (isotropic Gaussian in $d$ dimensions), and $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1-\bar{\alpha}\_t} \epsilon$, then:

$$
I(x_0; x_t) = -\frac{d}{2}\log(1 - \bar{\alpha}_t)
$$

Derive this formula. (*Hint: use the fact that $I(x\_0; x\_t) = H(x\_t) - H(x\_t \mid x\_0)$, where $H$ is differential entropy. The entropy of a $d$-dimensional Gaussian $\mathcal{N}(\mu, \Sigma)$ is $\frac{d}{2}\log(2\pi e) + \frac{1}{2}\log|\Sigma|$.*)

Plot $I(x\_0; x\_t) / d$ vs. $t$ for both the linear and cosine schedules with $d = 784$ (MNIST) and $T = 1000$.

### Part (c): When Is the Image "Gone"?

Using the mutual information formula, determine the timestep $t^*$ at which $I(x\_0; x\_t) / d < 0.01$ nats (i.e., less than 0.01 nats per dimension) for both schedules. This gives a quantitative answer to "when is the image destroyed?"

---

## Problem 6: Designing a Custom Noise Schedule (Implementation)

### Part (a): The Problem with Linear

Implement the linear and cosine schedules from Problem 4. For 1000 uniformly sampled timesteps, compute $\log \text{SNR}(t)$ and plot its histogram. Is the distribution of log-SNR values roughly uniform? If not, where does it concentrate?

### Part (b): A Schedule from log-SNR

Design a noise schedule by specifying the desired log-SNR function directly.

The idea: if we want $\log\text{SNR}(t)$ to decrease linearly from $\gamma\_{\max}$ to $\gamma\_{\min}$:

$$
\log\text{SNR}(t) = \gamma_{\max} - \frac{t}{T}(\gamma_{\max} - \gamma_{\min})
$$

then we can recover $\bar{\alpha}\_t$ from $\log\text{SNR}(t) = \log\bar{\alpha}\_t - \log(1-\bar{\alpha}\_t)$, which gives:

$$
\bar{\alpha}_t = \text{sigmoid}(\log\text{SNR}(t))
$$

and then recover $\beta\_t$ from $\bar{\alpha}\_t$.

Implement this "linear-in-log-SNR" schedule. Use $\gamma\_{\max} = 10$ and $\gamma\_{\min} = -10$. Plot the same four diagnostic plots from Problem 4(a) and compare to the linear and cosine schedules.

### Part (c): Visual Comparison

Using your test image from Problem 4(c), display the noised image at equally spaced timesteps for all three schedules. Which schedule produces the most visually uniform progression from clean to noisy?

---

## Submission Checklist

- [ ] Problem 1: Gaussian marginalization derivation from first principles, conditioning, affine transformation applied to the forward step
- [ ] Problem 2: Reparameterization trick derivation and PyTorch implementation, convergence plot
- [ ] Problem 3: Detailed balance verification for discrete chain, construction of reversible chain, check for Gaussian kernel
- [ ] Problem 4: Noise schedule implementation, forward process implementation, visualization grid, Markov property verification
- [ ] Problem 5: Variance preservation proof, mutual information derivation and plot, "when is the image gone" analysis
- [ ] Problem 6: Custom noise schedule design from log-SNR, visual comparison of three schedules

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs.
