# Week 7: Regularized Autoencoders

> *"The art of being wise is the art of knowing what to overlook."*
> -- William James

---

## Overview

Last week we built autoencoders that compress data through a bottleneck and reconstruct it. The bottleneck -- having fewer latent dimensions than input dimensions -- forces the network to learn a compressed representation. But what happens when we remove the bottleneck? What if the latent space is *bigger* than the input space?

This week we confront that question and discover that the answer leads us to some of the most elegant ideas in representation learning. We will study three families of regularized autoencoders -- denoising, contractive, and sparse -- and find that they share a deep mathematical connection: they all force the network to learn something about the *structure* of the data distribution, rather than merely memorizing inputs.

### Prerequisites
- Week 6: Autoencoders (encoder-decoder architecture, reconstruction loss, undercomplete AE)
- Week 4: Regularization, dropout
- Week 1: Norms, Frobenius norm, Jacobian matrices

---

## 1. The Problem with Overcomplete Autoencoders

### 1.1 Recall: Undercomplete Autoencoders

In Week 6, we studied autoencoders with a bottleneck: the latent dimension $d_z$ is smaller than the input dimension $d_x$. The encoder $f: \mathbb{R}^{d_x} \to \mathbb{R}^{d_z}$ compresses, and the decoder $g: \mathbb{R}^{d_z} \to \mathbb{R}^{d_x}$ reconstructs:

$$\min_{\theta_f, \theta_g} \mathbb{E}_{x \sim p_{\text{data}}} \left[ \| x - g(f(x)) \|^2 \right]$$

Because $d_z < d_x$, the network *must* discard some information. If the network is smart about what it discards, it keeps the important structure and throws away noise. This is why undercomplete autoencoders learn useful representations.

### 1.2 What Goes Wrong When $d_z \geq d_x$?

Now suppose $d_z \geq d_x$. The autoencoder has enough capacity to represent every input exactly. In the simplest case -- a linear autoencoder with $d_z = d_x$ -- the encoder can learn the identity mapping $f(x) = x$ and the decoder can learn $g(z) = z$, achieving zero reconstruction error without learning anything about the data structure.

**Theorem (Identity Mapping in Overcomplete Linear AEs).** Let $f(x) = Wx + b$ with $W \in \mathbb{R}^{d_z \times d_x}$ and $g(z) = W'z + b'$ with $W' \in \mathbb{R}^{d_x \times d_z}$, where $d_z \geq d_x$. Then there exist $W, b, W', b'$ such that $g(f(x)) = x$ for all $x$.

*Proof.* Set $W = \begin{pmatrix} I_{d_x} \\ 0 \end{pmatrix}$, $b = 0$, $W' = \begin{pmatrix} I_{d_x} & 0 \end{pmatrix}$, $b' = 0$. Then $f(x) = \begin{pmatrix} x \\ 0 \end{pmatrix}$ and $g(f(x)) = x$. $\square$

This is not just a theoretical concern. Even nonlinear autoencoders with $d_z < d_x$ can effectively learn identity-like mappings if they have enough hidden-layer capacity. A sufficiently wide encoder can memorize every training example.

### 1.3 Why We Want Overcomplete Representations

Here is the tension: overcomplete representations ($d_z > d_x$) are actually *desirable* in many settings.

**Reason 1: Expressiveness.** An overcomplete representation can express richer structure. Consider representing colours. RGB uses 3 dimensions. But a representation with separate features for "warm," "cool," "saturated," "pastel," "earthy" might use 5+ dimensions and be far more useful for downstream tasks, even though colours live in a 3D space.

**Reason 2: Sparsity.** If we want each input to activate only a few features from a large vocabulary, we need $d_z \gg d_x$. This is exactly the regime of sparse coding and sparse autoencoders, which we will study in Weeks 9-10.

**Reason 3: Interpretability.** A larger dictionary of features is more likely to contain individually interpretable features. This is the core motivation behind the mechanistic interpretability work we will study in Weeks 11-12.

So we *want* overcomplete representations, but we need to prevent the identity mapping. The answer: **regularization**.

### 1.4 The Regularization Principle

The general form of a regularized autoencoder's objective is:

$$\mathcal{L} = \underbrace{\mathbb{E}_{x \sim p_{\text{data}}} \left[ \| x - g(f(x)) \|^2 \right]}_{\text{reconstruction}} + \underbrace{\lambda \cdot \Omega(f, g, x)}_{\text{regularization}}$$

Different choices of $\Omega$ give different autoencoder variants:

| Variant | Regularization $\Omega$ | What it penalizes |
|---------|------------------------|-------------------|
| Weight decay | $\|W\|_F^2$ | Large weights |
| **Denoising** | (implicit, via noise) | Sensitivity to noise |
| **Contractive** | $\|J_f(x)\|_F^2$ | Sensitivity to input perturbations |
| **Sparse** | $\|f(x)\|_1$ or KL penalty | Dense activations |

The first three are the focus of this week. The sparse autoencoder gets its own dedicated treatment in Weeks 9-10, but we will preview it here.

---

## 2. Denoising Autoencoders (DAE)

### 2.1 The Key Idea

The denoising autoencoder, introduced by Vincent et al. (2008), is beautifully simple: **corrupt the input, then train the network to recover the clean version.**

The training procedure:
1. Take a clean input $x$ from the training data
2. Create a corrupted version $\tilde{x}$ by adding noise: $\tilde{x} \sim q(\tilde{x} | x)$
3. Feed $\tilde{x}$ through the autoencoder
4. Compute the loss against the *clean* $x$ (not $\tilde{x}$!)

The objective:

$$\mathcal{L}_{\text{DAE}} = \mathbb{E}_{x \sim p_{\text{data}}} \mathbb{E}_{\tilde{x} \sim q(\tilde{x}|x)} \left[ \| x - g(f(\tilde{x})) \|^2 \right]$$

Notice: the input is $\tilde{x}$ but the target is $x$. The network sees a corrupted version and must reconstruct the original. This means it cannot simply learn the identity -- even if $d_z > d_x$ -- because the identity maps $\tilde{x}$ to $\tilde{x}$, not to $x$.

### 2.2 Types of Corruption

The corruption distribution $q(\tilde{x}|x)$ can take several forms:

**Gaussian noise (additive):**
$$\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

This is the most common choice. The noise level $\sigma$ is a hyperparameter that controls how much the autoencoder must learn to denoise.

**Masking noise (dropout):**
$$\tilde{x}_i = \begin{cases} 0 & \text{with probability } p \\ x_i & \text{with probability } 1-p \end{cases}$$

Each input dimension is independently set to zero with probability $p$. This forces the network to learn to infer missing values from context -- if pixel $(i,j)$ is masked out, the network must reconstruct it from neighbouring pixels.

**Salt-and-pepper noise:**
$$\tilde{x}_i = \begin{cases} 0 & \text{with probability } p/2 \\ 1 & \text{with probability } p/2 \\ x_i & \text{with probability } 1-p \end{cases}$$

This replaces random values with extreme values (for inputs normalized to $[0,1]$).

### 2.3 A Concrete Example

Suppose we have MNIST digits (28x28 grayscale images, so $d_x = 784$). We build an overcomplete DAE with $d_z = 1000$.

With masking noise at $p = 0.3$, about 30% of pixels are zeroed out. The network sees an image with holes and must fill them in. To do this well, it needs to learn:
- That certain pixel patterns form strokes
- That strokes combine to form digit shapes
- That digits have characteristic structures (a "3" has two bumps on the right, etc.)

None of this can be accomplished by the identity mapping. The DAE is forced to learn the *structure of the data manifold*.

### 2.4 Why DAEs Learn About the Data Manifold

Here is the geometric intuition, which is quite beautiful.

The data lies on (or near) a low-dimensional manifold $\mathcal{M}$ embedded in $\mathbb{R}^{d_x}$. When we corrupt $x$ to get $\tilde{x}$, we push the point off the manifold into the ambient space. The DAE learns to map $\tilde{x}$ back onto the manifold -- specifically, back to $x$.

The reconstruction function $r(\tilde{x}) = g(f(\tilde{x}))$ therefore learns a **projection onto the data manifold**. At each point, the learned mapping points from the corrupted input back toward the nearest point on the manifold.

This is closely related to **score matching**. The score function of a distribution $p(x)$ is:

$$s(x) = \nabla_x \log p(x)$$

This is a vector field that, at each point in space, points in the direction of increasing probability -- i.e., toward the data manifold. Alain and Bengio (2014) showed that the reconstruction error of a DAE, $r(\tilde{x}) - \tilde{x}$, estimates a scaled version of the score function:

$$r(\tilde{x}) - \tilde{x} \approx \sigma^2 \nabla_{\tilde{x}} \log p(\tilde{x})$$

In other words, the DAE learns the gradient of the log-density of the (smoothed) data distribution. This is a deep connection between autoencoders and probabilistic models.

### 2.5 DAE Training in Practice

```python
# Pseudocode for DAE training loop
for x_batch in dataloader:
    # Corrupt
    if noise_type == 'gaussian':
        noise = torch.randn_like(x_batch) * sigma
        x_corrupted = x_batch + noise
    elif noise_type == 'masking':
        mask = torch.bernoulli(torch.ones_like(x_batch) * (1 - p))
        x_corrupted = x_batch * mask

    # Forward pass
    z = encoder(x_corrupted)
    x_reconstructed = decoder(z)

    # Loss against CLEAN input
    loss = F.mse_loss(x_reconstructed, x_batch)

    # Backprop
    loss.backward()
    optimizer.step()
```

Key implementation details:
- **Noise schedule:** Some practitioners increase $\sigma$ or $p$ over training, starting easy and making the task harder.
- **Stacked DAEs:** Train one layer at a time as a DAE, then stack the encoders. This was a popular pre-training strategy before batch normalization and better optimizers made it less necessary.
- **The noise level matters:** Too little noise, and the DAE approximates the identity. Too much noise, and the task is impossibly hard. The sweet spot depends on the dataset.

---

## 3. Contractive Autoencoders (CAE)

### 3.1 The Key Idea

The contractive autoencoder, introduced by Rifai et al. (2011), takes a direct approach to preventing the encoder from being too sensitive to its input: **penalize the Jacobian of the encoder**.

The Jacobian of the encoder $f$ at input $x$ is the matrix of partial derivatives:

$$J_f(x) = \frac{\partial f(x)}{\partial x} \in \mathbb{R}^{d_z \times d_x}$$

where $[J_f(x)]_{ij} = \frac{\partial f_i(x)}{\partial x_j}$. This matrix describes how each component of the latent representation changes in response to small changes in each component of the input.

The CAE objective is:

$$\mathcal{L}_{\text{CAE}} = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \| x - g(f(x)) \|^2 + \lambda \| J_f(x) \|_F^2 \right]$$

where $\|J_f(x)\|_F^2 = \sum_{i,j} \left( \frac{\partial f_i(x)}{\partial x_j} \right)^2$ is the squared Frobenius norm of the Jacobian.

### 3.2 What the Jacobian Penalty Does

The Frobenius norm of the Jacobian measures how much the encoder's output changes when the input is perturbed slightly. Penalizing this quantity encourages the encoder to be **locally insensitive** to input perturbations.

But wait -- if the encoder were completely insensitive to its input, it would map everything to the same point, giving $f(x) = c$ for some constant $c$. Then the reconstruction loss would be terrible. So the two terms in the CAE objective are in tension:

- **Reconstruction loss** pushes the encoder to be faithful -- it must preserve enough information to reconstruct $x$.
- **Jacobian penalty** pushes the encoder to be insensitive -- it should not react to small input changes.

The resolution of this tension is beautiful: the encoder learns to be sensitive to changes **along the data manifold** (because those change which data point we are looking at, so reconstruction requires sensitivity) and insensitive to changes **perpendicular to the manifold** (because those are noise, and ignoring them does not hurt reconstruction).

### 3.3 Geometric Picture

Imagine the data lies on a 1D curve (manifold) embedded in 2D space. At any point on the curve, we can decompose a perturbation $\delta x$ into:

- A **tangent component** $\delta x_\parallel$ along the curve
- A **normal component** $\delta x_\perp$ perpendicular to the curve

The CAE learns an encoder where:
- $\| J_f(x) \cdot \delta x_\parallel \|$ is relatively large (sensitive along manifold)
- $\| J_f(x) \cdot \delta x_\perp \|$ is small (insensitive perpendicular to manifold)

This means the CAE implicitly learns the tangent space of the data manifold at each point -- a remarkably useful geometric property.

### 3.4 Computing the Jacobian Penalty

For a single hidden-layer encoder $f(x) = h(Wx + b)$ with element-wise activation $h$, the Jacobian is:

$$J_f(x) = \text{diag}(h'(Wx + b)) \cdot W$$

where $h'$ is the derivative of the activation function applied element-wise, and $\text{diag}(\cdot)$ creates a diagonal matrix.

The Frobenius norm squared is then:

$$\| J_f(x) \|_F^2 = \sum_{i=1}^{d_z} \sum_{j=1}^{d_x} \left( h'(w_i^\top x + b_i) \cdot w_{ij} \right)^2 = \sum_{i=1}^{d_z} h'(w_i^\top x + b_i)^2 \sum_{j=1}^{d_x} w_{ij}^2$$

$$= \sum_{i=1}^{d_z} h'(a_i)^2 \| w_i \|^2$$

where $a_i = w_i^\top x + b_i$ is the pre-activation value and $w_i$ is the $i$-th row of $W$.

**For sigmoid activation** $h(a) = \sigma(a)$, we have $h'(a) = \sigma(a)(1 - \sigma(a))$, so:

$$\| J_f(x) \|_F^2 = \sum_{i=1}^{d_z} [\sigma(a_i)(1 - \sigma(a_i))]^2 \| w_i \|^2$$

This is cheap to compute -- it is a function of the activations $f(x)$ and the weight matrix $W$, both of which we already have during the forward pass.

**For deeper encoders**, the Jacobian involves products of per-layer Jacobians (by the chain rule), making exact computation more expensive. In practice, people often penalize only the first layer's Jacobian, or use approximations.

### 3.5 A Numerical Example

Suppose we have a tiny encoder with $d_x = 3$, $d_z = 2$, sigmoid activation, and:

$$W = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}, \quad b = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

For input $x = (0, 0, 0)^\top$:
- Pre-activations: $a = (0, 0)^\top$
- Activations: $f(x) = (\sigma(0), \sigma(0))^\top = (0.5, 0.5)^\top$
- $\sigma'(0) = 0.5 \times 0.5 = 0.25$
- $\| w_1 \|^2 = 1^2 + 0^2 + 0^2 = 1$
- $\| w_2 \|^2 = 0^2 + 1^2 + 0^2 = 1$

$$\| J_f(x) \|_F^2 = 0.25^2 \cdot 1 + 0.25^2 \cdot 1 = 0.0625 + 0.0625 = 0.125$$

Now consider an encoder where the weights are 10 times larger: $W' = 10W$. The pre-activations are $a = (0, 0)^\top$ (same for this input), but the norms are $\|w'_i\|^2 = 100$. So:

$$\| J_{f'}(x) \|_F^2 = 0.25^2 \cdot 100 + 0.25^2 \cdot 100 = 12.5$$

The contractive penalty would strongly discourage these large weights. Note, however, that the effect is input-dependent: for inputs where the sigmoid is saturated ($\sigma'(a) \approx 0$), even large weights produce small Jacobian norms. The penalty is truly about *local* sensitivity.

### 3.6 CAE vs. Weight Decay

You might ask: doesn't weight decay ($\lambda \|W\|_F^2$) also discourage large weights? Yes, but the CAE penalty is more nuanced because it involves the activation derivatives $h'(a_i)$. Weight decay penalizes large weights everywhere; the Jacobian penalty only penalizes large weights *where the activation function is sensitive* (i.e., in the non-saturated regime). This is a data-dependent form of regularization.

---

## 4. The Equivalence of DAE and CAE

### 4.1 Statement of the Result

One of the most satisfying results in autoencoder theory is the connection between denoising and contractive autoencoders. Vincent et al. (2010) showed:

**Theorem.** For a denoising autoencoder with small Gaussian noise $\tilde{x} = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, the expected DAE reconstruction error (as a function of the learned reconstruction $r(x) = g(f(x))$) is, to first order in $\sigma^2$:

$$\mathcal{L}_{\text{DAE}} \approx \mathbb{E}_x \left[ \| x - r(x) \|^2 + \sigma^2 \| J_r(x) \|_F^2 \right]$$

where $J_r(x) = \frac{\partial r(x)}{\partial x}$ is the Jacobian of the full reconstruction function.

This means that **training a DAE with small noise implicitly penalizes the Jacobian of the reconstruction**, which is closely related to the CAE's explicit Jacobian penalty on the encoder.

### 4.2 Sketch of the Proof

We expand $r(\tilde{x}) = r(x + \epsilon)$ in a Taylor series around $x$:

$$r(x + \epsilon) = r(x) + J_r(x) \epsilon + O(\|\epsilon\|^2)$$

Now substitute into the DAE loss:

$$\| x - r(x + \epsilon) \|^2 = \| x - r(x) - J_r(x)\epsilon \|^2 + O(\|\epsilon\|^3)$$

Expanding:

$$= \|x - r(x)\|^2 - 2(x - r(x))^\top J_r(x)\epsilon + \|J_r(x)\epsilon\|^2 + O(\|\epsilon\|^3)$$

Taking the expectation over $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$:

- $\mathbb{E}[\epsilon] = 0$, so the middle term vanishes.
- $\mathbb{E}[\|J_r(x)\epsilon\|^2] = \mathbb{E}[\epsilon^\top J_r(x)^\top J_r(x) \epsilon] = \sigma^2 \text{tr}(J_r(x)^\top J_r(x)) = \sigma^2 \|J_r(x)\|_F^2$

Therefore:

$$\mathbb{E}_\epsilon \left[ \| x - r(\tilde{x}) \|^2 \right] = \|x - r(x)\|^2 + \sigma^2 \|J_r(x)\|_F^2 + O(\sigma^4)$$

Taking the expectation over $x$ gives us the result. $\square$

### 4.3 Interpretation

This result tells us that denoising and contractive autoencoders are not two unrelated ideas -- they are two sides of the same coin:

- **CAE** explicitly penalizes sensitivity via the Jacobian norm
- **DAE** implicitly penalizes sensitivity by requiring robustness to noise

The denoising approach achieves contractiveness *without explicitly computing the Jacobian*, which is a practical advantage for deep encoders where Jacobian computation is expensive.

There is a subtle difference: the DAE penalizes the Jacobian of the *reconstruction* $r = g \circ f$, while the standard CAE penalizes the Jacobian of the *encoder* $f$ alone. But since the decoder is also learned, both achieve the same qualitative effect: insensitivity to directions perpendicular to the data manifold.

### 4.4 Practical Implications

This equivalence suggests:
- Use **DAE** when the Jacobian is expensive to compute (deep networks, large input dimensions)
- Use **CAE** when you want fine-grained control over the regularization strength per layer
- Both achieve similar qualitative effects on the learned representation
- For theoretical analysis, the CAE formulation is often cleaner; for practical training, the DAE is often easier

---

## 5. Sparse Autoencoders: A Preview

### 5.1 A Different Kind of Regularization

Denoising and contractive autoencoders regularize by controlling *how the encoder responds to perturbations*. Sparse autoencoders take a different approach: they regularize the *activations* rather than the *function*.

The idea: instead of penalizing how much the representation *changes*, penalize how *dense* it is. Encourage most latent units to be zero (or near zero) for any given input.

$$\mathcal{L}_{\text{SAE}} = \mathbb{E}_x \left[ \| x - g(f(x)) \|^2 + \lambda \| f(x) \|_1 \right]$$

Here $\|f(x)\|_1 = \sum_{i} |f_i(x)|$ is the L1 norm of the activations, not the weights.

### 5.2 Why Sparsity Is Special

Sparsity is uniquely suited for overcomplete autoencoders. Consider $d_z = 10000$ and $d_x = 784$ (MNIST). The autoencoder has a dictionary of 10000 features. For any given input, only, say, 50 features activate. Different inputs activate different subsets.

This means:
- Each feature can be interpretable (it represents one specific pattern)
- The representation is efficient (despite being high-dimensional, most entries are zero)
- Different features can be combined combinatorially (exponentially many combinations of 50 from 10000)

We will develop this idea fully in Week 9 (sparse coding and dictionary learning) and Week 10 (sparse autoencoders). For now, note how it fits into the regularization family: all regularized autoencoders prevent trivial solutions, but sparse autoencoders do so by constraining the *activation pattern* rather than the *sensitivity*.

---

## 6. Comparing Regularization Strategies

### 6.1 Summary Table

| Property | Vanilla AE | DAE | CAE | Sparse AE |
|----------|-----------|-----|-----|-----------|
| **Regularization** | None (bottleneck only) | Implicit (noise) | Explicit (Jacobian) | Activation sparsity |
| **Overcomplete?** | No (fails) | Yes | Yes | Yes (designed for it) |
| **Hyperparameters** | Architecture | Noise type, $\sigma$ or $p$ | $\lambda$ for Jacobian | $\lambda$ for sparsity |
| **Computational cost** | Low | Low (just add noise) | Medium (Jacobian) | Low-Medium |
| **Learns manifold?** | Implicitly (bottleneck) | Yes (denoising = projection) | Yes (tangent space) | Differently (sparse features) |
| **Generative?** | Poor | Poor | Poor | Poor |
| **Good for features?** | Reasonable | Good | Good | Excellent |

### 6.2 When to Use Which

**Undercomplete AE (vanilla with bottleneck):** When the intrinsic dimensionality of your data is known and low. When you want the simplest approach. Think of it as nonlinear PCA.

**Denoising AE:** When your data is naturally noisy and you want robustness. When you want the benefits of contraction without computing Jacobians. A good general-purpose choice for learning representations.

**Contractive AE:** When you want explicit control over the smoothness of the learned mapping. When you are doing theoretical analysis and want a clean mathematical framework.

**Sparse AE:** When you want interpretable, disentangled features. When you need overcomplete representations. When you are building toward mechanistic interpretability. (This is where we are headed.)

### 6.3 The Common Thread

All regularized autoencoders share a philosophical commitment: **the autoencoder should learn something about the data distribution, not just memorize inputs.** They achieve this by different mechanisms:

- The **bottleneck** forces information compression
- **Denoising** forces learning the manifold's structure (to denoise, you must know what "clean" looks like)
- **Contraction** forces insensitivity to noise directions
- **Sparsity** forces a compact, selective representation

Each of these is a form of inductive bias -- an assumption we bake into the model about what makes a representation "good." The choice of bias should match the downstream task. For interpretability, sparsity turns out to be the winning bias, as we will see.

---

## 7. Deeper Connections (Optional)

### 7.1 Autoencoders and Score Matching

We mentioned briefly that DAEs learn the score function $\nabla_x \log p(x)$. Let us make this more precise.

The **score function** of a distribution $p(x)$ is the gradient of the log-density:

$$s(x) = \nabla_x \log p(x)$$

It is a vector field that points toward high-probability regions. Score matching (Hyvarinen, 2005) provides a way to estimate $s(x)$ without knowing the normalizing constant of $p(x)$.

For a DAE with small Gaussian noise ($\sigma \to 0$), the optimal reconstruction function satisfies:

$$r(x) - x \approx \sigma^2 s(x)$$

This means the "correction" the DAE applies (the difference between its output and input) is proportional to the score function. The DAE literally learns which direction to "push" a corrupted input to make it more likely under the data distribution.

This connection became the foundation for **diffusion models** (Song and Ermon, 2019; Ho et al., 2020), which iteratively denoise random noise into data samples using a learned score function. So denoising autoencoders are, in a sense, the intellectual ancestor of DALL-E and Stable Diffusion.

### 7.2 Autoencoders as Approximate MAP Inference

Each regularized autoencoder can be interpreted as performing approximate MAP (maximum a posteriori) inference under a different prior on the latent code $z$:

- **Weight decay** $\to$ Gaussian prior on weights (L2 regularization = MAP with $p(W) = \mathcal{N}(0, I/\lambda)$)
- **Sparse AE** $\to$ Laplace prior on activations (L1 on $z$ = MAP with $p(z) \propto e^{-\lambda \|z\|_1}$)
- **DAE** $\to$ Smoothness prior on the reconstruction (the data manifold should be smooth)
- **CAE** $\to$ Invariance prior (the representation should be locally constant off the manifold)

This Bayesian view unifies the regularization family under a single framework: different regularizers encode different prior beliefs about the structure of good representations.

---

## 8. Historical Context

The development of regularized autoencoders follows a clear intellectual arc:

1. **2006: Hinton & Salakhutdinov** popularize deep autoencoders for dimensionality reduction (undercomplete)
2. **2008: Vincent et al.** introduce denoising autoencoders -- the first clean solution to the overcomplete problem
3. **2011: Rifai et al.** introduce contractive autoencoders -- formalizing the geometric intuition
4. **2010: Vincent** proves the DAE-CAE connection -- unifying the two approaches
5. **2013-2014: Kingma & Welling** introduce VAEs -- next week's topic -- which regularize through a probabilistic framework

Meanwhile, the sparsity story has its own parallel history:
1. **1996: Olshausen & Field** introduce sparse coding (Week 9)
2. **2006-2011: Sparse autoencoders** are developed as an amortized version of sparse coding (Week 10)
3. **2022-2024: Anthropic and OpenAI** apply sparse autoencoders to mechanistic interpretability (Weeks 11-12)

We are tracing this arc, building up to the modern applications.

---

## Summary

This week we confronted the overcomplete autoencoder problem and discovered three elegant solutions:

1. **Denoising autoencoders** corrupt the input and train against the clean target, implicitly learning the data manifold's structure and the score function of the data distribution.

2. **Contractive autoencoders** explicitly penalize the encoder's Jacobian, encouraging the representation to be insensitive to directions perpendicular to the data manifold.

3. These two approaches are **mathematically equivalent** in the limit of small noise -- a satisfying unification.

4. **Sparse autoencoders** take a different approach, penalizing dense activations rather than sensitivity. They are uniquely suited for overcomplete representations and interpretability. We will study them in depth starting in Week 9.

The common thread: regularization forces the autoencoder to learn the *structure* of the data, not just a trivial mapping. The choice of regularizer encodes our prior beliefs about what makes a good representation.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| DAE objective | $\mathcal{L}_{\text{DAE}} = \mathbb{E}_{x, \tilde{x}} \left[ \|x - g(f(\tilde{x}))\|^2 \right]$ |
| CAE objective | $\mathcal{L}_{\text{CAE}} = \mathbb{E}_x \left[ \|x - g(f(x))\|^2 + \lambda \|J_f(x)\|_F^2 \right]$ |
| Jacobian (single layer) | $J_f(x) = \text{diag}(h'(Wx+b)) \cdot W$ |
| Frobenius norm squared | $\|J_f(x)\|_F^2 = \sum_i h'(a_i)^2 \|w_i\|^2$ |
| DAE-CAE equivalence | $\mathcal{L}_{\text{DAE}} \approx \mathbb{E}_x[\|x-r(x)\|^2 + \sigma^2 \|J_r(x)\|_F^2]$ |
| Score function | $r(x) - x \approx \sigma^2 \nabla_x \log p(x)$ |

---

## Suggested Reading

- **Vincent et al.** (2008), "Extracting and Composing Robust Features with Denoising Autoencoders" -- the original DAE paper.
- **Rifai et al.** (2011), "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction" -- the original CAE paper.
- **Vincent** (2010), "A Connection Between Score Matching and Denoising Autoencoders" -- the DAE-CAE equivalence.
- **Goodfellow et al.** (2016), *Deep Learning*, Chapter 14 -- covers all autoencoder variants in our notation.
- **Alain and Bengio** (2014), "What Regularized Auto-Encoders Learn from the Data-Generating Distribution" -- deep theoretical analysis.
