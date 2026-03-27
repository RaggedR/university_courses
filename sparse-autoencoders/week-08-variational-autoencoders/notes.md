# Week 8: Variational Autoencoders

> *"God does not play dice with the universe."* -- Albert Einstein
> *"Stop telling God what to do."* -- Niels Bohr

---

## Overview

Everything we have built so far -- autoencoders, denoising autoencoders, contractive autoencoders -- compresses and reconstructs data. These models learn representations, and good ones at that. But they cannot *generate* new data.

Try this thought experiment: take a vanilla autoencoder trained on MNIST, pick a random point in the latent space, and decode it. What do you get? Almost certainly: garbage. The latent space of a vanilla autoencoder has no obligation to be *structured* in a way that supports generation. Points between encoded digits might decode to nothing recognizable.

This week we fix this. The **Variational Autoencoder** (VAE), introduced by Kingma and Welling in 2014, reframes the autoencoder as a probabilistic generative model. By placing a probability distribution over the latent space, the VAE ensures that the latent space is smooth, continuous, and structured -- enabling both representation learning *and* generation.

The VAE is arguably the most elegant architecture we will study in this course. It sits at the intersection of deep learning, Bayesian inference, and information theory, and its derivation draws on all the foundations we built in Weeks 1-2.

### Prerequisites
- Week 2: Probability distributions, Bayes' theorem, KL divergence, expectation
- Week 6: Autoencoders (encoder-decoder architecture)
- Week 7: Regularized autoencoders (the need for regularization)

---

## 1. Motivation: Why Autoencoders Cannot Generate

### 1.1 The Problem with Unstructured Latent Spaces

A trained autoencoder gives us an encoder $f: \mathbb{R}^{d\_x} \to \mathbb{R}^{d\_z}$ and a decoder $g: \mathbb{R}^{d\_z} \to \mathbb{R}^{d\_x}$. The decoder can, in principle, map *any* latent vector $z$ to an output. But the decoder was only trained on latent vectors that are *outputs of the encoder* -- i.e., on the set $\lbrace f(x) : x \in \text{training data}\rbrace$.

If you feed the decoder a random $z$ that does not lie in this set, the decoder is being asked to extrapolate, and neural networks are notoriously bad at extrapolation.

Worse, the encoded training points may form a scattered, irregular cloud in latent space. Between any two encoded digits, there might be a vast desert of latent space that the decoder has never seen during training.

### 1.2 What We Want

For generation, we want a latent space where:
1. **Every point** decodes to a plausible output (no "dead zones")
2. **Nearby points** decode to similar outputs (continuity)
3. **We know how to sample** from the latent space (we have a distribution over it)

This is exactly what the VAE provides: it regularizes the latent space to be *close to a known distribution* (typically a standard Gaussian), ensuring all three properties.

### 1.3 A Concrete Illustration

Suppose we train a vanilla autoencoder and a VAE on MNIST, both with $d\_z = 2$ (for visualization).

**Vanilla AE latent space:** The encoded digits form tight, separated clusters with irregular shapes. The space between clusters is empty. Sampling a random point from this space and decoding it yields noise.

**VAE latent space:** The encoded digits form overlapping, roughly Gaussian clouds centered near the origin. The entire region around the origin is "filled in" with meaningful content. Sampling from $\mathcal{N}(0, I)$ and decoding produces recognizable (if sometimes blurry) digits.

The price of this structure: VAE reconstructions are typically blurrier than autoencoder reconstructions. There is a fundamental tension between reconstruction quality and latent space regularity, and the VAE makes a principled trade-off.

---

## 2. Latent Variable Models

### 2.1 The Generative Story

A latent variable model describes data generation as a two-step process:

1. **Sample a latent variable:** $z \sim p(z)$ (the prior)
2. **Generate data from the latent:** $x \sim p\_\theta(x | z)$ (the likelihood, parameterized by $\theta$)

The latent variable $z$ captures the "underlying causes" of the data. For handwritten digits, $z$ might encode the digit identity, slant, stroke width, etc. The generative process first decides these latent factors, then renders the pixel image.

The joint distribution is:

$$
p_\theta(x, z) = p_\theta(x | z) \cdot p(z)
$$

and the marginal likelihood (the probability of observing $x$ regardless of $z$) is:

$$
p_\theta(x) = \int p_\theta(x | z) \, p(z) \, dz
$$

### 2.2 The Inference Problem

Given observed data $x$, we want to infer the latent factors $z$ that generated it. By Bayes' theorem:

$$
p_\theta(z | x) = \frac{p_\theta(x | z) \, p(z)}{p_\theta(x)} = \frac{p_\theta(x | z) \, p(z)}{\int p_\theta(x | z') \, p(z') \, dz'}
$$

This posterior $p\_\theta(z|x)$ tells us which latent codes are likely to have generated a given observation. It is the "ideal encoder" -- the probabilistically correct mapping from data to latent space.

### 2.3 The Intractability Problem

Here is the fundamental difficulty: computing $p\_\theta(x) = \int p\_\theta(x|z) p(z) dz$ requires integrating over the entire latent space. When $p\_\theta(x|z)$ is a neural network (a complicated nonlinear function of $z$), this integral has no closed-form solution, and numerical integration is infeasible in high dimensions.

Since $p\_\theta(x)$ appears in the denominator of Bayes' rule, the posterior $p\_\theta(z|x)$ is also intractable.

We cannot compute the true posterior. We need an approximation. This is where variational inference enters.

---

## 3. Variational Inference and the ELBO

### 3.1 The Variational Approach

Since we cannot compute $p\_\theta(z|x)$, we will approximate it with a parametric distribution $q\_\phi(z|x)$, called the **variational posterior** or **recognition model**. Here $\phi$ are the parameters of a neural network that takes $x$ as input and outputs the parameters of a distribution over $z$.

We want $q\_\phi(z|x)$ to be close to $p\_\theta(z|x)$. The natural measure of closeness between distributions is the KL divergence (from Week 2):

$$
D_{\text{KL}}(q_\phi(z|x) \Vert p_\theta(z|x)) = \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{q_\phi(z|x)}{p_\theta(z|x)} \right]
$$

We would like to minimize this KL divergence. But it involves $p\_\theta(z|x)$, which we cannot compute! We seem stuck. The way out is one of the most important derivations in modern machine learning.

### 3.2 Deriving the ELBO

Start with the log marginal likelihood $\log p\_\theta(x)$. We want to relate it to something we *can* compute.

**Step 1.** Write $\log p\_\theta(x)$ and introduce $q\_\phi(z|x)$:

$$
\log p_\theta(x) = \log \int p_\theta(x, z) \, dz
$$

Since this does not depend on $z$, we can write:

$$
\log p_\theta(x) = \mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x)]
$$

**Step 2.** Use Bayes' rule $p\_\theta(x) = p\_\theta(x,z) / p\_\theta(z|x)$:

$$
\log p_\theta(x) = \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{p_\theta(x, z)}{p_\theta(z|x)} \right]
$$

**Step 3.** Multiply and divide by $q\_\phi(z|x)$:

$$
\log p_\theta(x) = \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right] + \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{q_\phi(z|x)}{p_\theta(z|x)} \right]
$$

**Step 4.** Identify the two terms:

$$
\log p_\theta(x) = \underbrace{\mathbb{E}_{z \sim q_\phi} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]}_{\text{ELBO: } \mathcal{L}(\theta, \phi; x)} + \underbrace{D_{\text{KL}}(q_\phi(z|x) \Vert p_\theta(z|x))}_{\geq 0}
$$

Since KL divergence is always non-negative, we have:

$$
\boxed{\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x) = \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]}
$$

This is the **Evidence Lower BOund (ELBO)**. The name is literal: it is a lower bound on the log-evidence $\log p\_\theta(x)$.

### 3.3 Decomposing the ELBO

We can rewrite the ELBO in a more interpretable form. Since $p\_\theta(x,z) = p\_\theta(x|z) p(z)$:

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{z \sim q_\phi} \left[ \log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x) \right]
$$

$$
= \mathbb{E}_{z \sim q_\phi} [\log p_\theta(x|z)] - \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{q_\phi(z|x)}{p(z)} \right]
$$

$$
\boxed{\mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)]}_{\text{Reconstruction term}} - \underbrace{D_{\text{KL}}(q_\phi(z|x) \Vert p(z))}_{\text{Regularization term}}}
$$

This decomposition is the heart of the VAE. Let us understand each term:

**Reconstruction term: $\mathbb{E}\_{z \sim q\_\phi(z|x)}[\log p\_\theta(x|z)]$.**
This says: sample a latent code $z$ from the approximate posterior, then evaluate how well the decoder reconstructs $x$ from $z$. Maximizing this term pushes the decoder to be a good reconstructor and the encoder to produce informative codes.

If $p\_\theta(x|z) = \mathcal{N}(x; g\_\theta(z), \sigma^2 I)$, then $\log p\_\theta(x|z) = -\frac{1}{2\sigma^2}\Vert x - g\_\theta(z)\Vert ^2 + \text{const}$, and maximizing this is equivalent to minimizing reconstruction MSE. Sound familiar? It is the same reconstruction loss as in a vanilla autoencoder.

**Regularization term: $D\_{\text{KL}}(q\_\phi(z|x) \Vert p(z))$.**
This penalizes the approximate posterior for being different from the prior. It encourages the encoder to produce latent codes that are distributed like $p(z)$. If $p(z) = \mathcal{N}(0, I)$, this term pushes all the encoded points toward a standard Gaussian -- exactly the structure we wanted for generation.

### 3.4 The Tightness of the Bound

Recall that:

$$
\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + D_{\text{KL}}(q_\phi(z|x) \Vert p_\theta(z|x))
$$

The gap between the true log-likelihood and the ELBO is exactly $D\_{\text{KL}}(q\_\phi(z|x) \Vert p\_\theta(z|x))$. This gap is zero if and only if $q\_\phi(z|x) = p\_\theta(z|x)$ -- i.e., the approximate posterior matches the true posterior exactly.

In practice, this gap is non-zero because we use a restricted family for $q\_\phi$ (typically Gaussian). The better our approximation, the tighter the bound, and the more faithfully we maximize the true likelihood.

### 3.5 An Alternative Derivation via Jensen's Inequality

Here is a quicker (but less informative) derivation:

$$
\log p_\theta(x) = \log \int p_\theta(x|z) p(z) dz = \log \int p_\theta(x|z) p(z) \frac{q_\phi(z|x)}{q_\phi(z|x)} dz
$$

$$
= \log \mathbb{E}_{z \sim q_\phi} \left[ \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} \right] \geq \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} \right]
$$

where the inequality is Jensen's inequality ($\log$ is concave). This directly gives the ELBO.

---

## 4. The VAE Architecture

### 4.1 The Encoder (Recognition Model)

The encoder is a neural network $q\_\phi(z|x)$ that maps an input $x$ to the *parameters* of a distribution over $z$. We choose $q\_\phi(z|x)$ to be Gaussian:

$$
q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))
$$

The encoder network takes $x$ and outputs two vectors:
- $\mu\_\phi(x) \in \mathbb{R}^{d\_z}$ -- the mean
- $\log \sigma^2\_\phi(x) \in \mathbb{R}^{d\_z}$ -- the log-variance (we parameterize in log-space for numerical stability)

Note the critical difference from a vanilla autoencoder: the encoder outputs a *distribution*, not a point. Each input $x$ maps to a cloud in latent space, not a single point.

Why output $\log \sigma^2$ instead of $\sigma^2$? Because variance must be positive, and the log transform maps $\mathbb{R} \to \mathbb{R}^+$, allowing the network to output any real number while ensuring valid variances.

### 4.2 The Decoder (Generative Model)

The decoder is a neural network $p\_\theta(x|z)$ that maps a latent code $z$ to the parameters of a distribution over $x$.

For continuous data (images with pixel values in $[0,1]$):
$$
p_\theta(x|z) = \prod_{i} \text{Bernoulli}(x_i; g_\theta(z)_i) \quad \text{(binary cross-entropy loss)}
$$

or

$$
p_\theta(x|z) = \mathcal{N}(x; g_\theta(z), \sigma^2 I) \quad \text{(MSE loss)}
$$

The Bernoulli likelihood is more appropriate for binary-valued pixels (like binarized MNIST), while the Gaussian likelihood is used for continuous-valued data.

### 4.3 The Prior

The prior on $z$ is chosen to be a standard Gaussian:

$$
p(z) = \mathcal{N}(0, I)
$$

This is a design choice, not a theoretical necessity. We choose it because:
1. It has a simple, known form
2. We can easily sample from it
3. It makes the KL divergence tractable (see below)
4. It encourages the latent space to be centered and not too spread out

### 4.4 The Full Architecture

```
Input x
    |
    v
[Encoder Network] --> mu(x), log_var(x)
    |                    |
    |    z = mu + exp(0.5 * log_var) * epsilon,  epsilon ~ N(0, I)
    |                    |
    v                    v
            [Decoder Network] --> x_reconstructed
```

The "sampling" step in the middle is the reparameterization trick, which we discuss next.

---

## 5. The KL Divergence: Closed Form

### 5.1 KL Between Two Multivariate Gaussians

Before deriving the VAE-specific case, let us state the general result. For two multivariate Gaussians:

$$
p = \mathcal{N}(\mu_1, \Sigma_1), \quad q = \mathcal{N}(\mu_2, \Sigma_2)
$$

The KL divergence is:

$$
D_{\text{KL}}(p \Vert q) = \frac{1}{2} \left[ \log \frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1} \Sigma_1) + (\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1) \right]
$$

where $d$ is the dimensionality. This is a standard result you will derive in the homework.

### 5.2 VAE-Specific Case

For the VAE, we need $D\_{\text{KL}}(q\_\phi(z|x) \Vert p(z))$ where:
- $q\_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ (diagonal covariance, depending on $x$)
- $p(z) = \mathcal{N}(0, I)$

Plugging into the general formula with $\mu\_1 = \mu$, $\Sigma\_1 = \text{diag}(\sigma^2)$, $\mu\_2 = 0$, $\Sigma\_2 = I$:

$$
D_{\text{KL}} = \frac{1}{2} \left[ \log \frac{|I|}{|\text{diag}(\sigma^2)|} - d_z + \text{tr}(\text{diag}(\sigma^2)) + \mu^\top \mu \right]
$$

Since $|I| = 1$, $|\text{diag}(\sigma^2)| = \prod\_j \sigma\_j^2$, and $\text{tr}(\text{diag}(\sigma^2)) = \sum\_j \sigma\_j^2$:

$$
\boxed{D_{\text{KL}}(q_\phi(z|x) \Vert p(z)) = \frac{1}{2} \sum_{j=1}^{d_z} \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)}
$$

Or equivalently, using log-variance $\gamma\_j = \log \sigma\_j^2$:

$$
D_{\text{KL}} = \frac{1}{2} \sum_{j=1}^{d_z} \left( \mu_j^2 + e^{\gamma_j} - \gamma_j - 1 \right)
$$

This is the form typically used in code, since the encoder outputs $\gamma\_j = \log \sigma\_j^2$ directly.

### 5.3 Intuition for the KL Terms

Let us understand each term in $\mu\_j^2 + \sigma\_j^2 - \log \sigma\_j^2 - 1$:

- **$\mu\_j^2$**: Penalizes the mean for being far from 0. This keeps encoded points centered.
- **$\sigma\_j^2$**: Penalizes the variance for being large. Prevents overly uncertain encodings.
- **$-\log \sigma\_j^2$**: Penalizes the variance for being *small*. Prevents the encoder from collapsing to a point (which would make $\sigma\_j^2 \to 0$).
- **$-1$**: A constant that makes the KL zero when $\mu\_j = 0$ and $\sigma\_j^2 = 1$.

The minimum of $f(\sigma^2) = \sigma^2 - \log \sigma^2$ is at $\sigma^2 = 1$ (you can verify: $f'(\sigma^2) = 1 - 1/\sigma^2 = 0$ when $\sigma^2 = 1$). So the KL term pushes each latent dimension toward a standard normal: mean 0, variance 1.

---

## 6. The Reparameterization Trick

### 6.1 The Problem: Sampling Is Not Differentiable

To compute the reconstruction term $\mathbb{E}\_{z \sim q\_\phi(z|x)}[\log p\_\theta(x|z)]$, we need to sample $z$ from $q\_\phi(z|x) = \mathcal{N}(\mu\_\phi(x), \text{diag}(\sigma^2\_\phi(x)))$. In practice, we estimate this expectation with a single Monte Carlo sample:

$$
\mathbb{E}_{z \sim q_\phi}[\log p_\theta(x|z)] \approx \log p_\theta(x|z), \quad z \sim q_\phi(z|x)
$$

But there is a problem: the sampling operation $z \sim \mathcal{N}(\mu, \sigma^2)$ is stochastic. The gradient $\nabla\_\phi \mathbb{E}\_{z \sim q\_\phi}[\log p\_\theta(x|z)]$ is not straightforward because $z$ depends on $\phi$ through $\mu\_\phi$ and $\sigma\_\phi$, and sampling introduces a non-differentiable step.

We cannot backpropagate through a random number generator.

### 6.2 The Solution: Reparameterize

The reparameterization trick (Kingma and Welling, 2014) separates the randomness from the parameters:

Instead of $z \sim \mathcal{N}(\mu, \sigma^2 I)$, write:

$$
\boxed{z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)}
$$

where $\odot$ is element-wise multiplication. The randomness is now in $\epsilon$, which does not depend on $\phi$. The dependence on $\phi$ enters only through $\mu$ and $\sigma$, which are deterministic functions of $x$.

Now $z$ is a deterministic, differentiable function of $\mu$, $\sigma$, and $\epsilon$. We can compute:

$$
\frac{\partial z}{\partial \mu} = I, \quad \frac{\partial z}{\partial \sigma} = \text{diag}(\epsilon)
$$

and backpropagate through the sampling step as usual.

### 6.3 Why This Is Clever

The reparameterization trick does not change the distribution of $z$ -- it is still $\mathcal{N}(\mu, \sigma^2 I)$. It merely reparameterizes the randomness so that it is "external" to the computation graph.

Think of it this way: instead of the network producing a random output, the network produces a deterministic transformation of external randomness. The external randomness ($\epsilon$) is like a random input to the network, and we know how to differentiate with respect to network parameters given fixed inputs.

### 6.4 Implementation

```python
def reparameterize(mu, log_var):
    """
    Sample z from N(mu, sigma^2 I) using the reparameterization trick.

    Args:
        mu: Mean of shape (batch_size, d_z)
        log_var: Log-variance of shape (batch_size, d_z)

    Returns:
        z: Sampled latent code of shape (batch_size, d_z)
    """
    std = torch.exp(0.5 * log_var)  # sigma = exp(0.5 * log(sigma^2))
    epsilon = torch.randn_like(std)  # Sample from N(0, I)
    z = mu + std * epsilon
    return z
```

During training, we sample $\epsilon$ fresh for each forward pass. During evaluation, we can either sample (for generation) or use $z = \mu$ (for deterministic encoding).

---

## 7. The Complete VAE Loss

### 7.1 Putting It All Together

The VAE training objective is to maximize the ELBO. Equivalently, we minimize the negative ELBO:

$$
\mathcal{L}_{\text{VAE}} = -\mathcal{L}(\theta, \phi; x) = \underbrace{-\mathbb{E}_{z \sim q_\phi}[\log p_\theta(x|z)]}_{\text{Reconstruction loss}} + \underbrace{D_{\text{KL}}(q_\phi(z|x) \Vert p(z))}_{\text{KL regularization}}
$$

With a Bernoulli decoder (for binary data):

$$
\text{Reconstruction loss} = -\sum_{i=1}^{d_x} [x_i \log \hat{x}_i + (1 - x_i) \log(1 - \hat{x}_i)]
$$

where $\hat{x} = g\_\theta(z)$ with sigmoid output.

With a Gaussian decoder (MSE):

$$
\text{Reconstruction loss} = \frac{1}{2\sigma^2} \Vert x - g_\theta(z)\Vert ^2 + \text{const}
$$

The KL term (closed form):

$$
D_{\text{KL}} = \frac{1}{2} \sum_{j=1}^{d_z} (\mu_j^2 + e^{\gamma_j} - \gamma_j - 1)
$$

where $\gamma\_j = \log \sigma\_j^2$.

### 7.2 A Numerical Example

Let us trace through a single training step. Suppose $d\_x = 4$, $d\_z = 2$.

**Input:** $x = (0.8, 0.2, 0.9, 0.1)$

**Encoder outputs:**
- $\mu = (0.5, -0.3)$
- $\log \sigma^2 = (-0.5, 0.2)$

**Reparameterization:**
- $\sigma = (\exp(-0.25), \exp(0.1)) = (0.779, 1.105)$
- $\epsilon = (0.3, -0.7)$ (sampled)
- $z = (0.5 + 0.779 \times 0.3, \; -0.3 + 1.105 \times (-0.7)) = (0.734, -1.074)$

**Decoder output:** $\hat{x} = g\_\theta(z) = (0.75, 0.25, 0.85, 0.15)$ (suppose)

**Reconstruction loss (MSE):** $\Vert x - \hat{x}\Vert ^2 = (0.05)^2 + (0.05)^2 + (0.05)^2 + (0.05)^2 = 0.01$

**KL divergence:**
- Dimension 1: $0.5^2 + e^{-0.5} - (-0.5) - 1 = 0.25 + 0.607 + 0.5 - 1 = 0.357$
- Dimension 2: $(-0.3)^2 + e^{0.2} - 0.2 - 1 = 0.09 + 1.221 - 0.2 - 1 = 0.111$
- Total: $\frac{1}{2}(0.357 + 0.111) = 0.234$

**Total loss:** $0.01 + 0.234 = 0.244$

In this example, the KL term dominates -- the encoder is producing distributions that are somewhat far from $\mathcal{N}(0, I)$. As training progresses, the two terms will balance.

---

## 8. Properties of the VAE Latent Space

### 8.1 Smoothness and Continuity

The KL regularization ensures that the latent space has no "holes." Since all encoded distributions are pushed toward $\mathcal{N}(0, I)$, the regions between different data points are filled in. Decoding any point near the origin produces something plausible.

This is the key difference from vanilla autoencoders: the VAE's latent space is *smooth*. Small movements in latent space produce small changes in the decoded output.

### 8.2 Interpolation

One of the most visually striking properties of VAEs: you can interpolate between two data points in latent space, and the intermediate decoded images form a smooth transition.

Given two inputs $x\_1$ and $x\_2$:
1. Encode: $\mu\_1 = \mu\_\phi(x\_1)$, $\mu\_2 = \mu\_\phi(x\_2)$
2. Interpolate: $z\_t = (1-t)\mu\_1 + t\mu\_2$ for $t \in [0, 1]$
3. Decode: $\hat{x}\_t = g\_\theta(z\_t)$

For MNIST, interpolating between a "3" and an "8" produces intermediate shapes that gradually morph from one digit to the other, passing through plausible intermediate forms. This is qualitatively different from pixel-space interpolation, which would produce ghostly overlaps.

### 8.3 Disentanglement

Ideally, each dimension of $z$ would control one interpretable factor of variation. For example:
- $z\_1$ controls digit identity
- $z\_2$ controls slant
- $z\_3$ controls stroke width

When this happens, the representation is called **disentangled**. Vanilla VAEs achieve some degree of disentanglement, but not perfectly. The $\beta$-VAE (Higgins et al., 2017) strengthens this by increasing the weight on the KL term.

### 8.4 Generation

To generate new data:
1. Sample $z \sim \mathcal{N}(0, I)$
2. Decode: $\hat{x} = g\_\theta(z)$

Because the KL term has regularized the latent space to be approximately $\mathcal{N}(0, I)$, these random samples fall in regions the decoder has been trained on, producing plausible outputs.

For 2D latent spaces, we can create a grid of $z$ values and decode each one, producing a "latent space atlas" showing how the decoded output varies across the latent space.

---

## 9. Training the VAE

### 9.1 The Training Loop

```python
for x_batch in dataloader:
    # Encode
    mu, log_var = encoder(x_batch)

    # Reparameterize
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)
    z = mu + std * epsilon

    # Decode
    x_recon = decoder(z)

    # Reconstruction loss (binary cross-entropy for Bernoulli decoder)
    recon_loss = F.binary_cross_entropy(x_recon, x_batch, reduction='sum')

    # KL divergence (closed form)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss
    loss = recon_loss + kl_loss

    loss.backward()
    optimizer.step()
```

Note: the reconstruction loss uses `reduction='sum'` (sum over dimensions and batch), and the KL is also summed. Some implementations average over the batch; just be consistent.

### 9.2 Training Dynamics

In the early stages of training, the reconstruction loss is large and dominates. The KL term is initially small because the encoder has not learned to produce anything useful (random weights produce high-entropy, vaguely Gaussian outputs).

As training progresses:
- The encoder learns to produce informative latent codes (reconstruction improves)
- The KL term grows as the encoder tries to encode information (pushing away from the prior)
- Eventually, a balance is reached

### 9.3 The Reconstruction-Regularization Trade-off

The two terms in the VAE loss pull in opposite directions:
- **Reconstruction** wants each $x$ to have a very specific, informative $z$ (small $\sigma$, distinct $\mu$)
- **KL regularization** wants all $z$ distributions to look like $\mathcal{N}(0, I)$ (large $\sigma$, $\mu$ near 0)

If the reconstruction term wins: the VAE degenerates toward a regular autoencoder (tight posteriors, unstructured latent space).

If the KL term wins: the encoder ignores the input and produces $q(z|x) \approx \mathcal{N}(0, I)$ for all $x$. The decoder then produces the average image for all $z$. This pathology is called **posterior collapse** or **KL vanishing**.

---

## 10. VAE Limitations and Extensions

### 10.1 Posterior Collapse

Posterior collapse occurs when the encoder "gives up" and sets $q\_\phi(z|x) \approx p(z)$ for all $x$, making the KL term zero. The decoder then ignores $z$ and produces a blurry average. This happens when:
- The decoder is too powerful (can produce good reconstructions without $z$)
- The KL term is weighted too strongly
- Training dynamics allow the KL to decrease too quickly

Mitigations:
- **KL annealing:** Start training with the KL term weight at 0 and gradually increase it to 1. This lets the encoder learn useful representations before the regularization kicks in.
- **Free bits:** Allow each KL dimension to contribute at least $\lambda$ nats before being penalized.
- **Use a weaker decoder:** If the decoder cannot model the data without $z$, it will be forced to use $z$.

### 10.2 Blurriness

VAE-generated images are often blurry. This is a consequence of the Gaussian assumption: the decoder outputs the *mean* of a Gaussian over pixel space, and the mean of a distribution over sharp images is a blurry image.

Consider two images of the digit "1," one leaning left and one leaning right. The mean of these two images is a blurry "1" in the middle. The VAE, by maximizing the expected log-likelihood under a Gaussian, learns to output this mean.

GANs (Generative Adversarial Networks) avoid this problem by using a different training objective, but at the cost of training instability and mode collapse.

### 10.3 $\beta$-VAE

The $\beta$-VAE (Higgins et al., 2017) modifies the VAE objective:

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{z \sim q_\phi}[\log p_\theta(x|z)] - \beta \cdot D_{\text{KL}}(q_\phi(z|x) \Vert p(z))
$$

where $\beta > 1$ increases the pressure to match the prior, encouraging more disentangled representations at the cost of reconstruction quality. When $\beta = 1$, this is the standard VAE.

The intuition: a stronger KL penalty forces the model to be more "efficient" in how it uses the latent space, which often leads to each dimension encoding a single factor of variation.

### 10.4 Beyond Gaussian: VQ-VAE

The Vector Quantized VAE (VQ-VAE, van den Oord et al., 2017) replaces the continuous Gaussian latent space with a discrete codebook. The encoder maps to the nearest codebook vector, producing a discrete representation.

This avoids the blurriness problem (the decoder sees discrete codes, not averages) and produces much sharper outputs. It was an important precursor to modern image generation systems.

### 10.5 Connection to Diffusion Models

Recall from Week 7 that denoising autoencoders learn the score function $\nabla\_x \log p(x)$. Diffusion models (Ho et al., 2020) formalize this into a multi-step denoising process: starting from pure noise, iteratively denoise to produce a sample. The VAE can be seen as a one-step version of this idea (encode to noise, decode to data), while diffusion models use many small steps.

The intellectual lineage: DAE $\to$ VAE $\to$ Diffusion Models. We are studying the history of a field that led to the image generation revolution.

---

## 11. VAE vs. Autoencoder: A Summary

| Property | Autoencoder | VAE |
|----------|------------|-----|
| Encoder output | Point $z$ | Distribution $q(z|x)$ |
| Latent space | Unstructured | Close to $\mathcal{N}(0,I)$ |
| Loss function | Reconstruction only | Reconstruction + KL |
| Generation | Poor | Good (sample from prior) |
| Interpolation | May be discontinuous | Smooth |
| Reconstruction quality | Better | Slightly blurry |
| Training | Simpler | Requires reparam. trick |

---

## Summary

This week we built the Variational Autoencoder from first principles:

1. **Motivation:** Vanilla autoencoders have unstructured latent spaces that do not support generation.

2. **Latent variable models:** We formulated data generation as $z \sim p(z)$, $x \sim p\_\theta(x|z)$, and discovered that the true posterior $p\_\theta(z|x)$ is intractable.

3. **Variational inference:** We approximated the posterior with $q\_\phi(z|x)$ and derived the **ELBO** -- a lower bound on $\log p\_\theta(x)$ that decomposes into reconstruction and KL regularization.

4. **The reparameterization trick:** By writing $z = \mu + \sigma \odot \epsilon$, we made the stochastic computation graph differentiable, enabling end-to-end training with backpropagation.

5. **KL divergence:** We derived the closed-form KL between the Gaussian encoder and the standard Gaussian prior.

6. **Properties:** The VAE latent space is smooth, supports interpolation and generation, and can be disentangled with $\beta$-VAE.

7. **Limitations:** Posterior collapse, blurriness, and the Gaussian assumption. Extensions like VQ-VAE and diffusion models address these.

The VAE is the last autoencoder *variant* we study. Starting next week, we turn to the mathematical foundations of sparsity (Week 9) and then build the sparse autoencoder (Week 10) -- the architecture at the heart of this course.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| ELBO | $\log p(x) \geq \mathbb{E}\_q[\log p(x|z)] - D\_{\text{KL}}(q(z|x) \Vert p(z))$ |
| ELBO gap | $\log p(x) = \text{ELBO} + D\_{\text{KL}}(q(z|x) \Vert p(z|x))$ |
| Reparameterization | $z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$ |
| KL (VAE) | $D\_{\text{KL}} = \frac{1}{2}\sum\_j (\mu\_j^2 + \sigma\_j^2 - \log \sigma\_j^2 - 1)$ |
| KL (general Gaussian) | $D\_{\text{KL}}(\mathcal{N}\_1 \Vert \mathcal{N}\_2) = \frac{1}{2}[\log\frac{|\Sigma\_2|}{|\Sigma\_1|} - d + \text{tr}(\Sigma\_2^{-1}\Sigma\_1) + \Delta\mu^\top \Sigma\_2^{-1} \Delta\mu]$ |

---

## Suggested Reading

- **Kingma and Welling** (2014), "Auto-Encoding Variational Bayes" -- the original VAE paper. Dense but essential.
- **Doersch** (2016), "Tutorial on Variational Autoencoders" -- an excellent tutorial, more accessible than the original paper.
- **Goodfellow et al.** (2016), *Deep Learning*, Chapter 20 -- covers deep generative models including VAEs.
- **Higgins et al.** (2017), "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" -- the disentanglement extension.
- **Blei, Kucukelbir, McAuliffe** (2017), "Variational Inference: A Review for Statisticians" -- broader context for variational methods.
