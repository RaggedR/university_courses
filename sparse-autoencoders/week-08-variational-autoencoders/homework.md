# Week 8: Variational Autoencoders -- Homework

**Estimated time:** 10-12 hours
**Prerequisites:** Probability and KL divergence (Week 2), autoencoder implementation (Week 6), PyTorch

---

## Problem 1: Deriving the ELBO (Theory)

This problem walks you through the complete ELBO derivation from scratch. Fill in every step -- do not skip any.

### Part (a): Setup

Write down the log marginal likelihood $\log p\_\theta(x)$ as the log of an integral over the joint distribution $p\_\theta(x, z)$. Then introduce the variational distribution $q\_\phi(z|x)$ by multiplying and dividing inside the integral.

### Part (b): Apply Jensen's Inequality

Recall Jensen's inequality: for a concave function $f$ and random variable $X$, $f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$.

Apply Jensen's inequality to the expression from Part (a) to derive:

$$
\log p_\theta(x) \geq \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]
$$

### Part (c): Decompose the ELBO

Starting from the ELBO:

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]
$$

Use $p\_\theta(x, z) = p\_\theta(x|z) p(z)$ to decompose this into:

$$
\mathcal{L} = \mathbb{E}_{z \sim q_\phi}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \Vert  p(z))
$$

Show every step of the algebra.

### Part (d): The Gap

Prove that:

$$
\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + D_{\text{KL}}(q_\phi(z|x) \Vert  p_\theta(z|x))
$$

*Hint: Start from the definition of $D\_{\text{KL}}(q\_\phi(z|x) \Vert  p\_\theta(z|x))$ and use Bayes' rule to substitute for $p\_\theta(z|x)$.*

### Part (e): Interpretation

1. Why does the gap being $D\_{\text{KL}}(q\_\phi(z|x) \Vert  p\_\theta(z|x)) \geq 0$ guarantee that the ELBO is indeed a lower bound?
2. Under what condition is the bound tight (gap = 0)?
3. Why can't we just minimize this KL divergence directly?

---

## Problem 2: KL Divergence Between Multivariate Gaussians (Theory)

### Part (a): General Case

Derive the KL divergence between two multivariate Gaussians:

$$
D_{\text{KL}}(\mathcal{N}(\mu_1, \Sigma_1) \Vert  \mathcal{N}(\mu_2, \Sigma_2))
$$

Start from the definition:

$$
D_{\text{KL}}(p \Vert  q) = \mathbb{E}_{x \sim p}\left[\log p(x) - \log q(x)\right]
$$

Write out the Gaussian log-densities, expand, and simplify. You will need these identities:
- $\mathbb{E}\_{x \sim \mathcal{N}(\mu, \Sigma)}[x] = \mu$
- $\mathbb{E}\_{x \sim \mathcal{N}(\mu, \Sigma)}[(x - \mu)(x - \mu)^\top] = \Sigma$
- $\mathbb{E}\_{x \sim \mathcal{N}(\mu, \Sigma)}[x^\top A x] = \mu^\top A \mu + \text{tr}(A\Sigma)$

Show that you arrive at:

$$
D_{\text{KL}} = \frac{1}{2}\left[\log \frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^\top \Sigma_2^{-1}(\mu_2 - \mu_1)\right]
$$

### Part (b): VAE Special Case

Specialize the result from Part (a) to the VAE case:
- $q\_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$
- $p(z) = \mathcal{N}(0, I)$

Show that:

$$
D_{\text{KL}} = \frac{1}{2}\sum_{j=1}^{d_z} \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)
$$

### Part (c): Verify with Numbers

Compute $D\_{\text{KL}}$ for $q = \mathcal{N}(\mu, \sigma^2 I)$ with $d\_z = 2$ for the following cases:
1. $\mu = (0, 0)$, $\sigma^2 = (1, 1)$ (should be 0)
2. $\mu = (1, 0)$, $\sigma^2 = (1, 1)$
3. $\mu = (0, 0)$, $\sigma^2 = (2, 2)$
4. $\mu = (0, 0)$, $\sigma^2 = (0.1, 0.1)$

Explain intuitively why case (4) has a higher KL than case (3).

---

## Problem 3: The VAE Special Case: KL Divergence (Theory)

### Part (a)

Starting from the definition of KL divergence as an expectation, derive the closed-form expression for the KL divergence $D\_{\text{KL}}(q(z|x) \Vert  p(z))$ where $q(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p(z) = \mathcal{N}(0, I)$.

Do this **without** using the general multivariate Gaussian result from Problem 2. Instead, work from:

$$
D_{\text{KL}} = \mathbb{E}_{z \sim q}\left[\log q(z|x) - \log p(z)\right]
$$

Write out the log-densities for the diagonal Gaussian $q$ and standard Gaussian $p$, expand, and take expectations.

*This is good practice: you should be able to derive the result both ways.*

### Part (b)

Prove that $D\_{\text{KL}}(q \Vert  p) = 0$ if and only if $\mu = 0$ and $\sigma^2 = \mathbf{1}$ (the vector of all ones).

*Hint: Show that for each dimension $j$, the function $f(\mu\_j, \sigma\_j^2) = \mu\_j^2 + \sigma\_j^2 - \log \sigma\_j^2 - 1$ has a unique minimum at $\mu\_j = 0, \sigma\_j^2 = 1$.*

---

## Problem 4: Implement a VAE (Implementation)

Implement a Variational Autoencoder in PyTorch for MNIST.

### Architecture

- **Encoder:** $784 \to 512 \to 256$, ReLU activations. The final layer branches into two heads:
  - $\mu$: Linear $256 \to d\_z$
  - $\log \sigma^2$: Linear $256 \to d\_z$
- **Decoder:** $d\_z \to 256 \to 512 \to 784$, ReLU activations, sigmoid on the final layer.
- **Latent dimension:** $d\_z = 20$ (but also experiment with $d\_z = 2$ for visualization).

### Requirements

Your implementation must include:

1. **A `VAE` class** with `encode(x)`, `reparameterize(mu, log_var)`, `decode(z)`, and `forward(x)` methods.

2. **A loss function** that computes both the reconstruction loss (binary cross-entropy) and the KL divergence. The function should return both terms separately (for logging).

3. **A training loop** that trains for at least 30 epochs on MNIST with Adam optimizer (lr=1e-3).

### Deliverables

For the $d\_z = 20$ model:
1. Plot the training loss (total, reconstruction, and KL) vs. epoch.
2. Show 10 original test images and their reconstructions side by side.
3. Compare the reconstruction quality to a vanilla autoencoder with the same architecture (from Week 6 or re-implement).

For the $d\_z = 2$ model:
4. Plot the latent space: encode all test images, plot $\mu\_1$ vs. $\mu\_2$, colour-coded by digit label. How well-separated are the clusters?
5. Create a 20x20 grid of latent points spanning $[-3, 3] \times [-3, 3]$. Decode each point and display the resulting images as a grid. This is the "latent space atlas."

---

## Problem 5: Generation and Sampling (Implementation)

Using your trained VAE from Problem 4 ($d\_z = 20$):

### Part (a): Random Generation

Sample 100 latent vectors $z \sim \mathcal{N}(0, I)$ and decode them. Display the results as a 10x10 grid. How many look like recognizable digits? How many are blurry or ambiguous?

### Part (b): Conditional Generation (Latent Space Exploration)

Encode 10 examples of each digit (0-9). Compute the average $\mu$ for each digit class. Then decode these 10 average latent vectors. Are the decoded images recognizable? Are they "prototypical" versions of each digit?

### Part (c): Arithmetic in Latent Space

A famous property of good latent spaces: "vector arithmetic" works. Try:
1. Compute $z\_{\text{smile}} = \bar{z}\_3 - \bar{z}\_2 + \bar{z}\_5$ (where $\bar{z}\_k$ is the average encoding of digit $k$). What digit does decoding this produce?
2. Try other combinations. Document your findings.

Note: this works better with richer data (faces, etc.) but can produce interesting results even on MNIST.

---

## Problem 6: Interpolation (Implementation)

### Part (a): VAE Interpolation

Select two test images of different digits (e.g., a "2" and a "7"). Encode both to get $\mu\_1$ and $\mu\_2$.

Create a sequence of 10 interpolated latent vectors:
$$
z_t = (1-t)\mu_1 + t\mu_2, \quad t \in \lbrace 0, 0.11, 0.22, \ldots, 1.0\rbrace 
$$

Decode each $z\_t$ and display the sequence. The transition should be smooth.

### Part (b): Vanilla AE Interpolation

Train a vanilla autoencoder with the same architecture (but outputting a point $z$ instead of $\mu, \sigma$). Perform the same interpolation.

### Part (c): Comparison

Display the VAE and vanilla AE interpolation sequences side by side. Which is smoother? Does the vanilla AE produce intermediate images that look like valid digits, or does it pass through unrecognizable states?

Explain why the VAE interpolation is smoother, relating your answer to the KL regularization term.

---

## Problem 7: $\beta$-VAE and Disentanglement (Implementation)

The $\beta$-VAE modifies the loss to:

$$
\mathcal{L}_{\beta\text{-VAE}} = \text{Reconstruction loss} + \beta \cdot D_{\text{KL}}
$$

### Part (a): Training

Train VAEs with $\beta \in \lbrace 0.1, 0.5, 1.0, 2.0, 5.0, 10.0\rbrace $, all with $d\_z = 10$. For each:
1. Report the final reconstruction loss and KL divergence separately.
2. Show 10 reconstructed test images.

### Part (b): Reconstruction vs. Regularization Trade-off

Plot reconstruction loss vs. KL divergence across the six $\beta$ values. You should see a clear trade-off curve (Pareto front).

### Part (c): Latent Traversals

For the $\beta = 1$ and $\beta = 5$ models, perform "latent traversals":
1. Encode a test image to get $\mu$.
2. For each latent dimension $j$, create a sequence where $z\_j$ varies from $-3$ to $+3$ while all other dimensions stay fixed at $\mu$.
3. Decode each point in the sequence.
4. Display as a grid: rows = latent dimensions, columns = traversal values.

Compare the $\beta = 1$ and $\beta = 5$ traversals. In which model does each latent dimension control a more distinct, interpretable factor? This is disentanglement.

### Part (d): Discussion

1. What happens as $\beta \to \infty$? What is the limiting behaviour of the encoder?
2. What happens as $\beta \to 0$? How does this relate to a vanilla autoencoder?
3. Why might disentanglement be useful for downstream tasks?

---

## Submission Checklist

- [ ] Problem 1: Complete ELBO derivation with all steps shown
- [ ] Problem 2: KL divergence derivation (general case), numerical verification
- [ ] Problem 3: Alternative KL derivation from scratch, proof of uniqueness of minimum
- [ ] Problem 4: Working VAE implementation, training curves, reconstructions, latent space visualizations
- [ ] Problem 5: Generated samples, conditional generation, latent arithmetic
- [ ] Problem 6: Interpolation comparison between VAE and vanilla AE
- [ ] Problem 7: $\beta$-VAE experiments, trade-off curve, latent traversals

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs.
