# CS 371: Final Examination — Solutions

---

## Section 1: Foundations

### Question 1.1

**(a)** [2 marks]

The eigenvectors of the covariance matrix $\mathbf{C}$ are the **principal directions** of the data — the orthogonal directions along which the data has maximum variance. The corresponding eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$ are the **variances** of the data projected along each principal direction. Geometrically, the eigenvectors define the axes of the ellipsoid that best approximates the data cloud, and the eigenvalues give the squared lengths of each axis.

**(b)** [4 marks]

Let $\mathbf{C} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top$ be the eigendecomposition, where $\mathbf{U} = [\mathbf{u}_1, \ldots, \mathbf{u}_d]$ and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_d)$ with $\lambda_1 \geq \cdots \geq \lambda_d$.

We want to maximize $\text{tr}(\mathbf{W}^\top \mathbf{C} \mathbf{W})$ subject to $\mathbf{W}^\top \mathbf{W} = \mathbf{I}_k$.

Substitute $\mathbf{C} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top$:

$$\text{tr}(\mathbf{W}^\top \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top \mathbf{W})$$

Let $\mathbf{V} = \mathbf{U}^\top \mathbf{W}$. Since $\mathbf{U}$ is orthogonal, $\mathbf{V}^\top \mathbf{V} = \mathbf{W}^\top \mathbf{U} \mathbf{U}^\top \mathbf{W} = \mathbf{W}^\top \mathbf{W} = \mathbf{I}_k$, so $\mathbf{V}$ also has orthonormal columns.

The objective becomes:

$$\text{tr}(\mathbf{V}^\top \boldsymbol{\Lambda} \mathbf{V}) = \sum_{j=1}^{k} \mathbf{v}_j^\top \boldsymbol{\Lambda} \mathbf{v}_j = \sum_{j=1}^{k} \sum_{i=1}^{d} \lambda_i v_{ij}^2$$

Since $\mathbf{v}_j$ is a unit vector, $\sum_i v_{ij}^2 = 1$ for each $j$. This is a weighted sum of the eigenvalues, where the weights for each $j$ form a probability distribution over eigenvalues. To maximize a weighted sum where weights must sum to 1, we should put all weight on the largest values.

The maximum is achieved when each $\mathbf{v}_j$ is a standard basis vector $\mathbf{e}_{\sigma(j)}$, selecting the top $k$ eigenvalues. The maximum is $\sum_{i=1}^{k} \lambda_i$.

This means $\mathbf{V} = [\mathbf{e}_1, \ldots, \mathbf{e}_k]$ (up to column permutations), so $\mathbf{W} = \mathbf{U} \mathbf{V} = [\mathbf{u}_1, \ldots, \mathbf{u}_k]$ — the top $k$ eigenvectors of $\mathbf{C}$. $\square$

---

### Question 1.2

**(a)** [3 marks]

The maximum learning rate that guarantees descent is $\eta = \frac{1}{L}$.

Justification: For an $L$-smooth function, we have the descent lemma:

$$f(\mathbf{w} - \eta \nabla f(\mathbf{w})) \leq f(\mathbf{w}) - \eta(1 - \frac{L\eta}{2}) \|\nabla f(\mathbf{w})\|^2$$

For this to guarantee descent, we need $1 - \frac{L\eta}{2} > 0$, i.e., $\eta < \frac{2}{L}$. The optimal fixed step size (maximizing the guaranteed decrease) is $\eta = \frac{1}{L}$, which gives a guaranteed decrease of $\frac{1}{2L}\|\nabla f(\mathbf{w})\|^2$ per step.

**(b)** [3 marks]

The first moment estimate $\mathbf{m}_t$ is an exponential moving average of the gradients, serving as a **momentum** term that smooths out gradient noise and accelerates optimization along consistent gradient directions. The second moment estimate $\mathbf{v}_t$ is an exponential moving average of the squared gradients, estimating the **variance** (or scale) of the gradient for each parameter.

Adam divides each parameter's gradient by $\sqrt{v_t}$, effectively normalizing the step size by the gradient's typical magnitude in that dimension. On loss surfaces with different curvatures along different dimensions (e.g., a narrow ravine), vanilla SGD takes the same step size everywhere, leading to oscillation along high-curvature directions or slow progress along low-curvature directions. Adam adapts: parameters with large, consistent gradients get smaller effective learning rates, and parameters with small gradients get larger effective learning rates, leading to more uniform progress.

---

### Question 1.3

**(a)** [3 marks]

The KL divergence between continuous distributions $p$ and $q$ is:

$$D_{\text{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$$

Two properties:

1. **Non-negativity:** $D_{\text{KL}}(p \| q) \geq 0$ for all $p, q$, with equality iff $p = q$ almost everywhere. This follows from Jensen's inequality applied to the concave function $\log$: $\mathbb{E}_p[\log \frac{q}{p}] \leq \log \mathbb{E}_p[\frac{q}{p}] = \log 1 = 0$, so $D_{\text{KL}} = -\mathbb{E}_p[\log \frac{q}{p}] \geq 0$.

2. **Asymmetry:** $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general. This means KL divergence is not a true metric. For example, $D_{\text{KL}}(p \| q)$ is large when $p(x) > 0$ but $q(x) \approx 0$ (it penalizes $q$ for assigning low probability where $p$ is high), but $D_{\text{KL}}(q \| p)$ does not penalize this scenario.

**(b)** [5 marks]

The empirical distribution is $\hat{p}_{\text{data}}(x) = \frac{1}{n}\sum_{i=1}^n \delta(x - x_i)$.

The KL divergence from $\hat{p}_{\text{data}}$ to $p_\theta$ is:

$$D_{\text{KL}}(\hat{p}_{\text{data}} \| p_\theta) = \int \hat{p}_{\text{data}}(x) \log \frac{\hat{p}_{\text{data}}(x)}{p_\theta(x)} \, dx$$

$$= \int \hat{p}_{\text{data}}(x) \log \hat{p}_{\text{data}}(x) \, dx - \int \hat{p}_{\text{data}}(x) \log p_\theta(x) \, dx$$

$$= -H(\hat{p}_{\text{data}}) - \mathbb{E}_{\hat{p}_{\text{data}}}[\log p_\theta(x)]$$

The first term $-H(\hat{p}_{\text{data}})$ is the negative entropy of the empirical distribution, which is a constant with respect to $\theta$.

Therefore:

$$\arg\min_\theta D_{\text{KL}}(\hat{p}_{\text{data}} \| p_\theta) = \arg\max_\theta \mathbb{E}_{\hat{p}_{\text{data}}}[\log p_\theta(x)]$$

$$= \arg\max_\theta \frac{1}{n} \sum_{i=1}^n \log p_\theta(x_i)$$

This is exactly the maximum likelihood objective (up to the constant $\frac{1}{n}$). $\square$

---

## Section 2: Autoencoders and Representation Learning

### Question 2.1

**(a)** [5 marks]

The reconstruction loss is:

$$\mathcal{L} = \mathbb{E}[\|\mathbf{x} - \mathbf{W}_d \mathbf{W}_e \mathbf{x}\|^2] = \text{tr}\left((\mathbf{I} - \mathbf{W}_d \mathbf{W}_e)^\top (\mathbf{I} - \mathbf{W}_d \mathbf{W}_e) \mathbf{C}\right)$$

Let $\mathbf{P} = \mathbf{W}_d \mathbf{W}_e$. The loss is $\mathbb{E}[\|(\mathbf{I} - \mathbf{P})\mathbf{x}\|^2] = \text{tr}((\mathbf{I} - \mathbf{P})^\top(\mathbf{I} - \mathbf{P}) \mathbf{C})$.

Since $\mathbf{W}_e \in \mathbb{R}^{k \times d}$ and $\mathbf{W}_d \in \mathbb{R}^{d \times k}$, $\mathbf{P}$ is a $d \times d$ matrix of rank at most $k$. The loss is minimized when $\mathbf{P}$ is the orthogonal projection onto a $k$-dimensional subspace that captures the most variance.

To see this, note that $\mathcal{L} = \text{tr}(\mathbf{C}) - 2\text{tr}(\mathbf{P}\mathbf{C}) + \text{tr}(\mathbf{P}^\top \mathbf{P} \mathbf{C})$. At the optimum, $\mathbf{P}$ should be an orthogonal projection ($\mathbf{P}^2 = \mathbf{P}$, $\mathbf{P}^\top = \mathbf{P}$), because for any rank-$k$ $\mathbf{P}$, the orthogonal projection onto its column space achieves lower or equal loss. For an orthogonal projection, $\mathbf{P}^\top \mathbf{P} = \mathbf{P}$, so:

$$\mathcal{L} = \text{tr}(\mathbf{C}) - \text{tr}(\mathbf{P}\mathbf{C})$$

Minimizing $\mathcal{L}$ is equivalent to maximizing $\text{tr}(\mathbf{P}\mathbf{C})$. Writing $\mathbf{P} = \sum_{j=1}^k \mathbf{w}_j \mathbf{w}_j^\top$ for orthonormal $\{\mathbf{w}_j\}$:

$$\text{tr}(\mathbf{P}\mathbf{C}) = \sum_{j=1}^k \mathbf{w}_j^\top \mathbf{C} \mathbf{w}_j$$

By the result from Q1.1(b), this is maximized when $\{\mathbf{w}_j\}$ are the top $k$ eigenvectors of $\mathbf{C}$. Therefore, the columns of $\mathbf{W}_d$ span the top-$k$ PCA subspace at the optimum. $\square$

**(b)** [3 marks]

The equivalence breaks for nonlinear autoencoders because they can capture **curved** (nonlinear) manifolds, not just flat (linear) subspaces. PCA projects onto a linear subspace, which fails when the data lies on a curved manifold.

**Concrete example:** Consider data points uniformly distributed along a half-circle (semicircle) in 2D: $\{(\cos\theta, \sin\theta) : \theta \in [0, \pi]\}$. The best 1D linear projection (PCA) would project onto the $x$-axis, losing information about whether a point is on the upper or lower part of the circle near the ends. A nonlinear autoencoder with a 1D bottleneck can learn to encode $\theta$ (the angle parameter), which perfectly parameterizes the semicircle — zero reconstruction error, whereas PCA necessarily has nonzero error.

---

### Question 2.2

**(a)** [6 marks]

Starting from the marginal log-likelihood:

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$$

Introduce an arbitrary distribution $q_\phi(\mathbf{z}|\mathbf{x})$:

$$\log p_\theta(\mathbf{x}) = \log \int q_\phi(\mathbf{z}|\mathbf{x}) \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \, d\mathbf{z}$$

$$= \log \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$$

By Jensen's inequality (since $\log$ is concave):

$$\geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$$

Expanding $p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})$:

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log p_\theta(\mathbf{x}|\mathbf{z}) + \log p(\mathbf{z}) - \log q_\phi(\mathbf{z}|\mathbf{x})\right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] + \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

This is the ELBO. $\square$

**(b)** [3 marks]

The reparameterization trick rewrites a sample from $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$ as:

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This is necessary because the expectation $\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]$ involves sampling from a distribution that depends on $\phi$. If we sample $\mathbf{z} \sim q_\phi$ directly, the sampling operation is non-differentiable with respect to $\phi$ — we cannot backpropagate through the stochastic sampling step. The reparameterization trick moves the stochasticity into the fixed distribution $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, making $\mathbf{z}$ a deterministic, differentiable function of $\phi$ (given $\boldsymbol{\epsilon}$), enabling standard backpropagation.

**(c)** [3 marks]

When $\beta \gg 1$: The KL term dominates the objective. The model is strongly penalized for having $q_\phi(\mathbf{z}|\mathbf{x})$ deviate from the prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$. The result is that the encoder collapses to roughly $q_\phi(\mathbf{z}|\mathbf{x}) \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$ for all inputs — the latent space becomes **uninformative** (posterior collapse). Reconstruction quality degrades severely because the latent code carries almost no information about the input. The latent space is smooth and well-structured but useless.

When $\beta \to 0$: The KL term vanishes. The model becomes a standard autoencoder with a stochastic bottleneck. Reconstruction quality is excellent, but the latent space has no structure — $q_\phi(\mathbf{z}|\mathbf{x})$ can be arbitrary, with no pressure to match the prior. The latent space is not useful for generation (sampling from $p(\mathbf{z})$ and decoding produces garbage).

The tradeoff: $\beta$ controls the balance between reconstruction fidelity and latent space regularity. The standard VAE ($\beta = 1$) balances both; increasing $\beta$ sacrifices reconstruction for a more disentangled, structured latent space.

---

## Section 3: Sparsity and Sparse Autoencoders

### Question 3.1

**(a)** [5 marks]

With $\mathbf{D} = \mathbf{I}$, the LASSO objective becomes:

$$\min_{\mathbf{z}} \frac{1}{2}\|\mathbf{x} - \mathbf{z}\|_2^2 + \lambda \|\mathbf{z}\|_1$$

This separates into independent scalar problems for each $i$:

$$\min_{z_i} \frac{1}{2}(x_i - z_i)^2 + \lambda |z_i|$$

**Case 1: $z_i > 0$.** The objective is $\frac{1}{2}(x_i - z_i)^2 + \lambda z_i$. Taking the derivative and setting to zero: $-(x_i - z_i) + \lambda = 0$, so $z_i = x_i - \lambda$. This is valid (positive) only if $x_i > \lambda$.

**Case 2: $z_i < 0$.** The objective is $\frac{1}{2}(x_i - z_i)^2 - \lambda z_i$. Taking the derivative: $-(x_i - z_i) - \lambda = 0$, so $z_i = x_i + \lambda$. This is valid (negative) only if $x_i < -\lambda$.

**Case 3: $z_i = 0$.** Valid when $|x_i| \leq \lambda$, which can be verified by checking that the subdifferential of the objective at $z_i = 0$ contains 0.

Combining all cases:

$$z_i^* = \text{sign}(x_i) \max(|x_i| - \lambda, 0) = S_\lambda(x_i)$$

This is the soft-thresholding operator. $\square$

**(b)** [3 marks]

The soft-thresholding operator is a piecewise linear function:
- For $|x_i| \leq \lambda$: output is 0 (the "dead zone").
- For $x_i > \lambda$: output is $x_i - \lambda$ (a line with slope 1, shifted down by $\lambda$).
- For $x_i < -\lambda$: output is $x_i + \lambda$ (a line with slope 1, shifted up by $\lambda$).

**Hard thresholding** is $z_i = x_i \cdot \mathbb{1}[|x_i| > \lambda]$: values below the threshold are zeroed, but values above are passed through unchanged.

**Soft thresholding introduces shrinkage bias** because all active values are reduced by $\lambda$ — the output $x_i - \lambda$ is always smaller in magnitude than the input $x_i$. Hard thresholding does not shrink active values, but it has a discontinuity at $|x_i| = \lambda$, making optimization harder (the objective with hard thresholding is non-convex).

---

### Question 3.2

**(a)** [4 marks]

One ISTA iteration has two steps:

1. **Gradient step** on the smooth part of the objective: $\mathbf{u}^{(t)} = \mathbf{z}^{(t)} + \frac{1}{L}\mathbf{D}^\top(\mathbf{x} - \mathbf{D}\mathbf{z}^{(t)})$. This computes the residual $\mathbf{x} - \mathbf{D}\mathbf{z}^{(t)}$, projects it back to the code space via $\mathbf{D}^\top$, and takes a gradient step with step size $\frac{1}{L}$ to reduce the reconstruction error.

2. **Proximal step** for the non-smooth L1 term: $\mathbf{z}^{(t+1)} = S_{\lambda/L}(\mathbf{u}^{(t)})$. This applies soft-thresholding, which is the proximal operator of the L1 norm. It enforces sparsity by shrinking small components to zero.

We cannot use standard gradient descent on the full objective $\frac{1}{2}\|\mathbf{x} - \mathbf{D}\mathbf{z}\|_2^2 + \lambda \|\mathbf{z}\|_1$ because the L1 term $\lambda \|\mathbf{z}\|_1$ is **non-differentiable** at $z_i = 0$. The subgradient exists but using subgradient descent converges very slowly and does not produce exact zeros. The proximal splitting approach (ISTA) handles the non-smooth term analytically via the proximal operator while using gradient descent on the smooth term.

**(b)** [3 marks]

A single forward pass of an SAE encoder computes $\mathbf{z} = \text{ReLU}(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)$. This has the same structural form as one ISTA step: a linear transformation of $\mathbf{x}$ followed by a nonlinear thresholding operation. The matrix $\mathbf{W}_e$ plays the role of $\mathbf{D}^\top$ (projecting from data space to code space), and ReLU plays the role of thresholding (though ReLU is not exactly soft-thresholding).

The SAE **amortizes** inference: instead of running ISTA for many iterations to convergence for each input $\mathbf{x}$, the SAE learns weights $\mathbf{W}_e$ that produce a good sparse code in a single step for all inputs in the training distribution. What the SAE sacrifices is **optimality per input** — ISTA converges to the exact minimizer for each $\mathbf{x}$, while the SAE uses a single linear + nonlinear step that is approximate. However, the SAE is much faster at test time (one forward pass vs. many iterations).

---

### Question 3.3

**(a)** [2 marks]

The expansion factor is $\frac{n\_\text{features}}{d\_\text{model}} = \frac{512 \times 8}{512} = 8$. The SAE is **overcomplete** because the hidden dimension (4096) is larger than the input dimension (512).

**(b)** [3 marks]

**Issue 1: No decoder column normalization.** The decoder weights are unconstrained, which creates a scaling degeneracy: the model can make encoder activations small (reducing L1 penalty) while making decoder columns large (maintaining reconstruction). This defeats the purpose of the sparsity penalty because the L1 loss can be minimized without actually making the representation sparse in a meaningful sense.

*Fix:* After each optimizer step, normalize each column of the decoder weight matrix to unit norm: `sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)`.

**Issue 2: No subtraction of the data mean (centering).** Activation vectors from neural networks typically have a nonzero mean. Without subtracting this mean before encoding (and adding it back after decoding), the SAE wastes capacity encoding the constant offset, and the bias terms absorb the mean rather than learning meaningful thresholds.

*Fix:* Compute a running mean of the activations and subtract it before encoding: `x_centered = x - running_mean`, then add it back: `x_hat = decoder(z) + running_mean`.

(Other valid issues: no learning rate warmup or decay, using `.sum(dim=-1)` for MSE instead of `.mean(dim=-1)` which makes the loss scale with dimension, no dead neuron resampling.)

**(c)** [2 marks]

60% dead features indicate that $\lambda = 10^{-3}$ is likely **too large**, pushing most features permanently below zero. Once below zero, ReLU blocks gradient flow, creating a death spiral. The L1 penalty overwhelms the reconstruction gradient for marginal features.

**Strategy 1: Reduce $\lambda$.** A smaller sparsity coefficient allows more features to maintain positive activations. Sweep $\lambda$ and monitor the fraction of active features.

**Strategy 2: Dead neuron resampling.** Periodically check for dead features (e.g., features that haven't fired in the last $N$ batches). Reinitialize dead features by setting their encoder weights to the direction of a poorly-reconstructed input (a data point with high residual error) and resetting their decoder column. This gives dead features a second chance to learn useful patterns.

**(d)** [3 marks]

To compute CE loss recovered:

1. Run the language model on a held-out evaluation set and record the cross-entropy loss $\mathcal{L}_{\text{orig}}$.
2. Run the model again, but at the SAE's target layer, replace the model's activations with the SAE's reconstructions. Record $\mathcal{L}_{\text{SAE}}$.
3. Run the model a third time, replacing the target layer's activations with zeros. Record $\mathcal{L}_{\text{zero}}$.
4. Compute: $\text{CE recovered} = \frac{\mathcal{L}_{\text{zero}} - \mathcal{L}_{\text{SAE}}}{\mathcal{L}_{\text{zero}} - \mathcal{L}_{\text{orig}}}$.

The baseline is $\mathcal{L}_{\text{zero}}$ (worst case: all information at this layer is destroyed) and the reference is $\mathcal{L}_{\text{orig}}$ (best case: perfect reconstruction).

A CE loss recovered of 0.85 means the SAE reconstruction preserves 85% of the useful information at this layer. The model's performance degrades by only 15% of the gap between "no information" and "perfect information." This is reasonably good — the SAE captures most of the functionally important structure — but there is still meaningful information being lost, which could correspond to subtle features the SAE fails to represent.

---

## Section 4: Mechanistic Interpretability

### Question 4.1

**(a)** [4 marks]

**Superposition** is the phenomenon where a neural network represents more features (meaningful directions in activation space) than it has dimensions (neurons). Instead of dedicating one neuron per feature, the network encodes features as directions in an overcomplete set — multiple features share the same neurons.

**Concrete example:** Consider a network layer with 100 neurons that needs to represent 500 binary features (e.g., "is the text about politics?", "is the word capitalized?", etc.). If each feature is sparse (active on only 1% of inputs), the network can assign each feature a nearly-orthogonal direction in 100-dimensional space. When a feature is active, it activates many neurons a little bit. The interference between features is small because they are rarely active simultaneously.

**Why networks use it:** Superposition allows networks to represent far more features than their dimensionality would suggest, increasing their effective capacity. It is favored when features are sparse — the interference cost (occasionally misrepresenting features that happen to co-occur) is outweighed by the benefit of representing more features.

**Why it makes interpretability difficult:** Superposition means individual neurons are **polysemantic** — they respond to multiple unrelated features. Looking at a single neuron and asking "what does it represent?" gives a confused answer because it participates in representing several features simultaneously. This is why naive neuron-level analysis fails and why we need tools like SAEs to decompose superimposed representations.

**(b)** [4 marks]

When features are sufficiently sparse, the expected interference cost of superposition is low. Consider two features $\mathbf{f}_1$ and $\mathbf{f}_2$ represented by columns $\mathbf{w}_1, \mathbf{w}_2$ of $\mathbf{W}$. The reconstruction error due to interference is proportional to $(\mathbf{w}_1^\top \mathbf{w}_2)^2$ times the probability that both features are simultaneously active. If features are active with probability $p$, the interference cost scales as $O(p^2)$, while the benefit of representing an additional feature (reducing the "not-represented" loss) is $O(p)$.

When $p$ is small enough, $O(p) \gg O(p^2)$, so the benefit of representing extra features exceeds the interference cost. The network can profitably represent $m > n$ features.

**Geometric structure:** The columns of $\mathbf{W}$ form a structure related to a **low-coherence frame** — a set of vectors whose pairwise inner products are small. In the extreme case of very high sparsity, the columns approach a structure called a near-orthogonal packing or a polytope inscribed in the unit sphere. For example, in 2D, five features might be arranged as the vertices of a regular pentagon (evenly spaced directions), maximizing the minimum angle between any pair.

The sparsity level determines the **transition point**: at high sparsity, many features can be superimposed with minimal interference. As sparsity decreases (features become more common), fewer features can be represented before interference dominates, and eventually the network falls back to representing at most $n$ features (one per neuron).

---

### Question 4.2

**(a)** [3 marks]

**Causal intervention experiment:**

1. Run the model on prompts where $f_{3421}$ is naturally active (e.g., text mentioning the Golden Gate Bridge) and record the model's output distribution (next-token probabilities).

2. **Activation patching / clamping:** Run the model on the same prompts but, at layer 8, set $f_{3421}$'s activation to zero (ablation) while keeping all other SAE features unchanged. Reconstruct the layer-8 activations from the modified SAE features and continue the forward pass.

3. **Measure:** Compare the output distribution with and without $f_{3421}$. Specifically, measure whether the probabilities of Golden Gate Bridge-related tokens (e.g., "Bridge," "San Francisco," "bay") decrease when $f_{3421}$ is ablated.

4. **Expected result supporting causal role:** Ablating $f_{3421}$ significantly reduces the probability of Golden Gate Bridge-related completions while having minimal effect on unrelated completions. Additionally, **clamping $f_{3421}$ to a high value** on prompts that do not mention the bridge should *increase* the probability of bridge-related completions.

**(b)** [4 marks]

The distinction is between the feature being **causal** (the model's computation flows through this feature) vs. being a **correlate** (the feature detects the same pattern as the model's computation but is not part of the causal chain).

**Experiment:**

1. **Ablation specificity test.** Ablate $f_{3421}$ and measure performance on a task that *requires* Golden Gate Bridge knowledge (e.g., "The Golden Gate Bridge is located in ___"). If the model's accuracy drops significantly, this suggests the feature is causal.

2. **Sufficiency test.** Take a prompt that does not mention the Golden Gate Bridge. Clamp $f_{3421}$ to a high activation value and check if the model begins producing Golden Gate Bridge-related output. If it does, the feature is sufficient to influence behavior.

3. **Alternative pathway test.** This is the key test to distinguish causation from correlation. Ablate $f_{3421}$ and check: does the model *still* correctly handle Golden Gate Bridge-related prompts, just using other features or pathways? If the model performs just as well without $f_{3421}$, then the feature is a correlate — the model has redundant pathways, and $f_{3421}$ is not on the critical path. If performance degrades and does not recover, $f_{3421}$ is causally necessary.

**Causal pattern:** Large performance drop on ablation AND successful steering on clamping = causal role.
**Correlate pattern:** No performance drop on ablation (even though the feature activates on bridge-related inputs) = the feature detects the concept but the model's computation does not depend on it.

---

### Question 4.3

**(a)** [4 marks]

**Shrinkage bias in L1 SAEs:** In a vanilla SAE with loss $\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \lambda \|\mathbf{z}\|_1$, the L1 penalty contributes a gradient of $\lambda \cdot \text{sign}(z_i)$ for each active feature $z_i > 0$. This pushes all active features toward zero, systematically reducing their magnitude. The result is that the SAE underestimates the true magnitude of active features.

**Gated SAE solution:** The Gated SAE decouples feature selection from magnitude estimation using two parallel paths:

- **Gating path:** $\boldsymbol{\pi}_{\text{gate}} = \mathbf{W}_e \mathbf{x} + \mathbf{b}_{\text{gate}}$, with gate $\mathbf{g} = \mathbb{1}[\boldsymbol{\pi}_{\text{gate}} > 0]$.
- **Magnitude path:** $\boldsymbol{\pi}_{\text{mag}} = \mathbf{W}_e \mathbf{x} + \mathbf{b}_{\text{mag}}$, with magnitude $\tilde{\mathbf{z}} = \text{ReLU}(\boldsymbol{\pi}_{\text{mag}})$.
- **Output:** $\mathbf{z} = \mathbf{g} \odot \tilde{\mathbf{z}}$.

The L1 penalty is applied to $\boldsymbol{\pi}_{\text{gate}}$ (the gate pre-activations), NOT to $\mathbf{z}$ or $\tilde{\mathbf{z}}$. This means:
- The sparsity penalty controls *which features are active* (through the gate), but does not touch the magnitude.
- For features where $g_i = 1$, the magnitude $\tilde{z}_i$ is determined purely by the magnitude path, which receives no L1 penalty. Hence, no shrinkage.

**(b)** [3 marks]

In a TopK SAE, the forward pass selects the $K$ largest pre-activations and zeros the rest. This is non-differentiable because the selection step involves a discontinuous indicator function.

Gradients are computed using the **straight-through estimator (STE)**: in the backward pass, gradients for the selected features (top-K) are passed through as if the selection were the identity function, and gradients for non-selected features are set to zero:

$$\frac{\partial \mathcal{L}}{\partial h_i} = \frac{\partial \mathcal{L}}{\partial z_i} \cdot \mathbb{1}[i \in \text{top-K}]$$

The key assumption is **local stability of the top-K set**: for small perturbations to the pre-activations $\mathbf{h}$, the identity of the top-K elements does not change (the same features remain in the top K). Under this assumption, the TopK operation is locally equivalent to the identity for selected features and locally equivalent to zero for unselected features, making the STE a valid local approximation to the gradient.

**(c)** [3 marks]

A key advantage of learned per-feature thresholds over both fixed-threshold and fixed-K approaches is **per-feature adaptivity to varying activation scales.** Different features may have pre-activations at very different scales. A feature detecting a rare concept might have pre-activations near 0.3, while a common-concept feature might have pre-activations near 5.0. A fixed threshold (zero in vanilla SAEs) cannot distinguish between a weakly active common feature and a strongly active rare feature. A fixed K forces the same number of features to fire for every input, regardless of whether the input genuinely involves many or few features.

Learned thresholds are most beneficial when the SAE's features have **heterogeneous activation distributions** — some features are rare with low-magnitude activations and some are common with high-magnitude activations. The thresholds adapt: rare features get low thresholds (easy to fire), common features get high thresholds (only fire when strongly indicated), leading to more faithful sparsity per feature.

---

## Section 5: Code Analysis and Design

### Question 5.1

**(a)** [2 marks]

The bug is that `torch.zeros_like(h)` creates a **new tensor disconnected from the computational graph**, and `scatter_` performs an **in-place operation** on this detached tensor. The `topk_vals` that are scattered in carry gradients, but the in-place scatter into a zeros tensor causes problems with autograd — specifically, the gradient does not properly flow back through the top-K selection because the mask construction is not part of the differentiable computation graph. The `scatter_` operation on a zeros tensor effectively detaches the gradient path from the original pre-activations `h`.

**(b)** [3 marks]

Corrected forward method:

```python
def forward(self, x):
    h = x @ self.W_enc + self.b_enc                # (B, n_features)
    topk_vals, topk_idx = torch.topk(h, self.k)    # (B, k)

    # Create mask from top-k indices
    mask = torch.zeros_like(h)
    mask.scatter_(1, topk_idx, 1.0)                 # binary mask

    # Apply mask to original h — gradients flow through h
    z = h * mask                                     # (B, n_features)

    x_hat = z @ self.W_dec + self.b_dec             # (B, d_model)
    return x_hat, z, h
```

The key fix: instead of scattering the values, we create a binary mask and multiply it element-wise with the original `h` tensor. Since `h` is in the computational graph and the mask is treated as a constant (no gradient needed for the mask itself), gradients flow through `h` for the selected positions. This implements the straight-through estimator: gradients pass through the top-K selected features.

**(c)** [3 marks]

Decoder column normalization should be done as a **post-processing step after `optimizer.step()`**, not inside `forward()` and not as a loss term.

**Why not inside `forward()`:** Normalizing inside the forward pass would change the computational graph at every step. The optimizer would compute gradients for unnormalized weights but then the next forward pass would use normalized weights, creating a mismatch. Also, normalization during forward would affect the gradient computation in confusing ways.

**Why not as a loss term:** Adding $\sum_j (\|\mathbf{w}_j\|^2 - 1)^2$ as a penalty would work in principle but is indirect — it encourages unit norm rather than enforcing it, and it introduces another hyperparameter (the penalty weight). The penalty must be quite strong to keep norms close to 1, which can interfere with optimization.

**Why post-processing works best:** Projecting the decoder columns to unit norm after each gradient step is a form of **projected gradient descent**. It is simple, exact (norms are exactly 1 after each step), and does not interfere with gradient computation. The optimizer computes and applies gradients freely, and then we project back to the constraint set. This is the standard approach in the SAE literature.

```python
optimizer.step()
with torch.no_grad():
    sae.W_dec.data = F.normalize(sae.W_dec.data, dim=0)
```

---

### Question 5.2

**(a)** [3 marks]

For a model with hidden dimension 2048, I would start with an **expansion factor of 8**, giving a dictionary size of $m = 2048 \times 8 = 16{,}384$ features.

Justification from scaling laws (Gao et al.):
- Expansion factors between 4 and 64 are commonly used. Starting at 8 is a practical middle ground that balances feature coverage against compute cost.
- With 1 billion training tokens, we have enough data to train a 16K-feature SAE without severe overfitting (Gao et al. show saturation at roughly $O(m)$ tokens, so 1B tokens comfortably supports $m = 16K$).
- After initial experiments at expansion factor 8, I would train additional SAEs at factors 4 and 16 to map out the Pareto frontier and determine if scaling up is worthwhile.

**(b)** [4 marks]

**Metric 1: Reconstruction MSE (or $R^2$).**
- *What it measures:* How well the SAE reconstructs the original activations.
- *How to compute:* On a held-out set of activation vectors, compute $\text{MSE} = \mathbb{E}[\|\mathbf{x} - \hat{\mathbf{x}}\|^2]$ and $R^2 = 1 - \text{MSE} / \text{Var}(\mathbf{x})$.
- *Good value:* $R^2 > 0.95$ indicates the SAE captures most activation variance. Below 0.9 suggests significant information loss.

**Metric 2: CE loss recovered.**
- *What it measures:* Functional fidelity — how much the model's behavior is preserved when real activations are replaced by SAE reconstructions.
- *How to compute:* Compute $\text{CE recovered} = (\mathcal{L}_{\text{zero}} - \mathcal{L}_{\text{SAE}}) / (\mathcal{L}_{\text{zero}} - \mathcal{L}_{\text{orig}})$ on an evaluation dataset.
- *Good value:* Above 0.90 is strong; above 0.95 is excellent. Below 0.85 suggests the SAE is losing information the model relies on.

**Metric 3: L0 (average number of active features per input).**
- *What it measures:* Sparsity of the representation.
- *How to compute:* Average $\|\mathbf{z}\|_0$ over the test set.
- *Good value:* Depends on dictionary size, but typically 10-100 active features out of 16K. The L0 should be much smaller than $m$ (strong sparsity) but large enough to reconstruct well.

**Metric 4: Feature utilization / dead feature fraction.**
- *What it measures:* Whether the SAE uses its full capacity.
- *How to compute:* For each feature, compute the fraction of test inputs where it activates. Count features that activate on 0% of inputs (dead) and features that activate on < 0.01% (nearly dead).
- *Good value:* Less than 10% dead features. A broad distribution of activation frequencies, with most features firing on 0.01%-10% of inputs.

---

## Section 6: Integration and Synthesis

### Question 6.1

**(a)** [2 marks]

PCA solves the problem of finding the low-dimensional linear subspace that best explains the variance of the data. Given zero-mean data, PCA finds orthogonal directions (eigenvectors of the covariance matrix) along which the data varies most, enabling dimensionality reduction with minimal information loss.

**Key limitations for neural networks:** PCA is restricted to linear projections and orthogonal features. Neural network representations are highly nonlinear — features interact, combine, and form hierarchies that cannot be captured by any linear projection. Moreover, the interesting structure in neural network activations may not align with the directions of maximum variance. A direction with small variance might still be critical for the model's computation.

**(b)** [2 marks]

Nonlinear autoencoders generalize PCA by replacing the linear encoder and decoder with neural networks, allowing them to learn curved, nonlinear manifolds rather than flat subspaces. This lets them capture complex structure: a nonlinear AE with a 2D bottleneck can learn to encode a spiral in 3D space, which PCA cannot do.

**New capabilities:** They can learn data-dependent, nonlinear features and can represent more complex low-dimensional structure.

**New problems:** (1) Optimization is non-convex — there is no guaranteed global optimum. (2) The learned features are opaque — unlike PCA's eigenvectors, the features of a nonlinear AE are not analytically interpretable. (3) They can memorize — without regularization, the autoencoder may learn the identity function rather than meaningful features.

**(c)** [2 marks]

Sparsity is important for two reasons that connect biology to interpretability:

**Natural data statistics (Olshausen & Field):** Natural signals (images, sounds, text) can be efficiently represented as sparse combinations of dictionary elements. Olshausen & Field showed that enforcing sparsity in a dictionary learning model applied to natural images produces receptive fields resembling those of V1 neurons — suggesting sparsity is a natural inductive bias for learning meaningful features of the world.

**Superposition hypothesis:** Neural networks represent many more features than they have neurons (superposition). If we want to decompose these superimposed representations into individual features, we need to assume the decomposition is sparse — each input activates only a few features from a large dictionary. Without sparsity, the decomposition is underdetermined (infinitely many ways to decompose a vector into an overcomplete dictionary).

**(d)** [2 marks]

Sparse autoencoders combine all three ideas: they use an overcomplete dictionary (many more features than input dimensions, like the representation space of a neural network) with a sparsity constraint (each input activates few features) to decompose neural network activations into interpretable components.

The key insight connecting dictionary learning to mechanistic interpretability: if the superposition hypothesis is correct, then neural network activations are sparse linear combinations of feature directions — exactly the structure that sparse coding / dictionary learning is designed to recover. An SAE trained on a layer's activations learns a dictionary whose elements correspond to the features the network has learned, decomposing polysemantic neuron activations into monosemantic feature activations. This transforms the interpretability problem from "what does this neuron do?" (confusing, because neurons are polysemantic) to "what does this feature do?" (clear, because SAE features are monosemantic by construction).

---

### Question 6.2

**(a)** [3 marks]

**Argument for the linear representation hypothesis:**

The core claim is that neural network features are represented as **directions** in activation space, and that the effect of a feature is a linear function of its activation strength (a feature's contribution to the activation is $\alpha_i \mathbf{d}_i$, where $\alpha_i$ is the strength and $\mathbf{d}_i$ is the direction).

**Evidence:**
1. **Probing classifiers.** Linear probes trained on neural network activations can often recover meaningful features (sentiment, part of speech, entity type) with high accuracy. If features were nonlinearly encoded, linear probes would fail.
2. **Activation addition / steering.** Adding a "feature direction" vector to activations (e.g., adding a "happiness" direction) predictably shifts model behavior in the corresponding semantic direction. This linearity of effect is consistent with features as directions.
3. **SAE success.** SAEs with linear decoders successfully decompose activations into interpretable features, achieving high reconstruction quality. If features were not well-approximated by directions, a linear decoder would fail.
4. **Toy models.** In Elhage et al.'s toy models with known ground-truth features, the features are indeed represented as directions, and their interference structure is consistent with the linear representation hypothesis.

**(b)** [2 marks]

**Argument against:**

The linear representation hypothesis may fail to capture **feature interactions and compositional structure**. Consider the concept "not hot" — this is not simply the negative of the "hot" direction. Or consider "bank" in the sense of a river bank vs. a financial bank — the meaning depends on context in a way that a single static direction cannot capture.

More formally, if features interact (e.g., feature A modulates the meaning of feature B), these interactions cannot be represented as independent linear contributions. **Nonlinear features** — features that are functions of multiple directions, or features whose direction depends on which other features are active — would require a richer representation than a linear dictionary. The residual stream likely contains such nonlinear structure, especially in the form of multi-token features and context-dependent representations.

**(c)** [2 marks]

**Proposed modification: Bilinear SAE with pairwise feature interactions.**

Instead of the standard linear decoder $\hat{\mathbf{x}} = \sum_i z_i \mathbf{d}_i + \mathbf{b}$, use a decoder that includes pairwise interaction terms:

$$\hat{\mathbf{x}} = \sum_i z_i \mathbf{d}_i + \sum_{i < j} z_i z_j \mathbf{d}_{ij} + \mathbf{b}$$

where $\mathbf{d}_{ij}$ are learned interaction directions. This allows the SAE to capture cases where the combined effect of two features is not the sum of their individual effects. For sparsity, most $z_i$ are zero, so the pairwise terms are also mostly zero (the number of nonzero interaction terms scales as $K^2$ where $K$ is the sparsity level, which is manageable).

**Motivation:** This directly addresses the limitation from (b) — compositional and context-dependent features can be represented as interactions between base features. The approach preserves the sparsity structure (most terms are zero) while adding expressiveness. The main cost is additional parameters ($O(m^2)$ interaction terms, though sparsity + pruning could keep this tractable).

---

**END OF SOLUTIONS**
