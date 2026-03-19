# Week 13: Advanced SAE Architectures & Open Problems

## Overview

This final week examines the frontier of sparse autoencoder research. We begin by confronting the concrete limitations of the vanilla L1-penalized SAE we built in Week 10, then study three architectural innovations — TopK, Gated, and JumpReLU SAEs — each of which addresses a specific failure mode. We discuss scaling laws, evaluation methodology, and the open problems that define the current research frontier. We close with a review of the entire course arc.

**Reading:**
- Gao et al., "Scaling and Evaluating Sparse Autoencoders" (OpenAI, 2024)
- Rajamanoharan et al., "Improving Dictionary Learning with Gated Sparse Autoencoders" (DeepMind, 2024)
- Rajamanoharan et al., "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders" (DeepMind, 2024)

---

## 1. Limitations of Vanilla SAEs

Recall the standard SAE from Week 10. Given an input activation $\mathbf{x} \in \mathbb{R}^n$, the encoder produces a hidden representation $\mathbf{z} \in \mathbb{R}^m$ (with $m \gg n$) and the decoder reconstructs $\hat{\mathbf{x}}$:

$$\mathbf{z} = \text{ReLU}(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)$$
$$\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_d$$

The training objective is:

$$\mathcal{L} = \| \mathbf{x} - \hat{\mathbf{x}} \|_2^2 + \lambda \| \mathbf{z} \|_1$$

This formulation has served us well, but it harbors three interrelated problems.

### 1.1 Shrinkage Bias

The L1 penalty $\lambda \| \mathbf{z} \|_1$ penalizes the *magnitude* of every active feature. This means gradient descent pushes active feature activations toward zero, not just inactive ones. If a feature genuinely has a large activation (it should be "strongly on"), the L1 penalty fights against this, biasing the magnitude downward.

Formally, consider the gradient contribution from the sparsity term for a single feature $z_i > 0$:

$$\frac{\partial}{\partial z_i} \lambda |z_i| = \lambda \cdot \text{sign}(z_i) = \lambda$$

This constant gradient toward zero is applied regardless of whether $z_i$ is large or small. The result: the SAE systematically underestimates the magnitude of active features. This is the same shrinkage bias known from LASSO regression (Week 9), now appearing in the autoencoder context.

**Practical consequence:** To compensate for suppressed feature magnitudes, the decoder must learn larger weights, distorting the geometry of the learned dictionary. Reconstruction quality degrades, especially for inputs where the "true" features are strongly active.

### 1.2 The $\lambda$ Tradeoff

The hyperparameter $\lambda$ controls a global tradeoff between reconstruction fidelity and sparsity. But this is a single knob controlling two qualitatively different objectives:

- Too large $\lambda$: features are too sparse, many are zeroed out, reconstruction suffers.
- Too small $\lambda$: features are not sparse enough, the representation is dense and hard to interpret.

Worse, the optimal $\lambda$ depends on the distribution of activations, which varies across layers and model architectures. There is no principled way to set $\lambda$ without extensive hyperparameter search.

**The deeper issue:** We want to control the number of active features (the L0 norm), but we are using the L1 norm as a differentiable proxy. The L1 norm conflates *which* features are active with *how much* they are active.

### 1.3 Dead Neurons

As we saw in Week 10, some features "die" during training — their pre-activation never exceeds zero, so they contribute nothing. With a large overcomplete dictionary ($m \gg n$), a significant fraction of features can die, wasting model capacity.

Dead neurons arise because:
1. Once a feature stops firing, it receives no gradient (ReLU has zero gradient for negative inputs).
2. The L1 penalty makes it harder for marginally active features to survive.
3. This creates a feedback loop: slight inactivity leads to permanent death.

Various heuristics exist (resampling dead neurons, auxiliary losses), but they are band-aids rather than solutions.

---

## 2. TopK Sparse Autoencoders

**Key idea:** Instead of using L1 to *encourage* sparsity, directly *enforce* it by keeping only the top $K$ activations.

**Reference:** Gao et al., "Scaling and Evaluating Sparse Autoencoders" (OpenAI, 2024).

### 2.1 Architecture

The encoder computes pre-activations as usual:

$$\mathbf{h} = \mathbf{W}_e \mathbf{x} + \mathbf{b}_e$$

Then, instead of applying ReLU + L1, we apply a TopK operation:

$$z_i = \begin{cases} h_i & \text{if } h_i \text{ is among the top } K \text{ values of } \mathbf{h} \\ 0 & \text{otherwise} \end{cases}$$

The decoder is unchanged: $\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_d$.

The loss function is simply reconstruction:

$$\mathcal{L} = \| \mathbf{x} - \hat{\mathbf{x}} \|_2^2$$

No sparsity penalty is needed — sparsity is baked into the architecture.

### 2.2 Advantages

1. **Exact sparsity control.** The L0 norm is exactly $K$ for every input. No tuning a continuous $\lambda$ — you choose $K$ directly.
2. **No shrinkage bias.** Active features pass through without penalty. Their magnitudes are not distorted.
3. **Simpler hyperparameter.** $K$ is an integer with an interpretable meaning ("how many features should be active?"), whereas $\lambda$ is a continuous weight with no direct interpretation.

### 2.3 The TopK Gradient: Straight-Through Estimator

The TopK operation is not differentiable — it involves a hard selection step. How do we backpropagate through it?

We use a **straight-through estimator (STE)**. In the forward pass, we apply the hard TopK mask. In the backward pass, we pass gradients through to all top-K selected elements as if the mask were the identity:

**Forward:**
$$z_i = h_i \cdot \mathbb{1}[h_i \in \text{top-K}(\mathbf{h})]$$

**Backward:**
$$\frac{\partial \mathcal{L}}{\partial h_i} = \frac{\partial \mathcal{L}}{\partial z_i} \cdot \mathbb{1}[h_i \in \text{top-K}(\mathbf{h})]$$

The gradient flows through unchanged for selected features and is zero for unselected features. This is the same principle as the straight-through estimator for binary quantization — we approximate the non-differentiable operation by ignoring its non-differentiability during the backward pass.

**Why this works:** The TopK mask is locally stable — small perturbations to $\mathbf{h}$ usually do not change which elements are in the top K. The gradient tells us how to adjust the magnitudes of the selected features to improve reconstruction, which is the right thing to optimize.

### 2.4 Choosing K

Gao et al. find that $K$ should scale with the dictionary size $m$. A useful rule of thumb from their experiments: $K \approx \sqrt{m}$ or a small fraction of $m$, adjusted so that the average L0 matches the sparsity level you want. In practice, one sweeps $K$ over a small range and selects based on the sparsity-reconstruction Pareto frontier.

### 2.5 Auxiliary Loss for Dead Features

TopK SAEs can still suffer from dead features. Gao et al. propose an auxiliary loss that encourages all features to be used. Let $\mathbf{h}^{(\text{dead})}$ denote the pre-activations of features that have not been in the top-K for many batches. The auxiliary loss reconstructs a residual using only dead features:

$$\mathcal{L}_{\text{aux}} = \| (\mathbf{x} - \hat{\mathbf{x}}) - \mathbf{W}_d^{(\text{dead})} \text{TopK}_{\text{aux}}(\mathbf{h}^{(\text{dead})}) \|_2^2$$

This gives dead features a gradient signal without interfering with the main reconstruction.

---

## 3. Gated Sparse Autoencoders

**Key idea:** Separate the decision of *which features fire* from the estimation of *how much they fire*.

**Reference:** Rajamanoharan et al., "Improving Dictionary Learning with Gated Sparse Autoencoders" (DeepMind, 2024).

### 3.1 Motivation

In a vanilla SAE, a single set of pre-activations $\mathbf{h} = \mathbf{W}_e \mathbf{x} + \mathbf{b}_e$ serves two roles:
1. **Selection:** Whether $h_i > 0$ determines if feature $i$ is active.
2. **Magnitude:** The value $h_i$ determines the strength of feature $i$.

The L1 penalty creates a tension between these roles: the penalty discourages large magnitudes, but the selection threshold (zero, for ReLU) is fixed. The result is that features are simultaneously pushed to be small (by L1) and need to be positive (to survive ReLU). This coupling is the root cause of shrinkage bias.

### 3.2 Architecture

The Gated SAE uses two parallel linear projections from the input:

**Gating path** (determines which features are active):
$$\boldsymbol{\pi}_{\text{gate}} = \mathbf{W}_{\text{gate}} \mathbf{x} + \mathbf{b}_{\text{gate}}$$
$$\mathbf{g} = \mathbb{1}[\boldsymbol{\pi}_{\text{gate}} > 0]$$

**Magnitude path** (determines feature strengths):
$$\boldsymbol{\pi}_{\text{mag}} = \mathbf{W}_{\text{mag}} \mathbf{x} + \mathbf{b}_{\text{mag}}$$
$$\tilde{\mathbf{z}} = \text{ReLU}(\boldsymbol{\pi}_{\text{mag}})$$

**Combined activation:**
$$\mathbf{z} = \mathbf{g} \odot \tilde{\mathbf{z}}$$

The decoder is standard: $\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_d$.

### 3.3 Training Objective

The loss has two components:

$$\mathcal{L} = \| \mathbf{x} - \hat{\mathbf{x}} \|_2^2 + \lambda \| \boldsymbol{\pi}_{\text{gate}} \|_1$$

Note that the L1 penalty is applied to the *gate pre-activations*, not the feature magnitudes. This is the key insight: sparsity pressure affects the gate (which features fire), while the magnitude path is free from shrinkage.

### 3.4 Why the Gate Separates Selection from Magnitude

With the gating architecture:
- The L1 penalty on $\boldsymbol{\pi}_{\text{gate}}$ encourages most gates to be off (most features inactive). This controls sparsity.
- For features where the gate is on ($g_i = 1$), the magnitude $\tilde{z}_i$ is determined entirely by the magnitude path, which has no L1 penalty. No shrinkage.

The gate path uses a Heaviside step function ($\mathbb{1}[\cdot > 0]$), which is non-differentiable. As with TopK, a straight-through estimator is used during backpropagation. In practice, Rajamanoharan et al. use the sigmoid function $\sigma(\boldsymbol{\pi}_{\text{gate}})$ as a smooth approximation during the backward pass.

### 3.5 Weight Sharing

In practice, $\mathbf{W}_{\text{gate}}$ and $\mathbf{W}_{\text{mag}}$ can share weights (same linear projection, different biases). This halves the parameter count of the encoder and works well empirically. The gating and magnitude computations then differ only in their bias terms:

$$\boldsymbol{\pi}_{\text{gate}} = \mathbf{W}_e \mathbf{x} + \mathbf{b}_{\text{gate}}, \qquad \boldsymbol{\pi}_{\text{mag}} = \mathbf{W}_e \mathbf{x} + \mathbf{b}_{\text{mag}}$$

This makes intuitive sense: the same projection detects features, but the threshold for "is this feature present?" (gate bias) may differ from the offset for "how strong is it?" (magnitude bias).

---

## 4. JumpReLU Sparse Autoencoders

**Key idea:** Use a ReLU-like activation with a *learned, per-feature threshold* instead of a fixed threshold at zero.

**Reference:** Rajamanoharan et al., "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders" (DeepMind, 2024).

### 4.1 The JumpReLU Activation

The JumpReLU function is defined as:

$$\text{JumpReLU}_\theta(x) = \begin{cases} x & \text{if } x > \theta \\ 0 & \text{if } x \leq \theta \end{cases}$$

where $\theta > 0$ is a threshold parameter. Unlike standard ReLU (where $\theta = 0$), the threshold is:
1. **Positive**, creating a "dead zone" between 0 and $\theta$ where the function is zero.
2. **Learnable**, so each feature $i$ has its own threshold $\theta_i$ that adapts during training.

The SAE architecture becomes:

$$\mathbf{h} = \mathbf{W}_e \mathbf{x} + \mathbf{b}_e$$
$$z_i = \text{JumpReLU}_{\theta_i}(h_i) = h_i \cdot \mathbb{1}[h_i > \theta_i]$$
$$\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_d$$

### 4.2 Why Learned Thresholds Help

Different features naturally have different baseline activation levels. A feature that detects "the token is a number" might have pre-activations clustered around 2.0, while a feature detecting "the token follows a period" might cluster around 0.5. A fixed threshold (zero for ReLU) treats both identically. Learned thresholds allow each feature to set its own decision boundary.

**Advantages:**
- **Adaptive sparsity per feature.** Rare features can have low thresholds (easy to activate); common features can have high thresholds (only fire when strongly indicated).
- **No shrinkage for active features.** When $h_i > \theta_i$, the output is $h_i$ — the full magnitude, undistorted.
- **No explicit sparsity penalty needed.** The thresholds themselves control sparsity. No $\lambda$ to tune (though in practice a light L0-targeting loss is often used).

### 4.3 Training with Non-Differentiable Thresholds

The JumpReLU has a discontinuity at $x = \theta$. The derivative $\frac{\partial z_i}{\partial \theta_i}$ does not exist in the classical sense — the function jumps from some positive value to zero.

Rajamanoharan et al. handle this by using a **straight-through estimator** for the indicator function. The key gradient approximations:

**Gradient w.r.t. the pre-activation $h_i$:**
$$\frac{\partial z_i}{\partial h_i} \approx \mathbb{1}[h_i > \theta_i]$$

This is just the ReLU gradient, shifted to the threshold $\theta_i$.

**Gradient w.r.t. the threshold $\theta_i$:**

This is trickier. The true derivative involves a Dirac delta at $h_i = \theta_i$. Rajamanoharan et al. use a smoothed approximation:

$$\frac{\partial z_i}{\partial \theta_i} \approx -h_i \cdot \sigma'(h_i - \theta_i) \cdot \frac{1}{\epsilon}$$

where $\sigma'$ is the derivative of the sigmoid function and $\epsilon$ controls the smoothing bandwidth. Alternatively, one can use the sparsity loss gradient: if the model is too sparse, thresholds should decrease; if too dense, thresholds should increase.

In practice, the threshold gradient is derived from an **L0-targeting loss**:

$$\mathcal{L}_{\text{sparsity}} = \left( \frac{1}{B} \sum_{b=1}^{B} \| \mathbf{z}^{(b)} \|_0 - K_{\text{target}} \right)^2$$

where $B$ is the batch size and $K_{\text{target}}$ is the desired average number of active features. The gradient of this loss w.r.t. $\theta_i$ is computed using the smoothed step function.

### 4.4 Comparison of the Three Architectures

| Property | L1 SAE | TopK SAE | Gated SAE | JumpReLU SAE |
|----------|--------|----------|-----------|--------------|
| Sparsity mechanism | L1 penalty | Hard top-K selection | L1 on gate | Learned thresholds |
| Shrinkage bias | Yes | No | No | No |
| Sparsity control | Indirect ($\lambda$) | Exact ($K$) | Indirect ($\lambda$) | Adaptive per feature |
| Per-feature adaptivity | No | No | Partial | Yes |
| Extra parameters | None | None | Gate biases | Thresholds ($m$ scalars) |
| Dead neuron risk | High | Moderate | Moderate | Low |
| Gradient estimation | Exact | STE for mask | STE for gate | STE for threshold |

All three advanced architectures achieve better reconstruction-sparsity tradeoffs than vanilla L1 SAEs. The choice between them depends on the application: TopK is simplest and gives exact L0 control; Gated SAEs provide a clean separation of concerns; JumpReLU offers the most adaptive per-feature behavior.

---

## 5. Scaling Laws for SAEs

Gao et al. (2024) systematically study how SAE quality scales with various dimensions.

### 5.1 Dimensions of Scaling

There are three axes along which we can scale SAEs:

1. **Dictionary size $m$:** More features in the SAE.
2. **Training data:** More activation vectors to train on.
3. **Model size:** Applying SAEs to larger neural networks.

### 5.2 Key Findings

**Larger dictionaries find more features.** As $m$ increases, the SAE discovers additional interpretable features. This is consistent with the superposition hypothesis: neural networks pack many more features into their activations than the dimension of those activations. Larger dictionaries can represent more of these superimposed features.

**Diminishing returns.** The relationship between $m$ and the number of interpretable features is sub-linear. Doubling the dictionary size does not double the number of useful features. Empirically, Gao et al. find that the loss improvement follows a power law:

$$\mathcal{L}(m) \approx \mathcal{L}_\infty + C \cdot m^{-\alpha}$$

where $\alpha$ is typically between 0.5 and 1.0, depending on the model and layer.

**More training data helps, but saturates.** For a fixed dictionary size $m$, there is a finite amount of structure to learn. Beyond a certain number of training tokens, the SAE converges and additional data provides no benefit. The saturation point scales with $m$ — larger dictionaries need more data.

**Larger models need larger SAEs.** When moving from a small model to a larger one, the SAE dictionary must grow to capture the additional features. Gao et al. find that the optimal dictionary size scales roughly linearly with the model's hidden dimension, suggesting a constant "expansion factor" $m / n$.

### 5.3 Practical Implications

- **Expansion factor.** A common starting point is $m = 4n$ to $m = 64n$, where $n$ is the model's hidden dimension. Start with smaller factors for initial experiments; scale up for thorough feature extraction.
- **Compute budget.** Training a large SAE is expensive but much cheaper than training the underlying model. Still, sweeping over dictionary sizes, sparsity levels, and architectures requires significant compute.
- **Layer selection.** SAE quality varies by layer. Middle layers of transformers tend to yield the most interpretable features, consistent with the idea that early layers encode local/positional features and late layers encode task-specific features.

---

## 6. Evaluation Methods for SAEs

Evaluation is one of the hardest problems in SAE research. How do we know an SAE is "good"?

### 6.1 Reconstruction Fidelity

The most basic metric: how well does the SAE reconstruct activations?

$$\text{MSE} = \mathbb{E}\left[ \| \mathbf{x} - \hat{\mathbf{x}} \|_2^2 \right]$$

Often normalized as the **fraction of variance explained**:

$$R^2 = 1 - \frac{\mathbb{E}[\| \mathbf{x} - \hat{\mathbf{x}} \|_2^2]}{\text{Var}(\mathbf{x})}$$

But reconstruction alone is insufficient — a dense, non-sparse autoencoder would score perfectly.

### 6.2 Downstream Loss (CE Loss Recovered)

A more meaningful metric: how much does the *model's behavior* change when we replace real activations with SAE reconstructions?

Given a language model $M$ and an SAE applied at layer $\ell$, we compute:
- $\mathcal{L}_{\text{orig}}$: the cross-entropy loss of the original model.
- $\mathcal{L}_{\text{SAE}}$: the cross-entropy loss when layer $\ell$'s activations are replaced by SAE reconstructions.
- $\mathcal{L}_{\text{zero}}$: the cross-entropy loss when layer $\ell$'s activations are zeroed out.

The **CE loss recovered** is:

$$\text{CE recovered} = \frac{\mathcal{L}_{\text{zero}} - \mathcal{L}_{\text{SAE}}}{\mathcal{L}_{\text{zero}} - \mathcal{L}_{\text{orig}}}$$

A value of 1.0 means the SAE reconstruction is as good as the original activations; 0.0 means the SAE reconstruction is no better than zeroing out the layer entirely.

This metric captures functional fidelity: does the SAE preserve the information the model actually uses?

### 6.3 Feature Interpretability

Are the features the SAE finds actually interpretable?

**Human evaluation:** Show annotators the top-activating examples for each feature and ask them to describe what the feature detects. Measure inter-annotator agreement and whether the descriptions are specific (good) or vague (bad).

**Automated interpretability scores:** Use a language model to generate descriptions of features (based on top-activating examples), then test whether those descriptions predict which new examples will activate the feature. This is the approach used in Bricken et al. (2023):

1. Show an LM the top activating examples for feature $i$ and ask it to generate a description $d_i$.
2. Show the LM new examples and, using only $d_i$, ask it to predict whether feature $i$ will activate.
3. The **interpretability score** is the correlation between the LM's predictions and actual activations.

### 6.4 Feature Absorption

A subtle failure mode: the SAE might *absorb* information about a concept into another feature rather than dedicating a separate feature to it. For example, a feature that detects "capitalized words" might absorb information about "proper nouns" — the SAE has no separate proper-noun feature, but the capitalized-word feature partially encodes it.

Detecting absorption requires comparing the features an SAE finds against a known set of ground-truth concepts and checking for concepts that are represented *distributedly* across SAE features rather than *monosemantically*.

### 6.5 Feature Density and Utilization

Not all features should fire with equal frequency, but extreme imbalance is a problem:
- **Dead features:** Never fire. Wasted capacity.
- **Ultra-rare features:** Fire on < 0.01% of inputs. Hard to evaluate, may be noise.
- **Ultra-common features:** Fire on > 50% of inputs. Likely not specific or interpretable.

A healthy SAE has a broad distribution of feature frequencies, with most features firing on 0.1% to 10% of inputs.

### 6.6 The Sparsity-Fidelity Pareto Frontier

No single metric suffices. The standard practice is to plot the **Pareto frontier** of sparsity (L0) vs. fidelity (CE loss recovered or MSE) for different SAE configurations. Architectures that push this frontier upward and to the left are better.

---

## 7. Open Problems

The field of SAE-based interpretability is young, and many fundamental questions remain open.

### 7.1 Faithfulness

**The core question:** Are the features SAEs find the "true" features of the network?

The superposition hypothesis says neural networks represent more features than they have neurons, and SAEs decompose these superimposed features. But this raises a philosophical issue: is there a unique decomposition? If the network represents features in superposition, there may be many valid decompositions — much like how a matrix can be factored in many ways.

**What would convince us?** At minimum:
- **Causal relevance:** Clamping a feature on/off should produce the predicted change in model behavior.
- **Completeness:** The SAE features should account for all of the model's behavior, not just some of it.
- **Uniqueness:** Different SAE training runs should converge on similar features (up to permutation).

Current evidence is encouraging for causal relevance but mixed for completeness and uniqueness.

### 7.2 Compositional Features

Real concepts are compositional: "a red car" involves the features "red" and "car" composed together. How do SAE features compose?

Current SAEs treat features as independent — the activation of feature $i$ is independent of feature $j$. But real neural network representations may have features that interact, modulate each other, or are only meaningful in combination.

**Research direction:** Developing SAEs that can capture pairwise or higher-order feature interactions, perhaps through bilinear layers or attention-like mechanisms in the SAE itself.

### 7.3 Circuit Analysis

Features are interesting, but the real goal is understanding *circuits*: how do features connect across layers to implement algorithms?

The vision: decompose a neural network into (1) features at each layer and (2) the computational graph connecting them. SAEs provide step (1); connecting features into circuits is step (2).

**Challenges:**
- The number of potential connections is quadratic in the number of features per layer.
- Not all connections are meaningful; identifying the important ones requires causal intervention.
- Attention heads complicate the picture: they move information between positions, creating a complex routing structure.

### 7.4 Universality

Do different models learn the same features? If GPT-2 and Llama-3 both have a "French language" feature, this suggests something fundamental about the structure of the task (language modeling) rather than the architecture.

Early evidence suggests some features are universal (e.g., basic syntactic features), while others are model-specific. A systematic study of feature universality across architectures, training data, and model sizes remains open.

### 7.5 Better Architectures

TopK, Gated, and JumpReLU SAEs all improve on the vanilla architecture, but they share fundamental assumptions:
- **Linear decoder:** Features are represented as directions, and the decoder is a linear combination.
- **Independence:** Features are treated as independent (no interactions in the SAE).
- **Single-layer:** The SAE has one encoder layer and one decoder layer.

Are these assumptions correct? Research directions include:
- **Multi-layer SAEs** that can capture hierarchical feature structure.
- **Transcoders** that map from one layer's activations to the next, learning the transformation between representations rather than reconstructing a single layer.
- **Attention-based SAEs** that can capture position-dependent features.
- **Conditional SAEs** where the dictionary depends on context.

### 7.6 Scaling to Frontier Models

Can we interpret the largest models (hundreds of billions of parameters)? The challenges are:

- **Compute:** Training SAEs on frontier models requires enormous compute, though still a fraction of the base model training cost.
- **Feature count:** Frontier models may have millions of features. Can humans or automated systems make sense of millions of features?
- **Emergent capabilities:** The most interesting behaviors of frontier models (reasoning, planning, in-context learning) may involve complex circuits that are hard to decompose into simple features.

Templeton et al. (Anthropic, 2024) made initial progress by applying SAEs to Claude 3 Sonnet, finding millions of interpretable features. But even this represents a partial picture of the model's capabilities.

---

## 8. Course Review: The Arc of the Course

Let us trace the intellectual thread from Week 1 to Week 13.

### 8.1 The Foundation (Weeks 1-4)

We started with the mathematical tools: **linear algebra** (vector spaces, eigendecomposition, SVD), **optimization** (gradient descent, convexity, L1 vs. L2 regularization), **probability** (distributions, KL divergence, MLE), and **neural networks** (backpropagation, SGD, Adam).

These are not prerequisites to be checked off — they are the language in which every subsequent idea is expressed.

### 8.2 Representation Learning (Weeks 5-6)

The central question: what representations do neural networks learn? We introduced the **manifold hypothesis** (data lives on low-dimensional manifolds) and **PCA** as the simplest dimensionality reduction. We showed that PCA is equivalent to a linear autoencoder (Weeks 5-6), motivating the nonlinear autoencoder as a generalization.

### 8.3 Regularized and Variational Autoencoders (Weeks 7-8)

Plain autoencoders can memorize. We introduced regularization strategies: **denoising autoencoders** (robustness to corruption), **contractive autoencoders** (penalizing sensitivity), and **variational autoencoders** (learning a generative model with principled regularization via the ELBO).

The VAE introduced a crucial idea: structured latent spaces, where the regularization is chosen to enforce a prior (standard Gaussian) on the latent distribution.

### 8.4 Sparsity (Weeks 9-10)

We studied **sparsity** as a structural prior: most signals are described by a few active components from a large dictionary. This connects to neuroscience (sparse coding in V1, Olshausen & Field), compressed sensing, and the statistics of natural data.

We built **sparse autoencoders**: overcomplete autoencoders with sparsity-inducing penalties. The key innovation: rather than compressing (undercomplete bottleneck), we expand (overcomplete) and enforce sparsity. This allows the dictionary to be large enough to represent many features while ensuring each input uses only a few.

### 8.5 Interpretability (Weeks 11-13)

The **superposition hypothesis** (Week 11) provided the motivation: neural networks represent more features than they have neurons. Polysemantic neurons confound interpretation. SAEs are a tool to decompose superimposed representations into monosemantic features.

We applied SAEs to **language models** (Week 12), using TransformerLens to extract activations and training SAEs to find interpretable features in GPT-2 and other models. We saw that SAEs can discover features corresponding to named entities, syntactic structures, and semantic concepts.

This week (Week 13), we studied the **frontier**: advanced architectures (TopK, Gated, JumpReLU) that address the limitations of vanilla SAEs, scaling laws, evaluation methods, and the open problems that will define the next phase of research.

### 8.6 The Big Picture

The through-line of this course:

$$\text{PCA} \to \text{Autoencoders} \to \text{Sparse Coding} \to \text{Sparse Autoencoders} \to \text{Interpretability}$$

Each step adds something:
- PCA finds principal directions. Limited to linear, orthogonal features.
- Autoencoders learn nonlinear representations. But the features may not be individually meaningful.
- Sparse coding enforces that each input uses few dictionary elements. But optimization is expensive (iterative inference).
- Sparse autoencoders amortize inference into a feedforward encoder. Scalable to large datasets.
- Applied to neural network activations, SAEs become a tool for mechanistic interpretability.

The deepest lesson: understanding neural networks requires understanding their representations, and understanding their representations requires the right mathematical tools — from eigendecomposition to variational inference to sparsity.

---

## 9. Looking Forward

This course has given you the foundations to engage with one of the most exciting areas of AI research. If you continue in this direction, here are some starting points:

1. **Replicate a result.** Take one of the papers from Weeks 11-13 and reproduce a key experiment. You will learn more from debugging your replication than from reading ten more papers.

2. **Build tools.** The interpretability community needs better tooling — visualization, automated evaluation, circuit tracing. Building tools is a high-leverage contribution.

3. **Think critically.** Not every SAE feature is meaningful. Not every causal intervention is clean. The field needs researchers who can distinguish signal from noise.

4. **Read the literature.** The field moves fast. Follow Anthropic's research blog, the Alignment Forum, and arXiv mechanistic interpretability papers.

Good luck on the exam.

---

## Summary of Key Concepts

| Concept | Key Idea |
|---------|----------|
| Shrinkage bias | L1 penalty distorts magnitudes of active features |
| TopK SAE | Hard sparsity via keeping top K activations |
| Gated SAE | Separate gating (which features) from magnitude (how much) |
| JumpReLU SAE | Learned per-feature thresholds |
| Straight-through estimator | Pass gradients through non-differentiable operations |
| Scaling laws | Larger SAEs find more features; diminishing returns |
| CE loss recovered | Functional fidelity metric for SAEs |
| Feature absorption | SAE misses a concept by distributing it across other features |
| Faithfulness | Do SAE features reflect the network's true computation? |
| Universality | Do different models learn the same features? |

---

## References

1. Gao, L., et al. "Scaling and Evaluating Sparse Autoencoders." OpenAI (2024).
2. Rajamanoharan, S., et al. "Improving Dictionary Learning with Gated Sparse Autoencoders." DeepMind (2024).
3. Rajamanoharan, S., et al. "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders." DeepMind (2024).
4. Templeton, A., et al. "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Anthropic (2024).
5. Bricken, T., et al. "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." Anthropic (2023).
6. Elhage, N., et al. "Toy Models of Superposition." Anthropic (2022).
7. Olshausen, B.A. and Field, D.J. "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images." Nature (1996).
