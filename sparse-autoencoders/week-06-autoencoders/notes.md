# Week 6: The Autoencoder

## Overview

Last week we showed that a linear autoencoder learns PCA — the best linear compression of data. This week we remove the "linear" constraint and unleash the full power of neural networks on the problem of representation learning.

The **autoencoder** is one of the simplest and most elegant ideas in deep learning: train a network to reconstruct its input, but force the data through a bottleneck. The bottleneck cannot transmit all the information in the input, so the network must learn which information is essential and which can be discarded. What survives the bottleneck is the *representation* — a compressed summary of the data's most important features.

Autoencoders are the backbone of this course. Everything that follows — regularized autoencoders, variational autoencoders, sparse autoencoders — is a variation on the architecture we build this week. Get the fundamentals right, and the rest flows naturally.

---

## 1. Autoencoder Architecture

### 1.1 The Big Picture

An autoencoder consists of two parts:

- **Encoder** $f\_\theta$: Maps the input to a lower-dimensional latent representation
- **Decoder** $g\_\phi$: Maps the latent representation back to the original input space

The full autoencoder computes: $\hat{\mathbf{x}} = g\_\phi(f\_\theta(\mathbf{x}))$

The training objective is to minimize the **reconstruction error**:

$$
\mathcal{L}(\theta, \phi) = \frac{1}{N}\sum_{i=1}^N \|\mathbf{x}_i - g_\phi(f_\theta(\mathbf{x}_i))\|^2
$$

That is it. No labels, no external supervision. The data is its own target. The only thing forcing the autoencoder to learn anything interesting is the **bottleneck** — the latent representation has fewer dimensions than the input.

### 1.2 Architecture Diagram

```
Input x ∈ ℝ^d          Latent z ∈ ℝ^k          Reconstruction x̂ ∈ ℝ^d
                         (k < d)
┌─────────┐        ┌──────────────┐        ┌─────────┐
│         │        │              │        │         │
│  784    │──────▶ │  Bottleneck  │──────▶ │  784    │
│  dims   │  Enc   │   k dims     │  Dec   │  dims   │
│         │ f_θ(x) │   z = f(x)   │ g_ϕ(z) │         │
│  Input  │        │              │        │  Output │
│  Image  │        │  Compressed  │        │  Image  │
│         │        │   Code       │        │ (recon) │
└─────────┘        └──────────────┘        └─────────┘

         ◀───────── Encoder ──────────▶◀────── Decoder ──────▶

Loss = ‖x - x̂‖²
```

### 1.3 Formal Definition

**Encoder:** $f\_\theta: \mathbb{R}^d \to \mathbb{R}^k$ is a neural network with parameters $\theta$. Typically:

$$
\mathbf{z} = f_\theta(\mathbf{x}) = \sigma(W_L \cdots \sigma(W_2 \sigma(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) \cdots + \mathbf{b}_L)
$$

where $\sigma$ is a nonlinear activation function (ReLU, etc.) and $L$ is the number of encoder layers.

**Decoder:** $g\_\phi: \mathbb{R}^k \to \mathbb{R}^d$ is a neural network with parameters $\phi$. Typically a mirror image of the encoder:

$$
\hat{\mathbf{x}} = g_\phi(\mathbf{z}) = \sigma(W_L' \cdots \sigma(W_2' \sigma(W_1' \mathbf{z} + \mathbf{b}_1') + \mathbf{b}_2') \cdots + \mathbf{b}_L')
$$

The final decoder layer often uses a sigmoid activation (if inputs are normalized to $[0,1]$) or no activation (if inputs are real-valued).

**Latent representation:** $\mathbf{z} = f\_\theta(\mathbf{x}) \in \mathbb{R}^k$ is the compressed code. When $k < d$, we call this an **undercomplete autoencoder**.

### 1.4 Why "Autoencoder"?

The name has two parts:
- **Auto:** The target output is the input itself — the network is self-supervised
- **Encoder:** The first half of the network encodes the input into a compact representation

The term dates to the 1980s (Rumelhart, Hinton, Williams, 1986), when these networks were studied as a way to discover useful representations via backpropagation.

---

## 2. Undercomplete Autoencoders

### 2.1 The Information Bottleneck

When $k < d$ (latent dimension smaller than input dimension), the encoder cannot simply pass all information through. It must *select* what to preserve and what to discard.

Think of it as a communication channel with limited bandwidth. If you can only transmit $k$ numbers to describe a 784-dimensional image, which $k$ numbers do you choose? The autoencoder answers this question by learning to encode the information that minimizes reconstruction error.

### 2.2 What the Bottleneck Learns

The bottleneck forces the network to discover the **most important factors of variation** in the data. For MNIST:
- With $k = 2$: The latent space might encode rough digit identity but lose fine details
- With $k = 10$: Enough room to encode digit class plus some style information
- With $k = 32$: Good reconstructions — the latent space captures digit identity, stroke width, tilt, etc.
- With $k = 128$: Near-perfect reconstruction — the latent space preserves essentially all relevant information

The optimal bottleneck size depends on the intrinsic dimensionality of the data manifold. If the data manifold is $k\_0$-dimensional, we need $k \geq k\_0$ for good reconstruction. If $k \gg k\_0$, the extra dimensions are wasted (or used to encode noise).

### 2.3 Reconstruction Loss

The standard loss function is **mean squared error (MSE)**:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 = \frac{1}{N}\sum_{i=1}^N \sum_{j=1}^d (x_{ij} - \hat{x}_{ij})^2
$$

For binary or normalized-to-$[0,1]$ inputs (like MNIST pixel values), **binary cross-entropy (BCE)** is often a better choice:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^d \left[x_{ij} \log \hat{x}_{ij} + (1 - x_{ij})\log(1 - \hat{x}_{ij})\right]
$$

BCE treats each pixel as a Bernoulli random variable and measures the log-likelihood. It gives sharper reconstructions than MSE for binary-ish data because it more heavily penalizes confident wrong predictions.

**When to use which:**
- MSE: real-valued data, or when simplicity is preferred
- BCE: binary or $[0,1]$-valued data (use sigmoid on the decoder output)

### 2.4 Capacity and Compression

The **compression ratio** is $d/k$ — how much we are squeezing the data. For MNIST with $k = 32$: the ratio is $784/32 \approx 25$. Each image is described by just 32 numbers instead of 784.

But the effective compression depends on the network capacity. A very wide encoder (e.g., 784 -> 4096 -> 32) has more capacity to find a good 32-dimensional code than a narrow one (784 -> 32 directly). More encoder capacity generally means better reconstruction at the same bottleneck size, up to the point where the encoder has learned the optimal encoding.

---

## 3. Relationship to PCA (Revisited)

### 3.1 The Linear Case: A Review

From Week 5, we proved that a linear autoencoder (no activation functions) with MSE loss converges to PCA. Specifically, the product $W\_d W\_e$ converges to the projection matrix $U\_k U\_k^\top$ onto the top-$k$ eigenvectors of the data covariance matrix.

Let us be more precise about what this means:

**Linear autoencoder:**
- Encoder: $\mathbf{z} = W\_e \mathbf{x}$ (no activation)
- Decoder: $\hat{\mathbf{x}} = W\_d \mathbf{z}$ (no activation)
- Loss: $\mathcal{L} = \frac{1}{N}\sum\_i \Vert x\_i - W\_d W\_e \mathbf{x}\_i\Vert ^2$

**At the global minimum:** $W\_d W\_e = U\_k U\_k^\top$, and the reconstruction error equals $\sum\_{j=k+1}^d \lambda\_j$ — the variance in the discarded dimensions.

### 3.2 Adding Nonlinearity: Beyond PCA

Now add nonlinear activations:

- Encoder: $\mathbf{z} = \sigma(W\_e \mathbf{x} + \mathbf{b}\_e)$
- Decoder: $\hat{\mathbf{x}} = \sigma(W\_d \mathbf{z} + \mathbf{b}\_d)$

This can no longer be reduced to PCA. The encoder can learn **nonlinear** transformations, mapping the data to a latent space that respects the curved geometry of the data manifold.

**Example:** Consider data lying on a circle in 2D. PCA would project onto a line (the diameter), losing half the information. A nonlinear autoencoder can learn to map each point on the circle to its angle $\theta \in [0, 2\pi)$ — a 1-dimensional representation that perfectly captures the circular structure.

More generally: PCA can only find linear subspaces, but data manifolds are typically curved. A nonlinear autoencoder can find the manifold itself, providing a much more compact and faithful representation.

### 3.3 The Spectrum of Autoencoders

We can think of autoencoders along a spectrum of complexity:

| Model | Encoder | Decoder | What it learns |
|-------|---------|---------|---------------|
| PCA | Linear | Linear | Top-$k$ eigenvectors (best linear subspace) |
| Shallow nonlinear AE | 1 nonlinear layer | 1 nonlinear layer | Mildly nonlinear manifold |
| Deep AE | Multiple layers | Multiple layers | Complex nonlinear manifold |
| Conv AE | Conv layers | Transposed conv | Manifold with spatial structure |

Each step up the spectrum adds expressive power but also adds the risk of learning degenerate or uninterpretable representations (a theme we will address in Weeks 7-8).

---

## 4. Training Autoencoders

### 4.1 Architecture Design

**Symmetric encoder-decoder:** The most common design mirrors the encoder in the decoder. If the encoder is `784 -> 256 -> 64 -> 32`, the decoder is `32 -> 64 -> 256 -> 784`. This is not strictly necessary — asymmetric designs work too — but symmetry is a natural default.

**Activation functions:**
- Hidden layers: ReLU (or variants: LeakyReLU, ELU)
- Final encoder layer: typically no constraint, or ReLU if non-negative codes are desired
- Final decoder layer: sigmoid (for $[0,1]$ data) or none (for real-valued data)

**Bottleneck size:** The key hyperparameter. Too small: poor reconstruction, information loss. Too large: the autoencoder can learn trivial mappings. Start with a bottleneck that is 5-10% of the input dimension and adjust based on reconstruction quality.

### 4.2 Tied Weights

A common constraint: require the decoder weights to be the transpose of the encoder weights.

If the encoder has weight matrix $W\_e$, the tied decoder uses $W\_d = W\_e^\top$.

**Advantages:**
- Halves the number of parameters
- Acts as a regularizer (constraining the decoder reduces overfitting)
- The encoder and decoder are forced to agree on the importance of each direction

**Disadvantages:**
- Reduces model expressivity
- The optimal encoder and decoder weights may not be transposes of each other

For deep autoencoders with multiple layers, tied weights means each decoder layer uses the transpose of the corresponding encoder layer (in reverse order).

**In PyTorch:**

```python
class TiedAutoencoder(nn.Module):
    def __init__(self, d_in, d_hidden, d_latent):
        super().__init__()
        self.encoder_w1 = nn.Parameter(torch.randn(d_hidden, d_in) * 0.01)
        self.encoder_w2 = nn.Parameter(torch.randn(d_latent, d_hidden) * 0.01)
        self.encoder_b1 = nn.Parameter(torch.zeros(d_hidden))
        self.encoder_b2 = nn.Parameter(torch.zeros(d_latent))
        self.decoder_b1 = nn.Parameter(torch.zeros(d_hidden))
        self.decoder_b2 = nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        h = F.relu(F.linear(x, self.encoder_w1, self.encoder_b1))
        return F.relu(F.linear(h, self.encoder_w2, self.encoder_b2))

    def decode(self, z):
        # Tied weights: decoder uses transposed encoder weights
        h = F.relu(F.linear(z, self.encoder_w2.t(), self.decoder_b1))
        return torch.sigmoid(F.linear(h, self.encoder_w1.t(), self.decoder_b2))

    def forward(self, x):
        return self.decode(self.encode(x))
```

### 4.3 Training Procedure

Training an autoencoder is no different from training any neural network:

1. **Data:** Load the dataset. Normalize pixel values to $[0, 1]$ (divide by 255). No labels needed.
2. **Model:** Define encoder and decoder architectures.
3. **Optimizer:** Adam with $\eta = 10^{-3}$ is the standard starting point.
4. **Loss:** MSE or BCE.
5. **Training loop:** Standard mini-batch gradient descent. Monitor reconstruction loss on both training and validation sets.

**Key differences from supervised training:**
- No labels! The input is both the input and the target.
- Accuracy is not a meaningful metric. Instead, track reconstruction loss and visualize reconstructions.
- Overfitting is less of a concern in practice (the bottleneck is itself a powerful regularizer), but it can still happen with very high-capacity decoders.

### 4.4 Measuring Success

**Quantitative:** Reconstruction loss on a held-out test set. Lower is better, but compare against baselines (PCA with the same $k$).

**Qualitative:** Visualize input-reconstruction pairs. Good reconstructions should capture the overall structure (digit identity, approximate shape) even if fine details are smoothed out.

**Representation quality:** This is harder to measure directly. We will use:
- t-SNE/UMAP visualization of the latent space
- Linear probe accuracy (train a linear classifier on latent codes)
- Interpolation quality (see Section 5)

---

## 5. What Do Autoencoders Learn?

### 5.1 Visualizing the Latent Space

One of the most informative things you can do with a trained autoencoder is visualize its latent space.

**For $k = 2$ (two-dimensional latent space):** We can directly plot $\mathbf{z} = f\_\theta(\mathbf{x})$ in 2D, coloring points by digit label. If the autoencoder has learned a good representation, we should see digit clusters.

**For $k > 2$:** Use t-SNE or UMAP to reduce the $k$-dimensional latent codes to 2D for visualization.

**What to look for:**
- Do same-class examples cluster together?
- Are similar classes nearby (3 and 8, 4 and 9)?
- Is the space well-utilized (points spread across the space, not crammed into a corner)?
- Are there smooth transitions between classes, or sharp boundaries?

### 5.2 Interpolation in Latent Space

A powerful test of representation quality: take two data points $\mathbf{x}\_A$ and $\mathbf{x}\_B$, encode them to $\mathbf{z}\_A$ and $\mathbf{z}\_B$, and decode points along the line between them:

$$
\hat{\mathbf{x}}_\alpha = g_\phi\left((1 - \alpha)\mathbf{z}_A + \alpha \mathbf{z}_B\right), \quad \alpha \in [0, 1]
$$

If the latent space is well-organized:
- The interpolation should produce smooth, realistic transitions
- Interpolating between a 3 and an 8 should pass through plausible intermediate shapes
- The intermediates should look like real digits, not noise or artifacts

**If the interpolation looks bad** — producing blurry, unrealistic, or discontinuous results — the latent space has "holes" or "gaps" where no training data lies. The decoder has never been asked to reconstruct from those latent codes, so it produces garbage.

This is a fundamental limitation of vanilla autoencoders: they only learn a decoder for regions of latent space where training data maps to. There is no guarantee that the latent space is smooth or continuous. This is one of the key motivations for the **variational autoencoder** (Week 8), which regularizes the latent space to be smooth.

### 5.3 Latent Space Structure (Or Lack Thereof)

A trained autoencoder's latent space typically has these properties:
- Training data maps to a complex, irregular region of the latent space
- Different classes may be entangled (interleaved, not cleanly separated)
- The latent space contains "dead zones" — regions far from any training example where the decoder produces nonsense
- The scale and distribution of latent codes can be arbitrary

This lack of structure is not a failure — the autoencoder was only asked to minimize reconstruction error, and it did that well. But it limits the usefulness of the latent space for generation, interpolation, and interpretation.

Improving latent space structure requires adding constraints beyond reconstruction error:
- **Regularization** (Week 7): Penalize complex or noisy latent codes
- **Probabilistic structure** (Week 8): Force the latent space to match a known distribution (e.g., Gaussian)
- **Sparsity** (Weeks 9-10): Force the latent code to use only a few active dimensions per example

### 5.4 What Each Latent Dimension Encodes

In an ideal world, each latent dimension would correspond to a single, interpretable factor of variation (stroke width, digit tilt, loop size, etc.). In practice, vanilla autoencoders rarely achieve this — the latent dimensions tend to encode entangled mixtures of factors.

To investigate what a latent dimension encodes, you can:
1. Take a latent code $\mathbf{z}$ for a test image
2. Vary one dimension $z\_j$ while holding the others fixed
3. Decode and observe how the reconstruction changes

This is called a **latent traversal**. In a well-structured latent space, each traversal changes one semantic attribute. In a poorly structured space, each traversal changes multiple attributes simultaneously.

---

## 6. Convolutional Autoencoders

### 6.1 Why Convolutions for Images?

Fully-connected autoencoders treat each pixel independently — the network must learn from scratch that neighboring pixels are related. For images, this is wasteful. Convolutional neural networks (CNNs) build in the prior knowledge that images have local spatial structure.

**Key CNN concepts** (brief introduction — we focus on what is needed for autoencoders):

**Convolutional layer:** Applies a set of small learned filters (e.g., 3x3 or 5x5) to local patches of the input. Each filter slides across the image, computing a dot product at each position. The output is a set of **feature maps** — one per filter.

$$
(\text{feature map})_{ij} = \sum_{m,n} w_{mn} \cdot \text{input}_{i+m, j+n} + b
$$

**Key properties:**
- **Translation equivariance:** The same filter is applied everywhere, so a feature detected in one location is detected in all locations
- **Local connectivity:** Each output depends only on a small local patch, not the entire image
- **Parameter sharing:** The same filter weights are used at every spatial position, drastically reducing the parameter count

A single 3x3 filter has 9 parameters, regardless of the image size. A fully-connected layer mapping a 28x28 image to 28x28 output would need $784^2 \approx 600{,}000$ parameters. That is the power of convolutions.

**Pooling (downsampling):** Max pooling or average pooling reduces the spatial dimensions (e.g., from 28x28 to 14x14), creating a coarser representation. In the encoder, pooling compresses spatial information.

**Transposed convolution (upsampling):** The inverse of convolution + pooling. In the decoder, transposed convolutions increase spatial dimensions (e.g., from 7x7 to 14x14 to 28x28), reconstructing the spatial structure.

### 6.2 Convolutional Autoencoder Architecture

```
Input: 1x28x28

Encoder:
  Conv2d(1, 16, 3, stride=2, padding=1)   →  16x14x14   (downsample)
  ReLU
  Conv2d(16, 32, 3, stride=2, padding=1)  →  32x7x7     (downsample)
  ReLU
  Flatten                                  →  32*7*7 = 1568
  Linear(1568, k)                          →  k          (bottleneck)

Decoder:
  Linear(k, 1568)                          →  1568
  Reshape                                  →  32x7x7
  ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1) → 16x14x14
  ReLU
  ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)  → 1x28x28
  Sigmoid
```

### 6.3 Advantages Over Fully-Connected Autoencoders

**Fewer parameters:** The convolutional encoder above has ~50K parameters vs. ~200K for a comparable fully-connected encoder. The parameter savings grow dramatically with image size.

**Better inductive bias:** Convolutions encode the prior that images have local structure and translation symmetry. The network does not need to learn these properties from data — they are built in.

**Spatially structured latent representation:** Before the final bottleneck linear layer, the encoder produces feature maps (e.g., 32x7x7) that preserve spatial layout. The decoder can use this spatial structure to reconstruct coherent images.

**Sharper reconstructions:** Convolutional autoencoders typically produce sharper, more detailed reconstructions than fully-connected ones, especially for larger images.

### 6.4 A Note on Architecture Design

Convolutional autoencoder design follows a simple pattern:

**Encoder:** Progressively increase channels while decreasing spatial dimensions.
$$
\text{channels: } 1 \to 16 \to 32 \to 64, \quad \text{spatial: } 28 \to 14 \to 7 \to 3
$$

**Decoder:** Mirror the encoder — decrease channels while increasing spatial dimensions.
$$
\text{channels: } 64 \to 32 \to 16 \to 1, \quad \text{spatial: } 3 \to 7 \to 14 \to 28
$$

The bottleneck can be a fully-connected layer (flattening the final feature maps) or the feature maps themselves can serve as the latent representation.

---

## 7. Overcomplete Autoencoders: A Warning

### 7.1 What If $k > d$?

What happens when the latent dimension is *larger* than the input dimension? With $k > d$, the autoencoder is **overcomplete** — it has more than enough capacity to represent the input.

In this case, the network can learn the **identity function**: $\hat{\mathbf{x}} = \mathbf{x}$ with zero reconstruction error. The encoder and decoder simply pass the data through unchanged (padded with zeros in the extra dimensions).

### 7.2 Why the Identity Function Is Useless

Zero reconstruction error sounds great — but the representation is completely useless:
- The "encoding" is just the original data (or a trivial transformation of it)
- No compression has occurred
- No structure has been learned
- The latent space is not organized in any meaningful way

The whole point of an autoencoder is to force the network to discover compact, meaningful representations. If the network can cheat by learning the identity, it will.

### 7.3 The Linear Case: A Proof

Let us prove this formally for the linear case. Suppose $k \geq d$ and the autoencoder is linear:

$$
\hat{\mathbf{x}} = W_d W_e \mathbf{x}
$$

where $W\_e \in \mathbb{R}^{k \times d}$ and $W\_d \in \mathbb{R}^{d \times k}$.

Since $k \geq d$, the product $W\_d W\_e$ is a $d \times d$ matrix that can have rank up to $d$. In particular, we can set $W\_e = \begin{pmatrix} I\_d \\ 0 \end{pmatrix}$ (identity padded with zeros) and $W\_d = \begin{pmatrix} I\_d & 0 \end{pmatrix}$ (identity with zero columns), giving:

$$
W_d W_e = I_d
$$

The reconstruction error is zero. The encoder simply copies the input to the first $d$ latent dimensions.

This is not just a theoretical concern — gradient descent on an overcomplete linear autoencoder will converge to (or near) such a solution.

### 7.4 The Nonlinear Case

For nonlinear overcomplete autoencoders, the identity function is not the only solution — but it is one of many trivially perfect solutions. The network might learn a bijection (one-to-one mapping) between the input space and a $d$-dimensional submanifold of the $k$-dimensional latent space, with zero reconstruction error but no useful compression.

### 7.5 The Motivation for Regularization

The overcomplete case is not inherently useless — it just requires *additional constraints* beyond reconstruction error. If we want an overcomplete autoencoder to learn a meaningful representation, we must add a regularization term to the loss:

$$
\mathcal{L} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|^2}_{\text{reconstruction}} + \lambda \cdot \underbrace{R(\mathbf{z})}_{\text{regularization}}
$$

Different choices of $R(\mathbf{z})$ give different autoencoder variants:

| Regularization | Variant | Week |
|---------------|---------|------|
| None | Undercomplete AE | This week |
| Noise on input | Denoising AE | 7 |
| Jacobian penalty | Contractive AE | 7 |
| KL divergence from prior | Variational AE | 8 |
| L1 penalty on $\mathbf{z}$ | Sparse AE | 9-10 |

This table is the roadmap for the next four weeks. Each regularization strategy addresses the overcomplete problem in a different way, and each teaches us something different about what makes a good representation.

The sparse autoencoder — our ultimate destination — uses an overcomplete latent space ($k \gg d$) with an L1 penalty that forces most latent dimensions to be zero for any given input. The result: a large "dictionary" of features, of which only a few are active at a time. This is the key to interpretable feature extraction in neural networks.

---

## 8. Practical Considerations

### 8.1 Choosing the Bottleneck Size

The bottleneck size $k$ is the most important hyperparameter. Here is a systematic approach:

1. **Start with PCA:** Compute PCA on your data and look at the explained variance curve. If 95% of variance is captured by 50 components, $k = 50$ is a reasonable starting point for the autoencoder.

2. **Reconstruction quality sweep:** Train autoencoders with $k \in \lbrace 2, 5, 10, 20, 50, 100\rbrace $ and plot reconstruction loss vs. $k$. There is usually a "knee" — below it, reconstruction degrades rapidly; above it, adding dimensions yields diminishing returns.

3. **Task-dependent:** If the autoencoder feeds into a downstream task (classification, generation), choose $k$ by cross-validation on the downstream metric.

### 8.2 Depth vs. Width

**Deeper encoders** can learn more complex nonlinear mappings but are harder to train. **Wider layers** increase capacity within each layer but at higher computational cost.

A reasonable default for MNIST: 2-3 hidden layers per side, with widths decreasing by factors of 2-4 (e.g., 784 -> 256 -> 64 -> $k$ -> 64 -> 256 -> 784).

For more complex data (CIFAR-10, face images), convolutional architectures with 4-6 encoder layers are typical.

### 8.3 Loss Function Choice

| Data type | Recommended loss | Decoder output activation |
|-----------|-----------------|--------------------------|
| Real-valued | MSE | None |
| Binary / $[0,1]$ | BCE | Sigmoid |
| Categorical | Cross-entropy | Softmax |

For MNIST (pixel values in $[0, 1]$): either MSE or BCE works. BCE often gives slightly sharper reconstructions.

### 8.4 Common Failure Modes

**Blurry reconstructions:** This is the most common issue, especially with MSE loss. MSE penalizes large errors heavily, so the decoder learns to output the *average* of possible reconstructions — which is blurry. Solutions: use BCE loss, increase bottleneck size, use convolutional architecture.

**Mode collapse:** The decoder ignores the latent code and outputs the same (average) reconstruction for all inputs. This happens when the decoder is too powerful relative to the encoder, or when the bottleneck is too small. Solution: reduce decoder capacity or increase $k$.

**Latent space collapse:** All inputs map to nearly the same latent code. This happens when the encoder is not expressive enough, or when the learning rate is too high early in training. Solution: increase encoder capacity, use warmup.

---

## 9. Autoencoders in the Context of This Course

Let us take a step back and see where we are in the course arc.

**Weeks 1-4** built the foundation: the mathematics and the machinery of neural networks.

**Week 5** introduced the key idea: representation learning. We showed that learning good representations is equivalent to finding the data manifold, and that PCA is the simplest way to do this.

**This week** (Week 6) introduced the autoencoder — a neural network that learns representations via reconstruction. We saw that:
- Linear autoencoders reduce to PCA
- Nonlinear autoencoders can capture manifold structure that PCA cannot
- The bottleneck forces compression, but does not guarantee useful structure
- Overcomplete autoencoders need regularization to avoid trivial solutions

**Next week** (Week 7) we add regularization to get better-structured latent spaces. **Week 8** introduces the VAE, which gives the latent space probabilistic structure. **Weeks 9-10** arrive at the sparse autoencoder — our ultimate destination — which uses sparsity to extract interpretable features.

Every week builds on the previous one. The autoencoder is the scaffold on which everything else is built.

---

## 10. Summary

1. **Autoencoders** learn representations by training a network to reconstruct its input through a bottleneck
2. **Undercomplete AEs** ($k < d$) force compression — the bottleneck must capture the most important factors of variation
3. **Linear AEs learn PCA** — established last week, reinforced this week as a special case
4. **Nonlinear AEs** learn nonlinear manifolds, generalizing PCA to curved structure
5. **Training** is straightforward: minimize reconstruction error with standard optimizers
6. **The latent space** of a vanilla AE is not guaranteed to be structured or continuous — motivating regularized variants
7. **Convolutional AEs** leverage spatial structure for better image reconstruction with fewer parameters
8. **Overcomplete AEs** ($k > d$) can learn the identity — motivating the regularization we study next week and the sparsity we study in Weeks 9-10

---

## References

- Rumelhart, Hinton, Williams, "Learning Internal Representations by Error Propagation" (1986)
- Hinton & Salakhutdinov, "Reducing the Dimensionality of Data with Neural Networks" (2006) — the deep autoencoder paper
- Baldi & Hornik, "Neural Networks and Principal Component Analysis: Learning from Examples Without Local Minima" (1989)
- Masci et al., "Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction" (2011)
- Goodfellow, Bengio, Courville, *Deep Learning*, Chapter 14 (2016)
