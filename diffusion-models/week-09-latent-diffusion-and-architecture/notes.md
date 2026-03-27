# Week 9: Latent Diffusion and Architecture

> *"The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise."*
> -- Edsger Dijkstra

---

## Overview

In Weeks 5-8, we developed the theory and practice of diffusion models operating directly on images -- pixel-space diffusion. The mathematics is elegant, the results are compelling, but there is a blunt practical problem: **high-resolution images are enormous**.

A 512x512 RGB image has 786,432 dimensions. Running 20-50 denoising steps through a U-Net on this space is computationally brutal. Training is worse: hundreds of GPU-days on clusters of A100s.

Rombach et al. (2022) proposed a simple, powerful solution: **compress the image first, then do diffusion in the compressed space**. A variational autoencoder (VAE) compresses a 512x512x3 image to a 64x64x4 latent representation -- a 48x reduction in dimensionality. Diffusion in this latent space is dramatically cheaper, and the VAE's decoder converts the result back to pixel space.

This is the architecture behind Stable Diffusion, the model that made high-quality image generation accessible on consumer hardware. It is a two-stage system: a perceptual compression stage (the VAE) and a generative stage (the diffusion model in latent space). Each stage can be trained and optimized independently.

This week, we will study both stages in depth. We will examine the autoencoder design, the U-Net architecture that performs denoising in latent space, and the emerging alternative: the Diffusion Transformer (DiT).

### Prerequisites
- Week 5: DDPM training objective and forward/reverse process
- Week 7: Probability flow ODE, continuous-time formulation
- Week 8: Sampling methods (DDIM, DPM-Solver)
- Familiarity with convolutional neural networks and attention mechanisms

---

## 1. The Pixel-Space Problem

### 1.1 Computational Cost

Consider training a DDPM on 256x256 RGB images. The denoising network must:
- Accept a 256x256x3 = 196,608-dimensional input (plus a time embedding)
- Output a 196,608-dimensional prediction ($\hat{\epsilon}$ or $\hat{x}\_0$)
- Process this at sufficient depth and width to model complex image statistics

A typical U-Net for this resolution has 200-500 million parameters. Training requires computing gradients through this network for millions of (image, noise level) pairs. On 8 A100 GPUs, this takes days to weeks.

At 512x512, the cost roughly quadruples (the spatial dimensions double in each direction, and the U-Net must be deeper to handle the larger receptive field). At 1024x1024, it becomes prohibitive for most research groups.

### 1.2 The Redundancy in Pixel Space

Most of the information in a natural image is low-frequency perceptual content. High-frequency pixel-level details (exact shade of each pixel, imperceptible textures) carry relatively little semantic information. This is why JPEG works: it discards high-frequency components with minimal perceptual impact.

A diffusion model operating in pixel space must learn to generate *all* of these details, including the perceptually irrelevant ones. This is wasteful. The model spends capacity on details that humans cannot distinguish.

### 1.3 The Latent Diffusion Insight

Rombach et al. (2022) observed that image generation can be decomposed into two largely independent problems:

1. **Perceptual compression**: map images to a lower-dimensional latent space that preserves perceptual quality
2. **Generative modeling**: learn the distribution of these latent representations

If the latent space is well-designed -- capturing the important structure of images while discarding perceptually irrelevant details -- then the generative modeling problem becomes much easier.

---

## 2. The Autoencoder: Perceptual Compression

### 2.1 Architecture

The autoencoder has two components:

**Encoder** $\mathcal{E}$: Maps an image $x \in \mathbb{R}^{H \times W \times 3}$ to a latent $z \in \mathbb{R}^{h \times w \times c}$, where typically $h = H/f$, $w = W/f$, and $f$ is the downsampling factor.

**Decoder** $\mathcal{D}$: Maps a latent $z$ back to an image $\hat{x} = \mathcal{D}(z) \in \mathbb{R}^{H \times W \times 3}$.

For Stable Diffusion: $f = 8$, $c = 4$. A 512x512x3 image becomes a 64x64x4 latent. The dimensionality reduction is $512 \times 512 \times 3 = 786{,}432$ down to $64 \times 64 \times 4 = 16{,}384$ -- a factor of 48.

The encoder and decoder are convolutional networks with residual blocks and (optionally) attention layers at lower resolutions.

### 2.2 Training Objectives

A plain autoencoder trained with MSE loss $\Vert x - \mathcal{D}(\mathcal{E}(x))\Vert ^2$ produces blurry reconstructions -- the L2 loss averages over possible reconstructions. Latent diffusion models use a combination of losses:

**Reconstruction loss (L1 or L2):**
$$
\mathcal{L}_{\text{rec}} = \|x - \mathcal{D}(\mathcal{E}(x))\|_1
$$

L1 is preferred over L2 because it produces sharper results (it penalizes all errors equally, rather than disproportionately penalizing large errors).

**Perceptual loss (LPIPS):**
$$
\mathcal{L}_{\text{perc}} = \sum_l \|F_l(x) - F_l(\hat{x})\|_2^2
$$

where $F\_l$ extracts features from layer $l$ of a pretrained VGG network. This loss measures perceptual similarity rather than pixel-wise similarity. Two images can have large pixel-wise differences but small perceptual differences (e.g., a slight spatial shift).

**Adversarial loss:**
$$
\mathcal{L}_{\text{adv}} = -\log D_\psi(\mathcal{D}(\mathcal{E}(x)))
$$

A patch-based discriminator $D\_\psi$ distinguishes real images from reconstructions. This encourages the decoder to produce sharp, realistic textures rather than blurry averages.

**KL regularization:**
$$
\mathcal{L}_{\text{KL}} = D_{\text{KL}}(q(z|x) \| \mathcal{N}(0, I))
$$

This is the crucial term for latent diffusion. Without it, the latent space can have an arbitrary, pathological distribution. The KL term encourages the latent distribution to be close to a standard Gaussian, making it smooth and well-behaved -- exactly what a diffusion model needs.

The encoder outputs mean $\mu$ and log-variance $\log \sigma^2$ for each latent dimension, and $z$ is sampled via the reparameterization trick: $z = \mu + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$. This makes the autoencoder a VAE.

The total loss:
$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}
$$

### 2.3 The KL Weight: A Critical Trade-Off

The KL weight $\lambda\_{\text{KL}}$ controls a fundamental trade-off:

- **Large $\lambda\_{\text{KL}}$**: The latent space is smooth and Gaussian-like, making the diffusion model's job easy. But reconstruction quality suffers -- the encoder is forced to discard information to match the prior.

- **Small $\lambda\_{\text{KL}}$**: Reconstruction is excellent, but the latent space can have sharp, irregular structure that is hard for the diffusion model to learn.

In practice, $\lambda\_{\text{KL}}$ is set very small (around $10^{-6}$). The latent space is only mildly regularized -- just enough to prevent degenerate behavior (extreme values, dead dimensions) without sacrificing reconstruction quality. This is sometimes called a "KL-regularized autoencoder" rather than a true VAE, because the KL term is so weak that the latents are nearly deterministic.

### 2.4 Downsampling Factor

The choice of $f$ (downsampling factor) is another design decision with real consequences:

| Factor $f$ | Latent size (512 input) | Compression | Reconstruction | Diffusion cost |
|---|---|---|---|---|
| 1 | 512x512x3 | 1x | Perfect | Very high |
| 4 | 128x128x3 | 16x | Excellent | Moderate |
| 8 | 64x64x4 | 48x | Good | Low |
| 16 | 32x32x8 | 24x | Acceptable | Very low |
| 32 | 16x16x16 | 12x | Lossy | Minimal |

Stable Diffusion uses $f = 8$, which strikes a good balance. At $f = 4$, reconstruction is better but diffusion is more expensive. At $f = 16$, some fine detail is lost and the reconstructions start to look subtly wrong (smoothed textures, lost fine lines).

---

## 3. Two-Stage Training

### 3.1 Stage 1: Train the Autoencoder

The autoencoder is trained on the full image dataset with the combined loss from Section 2.2. This is a standard deep learning training pipeline -- no diffusion involved yet.

The autoencoder should achieve:
- High perceptual quality: $\mathcal{D}(\mathcal{E}(x)) \approx x$ visually
- Smooth latent space: $z = \mathcal{E}(x)$ has approximately Gaussian statistics
- Compression: $z$ is much smaller than $x$

### 3.2 Stage 2: Train the Diffusion Model in Latent Space

Once the autoencoder is trained and frozen, we encode the entire training set into latents: $z\_n = \mathcal{E}(x\_n)$. Then we train a diffusion model (DDPM, score-based, etc.) on these latent representations.

The training objective is identical to the standard DDPM objective, but applied to latents:

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{z_0, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(z_t, t)\|^2 \right]
$$

where $z\_t = \sqrt{\bar{\alpha}\_t}\, z\_0 + \sqrt{1 - \bar{\alpha}\_t}\, \epsilon$ and $z\_0 = \mathcal{E}(x)$.

### 3.3 Sampling

To generate a new image:
1. Sample $z\_T \sim \mathcal{N}(0, I)$ in the latent space
2. Run the reverse diffusion process (DDIM, DPM-Solver, etc.) to get $z\_0$
3. Decode: $x = \mathcal{D}(z\_0)$

Step 2 is now much cheaper because $z$ is 48x smaller than $x$.

### 3.4 Advantages of Two-Stage Training

**Modularity.** The autoencoder and diffusion model are trained independently. You can swap the diffusion model (e.g., replace DDPM with a flow matching model) without retraining the autoencoder. You can upgrade the autoencoder without retraining the diffusion model (if the latent space statistics are similar).

**Efficiency.** The diffusion model trains on small latent representations. This dramatically reduces memory, compute, and training time. Stable Diffusion was trained on 256 A100 GPUs -- expensive, but feasible. The same model in pixel space would require an order of magnitude more.

**Flexibility.** Conditioning (text, class labels, images) can be injected into the diffusion model via cross-attention without modifying the autoencoder.

---

## 4. The U-Net: Workhorse of Diffusion

### 4.1 Origins

The U-Net was originally designed by Ronneberger et al. (2015) for biomedical image segmentation. Its characteristic structure -- a contracting encoder path, an expanding decoder path, and skip connections between them -- turned out to be remarkably well-suited for denoising.

### 4.2 Architecture Overview

The U-Net for diffusion models follows this structure:

```
Input z_t (64x64x4) + time embedding
    |
    v
[Conv 64x64x320] ----skip----> [Conv 64x64x320]
    |                                   ^
    v                                   |
[Down 32x32x640] ----skip----> [Up 32x32x640]
    |                                   ^
    v                                   |
[Down 16x16x1280] ---skip----> [Up 16x16x1280]
    |                                   ^
    v                                   |
           [Middle 8x8x1280]
```

Each resolution level has one or more **ResNet blocks** and (at some resolutions) **self-attention blocks**. The skip connections concatenate encoder features with decoder features at each resolution.

### 4.3 ResNet Blocks

Each ResNet block has the structure:

$$
\text{ResBlock}(h, t_{\text{emb}}) = h + \text{Conv}(\text{SiLU}(\text{GN}(\text{Conv}(\text{SiLU}(\text{GN}(h))) + \text{MLP}(t_{\text{emb}}))))
$$

Unpacked:
1. **Group Normalization (GN)**: Normalizes across groups of channels. Unlike batch normalization, GN works well with small batch sizes (common in high-resolution generation). Typically 32 groups.
2. **SiLU activation**: $\text{SiLU}(x) = x \cdot \sigma(x)$, also called Swish. Smoother than ReLU, consistently better for diffusion models.
3. **Convolution**: 3x3 convolutions with appropriate padding.
4. **Time embedding addition**: The time embedding (a vector) is projected by an MLP and added to the intermediate feature map (broadcast across spatial dimensions).
5. **Residual connection**: The input $h$ is added to the output (with a 1x1 conv if dimensions differ).

The time embedding addition is crucial: it tells each block *how noisy the input is*, allowing the network to adapt its behavior across noise levels.

### 4.4 Time Embedding

The timestep $t$ is embedded via sinusoidal positional encoding (borrowed from the Transformer):

$$
\text{PE}(t, 2i) = \sin(t / 10000^{2i/d})
$$
$$
\text{PE}(t, 2i+1) = \cos(t / 10000^{2i/d})
$$

This produces a vector in $\mathbb{R}^d$ (typically $d = 320$). The sinusoidal encoding has the desirable property that nearby timesteps have similar embeddings, while distant timesteps are easily distinguished.

The positional encoding is then passed through a small MLP (two linear layers with SiLU activation) to produce the final time embedding $t\_{\text{emb}} \in \mathbb{R}^{d\_{\text{model}}}$, which is injected into every ResNet block.

### 4.5 Self-Attention

At lower resolutions (typically 32x32, 16x16, and 8x8), the U-Net includes self-attention layers. The feature map $h \in \mathbb{R}^{h \times w \times c}$ is reshaped to a sequence of $hw$ tokens of dimension $c$, and standard multi-head self-attention is applied:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

where $Q = W\_Q h$, $K = W\_K h$, $V = W\_V h$.

Self-attention is computationally expensive ($O(n^2)$ in the number of spatial tokens $n = hw$), which is why it is only used at lower resolutions. At 64x64, there are 4096 tokens, making attention costly. At 16x16, there are only 256 tokens -- quite manageable.

Self-attention allows the network to model long-range spatial dependencies. Without it, the network's receptive field is limited by the depth of the convolutional stack. With it, every spatial position can attend to every other position, enabling global coherence (e.g., symmetry in faces, consistent perspective in scenes).

### 4.6 Downsampling and Upsampling

**Downsampling**: Strided convolutions (stride 2) or average pooling, reducing spatial dimensions by 2x.

**Upsampling**: Nearest-neighbor interpolation followed by a convolution, or transposed convolution. Nearest-neighbor + conv is preferred because transposed convolutions can produce checkerboard artifacts.

### 4.7 Skip Connections

The skip connections are the defining feature of the U-Net. They concatenate the encoder features with the decoder features at matching resolutions:

$$
h_{\text{dec}} = \text{Conv}([h_{\text{enc}}; h_{\text{up}}])
$$

where $[\cdot;\cdot]$ denotes channel-wise concatenation.

Why are skip connections so important for diffusion? Consider the denoising task: at low noise levels, the model should make small, precise adjustments to the input. This requires access to the fine-grained features from the encoder. Without skip connections, these features are lost by the bottleneck. With skip connections, the decoder can directly access multi-scale features from the encoder and make precision corrections at each resolution.

---

## 5. The Full Stable Diffusion Architecture

### 5.1 Three Components

Stable Diffusion consists of:

1. **VAE** (autoencoder): $\mathcal{E}: \mathbb{R}^{512 \times 512 \times 3} \to \mathbb{R}^{64 \times 64 \times 4}$ and $\mathcal{D}: \mathbb{R}^{64 \times 64 \times 4} \to \mathbb{R}^{512 \times 512 \times 3}$
2. **U-Net** (denoising network): $\epsilon\_\theta: \mathbb{R}^{64 \times 64 \times 4} \times \mathbb{R} \times \mathbb{R}^{L \times d} \to \mathbb{R}^{64 \times 64 \times 4}$, where $L \times d$ is the text conditioning
3. **Text encoder**: CLIP (or OpenCLIP) maps text prompts to a sequence of embeddings $c \in \mathbb{R}^{L \times d}$

The text conditioning enters the U-Net via cross-attention (which we will study in detail in Week 10). For now, the important point is that the U-Net takes three inputs: the noisy latent $z\_t$, the timestep $t$, and the text embedding $c$.

### 5.2 Model Sizes

| Component | Parameters | Purpose |
|-----------|-----------|---------|
| VAE encoder | ~34M | Image $\to$ latent |
| VAE decoder | ~49M | Latent $\to$ image |
| U-Net | ~860M | Denoising in latent space |
| CLIP text encoder | ~123M | Text $\to$ embeddings |
| **Total** | **~1.07B** | |

The U-Net dominates the parameter count and the computational cost. Every denoising step is a forward pass through this 860M-parameter network.

### 5.3 Stable Diffusion Versions

| Version | VAE | U-Net | Text Encoder | Key Change |
|---------|-----|-------|-------------|------------|
| SD 1.x | KL-f8 | 860M | CLIP ViT-L/14 | Original |
| SD 2.x | KL-f8 | 865M | OpenCLIP ViT-H/14 | Better text encoder |
| SDXL | KL-f8 | ~2.6B | CLIP-G + CLIP-L | Larger U-Net, dual text encoders |
| SD 3 | KL-f8 | DiT ~2B | T5 + CLIP | Transformer replaces U-Net |

The trend is clear: larger models, better text encoders, and a shift from U-Nets to transformers.

---

## 6. Diffusion Transformers (DiT)

### 6.1 Motivation

The U-Net architecture was inherited from image segmentation. It works well, but it has several limitations:
- The architecture is complex, with many design choices (number of ResNet blocks per level, which resolutions get attention, channel multipliers)
- Scaling is awkward -- you can scale width or depth, but the interaction between convolutional and attention layers complicates analysis
- Modern large-scale training infrastructure (Megatron, FSDP) is better optimized for homogeneous architectures like transformers

Peebles and Xie (2023) asked: can we replace the U-Net with a standard vision transformer?

### 6.2 Architecture

DiT treats the latent representation $z \in \mathbb{R}^{h \times w \times c}$ as a sequence of patches. Each patch of size $p \times p \times c$ is flattened and linearly projected to a token of dimension $d$:

$$
\text{tokens} = \text{PatchEmbed}(z) \in \mathbb{R}^{(hw/p^2) \times d}
$$

For $z \in \mathbb{R}^{32 \times 32 \times 4}$ and patch size $p = 2$: there are $32 \times 32 / 4 = 256$ tokens.

The tokens are processed by a stack of transformer blocks, each containing:
1. Layer normalization
2. Multi-head self-attention
3. Layer normalization
4. MLP (feedforward network)

with residual connections. This is the standard ViT architecture.

### 6.3 Conditioning via Adaptive Layer Norm (adaLN)

How does the timestep $t$ (and class label $y$) enter the transformer? DiT uses **adaptive layer normalization (adaLN)**, inspired by style transfer:

Instead of using fixed layer norm parameters $\gamma, \beta$, DiT computes them from the conditioning:

$$
\gamma, \beta, \alpha = \text{MLP}(t_{\text{emb}} + y_{\text{emb}})
$$

$$
\text{adaLN}(h) = \alpha \cdot (\gamma \cdot \text{LayerNorm}(h) + \beta)
$$

where $\alpha$ is a per-element scale applied to the residual connection. This allows each transformer block to modulate its behavior based on the noise level and class label.

The adaLN-Zero variant initializes $\alpha = 0$, so the transformer initially behaves as the identity function. This stabilizes training at initialization.

### 6.4 Scaling Properties

A major advantage of DiT: scaling behavior is clean and predictable. Peebles and Xie tested DiT at sizes from DiT-S (33M parameters) to DiT-XL (675M parameters) and found:

- FID decreases log-linearly with compute (Gflops)
- Larger models are strictly better at equal training compute
- The scaling follows the same laws observed for language models

This predictability is valuable: you can estimate the quality of a larger model before training it, enabling efficient allocation of compute budgets.

### 6.5 DiT vs. U-Net: Trade-offs

| Aspect | U-Net | DiT |
|--------|-------|-----|
| Inductive bias | Strong (multi-scale, local convolutions) | Weak (learns everything from data) |
| Architecture complexity | High (many hyperparameters) | Low (standard transformer) |
| Scaling | Awkward (width/depth interaction) | Clean (just add layers/heads) |
| Training efficiency | Good for small-medium models | Better for very large models |
| Inference speed | Fast (convolutions are efficient) | Slower per parameter (attention is $O(n^2)$) |
| Low-data regime | Better (inductive bias helps) | Worse (needs more data) |

The trend in the field is toward DiT-like architectures for large-scale models (SD3, Flux, Sora) while U-Nets remain practical for smaller models and research experiments.

---

## 7. Putting It All Together: A Concrete Example

### 7.1 Generating an Image with Stable Diffusion

Let us trace the full pipeline for generating a 512x512 image from the prompt "a cat sitting on a windowsill, golden hour":

**Step 1: Text encoding.**
The CLIP text encoder tokenizes the prompt and produces a sequence of embeddings $c \in \mathbb{R}^{77 \times 768}$ (77 tokens, 768 dimensions per token).

**Step 2: Initial noise.**
Sample $z\_T \sim \mathcal{N}(0, I)$ where $z\_T \in \mathbb{R}^{64 \times 64 \times 4}$.

**Step 3: Iterative denoising (e.g., 20 steps of DPM-Solver++).**
At each step $t$:
- The U-Net takes $(z\_t, t, c)$ and predicts $\hat{\epsilon}$ (or $\hat{x}\_0$)
- The sampler (DPM-Solver++) computes $z\_{t-1}$

This requires 20 forward passes through the 860M-parameter U-Net.

**Step 4: Decode.**
The VAE decoder maps $z\_0 \in \mathbb{R}^{64 \times 64 \times 4}$ to $x\_0 \in \mathbb{R}^{512 \times 512 \times 3}$. One forward pass through the 49M-parameter decoder.

**Step 5: Post-processing.**
Clip to $[0, 1]$ and convert to uint8. Done.

Total neural network evaluations: 1 (text encoder) + 20 (U-Net) + 1 (VAE decoder) = 22. On a modern GPU, this takes 2-5 seconds.

### 7.2 Latent Space Arithmetic

Because DDIM ($\eta = 0$) is deterministic, the mapping from latent noise $z\_T$ to image $x\_0$ is a fixed function. This enables latent space operations:

**Interpolation.** Given two noise vectors $z\_T^{(a)}$ and $z\_T^{(b)}$ that produce images $a$ and $b$, the interpolation $z\_T(\alpha) = \sqrt{\alpha}\, z\_T^{(a)} + \sqrt{1-\alpha}\, z\_T^{(b)}$ (spherical interpolation) produces a smooth blend between $a$ and $b$.

**Inversion.** Given an image $x$, run the DDIM forward process: $z\_0 = \mathcal{E}(x)$, then integrate the probability flow ODE *forward* from $t = 0$ to $t = T$ to find $z\_T$. This $z\_T$ will reconstruct $x$ when passed through the reverse process.

**Editing.** Invert an image to $z\_T$, modify the text prompt, and run the reverse process with the new prompt. The result is an edited version of the original image.

---

## 8. Practical Considerations

### 8.1 VAE Quality Matters

The VAE sets a ceiling on the final image quality. If the VAE cannot reconstruct fine details (small text, intricate patterns, subtle textures), then no amount of diffusion model improvement will recover them. This is why VAE quality has been a focus of recent work: SDXL's VAE was improved specifically to handle text rendering better.

### 8.2 Latent Space Distribution

The diffusion model assumes $z\_0 \sim p\_{\text{data}}$ has roughly unit variance. If the autoencoder produces latents with very different statistics (e.g., mean far from zero, or variance much larger than 1), the diffusion process will not work well. In practice, the latents are scaled by a constant factor to bring their variance close to 1. Stable Diffusion uses a scaling factor of 0.18215.

### 8.3 The Reconstruction-Generation Gap

There is a subtle gap between reconstruction quality (how well $\mathcal{D}(\mathcal{E}(x))$ matches $x$) and generation quality (how good $\mathcal{D}(z\_0)$ looks for a diffusion-generated $z\_0$). The diffusion model may produce latents that are slightly out-of-distribution for the decoder, leading to artifacts. This gap is usually small but can be visible in challenging cases (fine text, highly detailed textures).

---

## Summary

1. **Pixel-space diffusion** is expensive because images are high-dimensional and most pixel-level detail is perceptually irrelevant. A 512x512 image has ~786K dimensions.

2. **Latent diffusion** compresses images via a VAE (encoder-decoder) to a low-dimensional latent space (e.g., 64x64x4 = ~16K dimensions), then runs diffusion in that space. This is a 48x reduction in dimensionality.

3. **The autoencoder** is trained with a combination of reconstruction, perceptual, adversarial, and KL-regularization losses. The KL weight is kept small to preserve reconstruction quality while mildly regularizing the latent space.

4. **Two-stage training** (autoencoder, then diffusion model on frozen latents) provides modularity and efficiency. Each stage can be optimized independently.

5. **The U-Net** is the standard denoising architecture, featuring an encoder-decoder structure with skip connections, ResNet blocks, self-attention at lower resolutions, and time conditioning via sinusoidal embeddings injected into each block.

6. **Diffusion Transformers (DiT)** replace the U-Net with a vision transformer. They use adaptive layer normalization for conditioning and exhibit clean scaling properties. DiT is the architecture behind recent large-scale models.

7. **Stable Diffusion** combines a VAE, a U-Net (or DiT), and a text encoder (CLIP), with the text conditioning entering via cross-attention (Week 10).

---

## Key Equations

| Concept | Equation |
|---------|----------|
| VAE encoding | $z = \mathcal{E}(x), \quad z \in \mathbb{R}^{h \times w \times c}$ |
| VAE decoding | $\hat{x} = \mathcal{D}(z), \quad \hat{x} \in \mathbb{R}^{H \times W \times 3}$ |
| KL regularization | $\mathcal{L}\_{\text{KL}} = D\_{\text{KL}}(q(z|x) \Vert  \mathcal{N}(0,I))$ |
| Latent diffusion training | $\mathcal{L} = \mathbb{E}\_{z\_0, \epsilon, t}[\Vert \epsilon - \epsilon\_\theta(z\_t, t)\Vert ^2]$ |
| Latent forward process | $z\_t = \sqrt{\bar{\alpha}\_t}\, z\_0 + \sqrt{1-\bar{\alpha}\_t}\, \epsilon$ |
| Time embedding | $\text{PE}(t, 2i) = \sin(t / 10000^{2i/d})$ |
| adaLN (DiT) | $\text{adaLN}(h) = \alpha \cdot (\gamma \cdot \text{LN}(h) + \beta)$, $(\gamma,\beta,\alpha)=\text{MLP}(t\_{\text{emb}})$ |

---

## Suggested Reading

- **Rombach et al.** (2022), "High-Resolution Image Synthesis with Latent Diffusion Models" -- the foundational latent diffusion paper. Read Sections 3-4 for the autoencoder and latent diffusion framework.
- **Ronneberger, Fischer, and Brox** (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation" -- the original U-Net paper. Short and clear.
- **Peebles and Xie** (2023), "Scalable Diffusion Models with Transformers" -- the DiT paper. Focus on Section 3 for the architecture and Section 4 for scaling experiments.
- **Dhariwal and Nichol** (2021), "Diffusion Models Beat GANs on Image Synthesis" -- Section 4 describes the U-Net architecture improvements for diffusion models in detail.
- **Esser et al.** (2024), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" -- the Stable Diffusion 3 paper, showing the DiT architecture at scale.
