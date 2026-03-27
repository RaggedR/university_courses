---
title: "Week 9: Latent Diffusion and Architecture -- Homework"
---

# Week 9: Latent Diffusion and Architecture -- Homework

**Estimated time:** 14-18 hours
**Prerequisites:** DDPM training (Week 5), sampling methods (Week 8), PyTorch, convolutional networks

---

## Problem 1: The Autoencoder as a Bottleneck (Theory)

### Part (a): Information-Theoretic Bound

A VAE encoder maps $x \in \mathbb{R}^{H \times W \times 3}$ to $z \in \mathbb{R}^{h \times w \times c}$ where $h = H/f$, $w = W/f$.

1. For a 256x256x3 image and downsampling factor $f = 8$ with $c = 4$ latent channels, compute the dimensionality of the input and the latent. What is the compression ratio?

2. Argue that the autoencoder cannot represent all possible 256x256 images faithfully. What must the encoder learn to discard? Why is this actually desirable for generation?

3. The KL regularization term $D\_{\text{KL}}(q(z|x) \Vert \mathcal{N}(0,I))$ penalizes the encoder for deviating from a standard Gaussian. Write out the KL divergence for a diagonal Gaussian encoder $q(z|x) = \mathcal{N}(\mu(x), \text{diag}(\sigma^2(x)))$ in closed form. (Use the formula for KL between two multivariate Gaussians.)

### Part (b): Reconstruction Quality vs. Latent Smoothness

Consider two autoencoders trained on the same dataset:
- **AE-A**: Trained with $\lambda\_{\text{KL}} = 10^{-2}$ (strong regularization)
- **AE-B**: Trained with $\lambda\_{\text{KL}} = 10^{-6}$ (weak regularization)

1. Which autoencoder will have better reconstruction quality? Why?
2. Which autoencoder will produce a latent space more suitable for diffusion? Why?
3. If you sample $z \sim \mathcal{N}(0, I)$ and decode with each autoencoder, which is more likely to produce a coherent image? Explain.

### Part (c): Downsampling Factor Trade-offs

A team is designing a latent diffusion model for 1024x1024 images. They are deciding between $f = 8$ (latent 128x128x4) and $f = 16$ (latent 64x64x8).

1. Compute the computational cost of one U-Net forward pass for each option, assuming cost scales as $O(h^2 \cdot w^2 \cdot c)$ (due to attention at the lowest resolution). Which is cheaper and by how much?

2. What aspects of image quality might suffer at $f = 16$? Give two specific examples.

3. Propose a way to get the computational benefits of $f = 16$ while mitigating the quality loss. (There is no single right answer -- think creatively.)

---

## Problem 2: Build a Convolutional VAE (Implementation)

### Part (a): Encoder-Decoder Architecture

Implement a convolutional VAE with the following specification:

```python
class Encoder(nn.Module):
    """
    Input: (B, 3, 64, 64)
    Output: mu (B, 4, 8, 8), log_var (B, 4, 8, 8)

    Architecture:
    - Conv2d(3, 64, 3, stride=2, padding=1)   -> (B, 64, 32, 32)
    - ResBlock(64)
    - Conv2d(64, 128, 3, stride=2, padding=1)  -> (B, 128, 16, 16)
    - ResBlock(128)
    - Conv2d(128, 256, 3, stride=2, padding=1) -> (B, 256, 8, 8)
    - ResBlock(256)
    - Conv2d(256, 8, 1)                         -> (B, 8, 8, 8)
    Split into mu (B, 4, 8, 8) and log_var (B, 4, 8, 8)
    """
    pass

class Decoder(nn.Module):
    """
    Input: (B, 4, 8, 8)
    Output: (B, 3, 64, 64)

    Architecture: mirror of encoder with transposed convolutions or
    nearest-neighbor upsampling + conv.
    """
    pass

class VAE(nn.Module):
    """
    Combines encoder and decoder with reparameterization trick.
    """
    pass
```

Implement the `ResBlock` as: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv + skip connection.

### Part (b): Training

Train the VAE on a dataset of 64x64 images (CelebA, CIFAR-10 resized, or LSUN bedrooms resized). Use the following loss:

$$
\mathcal{L} = \Vert x - \hat{x}\Vert _1 + \lambda_{\text{KL}} \cdot D_{\text{KL}}(q(z|x) \Vert \mathcal{N}(0, I))
$$

Train with three different $\lambda\_{\text{KL}}$ values: $10^{-4}$, $10^{-6}$, $10^{-8}$.

For each, after training, display:
1. 16 input images and their reconstructions side by side
2. The mean and variance of the latent codes across 1000 images (are they close to $\mathcal{N}(0,I)$?)

### Part (c): Latent Space Smoothness

For the best-performing VAE:
1. Take two images $x\_a$ and $x\_b$, encode them to $z\_a = \mathcal{E}(x\_a)$ and $z\_b = \mathcal{E}(x\_b)$.
2. Interpolate: $z\_\alpha = (1-\alpha) z\_a + \alpha z\_b$ for $\alpha \in \lbrace 0, 0.1, 0.2, \ldots, 1.0\rbrace$.
3. Decode each $z\_\alpha$ and display the sequence. Does the interpolation look smooth?
4. Repeat with a plain autoencoder (no KL term). Is the interpolation smoother or rougher?

---

## Problem 3: The U-Net with Time Conditioning (Implementation)

### Part (a): Sinusoidal Time Embedding

Implement the sinusoidal time embedding:

```python
def sinusoidal_embedding(t, dim):
    """
    Args:
        t: (B,) tensor of timesteps
        dim: embedding dimension

    Returns:
        (B, dim) tensor of sinusoidal embeddings
    """
    # Your implementation here
    pass
```

Test: embed timesteps $t = 0, 100, 500, 900, 999$ into dimension 128. Plot the embedding vectors as heatmaps. Verify that nearby timesteps have similar embeddings (compute cosine similarity between embeddings of $t$ and $t+1$, $t+10$, $t+100$).

### Part (b): ResNet Block with Time Conditioning

Implement a ResNet block that incorporates the time embedding:

```python
class TimeConditionedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        # GroupNorm -> SiLU -> Conv -> time_proj -> GroupNorm -> SiLU -> Conv
        # with skip connection (1x1 conv if in_channels != out_channels)
        pass

    def forward(self, x, t_emb):
        """
        Args:
            x: (B, C, H, W) feature map
            t_emb: (B, time_dim) time embedding
        """
        pass
```

The time embedding should be projected via a linear layer and added to the feature map after the first convolution (broadcast across spatial dimensions).

### Part (c): Simplified U-Net

Build a simplified U-Net for denoising 32x32 single-channel images (e.g., MNIST):

```python
class SimpleUNet(nn.Module):
    """
    Input: (B, 1, 32, 32) noisy image + timestep t
    Output: (B, 1, 32, 32) predicted noise

    Architecture:
    - Time embedding: sinusoidal(t, 128) -> MLP -> 256-dim
    - Down: 32x32x64 -> 16x16x128 -> 8x8x256
    - Middle: 8x8x256 with self-attention
    - Up: 8x8x256 -> 16x16x128 -> 32x32x64
    - Out: Conv to 1 channel
    - Skip connections from each encoder level to corresponding decoder level
    """
    pass
```

For the self-attention in the middle block, reshape the 8x8x256 feature map to 64 tokens of dimension 256 and apply multi-head attention with 4 heads.

### Part (d): Train as a Denoiser

Train the U-Net as a DDPM denoiser on MNIST:

$$
\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\left[\Vert \epsilon - \epsilon_\theta(x_t, t)\Vert ^2\right]
$$

where $x\_t = \sqrt{\bar{\alpha}\_t}\, x\_0 + \sqrt{1-\bar{\alpha}\_t}\, \epsilon$.

Train for 50-100 epochs. Then generate samples using DDIM with 50 steps. Display a grid of 64 generated digits. Are they recognizable?

---

## Problem 4: Pixel-Space vs. Latent Diffusion (Implementation)

This is the central experiment of the week: directly compare pixel-space and latent-space diffusion.

### Part (a): Pixel-Space Diffusion

Train a DDPM on 32x32 images (CIFAR-10 or CelebA-32) in pixel space. Use the U-Net from Problem 3, adapted for 3-channel input/output.

Train for a fixed compute budget (e.g., 50 epochs or 6 hours, whichever comes first). Record:
1. Training loss curve
2. Sample quality at epochs 10, 25, 50 (generate 64 samples each time)
3. Total training time

### Part (b): Latent Diffusion

1. Train a VAE (from Problem 2, adapted for 32x32 input) with $f = 4$, producing 8x8x4 latents.
2. Freeze the VAE and encode the entire training set.
3. Train a smaller U-Net on the 8x8x4 latents, using the same total compute budget.

Record the same metrics as Part (a).

### Part (c): Comparison

Create a comparison table:

| Metric | Pixel-Space | Latent-Space |
|--------|-------------|-------------|
| Input dimensions | | |
| U-Net parameters | | |
| Training time (seconds/epoch) | | |
| Sample quality (FID or visual) at epoch 10 | | |
| Sample quality at epoch 50 | | |
| Best generated samples (grid) | | |

Answer:
1. At equal compute, which approach produces better samples? Why?
2. What is the main disadvantage of latent diffusion at this small scale (32x32)?
3. At what resolution would you expect latent diffusion to become clearly superior? Justify your answer.

---

## Problem 5: DiT Block (Implementation)

### Part (a): Adaptive Layer Norm

Implement the adaLN-Zero block used in DiT:

```python
class AdaLNBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        # Self-attention with adaLN
        # MLP with adaLN
        # Conditioning MLP that produces (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        pass

    def forward(self, x, c):
        """
        Args:
            x: (B, N, D) token sequence
            c: (B, D) conditioning vector (time + class embedding)
        Returns:
            (B, N, D) updated tokens
        """
        pass
```

The conditioning MLP should output 6 vectors (2 per sub-layer: scale $\gamma$, shift $\beta$, and gate $\alpha$ for the attention sub-layer and the MLP sub-layer). Initialize the output layer with zeros so that $\alpha = 0$ at initialization (the adaLN-Zero trick).

### Part (b): Mini-DiT

Build a minimal DiT for 16x16 single-channel images:

```python
class MiniDiT(nn.Module):
    """
    Input: (B, 1, 16, 16) + timestep t + class label y
    Output: (B, 1, 16, 16) predicted noise

    Architecture:
    - Patch embedding: 2x2 patches -> 64 tokens of dim 256
    - Positional embedding: learnable (64, 256)
    - 6 AdaLN blocks (dim=256, 4 heads)
    - Final layer: adaLN -> Linear -> unpatchify
    """
    pass
```

### Part (c): Train and Compare

Train MiniDiT on MNIST with class conditioning ($y \in \lbrace 0, \ldots, 9\rbrace$, embedded via a learnable embedding table).

Compare to the U-Net from Problem 3 (also with class conditioning, added to the time embedding):
1. Parameter count of each model
2. Training loss curves (same number of epochs)
3. Sample quality (generate 10 samples per class)

At this small scale, is DiT competitive with the U-Net?

---

## Problem 6: Analyzing a Pretrained VAE (Implementation)

If you have access to the Stable Diffusion VAE (available via `diffusers` library), use it. Otherwise, use your trained VAE from Problem 2.

### Part (a): Reconstruction Quality

Load 20 diverse images (faces, landscapes, text, fine patterns). For each:
1. Encode to latent: $z = \mathcal{E}(x)$
2. Decode: $\hat{x} = \mathcal{D}(z)$
3. Display $x$ and $\hat{x}$ side by side
4. Compute PSNR and SSIM between $x$ and $\hat{x}$

Which types of images are reconstructed best? Which are worst? Does fine text survive the encode-decode cycle?

### Part (b): Latent Statistics

Encode 1000 images and analyze the latent statistics:
1. Compute the mean and standard deviation of each latent channel. Are they close to $\mathcal{N}(0, 1)$?
2. Plot the histogram of latent values for one channel. Is it approximately Gaussian?
3. Compute the correlation matrix between latent channels at one spatial position. Are the channels independent?
4. Why does Stable Diffusion multiply latents by 0.18215 before feeding them to the diffusion model?

### Part (c): Latent Arithmetic

Using the pretrained VAE:
1. Encode two images $x\_a$ ("smiling face") and $x\_b$ ("neutral face") to $z\_a, z\_b$.
2. Compute $z\_c = z\_b + (z\_a - z\_b)$. Decode $z\_c$. Does it add a "smile" to a neutral face?
3. Take a latent $z$ and add Gaussian noise at different scales: $z + \sigma \epsilon$ for $\sigma \in \lbrace 0.1, 0.5, 1.0, 2.0\rbrace$. Decode each. At what noise level does the decoded image become unrecognizable?

This last experiment helps build intuition for what happens during the diffusion process in latent space.

---

## Submission Checklist

- [ ] Problem 1: Information-theoretic analysis, KL derivation, trade-off discussion
- [ ] Problem 2: VAE implementation, training with different $\lambda\_{\text{KL}}$, interpolation experiment
- [ ] Problem 3: U-Net with time conditioning, sinusoidal embedding, MNIST generation
- [ ] Problem 4: Pixel-space vs. latent-space comparison table and analysis
- [ ] Problem 5: DiT block implementation, MiniDiT, comparison with U-Net
- [ ] Problem 6: Pretrained VAE analysis -- reconstruction, latent statistics, arithmetic

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs. Include all generated images and plots.
