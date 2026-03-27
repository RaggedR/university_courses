---
title: "Week 13: Advanced Topics and Open Problems"
---

# Week 13: Advanced Topics and Open Problems

> *"The only way to discover the limits of the possible is to go beyond them into the impossible."*
> -- Arthur C. Clarke

---

## Overview

We have arrived at the final week. Over twelve weeks, we built diffusion models from their mathematical foundations (probability, stochastic calculus, score matching) through the practical machinery (DDPM, latent diffusion, guidance, flow matching, distillation) that drives modern generative AI. This week, we survey the frontiers: where diffusion models are being extended beyond images, what theoretical questions remain open, and where the field is heading.

This is not a week of derivations. It is a week of *questions* -- many without definitive answers. The goal is to give you a map of the territory so you can find problems worth working on.

### Prerequisites
- All previous weeks. This is a survey that draws on everything we have covered.

---

## 1. Video Diffusion

### 1.1 The Challenge

Video generation is the natural next step after image generation, and arguably the most commercially significant application of diffusion models. It is also dramatically harder.

A 5-second, 24fps, 1024x1024 video has $5 \times 24 \times 1024 \times 1024 \times 3 \approx 377$ million pixels. Even in latent space (128x128x4 spatial, 120 frames), the tensor has $\sim$8 million elements. The computational and memory requirements are enormous.

But the real challenge is not scale -- it is **temporal consistency**. An image generator can produce each image independently. A video generator must produce frames that are consistent across time: objects must persist, physics must be plausible, camera motion must be smooth. A single inconsistent frame breaks the illusion.

### 1.2 Architectural Approaches

**3D U-Net / Temporal Attention**: The most direct approach extends the image U-Net to 3D by adding temporal convolutions and temporal attention layers. Spatial layers process each frame independently (or nearly so), while temporal layers model cross-frame dependencies. Ho et al. (2022) showed this works for short, low-resolution video.

**Factored attention**: Full 3D attention over space and time is $O(T^2 H^2 W^2)$ -- prohibitively expensive. Practical systems factorize attention:
- Spatial attention within each frame: $O(T \cdot (HW)^2)$
- Temporal attention across frames at each spatial position: $O(HW \cdot T^2)$

This reduces cost from cubic to quadratic in each dimension but may miss long-range spatiotemporal dependencies.

**Autoregressive + diffusion hybrids**: Generate video in chunks. A diffusion model generates a keyframe or short clip; an autoregressive model extends it temporally. This is believed to be the approach used by systems like Sora, Runway Gen-3, and Kling, though implementation details are not fully public.

### 1.3 The State of the Art (Early 2025)

- **Sora** (OpenAI, 2024): Generates up to 60 seconds of video at 1080p. Uses a DiT architecture operating on spacetime patches in a compressed latent space. Demonstrated impressive physical understanding but remains unreleased for general use.
- **Runway Gen-3** (Runway, 2024): Commercially available video generation. 10-second clips at 1080p.
- **Kling** (Kuaishou, 2024): Open-weight video model. Competitive quality at lower resolution.
- **CogVideoX** (Tsinghua/Zhipu, 2024): Open-source video diffusion with 3D causal attention.
- **Wan** (Alibaba, 2025): Open-weight, available in 1.3B and 14B parameter versions.

The quality of video generation has improved dramatically since 2023, but temporal consistency, physics simulation, and long-duration coherence remain significant challenges.

---

## 2. 3D Generation

### 2.1 DreamFusion and Score Distillation

We introduced score distillation sampling (SDS) in Week 12. DreamFusion (Poole et al., 2023) applied SDS to generate 3D scenes from text prompts by optimizing a NeRF (neural radiance field) using a pre-trained 2D diffusion model as a loss function.

The pipeline:
1. Initialize a random NeRF
2. Render from a random viewpoint $c$
3. Add noise to the rendering
4. Compute the SDS gradient using the diffusion model
5. Update the NeRF parameters
6. Repeat from step 2 with a new viewpoint

The result is a 3D scene that looks good from all viewpoints -- the diffusion model's understanding of 2D images is "lifted" into 3D.

### 2.2 Multi-View Diffusion

A more direct approach: train diffusion models that generate multiple views of the same object simultaneously.

**Zero-1-to-3** (Liu et al., 2023): Given a single image, generate novel views at specified camera angles. The diffusion model is conditioned on the input image and the relative camera transformation.

**Multi-view diffusion** (Shi et al., 2024): Generate $N$ consistent views simultaneously using cross-attention between views. The views share information, so the model learns to produce 3D-consistent outputs.

**3D-native diffusion**: Instead of operating on 2D images and lifting to 3D, operate directly on 3D representations (point clouds, triplanes, or 3D Gaussians). Models like Point-E (OpenAI), Shap-E (OpenAI), and Instant3D generate 3D assets directly.

### 2.3 Gaussian Splatting Meets Diffusion

3D Gaussian splatting (Kerbl et al., 2023) has emerged as a fast, high-quality 3D representation. Recent work combines it with diffusion:

- **DreamGaussian** (Tang et al., 2024): Optimizes a set of 3D Gaussians using SDS, much faster than NeRF-based DreamFusion.
- **GaussianDreamer** (Yi et al., 2024): Generates 3D Gaussian splats directly using a diffusion model over Gaussian parameters.

The convergence of diffusion models with fast 3D representations is an active and rapidly evolving area.

---

## 3. Discrete Diffusion

### 3.1 Diffusion Beyond Continuous Data

Everything we have studied assumes continuous data -- images, latent vectors, audio waveforms. But many important data types are discrete: text, DNA sequences, program code, molecular graphs. Can diffusion models handle discrete data?

The answer is yes, but it requires rethinking the noise process. You cannot add Gaussian noise to a discrete token.

### 3.2 D3PM: Discrete Denoising Diffusion

Austin et al. (2021) introduced **D3PM** (Discrete Denoising Diffusion Probabilistic Models), which defines a forward process over discrete state spaces using transition matrices.

For a token $x\_t \in \lbrace 1, 2, \ldots, K\rbrace$, the forward process is defined by a sequence of transition matrices $Q\_t \in \mathbb{R}^{K \times K}$:

$$
q(x_t | x_{t-1}) = \text{Cat}(x_t; Q_t^\top e_{x_{t-1}})
$$

where $e\_{x\_{t-1}}$ is the one-hot vector for state $x\_{t-1}$ and $\text{Cat}$ denotes the categorical distribution.

Common choices for $Q\_t$:

**Uniform diffusion**: Each token has a small probability of transitioning to any other token:

$$
Q_t = (1 - \beta_t) I + \frac{\beta_t}{K} \mathbf{1}\mathbf{1}^\top
$$

As $t$ increases, the distribution converges to the uniform distribution over all tokens.

**Absorbing state diffusion**: Each token transitions to a special [MASK] token with probability $\beta\_t$:

$$
[Q_t]_{ij} = \begin{cases} 1 - \beta_t & \text{if } i = j \neq [\text{MASK}] \\\\ \beta_t & \text{if } j = [\text{MASK}], i \neq [\text{MASK}] \\\\ 1 & \text{if } i = j = [\text{MASK}] \end{cases}
$$

The fully corrupted state is all [MASK] tokens. The reverse process "fills in" the masks.

### 3.3 BERT as a (Nearly) Discrete Diffusion Model

There is a striking connection between absorbing-state discrete diffusion and masked language models like BERT.

BERT's training: mask 15% of tokens, predict the masked tokens from context. This is one step of a discrete denoising process -- taking a partially masked sequence and predicting the clean version.

Discrete diffusion generalizes this by:
1. Allowing variable masking rates (controlled by $t$)
2. Defining a full forward-reverse process, not just a single denoising step
3. Enabling iterative refinement during generation

The generation process for absorbing-state diffusion is: start with all [MASK], predict all tokens, remask some fraction, predict again, repeat. This is essentially the iterative refinement strategy used by MaskGIT (Chang et al., 2022) and related non-autoregressive language models.

### 3.4 MDLM: Masked Diffusion Language Models

Sahoo et al. (2024) formalized this connection with **MDLM** (Masked Diffusion Language Models), showing that masked language model training is equivalent to a specific parameterization of discrete diffusion with absorbing states.

The key result: the ELBO for absorbing-state discrete diffusion can be written as a weighted sum of masked language model losses at different masking rates. This means we can train a discrete diffusion model using exactly the same objective as BERT, just with a different masking schedule.

This opens the door to applying diffusion model techniques (guidance, distillation, flow matching) to language generation, potentially offering an alternative to autoregressive models. Early results are competitive with autoregressive models for some tasks, though autoregressive models still dominate on most language benchmarks.

---

## 4. Audio and Music Generation

### 4.1 Audio as a Diffusion Problem

Audio is a natural fit for diffusion models -- it is continuous, high-dimensional, and has well-understood spectral structure.

**WaveGrad** (Chen et al., 2021): Applies diffusion directly to waveforms. A conditional diffusion model takes a mel spectrogram as conditioning and generates the raw audio waveform. Achieves high quality with 6-50 sampling steps.

**Riffusion** (Forsgren and Martiros, 2022): A creative approach -- fine-tune Stable Diffusion to generate mel spectrograms as images, then convert to audio using Griffin-Lim or a vocoder. Demonstrates that image diffusion models can be repurposed for audio with minimal modification.

### 4.2 Music Generation

**Stable Audio** (Stability AI, 2024): A latent diffusion model for music and sound effects. Uses a variational autoencoder to compress stereo audio into a latent space, then applies a DiT-based diffusion model. Generates up to 3 minutes of 44.1kHz stereo audio.

**MusicGen** (Meta, 2023): While technically an autoregressive model (not diffusion), it represents the main competition and illustrates the design space.

**Music generation challenges**: Long-range structure (verse-chorus-verse), harmonic consistency, rhythmic stability, and multitrack arrangement. These are analogous to the temporal consistency challenges in video generation.

---

## 5. Connections to Optimal Transport

### 5.1 Schrödinger Bridges

The **Schrödinger bridge problem** (1932) asks: given two distributions $p\_0$ and $p\_1$ and a reference stochastic process $Q$ (typically Brownian motion), find the stochastic process $P^*$ that is closest to $Q$ (in KL divergence) while having marginals $p\_0$ at time 0 and $p\_1$ at time 1:

$$
P^* = \arg\min_{P : P_0 = p_0, P_1 = p_1} D_{\text{KL}}(P \Vert Q)
$$

This is a generalization of optimal transport to stochastic processes. When the reference process $Q$ has zero diffusion (pure ODE), the Schrödinger bridge reduces to the Monge optimal transport problem.

The connection to diffusion models: the diffusion forward process is a Brownian bridge from data to noise. The *optimal* forward process (the one that minimizes the KL divergence while matching the data distribution) is a Schrödinger bridge. Learning this optimal process rather than using a fixed forward process can improve generation quality.

**Iterative Proportional Fitting (IPF)** / **Bridge Matching**: Algorithms that iteratively refine the forward and backward processes to converge to the Schrödinger bridge. These are related to the reflow procedure from Week 11.

### 5.2 Flow Matching and Monge's Problem

In the deterministic limit ($\sigma \to 0$), the Schrödinger bridge reduces to the **Monge optimal transport map**: the deterministic map $T : p\_0 \to p\_1$ that minimizes the total transport cost $\mathbb{E}[\Vert x\_0 - T(x\_0)\Vert ^2]$.

Flow matching with linear interpolation $x\_t = (1-t)x\_0 + tx\_1$ is *not* the optimal transport flow (because $x\_0$ and $x\_1$ are sampled independently, not paired optimally). But it approximates it, and methods like OT-CFM (Tong et al., 2024) incorporate mini-batch optimal transport to better approximate the Monge map.

The mathematical thread from Monge (1781) to modern flow matching is surprisingly direct: how do you move mass from one distribution to another as efficiently as possible? Diffusion models solve this problem approximately; optimal transport theory tells us what the exact solution looks like.

---

## 6. Scaling Laws

### 6.1 Empirical Scaling Laws

Like language models, diffusion models obey scaling laws -- power-law relationships between compute, data, model size, and performance.

Mei et al. (2024) established scaling laws for diffusion models on ImageNet:

$$
\text{FID} \propto C^{-\alpha}
$$

where $C$ is the total training compute and $\alpha \approx 0.2$. This means a 10x increase in compute yields roughly a 37% reduction in FID -- significant but less dramatic than language model scaling, where performance improvements are often more pronounced.

Key findings:
- **Model size**: Larger models are more compute-efficient (same as language models)
- **Data**: More data helps, but with diminishing returns
- **Resolution**: Higher resolution requires disproportionately more compute
- **Architecture**: Transformers (DiT) scale more predictably than U-Nets

### 6.2 Implications

If these scaling laws hold, we can predict the compute needed for future capability milestones. For example, if current models achieve FID $\approx 2$ on ImageNet 256x256 with $C$ compute, reaching FID $\approx 1$ would require roughly $C \cdot 2^{1/\alpha} \approx 32C$ -- a 32x increase in compute.

The practical implication: raw scaling alone may not be sufficient for the next leaps in quality. Algorithmic improvements (better architectures, training procedures, noise schedules, distillation methods) provide multiplicative gains on top of scaling.

---

## 7. Open Problems

### 7.1 Theoretical: Why Do Diffusion Models Generalize?

Perhaps the most fundamental open question: why do diffusion models generalize rather than memorize?

A diffusion model trained on $N$ images can generate novel images that are clearly not in the training set. This is empirically undeniable. But the model has enough parameters to memorize the entire training set (models with hundreds of millions of parameters trained on datasets of tens of millions of images). Why does it choose to generalize?

**Possible explanations**:
- **Score function smoothness**: The score $\nabla \log p\_t(x)$ is a smooth function of $x$ (especially at high noise levels). A neural network learns this smooth function, which automatically interpolates between training examples.
- **Inductive bias of the architecture**: U-Nets and transformers have built-in biases toward smooth, structured functions. These biases prevent memorization.
- **The noise schedule as regularizer**: At high noise levels, many training examples contribute to the score at each point, forcing the model to learn shared structure rather than individual examples.
- **Implicit regularization of SGD**: Stochastic gradient descent favors flat minima, which tend to correspond to models that generalize.

None of these explanations are fully rigorous. A satisfying theory of generalization for diffusion models does not yet exist.

### 7.2 Theoretical: What Is the Optimal Noise Process?

We have used Gaussian noise throughout this course. Is Gaussian the right choice?

**Arguments for Gaussian**:
- The central limit theorem makes Gaussian noise natural
- Gaussian forward processes have closed-form marginals
- The score function has a clean mathematical form

**Arguments against**:
- For image data with bounded pixel values, Gaussian noise extends outside the data range
- For data with specific structure (e.g., symmetries, discrete components), a matched noise process might be more efficient
- Information-theoretic arguments suggest that the optimal noise process depends on the data distribution

This is connected to the choice of probability paths in flow matching: the linear interpolation $x\_t = (1-t)x\_0 + tx\_1$ assumes Gaussian source noise. Other source distributions (e.g., uniform, or learned) might give straighter paths and faster convergence.

### 7.3 Practical: Real-Time High-Resolution Video

Current video generation requires minutes to hours of compute per second of video. Real-time video generation (generating 30 fps video faster than real time) would unlock interactive applications: video games, virtual reality, real-time visual effects.

The gap is roughly 3-4 orders of magnitude: current systems generate $\sim$0.01 fps of 1080p video in real time, while the target is 30+ fps.

Closing this gap requires advances in:
- **Architecture efficiency**: Smaller, faster models that maintain quality
- **Distillation**: 1-2 step generation for video (current distillation methods work well for images but are less explored for video)
- **Temporal coherence**: Efficient mechanisms for cross-frame consistency
- **Hardware**: Purpose-built accelerators for diffusion inference

### 7.4 Practical: Controllability

Current diffusion models offer limited control: text prompts, class labels, reference images, depth maps, edge maps (via ControlNet-style conditioning). But precise spatial control -- "move the dog 10 pixels to the right while keeping the background fixed" -- remains difficult.

Open directions:
- **Compositional generation**: Generating scenes with specific spatial relationships between objects
- **Physical consistency**: Ensuring generated images obey physics (lighting, shadows, reflections)
- **Temporal control in video**: Specifying the trajectory of objects across frames
- **Fine-grained editing**: Modifying specific attributes while preserving everything else

### 7.5 Evaluation: FID Is Broken

The Fréchet Inception Distance (FID) has been the standard metric for generative model quality since 2018. It computes the Fréchet distance between the Inception-v3 feature distributions of real and generated images.

**Known problems with FID**:
- It uses Inception-v3 features, which are trained on ImageNet -- biased toward ImageNet-like content
- It assumes Gaussian feature distributions, which is a poor approximation
- It is insensitive to spatial artifacts and texture details
- It can be gamed by generating ImageNet-like images regardless of the prompt
- Reported FID values vary significantly with implementation details (preprocessing, number of samples, random seed)

**Proposed alternatives**:
- **CLIP-FID**: Use CLIP features instead of Inception features. Better aligned with human perception.
- **FDD (Fréchet DINOv2 Distance)**: Use DINOv2 features. Better visual feature extraction.
- **Human evaluation**: The gold standard, but expensive and not reproducible.
- **Precision and recall**: Measure both quality (precision: are generated images realistic?) and diversity (recall: does the generator cover the full distribution?).

No consensus replacement has emerged. This matters because metric choice influences which models and methods get published and adopted.

### 7.6 Safety: Deepfakes, Watermarking, and Detection

Diffusion models can generate photorealistic images of people, places, and events that never existed. This creates obvious risks: misinformation, fraud, harassment.

**Watermarking**: Embedding invisible signatures in generated images so they can be identified as AI-generated. Approaches include:
- **Post-hoc watermarking**: Add a watermark after generation (e.g., StegaStamp, Tree-Ring Watermarks)
- **In-model watermarking**: Modify the diffusion process to embed a watermark during generation (the noise is chosen to encode a message)

**Detection**: Identifying AI-generated images without a watermark. Approaches include:
- **Forensic classifiers**: Train a binary classifier on real vs. generated images
- **Spectral analysis**: AI-generated images have distinctive frequency characteristics
- **The cat-and-mouse problem**: As generators improve, detection becomes harder. And as detectors improve, generators can be trained to evade them.

**The fundamental tension**: The same properties that make diffusion models powerful (high-quality, controllable generation) make them dangerous. There is no purely technical solution -- policy, education, and social norms must complement technical measures.

---

## 8. The View from Here

Let us end with some perspective on where we are and where we might be going.

**What diffusion models have achieved**: In three years (2020-2023), diffusion models went from a mathematical curiosity to the dominant paradigm for image generation, surpassing GANs in quality, diversity, and controllability. They have been extended to video, 3D, audio, and molecular design. The flow matching reformulation (2023) simplified the framework while improving performance.

**What remains hard**: Real-time high-resolution video generation, precise controllability, temporal consistency over long durations, and generation of discrete data (where autoregressive models still dominate).

**The big questions**:

1. Will diffusion models scale to the same degree as language models? The scaling laws suggest yes, but the exponents are smaller, and the compute requirements for video and 3D are formidable.

2. Will discrete diffusion replace autoregressive generation for text? Current evidence suggests not in the near term, but the gap is closing. Hybrid approaches (autoregressive for global structure, diffusion for local refinement) are promising.

3. Is there a unified framework for generation across all modalities? Flow matching is a candidate -- it applies to images, video, audio, and (with discrete extensions) text. Whether a single architecture and training procedure can handle all modalities remains to be seen.

4. What is the right theoretical framework? We have studied diffusion from the perspectives of score matching, SDEs, probability flow ODEs, flow matching, and optimal transport. Each illuminates different aspects. A unified theory that explains when and why diffusion models work well -- and predicts when they will fail -- does not yet exist.

These are the questions your generation will answer. The mathematical tools are in your hands.

---

## Summary

1. **Video diffusion** extends image diffusion with temporal attention and 3D architectures. Temporal consistency remains the core challenge.

2. **3D generation** uses SDS (DreamFusion), multi-view diffusion, or direct 3D diffusion. Gaussian splatting is emerging as the preferred 3D representation.

3. **Discrete diffusion** (D3PM, MDLM) extends diffusion to text and other discrete data using transition matrices. Absorbing-state diffusion is nearly equivalent to masked language modeling.

4. **Audio diffusion** (WaveGrad, Stable Audio) applies diffusion to waveforms or spectrograms for speech, music, and sound effects.

5. **Optimal transport** connections (Schrödinger bridges, Monge's problem) provide theoretical grounding for flow matching and suggest paths to optimal training procedures.

6. **Scaling laws** show power-law improvement with compute, but the exponents are smaller than for language models.

7. **Open problems** include theoretical questions (generalization, optimal noise process), practical challenges (real-time video, controllability), evaluation issues (FID alternatives), and safety concerns (deepfakes, watermarking).

---

## Suggested Reading

- **Ho et al.** (2022), "Video Diffusion Models" -- foundational video diffusion paper.
- **Poole et al.** (2023), "DreamFusion: Text-to-3D using 2D Diffusion" -- SDS and text-to-3D.
- **Austin et al.** (2021), "Structured Denoising Diffusion Models in Discrete State-Spaces" -- D3PM, the foundational discrete diffusion paper.
- **Sahoo et al.** (2024), "Simple and Effective Masked Diffusion Language Models" -- MDLM, connecting masked language models to discrete diffusion.
- **Liu et al.** (2023), "Zero-1-to-3: Zero-shot One Image to 3D Object" -- single-image 3D generation.
- **De Bortoli et al.** (2021), "Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling" -- Schrödinger bridges for diffusion.
- **Tong et al.** (2024), "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport" -- OT-CFM.
- **Mei et al.** (2024), "Bigger is not Always Better: Scaling Properties of Latent Diffusion Models" -- scaling laws for diffusion.
- **Stein et al.** (2023), "Exposing Flaws of Generative Model Evaluation Metrics and Their Unfair Treatment of Diffusion Models" -- critique of FID.
