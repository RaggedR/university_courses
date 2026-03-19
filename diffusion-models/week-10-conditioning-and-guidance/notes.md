# Week 10: Conditioning and Guidance

> *"The question is not whether machines can think, but whether we can tell them what to think about."*
> -- loosely after Alan Turing

---

## Overview

An unconditional diffusion model generates images from the data distribution $p(x)$ -- it produces "some image" without any control over what appears. This is mathematically satisfying but practically useless. We almost always want to generate a *specific* image: one that matches a text description, belongs to a particular class, or follows a spatial layout.

The problem of **conditional generation** is: given a condition $y$ (text, class label, image, sketch, etc.), sample from the conditional distribution $p(x|y)$. Bayes' theorem gives us the decomposition:

$$\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$$

The first term is the unconditional score (which we already have from the diffusion model). The second term tells us how to push the generation toward satisfying the condition $y$. The entire story of guidance is about how to estimate and use this second term.

We will develop two approaches -- **classifier guidance** and **classifier-free guidance** -- and then see how conditioning signals (text, images, spatial controls) are injected into the architecture.

### Prerequisites
- Week 5: DDPM training and sampling
- Week 6: Score functions, $\nabla_x \log p(x) \approx -\epsilon_\theta(x,t)/\sigma_t$
- Week 8: DDIM and ODE-based sampling
- Week 9: U-Net architecture, cross-attention

---

## 1. Conditional Diffusion Models

### 1.1 The Setup

A conditional diffusion model learns the conditional score function:

$$s_\theta(x_t, t, y) \approx \nabla_x \log p_t(x|y)$$

or equivalently, a conditional noise predictor:

$$\epsilon_\theta(x_t, t, y) \approx -\sigma_t \nabla_x \log p_t(x|y)$$

The training objective is the same as unconditional DDPM, but with $y$ as an additional input:

$$\mathcal{L} = \mathbb{E}_{x_0, y, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t, y)\|^2\right]$$

where $(x_0, y)$ are paired training examples (e.g., image-caption pairs).

### 1.2 How Conditioning Enters the Network

The condition $y$ can enter the denoising network in several ways:

**Concatenation.** Append $y$ (or an embedding of $y$) to the input. For spatial conditions (e.g., segmentation maps), concatenate along the channel dimension.

**Addition.** Add an embedding of $y$ to the time embedding, so it is injected into every ResNet block. This works well for simple conditions like class labels.

**Cross-attention.** For sequential conditions (text), use cross-attention between the U-Net features and the conditioning embeddings. This is the dominant approach and we will study it in detail in Section 5.

**Adaptive normalization.** Modulate the normalization layers using the condition, as in DiT's adaLN (Week 9). Effective for class labels and style codes.

### 1.3 The Limitation of Pure Conditioning

A conditional model $\epsilon_\theta(x_t, t, y)$ produces reasonable results, but often the conditioning is *weak*: the generated images are plausible but do not strongly reflect the condition. A model trained on image-caption pairs may generate a vaguely correct scene but miss specific details from the caption.

This is because the model is trained to minimize the *expected* prediction error over all (image, caption) pairs, which encourages it to produce safe, average predictions rather than sharp, condition-specific ones. Guidance is the solution.

---

## 2. Classifier Guidance

### 2.1 The Idea

Dhariwal and Nichol (2021) proposed a simple, beautiful approach: use a pretrained classifier to steer the diffusion process.

Suppose we have a classifier $p_\phi(y|x_t)$ that can predict the class label $y$ from a *noisy* image $x_t$ at any noise level $t$. By Bayes' theorem, the conditional score is:

$$\nabla_x \log p_t(x|y) = \nabla_x \log p_t(x) + \nabla_x \log p_\phi(y|x_t)$$

The first term is the unconditional score, estimated by the diffusion model. The second term is the gradient of the log-classifier with respect to $x_t$ -- it points in the direction that makes $x_t$ more strongly classified as class $y$.

### 2.2 The Guided Sampling Update

Substituting into the DDPM sampling formula:

$$\hat{\epsilon}_{\text{guided}} = \epsilon_\theta(x_t, t) - s \cdot \sigma_t \cdot \nabla_x \log p_\phi(y|x_t)$$

where $s$ is the **guidance scale**. With $s = 0$, we recover unconditional sampling. With $s > 0$, we push the generation toward images that the classifier confidently labels as class $y$.

The modified reverse step:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_{\text{guided}}\right) + \sigma_t z$$

### 2.3 The Guidance Scale

The guidance scale $s$ controls the strength of conditioning:

- **$s = 0$**: Unconditional sampling. The classifier has no effect.
- **$s = 1$**: Standard conditional sampling. This is the Bayes-optimal balance between the prior $p(x)$ and the likelihood $p(y|x)$.
- **$s > 1$**: Over-conditioning. The generation is pushed harder toward the classifier's preferred region. Images become more recognizable as class $y$ but less diverse. At extreme values, the model generates stereotypical, oversaturated examples.

The trade-off is between **fidelity** (how well the image matches the condition) and **diversity** (how varied the generated images are). Dhariwal and Nichol showed that guidance with $s \approx 2$-$4$ dramatically improves FID and classification accuracy at the cost of some diversity. This was the first time diffusion models convincingly beat GANs on ImageNet generation.

### 2.4 Training the Noisy Classifier

A crucial requirement: the classifier must work on *noisy* images $x_t$, not clean images $x_0$. Standard ImageNet classifiers fail on noisy inputs because they were trained on clean images.

The solution is to train a classifier $p_\phi(y|x_t, t)$ on pairs $(x_t, y)$ where $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$ is the noisy version of a training image $x_0$ with known label $y$. The classifier receives the noise level $t$ as an additional input (via the same sinusoidal embedding used in the diffusion model).

This is essentially training a classifier with aggressive data augmentation (Gaussian noise at many scales).

### 2.5 Limitations

Classifier guidance has clear drawbacks:

1. **Requires training a separate classifier.** For each new condition type, you need a new classifier. For text conditioning, you would need a classifier that maps noisy images to text descriptions -- impractical.

2. **The classifier must handle noisy inputs.** It must be trained specifically for this purpose, with noise-conditioned architecture.

3. **Gradient quality.** The classifier's gradients must be informative in high-dimensional image space. Deep classifiers can have adversarial-like gradients that push images toward classifier confidence without improving visual quality.

4. **Only works for classification-like conditions.** Text, layout, style -- these do not fit naturally into a classifier framework.

---

## 3. Classifier-Free Guidance

### 3.1 The Breakthrough

Ho and Salimans (2022) eliminated the need for an external classifier entirely. The key insight: **train the diffusion model itself to be both conditional and unconditional**, then use the difference between them as an implicit classifier gradient.

### 3.2 Training

During training, randomly drop the conditioning $y$ with some probability $p_{\text{uncond}}$ (typically 10-20%):

$$\epsilon_\theta(x_t, t, y) = \begin{cases} \epsilon_\theta(x_t, t, y) & \text{with probability } 1 - p_{\text{uncond}} \\ \epsilon_\theta(x_t, t, \varnothing) & \text{with probability } p_{\text{uncond}} \end{cases}$$

where $\varnothing$ is a null/empty condition (e.g., a zero vector, an empty string embedding, or a learned "null" token).

After training, the single network $\epsilon_\theta$ can produce both:
- **Conditional predictions**: $\epsilon_\theta(x_t, t, y)$ (the model's best guess of the noise, given the condition)
- **Unconditional predictions**: $\epsilon_\theta(x_t, t, \varnothing)$ (the model's best guess without any condition)

### 3.3 Sampling with Classifier-Free Guidance

At sampling time, the guided noise prediction is:

$$\boxed{\tilde{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot \left[\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing)\right]}$$

Equivalently:

$$\tilde{\epsilon} = (1 - w) \cdot \epsilon_\theta(x_t, t, \varnothing) + w \cdot \epsilon_\theta(x_t, t, y)$$

where $w$ is the **guidance weight** (also called the guidance scale or CFG scale). The default in most applications is $w = 7.5$.

### 3.4 Interpreting the Formula

Let us understand what this formula does.

**The conditional-unconditional difference** $\Delta\epsilon = \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing)$ is a vector in the noise prediction space. It points in the direction that "makes the prediction more consistent with the condition $y$."

When $w = 1$, we use the conditional prediction directly: $\tilde{\epsilon} = \epsilon_\theta(x_t, t, y)$. This is standard conditional generation -- no guidance.

When $w > 1$, we **extrapolate beyond** the conditional prediction, moving further in the direction away from the unconditional prediction. We are saying: "the conditional model wants to go this way, so go *even further* in that direction."

When $w = 0$, we use the unconditional prediction: $\tilde{\epsilon} = \epsilon_\theta(x_t, t, \varnothing)$. The condition is ignored.

### 3.5 Why It Is Called "Classifier-Free"

Recall that classifier guidance uses:

$$\hat{\epsilon}_{\text{guided}} = \epsilon_\theta(x_t, t) - s \cdot \sigma_t \cdot \nabla_x \log p_\phi(y|x_t)$$

Classifier-free guidance achieves the same effect without an explicit classifier. To see this, note that:

$$\nabla_x \log p(y|x_t) = \nabla_x \log p(x_t|y) - \nabla_x \log p(x_t)$$

Converting to noise predictions:

$$-\sigma_t \nabla_x \log p(y|x_t) \approx \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing)$$

So the "implicit classifier gradient" is exactly the difference between conditional and unconditional noise predictions. Classifier-free guidance uses this implicit gradient with scale $w - 1$ (i.e., $w = 1$ corresponds to $s = 0$, and $w = 1 + s$).

### 3.6 The Guidance Scale in Practice

The guidance weight $w$ has a profound effect on generation quality:

| $w$ | Effect |
|-----|--------|
| 0 | Unconditional: condition is ignored |
| 1 | Standard conditional: no guidance |
| 3-5 | Mild guidance: improved coherence |
| 7-8 | Standard (Stable Diffusion default): good fidelity-diversity trade-off |
| 10-15 | Strong guidance: high fidelity but reduced diversity, may look "overcooked" |
| 20+ | Extreme: oversaturated colors, exaggerated features, artifacts |

The sweet spot depends on the model and task. Most Stable Diffusion users find $w = 7$-$8$ works well for general use, with lower values for artistic/creative generation and higher values for photorealistic or specification-adherent generation.

### 3.7 Computational Cost

Classifier-free guidance requires **two forward passes** per sampling step: one conditional ($\epsilon_\theta(x_t, t, y)$) and one unconditional ($\epsilon_\theta(x_t, t, \varnothing)$). This doubles the cost per step compared to unguided sampling.

In practice, the two passes can be batched (process both in a single batched forward pass with batch size 2), so the overhead is less than 2x due to GPU parallelism. Still, it is a meaningful cost, and reducing it is an active area of research.

---

## 4. Geometric Interpretation

### 4.1 The $\epsilon$-Prediction Space

Consider the space of all possible noise predictions $\epsilon \in \mathbb{R}^d$ (where $d$ is the image or latent dimension). At each step, the model produces:

- $\epsilon_{\text{uncond}} = \epsilon_\theta(x_t, t, \varnothing)$: the unconditional prediction
- $\epsilon_{\text{cond}} = \epsilon_\theta(x_t, t, y)$: the conditional prediction

These are two points in $\mathbb{R}^d$. The vector $\Delta = \epsilon_{\text{cond}} - \epsilon_{\text{uncond}}$ is the "direction of conditioning."

Classifier-free guidance computes:

$$\tilde{\epsilon} = \epsilon_{\text{uncond}} + w \cdot \Delta$$

Geometrically, this is a point on the ray from $\epsilon_{\text{uncond}}$ through $\epsilon_{\text{cond}}$. With $w = 1$, we land exactly at $\epsilon_{\text{cond}}$. With $w > 1$, we overshoot, extrapolating beyond the conditional prediction.

### 4.2 Why Extrapolation Helps

The conditional model $\epsilon_\theta(x_t, t, y)$ is trained on finite data and learns an average prediction over all images matching condition $y$. This average tends to be conservative -- it hedges between different possible images.

Extrapolation ($w > 1$) amplifies the conditioning signal, effectively sharpening the conditional distribution. The unconditional model represents the "average over everything"; the conditional model represents the "average over images matching $y$"; extrapolation represents a sharpened version of the conditional distribution.

In probabilistic terms, guidance with scale $w$ corresponds to sampling from:

$$\tilde{p}(x|y) \propto p(x) \cdot p(y|x)^w$$

When $w > 1$, the likelihood is raised to a power greater than 1, concentrating the distribution on the highest-likelihood images. This is a "tempered" or "sharpened" posterior.

---

## 5. Text Conditioning: Cross-Attention

### 5.1 The Text Encoding Pipeline

In text-to-image models like Stable Diffusion, the text prompt goes through:

1. **Tokenization**: The text is split into tokens (subwords). CLIP uses BPE tokenization with a vocabulary of ~49K tokens.
2. **Token embedding**: Each token is mapped to a learned embedding vector.
3. **Transformer encoding**: The sequence of embeddings is processed by the CLIP text encoder (a transformer), producing a sequence of context vectors $c = [c_1, c_2, \ldots, c_L] \in \mathbb{R}^{L \times d}$.

For Stable Diffusion 1.x: $L = 77$ (maximum sequence length), $d = 768$ (CLIP ViT-L/14 embedding dimension).

### 5.2 Cross-Attention Mechanism

The text embeddings enter the U-Net via **cross-attention** layers, inserted after the self-attention layers at certain resolutions (typically 16x16, 32x32, and 64x64 in the latent space).

In self-attention, queries, keys, and values all come from the same source (the U-Net features). In cross-attention, the queries come from the U-Net features and the keys/values come from the text embeddings:

$$Q = W_Q h, \quad K = W_K c, \quad V = W_V c$$

$$\text{CrossAttn}(h, c) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $h \in \mathbb{R}^{n \times d_h}$ is the (flattened) spatial feature map from the U-Net and $c \in \mathbb{R}^{L \times d_c}$ is the text embedding sequence.

Each spatial position in $h$ attends to all text tokens. A position that should contain a "cat" will learn to attend to the "cat" token; a position that should be "blue sky" will attend to "blue" and "sky".

### 5.3 Why Cross-Attention Works

Cross-attention provides:

**Spatial grounding.** Each spatial location independently queries the text embedding, allowing different parts of the image to "read" different parts of the prompt. This enables spatial composition: "a cat on the left and a dog on the right" can be satisfied because left-side positions attend to "cat" and right-side positions attend to "dog."

**Flexible conditioning length.** The text can be any length (up to the maximum). Short prompts result in many padding tokens that the attention mechanism learns to ignore. Long prompts provide more information for the model to use.

**Semantic alignment.** CLIP's text encoder is trained on image-text pairs, so its embeddings already carry visual semantics. The word "sunset" in CLIP space is already near visual features of sunsets. Cross-attention leverages this pre-existing alignment.

### 5.4 Attention Maps as Interpretability Tools

The cross-attention maps $A = \text{softmax}(QK^\top/\sqrt{d_k}) \in \mathbb{R}^{n \times L}$ reveal which text tokens influence which spatial positions. Visualizing these maps shows where the model "places" each concept:

- The attention map for "cat" lights up where the cat appears in the image
- The attention map for "sunset" lights up in the sky region
- The attention map for "sitting" lights up at the boundary between the cat and its support surface

This interpretability has been exploited for editing (Prompt-to-Prompt: modifying attention maps to edit images) and for improving spatial composition (Attend-and-Excite: ensuring all mentioned objects get sufficient attention).

---

## 6. Negative Prompts

### 6.1 How They Work

Negative prompts are a user-facing feature of classifier-free guidance. The standard CFG formula uses the null embedding $\varnothing$ for the unconditional prediction. Negative prompts replace $\varnothing$ with an embedding of the negative text $y_{\text{neg}}$:

$$\tilde{\epsilon} = \epsilon_\theta(x_t, t, y_{\text{neg}}) + w \cdot \left[\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, y_{\text{neg}})\right]$$

This steers the generation *away from* the negative prompt and *toward* the positive prompt. The guidance direction is the difference between the positive and negative conditional predictions.

### 6.2 Why This Works

In the $\epsilon$-prediction space, the negative prompt shifts the "baseline" from $\epsilon_{\text{uncond}}$ to $\epsilon_{\text{neg}}$. The guidance direction becomes $\epsilon_{\text{pos}} - \epsilon_{\text{neg}}$, which points from the negative concept toward the positive concept.

For example: positive prompt "detailed, sharp, high quality" and negative prompt "blurry, low quality, deformed" creates a guidance direction that points from "blurry" toward "sharp" -- more strongly than just pointing away from the null prompt.

Common negative prompts like "low quality, blurry, deformed hands" work because they shift the baseline toward the model's failure modes, and the guidance direction then points strongly away from those modes.

---

## 7. Image Conditioning: ControlNet and Beyond

### 7.1 The Problem

Text is good for describing *what* to generate but bad for specifying *where* and *how*. "A person standing" does not specify the pose. "A building" does not specify the perspective. We need spatial conditioning.

### 7.2 ControlNet

Zhang et al. (2023) proposed ControlNet: a method to add spatial control to a pretrained text-to-image model without modifying its weights.

**Architecture.** ControlNet creates a trainable copy of the U-Net's encoder blocks. The control signal (edge map, depth map, pose skeleton) is processed by this copy, and its outputs are added to the frozen U-Net's skip connections:

$$h_{\text{dec}}^{(l)} = h_{\text{dec, original}}^{(l)} + \text{zero\_conv}(h_{\text{control}}^{(l)})$$

The "zero convolution" is a 1x1 convolution initialized with zero weights. At the start of training, ControlNet has no effect on the output (because the zero convolutions output zeros), preserving the pretrained model's quality. During training, the zero convolutions gradually learn to inject the control signal.

**Training.** ControlNet is trained on (image, control signal, text) triplets:
- Edge maps: extracted using Canny edge detection
- Depth maps: estimated using a pretrained depth model (MiDaS)
- Pose: estimated using OpenPose
- Segmentation: from annotated datasets or pretrained segmentation models

The training objective is the standard diffusion loss, with the control signal as an additional input.

### 7.3 IP-Adapter: Image Prompt Conditioning

IP-Adapter (Ye et al., 2023) conditions generation on a reference image rather than text. It uses CLIP image embeddings (instead of text embeddings) as the cross-attention keys and values:

$$K_{\text{ip}} = W_{K'} \cdot \text{CLIP}_{\text{image}}(x_{\text{ref}}), \quad V_{\text{ip}} = W_{V'} \cdot \text{CLIP}_{\text{image}}(x_{\text{ref}})$$

These are injected via an additional cross-attention layer that runs in parallel with the text cross-attention:

$$h = h + \text{CrossAttn}(h, c_{\text{text}}) + \lambda \cdot \text{CrossAttn}(h, c_{\text{image}})$$

where $\lambda$ controls the strength of the image conditioning.

### 7.4 Other Conditioning Modalities

The same principles extend to many types of conditioning:
- **Inpainting**: Concatenate a binary mask and the masked image to the U-Net input
- **Super-resolution**: Concatenate the low-resolution image (upsampled) to the input
- **Style transfer**: Use style features from a reference image as conditioning
- **Video**: Condition each frame on the previous frame(s) via cross-attention or concatenation

The architectural pattern is consistent: convert the condition into an appropriate representation and inject it via concatenation, addition, or cross-attention.

---

## 8. Practical Guidance Strategies

### 8.1 Dynamic Guidance

Some recent approaches vary the guidance scale across the sampling trajectory:
- **High guidance early** (high noise): strong conditioning to establish the global structure
- **Low guidance late** (low noise): reduce guidance to allow natural detail and avoid oversaturation

This is motivated by the observation that guidance with $w > 1$ can introduce artifacts at low noise levels (sharp edges, oversaturated colors) while being essential at high noise levels.

### 8.2 Rescaled Classifier-Free Guidance

Lin et al. (2024) observed that CFG with $w > 1$ can push the noise prediction $\tilde{\epsilon}$ outside the expected range, causing oversaturation. Their fix: rescale the guided prediction to match the standard deviation of the unguided prediction:

$$\tilde{\epsilon}_{\text{rescaled}} = \tilde{\epsilon} \cdot \frac{\text{std}(\epsilon_{\text{cond}})}{\text{std}(\tilde{\epsilon})} \cdot \phi + \tilde{\epsilon} \cdot (1 - \phi)$$

where $\phi \in [0, 1]$ controls the rescaling strength.

### 8.3 Perp-Neg: Perpendicular Negative Guidance

When multiple concepts are involved (e.g., "a cat" with negative prompt "a dog"), standard negative guidance can accidentally suppress features shared between the positive and negative concepts (both cats and dogs have fur, eyes, legs). Perp-Neg (Armandpour et al., 2023) projects the negative guidance to be perpendicular to the positive direction, preserving shared features while suppressing only the features unique to the negative concept.

---

## Summary

1. **Conditional diffusion** models learn $\epsilon_\theta(x_t, t, y)$ to generate images conditioned on $y$. The condition enters via concatenation, addition, cross-attention, or adaptive normalization.

2. **Classifier guidance** uses an external classifier $p_\phi(y|x_t)$ trained on noisy images. The modified score is $\nabla_x \log p_t(x|y) = \nabla_x \log p_t(x) + s \cdot \nabla_x \log p_\phi(y|x_t)$, with guidance scale $s$ trading diversity for fidelity.

3. **Classifier-free guidance** eliminates the external classifier by training the diffusion model with random conditioning dropout. The guided prediction $\tilde{\epsilon} = \epsilon_{\text{uncond}} + w \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$ extrapolates in the conditioning direction. This is equivalent to an implicit classifier with scale $w - 1$.

4. **The guidance scale** ($s$ or $w$) trades diversity for fidelity. $w = 7$-$8$ is standard for text-to-image generation. Higher values produce more condition-adherent but less diverse (and potentially oversaturated) images.

5. **Text conditioning** uses CLIP/T5 text embeddings injected via cross-attention. Each spatial position in the U-Net attends to all text tokens, enabling spatial grounding and flexible-length prompts.

6. **Negative prompts** modify the CFG baseline: $\tilde{\epsilon} = \epsilon_{\text{neg}} + w \cdot (\epsilon_{\text{pos}} - \epsilon_{\text{neg}})$, steering generation away from the negative concept.

7. **ControlNet** adds spatial conditioning (edges, depth, pose) to a frozen pretrained model via a trainable copy of the encoder with zero-initialized injection. **IP-Adapter** adds image conditioning via additional cross-attention.

---

## Key Equations

| Concept | Equation |
|---------|----------|
| Conditional score (Bayes) | $\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$ |
| Classifier guidance | $\hat{\epsilon} = \epsilon_\theta(x_t,t) - s \sigma_t \nabla_x \log p_\phi(y|x_t)$ |
| Classifier-free guidance | $\tilde{\epsilon} = \epsilon_\theta(x_t,t,\varnothing) + w[\epsilon_\theta(x_t,t,y) - \epsilon_\theta(x_t,t,\varnothing)]$ |
| Equivalent form | $\tilde{\epsilon} = (1-w)\,\epsilon_\theta(x_t,t,\varnothing) + w\,\epsilon_\theta(x_t,t,y)$ |
| Tempered posterior | $\tilde{p}(x|y) \propto p(x) \cdot p(y|x)^w$ |
| Cross-attention | $\text{CrossAttn}(h,c) = \text{softmax}(QK^\top/\sqrt{d_k})V$, $Q=W_Q h$, $K=W_K c$, $V=W_V c$ |
| Negative prompts | $\tilde{\epsilon} = \epsilon_{\text{neg}} + w(\epsilon_{\text{pos}} - \epsilon_{\text{neg}})$ |

---

## Suggested Reading

- **Dhariwal and Nichol** (2021), "Diffusion Models Beat GANs on Image Synthesis" -- the classifier guidance paper. Sections 4-5 introduce guidance and the noisy classifier. The FID results in Table 1 are striking.
- **Ho and Salimans** (2022), "Classifier-Free Diffusion Guidance" -- the classifier-free guidance paper. Short and clearly written. Read the full paper.
- **Zhang, Rao, and Agrawala** (2023), "Adding Conditional Control to Text-to-Image Diffusion Models" -- the ControlNet paper. Focus on the architecture in Section 3.
- **Rombach et al.** (2022), "High-Resolution Image Synthesis with Latent Diffusion Models" -- Section 3.3 describes the cross-attention conditioning mechanism.
- **Hertz et al.** (2023), "Prompt-to-Prompt Image Editing with Cross Attention Control" -- shows how cross-attention maps can be manipulated for image editing. Provides excellent intuition for what cross-attention does.
- **Lin et al.** (2024), "Common Diffusion Noise Schedules and Sample Steps are Flawed" -- Section 3 on rescaled CFG and guidance artifacts.
