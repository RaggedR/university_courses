# Week 10: Conditioning and Guidance -- Homework

**Estimated time:** 12-16 hours
**Prerequisites:** Score functions (Week 6), DDIM sampling (Week 8), U-Net with time conditioning (Week 9), PyTorch

---

## Problem 1: Classifier Guidance from Bayes' Theorem (Theory)

### Part (a): Derive the Guided Score

Starting from Bayes' theorem $p(x|y) = p(y|x)p(x)/p(y)$, take the gradient $\nabla\_x \log$ of both sides to derive:

$$
\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)
$$

Explain why the $\nabla\_x \log p(y)$ term vanishes.

### Part (b): From Score to $\epsilon$

Using the relationship $s\_\theta(x\_t, t) = -\epsilon\_\theta(x\_t, t) / \sigma\_t$ (where $\sigma\_t = \sqrt{1-\bar{\alpha}\_t}$), show that the guided noise prediction is:

$$
\hat{\epsilon}_{\text{guided}} = \epsilon_\theta(x_t, t) - s \cdot \sigma_t \cdot \nabla_x \log p_\phi(y|x_t)
$$

where $s$ is the guidance scale. What happens to the guidance term as $t \to 0$ (low noise)? As $t \to T$ (high noise)? Is this intuitively correct?

### Part (c): The Tempered Posterior

Classifier-free guidance with weight $w$ corresponds to sampling from:

$$
\tilde{p}(x|y) \propto p(x) \cdot p(y|x)^{w}
$$

1. Derive this by showing that the guided score $\tilde{s} = \nabla\_x \log p(x) + w \cdot \nabla\_x \log p(y|x)$ is the score of $\tilde{p}$.

2. What distribution does this reduce to when $w = 0$? When $w = 1$?

3. For a Gaussian likelihood $p(y|x) = \mathcal{N}(y; f(x), \sigma^2 I)$ and Gaussian prior $p(x) = \mathcal{N}(0, I)$, sketch (in 1D) the tempered posterior for $w = 0.5, 1, 2, 5$. How does increasing $w$ affect the mode and the spread?

4. Explain in 2-3 sentences why this temperature-scaling interpretation makes the diversity-fidelity trade-off intuitive.

---

## Problem 2: Implement Classifier Guidance (Implementation)

### Part (a): Train a Noisy Classifier

Train a classifier on MNIST that can handle noisy inputs:

```python
class NoisyClassifier(nn.Module):
    """
    Classifies noisy MNIST digits.
    Input: noisy image x_t (B, 1, 28, 28) + noise level t (B,)
    Output: class logits (B, 10)
    """
    def __init__(self):
        super().__init__()
        # CNN with time conditioning (similar to U-Net ResBlocks)
        # Time embedding: sinusoidal -> MLP
        # Convolutional layers with time-conditioned normalization
        # Global average pooling -> linear -> 10 classes
        pass

    def forward(self, x_t, t):
        pass
```

Training procedure:
1. For each batch of $(x\_0, y)$, sample $t \sim \text{Uniform}(0, T)$ and $\epsilon \sim \mathcal{N}(0, I)$
2. Compute $x\_t = \sqrt{\bar{\alpha}\_t}\, x\_0 + \sqrt{1-\bar{\alpha}\_t}\, \epsilon$
3. Predict $\hat{y} = \text{Classifier}(x\_t, t)$
4. Loss: cross-entropy between $\hat{y}$ and $y$

After training, report the classification accuracy at noise levels $t = 0, 100, 500, 900$ on the test set. The classifier should work well at low noise and degrade gracefully at high noise.

### Part (b): Guided Sampling

Using a pretrained unconditional DDPM on MNIST (from Week 5 homework, or train a new one):

1. Implement classifier-guided DDPM sampling:

```python
def classifier_guided_sample(diffusion_model, classifier, class_label,
                              guidance_scale, num_steps=1000):
    """
    Generate a sample of the given class using classifier guidance.

    At each step:
    1. Compute epsilon_theta(x_t, t) from the diffusion model
    2. Compute grad_x log p(y|x_t) from the classifier
    3. Modify: epsilon_guided = epsilon_theta - s * sigma_t * grad_log_p
    4. Apply the DDPM reverse step with epsilon_guided
    """
    pass
```

2. Generate 10 samples of each digit (0-9) with guidance scales $s \in \lbrace 0, 1, 3, 5, 10\rbrace$.

3. Display the results as a 10x5 grid (rows = digits, columns = guidance scales). Describe the visual trend as $s$ increases.

### Part (c): Guidance Strength Analysis

For class "7":
1. Generate 100 samples at each $s \in \lbrace 0, 0.5, 1, 2, 3, 5, 10, 20\rbrace$
2. Classify each sample with a clean (not noise-conditioned) MNIST classifier
3. Report: (i) fraction correctly classified as "7", (ii) visual diversity (qualitative), (iii) presence of artifacts at high $s$

Plot classification accuracy vs. $s$. Identify the approximate $s$ at which accuracy saturates and the $s$ at which artifacts begin to appear.

---

## Problem 3: Implement Classifier-Free Guidance (Implementation)

### Part (a): Class-Conditional DDPM with Dropout

Modify your DDPM model to accept a class label, with random dropout during training:

```python
class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes, uncond_prob=0.1):
        super().__init__()
        self.uncond_prob = uncond_prob
        self.class_embedding = nn.Embedding(num_classes + 1, embed_dim)
        # +1 for the "null" class (unconditional)
        self.null_class = num_classes  # index for unconditional
        # ... U-Net architecture (from Week 9)

    def forward(self, x_t, t, y):
        """
        During training: randomly replace y with null_class
        with probability uncond_prob.
        """
        if self.training:
            # Randomly drop conditioning
            mask = torch.rand(y.shape[0]) < self.uncond_prob
            y = y.clone()
            y[mask] = self.null_class
        # Embed y, add to time embedding, pass through U-Net
        pass
```

Train on MNIST for 50-100 epochs.

### Part (b): CFG Sampling

Implement classifier-free guided sampling:

```python
def cfg_sample(model, class_label, guidance_weight, num_steps=50):
    """
    Classifier-free guidance sampling with DDIM.

    At each step:
    1. eps_uncond = model(x_t, t, null_class)
    2. eps_cond = model(x_t, t, class_label)
    3. eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
    4. Apply DDIM step with eps_guided
    """
    pass
```

Generate samples for each digit with $w \in \lbrace 1, 3, 5, 7.5, 10, 15\rbrace$.

### Part (c): Compare with Classifier Guidance

Using the same model and dataset, compare:
1. **Classifier guidance** ($s = 3$) from Problem 2
2. **Classifier-free guidance** ($w = 4$, which corresponds roughly to $s = 3$ since $w = 1 + s$)

For 10 samples of each digit:
1. Display both sets of samples side by side
2. Which produces better quality? Which is more diverse?
3. Which requires more computation per sample? (Count NFEs.)

### Part (d): The Unconditional Dropout Rate

Train three models with $p\_{\text{uncond}} \in \lbrace 0.05, 0.1, 0.2\rbrace$. For each, generate samples with $w = 7.5$.

1. Does the dropout rate significantly affect sample quality?
2. What happens if $p\_{\text{uncond}} = 0$ (no unconditional training)? Try to use CFG -- what goes wrong?
3. What happens if $p\_{\text{uncond}} = 0.5$ (half unconditional)? Does the conditional model still work well at $w = 1$?

---

## Problem 4: Guidance Scale Experiment (Implementation)

### Part (a): The Diversity-Fidelity Curve

Using your CFG model from Problem 3:

1. For the digit "3", generate 500 samples at each $w \in \lbrace 1, 2, 3, 5, 7, 10, 15, 20\rbrace$

2. For each set of 500 samples, compute:
   - **Fidelity**: Average confidence of a pretrained classifier on the correct class
   - **Diversity**: Average pairwise distance between samples (use L2 distance in pixel space, or LPIPS if available)

3. Plot a diversity-fidelity curve (diversity on x-axis, fidelity on y-axis, each point labeled with its $w$ value). The curve should show the trade-off: as $w$ increases, fidelity improves but diversity decreases.

### Part (b): Per-Class Behavior

Repeat Part (a) for all 10 digits at $w = 1, 5, 10$.

1. Which digits benefit most from guidance (biggest fidelity improvement from $w=1$ to $w=5$)?
2. Which digits lose the most diversity?
3. Is there a digit where guidance actually *hurts* quality? (This can happen for digits that are already well-modeled unconditionally.)

### Part (c): Visualizing the Guidance Direction

For one fixed noisy input $x\_t$ (at $t = 500$, mid-noise) and class "8":

1. Compute $\epsilon\_{\text{uncond}} = \epsilon\_\theta(x\_t, t, \varnothing)$
2. Compute $\epsilon\_{\text{cond}} = \epsilon\_\theta(x\_t, t, y=8)$
3. Compute $\Delta = \epsilon\_{\text{cond}} - \epsilon\_{\text{uncond}}$

Reshape $\Delta$ to image dimensions and display it. What does the "guidance direction" look like? Does it visually resemble the structure of an "8"?

Repeat for classes "0", "1", "7". How do the guidance directions differ?

---

## Problem 5: Cross-Attention Conditioning (Implementation)

### Part (a): Implement Cross-Attention

Implement a cross-attention layer:

```python
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=4, head_dim=64):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        """
        Args:
            x: (B, N, query_dim)  -- spatial features (queries)
            context: (B, L, context_dim)  -- conditioning sequence (keys/values)
        Returns:
            (B, N, query_dim)
        """
        pass
```

Test: create random $x$ (batch 2, 64 spatial tokens, dim 256) and $context$ (batch 2, 10 text tokens, dim 512). Verify the output shape is (2, 64, 256).

### Part (b): U-Net with Cross-Attention

Modify the U-Net from Week 9 (Problem 3) to accept cross-attention conditioning. Add a cross-attention layer after each self-attention layer in the U-Net.

For conditioning, use a simple "text encoder": embed each of 10 class labels as a sequence of learned tokens:

```python
class SimpleTextEncoder(nn.Module):
    def __init__(self, num_classes, seq_len=4, embed_dim=512):
        """
        Maps class label to a sequence of embeddings.
        This is a toy stand-in for CLIP.
        """
        super().__init__()
        self.embeddings = nn.Embedding(num_classes, embed_dim * seq_len)
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self, y):
        # (B,) -> (B, seq_len, embed_dim)
        return self.embeddings(y).reshape(-1, self.seq_len, self.embed_dim)
```

### Part (c): Train and Evaluate

Train the cross-attention conditioned U-Net on MNIST with CFG (10% unconditional dropout).

Generate samples with $w = 7.5$ for all 10 digits. Compare with the class-conditional model from Problem 3 (which used class embedding added to the time embedding).

1. Which conditioning method produces better class-specific samples?
2. Extract and visualize the cross-attention maps for one generated sample. For each class token, display the attention map (reshaped to spatial dimensions). Do the attention maps show meaningful spatial patterns?

---

## Problem 6: Negative Prompts and Guidance Geometry (Theory + Implementation)

### Part (a): Negative Prompt Derivation

In standard CFG: $\tilde{\epsilon} = \epsilon\_{\text{uncond}} + w(\epsilon\_{\text{cond}} - \epsilon\_{\text{uncond}})$

In CFG with a negative prompt: $\tilde{\epsilon} = \epsilon\_{\text{neg}} + w(\epsilon\_{\text{pos}} - \epsilon\_{\text{neg}})$

1. Show that when the negative prompt equals the null prompt ($y\_{\text{neg}} = \varnothing$), negative-prompt CFG reduces to standard CFG.

2. Show that the negative-prompt formula can be rewritten as:
   $$
   \tilde{\epsilon} = (1-w)\epsilon_{\text{neg}} + w\,\epsilon_{\text{pos}}
   $$
   What does this mean geometrically? Draw a diagram in 2D $\epsilon$-space.

3. Suppose $\epsilon\_{\text{pos}}$, $\epsilon\_{\text{neg}}$, and $\epsilon\_{\text{uncond}}$ are three distinct points in $\epsilon$-space. Compare the guidance directions (and magnitudes) for standard CFG vs. negative-prompt CFG. Under what conditions does the negative prompt produce a *different direction* (not just a different magnitude)?

### Part (b): Implement Negative Prompts

Using the class-conditional model from Problem 3, implement negative-prompt guidance:

```python
def cfg_sample_with_negative(model, pos_class, neg_class,
                              guidance_weight, num_steps=50):
    """
    CFG with negative prompt (negative class).
    eps_guided = eps_neg + w * (eps_pos - eps_neg)
    """
    pass
```

Experiment: generate digit "8" with different negative classes:
1. Negative = null (standard CFG)
2. Negative = "0" (avoid round shapes similar to 8?)
3. Negative = "3" (avoid the half-shape of 8?)
4. Negative = "1" (avoid something very different from 8)

Display samples for each. Does the negative class meaningfully affect the generated samples?

### Part (c): Guidance Direction Visualization

For the experiments in Part (b), compute and display the guidance direction $\Delta = \epsilon\_{\text{pos}} - \epsilon\_{\text{neg}}$ for each negative class. How do the guidance directions differ?

At a fixed noisy input $x\_t$ (at $t = 500$):
1. Compute $\epsilon\_\theta(x\_t, t, y)$ for all 10 classes plus null
2. Reduce to 2D using PCA on these 11 vectors
3. Plot the 11 points in 2D, labeled by class
4. Draw the guidance direction vectors for standard CFG and for negative-prompt CFG with different negative classes

This visualization shows the geometry of the guidance space. Are semantically similar digits close together in $\epsilon$-space?

---

## Submission Checklist

- [ ] Problem 1: Classifier guidance derivation, tempered posterior, geometric sketches
- [ ] Problem 2: Noisy classifier implementation, guided sampling, guidance strength analysis
- [ ] Problem 3: Classifier-free guidance implementation, comparison with classifier guidance, dropout rate experiment
- [ ] Problem 4: Diversity-fidelity curve, per-class analysis, guidance direction visualization
- [ ] Problem 5: Cross-attention implementation, cross-attention conditioned U-Net, attention map visualization
- [ ] Problem 6: Negative prompt derivation, implementation, guidance geometry visualization

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs. Include all generated images and plots.
