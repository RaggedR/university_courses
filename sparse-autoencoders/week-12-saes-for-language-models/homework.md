# Week 12: SAEs for Language Models -- Homework

## Overview

This problem set takes you from extracting activations from a real language model through to analyzing SAE features with causal methods. You will work with GPT-2 small and pre-trained SAEs, rather than training everything from scratch (training a good SAE on a language model takes significant compute; the goal here is to develop your analysis skills).

**Tools:** Python, PyTorch, TransformerLens, SAELens (or equivalent), matplotlib, numpy.

**Setup:** Install the required libraries:
```bash
pip install transformer-lens sae-lens einops fancy-einsum
```

If SAELens is unavailable or has compatibility issues, you can load pre-trained SAE weights manually (see the SAELens documentation or Neuronpedia for download links). The key operations (extract activations, apply encoder, decode) can be implemented in a few lines of PyTorch.

**Estimated time:** 10-14 hours.

---

## Problem 1: Exploring Transformer Activations (Warm-up)

Before applying SAEs, build familiarity with the activation space of a real language model.

### 1a. Load GPT-2 Small

Using TransformerLens, load GPT-2 small and run it on the following prompts:

```python
prompts = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "In the year 1969, humans first",
    "The quick brown fox jumps over the",
    "According to quantum mechanics, particles can",
]
```

For each prompt, extract the residual stream activations at layers 0, 4, 8, and 11 for the **last token** (the token the model is about to predict from).

### 1b. Dimensionality and Magnitude

For the extracted activations:
1. Report the dimension of the residual stream ($d_{\text{model}}$).
2. Compute the L2 norm of the activation vector at each layer for each prompt. How does the magnitude change across layers?
3. Compute the cosine similarity between the layer 8 activations of all pairs of prompts. Which prompts are most similar in representation space? Does this make intuitive sense?

### 1c. PCA of Activations

Collect residual stream activations at layer 8 for 1,000 different text passages (you can use a dataset like OpenWebText or simply generate prompts). Take the last-token activation from each.

1. Apply PCA to these 1,000 vectors.
2. Plot the explained variance ratio for the first 50 principal components.
3. How many dimensions capture 90% of the variance? What does this tell you about the "effective dimensionality" of the residual stream?

### 1d. Discussion

In 3-5 sentences, explain why PCA is insufficient for understanding the features in the residual stream, even if it captures most of the variance. (Hint: think about superposition and the relationship between variance and feature importance.)

---

## Problem 2: Working with Pre-trained SAEs (Core)

Load a pre-trained SAE for GPT-2 small and explore its features.

### 2a. Load the SAE

Using SAELens (or by downloading weights from Neuronpedia), load a pre-trained SAE for GPT-2 small's residual stream at a middle layer (layer 8 is recommended). Note the expansion factor and the number of features.

Verify the SAE works by:
1. Running a prompt through GPT-2 and extracting layer 8 residual stream activations.
2. Passing these activations through the SAE encoder to get sparse feature activations $\mathbf{z}$.
3. Reconstructing using the decoder: $\hat{\mathbf{h}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_d$.
4. Computing the reconstruction error (MSE and cosine similarity between $\mathbf{h}$ and $\hat{\mathbf{h}}$).

### 2b. Feature Exploration: Max-Activating Examples

Select 5 features at random (or choose features with interesting activation patterns). For each feature:

1. Run 100+ diverse text passages through the model and SAE.
2. For each token in each passage, record the feature's activation strength.
3. Find the top-10 tokens by activation strength. Record each token along with its surrounding context (at least 10 tokens before and after).
4. Examine the top-10 list. Can you identify what the feature detects?

Give each feature a short name/description based on your analysis (e.g., "conjunctions at sentence boundaries" or "proper nouns in news articles").

### 2c. Feature Activation Statistics

For the same 5 features, compute and report:
1. **Activation frequency:** What fraction of tokens activate this feature (activation > 0)?
2. **Mean activation** (when active): What is the average activation when the feature fires?
3. **Max activation:** What is the strongest activation observed?
4. **Activation histogram:** Plot a histogram of non-zero activation values for each feature.

### 2d. Dead Features

Across all features in the SAE, what fraction are "dead" (never activate on your 100+ passages)? How does this compare to the dead feature fraction reported in the literature (typically 5-30% depending on training)?

---

## Problem 3: Feature Ablation Experiment (Causal)

Test whether an SAE feature is causally relevant for model behavior.

### 3a. Select a Feature

From your analysis in Problem 2, select a feature that you are most confident about (i.e., you have a clear hypothesis about what it detects). Write down:
- The feature index.
- Your hypothesis about what it detects.
- Three example tokens/contexts where it strongly activates.

### 3b. Design Test Cases

Construct two sets of test inputs:
- **Positive set** (10 inputs): Text where you expect the feature to activate strongly.
- **Negative set** (10 inputs): Text where you expect the feature to NOT activate, but that is otherwise similar to the positive set (controls for confounds).

Verify that the feature indeed activates on the positive set and not on the negative set.

### 3c. Zero Ablation

For each input in the positive set:
1. Run the model normally and record the top-5 predicted next tokens and their probabilities.
2. Run the model with the selected feature's activation zeroed out:
   - Extract the residual stream at layer 8.
   - Pass through the SAE encoder to get $\mathbf{z}$.
   - Set the selected feature's activation to zero: $z_j \leftarrow 0$.
   - Reconstruct: $\hat{\mathbf{h}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_d$.
   - Replace the original activation with $\hat{\mathbf{h}}$ and continue the forward pass.
3. Record the new top-5 predicted tokens and their probabilities.

Compare the results. Does zeroing out the feature change the predictions in a way consistent with your hypothesis?

### 3d. Quantitative Measurement

Compute the KL divergence between the original and ablated output distributions:

$$D_{\text{KL}}(p_{\text{orig}} \| p_{\text{ablated}})$$

where $p_{\text{orig}}$ and $p_{\text{ablated}}$ are the full next-token probability distributions.

Report the mean KL divergence across your positive set and your negative set. If the feature is causally relevant for the positive set, KL divergence should be higher for positive examples than negative examples.

---

## Problem 4: Feature Steering (Causal)

Artificially boost a feature and observe how model behavior changes.

### 4a. Choose a Feature for Steering

Select a feature (the same one from Problem 3, or a different one if you find one better suited to steering). The ideal feature for this problem is one that, when amplified, would produce a clearly observable change in the model's text generation.

### 4b. Steering Implementation

Implement feature steering:
1. During model inference, at layer 8, modify the residual stream:
   $$\mathbf{h}_{\text{steered}} = \mathbf{h} + \alpha \cdot \mathbf{d}_j$$
   where $\mathbf{d}_j = \mathbf{W}_d[:, j]$ is the decoder column for feature $j$, and $\alpha$ is the steering strength.
2. Continue the forward pass with the modified activations.

### 4c. Steering Experiment

Choose a neutral prompt (one where the feature would not normally be active). For example, if your feature detects "formal academic writing," choose a casual, informal prompt.

Generate 50 tokens of text from this prompt at the following steering strengths:
- $\alpha = 0$ (baseline, no steering)
- $\alpha = 5$
- $\alpha = 20$
- $\alpha = 50$

Record the generated text for each $\alpha$.

### 4d. Analysis

1. At what steering strength does the feature's effect become noticeable?
2. At what strength does the generation become incoherent?
3. Is the behavioral change consistent with your hypothesis about the feature's meaning?
4. Compute the perplexity of the generated text at each steering strength. How does coherence degrade with increasing $\alpha$?

---

## Problem 5: Expansion Factor Comparison (Experimental)

Compare SAEs with different expansion factors to understand the effect of SAE capacity on feature granularity.

### 5a. Load Two SAEs

If available, load pre-trained SAEs for GPT-2 small at the same layer but with different expansion factors (e.g., 4x and 16x). If pre-trained SAEs at different expansion factors are not available, you may:
- Train two small SAEs yourself (with expansion factors 2x and 8x) for a shorter training run.
- Or use SAEs from different layers as a proxy for different capacities.

### 5b. Feature Comparison

Choose a concept that is clearly represented in both SAEs (e.g., "numbers" or "Python code" or "negation words"). For each SAE:
1. Find the feature(s) that correspond to this concept.
2. Examine their max-activating examples.

### 5c. Splitting Analysis

Does the larger SAE **split** the concept into finer-grained sub-features? For example:
- Does a "numbers" feature in the 4x SAE split into "small numbers," "large numbers," "decimal numbers," and "numbers in equations" in the 16x SAE?
- Does a "code" feature split into "Python code," "JavaScript code," "code comments"?

Document any splitting you observe with specific examples.

### 5d. Quantitative Comparison

Compare the two SAEs on:
1. **Reconstruction quality** (MSE and CE loss recovered).
2. **Average L0** (features active per token).
3. **Dead feature fraction.**
4. **Feature density** (average activation frequency across features).

Which SAE offers a better trade-off between interpretability and faithfulness?

---

## Problem 6: Feature Report (Analysis and Writing)

Write a detailed analysis of one feature you have discovered and validated.

### Requirements

Write a 1-2 page report (approximately 500-1000 words) about one specific SAE feature. Your report should include:

### 6a. Feature Identification
- Feature index and which SAE/model/layer it comes from.
- The top-10 maximally activating examples, with surrounding context.

### 6b. Hypothesis
- What concept or pattern does this feature represent?
- How confident are you? What is the evidence?
- Are there any ambiguous or surprising activating examples?

### 6c. Specificity
- Examples where the feature activates as expected (true positives).
- Examples where the feature does NOT activate despite seeming relevant (false negatives), if any.
- Examples where the feature activates unexpectedly (false positives), if any.

### 6d. Causal Evidence
- Ablation results: what changes when you remove this feature?
- Steering results (if applicable): what happens when you amplify it?

### 6e. Interpretation
- What does this feature tell us about how GPT-2 processes text?
- Is this feature "surprising" -- something you would not have predicted a language model would learn?
- How does this feature relate to the superposition hypothesis (does it seem to occupy its own dimension, or does it share space with other features)?

### 6f. Limitations
- What are you uncertain about?
- What additional experiments would increase your confidence?

---

## Submission Checklist

- [ ] Problem 1: Transformer activations -- dimension/magnitude analysis, PCA plot, similarity matrix, discussion
- [ ] Problem 2: Pre-trained SAE exploration -- 5 features with max-activating examples, activation statistics, dead feature analysis
- [ ] Problem 3: Feature ablation -- test design, ablation results, KL divergence comparison
- [ ] Problem 4: Feature steering -- generated text at multiple strengths, coherence analysis
- [ ] Problem 5: Expansion factor comparison -- feature splitting documentation, quantitative comparison
- [ ] Problem 6: Feature report -- 1-2 page written analysis of one feature

---

## Tips and Common Pitfalls

1. **Memory management:** GPT-2 small fits comfortably on a single GPU or even a CPU, but collecting millions of activations can fill up RAM. Process text in batches and save activations to disk if needed.

2. **Token vs. word:** Remember that language models operate on tokens, not words. A single word might be split into multiple tokens (e.g., "understanding" might become "under" + "standing"). Features activate on tokens, so pay attention to which specific token in a word activates the feature.

3. **Context matters:** A feature's activation depends on the full context, not just the token it fires on. The same word in different contexts may activate different features. Always examine features in their full context.

4. **Activation patching is tricky:** When you replace activations at layer 8, the remaining layers process the modified activations. This is the intended behavior but can produce unexpected results if the modification is too large (the later layers receive out-of-distribution inputs).

5. **Use Neuronpedia:** Before spending hours analyzing a feature, check if it has already been described on Neuronpedia. You can build on existing analyses rather than starting from scratch. (But do verify independently -- the automated descriptions are not always correct.)

6. **Be honest about uncertainty.** It is better to say "I think this feature detects X, but I am only 60% confident because of these ambiguous examples" than to claim certainty you do not have. Interpretability is hard, and honest uncertainty is scientifically valuable.
