---
title: "Week 12: SAEs for Language Models"
---

# Week 12: SAEs for Language Models

## From toy models to the real thing.

For two weeks we have been building up to this moment. In Week 10, we mastered the mechanics of sparse autoencoders. In Week 11, we understood *why* we need them -- the superposition hypothesis tells us that neural networks pack many features into few dimensions, and SAEs can decompose them.

Now we apply SAEs to real language models and ask: *what have they learned?*

The answer turns out to be remarkable. When we train SAEs on the internal activations of language models like GPT-2 or Claude, we discover thousands of interpretable features -- concepts, patterns, and behaviors that the model has learned to represent. Some are mundane ("this text is in French"). Some are surprising ("this is a code error"). And some are profound ("this response involves deception").

This week draws heavily on two landmark papers from Anthropic:
- "Towards Monosemanticity" (Bricken et al., 2023) -- SAEs on a one-layer transformer
- "Scaling Monosemanticity" (Templeton et al., 2024) -- SAEs on Claude 3 Sonnet

---

## 1. Transformer Architecture Review

Before we can apply SAEs to transformers, we need to understand where to apply them. Let us briefly review the key components. If you studied transformers in a previous course, this is a refresher; if not, this section provides the minimum you need.

### 1.1 The Transformer Block

A transformer processes a sequence of tokens. At each layer $\ell$, each token has an activation vector $\mathbf{h}^{(\ell)} \in \mathbb{R}^d$ (the **residual stream** representation). A single transformer block applies two operations:

$$
\mathbf{h}^{(\ell + 0.5)} = \mathbf{h}^{(\ell)} + \text{Attention}^{(\ell)}(\mathbf{h}^{(\ell)})
$$
$$
\mathbf{h}^{(\ell + 1)} = \mathbf{h}^{(\ell + 0.5)} + \text{MLP}^{(\ell)}(\mathbf{h}^{(\ell + 0.5)})
$$

Note the **residual connections** (the $+$ signs). Each layer *adds* to the residual stream rather than replacing it. This is crucial: the residual stream is a "highway" that carries information across layers. Any layer can write to it, and any later layer can read from it.

### 1.2 The Attention Mechanism

The attention mechanism allows each token to "look at" other tokens in the sequence and aggregate information from them. For a single attention head $h$:

$$
\text{head}_h(\mathbf{h}) = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}}\right) \mathbf{V}_h
$$

where $\mathbf{Q}\_h, \mathbf{K}\_h, \mathbf{V}\_h$ are linear projections of the input (query, key, value). The attention output is the sum across all heads:

$$
\text{Attention}(\mathbf{h}) = \sum_h \mathbf{W}_O^{(h)} \text{head}_h(\mathbf{h})
$$

For our purposes, the key facts are:
- Attention moves information between token positions.
- Each attention head implements a specific "information routing" pattern.
- The output is added to the residual stream.

### 1.3 The MLP Layer

The MLP (multi-layer perceptron) layer is a position-wise feedforward network:

$$
\text{MLP}(\mathbf{h}) = \mathbf{W}_{\text{out}} \cdot \sigma(\mathbf{W}_{\text{in}} \mathbf{h} + \mathbf{b}_{\text{in}}) + \mathbf{b}_{\text{out}}
$$

where $\mathbf{W}\_{\text{in}} \in \mathbb{R}^{d\_{\text{ff}} \times d}$ expands the dimension (typically $d\_{\text{ff}} = 4d$), $\sigma$ is an activation function (ReLU or GELU), and $\mathbf{W}\_{\text{out}} \in \mathbb{R}^{d \times d\_{\text{ff}}}$ projects back down.

The MLP is where much of the model's "knowledge" is believed to be stored. It processes each token independently (no cross-token interaction) and adds its output to the residual stream.

### 1.4 The Residual Stream View

A useful perspective (due to Elhage et al., 2021): think of the residual stream as a shared communication channel. Each attention head and MLP layer reads from the residual stream, computes something, and writes the result back. The final token representation is the sum of contributions from all layers:

$$
\mathbf{h}^{(L)} = \mathbf{h}^{(0)} + \sum_{\ell=1}^{L} \left[\text{Attention}^{(\ell)}(\cdot) + \text{MLP}^{(\ell)}(\cdot)\right]
$$

This view clarifies where SAEs can be applied: at any point along the residual stream, we have a $d$-dimensional vector that encodes "everything the model knows about this token so far." We can train an SAE on these vectors to decompose the information into interpretable features.

---

## 2. Where to Apply SAEs

### 2.1 Possible Locations

There are several natural locations to apply SAEs in a transformer:

**Residual stream activations** $\mathbf{h}^{(\ell)}$ at layer $\ell$:
- These contain the cumulative representation after $\ell$ layers of processing.
- Features found here reflect everything the model has computed up to this point.
- This is the most common choice for interpretability work.

**MLP outputs** $\text{MLP}^{(\ell)}(\mathbf{h})$:
- These capture what the MLP at layer $\ell$ specifically contributes.
- Features found here are "things the MLP is writing to the residual stream."
- This was the focus of Anthropic's first SAE paper ("Towards Monosemanticity").

**Attention outputs** $\text{Attention}^{(\ell)}(\mathbf{h})$:
- These capture cross-token information movement at layer $\ell$.
- Features found here often relate to syntactic or positional patterns.

**MLP hidden activations** $\sigma(\mathbf{W}\_{\text{in}} \mathbf{h} + \mathbf{b}\_{\text{in}})$:
- The pre-output activations of the MLP.
- Already high-dimensional ($d\_{\text{ff}} = 4d$), so the expansion factor of the SAE can be smaller.

### 2.2 Practical Considerations

The choice of location depends on what you want to learn:

| Location | Dimension | What it captures | When to use |
|----------|-----------|-----------------|-------------|
| Residual stream | $d$ | Cumulative representation | General feature discovery |
| MLP output | $d$ | What this MLP adds | Understanding specific layers |
| Attention output | $d$ | Cross-token information | Understanding information flow |
| MLP hidden | $4d$ | MLP's internal features | Already partially decomposed |

A key tradeoff: residual stream activations contain *everything*, which makes them rich but potentially harder to decompose. MLP outputs contain only one layer's contribution, which is cleaner but misses features that are passed through from earlier layers.

---

## 3. Towards Monosemanticity (Anthropic, 2023)

### 3.1 The Setup

Bricken et al. (2023) trained SAEs on a one-layer transformer with 512-dimensional MLP activations. This is a small model by modern standards, but large enough to exhibit superposition and polysemanticity.

**Model:** A one-layer transformer trained on a large text corpus. The MLP has a hidden dimension of 512 (the dimension where SAEs are applied).

**SAE architecture:**
- Input dimension: 512
- Hidden dimension: 4096 (expansion factor 8x) and 131,072 (expansion factor 256x)
- L1 penalty with decoder norm constraint
- Trained on millions of MLP activation vectors from diverse text

### 3.2 Key Finding: Interpretable Features Emerge

The headline result: many SAE features are individually interpretable. When you look at the inputs that maximally activate a specific feature, they share a coherent semantic property.

Here are some example features from the paper (paraphrased):

**Feature: Arabic script**
- Maximally activating examples: text containing Arabic characters
- Activation pattern: activates strongly and specifically on Arabic text
- Ablation: removing this feature reduces the model's ability to predict Arabic tokens

**Feature: DNA sequences**
- Maximally activating examples: strings like "ATCGATCG..."
- Activation pattern: specifically detects nucleotide sequences
- The model has learned a concept of "DNA" even though no one explicitly taught it

**Feature: First person pronouns**
- Maximally activating examples: "I think...", "My experience...", "We believe..."
- Activation pattern: fires on first-person pronouns across diverse contexts

**Feature: Code with errors**
- Maximally activating examples: Python and JavaScript code with bugs
- This is remarkable: the feature does not just detect "code" but specifically "code that contains errors"

**Feature: Academic citations**
- Maximally activating examples: text like "(Smith et al., 2019)" or "[23]"
- Specific to citation formats, not just numbers or parentheses

**Feature: Expressions of uncertainty**
- Maximally activating examples: "I'm not sure...", "It might be...", "Perhaps..."
- Detects hedging language across different phrasings

### 3.3 The Spectrum of Feature Types

Features discovered by SAEs range across several dimensions:

**Specificity:** From very specific ("base64 encoded strings") to very general ("text in English").

**Abstraction level:** From low-level ("the letter 'q' followed by 'u'") to high-level ("sarcasm").

**Compositionality:** Some features are atomic ("is a number"), others seem to combine multiple properties ("a number in a mathematical equation, specifically an exponent").

The paper found that the majority of features in the smaller SAE (4096 features) were at least partially interpretable, with a significant fraction being clearly monosemantic. The larger SAE (131,072 features) found finer-grained features -- more specific sub-types of the broader features found by the smaller SAE.

### 3.4 Feature Splitting and Absorption

An important nuance: as SAE size increases, features do not simply become more numerous while staying the same. They undergo **splitting** and **absorption**:

**Splitting:** A feature in a smaller SAE (e.g., "code") may split into multiple features in a larger SAE ("Python code," "JavaScript code," "code comments"). The coarse-grained feature is decomposed into finer-grained variants.

**Absorption:** Sometimes a feature that was distinct in a smaller SAE gets absorbed into a more general feature in a larger SAE, or splits in unexpected ways.

This raises a philosophical question: what is the "right" granularity for features? Is "code" one feature or many? The answer likely depends on the context and the resolution of analysis.

---

## 4. Scaling Monosemanticity (Anthropic, 2024)

### 4.1 The Leap to Production Scale

Templeton et al. (2024) took the next step: applying SAEs to Claude 3 Sonnet, a production-scale language model. This is a dramatically larger model than the one-layer transformer in the 2023 paper.

**Model:** Claude 3 Sonnet (a model comparable in capability to GPT-4).

**SAE:**
- Trained on residual stream activations at a middle layer
- 1 million and 34 million feature variants
- Training required significant computational resources (but far less than training the model itself)

### 4.2 Key Findings

**Abstract features emerge at scale.** While the one-layer model had mostly concrete features (Arabic text, DNA sequences), the larger model produced highly abstract features:

**Feature: Deception / sycophancy**
- Activates on text where someone is being deceptive or saying what they think the listener wants to hear
- This is an abstract social concept, not a surface-level text pattern
- Incredibly relevant for AI safety: if we can detect when a model "thinks about" deception, we might detect deceptive behavior before it manifests in outputs

**Feature: Safety-relevant content**
- Features that activate on discussions of dangerous activities, bioweapons, etc.
- These features appear to be part of the model's safety training -- they are concepts the model has learned to be cautious about

**Feature: Multilingual concepts**
- Features that activate on the *same concept* expressed in multiple languages
- For example, a feature for "expressions of gratitude" that fires on "thank you" (English), "merci" (French), "gracias" (Spanish), etc.
- This suggests the model has learned language-independent conceptual representations

**Feature: The Golden Gate Bridge**
- A feature that activates specifically on references to the Golden Gate Bridge
- This feature became famous because of the dramatic feature steering results (see below)

### 4.3 The Golden Gate Bridge Experiment

The most viral result from the paper: by artificially amplifying the Golden Gate Bridge feature, the researchers created "Golden Gate Claude" -- a version of the model that would steer every conversation toward the Golden Gate Bridge.

**How it works:**
1. Identify the SAE feature for "Golden Gate Bridge."
2. During model inference, at the layer where the SAE was trained, add a multiple of this feature's direction to the residual stream:
   $$
   \mathbf{h}_{\text{steered}} = \mathbf{h} + \alpha \cdot \mathbf{W}_d[:, j_{\text{GGB}}]
   $$
   where $\alpha$ controls the steering strength.
3. The model continues processing with the modified activations.

**The result:** With sufficient $\alpha$, the model would:
- Claim to *be* the Golden Gate Bridge when asked "Who are you?"
- Relate any topic back to the Golden Gate Bridge
- Express "feelings" about fog, traffic, and being a bridge

This is simultaneously hilarious and profound. It demonstrates that individual SAE features are causally meaningful: they are not just correlates of the model's computations but active participants in them. Amplifying a single feature dramatically and specifically changes the model's behavior.

### 4.4 Feature Clamping and Behavioral Control

More seriously, the ability to clamp features enables targeted behavioral interventions:

**Safety clamping:** If a feature corresponds to "willingness to discuss dangerous topics," clamping it to zero might make the model refuse such discussions.

**Debiasing:** If a feature corresponds to a gender or racial bias, reducing its activation might reduce biased outputs.

**Capability enhancement:** If a feature corresponds to "mathematical reasoning," boosting it might improve the model's math performance.

These possibilities are speculative and require careful validation, but they illustrate why SAE-based interpretability is not just an academic exercise -- it could eventually provide fine-grained control over model behavior.

---

## 5. Training SAEs on Language Models: A Practical Guide

### 5.1 The Toolchain

Several libraries make it feasible to train SAEs on language models without building everything from scratch:

**TransformerLens** (Neel Nanda): A library for mechanistic interpretability of transformers. It provides:
- Easy loading of pre-trained models (GPT-2, Pythia, etc.)
- Hook-based access to any internal activation
- Clean API for patching and ablation experiments

**SAELens** (Joseph Bloom et al.): A library specifically for training and analyzing SAEs on language models. Built on top of TransformerLens. It provides:
- Pre-built SAE architectures (vanilla, Gated, TopK)
- Training pipelines with logging
- Pre-trained SAE checkpoints for several models
- Feature analysis tools

**Neuronpedia** (Joseph Bloom): A web platform for exploring SAE features interactively. Contains pre-trained SAEs for GPT-2 and other models, with feature dashboards showing max-activating examples, activation histograms, and more.

### 5.2 Extracting Activations

The first step is to collect activation vectors from the model. Using TransformerLens:

```python
# Pseudocode (for illustration -- see homework for working code)
import transformer_lens as tl

model = tl.HookedTransformer.from_pretrained("gpt2-small")

# Run model on text and cache all activations
logits, cache = model.run_with_cache("The cat sat on the mat")

# Extract residual stream at layer 6
# Shape: (batch, seq_len, d_model) where d_model = 768 for GPT-2 small
h = cache["resid_post", 6]
```

For training an SAE, you need millions of activation vectors. The typical approach:
1. Stream text from a large corpus (e.g., OpenWebText, The Pile).
2. For each batch of text, run the model and extract activations at the desired layer.
3. Reshape to individual token activations: each token produces one vector $\mathbf{h} \in \mathbb{R}^{d}$.
4. Feed these vectors to the SAE training loop.

### 5.3 Architecture Choices

**Expansion factor** $r = d\_{\text{SAE}} / d\_{\text{model}}$:
- $r = 4$: Discovers coarse-grained features. Good for initial exploration.
- $r = 8$: Standard choice. Balances feature granularity and training cost.
- $r = 16$: Finer-grained features. More feature splitting.
- $r = 32$ or higher: Very fine-grained. Used in large-scale studies.

For GPT-2 small ($d = 768$):
- $r = 4$ gives 3,072 SAE features
- $r = 8$ gives 6,144 features
- $r = 16$ gives 12,288 features

**Sparsity coefficient $\lambda$:** Needs to be tuned for each model and layer. A common approach: target a specific L0 (e.g., 50 active features per token) and adjust $\lambda$ accordingly. Typical values: $\lambda \in [0.001, 0.1]$.

**Training data:** Use tokens from the model's training distribution or a similar distribution. Using out-of-distribution text will produce activations the model was not trained on, leading to poor SAE features.

**Number of training tokens:** Depends on model size and SAE size. For GPT-2 small with an 8x SAE, 100M-500M tokens is typical. Larger models and larger SAEs need more data.

### 5.4 Evaluation Metrics

For SAEs trained on language model activations, the standard metrics are:

**CE loss recovered** (the most important metric):
$$
\text{CE recovered} = 1 - \frac{L_{\text{SAE}} - L_{\text{orig}}}{L_{\text{zero}} - L_{\text{orig}}}
$$

where $L\_{\text{SAE}}$ is the cross-entropy loss of the model when the SAE's reconstructed activations replace the original activations, $L\_{\text{orig}}$ is the original model's loss, and $L\_{\text{zero}}$ is the loss when activations are replaced with zeros.

Good SAEs achieve CE recovered > 0.95, meaning they preserve almost all of the model's predictive ability.

**L0 (sparsity):** Average number of active features per token. Typical targets: 20-100.

**Dead feature fraction:** What fraction of SAE features never activate? Should be as low as possible (< 10%).

**Feature interpretability:** Manually or automatically assessed. What fraction of features have a clear, coherent meaning?

---

## 6. The Feature Analysis Pipeline

Once you have a trained SAE, the real work begins: understanding what the features mean.

### 6.1 Finding Maximally Activating Examples

For each feature $j$, find the input tokens (in context) that produce the highest activation $z\_j$:

1. Run the model on a large, diverse corpus.
2. For each token, compute the SAE hidden activations.
3. For feature $j$, record the top-$k$ tokens by activation strength, along with their surrounding context.

These max-activating examples are the primary evidence for what a feature "means."

**Example:** If feature 2847's top-10 activating tokens are all instances of the word "however" at the beginning of a sentence, that feature likely detects the discourse marker "however" in a sentence-initial position.

### 6.2 Activation Distributions

For each feature, compute its activation distribution across a large corpus:
- **Activation frequency:** What fraction of tokens activate this feature (activation > 0)?
- **Activation magnitude distribution:** When the feature is active, how strong are its activations? A narrow, peaked distribution suggests a "binary" feature (either on or off). A wide distribution suggests a graded feature.
- **Context patterns:** Does the feature activate on specific token positions? (E.g., always on the first token, always on the last token of a word, etc.)

### 6.3 Feature Specificity Testing

Max-activating examples tell you what a feature responds to, but they do not tell you what it does *not* respond to. For rigorous interpretability, you need **specificity tests**:

1. Form a hypothesis about the feature's meaning (e.g., "this feature detects cooking recipes").
2. Construct positive examples (text that should activate the feature) and negative examples (text that should not).
3. Run both through the model and SAE.
4. Check: does the feature activate on all positives and none of the negatives?

This is analogous to hypothesis testing in science. The max-activating examples generate the hypothesis; the specificity tests confirm it.

### 6.4 Ablation and Patching

As discussed in Week 11, causal methods verify that features are not just correlated with a concept but actively involved in processing it:

**Zero ablation:** Set the feature to zero and measure the change in model output.
- For a "French text" feature: does ablation reduce the model's ability to predict French words?
- For a "code error" feature: does ablation prevent the model from detecting bugs?

**Feature steering:** Amplify the feature and observe behavior changes.
- For a "formal writing" feature: does amplification make outputs more formal?
- For a "safety" feature: does amplification make the model more cautious?

### 6.5 Automated Interpretability

With thousands or millions of features, manual analysis is infeasible. **Automated interpretability** uses another language model to describe features:

1. Collect the top-$k$ maximally activating examples for a feature.
2. Present them to a language model (e.g., Claude or GPT-4) with a prompt like: "Here are the text passages that most strongly activate a specific neuron in a language model. What concept or pattern do they have in common?"
3. The describing model produces a natural language description.
4. Optionally, use the description to generate new test examples and verify the description's accuracy.

Anthropic has used this approach at scale, generating descriptions for hundreds of thousands of features. The descriptions are imperfect -- the describing model can hallucinate patterns, miss subtle commonalities, or be misled by superficial correlations -- but they provide a useful starting point for exploration.

A more sophisticated approach: have the describing model predict which new examples will activate the feature (based on its description), then check whether those predictions are correct. The accuracy of these predictions measures the quality of the description.

---

## 7. Open Questions and Limitations

SAE-based interpretability is a young field, and many fundamental questions remain open.

### 7.1 Are SAE Features the "True" Features?

The deepest question: do SAE features correspond to the features that the neural network "actually uses," or are they an artifact of the SAE training process?

**Arguments for "yes":**
- SAE features are often interpretable (they correspond to human-recognizable concepts).
- They are causally relevant (ablation and steering change model behavior predictably).
- Different random seeds produce similar features, suggesting the decomposition is not arbitrary.
- The toy model results (Week 11) show that SAEs can recover ground-truth features from superposition.

**Arguments for "maybe not":**
- We have no ground truth for what the "real" features of a large language model are.
- The SAE imposes specific assumptions (linearity, L1 sparsity) that may not match the network's actual structure.
- Different SAE architectures (L1 vs. TopK vs. Gated) find somewhat different features. Which is "right"?
- Feature splitting suggests that the granularity of features depends on the SAE size, not just the underlying model.

The honest answer: SAE features are *a* decomposition of the model's representations that is often useful and interpretable, but we cannot be sure it is *the* decomposition. This is an active area of research.

### 7.2 Faithfulness

A related concern: does replacing the model's activations with SAE-reconstructed activations preserve the model's behavior? If the answer is "mostly yes" (high CE recovered), then the SAE's decomposition is at least *consistent* with the model's computations.

But "mostly" is doing heavy lifting. A 95% CE recovered score means 5% of the information is lost. What is in that 5%? It might be noise, or it might be critical features that the SAE failed to capture.

### 7.3 Scaling Challenges

Training SAEs on large models is computationally expensive:
- Extracting activations requires running the model on millions of inputs.
- Training the SAE itself requires processing millions of high-dimensional vectors.
- Analysis (max-activating examples, ablation, automated interpretability) must be done for thousands of features.

The total cost of a full SAE analysis can be a significant fraction of the cost of training the model itself. This limits SAE-based interpretability to well-funded research labs (for now).

### 7.4 The Feature Completeness Problem

Even a large SAE may not capture all of the model's features. Some features might be:
- Too rare to be learned (the SAE never sees enough examples)
- Too nonlinear to be captured by a linear decomposition
- Distributed across multiple SAE features in complex ways

How do we know when we have found "all" the features? We do not, and this is a fundamental limitation.

### 7.5 Beyond Features: Circuits and Algorithms

Identifying individual features is only the beginning. The deeper goal is to understand *circuits* -- how features interact across layers to implement algorithms. Current SAE-based methods are good at finding features but have limited tools for tracing circuits.

Some promising directions:
- Training SAEs at every layer and studying how features at layer $\ell$ relate to features at layer $\ell + 1$.
- Using causal methods (activation patching) to trace the flow of information through features across layers.
- Combining SAEs with attention pattern analysis to understand cross-token feature interactions.

---

## 8. The State of the Art and Future Directions

### 8.1 Current Best Practices

As of 2024-2025, the state of the art for SAE-based interpretability involves:

1. **Gated SAEs or TopK SAEs** rather than vanilla L1 SAEs, for better Pareto frontiers (more sparsity for the same reconstruction quality).
2. **Multiple expansion factors** to study features at different granularities.
3. **Automated interpretability pipelines** using language models to describe and validate features at scale.
4. **Causal validation** (ablation, steering) as a standard part of the analysis, not an afterthought.
5. **Public tooling** (TransformerLens, SAELens, Neuronpedia) making the field accessible to researchers outside of major labs.

### 8.2 What Comes Next

Several exciting directions are actively being pursued:

**Circuit-level analysis:** Moving beyond individual features to understand how features compose into algorithms across layers.

**Real-time interpretability:** Using SAEs as a monitoring tool during model inference, detecting when safety-relevant features activate.

**Feature-based model editing:** Using SAE features to precisely edit a model's knowledge or behavior, as an alternative to retraining.

**Cross-model universality:** Do different models learn the same features? If so, there might be a "universal" vocabulary of neural network features.

**Theoretical foundations:** Why does superposition have the geometric structure it does? Can we predict which features a model will learn? Is there a principled way to determine the "right" number of SAE features?

---

## 9. A Complete Workflow: From Model to Understanding

Let us tie everything together with a concrete workflow for analyzing a language model using SAEs.

### Step 1: Choose your model and layer

Select a model (e.g., GPT-2 small) and a layer to analyze (e.g., residual stream at layer 8). Layer 8 is a reasonable middle-layer choice for GPT-2 small (which has 12 layers), as it has had enough layers to form interesting features but is not yet committed to specific output tokens.

### Step 2: Extract activations

Using TransformerLens, run the model on a large corpus and collect residual stream activations at layer 8. Aim for at least 10 million activation vectors (each $\in \mathbb{R}^{768}$).

### Step 3: Train the SAE

Train an SAE with expansion factor 8 (hidden dim = 6,144) and L1 penalty. Use decoder norm constraint. Train for 100,000+ steps with Adam.

Monitor: total loss, reconstruction loss, L1 loss, average L0, dead feature fraction.

### Step 4: Evaluate reconstruction quality

Compute CE loss recovered on a held-out set of text. Target > 0.95.

### Step 5: Analyze features

For each of the 6,144 features:
1. Find the top-20 maximally activating examples.
2. Compute the activation frequency and magnitude distribution.
3. (Optional) Use automated interpretability to generate a description.

### Step 6: Deep-dive on interesting features

Select 10-20 features that seem particularly interesting (highly specific, safety-relevant, or surprising). For each:
1. Manually verify interpretability by examining many activating and non-activating examples.
2. Perform ablation: zero out the feature and measure the change in model predictions.
3. Perform steering: amplify the feature and observe the change in model outputs.

### Step 7: Report findings

Document each interesting feature with:
- A name/description
- Representative max-activating examples
- Activation statistics
- Ablation results
- Steering results
- Confidence assessment (how sure are you this feature is what you think it is?)

---

## 10. Summary

| Concept | Key Idea |
|---------|----------|
| Transformer residual stream | The "highway" where information accumulates across layers |
| Where to apply SAEs | Residual stream, MLP outputs, attention outputs -- each captures different features |
| Towards Monosemanticity | SAEs on a 1-layer transformer find interpretable features (Arabic, DNA, code errors) |
| Scaling Monosemanticity | SAEs on Claude 3 Sonnet find abstract features (deception, safety, multilingual concepts) |
| Golden Gate Bridge | Feature steering dramatically changes model behavior -- proof of causal relevance |
| Training pipeline | Extract activations, train SAE, evaluate reconstruction + sparsity + interpretability |
| Feature analysis | Max-activating examples, specificity tests, ablation, steering, automated interpretability |
| Open questions | Faithfulness, feature completeness, scaling, true features vs. SAE artifacts |

---

## Further Reading

- **Bricken et al.**, "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning" (Anthropic, 2023). The paper that demonstrated SAE-based interpretability on a real language model. Beautifully written with detailed feature analyses.

- **Templeton et al.**, "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" (Anthropic, 2024). The scaling follow-up. The Golden Gate Bridge results are in this paper.

- **Cunningham et al.**, "Sparse Autoencoders Find Highly Interpretable Features in Language Models" (2023). A concurrent and complementary paper to Anthropic's work.

- **Bloom, Joseph**, "SAELens documentation." Practical guide to training and analyzing SAEs with the SAELens library. Essential for the homework.

- **Neuronpedia** (neuronpedia.org). Interactive exploration of SAE features for GPT-2 and other models. Spend some time browsing -- it is the best way to build intuition for what SAE features look like.

- **Neel Nanda**, "TransformerLens documentation." The standard library for mechanistic interpretability experiments.

---

*This concludes the core arc of the course. In Week 13, we will survey advanced SAE architectures (TopK, Gated, JumpReLU), scaling laws, and the open frontier of research. But the ideas from Weeks 10-12 are the foundation: sparse autoencoders decompose neural network representations into interpretable features, and those features are the key to understanding -- and eventually controlling -- what neural networks do.*
