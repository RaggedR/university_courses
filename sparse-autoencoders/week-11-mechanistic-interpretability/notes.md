# Week 11: Mechanistic Interpretability

## Why should we care what neural networks think?

Last week we built sparse autoencoders -- overcomplete autoencoders with sparsity constraints that learn interpretable, dictionary-like features. This week, we ask the motivating question: *why would anyone apply an SAE to the internals of a neural network?*

The answer lies in a problem that has haunted AI since deep learning took off: we can build systems that perform remarkably well, but we have very little idea how they work. A language model can write poetry, solve math problems, and carry on a conversation -- but if you ask "which internal computations lead to this output?", the honest answer is usually "we do not know."

Mechanistic interpretability is the field that tries to change this. It aims to reverse-engineer neural networks: to identify the algorithms and representations they use internally, not just measure their behavior from the outside. And sparse autoencoders have emerged as one of the most promising tools in this effort.

---

## 1. The Interpretability Problem

### 1.1 The Stakes

Why do we want to understand neural networks? Several reasons, ranging from practical to philosophical:

**Safety.** If a neural network is making high-stakes decisions (medical diagnosis, autonomous driving, content moderation), we need to know whether its reasoning is sound. A model might achieve 99% accuracy by learning the right patterns -- or by learning a spurious shortcut that will fail catastrophically in deployment.

**Trust.** Humans (rightly) struggle to trust systems they cannot understand. A doctor will not defer to an AI diagnosis if the AI cannot explain its reasoning. Interpretability is a prerequisite for appropriate trust.

**Scientific understanding.** Neural networks are trained on vast amounts of human-generated data. If they learn good representations of language, vision, or reasoning, those representations might teach us something about the structure of the problems themselves. What concepts does a language model form? How does it organize knowledge?

**Debugging.** When a model fails, interpretability helps us understand *why*. Without it, we are reduced to trial-and-error: change the data, change the architecture, retrain, and hope for the better. With interpretability, we can diagnose the specific computation that went wrong.

**Alignment.** As AI systems become more capable, ensuring they pursue intended goals becomes critical. If we can inspect a model's internal representations, we might detect misaligned objectives before they cause harm. Does this model have a concept of "deception"? Does it ever activate it?

### 1.2 Levels of Interpretability

Interpretability is not a single thing. It operates at different levels:

**Behavioral interpretability:** Study the model as a black box. What outputs does it produce for various inputs? This includes techniques like:
- Probing classifiers (train a small model on internal activations to predict a property)
- Behavioral testing (check model outputs on carefully designed inputs)
- Attention visualization (look at attention patterns -- though this has known limitations)

**Mechanistic interpretability:** Open the black box. Identify the specific computations, features, and circuits that produce the model's behavior. This is harder but more informative:
- What features does each neuron or direction in activation space represent?
- How do features interact through the network's layers?
- Can we identify "circuits" -- subnetworks that implement specific algorithms?

This week focuses on the mechanistic approach.

---

## 2. Features in Neural Networks

### 2.1 What Is a "Feature"?

We use the word "feature" casually, but it deserves careful definition. In the context of neural network interpretability, a **feature** is a property of the input that the network represents internally and uses for computation.

For example, in a language model processing the sentence "The cat sat on the mat":
- "The current token is a noun" might be a feature
- "The subject is an animal" might be a feature
- "This is a rhyming sentence" might be a feature
- "The grammatical structure is Subject-Verb-Preposition-Object" might be a feature

The critical question is: *where* in the network are these features represented, and *how*?

### 2.2 The Linear Representation Hypothesis

A foundational claim in mechanistic interpretability:

> **The Linear Representation Hypothesis:** Features in neural networks are represented as directions in activation space.

What does this mean concretely? Suppose we are looking at the residual stream of a transformer at some layer, which produces a vector $\mathbf{h} \in \mathbb{R}^d$ for each token. The hypothesis says that there exists a direction $\mathbf{v}\_{\text{cat}} \in \mathbb{R}^d$ such that the dot product $\mathbf{h} \cdot \mathbf{v}\_{\text{cat}}$ measures "how much the model thinks this token is about cats."

More precisely: the activation of a feature is a linear function of the representation vector. This is a strong and testable claim. There is mounting evidence for it:

- Word embeddings (Word2Vec, GloVe) exhibit linear structure: the vector for "king" minus "man" plus "woman" gives approximately "queen."
- Probing classifiers that are linear (logistic regression on activations) can often detect semantic properties with high accuracy, suggesting those properties are linearly accessible.
- Activation patching experiments show that adding or subtracting specific directions from activation vectors changes model behavior in predictable, targeted ways.

The linear representation hypothesis does not claim that *all* features are linear, or that linearity is *exact*. But it is a useful approximation that has proven surprisingly effective.

### 2.3 Neurons vs. Directions

If features are directions, a natural question: are those directions aligned with the *coordinate axes* of the activation space (i.e., individual neurons)?

In an idealized world, each neuron would represent one feature. Neuron 42 fires when the input is about cats, neuron 137 fires for plural nouns, and so on. This is the **monosemantic** ideal -- each neuron has one meaning.

In practice, this is often not the case. Individual neurons frequently respond to multiple unrelated concepts. A single neuron might activate for both cats and cars, or for both the number 7 and the color blue. This is the **polysemantic** reality.

Why? The answer, it turns out, is superposition.

---

## 3. The Superposition Hypothesis

### 3.1 The Core Idea

The **superposition hypothesis** (Elhage et al., 2022) proposes:

> Neural networks represent more features than they have dimensions by encoding features as nearly-orthogonal directions in activation space. This works when features are sparse (rarely active simultaneously).

Let us build intuition for this claim step by step.

### 3.2 The Geometric Picture

Consider a vector space $\mathbb{R}^d$ with $d$ dimensions. If we want to store $n$ features as directions in this space, how large can $n$ be?

**Without superposition ($n \leq d$):** We can store up to $d$ features as orthogonal directions (the coordinate axes, or any orthonormal basis). Each feature gets its own dimension, there is no interference, and we can read off any feature's activation perfectly.

**With superposition ($n > d$):** We can fit *more* than $d$ directions in $\mathbb{R}^d$ -- they just cannot all be orthogonal. But they can be *nearly* orthogonal. How nearly? This is where the geometry of high-dimensional spaces becomes remarkable.

**The Johnson-Lindenstrauss phenomenon:** In high-dimensional spaces, random vectors are nearly orthogonal. Specifically, if you draw $n$ random unit vectors in $\mathbb{R}^d$, the expected inner product between any two of them is 0, and the typical magnitude of their inner product is approximately $1/\sqrt{d}$.

For $d = 768$ (the hidden dimension of GPT-2 small), two random unit vectors will have an inner product of roughly $\pm 0.036$ -- nearly orthogonal. This means you can pack many more than 768 nearly-orthogonal directions into 768 dimensions.

The exact limit depends on how much interference you can tolerate, but the number of nearly-orthogonal vectors in $\mathbb{R}^d$ grows *exponentially* with $d$ (for a fixed tolerance on inner products). A 768-dimensional space can, in principle, represent thousands or millions of nearly-orthogonal features.

### 3.3 Why Sparsity Makes Superposition Work

Having nearly-orthogonal directions is not enough on its own. If *all* features are active simultaneously, the interference between them accumulates and the representation becomes noisy.

But if features are **sparse** -- only a small fraction are active for any given input -- then the interference is manageable. Here is the key insight:

Consider two features $\mathbf{v}\_A$ and $\mathbf{v}\_B$ stored as nearly-orthogonal directions with inner product $\epsilon = \mathbf{v}\_A \cdot \mathbf{v}\_B \approx 0.03$. If feature A is active with magnitude $a$ and feature B is active with magnitude $b$, the total activation vector is:

$$
\mathbf{h} = a \mathbf{v}_A + b \mathbf{v}_B
$$

When we try to read out feature A's activation, we get:

$$
\mathbf{h} \cdot \mathbf{v}_A = a \|\mathbf{v}_A\|^2 + b (\mathbf{v}_B \cdot \mathbf{v}_A) = a + b\epsilon
$$

The term $b\epsilon$ is interference from feature B. If features A and B are rarely active simultaneously (sparse), this interference is usually zero (because $b = 0$ most of the time). When it does occur, it is small (because $\epsilon$ is small).

This is exactly the situation where overcomplete sparse representations shine -- and exactly the situation where SAEs can help recover the individual features.

### 3.4 A Party Analogy

Imagine a party in a small room:

**Without superposition:** Each person gets their own designated corner. The room can hold $d$ people. No one talks over anyone else. But the room is small and the guest list is limited.

**With superposition and dense features:** You invite $2d$ people and everyone talks at once. The room is cacophonous. You cannot understand any individual conversation -- total interference.

**With superposition and sparse features:** You invite $2d$ people but most of them are quiet most of the time. At any given moment, only a few people are talking. You can follow each conversation because the speakers are spread out in the room and the quiet majority does not interfere.

The neural network is the room. The features are the people. Superposition is the strategy of inviting more people than the room "should" hold, relying on the fact that most of them are quiet most of the time.

---

## 4. Toy Models of Superposition

### 4.1 The Setup (Elhage et al., 2022)

To study superposition rigorously, Elhage et al. designed a toy model with the following structure:

**Task:** Given an input $\mathbf{x} \in \mathbb{R}^n$ consisting of $n$ independent features, compress it through a bottleneck of dimension $m < n$, then decompress it to recover $\mathbf{x}$.

**Model:** A linear model with a bottleneck:
$$
\hat{\mathbf{x}} = \mathbf{W}^T \mathbf{W} \mathbf{x} + \mathbf{b}
$$

where $\mathbf{W} \in \mathbb{R}^{m \times n}$ (compress $n$ dimensions to $m$), and $\mathbf{W}^T$ (expand back to $n$). This is essentially a linear autoencoder.

Note: the same weight matrix $\mathbf{W}$ is used for both encoding and decoding (tied weights, transposed). This is not essential but simplifies analysis.

**Feature structure:**
- Each feature $x\_i$ has an **importance** $I\_i$ (how much the model should care about reconstructing it). Features are ordered by decreasing importance: $I\_1 > I\_2 > \cdots > I\_n$.
- Each feature has a **sparsity** $S$ (probability of being zero). A feature $x\_i$ is independently zero with probability $1 - S$, and drawn from $U[0, 1]$ with probability $S$.

**Loss function:**
$$
\mathcal{L} = \sum_{i=1}^{n} I_i \cdot \mathbb{E}\left[(x_i - \hat{x}_i)^2\right]
$$

This is weighted MSE: features with higher importance contribute more to the loss.

### 4.2 The Key Parameter: Sparsity

The brilliant insight of the paper is to study what happens as the **sparsity** $S$ of the features varies from $S = 1$ (dense, features always active) to $S \approx 0$ (sparse, features rarely active).

**Dense features ($S = 1$):** Every feature is always active. The model must represent all $n$ features in $m < n$ dimensions. The best strategy is PCA-like: represent the top $m$ most important features perfectly and ignore the rest. This is compression by bottleneck, exactly as in an undercomplete autoencoder.

**Sparse features ($S \to 0$):** Features are rarely active. Now superposition becomes possible. The model can represent *more than $m$ features* in $m$ dimensions because the features are unlikely to interfere with each other.

### 4.3 Key Results

The paper demonstrates several striking phenomena:

**Result 1: Phase transition from compression to superposition.** As sparsity increases (features become rarer), the model abruptly switches from the "PCA strategy" (represent top-$m$ features, ignore the rest) to the "superposition strategy" (represent more than $m$ features, tolerating some interference). This is not a gradual transition -- it is a sharp phase change.

```
Features represented
      |
  n   |                         ─────────── superposition
      |                    ____/
      |                   /
  m   |──────────────────/
      |    compression    ↑ phase transition
      |________________________________
      dense                       sparse
              Feature sparsity →
```

**Result 2: Geometric structure.** In superposition, the learned feature directions arrange themselves in specific geometric patterns that minimize interference:

- **Antipodal pairs:** Two features sharing one dimension by pointing in opposite directions. Feature A is encoded as $+\mathbf{e}\_1$ and feature B as $-\mathbf{e}\_1$. Since ReLU can distinguish positive from negative, both features can be recovered.

- **Triangular configurations:** In 2D, three features can be arranged at 120-degree angles (vertices of an equilateral triangle), each pair having inner product $-0.5$.

- **Pentagon/polytope configurations:** More features in more dimensions form the vertices of regular polytopes (pentagons, hexagons, etc.).

These are not arbitrary -- they are optimal geometric packings that minimize the worst-case interference between features.

**Result 3: The number of features in superposition depends on sparsity and importance.** More important features get "cleaner" representations (closer to orthogonal to everything else). Less important features get "noisier" representations (more interference). As sparsity increases, the model is willing to accept more interference because it occurs less frequently.

### 4.4 Why This Matters

The toy model demonstrates that superposition is not a bug or an accident -- it is a *rational strategy* for networks that must represent more features than they have dimensions. A network that uses superposition is doing something smart: it is exploiting the sparsity structure of its inputs to achieve a higher effective capacity.

But it creates a problem for interpretability: if features are in superposition, individual neurons are not meaningful units of analysis. A neuron's activation is a *mixture* of multiple features, and understanding the neuron requires understanding the mixture.

---

## 5. Polysemantic Neurons

### 5.1 The Phenomenon

A **polysemantic neuron** is a neuron that activates for multiple, seemingly unrelated concepts. Classic examples from the interpretability literature:

- A neuron in InceptionV1 (a vision model) that responds to both cat faces and car fronts
- A neuron in GPT-2 that activates for both the word "and" and the beginning of sentences about baseball
- A neuron in a language model that fires for both academic citations and DNA sequences

At first glance, polysemanticity seems like a failure of the network -- why would a well-trained model conflate cats and cars? But superposition provides a clean explanation.

### 5.2 Superposition Explains Polysemanticity

If the "cat" feature and the "car" feature are both represented as directions in activation space, and these directions are *not aligned with any single neuron's axis*, then each neuron's activation will be a linear combination of multiple feature activations.

Concretely, suppose the "cat" feature direction is $\mathbf{v}\_{\text{cat}} = (0.7, 0.5, 0.3, \ldots)$ and the "car" feature direction is $\mathbf{v}\_{\text{car}} = (0.6, -0.4, 0.5, \ldots)$. Then neuron 1's activation is:

$$
h_1 = 0.7 \cdot a_{\text{cat}} + 0.6 \cdot a_{\text{car}} + \cdots
$$

Neuron 1 responds to *both* cats and cars because both feature directions have a positive component along neuron 1's axis. This neuron is polysemantic not because the network is confused, but because it is efficiently packing multiple features into a limited-dimensional space.

### 5.3 The Interpretability Bottleneck

Polysemanticity means that looking at individual neurons is fundamentally insufficient for understanding a neural network. When a neuron fires, we cannot know which of its multiple "meanings" is active without examining the full activation vector and decomposing it into features.

This is exactly what SAEs do. They take a $d$-dimensional activation vector that represents features in superposition and decompose it into a $d'$-dimensional ($d' \gg d$) sparse vector where (ideally) each dimension corresponds to a single, interpretable feature.

---

## 6. SAEs as the Solution

### 6.1 The Pitch

Given everything we have discussed, the logic for using SAEs in interpretability is:

1. Neural networks represent features as directions in activation space (linear representation hypothesis).
2. They represent more features than dimensions using superposition (superposition hypothesis).
3. This makes individual neurons polysemantic and uninterpretable.
4. A sparse autoencoder can decompose superimposed representations into individual features because:
   - Its overcomplete hidden layer has enough dimensions to represent all features without superposition.
   - Its sparsity constraint ensures each input activates only the relevant features.
   - The decoder weights learn the "dictionary" of feature directions.

### 6.2 The Setup

In practice, applying SAEs to a neural network works as follows:

1. **Choose a location** in the network to analyze (e.g., the residual stream at layer 6, or the MLP output at layer 3).
2. **Collect activations** by running many inputs through the network and saving the activation vectors at the chosen location. If the activation dimension is $d$, you collect a dataset of vectors $\lbrace \mathbf{h}\_i \in \mathbb{R}^d\rbrace $.
3. **Train an SAE** on these activation vectors: $\mathbf{h} \to \mathbf{z} \to \hat{\mathbf{h}}$. The hidden dimension $d'$ is typically $4d$ to $32d$.
4. **Analyze the SAE's features:** each hidden neuron in the SAE should (ideally) correspond to an interpretable concept that the original network represents.

```
Original network:                  SAE layer:

input → ... → layer k → ... → output
                 |
                 ↓
              h ∈ R^d          h → z ∈ R^(r*d) → h-hat ∈ R^d
              (activations)        (sparse, interpretable features)
```

### 6.3 What Makes a Good Feature?

An ideal SAE feature should be:

**Monosemantic:** It activates for one coherent concept, not a mixture of unrelated things.

**Specific:** It activates reliably for its concept and does not activate for other things.

**Causally relevant:** Ablating (removing) the feature changes the model's behavior in a way consistent with the feature's supposed meaning.

**Complete:** The set of all SAE features should collectively explain the model's behavior -- they should reconstruct the activation vector well (low reconstruction error).

### 6.4 Why Not Just Use More Neurons?

A reasonable question: if polysemantic neurons are the problem, why not just make the network wider (more neurons per layer) until each neuron is monosemantic?

The answer is efficiency. A network that uses superposition achieves higher effective capacity per parameter. Training a network wide enough to avoid superposition entirely would be enormously expensive and wasteful, since most features are sparse and do not need their own dedicated neuron most of the time.

SAEs offer a post-hoc solution: let the network train however it wants (with superposition), and then use an SAE to decompose its representations after the fact. This decouples the efficiency of the network's internal representations from the interpretability of the analysis.

---

## 7. Causal Interpretability

### 7.1 Correlation Is Not Causation

Finding that an SAE feature activates on inputs related to "French text" is interesting but insufficient. The feature might be *correlated* with French text without being *causally involved* in the model's processing of French text. Maybe it is actually detecting Unicode characters that happen to appear frequently in French, or maybe it is a downstream effect of some other feature that does the real work.

To establish causal relevance, we need intervention experiments.

### 7.2 Ablation Studies

The simplest causal test: **remove** a feature and observe what changes.

**Zero ablation:** Set feature $j$'s activation to zero:
$$
\mathbf{z}_{\text{ablated}} = \mathbf{z} \odot (\mathbf{1} - \mathbf{e}_j)
$$

where $\mathbf{e}\_j$ is the one-hot vector for feature $j$. Then reconstruct with the ablated code:
$$
\hat{\mathbf{h}}_{\text{ablated}} = \mathbf{W}_d \mathbf{z}_{\text{ablated}} + \mathbf{b}_d
$$

Feed this modified activation back into the network and observe the change in output. If the model's prediction changes specifically in ways related to the feature's hypothesized meaning, that is evidence of causal relevance.

**Mean ablation:** Instead of setting to zero, replace the feature's activation with its mean activation across a reference dataset. This is sometimes better because it avoids introducing an out-of-distribution activation (zero might never occur naturally for this feature).

### 7.3 Activation Patching

A more sophisticated technique: **swap** a feature's activation between two different inputs.

Given input A (where the feature is active) and input B (where the feature is inactive), replace feature $j$'s activation in input B with the value from input A. If this causes the model to behave more like it would for input A, the feature is causally relevant for the behavioral difference between A and B.

This is more targeted than ablation because it tests whether a specific feature accounts for a specific behavioral difference.

### 7.4 Feature Steering

The most dramatic demonstration of causal relevance: **artificially increase** a feature's activation and observe the change in behavior.

If a feature represents "formal writing style," then boosting its activation should make the model write more formally. If a feature represents "the Golden Gate Bridge," boosting it should make the model talk about the Golden Gate Bridge.

Feature steering has produced some of the most vivid results in SAE-based interpretability. In Anthropic's "Scaling Monosemanticity" work (which we will study in detail next week), they found a "Golden Gate Bridge" feature in Claude 3 Sonnet. By artificially increasing this feature's activation, they could make the model insert references to the Golden Gate Bridge into any conversation -- a striking and viral demonstration of causal influence.

### 7.5 The Faithfulness Question

All of these causal methods assume that the SAE's decomposition is faithful -- that the features in the SAE correspond to the "true" features used by the network. This is a deep and unresolved question.

It is possible that:
- The SAE's features are a reasonable but imperfect approximation of the network's true features.
- Different SAE architectures or random seeds find different but equally valid decompositions.
- Some features in the SAE are artifacts of the training process that do not correspond to anything meaningful in the original network.

The faithfulness problem is an active area of research. For now, causal validation (ablation, patching, steering) is the best available method for checking whether SAE features are "real."

---

## 8. Circuits: From Features to Algorithms

### 8.1 Beyond Individual Features

Identifying individual features is only the first step. The deeper goal of mechanistic interpretability is to understand **circuits** -- how features interact across layers to implement algorithms.

A circuit is a subnetwork that takes specific input features, performs specific computations, and produces specific output features. Examples:

**Induction circuit** (Olsson et al., 2022): A two-layer circuit in transformers that implements the pattern "if A followed B, then the next time A appears, predict B." This circuit uses attention heads in specific ways across two layers to copy patterns.

**Indirect Object Identification** (Wang et al., 2022): A circuit in GPT-2 that identifies the indirect object in sentences like "When Mary and John went to the store, John gave a drink to ___." The circuit involves specific attention heads that perform "subject tracking" and "object identification."

### 8.2 SAEs and Circuit Analysis

SAEs can help with circuit analysis by:

1. **Identifying the features** that are relevant for a specific behavior.
2. **Tracking features across layers:** train SAEs at multiple layers and see how features at layer $k$ relate to features at layer $k+1$.
3. **Simplifying the network:** instead of analyzing all $d$ neurons at each layer, analyze the $k \ll d$ active SAE features, which are (ideally) more interpretable.

This is an area of very active research. The combination of SAEs (for feature identification) and causal methods (for circuit tracing) is the current state of the art in mechanistic interpretability.

---

## 9. A Brief History

The ideas we have discussed did not emerge all at once. A brief timeline:

- **2017:** Olah et al. ("Feature Visualization") demonstrate that individual neurons and directions in vision models correspond to interpretable features, but also note widespread polysemanticity.

- **2020:** Elhage et al. ("A Mathematical Framework for Transformer Circuits") lay the groundwork for mechanistic analysis of transformers, introducing the "residual stream" view.

- **2022:** Elhage et al. ("Toy Models of Superposition") formalize the superposition hypothesis and demonstrate it in controlled settings. This paper is the theoretical foundation for SAE-based interpretability.

- **2022:** Sharkey et al. ("Taking Features out of Superposition with Sparse Autoencoders") demonstrate that SAEs can recover features from toy models of superposition.

- **2023:** Cunningham et al. and Bricken et al. apply SAEs to real language models, demonstrating interpretable features at scale. Anthropic's "Towards Monosemanticity" is the landmark paper.

- **2024:** Templeton et al. ("Scaling Monosemanticity") apply SAEs to Claude 3 Sonnet, demonstrating interpretable features in a production-scale model. The "Golden Gate Bridge Claude" goes viral.

We will study the 2023 and 2024 papers in detail next week.

---

## 10. Summary

| Concept | Key Idea |
|---------|----------|
| Interpretability problem | Neural networks are powerful but opaque; we want to understand their internals |
| Linear representation hypothesis | Features are directions in activation space |
| Superposition | Networks represent more features than dimensions using nearly-orthogonal directions |
| Why superposition works | Sparse features rarely interfere; high-dimensional spaces have many nearly-orthogonal directions |
| Polysemantic neurons | Individual neurons respond to multiple features (predicted by superposition) |
| Toy models | Phase transition from compression to superposition as sparsity increases |
| SAEs as solution | Overcomplete + sparse decomposition recovers individual features from superposition |
| Causal interpretability | Ablation, patching, and steering verify that features are causally relevant |
| Circuits | How features interact across layers to implement algorithms |

---

## Further Reading

- **Elhage et al.**, "Toy Models of Superposition" (Anthropic, 2022). The foundational paper for this week. Read at least Sections 1-3 and the results figures. Available at transformer-circuits.pub.

- **Olah et al.**, "Zoom In: An Introduction to Circuits" (2020). A beautifully written and illustrated introduction to mechanistic interpretability in vision models.

- **Neel Nanda**, "A Comprehensive Mechanistic Interpretability Explainer" (2023). An excellent blog post that covers superposition, SAEs, and circuits at an accessible level.

- **Olsson et al.**, "In-context Learning and Induction Heads" (Anthropic, 2022). The paper that identified the induction circuit, a paradigmatic example of a neural network "algorithm."

---

*Next week: We apply SAEs to real language models and discover what they have learned -- from "Arabic script" to "code errors" to "the Golden Gate Bridge."*
