# Week 11: Mechanistic Interpretability -- Homework

## Overview

This problem set has you build the toy model of superposition from Elhage et al. (2022), reproduce its key results, and then use a sparse autoencoder to recover features from superposition. The final problems develop your conceptual understanding of the geometry of superposition and the design of causal experiments.

**Tools:** Python, PyTorch, matplotlib, numpy.

**Estimated time:** 8-12 hours.

---

## Problem 1: Toy Model of Superposition (Implementation)

Implement the toy model from Elhage et al. (2022) and reproduce the key result: superposition emerges when features are sparse.

### 1a. Model Architecture

Implement a linear autoencoder with tied weights:

$$
\hat{\mathbf{x}} = \mathbf{W}^T \mathbf{W} \mathbf{x} + \mathbf{b}
$$

where $\mathbf{W} \in \mathbb{R}^{m \times n}$. Use $n = 20$ input features and $m = 5$ bottleneck dimensions. The bias $\mathbf{b} \in \mathbb{R}^n$ is optional but recommended.

Apply ReLU to the output: $\hat{\mathbf{x}} = \text{ReLU}(\mathbf{W}^T \mathbf{W} \mathbf{x} + \mathbf{b})$. This is important because input features are non-negative, and ReLU allows the model to use antipodal pairs in the bottleneck.

### 1b. Data Generation

Generate synthetic data where each input $\mathbf{x} \in \mathbb{R}^n$ has:
- Each feature $x\_i$ is independently zero with probability $(1 - S)$ and drawn from $U[0, 1]$ with probability $S$, where $S$ is the sparsity parameter (probability of being active).
- Feature importances decay geometrically: $I\_i = 0.9^i$ for $i = 0, 1, \ldots, n-1$.

### 1c. Loss Function

Use importance-weighted MSE:

$$
\mathcal{L} = \sum_{i=1}^{n} I_i \cdot \mathbb{E}\left[(x_i - \hat{x}_i)^2\right]
$$

In practice, compute this as a batch average:

$$
\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} \sum_{i=1}^{n} I_i (x_i^{(b)} - \hat{x}_i^{(b)})^2
$$

### 1d. Training

Train the model with Adam (lr $= 10^{-3}$) for 10,000 steps with batch size 256. Generate fresh data for each batch (do not use a fixed dataset).

Train **separate models** for each of the following sparsity levels:

$$
S \in \{1.0, 0.5, 0.1, 0.05, 0.01, 0.003\}
$$

### 1e. Visualization: The W Matrix

For each trained model, visualize the $m \times n$ weight matrix $\mathbf{W}$ as a heatmap.

Then create the key visualization from the paper: for each feature $i$, compute $\Vert \mathbf{W}\_{:,i}\Vert ^2$ (the squared norm of its column in $\mathbf{W}$). This measures how much of the bottleneck capacity is devoted to feature $i$.

Plot $\Vert \mathbf{W}\_{:,i}\Vert ^2$ vs. feature index $i$ for each sparsity level, all on the same plot. You should observe:
- At $S = 1.0$ (dense): only the top $m = 5$ features have significant weight (PCA-like).
- At $S = 0.01$ (sparse): many more than $m = 5$ features have significant weight (superposition).

---

## Problem 2: Phase Transition (Experimental)

### 2a. Quantifying Superposition

For each trained model from Problem 1, compute the **number of features represented**, defined as:

$$
N_{\text{rep}} = \sum_{i=1}^{n} \|\mathbf{W}_{:,i}\|^2
$$

This quantity equals $m$ when each bottleneck dimension is "used up" by exactly one feature (no superposition), and exceeds $m$ when features share dimensions (superposition).

An alternative metric: count the number of features $i$ for which $\Vert \mathbf{W}\_{:,i}\Vert ^2 > 0.1$ (a threshold indicating "meaningfully represented").

### 2b. Phase Transition Plot

Create a plot of $N\_{\text{rep}}$ (y-axis) vs. sparsity $S$ (x-axis, log scale). Use more sparsity levels than Problem 1 if needed (e.g., 15-20 values logarithmically spaced between 0.003 and 1.0).

You should observe a transition: at high density (large $S$), $N\_{\text{rep}} \approx m = 5$; at high sparsity (small $S$), $N\_{\text{rep}}$ is significantly larger, approaching $n = 20$.

Mark the approximate location of the phase transition on your plot.

### 2c. Discussion

In 3-5 sentences, explain:
- Why does the model switch from compression to superposition as sparsity increases?
- What is the "cost" of superposition (in terms of reconstruction error)?
- Why is this cost acceptable when features are sparse?

---

## Problem 3: Recovering Features with an SAE (Implementation)

This is the central problem of the set: demonstrate that an SAE can recover individual features from a model that uses superposition.

### 3a. Generate Superimposed Representations

Use the toy model trained with $S = 0.01$ from Problem 1. For a dataset of 10,000 inputs $\lbrace \mathbf{x}\_i\rbrace $:
1. Compute the bottleneck representations: $\mathbf{h}\_i = \mathbf{W} \mathbf{x}\_i \in \mathbb{R}^m$.
2. These are the "activations" that live in superposition -- $m = 5$ dimensions encoding information about $n = 20$ features.

### 3b. Train an SAE

Train a sparse autoencoder on the bottleneck representations $\lbrace \mathbf{h}\_i\rbrace $:
- **Input dimension:** $m = 5$
- **Hidden dimension:** $d = 40$ (expansion factor of 8; we expect around 20 true features)
- **Loss:** MSE + L1 penalty with $\lambda = 0.01$ (tune this if needed)
- **Decoder norm constraint:** Apply column normalization after each step.
- **Optimizer:** Adam, lr $= 10^{-3}$, 5000 steps, batch size 256.

### 3c. Feature Recovery Analysis

After training the SAE:

1. **Decoder column analysis:** Each column of the SAE's decoder $\mathbf{W}\_d[:, j] \in \mathbb{R}^m$ represents a learned feature direction in the bottleneck space. Each column of the toy model's $\mathbf{W}$ matrix, $\mathbf{W}[:, i] \in \mathbb{R}^m$, represents where a ground-truth feature is stored in the bottleneck.

   Compute the cosine similarity matrix between all SAE decoder columns and all ground-truth feature directions. This is a $40 \times 20$ matrix.

2. **Matching:** For each ground-truth feature $i$, find the SAE feature $j$ with the highest cosine similarity. Is the match close to 1.0? How many ground-truth features are successfully recovered (cosine similarity > 0.9)?

3. **Visualization:** Plot the cosine similarity matrix as a heatmap. If the SAE successfully recovers features, you should see a near-permutation matrix structure (each ground-truth feature matched to one SAE feature, with the remaining SAE features unused or dead).

### 3d. What If Sparsity Is Wrong?

Retrain the SAE with $\lambda = 0$ (no sparsity). Does it still recover the features? Retrain with $\lambda = 1.0$ (very high sparsity). What happens?

Write a paragraph explaining why the sparsity coefficient matters for feature recovery.

---

## Problem 4: Polysemanticity Demonstration (Experimental)

### 4a. Find a Polysemantic Neuron

Using the toy model from Problem 1 (with $S = 0.01$), examine the weight matrix $\mathbf{W}$ and find a bottleneck neuron (row of $\mathbf{W}$) that has significant weight on multiple input features.

Specifically, find neuron $k$ such that $|W\_{k,i}| > 0.3$ for at least two different features $i$.

### 4b. Demonstrate Polysemanticity

For your chosen neuron $k$:
1. Generate 1000 random inputs.
2. For each input, record neuron $k$'s activation: $h\_k = (\mathbf{W} \mathbf{x})\_k$.
3. Also record which input features are active.

Show that neuron $k$ activates when *any* of its multiple contributing features are active. Create a scatter plot or histogram that makes the polysemanticity visually clear.

### 4c. SAE Decomposition

Using the SAE from Problem 3, show that the SAE decomposes the polysemantic neuron's activity into separate features. Specifically:
1. For the same 1000 inputs, compute the SAE hidden activations $\mathbf{z}$.
2. Identify the SAE features that correspond to each of the ground-truth features that activate neuron $k$.
3. Show that each SAE feature activates only when its specific ground-truth feature is active, even though the bottleneck neuron activates for all of them.

This is the key result: the SAE successfully disentangles a polysemantic neuron into monosemantic features.

---

## Problem 5: The Geometry of Nearly-Orthogonal Vectors (Theory)

### 5a. Random Vectors in High Dimensions

Consider two random unit vectors $\mathbf{u}, \mathbf{v}$ drawn uniformly from the unit sphere in $\mathbb{R}^d$.

Using the fact that random projections of high-dimensional vectors are approximately Gaussian (by the central limit theorem), argue that:

$$
\mathbb{E}[\mathbf{u} \cdot \mathbf{v}] = 0, \qquad \text{Var}(\mathbf{u} \cdot \mathbf{v}) \approx \frac{1}{d}
$$

Conclude that for large $d$, the inner product $\mathbf{u} \cdot \mathbf{v}$ is concentrated near zero with standard deviation $\approx 1/\sqrt{d}$.

### 5b. Connection to Johnson-Lindenstrauss

The Johnson-Lindenstrauss lemma states that $n$ points in high-dimensional space can be projected to $d = O(\epsilon^{-2} \log n)$ dimensions while preserving all pairwise distances to within a factor of $(1 \pm \epsilon)$.

Explain (in 3-5 sentences) why this is relevant to superposition. Specifically:
- If a neural network wants to represent $n$ features as nearly-orthogonal directions in $d$ dimensions (with pairwise inner products at most $\epsilon$), how does $d$ need to scale with $n$?
- What does this tell us about the "capacity" of superposition?

### 5c. Counting Nearly-Orthogonal Vectors

For a concrete calculation: in $\mathbb{R}^d$, how many unit vectors can you place such that all pairwise inner products have absolute value at most $\epsilon$?

A known lower bound is approximately $(1/\epsilon)^{cd}$ for some constant $c > 0$. This is **exponential** in the dimension $d$.

For $d = 768$ (GPT-2 small's hidden dimension) and $\epsilon = 0.05$ (5% interference), give a rough lower bound on the number of nearly-orthogonal feature directions. Compare this to the number of features that Anthropic found in SAE studies of similar-scale models (on the order of 10,000-100,000).

Is superposition a plausible explanation for how language models represent so many features?

---

## Problem 6: Designing an Ablation Experiment (Conceptual)

This problem asks you to design (but not implement) a causal experiment to test whether an SAE feature is "real."

### Scenario

Suppose you have trained an SAE on the residual stream activations at layer 8 of GPT-2 small. You find a feature (call it Feature 4721) that appears to activate on text about cooking and recipes. Specifically, the top-10 maximally activating examples all contain cooking-related content:
- "Preheat the oven to 350 degrees..."
- "Add the flour and stir until combined..."
- "Season the chicken with salt and pepper..."

You want to test whether Feature 4721 is causally involved in the model's processing of cooking-related text.

### 6a. Ablation Experiment Design

Design an ablation experiment. Specifically:
1. What inputs would you use for testing? (Describe the test set you would construct.)
2. What metric would you measure? (Be specific -- not just "model behavior" but what exactly.)
3. What would a positive result look like? (What change in the metric would confirm causal relevance?)
4. What would a null result mean? (Feature activates on cooking text but is not causally involved in processing it.)

### 6b. Feature Steering Experiment Design

Design a feature steering experiment:
1. What inputs would you use? (These should be inputs where the feature is NOT naturally active.)
2. How would you artificially activate the feature? (At what magnitude? How do you choose?)
3. What behavior change would you look for?
4. What controls would you include to ensure the effect is specific to cooking and not just "the model behaves weirdly when you perturb its activations"?

### 6c. Potential Confounds

List at least three potential confounds that could lead you to a wrong conclusion about Feature 4721's meaning or causal role. For each confound, describe how you would test for it or mitigate it.

### 6d. Feature Completeness

Even if Feature 4721 passes all your causal tests, it might not be the *only* cooking-related feature. How would you test whether the SAE has captured the *complete* set of cooking-related features, or whether some cooking-related information is spread across other features?

---

## Submission Checklist

- [ ] Problem 1: Toy model -- code, weight matrix heatmaps, feature representation plot across sparsity levels
- [ ] Problem 2: Phase transition -- plot of features represented vs. sparsity, discussion
- [ ] Problem 3: SAE feature recovery -- cosine similarity matrix, matching results, effect of lambda, discussion
- [ ] Problem 4: Polysemanticity -- identification of polysemantic neuron, demonstration, SAE decomposition
- [ ] Problem 5: Geometry -- theoretical arguments, JL connection, capacity calculation
- [ ] Problem 6: Ablation design -- detailed experimental designs, confound analysis, completeness test
