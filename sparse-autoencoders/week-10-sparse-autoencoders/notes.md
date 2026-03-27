# Week 10: Sparse Autoencoders

## This is the week everything we have built leads to.

In Week 5, we saw that representations matter -- that the right encoding of data can make downstream tasks trivial. In Week 6, we built autoencoders that learn representations by compressing data through a bottleneck. In Week 9, we studied sparsity as a structural prior -- how insisting that only a few dictionary atoms are active per input leads to interpretable, robust representations.

Now we fuse these ideas. A **sparse autoencoder** (SAE) is an overcomplete autoencoder -- one whose hidden dimension is *larger* than its input dimension -- that is prevented from learning a trivial identity mapping by a sparsity constraint on its hidden activations. This seemingly simple idea turns out to be extraordinarily powerful, and in the coming weeks, we will see it applied to one of the deepest open problems in AI: understanding what neural networks have learned.

---

## 1. The SAE Architecture

### 1.1 Setup and Notation

Let $\mathbf{x} \in \mathbb{R}^n$ be an input vector. A sparse autoencoder maps:

$$
\mathbf{x} \xrightarrow{\text{encode}} \mathbf{z} \xrightarrow{\text{decode}} \hat{\mathbf{x}}
$$

where the hidden representation $\mathbf{z} \in \mathbb{R}^d$ has $d > n$ (overcomplete), but most entries of $\mathbf{z}$ are zero for any given input.

**Encoder:**
$$
\mathbf{z} = \sigma(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)
$$

where $\mathbf{W}\_e \in \mathbb{R}^{d \times n}$ is the encoder weight matrix, $\mathbf{b}\_e \in \mathbb{R}^d$ is the encoder bias, and $\sigma$ is an activation function (typically ReLU).

**Decoder:**
$$
\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_d
$$

where $\mathbf{W}\_d \in \mathbb{R}^{n \times d}$ is the decoder weight matrix and $\mathbf{b}\_d \in \mathbb{R}^n$ is the decoder bias.

Note: some formulations use a pre-encoder bias (subtracting the mean of the data before encoding and adding it back after decoding). This is especially common when training SAEs on neural network activations, where a "geometric median" centering can improve training stability:

$$
\mathbf{z} = \sigma(\mathbf{W}_e (\mathbf{x} - \mathbf{b}_{\text{pre}}) + \mathbf{b}_e), \qquad \hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_{\text{pre}}
$$

### 1.2 Architecture Diagram

```
Input x            Hidden z (sparse)           Output x-hat
[n dims]           [d dims, d >> n]            [n dims]

 x_1 ─────┐   ┌─ z_1 = 0.0                ┌──── x-hat_1
 x_2 ─────┤   ├─ z_2 = 0.0                ├──── x-hat_2
 x_3 ─────┼───┤  z_3 = 2.7  * ────────────┼──── x-hat_3
 x_4 ─────┤   ├─ z_4 = 0.0                ├──── x-hat_4
   .       │   │  z_5 = 0.0                │      .
   .       │   │  z_6 = 1.3  * ────────────┤      .
   .       │   │  z_7 = 0.0                │      .
 x_n ─────┘   ├─ z_8 = 0.0                └──── x-hat_n
               │    .
               │    .
               └─ z_d = 0.4  *
                  ^
         Most activations are 0.
         Only a few (*) are active.

         <-- W_e, b_e -->   <-- W_d, b_d -->
            (encode)            (decode)
```

The key visual: the hidden layer is *wider* than the input, but *sparser*. Only a handful of hidden units fire for any given input.

### 1.3 The Loss Function (General Form)

$$
\mathcal{L} = \underbrace{\Vert \mathbf{x} - \hat{\mathbf{x}}\Vert _2^2}_{\text{reconstruction}} + \underbrace{\lambda \cdot \Omega(\mathbf{z})}_{\text{sparsity penalty}}
$$

where $\Omega(\mathbf{z})$ is some function that penalizes non-sparse activations, and $\lambda > 0$ controls the trade-off between faithful reconstruction and sparsity.

This is the same structure we saw in regularized regression (Week 1) and sparse coding (Week 9). The reconstruction term wants the autoencoder to be a good compressor; the sparsity term wants the representation to be economical.

---

## 2. Why Overcomplete + Sparse?

### 2.1 Two Philosophies of Compression

We have now seen two fundamentally different approaches to learning compact representations:

**Undercomplete autoencoders (Week 6):** Force compression by making $d < n$. The bottleneck is *architectural* -- there simply are not enough dimensions to store everything, so the network must learn which information to keep. This is like being given a small suitcase: you must choose what to pack.

**Sparse autoencoders:** Force compression by penalizing activation. The hidden layer has $d \gg n$ dimensions, but the sparsity constraint ensures that only $k \ll d$ of them are active for any given input. This is like having a huge warehouse of tools, but being told you can only carry $k$ of them to each job site.

### 2.2 The Flexibility Advantage

Why might the second approach be better? Consider representing handwritten digits.

An undercomplete AE with $d = 10$ must represent *every* digit as a point in 10-dimensional space. The network must learn a global coordinate system where all digits coexist.

A sparse AE with $d = 500$ and typical sparsity $k \approx 10$ has 500 possible features available. For any given digit, it activates roughly 10 of them. But different digits can activate *different subsets* of the 500 features. The representational capacity is vastly larger: there are $\binom{500}{10} \approx 2.6 \times 10^{20}$ possible activation patterns.

This is a form of **combinatorial coding**. With $d$ features and $k$ active at a time, the number of distinct codes is $\binom{d}{k}$, which grows combinatorially. Compare this to a dense code with $k$ dimensions, which has a continuous $k$-dimensional space but no discrete structure.

### 2.3 The Dictionary Analogy

Think of a dictionary of a natural language. English has roughly 170,000 words in current use. A typical sentence uses maybe 10-20 of them. No one would argue we should reduce the dictionary to 20 words for "efficiency" -- the whole power comes from having a huge vocabulary while using only a small, contextually appropriate subset.

This is exactly the sparse autoencoder philosophy: learn a large dictionary of features, but use only a few per input.

### 2.4 Connection to Biology

This also connects to what we learned in Week 9 about neural coding in the brain. The mammalian visual cortex has far more neurons than the dimensionality of the visual input, yet neural activity is sparse -- only a small fraction of neurons fire for any given visual scene. The brain appears to implement an overcomplete + sparse strategy. There may be deep computational reasons why this is a good idea, related to energy efficiency, robustness, and compositionality of representations.

---

## 3. Sparsity via L1 Penalty

### 3.1 The L1 Approach

The simplest way to encourage sparsity is to add an L1 penalty on the hidden activations:

$$
\mathcal{L}_{\text{L1}} = \Vert \mathbf{x} - \hat{\mathbf{x}}\Vert _2^2 + \lambda \Vert \mathbf{z}\Vert _1 = \Vert \mathbf{x} - \hat{\mathbf{x}}\Vert _2^2 + \lambda \sum_{j=1}^{d} |z_j|
$$

We studied L1 regularization extensively in Week 1 (LASSO) and Week 9 (sparse coding). The key property is that L1 drives values exactly to zero, unlike L2 which merely shrinks them. The non-differentiability of the absolute value at zero creates a "dead zone" where the gradient of the penalty can overpower the gradient of the reconstruction loss, forcing the activation to stay at exactly zero.

**Important distinction:** We are penalizing the *activations* $\mathbf{z}$, not the *weights*. Weight decay (L2 on weights) prevents individual weights from growing too large. The L1 activation penalty prevents too many hidden units from being active simultaneously. These are very different regularizers with very different effects.

### 3.2 Gradient Derivation

Since $\mathbf{z} = \text{ReLU}(\mathbf{W}\_e \mathbf{x} + \mathbf{b}\_e)$, we have $z\_j \geq 0$ for all $j$. This simplifies the L1 penalty:

$$
\Vert \mathbf{z}\Vert _1 = \sum_j |z_j| = \sum_j z_j \qquad \text{(since } z_j \geq 0 \text{ after ReLU)}
$$

The gradient of the sparsity penalty with respect to the pre-activation $\mathbf{a} = \mathbf{W}\_e \mathbf{x} + \mathbf{b}\_e$ is:

$$
\frac{\partial (\lambda \Vert \mathbf{z}\Vert _1)}{\partial a_j} = \lambda \cdot \frac{\partial z_j}{\partial a_j} = \lambda \cdot \mathbb{1}[a_j > 0]
$$

That is, for every active neuron ($a\_j > 0$), there is a constant penalty gradient of $\lambda$ pushing the activation toward zero. For inactive neurons ($a\_j \leq 0$), the gradient is zero (they are already "sparse").

For the full loss, the gradient with respect to $a\_j$ combines the reconstruction gradient and the sparsity gradient:

$$
\frac{\partial \mathcal{L}}{\partial a_j} = \frac{\partial \Vert \mathbf{x} - \hat{\mathbf{x}}\Vert _2^2}{\partial a_j} + \lambda \cdot \mathbb{1}[a_j > 0]
$$

The reconstruction gradient tries to keep neurons active (to reconstruct well); the sparsity gradient tries to turn them off. The balance between these forces, governed by $\lambda$, determines the final sparsity level.

### 3.3 The Effect of Varying Lambda

| $\lambda$ | Effect | Reconstruction | Sparsity |
|-----------|--------|----------------|----------|
| $\lambda = 0$ | No sparsity penalty | Excellent (but trivial -- may learn identity) | Low |
| $\lambda$ small | Mild sparsity | Good | Moderate |
| $\lambda$ moderate | Strong sparsity | Acceptable | High |
| $\lambda$ large | Extreme sparsity | Poor (too constrained) | Very high (nearly all zeros) |

**Concrete example:** Suppose we train on MNIST ($n = 784$) with $d = 2000$.

- At $\lambda = 0$: the network might activate all 2000 hidden units for each input. Reconstruction is perfect but the representation is dense and uninterpretable.
- At $\lambda = 0.001$: perhaps 100-200 units are active per input. Features start to look like oriented strokes and edges.
- At $\lambda = 0.01$: perhaps 20-40 units active per input. Features become crisp, localized edge detectors. Reconstruction is slightly blurry but structurally faithful.
- At $\lambda = 0.1$: perhaps 3-5 units active. Reconstruction degrades noticeably. Features are very high-level (maybe "loop detector," "vertical stroke").
- At $\lambda = 1.0$: essentially everything is pushed to zero. The network barely reconstructs.

The "sweet spot" depends on your goals. For interpretability, you often want fairly aggressive sparsity -- enough active features to faithfully represent the input, but few enough that each feature is individually meaningful.

---

## 4. Sparsity via KL Divergence

### 4.1 Andrew Ng's Approach

An alternative to L1 is the **KL divergence penalty**, popularized by Andrew Ng in his Stanford CS294A lecture notes. Instead of penalizing individual activations, we penalize the *average activation* of each hidden neuron across the training set.

**Define the average activation** of hidden neuron $j$:

$$
\hat{\rho}_j = \frac{1}{m} \sum_{i=1}^{m} z_j(\mathbf{x}_i)
$$

where the sum is over $m$ training examples (or a minibatch). This measures how often neuron $j$ is active on average.

**Set a target sparsity** $\rho$ (e.g., $\rho = 0.05$). We want $\hat{\rho}\_j \approx \rho$ for all $j$ -- each neuron should be active about 5% of the time.

**Penalize deviations** using KL divergence between Bernoulli distributions:

$$
\Omega_{\text{KL}} = \sum_{j=1}^{d} \text{KL}(\text{Bernoulli}(\rho) \Vert \text{Bernoulli}(\hat{\rho}_j))
$$

### 4.2 Deriving the KL Term

The KL divergence between two Bernoulli distributions with parameters $\rho$ and $\hat{\rho}$ is:

$$
\text{KL}(\rho \Vert \hat{\rho}) = \rho \log \frac{\rho}{\hat{\rho}} + (1 - \rho) \log \frac{1 - \rho}{1 - \hat{\rho}}
$$

Let us verify the key properties:

1. **Minimum at $\hat{\rho} = \rho$:** Taking the derivative and setting it to zero:
   $$
   \frac{\partial \text{KL}}{\partial \hat{\rho}} = -\frac{\rho}{\hat{\rho}} + \frac{1-\rho}{1-\hat{\rho}} = 0 \implies \hat{\rho} = \rho
   $$

2. **Always non-negative:** By Gibbs' inequality, $\text{KL}(\rho \Vert \hat{\rho}) \geq 0$ with equality iff $\hat{\rho} = \rho$.

3. **Blows up at boundaries:** As $\hat{\rho} \to 0$ or $\hat{\rho} \to 1$ (when $\rho$ is neither 0 nor 1), the KL divergence goes to infinity. This prevents neurons from being permanently dead or permanently active.

**Numerical example** with $\rho = 0.05$:

| $\hat{\rho}\_j$ | $\text{KL}(\rho \Vert \hat{\rho}\_j)$ | Interpretation |
|---|---|---|
| 0.01 | 0.0770 | Too inactive |
| 0.05 | 0.0000 | Just right |
| 0.10 | 0.0214 | Slightly too active |
| 0.30 | 0.2015 | Way too active |
| 0.50 | 0.4515 | Very heavily penalized |
| 0.90 | 1.6094 | Extreme penalty |

The full loss becomes:

$$
\mathcal{L}_{\text{KL}} = \Vert \mathbf{x} - \hat{\mathbf{x}}\Vert _2^2 + \beta \sum_{j=1}^{d} \text{KL}(\rho \Vert \hat{\rho}_j)
$$

where $\beta > 0$ is the sparsity weight (analogous to $\lambda$ in the L1 case).

### 4.3 Gradient of the KL Penalty

The gradient with respect to $\hat{\rho}\_j$ is:

$$
\frac{\partial \text{KL}(\rho \Vert \hat{\rho}_j)}{\partial \hat{\rho}_j} = -\frac{\rho}{\hat{\rho}_j} + \frac{1 - \rho}{1 - \hat{\rho}_j}
$$

And since $\hat{\rho}\_j = \frac{1}{m} \sum\_i z\_j(\mathbf{x}\_i)$, the gradient with respect to $z\_j(\mathbf{x}\_i)$ is:

$$
\frac{\partial \text{KL}(\rho \Vert \hat{\rho}_j)}{\partial z_j(\mathbf{x}_i)} = \frac{1}{m}\left(-\frac{\rho}{\hat{\rho}_j} + \frac{1 - \rho}{1 - \hat{\rho}_j}\right)
$$

This gradient is negative when $\hat{\rho}\_j < \rho$ (pushing the neuron to be more active) and positive when $\hat{\rho}\_j > \rho$ (pushing it to be less active). The penalty acts like a thermostat, maintaining each neuron's average activation near the target.

### 4.4 L1 vs. KL: A Comparison

| Aspect | L1 Penalty | KL Divergence Penalty |
|--------|------------|----------------------|
| What it penalizes | Individual activation magnitude | Average activation level |
| Effect per input | Pushes each $z\_j$ toward 0 | Indirectly encourages sparsity through averages |
| Gradient for active neurons | Constant ($\lambda$) | Depends on current $\hat{\rho}\_j$ relative to target |
| Dead neurons | Can happen (gradient is zero for $z\_j = 0$) | Explicitly prevented ($\text{KL} \to \infty$ as $\hat{\rho}\_j \to 0$) |
| Hyperparameters | $\lambda$ | $\rho$ (target), $\beta$ (weight) |
| Batch dependence | No (per-example) | Yes (needs batch to estimate $\hat{\rho}\_j$) |
| In practice | More common in modern SAE work | Historical significance (Ng's notes); less common now |

**The modern consensus** tends to favor L1 for its simplicity and per-example nature, but the KL approach offers a valuable insight: sparsity is about *resource allocation across the population of neurons*, not just about shrinking individual values.

In practice, there are also other approaches: **TopK activation** (keep only the $k$ largest activations, set the rest to zero), **Gated SAEs** (learn a separate gating network), and **JumpReLU** (a shifted ReLU that creates a hard sparsity threshold). We will see these in Week 13.

---

## 5. Training SAEs: Practical Challenges

Building an SAE that works on paper is straightforward. Building one that works *in practice* is an art. This section covers the demons you will encounter.

### 5.1 Dead Neurons

**The problem:** After some training, a significant fraction of hidden neurons may have activations that are *always zero* for every input in the dataset. These "dead neurons" contribute nothing to reconstruction and waste capacity.

**Why it happens:** Consider a ReLU neuron with pre-activation $a\_j = \mathbf{w}\_j^\top \mathbf{x} + b\_j$. If, due to some unlucky gradient updates early in training, $a\_j < 0$ for all inputs $\mathbf{x}$ in the training set, then:
- $z\_j = \text{ReLU}(a\_j) = 0$ for all inputs
- The gradient of the reconstruction loss with respect to $\mathbf{w}\_j$ is $\frac{\partial \mathcal{L}}{\partial z\_j} \cdot \frac{\partial z\_j}{\partial a\_j} = \frac{\partial \mathcal{L}}{\partial z\_j} \cdot 0 = 0$
- The neuron receives no learning signal. It is stuck dead.

This is the "dying ReLU" problem we encountered in Week 3, but it is much worse in SAEs because the L1 penalty *actively pushes neurons toward death*. The sparsity penalty wants neurons to be off; once off, they stay off.

**How bad is it?** In early SAE work, it was not uncommon to see 50-90% of neurons dead after training. This means you paid for a 10,000-dimensional hidden layer but only 1,000 neurons are actually doing anything.

**Solutions:**

**(a) Neuron resampling.** Periodically check which neurons are dead (zero activation on a large batch). For dead neurons, reinitialize their weights to point toward data points with high reconstruction error. The intuition: these are the inputs the SAE is struggling with, so new neurons should try to help with them.

A typical resampling strategy:
1. Every $T$ steps (e.g., $T = 25000$), identify neurons that have not activated in the last $T$ steps.
2. Sample data points $\mathbf{x}\_i$ with probability proportional to their reconstruction error $\Vert \mathbf{x}\_i - \hat{\mathbf{x}}\_i\Vert ^2$.
3. Set the dead neuron's encoder weights to $\mathbf{w}\_j \leftarrow \frac{\mathbf{x}\_i - \hat{\mathbf{x}}\_i}{\Vert \mathbf{x}\_i - \hat{\mathbf{x}}\_i\Vert }$ (pointing toward the residual).
4. Set the corresponding decoder column to the same direction, scaled down.
5. Reset the encoder bias to a small negative value.

**(b) Leaky ReLU or other activations.** Replace $\text{ReLU}(a) = \max(0, a)$ with $\text{LeakyReLU}(a) = \max(\alpha a, a)$ where $\alpha = 0.01$. Now even for $a < 0$, the gradient is $\alpha \neq 0$, so dead neurons can recover. The trade-off: Leaky ReLU does not produce exact zeros, so "sparsity" becomes approximate.

**(c) Auxiliary loss.** Add a small auxiliary loss that specifically penalizes dead neurons. For example, Anthropic's approach adds a loss term based on the dead neurons' contribution to the reconstruction error from the top-$k$ most erroneously reconstructed examples. This gives dead neurons a gradient signal to "wake up."

### 5.2 Feature Splitting

**The problem:** A single conceptual feature (e.g., "vertical stroke in the left half of the image") may be split across multiple SAE neurons. Instead of one neuron cleanly detecting this feature, you get three neurons that each detect a slightly different variant.

**Why it happens:** When the hidden dimension $d$ is large relative to the true number of features, the SAE has more capacity than it needs. Multiple neurons may "share" the same feature, each capturing a slightly different aspect. This is especially common when $\lambda$ is too small (weak sparsity pressure).

**How to detect it:** Compute the cosine similarity between decoder columns $\mathbf{d}\_j = \mathbf{W}\_d[:, j]$. If two decoder columns have high cosine similarity (e.g., $> 0.9$), their corresponding features are likely splits of the same underlying concept.

**Mitigation:** Increase $\lambda$ (stronger sparsity pressure makes it costly to maintain redundant features). Use decoder column orthogonality regularization. Or accept some degree of splitting and post-process by clustering similar features.

### 5.3 Choosing Hyperparameters

The key hyperparameters for an SAE are:

**Hidden dimension $d$:** Usually expressed as an "expansion factor" $r = d/n$. Common choices: $r = 4, 8, 16, 32$. Larger $r$ allows more features to be discovered but increases training cost and the risk of feature splitting. A useful principle: start with $r = 4$ and increase if your features look coarse-grained.

**Sparsity coefficient $\lambda$:** Controls the sparsity-reconstruction trade-off. This is the most important hyperparameter to tune. A practical approach:
1. Train several SAEs with different $\lambda$ values (e.g., $\lambda \in \lbrace 0.0001, 0.001, 0.01, 0.1\rbrace$).
2. For each, measure the average L0 (number of non-zero activations per input).
3. Plot reconstruction error vs. L0.
4. Choose the $\lambda$ that gives the desired L0 (often 10-50 active features per input for interpretability).

**The L0 metric:** The L0 "norm" counts the number of non-zero entries: $\Vert \mathbf{z}\Vert \_0 = |\lbrace j : z\_j \neq 0\rbrace |$. This is not differentiable (and is not actually a norm), but it is the most natural measure of sparsity. Report it alongside reconstruction quality when evaluating SAEs.

**Learning rate:** SAEs can be sensitive to learning rate. Too high and neurons die rapidly; too low and training is slow. Adam with $\text{lr} \in [10^{-4}, 10^{-3}]$ is a reasonable starting point. Some practitioners use learning rate warmup to reduce early neuron death.

### 5.4 Decoder Column Normalization

**The problem:** Without constraints on the decoder, the SAE can "cheat" the sparsity penalty. Here is how:

Consider the L1-penalized loss:
$$
\mathcal{L} = \Vert \mathbf{x} - \mathbf{W}_d \mathbf{z}\Vert _2^2 + \lambda \Vert \mathbf{z}\Vert _1
$$

Suppose the network scales its decoder by a factor $\alpha > 1$ and its encoder by $1/\alpha$:
$$
\mathbf{z}' = \mathbf{z}/\alpha, \quad \mathbf{W}_d' = \alpha \mathbf{W}_d
$$

Then:
- Reconstruction is unchanged: $\mathbf{W}\_d' \mathbf{z}' = \alpha \mathbf{W}\_d \cdot \mathbf{z}/\alpha = \mathbf{W}\_d \mathbf{z}$
- But L1 penalty *decreases*: $\lambda \Vert \mathbf{z}'\Vert \_1 = \lambda \Vert \mathbf{z}\Vert \_1 / \alpha < \lambda \Vert \mathbf{z}\Vert \_1$

By making $\alpha$ arbitrarily large, the network can make the L1 penalty arbitrarily small without any cost to reconstruction! The "sparsity" is fake -- the activations are small numbers, not zeros.

**The solution:** Constrain each column of the decoder to have unit norm:

$$
\Vert \mathbf{W}_d[:, j]\Vert _2 = 1 \quad \text{for all } j
$$

With this constraint, scaling the decoder by $\alpha$ would violate the norm constraint, so the cheating strategy does not work. The L1 penalty on activations then truly measures the "importance" of each feature.

**Implementation:** After each gradient step, project each decoder column to unit norm:

$$
\mathbf{W}_d[:, j] \leftarrow \frac{\mathbf{W}_d[:, j]}{\Vert \mathbf{W}_d[:, j]\Vert _2}
$$

This is a simple post-gradient-step projection, not a reparameterization. It is cheap and effective.

An alternative (used in some implementations) is to include the decoder norms in the loss as a soft constraint rather than a hard projection. But the hard projection is more common in practice because it is simpler and removes one more hyperparameter.

---

## 6. SAE as Amortized Sparse Coding (Revisited)

### 6.1 The Classical Sparse Coding Problem

Recall from Week 9 that sparse coding solves, for each input $\mathbf{x}$:

$$
\mathbf{z}^* = \arg\min_{\mathbf{z}} \Vert \mathbf{x} - \mathbf{D}\mathbf{z}\Vert _2^2 + \lambda \Vert \mathbf{z}\Vert _1
$$

where $\mathbf{D} \in \mathbb{R}^{n \times d}$ is a dictionary. This is solved iteratively (e.g., by ISTA) for *each input* -- an inner optimization loop nested inside the outer loop that learns $\mathbf{D}$.

### 6.2 The Amortization Insight

A sparse autoencoder replaces the inner optimization with a feedforward pass:

$$
\mathbf{z} = f_{\text{enc}}(\mathbf{x}) = \text{ReLU}(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)
$$

The encoder *learns to approximate* the iterative solver. Instead of running ISTA for 100 iterations to find $\mathbf{z}^*$, we train a neural network that produces an approximate $\mathbf{z}$ in a single forward pass.

This is called **amortized inference** -- we pay a one-time training cost so that future inference is cheap. The analogy: memorizing multiplication tables (expensive upfront, cheap per query) vs. computing each multiplication from scratch (cheap upfront, expensive per query).

### 6.3 Making the Connection Precise

| Sparse Coding | Sparse Autoencoder |
|---|---|
| Dictionary $\mathbf{D}$ | Decoder weights $\mathbf{W}\_d$ |
| Sparse code $\mathbf{z}^* = \arg\min$ | Hidden activations $\mathbf{z} = f\_{\text{enc}}(\mathbf{x})$ |
| ISTA / FISTA (iterative) | Encoder (feedforward) |
| $\Vert \mathbf{x} - \mathbf{D}\mathbf{z}\Vert ^2 + \lambda\Vert \mathbf{z}\Vert \_1$ | $\Vert \mathbf{x} - \hat{\mathbf{x}}\Vert ^2 + \lambda\Vert \mathbf{z}\Vert \_1$ |
| Learn $\mathbf{D}$ only | Learn $\mathbf{W}\_e, \mathbf{b}\_e, \mathbf{W}\_d, \mathbf{b}\_d$ jointly |

The decoder weights $\mathbf{W}\_d$ play the role of the dictionary $\mathbf{D}$. Each column of $\mathbf{W}\_d$ is a "dictionary atom" or "feature direction." The reconstruction $\hat{\mathbf{x}} = \mathbf{W}\_d \mathbf{z}$ expresses the input as a sparse linear combination of these atoms.

### 6.4 When Does Amortization Help?

Amortization is most valuable when:
- You have many inputs and need fast inference
- The optimization landscape is smooth enough that a feedforward network can approximate the solution well
- Approximate solutions are acceptable (we do not need the global optimum)

The cost: the encoder may not find the *exact* optimum of the sparse coding problem. It trades optimality for speed. In practice, well-trained SAE encoders produce codes that are very close to the ISTA solution, especially for inputs similar to the training distribution.

### 6.5 A Concrete Comparison

Suppose we have 10,000 input vectors and a dictionary with 2,000 atoms.

**Sparse coding (ISTA):** For each of the 10,000 inputs, run 100 iterations of ISTA. Total: 1,000,000 optimization steps.

**SAE:** Run a single forward pass through the encoder for each input. Total: 10,000 forward passes.

If both achieve similar reconstruction quality and sparsity, the SAE wins dramatically on inference speed. The training cost of the SAE is higher (we have to learn the encoder), but this is a one-time cost.

---

## 7. Evaluating Sparse Autoencoders

How do we know if an SAE is any good? There is no single metric -- you need to evaluate along multiple axes.

### 7.1 Reconstruction Quality

**MSE (Mean Squared Error):**
$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} \Vert \mathbf{x}_i - \hat{\mathbf{x}}_i\Vert _2^2
$$

Lower is better. But MSE alone is not sufficient -- an SAE with no sparsity penalty will have the lowest MSE (possibly zero) but useless features.

**Explained Variance:**
$$
R^2 = 1 - \frac{\sum_i \Vert \mathbf{x}_i - \hat{\mathbf{x}}_i\Vert ^2}{\sum_i \Vert \mathbf{x}_i - \bar{\mathbf{x}}\Vert ^2}
$$

Values close to 1 mean the SAE captures most of the variance. This normalizes for the scale of the data.

**Cross-Entropy Loss Recovered (for language model SAEs):**
When an SAE is trained on a language model's internal activations, we can measure quality by how well the reconstructed activations preserve the model's behavior. If the original model achieves cross-entropy loss $L\_{\text{orig}}$ and the model with SAE-reconstructed activations achieves $L\_{\text{SAE}}$, then:

$$
\text{CE loss recovered} = 1 - \frac{L_{\text{SAE}} - L_{\text{orig}}}{L_{\text{zero}} - L_{\text{orig}}}
$$

where $L\_{\text{zero}}$ is the loss when replacing activations with zeros (the worst case). A value of 1.0 means the SAE introduces no degradation; 0.0 means it is as bad as zeroing out activations entirely. Good SAEs typically achieve 0.95+ on this metric.

### 7.2 Sparsity Metrics

**L0 (average number of active features per input):**
$$
\text{L0} = \frac{1}{m} \sum_{i=1}^{m} \Vert \mathbf{z}_i\Vert _0
$$

This is the most interpretable sparsity metric. For mechanistic interpretability work, typical targets are L0 in the range of 10-100.

**L1 (average sum of activations):**
$$
\text{L1} = \frac{1}{m} \sum_{i=1}^{m} \Vert \mathbf{z}_i\Vert _1
$$

This is what the L1 penalty directly optimizes. It is correlated with L0 but also accounts for activation magnitudes.

**Feature density:** What fraction of the hidden neurons are "alive" (activated by at least one input in a large batch)? Ideally close to 1.0 -- we want every neuron to contribute.

### 7.3 The Pareto Frontier

The most informative evaluation plots **reconstruction quality vs. sparsity** across different $\lambda$ values, producing a Pareto frontier:

```
Reconstruction   |
Error (MSE)      |
                 | *  (lambda too large)
                 |
                 |   *
                 |     *
                 |       *  <-- Pareto frontier
                 |         *
                 |           * *
                 |                * (lambda too small)
                 |________________________
                       L0 (avg active features)
                 fewer <--           --> more
```

A better SAE shifts this frontier down and to the left (less error for the same sparsity, or more sparsity for the same error). Comparing different architectures or training strategies using this Pareto frontier is the gold standard for SAE evaluation.

### 7.4 Feature Interpretability

This is the hardest to quantify but often the most important. Methods include:

**Visual inspection (for image data):** Visualize each decoder column $\mathbf{W}\_d[:, j]$ reshaped as an image. Do the features look like meaningful patterns (edges, strokes, textures)?

**Max-activating examples:** For each feature $j$, find the inputs $\mathbf{x}\_i$ that produce the highest activation $z\_j(\mathbf{x}\_i)$. Do these inputs share a coherent property?

**Automated interpretability (for language model SAEs):** Feed the max-activating examples to another language model and ask it to describe what they have in common. This scales to thousands of features but has obvious limitations (the describing model might hallucinate patterns).

**Ablation studies:** Zero out feature $j$ and observe the change in model behavior. If removing a feature consistently changes one specific behavior, that is strong evidence the feature is causally meaningful.

### 7.5 Downstream Task Performance

If the goal is representation learning, we can evaluate SAE features by how useful they are for downstream tasks:

1. Train an SAE on the data.
2. Encode the data: $\mathbf{z}\_i = f\_{\text{enc}}(\mathbf{x}\_i)$ for all inputs.
3. Train a simple classifier (e.g., logistic regression) on $\mathbf{z}\_i$ to predict labels.
4. Compare accuracy to classifiers trained on: raw inputs, PCA features, features from undercomplete AEs.

If SAE features are more useful than PCA features (for a simple classifier), the SAE has learned something genuinely helpful -- it has not just found a linear projection.

---

## 8. A Complete Example: SAE on MNIST

Let us trace through the entire pipeline concretely.

### 8.1 Setup

- **Data:** MNIST, 60,000 images of $28 \times 28 = 784$ pixels, flattened to $\mathbf{x} \in \mathbb{R}^{784}$.
- **Architecture:** Encoder: $784 \to 2000$ (ReLU). Decoder: $2000 \to 784$ (linear). Expansion factor: $r = 2000/784 \approx 2.6$.
- **Loss:** $\mathcal{L} = \Vert \mathbf{x} - \hat{\mathbf{x}}\Vert \_2^2 + 0.005 \cdot \Vert \mathbf{z}\Vert \_1$
- **Optimizer:** Adam, lr $= 3 \times 10^{-4}$
- **Decoder norm constraint:** Project decoder columns to unit norm after each step.

### 8.2 What to Expect

After training for around 50 epochs:

**Reconstruction:** MSE around $10^{-3}$ to $10^{-2}$. Reconstructed digits are clearly recognizable, perhaps slightly blurry.

**Sparsity:** Average L0 around 20-50 (out of 2000). Each digit activates about 1-2.5% of hidden neurons.

**Features:** Visualizing the 2000 decoder columns (each reshaped to $28 \times 28$) reveals:
- Short oriented strokes at various positions and angles
- Curve segments (arcs, hooks)
- Some more global features (overall shape templates)
- A few dead neurons (featureless noise patterns)

The features look remarkably like the Gabor-like filters that Olshausen and Field (Week 9) found in natural images, adapted to the structure of handwritten digits.

### 8.3 Comparison to Other Methods

| Method | Features Look Like | Sparsity | Interpretability |
|---|---|---|---|
| PCA ($k=50$) | Global, blurry eigendigits | Dense | Low |
| Undercomplete AE ($d=50$) | Abstract, distributed | Dense | Low-Medium |
| Dictionary Learning ($d=2000$) | Localized strokes and edges | Sparse | High |
| **SAE** ($d=2000$, L1) | Localized strokes and edges | Sparse | High |
| SAE ($d=2000$, KL) | Similar to L1, slightly different selection | Sparse | High |

SAE features and dictionary learning features should look qualitatively similar -- this is the amortization insight from Section 6. The SAE just learns them faster and applies them faster.

---

## 9. The Bigger Picture: Why SAEs Matter

We have spent a lot of time on the mechanics of SAEs. Let us step back and ask: why should we care?

### 9.1 For Representation Learning

SAEs provide a principled way to learn overcomplete, sparse, interpretable representations of data. The features they discover often correspond to meaningful attributes of the data -- edges in images, strokes in handwriting, parts of objects. This makes them useful for:
- Feature extraction for downstream tasks
- Data visualization and exploration
- Anomaly detection (unusual inputs activate unusual feature combinations)

### 9.2 For Understanding Neural Networks

This is the application that makes SAEs one of the most exciting tools in modern AI research, and the subject of Weeks 11-12.

Modern neural networks (GPT-4, Claude, etc.) are powerful but opaque. They have billions of parameters and we have essentially no idea how they work internally. SAEs offer a path toward understanding: if we can decompose a neural network's internal representations into sparse, interpretable features, we might be able to understand *what concepts the network has learned* and *how it uses them*.

The key insight (which we will develop fully next week): neural networks may represent many more "features" (concepts, patterns) than they have dimensions. They do this by packing features into overlapping directions -- a phenomenon called **superposition**. SAEs, with their overcomplete and sparse structure, are ideally suited to undo this packing and recover the individual features.

### 9.3 The Road Ahead

- **Week 11:** We will study the superposition hypothesis in detail and understand *why* SAEs are the right tool for decomposing neural network representations.
- **Week 12:** We will see SAEs applied to real language models, discovering interpretable features like "code syntax," "French text," and "expressions of uncertainty."

This week, make sure you deeply understand the mechanics -- architecture, loss functions, training challenges. Next week, we will need all of it.

---

## 10. Summary

| Concept | Key Idea |
|---------|----------|
| SAE architecture | Overcomplete hidden layer ($d > n$) + sparsity constraint on activations |
| L1 penalty | $\lambda \sum\_j \Vert z\_j\Vert$ -- direct, per-example, drives values to zero |
| KL penalty | $\beta \sum\_j \text{KL}(\rho \Vert \hat{\rho}\_j)$ -- targets average activation per neuron |
| Dead neurons | Neurons stuck at zero activation; fix with resampling, LeakyReLU, or auxiliary loss |
| Feature splitting | One concept split across multiple neurons; fix with stronger sparsity or clustering |
| Decoder norm constraint | Prevents trivial "cheat" of shrinking decoder and inflating encoder |
| Amortized sparse coding | SAE encoder approximates iterative solver (ISTA) in a single forward pass |
| Evaluation | Reconstruction (MSE), sparsity (L0, L1), interpretability, downstream task performance |

---

## Further Reading

- **Andrew Ng**, "Sparse Autoencoder," CS294A Lecture Notes (2011). The classic reference for SAEs with KL divergence penalty. Clear and accessible.

- **Makhzani and Frey**, "k-Sparse Autoencoders" (2013). An alternative approach: instead of L1 penalty, simply keep the top-$k$ activations and zero out the rest. Clean and effective.

- **Sharkey et al.**, "Taking features out of superposition with sparse autoencoders" (2022). Connects SAEs to superposition and shows they can recover features from toy models -- a bridge to Week 11.

- **Cunningham et al.**, "Sparse Autoencoders Find Highly Interpretable Features in Language Models" (2023). An early application of SAEs to language models.

---

*Next week: Why do we need overcomplete representations to understand neural networks? The answer lies in a phenomenon called superposition -- and it changes how we think about what neurons in a neural network "mean."*
