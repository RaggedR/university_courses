---
title: "Week 3: Neural Networks from First Principles"
---

# Week 3: Neural Networks from First Principles

> *"A neural network is just a function from numbers to numbers, built by composing simple pieces."*
> — Your sensible inner voice, cutting through the hype

We've now rebuilt our linear algebra (Week 1) and probability (Week 2). It's time to build neural networks. But we're going to do it slowly and deliberately — understanding every component, questioning every design choice, and computing everything by hand at least once before reaching for PyTorch. If you understand neural networks at the level of "I can implement one with nothing but NumPy and basic calculus," then the autoencoder variants in Weeks 6-10 will feel like natural extensions rather than mysterious new architectures.

A neural network is, at bottom, a parameterized function. The art is in choosing the right structure (architecture) and finding the right parameters (training). This week we cover architecture and forward computation. Next week we'll cover training (backpropagation and optimization).

---

## 1. The Perceptron

### 1.1 A Single Neuron

The simplest neural network has one neuron. It takes an input vector $\mathbf{x} \in \mathbb{R}^d$, computes a weighted sum, adds a bias, and passes the result through an activation function:

$$
y = \sigma(\mathbf{w}^T\mathbf{x} + b) = \sigma\left(\sum_{i=1}^d w_i x_i + b\right)
$$

where $\mathbf{w} \in \mathbb{R}^d$ is the **weight vector**, $b \in \mathbb{R}$ is the **bias**, and $\sigma$ is the **activation function**.

If $\sigma$ is the step function ($\sigma(z) = 1$ if $z > 0$, else $0$), this is the **perceptron** — Frank Rosenblatt's 1958 model, inspired by biological neurons.

### 1.2 Decision Boundary Geometry

The perceptron classifies by testing which side of a hyperplane the input falls on. The equation $\mathbf{w}^T\mathbf{x} + b = 0$ defines a hyperplane in $\mathbb{R}^d$:

- $\mathbf{w}$ is the **normal vector** to the hyperplane (it points toward the positive side)
- $b$ controls the **offset** from the origin
- The distance from the origin to the hyperplane is $|b|/\Vert \mathbf{w}\Vert$

**Concrete example in $\mathbb{R}^2$.** Let $\mathbf{w} = (2, 1)^T$ and $b = -3$. The decision boundary is $2x\_1 + x\_2 = 3$, a line in the plane. Points above the line (where $2x\_1 + x\_2 > 3$) are classified as positive; points below as negative.

The weight vector $(2, 1)$ tells us that $x\_1$ matters twice as much as $x\_2$ for the classification decision. The bias $-3$ shifts the boundary away from the origin. This geometric reading of parameters is useful and often overlooked — every parameter in a neural network has a geometric interpretation, even when the geometry is in high dimensions.

### 1.3 The Perceptron Learning Algorithm

Given labeled training data $\lbrace (\mathbf{x}\_i, y\_i)\rbrace$ with $y\_i \in \lbrace -1, +1\rbrace$, the perceptron learning algorithm updates the weights whenever it makes a mistake:

```
Initialize w = 0, b = 0
For each training example (x_i, y_i):
    If y_i * (w^T x_i + b) <= 0:      # misclassified
        w = w + y_i * x_i
        b = b + y_i
```

**Theorem (Perceptron Convergence).** If the data is linearly separable (there exists a hyperplane that correctly classifies all points), the perceptron algorithm converges in a finite number of steps. The number of steps depends on the **margin** — the distance from the closest point to the separating hyperplane. Larger margin = fewer steps.

This is a clean result, but it has a devastating limitation: the data must be linearly separable. For data that isn't, the perceptron never converges — it oscillates forever, adjusting the boundary back and forth without settling.

### 1.4 The XOR Problem

The **XOR function** maps:

| $x\_1$ | $x\_2$ | $y$ |
|--------|--------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plot these four points. The positive examples $(0,1)$ and $(1,0)$ are on opposite corners of a square; the negative examples $(0,0)$ and $(1,1)$ are on the other two corners. No single line can separate the positives from the negatives.

**Proof that XOR is not linearly separable.** Suppose there exist $w\_1, w\_2, b$ such that:

- $w\_1 \cdot 0 + w\_2 \cdot 0 + b \leq 0$ (classify (0,0) as 0) $\Rightarrow$ $b \leq 0$
- $w\_1 \cdot 0 + w\_2 \cdot 1 + b > 0$ (classify (0,1) as 1) $\Rightarrow$ $w\_2 + b > 0$
- $w\_1 \cdot 1 + w\_2 \cdot 0 + b > 0$ (classify (1,0) as 1) $\Rightarrow$ $w\_1 + b > 0$
- $w\_1 \cdot 1 + w\_2 \cdot 1 + b \leq 0$ (classify (1,1) as 0) $\Rightarrow$ $w\_1 + w\_2 + b \leq 0$

Adding the second and third inequalities: $w\_1 + w\_2 + 2b > 0$, so $w\_1 + w\_2 > -2b \geq 0$. But the fourth inequality says $w\_1 + w\_2 \leq -b \leq 0$. Contradiction.

This is Minsky and Papert's famous 1969 result. It caused a temporary collapse of interest in neural networks (the first "AI winter"). The resolution: use more than one layer.

### 1.5 Solving XOR with Two Layers

XOR can be written as $(x\_1 \text{ OR } x\_2) \text{ AND NOT } (x\_1 \text{ AND } x\_2)$. Each of these sub-functions IS linearly separable. A two-layer network can compute each one separately and combine them:

**Layer 1 (two hidden neurons):**
- Neuron 1 computes OR: $h\_1 = \sigma(x\_1 + x\_2 - 0.5)$
- Neuron 2 computes NAND: $h\_2 = \sigma(-x\_1 - x\_2 + 1.5)$

**Layer 2 (one output neuron):**
- Output computes AND: $y = \sigma(h\_1 + h\_2 - 1.5)$

(Using step activation $\sigma$.) The first layer creates a new representation $(h\_1, h\_2)$ in which the XOR problem IS linearly separable. That's the key insight: the hidden layer transforms the data into a representation where the problem becomes easy.

This principle — **learned feature extraction** — is the entire foundation of deep learning. Each layer learns a representation that makes the next layer's job easier.

---

## 2. Multi-Layer Perceptrons

### 2.1 Architecture

A **multi-layer perceptron (MLP)** or **feedforward neural network** is a sequence of layers:

$$
\mathbf{x} \xrightarrow{W_1, \mathbf{b}_1} \mathbf{h}_1 \xrightarrow{W_2, \mathbf{b}_2} \mathbf{h}_2 \xrightarrow{} \cdots \xrightarrow{W_L, \mathbf{b}_L} \mathbf{y}
$$

Each layer applies a linear transformation followed by a nonlinear activation:

$$
\mathbf{h}_l = \sigma_l(W_l \mathbf{h}_{l-1} + \mathbf{b}_l)
$$

where $\mathbf{h}\_0 = \mathbf{x}$ is the input. The **parameters** of the network are all the weights $\lbrace W\_l\rbrace$ and biases $\lbrace \mathbf{b}\_l\rbrace$.

**Terminology:**
- **Input layer:** the raw input $\mathbf{x}$
- **Hidden layers:** $\mathbf{h}\_1, \ldots, \mathbf{h}\_{L-1}$ — not directly observed
- **Output layer:** $\mathbf{h}\_L = \mathbf{y}$ — the network's prediction
- **Depth:** the number of layers (usually counting hidden + output)
- **Width:** the number of neurons in a layer

### 2.2 Hidden Layers as Feature Extraction

Each hidden layer computes a new representation of the input. The first layer extracts simple features (edges, in images; character n-grams, in text). Subsequent layers combine these into increasingly complex features (shapes, then objects, then scenes).

This is not metaphorical. In trained image classifiers, you can literally visualize what each layer detects:

- **Layer 1:** edges at different orientations and positions
- **Layer 2:** corners, textures, color gradients
- **Layer 3:** parts of objects (eyes, wheels, petals)
- **Layer 4+:** whole objects and scenes

The hidden layers are doing exactly what we want autoencoders to do: learning a useful representation. The difference is that in a classifier, the representation is optimized for a specific task (classification), while in an autoencoder, it's optimized for reconstruction — which forces it to capture the general structure of the data.

### 2.3 Width vs. Depth

Two choices define a network's architecture: how wide and how deep.

**Width** (more neurons per layer): more capacity to represent complex functions at each level of abstraction. Very wide networks can memorize training data — they have enough parameters to "look up" every example. But this doesn't generalize.

**Depth** (more layers): enables compositional, hierarchical representations. Each layer builds on the previous one, creating abstractions at increasing levels. Deep networks can represent functions efficiently that would require exponentially many neurons in a shallow network.

**The practical answer:** Depth matters more than width for most tasks, but you need "enough" width too. The right architecture depends on the problem and is found empirically. For our autoencoders, we'll use relatively shallow networks (2-5 layers in each half) with varying widths.

### 2.4 The Computational Graph Perspective

Every neural network computation can be viewed as a **directed acyclic graph (DAG)**, where:
- Each node is an operation (matrix multiply, add, apply activation)
- Edges represent data flow (tensors flowing between operations)

This perspective is crucial for two reasons:
1. **Forward pass** = evaluate the graph from inputs to outputs
2. **Backward pass** = reverse the graph to compute gradients (Week 4)

PyTorch and other deep learning frameworks build this graph automatically. When you write `y = model(x)`, PyTorch records every operation, creating a graph that it can later traverse backward to compute gradients. This is **automatic differentiation** — we'll explore it in depth next week.

---

## 3. Activation Functions

### 3.1 Why Nonlinearity is Essential

Without activation functions, a multi-layer network collapses to a single linear transformation:

$$
W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2) = W'\mathbf{x} + \mathbf{b}'
$$

No matter how many layers you stack, the result is always linear. The activation function is what gives neural networks the ability to learn nonlinear relationships. It's the bridge between the linear algebra of Week 1 and the complex functions networks actually learn.

### 3.2 The Classic: Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Properties:**
- Range: $(0, 1)$ — outputs can be interpreted as probabilities
- Smooth, monotonically increasing
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- Saturates at both ends: when $|z|$ is large, $\sigma'(z) \approx 0$

**The vanishing gradient problem.** The maximum value of $\sigma'(z)$ is $0.25$ (at $z = 0$). In a deep network, gradients are multiplied across layers. If each layer contributes a factor of at most $0.25$, the gradient after $L$ layers is at most $0.25^L$. For $L = 10$: $0.25^{10} \approx 10^{-6}$. The gradient effectively vanishes — early layers learn extremely slowly.

This is not a theoretical curiosity. It's the primary reason deep networks couldn't be trained effectively until around 2010, and it's why sigmoid has been largely replaced by ReLU for hidden layers. (Sigmoid is still used in the output layer for binary classification, where its probabilistic interpretation is useful.)

### 3.3 Tanh

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1
$$

**Properties:**
- Range: $(-1, 1)$ — zero-centered, unlike sigmoid
- Derivative: $\tanh'(z) = 1 - \tanh^2(z)$
- Maximum derivative: $1$ (at $z = 0$) — better than sigmoid's $0.25$
- Still saturates for large $|z|$

Tanh was preferred over sigmoid for hidden layers because its zero-centered outputs help with gradient flow. But it still suffers from vanishing gradients in deep networks.

### 3.4 ReLU: The Modern Default

$$
\text{ReLU}(z) = \max(0, z)
$$

**Properties:**
- Range: $[0, \infty)$
- Derivative: $1$ for $z > 0$, $0$ for $z < 0$ (undefined at $z = 0$, typically set to $0$)
- **No saturation** for positive inputs — gradient is always $1$
- Extremely cheap to compute (just a comparison)
- Introduces sparsity: negative inputs map to exactly $0$

ReLU solved the vanishing gradient problem for positive activations and is computationally faster than sigmoid/tanh. It became the default activation function around 2012 (with the AlexNet breakthrough) and remains dominant.

**The dying ReLU problem.** If a neuron's input is always negative (perhaps due to a large negative bias learned during training), it always outputs zero. Since the gradient is also zero, it can never recover. The neuron "dies" — it stops learning permanently. This can happen to a significant fraction of neurons during training.

**Concrete example.** Consider a neuron with weights $\mathbf{w} = (-1, -1)^T$ and bias $b = -5$. For any non-negative input $(x\_1, x\_2)$, the pre-activation $-x\_1 - x\_2 - 5 < 0$, so $\text{ReLU}$ outputs 0. This neuron is dead and will stay dead because the gradient through a dead ReLU is zero.

### 3.5 ReLU Variants

Several modifications address the dying ReLU problem:

**Leaky ReLU:**
$$
\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\\\ \alpha z & z \leq 0 \end{cases}
$$
where $\alpha$ is a small positive constant (typically $0.01$). The small negative slope ensures the gradient is never exactly zero.

**GELU (Gaussian Error Linear Unit):**
$$
\text{GELU}(z) = z \cdot \Phi(z)
$$
where $\Phi$ is the CDF of the standard normal distribution. GELU is smooth, non-monotonic, and has become the activation of choice in transformer models. It can be approximated as $0.5z(1 + \tanh[\sqrt{2/\pi}(z + 0.044715z^3)])$.

**SiLU/Swish:**
$$
\text{SiLU}(z) = z \cdot \sigma(z)
$$
Similar to GELU in practice. Smooth, allows small negative outputs.

### 3.6 Choosing an Activation Function

| Activation | Use Case | Why |
|-----------|----------|-----|
| ReLU | Default for hidden layers | Fast, no vanishing gradient, sparsity |
| Leaky ReLU | When dead neurons are a concern | Prevents dying ReLU |
| GELU | Transformer models | Empirically best for attention mechanisms |
| Sigmoid | Output layer (binary classification) | Outputs a probability |
| Softmax | Output layer (multi-class classification) | Outputs a probability distribution |
| None (linear) | Output layer (regression) | No bounds on output range |

For the autoencoders in this course, we'll typically use ReLU for hidden layers and sigmoid for the decoder output (when inputs are in $[0, 1]$, like pixel values).

---

## 4. The Universal Approximation Theorem

### 4.1 Statement

**Theorem (Cybenko, 1989; Hornik, 1991).** Let $\sigma$ be any continuous, non-constant, bounded activation function (e.g., sigmoid). Then for any continuous function $f$ on a compact subset of $\mathbb{R}^n$ and any $\epsilon > 0$, there exists a single-hidden-layer neural network $g$ such that:

$$
|f(\mathbf{x}) - g(\mathbf{x})| < \epsilon \quad \text{for all } \mathbf{x} \text{ in the domain}
$$

The result has been extended to ReLU and other unbounded activations.

### 4.2 Intuition: How a Single Layer Approximates

Here's the idea for 1D functions with sigmoid activation. Each hidden neuron $\sigma(wx + b)$ is roughly a "soft step" — it transitions from 0 to 1 over a small interval. By combining many such neurons with different positions and heights, you can approximate any shape:

1. Each neuron creates a "bump" when you subtract two shifted sigmoids: $\sigma(w(x-a) + c) - \sigma(w(x-b) + c)$ is approximately 1 on $[a, b]$ and 0 elsewhere (for large $w$).
2. Scale each bump to the right height.
3. Sum them up to approximate the target function like a bar chart.

With enough neurons (width), you can make the bar chart as fine-grained as needed.

### 4.3 What the Theorem Does NOT Say

This is worth emphasizing:

1. **It doesn't say the network is easy to train.** Gradient descent might not find the right parameters. The theorem is about existence, not constructability.

2. **It doesn't say the network is efficient.** The required width might be exponential in the input dimension. A network with $2^n$ neurons might be needed to approximate a function on $\mathbb{R}^n$ to a given precision.

3. **It doesn't say one hidden layer is better than multiple.** Deep networks often approximate the same functions with exponentially fewer parameters than shallow ones. The theorem says shallow networks *can* work in principle, not that they *should* be preferred.

The universal approximation theorem is a theoretical guarantee that neural networks are expressive enough. The practical question is always about sample efficiency, computational cost, and generalization — which are about depth, architecture, and training, not just expressiveness.

### 4.4 Depth Separations

There exist functions that require exponentially many neurons in a shallow network but only polynomially many in a deep one. The classic example: the function that computes the parity of $n$ binary inputs. A shallow network needs $\Omega(2^n)$ neurons; a network of depth $O(\log n)$ needs only $O(n)$.

This is why depth matters. Deep networks learn hierarchical representations, and many natural functions have hierarchical structure. An image of a face is composed of parts (eyes, nose, mouth), which are composed of sub-parts (edges, textures), which are composed of pixels. A deep network mirrors this hierarchy; a shallow network must represent it all in one flat layer.

---

## 5. The Forward Pass

### 5.1 Matrix Multiplication View

A single fully-connected layer transforms an input $\mathbf{h} \in \mathbb{R}^{d\_{\text{in}}}$ to an output $\mathbf{h}' \in \mathbb{R}^{d\_{\text{out}}}$:

$$
\mathbf{z} = W\mathbf{h} + \mathbf{b} \quad \text{(pre-activation)}
$$
$$
\mathbf{h}' = \sigma(\mathbf{z}) \quad \text{(post-activation)}
$$

where $W \in \mathbb{R}^{d\_{\text{out}} \times d\_{\text{in}}}$ and $\mathbf{b} \in \mathbb{R}^{d\_{\text{out}}}$.

**Parameter count:** This single layer has $d\_{\text{out}} \times d\_{\text{in}} + d\_{\text{out}}$ parameters. For a layer with 512 input neurons and 256 output neurons: $512 \times 256 + 256 = 131{,}328$ parameters. Neural networks have a lot of parameters. Modern language models have billions.

### 5.2 Batch Processing

In practice, we process multiple inputs simultaneously as a **batch**. If $X \in \mathbb{R}^{B \times d\_{\text{in}}}$ is a batch of $B$ input vectors (one per row), the layer computation is:

$$
Z = XW^T + \mathbf{1}\mathbf{b}^T \quad \text{(broadcasting the bias)}
$$
$$
H' = \sigma(Z)
$$

where $Z, H' \in \mathbb{R}^{B \times d\_{\text{out}}}$. This is a single matrix multiplication — extremely efficient on GPUs, which are designed for exactly this operation.

**Why batching matters:** A GPU can multiply a $256 \times 512$ matrix by a $512 \times 256$ matrix about as fast as it can multiply a $1 \times 512$ vector by a $512 \times 256$ matrix. Processing 256 samples takes barely longer than processing 1. This is why mini-batch SGD (Week 1) is not just a statistical trick but a computational necessity.

### 5.3 Complete Worked Example

Let's trace a complete forward pass through a small network with actual numbers.

**Network architecture:** Input dimension 3, hidden layer with 2 neurons (ReLU), output layer with 1 neuron (sigmoid).

**Parameters:**
$$
W_1 = \begin{pmatrix} 0.5 & -0.3 & 0.8 \\\\ -0.2 & 0.7 & 0.1 \end{pmatrix}, \quad \mathbf{b}_1 = \begin{pmatrix} 0.1 \\\\ -0.1 \end{pmatrix}
$$

$$
W_2 = \begin{pmatrix} 0.6 & -0.4 \end{pmatrix}, \quad b_2 = 0.2
$$

**Input:** $\mathbf{x} = (1.0, 0.5, -1.0)^T$

**Layer 1 (pre-activation):**
$$
\mathbf{z}_1 = W_1\mathbf{x} + \mathbf{b}_1 = \begin{pmatrix} 0.5(1) + (-0.3)(0.5) + 0.8(-1) \\\\ (-0.2)(1) + 0.7(0.5) + 0.1(-1) \end{pmatrix} + \begin{pmatrix} 0.1 \\\\ -0.1 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0.5 - 0.15 - 0.8 \\\\ -0.2 + 0.35 - 0.1 \end{pmatrix} + \begin{pmatrix} 0.1 \\\\ -0.1 \end{pmatrix} = \begin{pmatrix} -0.45 \\\\ 0.05 \end{pmatrix} + \begin{pmatrix} 0.1 \\\\ -0.1 \end{pmatrix} = \begin{pmatrix} -0.35 \\\\ -0.05 \end{pmatrix}
$$

**Layer 1 (post-activation, ReLU):**
$$
\mathbf{h}_1 = \text{ReLU}(\mathbf{z}_1) = \begin{pmatrix} \max(0, -0.35) \\\\ \max(0, -0.05) \end{pmatrix} = \begin{pmatrix} 0 \\\\ 0 \end{pmatrix}
$$

Both neurons are dead for this input! ReLU clipped both pre-activations to zero.

**Layer 2 (pre-activation):**
$$
z_2 = W_2\mathbf{h}_1 + b_2 = 0.6(0) + (-0.4)(0) + 0.2 = 0.2
$$

**Layer 2 (post-activation, sigmoid):**
$$
y = \sigma(0.2) = \frac{1}{1 + e^{-0.2}} \approx 0.550
$$

**Result:** The network outputs $0.550$, which could be interpreted as $P(\text{positive class}) \approx 55\%$. Notice that the output is determined entirely by the bias $b\_2 = 0.2$, since both hidden neurons output zero. This is a real (if extreme) illustration of how ReLU can suppress information flow.

### 5.4 Notation Conventions

The notation in deep learning papers is unfortunately inconsistent. Here are the most common conventions:

| Convention | Layer index | Pre-activation | Post-activation | Weights |
|-----------|------------|---------------|----------------|---------|
| This course | Superscript: $l = 1, \ldots, L$ | $\mathbf{z}^{(l)}$ | $\mathbf{h}^{(l)}$ (or $\mathbf{a}^{(l)}$) | $W^{(l)}$ |
| Goodfellow | Same | $\mathbf{a}^{(l)}$ | $\mathbf{h}^{(l)}$ | $W^{(l)}$ |
| PyTorch | 0-indexed layers | N/A | accessed via hooks | `model.layer_name.weight` |

The key distinction to keep straight: **pre-activation** $\mathbf{z} = W\mathbf{h} + \mathbf{b}$ (before the nonlinearity) vs. **post-activation** $\mathbf{h}' = \sigma(\mathbf{z})$ (after). Some papers use $\mathbf{a}$ for one and $\mathbf{h}$ for the other; the letters vary but the concept doesn't.

---

## 6. Loss Functions

### 6.1 The Role of the Loss Function

The **loss function** $\mathcal{L}(\theta)$ measures how wrong the network is. Training is the process of minimizing this loss over the parameters $\theta = \lbrace W\_l, \mathbf{b}\_l\rbrace \_{l=1}^L$ using gradient descent (Week 1).

The choice of loss function is not arbitrary — it encodes our assumptions about the data and the task. As we saw in Week 2, many loss functions correspond to maximum likelihood under specific probabilistic models.

### 6.2 Mean Squared Error (MSE)

For regression (predicting continuous values):

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N \Vert \mathbf{y}_i - \hat{\mathbf{y}}_i\Vert ^2
$$

where $\mathbf{y}\_i$ is the true value and $\hat{\mathbf{y}}\_i = f\_\theta(\mathbf{x}\_i)$ is the prediction.

**Probabilistic interpretation (Week 2):** MSE corresponds to MLE under the assumption that the targets have Gaussian noise: $y\_i = f\_\theta(\mathbf{x}\_i) + \epsilon\_i$ with $\epsilon\_i \sim \mathcal{N}(0, \sigma^2)$.

MSE penalizes large errors quadratically — a prediction off by 10 is 100 times worse than one off by 1. This makes MSE sensitive to outliers.

### 6.3 Cross-Entropy Loss

For classification (predicting discrete categories):

**Binary cross-entropy (2 classes):**
$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right]
$$

where $y\_i \in \lbrace 0, 1\rbrace$ and $\hat{p}\_i = \sigma(f\_\theta(\mathbf{x}\_i))$ is the predicted probability.

**Categorical cross-entropy ($K$ classes):**
$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log \hat{p}_{ik}
$$

where $\hat{\mathbf{p}}\_i = \text{softmax}(f\_\theta(\mathbf{x}\_i))$ and $\text{softmax}(\mathbf{z})\_k = \frac{e^{z\_k}}{\sum\_j e^{z\_j}}$.

### 6.4 Why Cross-Entropy Beats MSE for Classification

Consider a binary classification problem where the true label is $y = 1$ and the model predicts probability $\hat{p}$.

**Cross-entropy gradient with respect to $\hat{p}$:**
$$
\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \hat{p}} = -\frac{1}{\hat{p}}
$$

When $\hat{p} = 0.01$ (very wrong): gradient $\approx -100$. Large gradient = fast learning.

**MSE gradient with respect to $\hat{p}$:**
$$
\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{p}} = -2(1 - \hat{p})
$$

When $\hat{p} = 0.01$ (very wrong): gradient $\approx -1.98$. Small gradient = slow learning.

The cross-entropy gradient is 50 times larger when the prediction is very wrong. Cross-entropy screams "you're wrong, fix this NOW!"; MSE just shrugs. This is why cross-entropy is strongly preferred for classification.

There's a deeper reason: the sigmoid function saturates for extreme values ($\sigma'(z) \approx 0$ when $|z|$ is large). For MSE, this saturation compounds with the small gradient, making learning extremely slow. For cross-entropy, the loss derivative $-1/\hat{p}$ exactly cancels the sigmoid saturation, producing a well-behaved gradient $\hat{p} - y$ that depends only on how wrong the prediction is, not on where we are on the sigmoid curve.

### 6.5 Loss Functions for Autoencoders

Looking ahead to Weeks 6-10, autoencoders use **reconstruction loss**: how well the decoder output matches the original input.

- **MSE reconstruction:** $\mathcal{L} = \Vert \mathbf{x} - \hat{\mathbf{x}}\Vert ^2$ — appropriate when inputs are continuous
- **Binary cross-entropy reconstruction:** $\mathcal{L} = -\sum\_j [x\_j \log \hat{x}\_j + (1-x\_j)\log(1-\hat{x}\_j)]$ — appropriate when inputs are in $[0, 1]$ (e.g., normalized pixel values)

The autoencoder loss will later be augmented with regularization terms (KL divergence in VAEs, sparsity penalties in SAEs), but the reconstruction loss is always the foundation.

---

## 7. PyTorch Introduction

### 7.1 Why PyTorch

PyTorch is the dominant deep learning framework in research (TensorFlow is more common in production). We use it because:

1. **Automatic differentiation:** You define the forward pass; PyTorch computes gradients for free.
2. **Dynamic computation graphs:** The graph is built on the fly, making debugging natural.
3. **NumPy-like interface:** If you know NumPy, you're most of the way there.

### 7.2 Tensors

A **tensor** is PyTorch's equivalent of a NumPy array, with two additions: it can live on a GPU, and it tracks operations for automatic differentiation.

```python
import torch

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])                    # from a list
x = torch.randn(3, 4)                                  # random normal, shape (3, 4)
x = torch.zeros(2, 3)                                  # all zeros
x = torch.from_numpy(numpy_array)                      # from NumPy

# Operations (same as NumPy)
y = x + 1                    # element-wise addition
y = x @ W                   # matrix multiplication
y = x.T                     # transpose
y = x.reshape(6, 1)         # reshape
y = x.sum(dim=0)            # sum along axis 0

# GPU
if torch.cuda.is_available():
    x = x.cuda()             # move to GPU
    x = x.to('cuda')         # alternative syntax
```

### 7.3 Autograd

PyTorch tracks operations on tensors with `requires_grad=True` and can compute gradients automatically:

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x[0]**2 + 3*x[1]     # y = x0^2 + 3*x1 = 4 + 9 = 13
y.backward()               # compute dy/dx
print(x.grad)              # tensor([4., 3.])  -- dy/dx0 = 2*x0 = 4, dy/dx1 = 3
```

This is the magic that makes training neural networks practical. You define the forward computation (any Python code involving tensors), call `.backward()`, and PyTorch gives you all the gradients. No manual derivative computation needed.

### 7.4 Building a Network with nn.Module

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.layer1(x))      # hidden layer
        y = self.sigmoid(self.layer2(h))    # output layer
        return y

# Create model
model = SimpleMLP(input_dim=784, hidden_dim=128, output_dim=10)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# 784*128 + 128 + 128*10 + 10 = 100,352 + 128 + 1,280 + 10 = 101,770
```

Key patterns:
- Inherit from `nn.Module`
- Define layers in `__init__`
- Define the forward pass in `forward`
- PyTorch handles the backward pass automatically

### 7.5 Training Loop

```python
import torch.optim as optim

# Model, loss, optimizer
model = SimpleMLP(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (one epoch)
for batch_x, batch_y in dataloader:
    # Forward pass
    predictions = model(batch_x)
    loss = criterion(predictions, batch_y)

    # Backward pass
    optimizer.zero_grad()    # clear old gradients
    loss.backward()          # compute new gradients
    optimizer.step()         # update parameters
```

The three-line backward pass (`zero_grad`, `backward`, `step`) is the universal training pattern in PyTorch. Every model we build in this course — classifiers, autoencoders, VAEs, sparse autoencoders — uses this same structure.

### 7.6 A Complete Example: MLP on Synthetic Data

Here's a complete, runnable example that ties together everything in this week:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset: two interleaving moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

# Define model
class MoonClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = MoonClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(500):
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        acc = ((pred > 0.5).float() == y_tensor).float().mean()
        print(f"Epoch {epoch}: loss={loss.item():.4f}, accuracy={acc.item():.4f}")
```

This 30-line script trains a two-layer MLP to classify the "two moons" dataset — a non-linearly-separable problem that a single perceptron cannot solve. The network learns a curved decision boundary that separates the two interleaving crescents.

---

## Summary

| Concept | Key Idea | Where It Shows Up Later |
|---------|----------|------------------------|
| Perceptron | Linear classifier; limited to linearly separable data | Foundation for all neural nets |
| Hidden layers | Transform data into representations where the task is easier | Encoder/decoder in autoencoders |
| Activation functions | Nonlinearity enables learning complex functions; ReLU is default | ReLU in SAEs introduces sparsity naturally |
| Universal approx. theorem | Wide enough networks can approximate anything (existence, not practice) | Theoretical justification for autoencoder capacity |
| Forward pass | Matrix multiplications + activations; efficient in batches | The encoder and decoder are forward passes |
| Loss functions | CE for classification, MSE for regression; both are MLE | Reconstruction loss in autoencoders |
| PyTorch | Define forward pass; autograd handles backward pass | Tool for all implementations |

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$ | Network input |
| $\mathbf{h}^{(l)}$ | Activation (post-nonlinearity) at layer $l$ |
| $\mathbf{z}^{(l)}$ | Pre-activation at layer $l$ |
| $W^{(l)}$, $\mathbf{b}^{(l)}$ | Weights and bias of layer $l$ |
| $\sigma(\cdot)$ | Activation function |
| $\hat{\mathbf{y}}$ | Network prediction |
| $\mathcal{L}$ | Loss function |
| $\theta$ | All network parameters collectively |

---

## Further Reading

- **Goodfellow et al.** *Deep Learning*, Chapter 6 (Deep Feedforward Networks). Available free at deeplearningbook.org. Comprehensive treatment of MLPs.
- **3Blue1Brown.** *Neural Networks* (YouTube, 4 videos). Beautiful visual explanation of what neural networks are doing geometrically.
- **PyTorch Tutorials.** pytorch.org/tutorials — start with "Deep Learning with PyTorch: A 60 Minute Blitz."
- **Minsky, M. and Papert, S.** *Perceptrons* (1969). The book that killed and later resurrected neural network research. Chapter 0 and the XOR analysis are worth reading for historical perspective.
- **Hornik, K.** "Approximation capabilities of multilayer feedforward networks" (1991). The original universal approximation theorem paper. Readable and short.
