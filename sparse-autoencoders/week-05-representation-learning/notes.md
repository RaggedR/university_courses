# Week 5: Representation Learning & Dimensionality Reduction

## Overview

For the past four weeks, we have been building mathematical machinery and learning to train neural networks. Now we ask the deeper question: *what do neural networks actually learn?*

The answer, it turns out, is **representations** — transformed versions of the input data that make downstream tasks easier. A well-trained image classifier does not just map pixels to labels; it builds an internal hierarchy of increasingly abstract features: edges, textures, parts, objects. These intermediate representations are often more valuable than the final classification itself.

This week we study representation learning from first principles. We begin with the classical approach — Principal Component Analysis (PCA) — and show that it is secretly a linear autoencoder in disguise. This connection between classical statistics and neural networks is the conceptual bridge to everything that follows.

---

## 1. What Are Features?

### 1.1 Raw Inputs vs. Learned Representations

Consider the problem of recognizing handwritten digits. The raw input is a $28 \times 28$ grayscale image — a vector in $\mathbb{R}^{784}$. But this representation is terrible for classification. Two images of the same digit can have vastly different pixel values (different handwriting styles, stroke widths, positions), while two images of different digits can be very similar in pixel space (a poorly written 3 vs. 8).

What we need is a **representation** — a transformation of the raw input into a space where the structure we care about (digit identity) becomes easy to extract. In a good representation:
- Same-class examples are nearby
- Different-class examples are far apart
- Irrelevant variation (style, position, stroke width) has been factored out

### 1.2 Hand-Crafted Features: The Old Way

Before deep learning, practitioners spent enormous effort designing features by hand:

**SIFT (Scale-Invariant Feature Transform):** Detect keypoints in images and describe them using local gradient histograms. Invariant to scale, rotation, and illumination changes. Took years of expert engineering to develop.

**HOG (Histogram of Oriented Gradients):** Divide an image into cells, compute gradient orientation histograms in each cell. Excellent for pedestrian detection. Again, years of domain-specific engineering.

**Bag of Words (NLP):** Represent a document as a vector of word counts, ignoring order. Simple but surprisingly effective for topic classification.

The pattern: experts study a domain, identify what matters, and design mathematical descriptors that capture those properties. This works — but it scales poorly. Each new domain or task requires new features. The features encode the designer's assumptions about what is relevant, which may be wrong or incomplete.

### 1.3 Learned Features: The Deep Learning Revolution

The core insight of deep learning: **let the network learn features from data**.

Instead of designing features by hand, we train a deep neural network end-to-end — from raw pixels to the final prediction. The intermediate layers are forced to discover useful features as a byproduct of minimizing the loss function. No human engineer decides what features to extract; the data and the gradient determine what the network learns.

This is why deep learning has dominated computer vision, natural language processing, and speech recognition since ~2012. The learned features are often *better* than hand-crafted ones, and they require no domain expertise to design.

But this raises new questions: what exactly are these learned features? Can we understand and interpret them? Can we learn useful features *without labels*? These are the questions that drive the rest of this course.

---

## 2. The Manifold Hypothesis

### 2.1 The Curse of Dimensionality

MNIST images live in $\mathbb{R}^{784}$. If we were to uniformly sample points in this space, virtually none of them would look like handwritten digits. The volume of a high-dimensional space is enormous, and the set of "realistic" images is a vanishingly small subset.

To get a sense of scale: a 28x28 grayscale image has $256^{784} \approx 10^{1888}$ possible pixel configurations. There are roughly $10^{80}$ atoms in the observable universe. The space of possible images is incomprehensibly vast, and meaningful images occupy an infinitesimal fraction of it.

### 2.2 Data Lies on Low-Dimensional Manifolds

The **manifold hypothesis** states: high-dimensional real-world data (images, text, audio) lies on or near a low-dimensional manifold embedded in the high-dimensional ambient space.

**What is a manifold?** Informally, a manifold is a smooth surface that can be locally approximated by a Euclidean space of lower dimension. The surface of a sphere is a 2-dimensional manifold embedded in 3D space: at any point, a small enough patch looks like a flat plane.

**Intuition for images:** Consider face images. A face image might be 1000x1000 pixels (dimension $3 \times 10^6$ for RGB), but the space of faces can be parameterized by a much smaller number of factors:
- Head pose (yaw, pitch, roll): 3 dimensions
- Lighting direction and intensity: ~4 dimensions
- Identity: maybe ~50 dimensions
- Expression: maybe ~10 dimensions
- Total: perhaps 50-100 dimensions

The manifold of face images is a ~100-dimensional surface winding through a 3-million-dimensional pixel space.

### 2.3 Why the Manifold Hypothesis Matters

If data lies on a low-dimensional manifold, then:

1. **Compression is possible.** We can find a low-dimensional representation that captures the essential structure. This is what autoencoders do.

2. **Distance in ambient space is misleading.** Two face images that differ only in lighting might be far apart in pixel space but close on the manifold. A good representation preserves manifold distances, not pixel distances.

3. **Interpolation is meaningful.** Moving along the manifold between two data points should produce realistic intermediate points. (This is why interpolation in the latent space of autoencoders produces plausible images.)

4. **Generalization is possible.** If the data manifold is low-dimensional, we can learn its structure from a reasonable number of samples, despite the high ambient dimension.

### 2.4 Evidence for the Manifold Hypothesis

The manifold hypothesis is supported by overwhelming empirical evidence:

- Dimensionality reduction methods (PCA, t-SNE, UMAP) consistently show that high-dimensional datasets have low intrinsic dimensionality
- Autoencoders can reconstruct images well from low-dimensional bottlenecks (we will see this next week)
- Interpolation in learned latent spaces produces realistic examples
- The success of deep learning itself: if data did not have low-dimensional structure, learning would require exponentially many samples

---

## 3. Principal Component Analysis (PCA)

PCA is the oldest and most fundamental dimensionality reduction method. It finds the directions of maximum variance in the data and projects onto those directions. We derive it from two perspectives, then show they are equivalent.

### 3.1 Setup

Given $N$ data points $\mathbf{x}\_1, \ldots, \mathbf{x}\_N \in \mathbb{R}^d$. Assume the data is centered: $\frac{1}{N}\sum\_i \mathbf{x}\_i = \mathbf{0}$. (If not, subtract the mean first.)

The data covariance matrix is:

$$
C = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i \mathbf{x}_i^\top = \frac{1}{N} X^\top X
$$

where $X \in \mathbb{R}^{N \times d}$ is the data matrix (each row is a data point). $C$ is $d \times d$, symmetric, and positive semi-definite.

### 3.2 Derivation: Maximum Variance Perspective

**Goal:** Find the unit vector $\mathbf{u}\_1 \in \mathbb{R}^d$ (with $\Vert \mathbf{u}\_1\Vert = 1$) such that the variance of the projected data is maximized.

The projection of $\mathbf{x}\_i$ onto $\mathbf{u}$ is the scalar $\mathbf{u}^\top \mathbf{x}\_i$. The variance of these projections is:

$$
\text{Var} = \frac{1}{N} \sum_{i=1}^N (\mathbf{u}^\top \mathbf{x}_i)^2 = \mathbf{u}^\top \left(\frac{1}{N} \sum_i \mathbf{x}_i \mathbf{x}_i^\top\right) \mathbf{u} = \mathbf{u}^\top C \mathbf{u}
$$

We want to maximize $\mathbf{u}^\top C \mathbf{u}$ subject to $\Vert \mathbf{u}\Vert ^2 = 1$.

**Using Lagrange multipliers:**

$$
\mathcal{L}(\mathbf{u}, \lambda) = \mathbf{u}^\top C \mathbf{u} - \lambda(\mathbf{u}^\top \mathbf{u} - 1)
$$

Setting the gradient to zero:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}} = 2C\mathbf{u} - 2\lambda\mathbf{u} = \mathbf{0}
$$

$$
C\mathbf{u} = \lambda\mathbf{u}
$$

This is an eigenvalue equation. The optimal $\mathbf{u}\_1$ is an eigenvector of $C$, and the variance it captures is:

$$
\mathbf{u}_1^\top C \mathbf{u}_1 = \mathbf{u}_1^\top (\lambda_1 \mathbf{u}_1) = \lambda_1
$$

To maximize variance, choose $\mathbf{u}\_1$ as the eigenvector corresponding to the **largest eigenvalue** $\lambda\_1$.

For the second principal component, we maximize variance subject to orthogonality to $\mathbf{u}\_1$. This gives the eigenvector for the second-largest eigenvalue $\lambda\_2$. And so on.

**PCA solution:** The $k$ principal components are the eigenvectors of $C$ corresponding to the $k$ largest eigenvalues:

$$
\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_k \quad \text{with} \quad \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_k
$$

The projection of a data point $\mathbf{x}$ onto these components gives the $k$-dimensional representation:

$$
\mathbf{z} = U_k^\top \mathbf{x} \in \mathbb{R}^k
$$

where $U\_k = [\mathbf{u}\_1, \ldots, \mathbf{u}\_k] \in \mathbb{R}^{d \times k}$.

### 3.3 Derivation: Minimum Reconstruction Error Perspective

**Goal:** Find a $k$-dimensional subspace that minimizes the reconstruction error when data is projected onto it and reconstructed.

The projection of $\mathbf{x}\_i$ onto the subspace spanned by $U\_k$ is:

$$
\hat{\mathbf{x}}_i = U_k U_k^\top \mathbf{x}_i
$$

($U\_k U\_k^\top$ is the projection matrix onto the subspace.) The reconstruction error is:

$$
\mathcal{E} = \frac{1}{N} \sum_{i=1}^N \Vert \mathbf{x}_i - \hat{\mathbf{x}}_i\Vert ^2 = \frac{1}{N} \sum_i \Vert \mathbf{x}_i - U_k U_k^\top \mathbf{x}_i\Vert ^2
$$

**To minimize this**, decompose the error using the Pythagorean theorem (since projection is orthogonal):

$$
\Vert \mathbf{x}_i\Vert ^2 = \Vert U_k U_k^\top \mathbf{x}_i\Vert ^2 + \Vert \mathbf{x}_i - U_k U_k^\top \mathbf{x}_i\Vert ^2
$$

Summing over data and rearranging:

$$
\mathcal{E} = \frac{1}{N}\sum_i \Vert \mathbf{x}_i\Vert ^2 - \frac{1}{N}\sum_i \Vert U_k U_k^\top \mathbf{x}_i\Vert ^2
$$

The first term is constant (total data variance). So minimizing reconstruction error is equivalent to **maximizing the variance captured by the projection** — which is:

$$
\frac{1}{N}\sum_i \Vert U_k^\top \mathbf{x}_i\Vert ^2 = \text{tr}(U_k^\top C \, U_k) = \sum_{j=1}^k \lambda_j
$$

This is maximized by choosing the top-$k$ eigenvectors, the same solution as before.

### 3.4 The Equivalence

We have derived PCA from two different perspectives:
1. **Maximize variance:** Find the directions that capture the most variance in the data
2. **Minimize reconstruction error:** Find the subspace that best approximates the data

These give the **same solution**: the top eigenvectors of the covariance matrix. This is not a coincidence — it follows from the Pythagorean theorem. Total variance = captured variance + lost variance. Maximizing captured variance is the same as minimizing lost variance (reconstruction error).

This equivalence is important for the autoencoder connection: autoencoders minimize reconstruction error, so when they are linear, they learn PCA.

### 3.5 Connection to SVD

PCA is intimately connected to the Singular Value Decomposition from Week 1.

If we compute the SVD of the centered data matrix $X = U \Sigma V^\top$, then:

$$
C = \frac{1}{N} X^\top X = \frac{1}{N} V \Sigma^2 V^\top
$$

The right singular vectors $V$ are the eigenvectors of $C$, and the eigenvalues are $\lambda\_j = \sigma\_j^2 / N$ where $\sigma\_j$ are the singular values.

In practice, PCA is often computed via SVD of the data matrix rather than eigendecomposition of the covariance matrix, because SVD is more numerically stable.

### 3.6 Choosing the Number of Components

How many components $k$ should we keep? Several approaches:

**Explained variance ratio:** The fraction of total variance captured by the top $k$ components:

$$
\frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j}
$$

A common rule of thumb: keep enough components to explain 95% or 99% of the variance.

**Scree plot:** Plot $\lambda\_j$ vs. $j$ (eigenvalue vs. component index). Look for an "elbow" — a sharp drop-off. Components after the elbow contribute little.

**Task-dependent:** Choose $k$ by cross-validation on the downstream task (classification accuracy with $k$ components).

### 3.7 A Concrete Example

Consider 2D data points clustered along a tilted ellipse:

```
     *  *
   *  *  *  *
  *  *  *  *  *       ← data lies along a tilted direction
   *  *  *  *
     *  *
```

If the major axis of the ellipse is along the direction $(0.8, 0.6)$ with eigenvalue 4.0, and the minor axis is along $(-0.6, 0.8)$ with eigenvalue 0.5, then:
- PC1 = $(0.8, 0.6)$, capturing $4.0 / 4.5 = 89\%$ of the variance
- PC2 = $(-0.6, 0.8)$, capturing $0.5 / 4.5 = 11\%$ of the variance

Projecting onto PC1 alone gives a 1D representation that preserves most of the data structure, at the cost of losing information about the spread along the minor axis.

For MNIST: the data lives in $\mathbb{R}^{784}$, but the top 50 principal components capture ~85% of the variance, and the top 150 capture ~95%. This confirms the manifold hypothesis — the intrinsic dimensionality is far lower than 784.

---

## 4. PCA as a Linear Autoencoder

This section is the conceptual heart of the week. It connects classical statistics (PCA) to neural networks (autoencoders), setting up everything that follows.

### 4.1 The Linear Autoencoder

Consider a neural network with one hidden layer, **no activation functions** (purely linear), and trained to reconstruct its input:

- Encoder: $\mathbf{z} = W\_e \mathbf{x}$, where $W\_e \in \mathbb{R}^{k \times d}$
- Decoder: $\hat{\mathbf{x}} = W\_d \mathbf{z}$, where $W\_d \in \mathbb{R}^{d \times k}$
- Reconstruction: $\hat{\mathbf{x}} = W\_d W\_e \mathbf{x}$

The loss is mean squared reconstruction error:

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \Vert \mathbf{x}_i - W_d W_e \mathbf{x}_i\Vert ^2
$$

Note that $W\_d W\_e$ is a $d \times d$ matrix of rank at most $k$ (since $W\_e$ maps to $k$ dimensions). So the autoencoder is finding the rank-$k$ linear map that best reconstructs the data.

### 4.2 The Theorem

**Claim:** A linear autoencoder with MSE loss, at any global minimum, satisfies:
$$
W_d W_e = U_k U_k^\top
$$
where $U\_k$ are the top $k$ eigenvectors of the data covariance matrix.

In other words: **the optimal linear autoencoder is PCA**.

### 4.3 Proof Sketch

The reconstruction error is:

$$
\mathcal{L} = \frac{1}{N}\sum_i \Vert \mathbf{x}_i - M\mathbf{x}_i\Vert ^2
$$

where $M = W\_d W\_e$ has rank $\leq k$. This is minimized by the rank-$k$ matrix $M^*$ that best approximates the identity in the Frobenius norm on the data:

$$
M^* = \arg\min_{\text{rank}(M) \leq k} \frac{1}{N}\sum_i \Vert (\mathbf{I} - M)\mathbf{x}_i\Vert ^2
$$

By the Eckart-Young-Mirsky theorem (which we met in Week 1 as the connection between SVD and best low-rank approximation), the optimal $M^*$ is the projection onto the top $k$ principal components of the data — i.e., $M^* = U\_k U\_k^\top$.

**Important subtlety:** The individual matrices $W\_e$ and $W\_d$ are not uniquely determined — only their product $W\_d W\_e$ is. The encoder might not directly output the PCA coordinates; it might output a rotated or scaled version. But the subspace spanned by the rows of $W\_e$ (and columns of $W\_d$) is exactly the PCA subspace.

### 4.4 Why This Matters

This theorem is the bridge between two worlds:

- **Classical statistics:** PCA is a venerable method from the early 1900s, derived from eigendecomposition
- **Neural networks:** Autoencoders are neural networks trained to reconstruct their input

They meet at the linear case: a linear autoencoder *is* PCA. This tells us:

1. **Autoencoders generalize PCA.** If we add nonlinear activation functions, the autoencoder can learn a *nonlinear* generalization of PCA — capturing manifold structure that PCA cannot.

2. **The autoencoder objective is well-motivated.** Minimizing reconstruction error has deep connections to variance maximization and information preservation.

3. **The bottleneck forces compression.** When $k < d$, the autoencoder cannot simply learn the identity — it must discover the most important $k$-dimensional subspace. This is the **information bottleneck** principle.

Next week, we will add nonlinear activations and build the full autoencoder. The linear case gives us a solid foundation: we know what the autoencoder *should* learn in the simplest setting, so we can evaluate whether the nonlinear version is doing something sensible.

---

## 5. Limitations of Linear Methods

### 5.1 PCA Can Only Capture Linear Structure

PCA finds the best linear subspace — but real data often has *nonlinear* structure that a linear subspace cannot capture.

**The Swiss Roll example:** Consider data sampled from a 2D surface that is "rolled up" in 3D, like a Swiss roll. The intrinsic dimensionality is 2, but the 2D manifold is curved. PCA would find the 2D plane of maximum variance, which cuts through the roll and mixes up points that are far apart on the manifold but close in 3D.

```
Side view:                    PCA projection:
   ___                        Everything
  /   \                       squashed onto
 /     \   ← rolled up        a plane that
|       |    surface           cuts through
 \     /                       the roll
  \___/                        ← BAD
```

The same issue arises with any dataset where the relevant structure is curved: handwritten digits, natural images, molecular conformations, etc.

### 5.2 t-SNE: Nonlinear Visualization

**t-distributed Stochastic Neighbor Embedding (t-SNE)** (van der Maaten & Hinton, 2008) is a nonlinear method designed for *visualization* — mapping high-dimensional data to 2D or 3D while preserving neighborhood structure.

**Key ideas** (intuitive, not full derivation):

1. In the high-dimensional space, define a probability distribution over pairs of points: nearby points get high probability, distant points get low probability (using a Gaussian kernel).

2. In the low-dimensional embedding, define a similar distribution using a Student t-distribution (hence "t-SNE") — the heavy tails of the t-distribution prevent distant points from being forced too close together.

3. Minimize the KL divergence between the two distributions. This forces the low-dimensional embedding to preserve the neighborhood structure of the high-dimensional data.

**Strengths:** t-SNE produces visually striking 2D plots where clusters are clearly separated. It is excellent at revealing the *existence* of clusters.

**Limitations:**
- The axes of a t-SNE plot have no interpretable meaning
- Distances between clusters are not meaningful (only within-cluster structure is preserved)
- Different random initializations give different results
- The perplexity hyperparameter significantly affects the result
- It is a visualization tool, not a feature extraction method — you cannot use t-SNE embeddings as inputs to a downstream model

### 5.3 UMAP: A Modern Alternative

**Uniform Manifold Approximation and Projection (UMAP)** (McInnes et al., 2018) is a newer nonlinear embedding method based on topological data analysis. In practice, it produces results similar to t-SNE but is much faster and arguably better at preserving global structure.

Like t-SNE, UMAP is primarily a visualization tool. We will use both t-SNE and UMAP in the homework to visualize MNIST digits and the representations learned by neural networks.

### 5.4 The Gap That Autoencoders Fill

Here is the landscape:

| Method | Linear? | Learns features? | Invertible? | Scalable? |
|--------|---------|-------------------|-------------|-----------|
| PCA | Yes | Yes (linear) | Yes | Yes |
| t-SNE | No | Yes | No | Moderate |
| UMAP | No | Yes | No | Yes |
| Autoencoder | No | Yes | Yes (decoder) | Yes |

PCA is linear and invertible but cannot capture nonlinear structure. t-SNE/UMAP capture nonlinear structure but are not invertible (you cannot map from the 2D embedding back to the original space). Autoencoders are nonlinear, invertible (the decoder gives you the reconstruction), and scalable.

Autoencoders learn both an *encoder* (compression) and a *decoder* (reconstruction). This makes them useful not just for visualization but for feature extraction, denoising, generation, and more.

---

## 6. Learned Representations in Neural Networks

### 6.1 Hidden Layers as Feature Extractors

When we train a neural network for classification, the hidden layers are not just "intermediate computations" — they are learning *representations* of the input that make the classification task progressively easier.

Consider a network trained to classify MNIST digits:
- **Input layer:** Raw pixels (784 dimensions). Digit identity is deeply entangled with style, position, and noise.
- **First hidden layer:** Learns to detect simple local patterns — edges, strokes, corners. These are features shared across digit classes.
- **Middle hidden layers:** Combine local patterns into larger structures — loops, line segments, crossings. These are more specific to particular digits.
- **Final hidden layer:** A high-level representation where each digit class forms a distinct cluster. A simple linear classifier can separate them.

The network's internal computation is a progressive disentangling: each layer transforms the data so that the relevant structure (digit identity) becomes more explicit and the irrelevant variation (style, noise) is factored out.

### 6.2 Visualizing What Layers Learn

There is a beautiful hierarchical structure to what different layers of a neural network learn:

**In image classification networks (e.g., trained on ImageNet):**
- **Layer 1:** Gabor filters and color blobs (edge detectors at various orientations and spatial frequencies)
- **Layer 2:** Textures and repeated patterns (composed from Layer 1 edges)
- **Layer 3:** Parts of objects (eyes, wheels, leaves — composed from Layer 2 textures)
- **Layer 4-5:** Whole objects and scenes

This hierarchy was first clearly documented by Zeiler & Fergus (2014) and is one of the most important empirical findings in deep learning. It shows that neural networks independently rediscover many of the hand-crafted features that computer vision researchers spent decades designing.

### 6.3 Transfer Learning

If the intermediate layers learn general features (edges, textures, parts), then these features should be useful for tasks beyond the one the network was trained for. This is the idea behind **transfer learning**:

1. Train a network on a large dataset (e.g., ImageNet with 1.4M images)
2. Freeze the early layers (which have learned general features)
3. Replace the final layer(s) and fine-tune on a new, smaller dataset

This works remarkably well in practice. A network pre-trained on ImageNet can be fine-tuned for medical imaging, satellite imagery, or art classification with just a few hundred examples of the new task. The pre-trained features provide a strong starting point.

**Why transfer learning works:** The early layers learn features that are universal — edges and textures appear in all natural images. Only the later layers need to be task-specific. By reusing the early layers, we avoid having to learn general features from scratch on the small target dataset.

### 6.4 Can We Learn Representations Without Labels?

Transfer learning relies on having a large labeled dataset (like ImageNet) to learn the initial features. But labeling data is expensive. Can we learn useful representations from *unlabeled* data?

This is the central question of **unsupervised representation learning**, and the answer is yes — using autoencoders.

The key insight: if a network can compress data into a small representation and then reconstruct it accurately, the representation must capture the essential structure of the data. No labels needed — the data is its own supervision (this is why autoencoders are sometimes called "self-supervised").

This is where we are headed:
- **Week 6:** Build autoencoders — learn representations by reconstruction
- **Weeks 7-8:** Improve the representations with regularization and probabilistic structure
- **Weeks 9-10:** Enforce sparsity to get interpretable representations
- **Weeks 11-12:** Apply sparse autoencoders to understand what language models have learned

The chain from PCA to sparse autoencoders for interpretability passes through every concept we have studied: linear algebra, optimization, information theory, neural networks, and representation learning. It all connects.

---

## 7. Representations as Geometry

### 7.1 The Geometric View

A representation maps each data point to a point in a new space. The *geometry* of this mapping — which points are close, which are far, how the space is organized — determines how useful the representation is.

**Example:** Consider the XOR problem. Two input features, $x\_1$ and $x\_2$, each 0 or 1. The output is $x\_1 \oplus x\_2$. In the original space, the positive examples (0,1) and (1,0) are not linearly separable from the negative examples (0,0) and (1,1).

A hidden layer can learn a representation where they *are* linearly separable. For instance, if the hidden layer computes $h\_1 = \text{ReLU}(x\_1 + x\_2 - 0.5)$ and $h\_2 = \text{ReLU}(-x\_1 - x\_2 + 1.5)$, the four points become:

| $(x\_1, x\_2)$ | Label | $(h\_1, h\_2)$ |
|---------------|-------|---------------|
| (0, 0) | 0 | (0, 1.5) |
| (0, 1) | 1 | (0.5, 0.5) |
| (1, 0) | 1 | (0.5, 0.5) |
| (1, 1) | 0 | (1.5, 0) |

In the $(h\_1, h\_2)$ space, the positive and negative examples are linearly separable. The network has found a representation that makes the classification trivial.

### 7.2 Good Representations Disentangle Factors of Variation

An ideal representation separates the independent factors that generate the data. For face images:
- One dimension might encode identity (who the person is)
- Another might encode pose (which direction they are facing)
- Another might encode lighting

If these factors are disentangled, we can vary one while holding the others fixed — for example, rotating a face while preserving identity and lighting. Disentangled representations are more interpretable, more robust, and more useful for downstream tasks.

This is precisely what we want from autoencoders: a latent space where each dimension captures a meaningful factor of variation. As we will see in later weeks, achieving this requires careful regularization — vanilla autoencoders do not automatically produce disentangled representations.

### 7.3 Measuring Representation Quality

How do we know if a learned representation is good? Several approaches:

**Reconstruction quality:** How well can the decoder reconstruct the original input from the representation? (Necessary but not sufficient — a good reconstruction does not guarantee a useful representation.)

**Linear probe accuracy:** Train a simple linear classifier on top of the frozen representation. If it achieves high accuracy, the representation has made the classification task linearly separable — a sign of good features.

**Transfer performance:** Use the representation as input to a model on a new task. Good representations transfer well.

**Visualization:** Use t-SNE or UMAP to visualize the representation in 2D. If same-class points cluster together, the representation captures class structure.

We will use all of these metrics throughout the autoencoder weeks.

---

## 8. Summary and Looking Ahead

This week established the conceptual framework for everything that follows:

1. **Features** are transformed representations that make data structure explicit
2. **The manifold hypothesis** tells us that high-dimensional data has low-dimensional structure, making compression possible
3. **PCA** finds the linear subspace of maximum variance (or minimum reconstruction error)
4. **A linear autoencoder learns PCA** — the bridge between classical statistics and neural networks
5. **Nonlinear methods** (t-SNE, UMAP) reveal structure that PCA misses
6. **Neural networks learn hierarchical representations** that are often better than hand-crafted features

The key theorem — that linear autoencoders learn PCA — is our bridge to next week. When we add nonlinear activations, we get a nonlinear generalization of PCA that can capture the curved manifold structure that PCA cannot. That is the autoencoder, and it is where the main arc of this course begins.

---

## References

- Pearson, "On Lines and Planes of Closest Fit" (1901) — the original PCA paper
- Hotelling, "Analysis of a Complex of Statistical Variables into Principal Components" (1933)
- van der Maaten & Hinton, "Visualizing Data using t-SNE" (2008)
- McInnes, Healy, Melville, "UMAP: Uniform Manifold Approximation and Projection" (2018)
- Bengio, Courville, Vincent, "Representation Learning: A Review and New Perspectives" (2013)
- Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks" (2014)
- Baldi & Hornik, "Neural Networks and Principal Component Analysis" (1989) — the linear AE = PCA result
- Goodfellow, Bengio, Courville, *Deep Learning*, Chapter 14: Autoencoders (2016)
