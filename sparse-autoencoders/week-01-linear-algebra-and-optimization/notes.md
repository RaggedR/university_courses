# Week 1: Linear Algebra and Optimization

> *"The introduction of numbers as coordinates is an act of violence."*
> — Hermann Weyl

This week we rebuild the mathematical machinery that everything else in this course rests on. If you've been away from linear algebra for a while, resist the temptation to skim — the concepts here aren't just prerequisites, they're the *language* in which autoencoders, feature extraction, and interpretability are expressed. When we decompose a neural network's activations into interpretable features in Week 11, we'll be doing linear algebra. When we train a sparse autoencoder in Week 10, we'll be doing optimization. This week, we sharpen our tools.

---

## 1. Vector Spaces

### 1.1 Vectors: Two Ways to Think About Them

A vector is simultaneously two things:

1. **A geometric object** — an arrow in space with magnitude and direction
2. **An algebraic object** — an ordered list of numbers $(x\_1, x\_2, \ldots, x\_n)$

Both views are useful. The geometric view gives intuition; the algebraic view gives computation. The power of linear algebra is that you can switch between them freely.

In this course, vectors will most often represent **neural network activations**. When we pass an input through a neural network, each hidden layer produces a vector of activations — a point in a high-dimensional space. A layer with 512 neurons produces a vector in $\mathbb{R}^{512}$. You can't visualize 512 dimensions, but the algebra doesn't care.

We write vectors as bold lowercase: $\mathbf{x} \in \mathbb{R}^n$. By convention, vectors are column vectors:

$$
\mathbf{x} = \begin{pmatrix} x_1 \\\\ x_2 \\\\ \vdots \\\\ x_n \end{pmatrix}
$$

### 1.2 Vector Spaces: The Abstract View

A **vector space** $V$ over $\mathbb{R}$ is a set equipped with two operations — addition and scalar multiplication — satisfying the usual axioms (commutativity, associativity, distributivity, existence of zero vector, etc.). The standard example is $\mathbb{R}^n$, but vector spaces are more general than that. The space of all continuous functions on $[0,1]$ is a vector space. The space of all $m \times n$ matrices is a vector space.

Why bother with the abstraction? Because neural network representations live in spaces that are *isomorphic* to $\mathbb{R}^n$ but aren't literally tuples of numbers. Understanding the abstract structure lets you reason about representations without getting tangled in coordinate details.

### 1.3 Subspaces, Span, and Linear Independence

A **subspace** of $V$ is a subset that is itself a vector space (closed under addition and scalar multiplication). Subspaces always contain the zero vector — this is a quick sanity check.

Given vectors $\mathbf{v}\_1, \ldots, \mathbf{v}\_k$, their **span** is the set of all linear combinations:

$$
\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \left\lbrace \sum_{i=1}^k \alpha_i \mathbf{v}_i \;\middle|\; \alpha_i \in \mathbb{R} \right\rbrace 
$$

Vectors $\mathbf{v}\_1, \ldots, \mathbf{v}\_k$ are **linearly independent** if the only solution to $\sum\_{i=1}^k \alpha\_i \mathbf{v}\_i = \mathbf{0}$ is $\alpha\_1 = \cdots = \alpha\_k = 0$. Intuitively: no vector in the set is redundant.

### 1.4 Basis and Dimension

A **basis** for $V$ is a linearly independent set that spans $V$. Every finite-dimensional vector space has a basis, and all bases have the same number of vectors — this number is the **dimension** of $V$.

The **standard basis** for $\mathbb{R}^n$ is $\lbrace \mathbf{e}\_1, \ldots, \mathbf{e}\_n\rbrace$, where $\mathbf{e}\_i$ has a 1 in position $i$ and 0s elsewhere. But there are infinitely many other bases, and choosing a good one can make your problem much easier. This idea — choosing the right basis — is the heart of dimensionality reduction and feature extraction.

**Concrete example.** In $\mathbb{R}^2$, the vectors $\mathbf{v}\_1 = (1, 1)^T$ and $\mathbf{v}\_2 = (1, -1)^T$ form a basis. Any vector $\mathbf{x} = (a, b)^T$ can be written as:

$$
\mathbf{x} = \frac{a+b}{2}\mathbf{v}_1 + \frac{a-b}{2}\mathbf{v}_2
$$

These coefficients $\left(\frac{a+b}{2}, \frac{a-b}{2}\right)$ are the **coordinates of $\mathbf{x}$ in the $\lbrace \mathbf{v}\_1, \mathbf{v}\_2\rbrace$ basis**. Different basis, different coordinates, same vector.

### 1.5 Inner Products, Norms, and Orthogonality

The **inner product** (dot product) of $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ is:

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T \mathbf{y} = \sum_{i=1}^n x_i y_i
$$

This gives us three critical tools:

- **Norm (length):** $\Vert \mathbf{x}\Vert = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle} = \sqrt{\sum\_i x\_i^2}$ — the Euclidean or $L\_2$ norm.
- **Angle:** $\cos\theta = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\Vert \mathbf{x}\Vert \Vert \mathbf{y}\Vert }$ — this measures similarity between vectors.
- **Orthogonality:** $\mathbf{x} \perp \mathbf{y}$ iff $\langle \mathbf{x}, \mathbf{y} \rangle = 0$.

An **orthonormal basis** is a basis where every vector has unit norm and every pair is orthogonal. Orthonormal bases are computationally convenient because finding coordinates reduces to taking dot products: if $\lbrace \mathbf{u}\_1, \ldots, \mathbf{u}\_n\rbrace$ is orthonormal, then the coordinate of $\mathbf{x}$ along $\mathbf{u}\_i$ is simply $\langle \mathbf{x}, \mathbf{u}\_i \rangle$.

**The Cauchy-Schwarz inequality** deserves a mention: $|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \Vert \mathbf{x}\Vert \cdot \Vert \mathbf{y}\Vert$, with equality iff $\mathbf{x}$ and $\mathbf{y}$ are parallel. This is what makes the "angle" formula above well-defined (the ratio is always in $[-1, 1]$).

**Why this matters for us:** When we talk about "features as directions" in neural activation space (Week 11), we're saying that each interpretable feature corresponds to a direction $\mathbf{u}$ in activation space, and the "amount" of that feature in an activation $\mathbf{x}$ is the inner product $\langle \mathbf{x}, \mathbf{u} \rangle$. Cosine similarity $\frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\Vert \mathbf{x}\Vert \Vert \mathbf{y}\Vert }$ is the standard way to measure whether two activation vectors represent "similar" inputs.

### 1.6 High-Dimensional Geometry: Surprises

Our geometric intuition is built in 2D and 3D, and it fails spectacularly in high dimensions. Since neural network activations live in spaces with hundreds or thousands of dimensions, we need to update our intuition.

**Surprise 1: Almost everything is orthogonal.** If you pick two random unit vectors in $\mathbb{R}^n$, their inner product concentrates around zero as $n$ grows. Specifically, $\mathbb{E}[\langle \mathbf{u}, \mathbf{v} \rangle^2] = 1/n$. In $\mathbb{R}^{1000}$, two random vectors are almost always nearly orthogonal. This means high-dimensional spaces can accommodate far more "almost orthogonal" directions than they have dimensions — a fact that directly underlies the **superposition hypothesis** in mechanistic interpretability (Week 11).

**Surprise 2: Volume concentrates near the surface.** The fraction of the volume of a unit ball in $\mathbb{R}^n$ that lies within distance $\epsilon$ of the surface goes to 1 as $n$ increases. In 1000 dimensions, virtually all the volume is in a thin shell near the boundary. This means that random samples from a uniform distribution on a high-dimensional ball are all approximately the same distance from the center.

**Surprise 3: The curse of dimensionality.** To uniformly sample a $d$-dimensional cube at a resolution of $k$ points per axis, you need $k^d$ points. At just $k = 10$ and $d = 100$, that's $10^{100}$ points — more than the atoms in the observable universe. This is why you cannot learn high-dimensional functions by brute-force sampling, and why learning algorithms must exploit *structure* (like sparsity or manifold structure) to work in practice.

**Concrete example.** Consider $n = 2$ versus $n = 100$. In $\mathbb{R}^2$, you can fit at most 4 mutually "nearly orthogonal" unit vectors (the standard basis and their negatives). In $\mathbb{R}^{100}$, you can fit 100 exactly orthogonal directions — but you can also fit roughly $e^{cn}$ directions that are all *nearly* orthogonal (pairwise angles close to $90°$). The number of nearly orthogonal directions grows exponentially faster than the dimension. Neural networks exploit this: a 100-dimensional activation space can represent far more than 100 features, as long as the features aren't all active simultaneously.

### 1.7 Projection and Orthogonal Decomposition

The **orthogonal projection** of $\mathbf{x}$ onto a subspace $W$ is the vector $\hat{\mathbf{x}} \in W$ that is closest to $\mathbf{x}$:

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{w} \in W} \Vert \mathbf{x} - \mathbf{w}\Vert 
$$

If $\lbrace \mathbf{u}\_1, \ldots, \mathbf{u}\_k\rbrace$ is an orthonormal basis for $W$, the projection is:

$$
\hat{\mathbf{x}} = \sum_{i=1}^k \langle \mathbf{x}, \mathbf{u}_i \rangle \mathbf{u}_i
$$

The residual $\mathbf{x} - \hat{\mathbf{x}}$ is orthogonal to $W$. This gives the **orthogonal decomposition**: any vector can be written as its projection onto $W$ plus an orthogonal residual.

This is exactly what low-rank approximation (SVD) does: it finds the best $k$-dimensional subspace to project your data onto. And it's what a linear autoencoder with a bottleneck of dimension $k$ learns to do. The encoder computes the projection; the decoder reconstructs from the projected coordinates.

**Concrete example.** Project $\mathbf{x} = (3, 4, 1)^T$ onto the subspace spanned by $\mathbf{u}\_1 = \frac{1}{\sqrt{2}}(1, 1, 0)^T$:

$$
\hat{\mathbf{x}} = \langle \mathbf{x}, \mathbf{u}_1 \rangle \mathbf{u}_1 = \frac{3 + 4}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix}1\\\\1\\\\0\end{pmatrix} = \frac{7}{2}\begin{pmatrix}1\\\\1\\\\0\end{pmatrix} = \begin{pmatrix}3.5\\\\3.5\\\\0\end{pmatrix}
$$

The residual is $(3, 4, 1)^T - (3.5, 3.5, 0)^T = (-0.5, 0.5, 1)^T$. You can verify: $\langle (-0.5, 0.5, 1), (1, 1, 0)/\sqrt{2} \rangle = 0$.

---

## 2. Matrices as Linear Maps

### 2.1 The Dual Life of a Matrix

A matrix $A \in \mathbb{R}^{m \times n}$ is simultaneously:

1. **A table of numbers** — $m$ rows, $n$ columns
2. **A linear map** $T: \mathbb{R}^n \to \mathbb{R}^m$ — it transforms vectors

The matrix-vector product $A\mathbf{x}$ is a linear combination of the columns of $A$:

$$
A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n
$$

where $\mathbf{a}\_j$ is the $j$-th column of $A$. This is the **column picture** of matrix multiplication, and it's the right way to think about what matrices do.

**Concrete example.** The matrix $A = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$ stretches space by a factor of 2 in the $x$-direction and leaves the $y$-direction alone. The matrix $R = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ rotates space 90 degrees counterclockwise. The product $RA$ first stretches, then rotates. Note the order: $RA$ means "apply $A$ first, then $R$." Matrix multiplication composes transformations right-to-left.

### 2.2 Fundamental Subspaces

Every matrix $A \in \mathbb{R}^{m \times n}$ has four fundamental subspaces:

| Subspace | Definition | Lives in |
|----------|-----------|----------|
| **Column space** $\mathcal{C}(A)$ | $\lbrace A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\rbrace$ — the range/image | $\mathbb{R}^m$ |
| **Null space** $\mathcal{N}(A)$ | $\lbrace \mathbf{x} : A\mathbf{x} = \mathbf{0}\rbrace$ — the kernel | $\mathbb{R}^n$ |
| **Row space** $\mathcal{C}(A^T)$ | The column space of $A^T$ | $\mathbb{R}^n$ |
| **Left null space** $\mathcal{N}(A^T)$ | The null space of $A^T$ | $\mathbb{R}^m$ |

The **rank** of $A$ is the dimension of its column space (equivalently, row space). The rank-nullity theorem ties these together: $\text{rank}(A) + \dim(\mathcal{N}(A)) = n$. The rank tells you the "true dimensionality" of the transformation. A rank-deficient matrix maps some non-zero vectors to zero — it loses information.

**Connection to neural networks:** A weight matrix $W \in \mathbb{R}^{d\_{\text{out}} \times d\_{\text{in}}}$ maps activations from one layer to the next. If $\text{rank}(W) < \min(d\_{\text{out}}, d\_{\text{in}})$, the layer is not using its full capacity. This connects to the bottleneck idea in autoencoders (Week 6).

### 2.3 Matrix Multiplication as Data Transformation

Here's a perspective that connects matrices to data. Suppose you have a dataset of $N$ samples, each with $d$ features, stored as an $N \times d$ matrix $X$ (one row per sample). Multiplying $X$ by a matrix $W^T \in \mathbb{R}^{d \times k}$ gives you an $N \times k$ matrix $XW^T$ — the same $N$ samples, but now each described by $k$ features instead of $d$.

If $k < d$, this is **dimensionality reduction**: the matrix $W^T$ projects each sample from $\mathbb{R}^d$ down to $\mathbb{R}^k$. If $k > d$, this is **feature expansion**: the matrix maps each sample to a higher-dimensional space where linear separation may be easier.

Both of these operations appear in autoencoders. The encoder reduces dimensionality; the decoder expands it back. The autoencoder learns the matrices $W$ that make this round-trip as lossless as possible.

### 2.4 Change of Basis

If $P$ is the matrix whose columns are a new basis $\lbrace \mathbf{v}\_1, \ldots, \mathbf{v}\_n\rbrace$, then the coordinates of $\mathbf{x}$ in the new basis are $P^{-1}\mathbf{x}$. A linear map $A$ in the original basis becomes $P^{-1}AP$ in the new basis.

This is what eigendecomposition and SVD are really about: finding a basis in which the matrix takes a particularly simple form.

---

## 3. Eigendecomposition

### 3.1 Eigenvalues and Eigenvectors

An **eigenvector** of a square matrix $A \in \mathbb{R}^{n \times n}$ is a non-zero vector $\mathbf{v}$ such that:

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

for some scalar $\lambda$ (the **eigenvalue**). Geometrically: $A$ acts on $\mathbf{v}$ by merely scaling it. The eigenvectors are the "natural axes" of the transformation — the directions that the matrix doesn't rotate, only stretches or flips.

To find eigenvalues, solve $\det(A - \lambda I) = 0$ (the **characteristic equation**). This is a polynomial of degree $n$ in $\lambda$, so there are at most $n$ eigenvalues (counted with multiplicity).

**Concrete example.** Consider $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$.

The characteristic equation: $(3 - \lambda)(2 - \lambda) = 0$, giving $\lambda\_1 = 3$, $\lambda\_2 = 2$.

For $\lambda\_1 = 3$: solve $(A - 3I)\mathbf{v} = \mathbf{0}$, giving $\begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\mathbf{v} = \mathbf{0}$, so $\mathbf{v}\_1 = (1, 0)^T$.

For $\lambda\_2 = 2$: solve $(A - 2I)\mathbf{v} = \mathbf{0}$, giving $\begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\mathbf{v} = \mathbf{0}$, so $\mathbf{v}\_2 = (-1, 1)^T$.

### 3.2 Diagonalization

If $A$ has $n$ linearly independent eigenvectors, we can write:

$$
A = PDP^{-1}
$$

where $P = [\mathbf{v}\_1 | \cdots | \mathbf{v}\_n]$ is the matrix of eigenvectors and $D = \text{diag}(\lambda\_1, \ldots, \lambda\_n)$ is the diagonal matrix of eigenvalues. This is **eigendecomposition** or **diagonalization**.

Why is this useful? Because in the eigenbasis, $A$ is just scaling along each axis. Powers become trivial: $A^k = PD^kP^{-1}$. The matrix exponential becomes easy. And it reveals the geometric structure of the transformation.

Not every matrix is diagonalizable (you need $n$ linearly independent eigenvectors). But the matrices we care about most — symmetric matrices — always are.

### 3.3 Positive Definiteness

A symmetric matrix $A$ is:

- **Positive definite** if $\mathbf{x}^TA\mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$ (equivalently, all eigenvalues are positive)
- **Positive semi-definite** if $\mathbf{x}^TA\mathbf{x} \geq 0$ for all $\mathbf{x}$ (all eigenvalues non-negative)

The expression $\mathbf{x}^TA\mathbf{x}$ is a **quadratic form**. For a $2 \times 2$ matrix $\begin{pmatrix} a & b \\ b & c \end{pmatrix}$, this is $ax\_1^2 + 2bx\_1x\_2 + cx\_2^2$.

**Why this matters:** The Hessian of a convex function is positive semi-definite. Covariance matrices are always positive semi-definite. These constraints tell you a lot about the geometric and statistical structure of your problem.

### 3.4 The Spectral Theorem

**Theorem (Spectral Theorem for Real Symmetric Matrices).** If $A \in \mathbb{R}^{n \times n}$ is symmetric ($A = A^T$), then:

1. All eigenvalues of $A$ are **real**.
2. Eigenvectors corresponding to distinct eigenvalues are **orthogonal**.
3. $A$ can be written as $A = Q\Lambda Q^T$ where $Q$ is orthogonal ($Q^TQ = I$) and $\Lambda$ is diagonal.

This is one of the most important theorems in applied mathematics. The decomposition $A = Q\Lambda Q^T$ can also be written as:

$$
A = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T
$$

This is the **spectral decomposition**: $A$ is a sum of rank-1 matrices, each scaled by an eigenvalue. This "take a complicated thing and decompose it into simple pieces" pattern is exactly what feature extraction does.

**Why this matters:** Covariance matrices are symmetric and positive semi-definite ($\lambda\_i \geq 0$). PCA (Week 5) is eigendecomposition of the covariance matrix. The eigenvectors with the largest eigenvalues are the principal components — the directions of greatest variance in your data.

**Concrete example.** Consider the covariance matrix $\Sigma = \begin{pmatrix} 5 & 3 \\ 3 & 5 \end{pmatrix}$.

The eigenvalues are $\lambda\_1 = 8$ and $\lambda\_2 = 2$, with eigenvectors $\mathbf{q}\_1 = \frac{1}{\sqrt{2}}(1, 1)^T$ and $\mathbf{q}\_2 = \frac{1}{\sqrt{2}}(1, -1)^T$. The spectral decomposition tells us that the data varies most along the direction $(1, 1)$ (eigenvalue 8) and least along $(1, -1)$ (eigenvalue 2). If you had to keep only one direction, you'd keep $(1, 1)$ — and that's exactly what PCA does.

---

## 4. Singular Value Decomposition

### 4.1 SVD: The Statement

The SVD generalizes eigendecomposition to *any* matrix, not just square ones. For any $A \in \mathbb{R}^{m \times n}$:

$$
A = U\Sigma V^T
$$

where:
- $U \in \mathbb{R}^{m \times m}$ is orthogonal (columns are **left singular vectors**)
- $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with non-negative entries $\sigma\_1 \geq \sigma\_2 \geq \cdots \geq 0$ (the **singular values**)
- $V \in \mathbb{R}^{n \times n}$ is orthogonal (columns are **right singular vectors**)

The relationship to eigendecomposition: the singular values are the square roots of the eigenvalues of $A^TA$ (or $AA^T$). The right singular vectors are the eigenvectors of $A^TA$; the left singular vectors are the eigenvectors of $AA^T$.

### 4.2 Geometric Interpretation

The SVD says that *every* linear transformation can be decomposed into three steps:

1. **Rotate** (apply $V^T$) — align with the "input axes" of the transformation
2. **Scale** (apply $\Sigma$) — stretch or shrink along each axis
3. **Rotate** (apply $U$) — align with the "output axes"

This is the deepest statement about what matrices *do*. Any linear map, no matter how complicated, is just rotation-scaling-rotation.

**Concrete example.** Consider $A = \begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix}$.

Computing: $A^TA = \begin{pmatrix} 25 & -15 \\ -15 & 25 \end{pmatrix}$

Eigenvalues of $A^TA$: $\lambda\_1 = 40$, $\lambda\_2 = 10$, so $\sigma\_1 = \sqrt{40} = 2\sqrt{10}$, $\sigma\_2 = \sqrt{10}$.

The singular values tell you by how much the matrix stretches space in each principal direction. The ratio $\sigma\_1/\sigma\_n$ (the **condition number**) tells you how "distorted" the transformation is. A condition number of $2\sqrt{10}/\sqrt{10} = 2$ is quite mild; a condition number of $10^6$ means the matrix nearly collapses one direction to zero.

### 4.3 Low-Rank Approximation

Writing the SVD in summation form:

$$
A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

where $r = \text{rank}(A)$. Each term $\sigma\_i \mathbf{u}\_i \mathbf{v}\_i^T$ is a rank-1 matrix. The terms are ordered by importance (decreasing $\sigma\_i$).

**Theorem (Eckart-Young).** The best rank-$k$ approximation to $A$ (in Frobenius or spectral norm) is:

$$
A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

The approximation error is $\Vert A - A\_k\Vert \_F = \sqrt{\sigma\_{k+1}^2 + \cdots + \sigma\_r^2}$.

This theorem is why SVD is the backbone of data compression. If the singular values drop off quickly, you can approximate a large matrix with a small one and lose very little information.

**Concrete example: image compression.** A grayscale image is an $m \times n$ matrix. Its SVD has $\min(m,n)$ singular values. Keeping only the top $k$ singular values gives a rank-$k$ approximation that requires storing only $k(m + n + 1)$ numbers instead of $mn$. For a $1000 \times 1000$ image, rank-50 uses about $100{,}000$ numbers instead of $1{,}000{,}000$ — a 10x compression with usually excellent visual quality.

**What makes real images compressible?** Photographs have enormous redundancy: neighboring pixels are correlated, textures repeat, large regions have similar colors. These correlations cause the singular values to decay rapidly — the first few capture the dominant structure, and the rest are small corrections. Random noise, by contrast, has no structure, so its singular values don't decay — you can't compress noise. The gap between "structured" and "random" singular value spectra is a fundamental theme that runs through this entire course.

**Connection to autoencoders:** A linear autoencoder (no activation functions) with a $k$-dimensional bottleneck learns *exactly* the rank-$k$ SVD approximation. Nonlinear autoencoders generalize this to curved surfaces (manifolds) instead of flat subspaces. This is the key insight of Week 6.

### 4.4 The Frobenius Norm and Energy

The **Frobenius norm** of a matrix is:

$$
\Vert A\Vert _F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\sum_{i=1}^r \sigma_i^2}
$$

The second equality (sum of squared singular values) gives us an "energy" interpretation. The rank-$k$ approximation captures a fraction $\frac{\sum\_{i=1}^k \sigma\_i^2}{\sum\_{i=1}^r \sigma\_i^2}$ of the total energy. In image compression, we often target 95% or 99% energy retention and choose $k$ accordingly.

### 4.5 SVD and the Four Fundamental Subspaces

The SVD neatly reveals all four fundamental subspaces:

- The first $r$ columns of $U$ span the column space of $A$
- The last $m - r$ columns of $U$ span the left null space
- The first $r$ columns of $V$ span the row space
- The last $n - r$ columns of $V$ span the null space

This is the most complete "X-ray" of a matrix you can get.

---

## 5. Optimization

### 5.1 Functions of Several Variables

A function $f: \mathbb{R}^n \to \mathbb{R}$ maps a vector to a scalar. In machine learning, $f$ is typically a **loss function** that measures how far our model's predictions are from the truth. The goal is to find the input $\mathbf{x}^*$ that minimizes $f$.

The **gradient** of $f$ at $\mathbf{x}$ is the vector of partial derivatives:

$$
\nabla f(\mathbf{x}) = \begin{pmatrix} \partial f / \partial x_1 \\\\ \partial f / \partial x_2 \\\\ \vdots \\\\ \partial f / \partial x_n \end{pmatrix}
$$

The gradient points in the direction of steepest ascent. Its magnitude tells you how steep the slope is. At a local minimum, $\nabla f = \mathbf{0}$.

**Key property:** The gradient is orthogonal to the level sets of $f$. If you stand on a contour line of a topographic map, the gradient points directly uphill, perpendicular to the contour. This is why gradient descent moves perpendicular to level sets.

The **Hessian** is the matrix of second partial derivatives:

$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

The Hessian is symmetric (assuming continuous second derivatives, by Clairaut's theorem) and encodes curvature. If $H$ is positive definite (all eigenvalues positive) at a critical point, that point is a local minimum. If $H$ has both positive and negative eigenvalues, it's a saddle point. The eigenvalues of $H$ tell you the curvature in each principal direction.

### 5.2 Convexity

A function $f$ is **convex** if for all $\mathbf{x}, \mathbf{y}$ and $t \in [0,1]$:

$$
f(t\mathbf{x} + (1-t)\mathbf{y}) \leq tf(\mathbf{x}) + (1-t)f(\mathbf{y})
$$

Geometrically: the line segment between any two points on the graph lies above the graph. The function is "bowl-shaped."

Equivalently (for twice-differentiable functions): $f$ is convex iff its Hessian is positive semi-definite everywhere. This connects eigendecomposition to optimization: checking convexity is an eigenvalue problem.

**Why convexity matters (enormously):** For convex functions, every local minimum is a global minimum. You can't get stuck in bad local minima. Gradient descent is guaranteed to find the optimum (given enough time and a small enough learning rate).

Linear regression, logistic regression, and SVMs all have convex loss functions. Neural networks do *not* — their loss landscapes are highly non-convex with many local minima and saddle points. Understanding convexity helps you appreciate both why simple models are nice and why training neural networks is fundamentally hard.

**Common convex functions:**
- $f(x) = x^2$ (and more generally $f(\mathbf{x}) = \mathbf{x}^TA\mathbf{x}$ for positive semi-definite $A$)
- $f(x) = |x|$
- $f(x) = e^x$
- $f(x) = -\log(x)$ for $x > 0$
- Norms: $f(\mathbf{x}) = \Vert \mathbf{x}\Vert \_p$ for any $p \geq 1$

**Useful composition rules:** The sum of convex functions is convex. The maximum of convex functions is convex. The composition $g(f(\mathbf{x}))$ is convex if $g$ is convex and non-decreasing and $f$ is convex.

### 5.3 Gradient Descent

The simplest optimization algorithm: start somewhere, repeatedly step in the direction of steepest descent.

**Algorithm (Gradient Descent):**

1. Initialize $\mathbf{x}\_0$
2. For $t = 0, 1, 2, \ldots$:
   - $\mathbf{x}\_{t+1} = \mathbf{x}\_t - \eta \nabla f(\mathbf{x}\_t)$

Here $\eta > 0$ is the **learning rate**. Too small: convergence is painfully slow. Too large: you overshoot and diverge. The art of optimization is largely about managing this trade-off.

**Convergence for convex functions.** For a convex function with Lipschitz-continuous gradients (i.e., $\Vert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\Vert \leq L\Vert \mathbf{x} - \mathbf{y}\Vert$), gradient descent with learning rate $\eta = 1/L$ converges at rate $O(1/t)$: after $t$ steps, $f(\mathbf{x}\_t) - f^* \leq O(1/t)$. For **strongly convex** functions (Hessian eigenvalues bounded below by $\mu > 0$), the convergence is linear: $f(\mathbf{x}\_t) - f^* \leq O((1 - \mu/L)^t)$. The ratio $\kappa = L/\mu$ is the **condition number**, and it controls how fast gradient descent converges.

**Concrete example.** Minimize $f(x, y) = x^2 + 4y^2$.

The gradient is $\nabla f = (2x, 8y)^T$. The Hessian is $\begin{pmatrix} 2 & 0 \\ 0 & 8 \end{pmatrix}$ with eigenvalues $2$ and $8$, so $\kappa = 8/2 = 4$.

Starting at $(4, 2)$ with $\eta = 0.1$:

| Step | $(x, y)$ | $f(x, y)$ |
|------|-----------|-----------|
| 0 | $(4.00, 2.00)$ | $32.00$ |
| 1 | $(3.20, 0.40)$ | $10.88$ |
| 2 | $(2.56, 0.08)$ | $6.58$ |
| 3 | $(2.05, 0.02)$ | $4.19$ |

The $y$-direction converges much faster because the curvature is higher there (eigenvalue 8 vs. 2). The trajectory oscillates slightly in $y$ while slowly descending in $x$. This illustrates a general problem: gradient descent struggles when the curvature varies a lot across directions (high condition number).

### 5.4 Stochastic Gradient Descent (SGD)

In machine learning, the loss function is typically a sum over data points:

$$
L(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(\theta; \mathbf{x}_i, y_i)
$$

Computing the full gradient requires processing all $N$ data points — expensive when $N$ is millions. SGD approximates the gradient using a random **mini-batch** of $B$ samples:

$$
\nabla L(\theta) \approx \frac{1}{B} \sum_{i \in \text{batch}} \nabla \ell(\theta; \mathbf{x}_i, y_i)
$$

This estimate is noisy (high variance) but **unbiased** (correct in expectation). Remarkably, this noise often *helps* — it acts as a form of regularization and helps escape shallow local minima. SGD is the workhorse of deep learning.

**Key trade-off:** Larger batch size $\to$ more accurate gradient estimate but more computation per step. Smaller batch size $\to$ noisier gradient but more updates per epoch. In practice, batch sizes of 32-256 work well for most problems.

**An important subtlety:** One full pass through all $N$ training samples is called an **epoch**. With batch size $B$, each epoch contains $N/B$ gradient updates. Training typically runs for tens to hundreds of epochs.

### 5.5 Beyond Vanilla SGD

Vanilla SGD has known failure modes: it oscillates in directions of high curvature and crawls along flat directions. Modern optimizers address this:

- **Momentum:** Accumulate a running average of past gradients, like a ball rolling downhill. This smooths out oscillations and accelerates progress along consistent gradient directions.

$$
\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t), \quad \mathbf{x}_{t+1} = \mathbf{x}_t - \eta \mathbf{v}_{t+1}
$$

Typical $\beta = 0.9$, meaning the velocity is a weighted average of the last ~10 gradients.

- **Adam (Adaptive Moment Estimation):** Maintains per-parameter learning rates that adapt based on the history of gradients. It tracks both the first moment (mean) and second moment (variance) of gradients for each parameter. Parameters with consistently small gradients get larger effective learning rates; parameters with large, noisy gradients get smaller ones.

We'll implement these in Week 4. For now, know that they exist and that Adam is the default choice for most deep learning today.

---

## 6. Norms and Regularization

### 6.1 The $L\_p$ Norm Family

For $p \geq 1$, the $L\_p$ norm of $\mathbf{x} \in \mathbb{R}^n$ is:

$$
\Vert \mathbf{x}\Vert _p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}
$$

The three most important cases:

| Norm | Formula | Unit ball shape (2D) |
|------|---------|---------------------|
| $L\_1$ | $\sum\_i \Vert x\_i\Vert$ | Diamond (rotated square) |
| $L\_2$ | $\sqrt{\sum\_i x\_i^2}$ | Circle |
| $L\_\infty$ | $\max\_i \Vert x\_i\Vert$ | Square |

The $L\_0$ "norm" (in scare quotes because it's not actually a norm — it doesn't satisfy the triangle inequality) counts the number of non-zero entries: $\Vert \mathbf{x}\Vert \_0 = |\lbrace i : x\_i \neq 0\rbrace |$. It directly measures sparsity, but it's combinatorial and non-convex, making it impossible to optimize with gradient-based methods. The $L\_1$ norm is the closest *convex* approximation to $L\_0$ — this is why $L\_1$ is so important for sparsity.

### 6.2 Why $L\_1$ Promotes Sparsity

This is one of the most important geometric arguments in machine learning, and it's central to the entire second half of this course.

Consider the optimization problem: minimize $f(\mathbf{x})$ subject to $\Vert \mathbf{x}\Vert \_p \leq C$. Geometrically, we're looking for the lowest level set of $f$ that touches the $L\_p$ ball.

The level sets of $f$ are typically smooth curves (ellipses for quadratic $f$). Now compare:

- **$L\_2$ ball (circle):** The level sets of $f$ will generically touch the circle at a point with *all* coordinates non-zero. The solution is usually dense.
- **$L\_1$ ball (diamond):** The diamond has *corners* on the coordinate axes. The level sets of $f$ will preferentially touch the diamond at these corners, giving solutions where some coordinates are exactly zero. The solution is often **sparse**.

The corners of the $L\_1$ ball are the key: they're the points $(0, \ldots, 0, \pm C, 0, \ldots, 0)$ with all but one coordinate zero. Smooth level curves are much more likely to first touch a corner than a flat face.

**Why does the corner "attract" the optimum?** Consider $\mathbb{R}^2$. The $L\_1$ ball has 4 corners and 4 edges. In $\mathbb{R}^n$, it has $2n$ corners and lower-dimensional faces. As $n$ increases, the fraction of the surface that consists of corners and low-dimensional faces grows. In high dimensions, the $L\_1$ ball is almost entirely corners — almost any random direction from the origin hits a low-dimensional face. This means the sparsity-promoting effect gets *stronger* in higher dimensions, which is exactly when we need it most.

**Concrete example.** Minimize $f(x, y) = (x - 3)^2 + (y - 2)^2$ subject to $|x| + |y| \leq 2$ (the $L\_1$ constraint). The unconstrained minimum is at $(3, 2)$, which is outside the diamond. The constrained minimum turns out to be at $(2, 0)$ — on a corner, with $y$ pushed to exactly zero. With an $L\_2$ constraint $x^2 + y^2 \leq 4$, the solution would be approximately $(1.66, 1.11)$ — both coordinates non-zero.

### 6.3 Regularization in Machine Learning

In practice, we don't use hard constraints. Instead, we add a penalty term to the loss:

$$
\mathcal{L}_{\text{regularized}}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \lambda \Vert \theta\Vert _p^p
$$

- **$L\_2$ regularization (Ridge/weight decay):** $\lambda \Vert \theta\Vert \_2^2 = \lambda \sum\_i \theta\_i^2$. Pushes weights to be small but not exactly zero. The gradient of $L\_2$ is $2\lambda\theta\_i$, which is proportional to $\theta\_i$ — large weights are penalized more.
- **$L\_1$ regularization (LASSO):** $\lambda \Vert \theta\Vert \_1 = \lambda \sum\_i |\theta\_i|$. Pushes some weights to be exactly zero — **sparsity**. The "gradient" of $L\_1$ is $\lambda \cdot \text{sign}(\theta\_i)$, which applies constant pressure toward zero regardless of magnitude. This constant pressure is what drives small weights all the way to zero.

The hyperparameter $\lambda$ controls the trade-off: larger $\lambda$ means more regularization (simpler model), smaller $\lambda$ means less (more complex model).

### 6.4 The Proximal View (Optional but Illuminating)

The $L\_1$ norm is not differentiable at zero, which complicates gradient descent. The **proximal operator** handles this cleanly. For the $L\_1$ penalty, the proximal operator is the **soft thresholding** function:

$$
\text{prox}_{\lambda \Vert \cdot\Vert _1}(x_i) = \text{sign}(x_i) \max(|x_i| - \lambda, 0)
$$

This shrinks each coordinate toward zero by $\lambda$, and if the coordinate is already within $\lambda$ of zero, it sets it to exactly zero. The **ISTA (Iterative Shrinkage-Thresholding Algorithm)** alternates gradient steps (for the data term) with proximal steps (for the $L\_1$ term). We'll encounter ISTA in Week 9 when we study sparse coding.

### 6.5 Preview: Why Sparsity Matters for Autoencoders

Here's a taste of where all this is headed. In a sparse autoencoder (Week 10), we have a hidden layer with many neurons (possibly more than the input dimension). Without regularization, every input would activate every neuron — the representation would be dense and hard to interpret.

By adding an $L\_1$ penalty (or a KL divergence penalty, which acts similarly) on the hidden layer activations, we encourage each input to activate only a *few* neurons. The result: each neuron becomes a specialist, responding strongly to one specific feature and being silent otherwise. This makes the representation interpretable — you can look at what activates each neuron and assign it a meaning.

The geometry is the same: the $L\_1$ penalty creates corners in the optimization landscape that correspond to sparse activation patterns. The mathematics of this week directly enables the interpretability of Week 11.

---

## Summary

| Concept | Key Idea | Where It Shows Up Later |
|---------|----------|------------------------|
| Vector spaces & bases | Representations are vectors; choosing the right basis reveals structure | Features as directions (Week 11) |
| High-dim geometry | Nearly orthogonal directions grow exponentially with dimension | Superposition hypothesis (Week 11) |
| Projection | Best approximation by a subspace; orthogonal residual | Linear autoencoders, PCA (Weeks 5-6) |
| Eigendecomposition | Find the "natural axes" of a symmetric transformation | PCA (Week 5), covariance analysis |
| SVD | Any matrix = rotate + scale + rotate; best low-rank approximation | Dimensionality reduction, linear autoencoders (Week 6) |
| Gradient descent | Iteratively follow the steepest descent direction | Training all neural networks (Weeks 3-12) |
| Condition number | Ratio of largest to smallest eigenvalue; controls optimization difficulty | Learning rate selection, architecture design |
| $L\_1$ sparsity | Diamond geometry pushes solutions to corners (zeros) | Sparse autoencoders (Weeks 9-10) |

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$ | Column vector |
| $\mathbf{x}^T$ | Row vector (transpose) |
| $\Vert \mathbf{x}\Vert \_p$ | $L\_p$ norm |
| $\langle \mathbf{x}, \mathbf{y} \rangle$ | Inner product |
| $A^T$ | Matrix transpose |
| $A^{-1}$ | Matrix inverse |
| $\text{rank}(A)$ | Rank of matrix $A$ |
| $\mathcal{C}(A)$ | Column space of $A$ |
| $\mathcal{N}(A)$ | Null space of $A$ |
| $\sigma\_i$ | $i$-th singular value |
| $\lambda\_i$ | $i$-th eigenvalue |
| $\nabla f$ | Gradient of $f$ |
| $H$ | Hessian matrix |
| $\eta$ | Learning rate |
| $\kappa$ | Condition number |

---

## Further Reading

- **Strang, G.** *Introduction to Linear Algebra*, 5th ed. Chapters 1-8 cover everything in Sections 1-4 of these notes, with excellent geometric intuition.
- **Boyd, S. and Vandenberghe, L.** *Convex Optimization*. Free online. Chapter 1 and Sections 9.1-9.4 (gradient descent) are most relevant.
- **3Blue1Brown.** *Essence of Linear Algebra* (YouTube). 16 short videos that build visual intuition for vectors, matrices, eigenvalues, and more. Highly recommended if you're rusty.
- **Goodfellow et al.** *Deep Learning*, Chapter 2 (Linear Algebra) and Chapter 4 (Numerical Computation). Available free at deeplearningbook.org.
- **Vershynin, R.** *High-Dimensional Probability*. For the curious: rigorous treatment of the geometric phenomena described in Section 1.6. Chapters 1-5 are the relevant ones.
