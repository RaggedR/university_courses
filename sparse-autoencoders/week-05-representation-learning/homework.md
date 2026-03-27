# Week 5: Representation Learning & Dimensionality Reduction — Homework

## Problem Set Overview

This problem set connects PCA theory to practice, establishes the bridge between PCA and autoencoders, and builds intuition for representation learning through visualization. Problems 1-2 are mathematical. Problems 3-6 are implementation and conceptual exercises.

**Estimated time:** 7-9 hours

**Submission:** A Jupyter notebook containing your solutions, code, plots, and written explanations.

---

## Problem 1: PCA from Maximum Variance (Pen and Paper)

**(a)** State the PCA optimization problem: maximize the variance of the projected data, subject to the constraint that the projection direction is a unit vector. Write it formally using the covariance matrix $C$.

**(b)** Use the method of Lagrange multipliers to show that the optimal projection direction must satisfy $C\mathbf{u} = \lambda\mathbf{u}$ for some scalar $\lambda$. Identify $\lambda$ as the variance captured by this direction.

**(c)** Explain why the first principal component is the eigenvector corresponding to the *largest* eigenvalue, not just any eigenvalue.

**(d)** Show that the second principal component — the direction of maximum variance orthogonal to $\mathbf{u}\_1$ — is the eigenvector corresponding to the second-largest eigenvalue. (Hint: add the constraint $\mathbf{u}\_2^\top \mathbf{u}\_1 = 0$ and use a second Lagrange multiplier.)

**(e)** The total variance of the data is $\text{tr}(C) = \sum\_{j=1}^d \lambda\_j$. Show that the fraction of variance captured by the top $k$ components is $\frac{\sum\_{j=1}^k \lambda\_j}{\sum\_{j=1}^d \lambda\_j}$.

**(f)** Conversely, show that the reconstruction error from projecting onto the top $k$ components is $\sum\_{j=k+1}^d \lambda\_j$. Explain in one sentence why maximizing captured variance and minimizing reconstruction error are equivalent.

---

## Problem 2: Linear Autoencoder = PCA (Guided Proof)

Consider a linear autoencoder with encoder $W\_e \in \mathbb{R}^{k \times d}$ and decoder $W\_d \in \mathbb{R}^{d \times k}$, trained to minimize:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{x}_i - W_d W_e \mathbf{x}_i\|^2
$$

Assume the data is centered ($\bar{\mathbf{x}} = \mathbf{0}$) and has covariance $C$ with eigendecomposition $C = V \Lambda V^\top$.

**(a)** Define $M = W\_d W\_e$. What is the maximum possible rank of $M$? (Answer in terms of $k$ and $d$.)

**(b)** Show that the loss can be written as:

$$
\mathcal{L} = \text{tr}(C) - 2\,\text{tr}(MC) + \text{tr}(MCM^\top)
$$

(Hint: expand $\Vert \mathbf{x} - M\mathbf{x}\Vert ^2 = (\mathbf{x} - M\mathbf{x})^\top(\mathbf{x} - M\mathbf{x})$ and use the fact that $\frac{1}{N}\sum\_i \mathbf{x}\_i\mathbf{x}\_i^\top = C$.)

**(c)** If $M$ is a projection matrix (i.e., $M^2 = M$ and $M = M^\top$), show that the loss simplifies to $\mathcal{L} = \text{tr}(C) - \text{tr}(MC)$.

(Hint: for projection matrices, $M M^\top = M^2 = M$.)

**(d)** Among all rank-$k$ projection matrices, which one minimizes the loss? (Express your answer in terms of the eigenvectors of $C$. Recall that $\text{tr}(MC) = \sum\_j \mathbf{u}\_j^\top C \mathbf{u}\_j$ where $\mathbf{u}\_j$ are the columns of the projection.)

**(e)** Does gradient descent on $W\_e$ and $W\_d$ necessarily converge to a solution where $W\_d W\_e$ is a projection matrix? Or could the product be a non-projection matrix that still minimizes the loss? Discuss whether the individual matrices $W\_e, W\_d$ are uniquely determined at the optimum, even if their product $W\_d W\_e$ is.

**(f)** Suppose you add an L2 penalty $\frac{\mu}{2}(\Vert W\_e\Vert \_F^2 + \Vert W\_d\Vert \_F^2)$ to the loss. How would this affect the solution? Would the product $W\_d W\_e$ still be the PCA projection at the optimum?

---

## Problem 3: PCA from Scratch on MNIST

Implement PCA from scratch (no `sklearn.decomposition.PCA`) and apply it to MNIST.

**(a)** Load MNIST and flatten each 28x28 image to a 784-dimensional vector. Center the data by subtracting the mean image. Visualize the mean image as a 28x28 grayscale plot — what does it look like? Why does it look this way?

**(b)** Compute the covariance matrix $C = \frac{1}{N}X^\top X$ where $X$ is the $N \times 784$ centered data matrix. Compute its eigendecomposition. (You may use `numpy.linalg.eigh` for the eigendecomposition itself — we are not asking you to implement the eigenvalue algorithm.)

Report: What are the top 5 eigenvalues? What is the ratio of the largest to smallest non-zero eigenvalue? How many eigenvalues are essentially zero?

**(c)** Plot the top 50 eigenvalues (scree plot). On a separate subplot, plot the cumulative explained variance ratio vs. number of components (from 1 to 200). How many components are needed to explain 90% of the variance? 95%? 99%?

**(d)** Visualize the top 10 principal components as $28 \times 28$ images. What patterns do you see? The first few should look like basic digit shapes or strokes. How would you describe the structure that each component captures in words?

**(e)** Reconstruct 5 sample digits (choose one example each of digits 0, 3, 5, 7, and 9) using $k = 10, 50, 100, 200, 500$ principal components. Display the reconstructions in a grid (5 digits x 5 values of k, plus the original in the first column). At what value of $k$ do the reconstructions become visually indistinguishable from the originals?

**(f)** Compute the mean squared error (MSE) of reconstruction as a function of $k$, for $k = 1, 2, 5, 10, 20, 50, 100, 200, 500, 784$. Plot on a log-log scale. Verify that at $k = 784$, the reconstruction error is (essentially) zero.

**(g)** **Bonus:** Compute PCA via the SVD of the centered data matrix instead of the eigendecomposition of the covariance matrix. Verify that you get the same principal components (up to sign). Which method is faster for MNIST-sized data?

---

## Problem 4: t-SNE Visualization of MNIST

**(a)** Take a random subset of 5000 MNIST images (for computational tractability). Apply t-SNE to reduce from 784 dimensions to 2 dimensions. You may use `sklearn.manifold.TSNE`. Set `random_state=42` for reproducibility.

**(b)** Create a scatter plot of the 2D embeddings, coloring each point by its digit label (0-9). Use a colormap with 10 distinct colors and include a legend. Use a small marker size and some transparency (`alpha=0.5`) so overlapping points are visible.

**(c)** Observe the plot carefully and answer:
- Are the 10 digit classes clearly separated into distinct clusters?
- Which digits are closest to each other in the embedding? Does this match your intuition about visual similarity? (For example, 4 and 9 are visually similar, as are 3 and 8.)
- Are there any outlier points — digits that land far from their class cluster? Can you explain why certain digits might be ambiguous?

**(d)** Run t-SNE again with different `perplexity` values: 5, 30, and 100. Display all three plots side by side. How does changing perplexity affect:
- The size and compactness of clusters?
- The spacing between clusters?
- The presence of fine structure within clusters?

Which perplexity gives the most informative plot for understanding MNIST structure?

**(e)** **Comparison with PCA:** Project the same 5000 points to 2D using PCA and create a similar colored scatter plot. How does the PCA 2D projection compare to t-SNE? Which method shows better cluster separation? Why is PCA unable to separate the digits as well?

**(f)** **First-then-t-SNE:** Apply PCA to reduce from 784 to 50 dimensions, then apply t-SNE on the 50-dimensional data. Compare the result to t-SNE on the raw 784-dimensional data. Is there a noticeable difference? (This PCA-then-t-SNE pipeline is a common practice — explain why it might be beneficial.)

---

## Problem 5: Visualizing Learned Representations

This problem explores how a trained neural network transforms data through its layers.

**(a)** Train a simple classifier on MNIST:
- Architecture: 784 -> 256 -> 64 -> 10
- Activation: ReLU on hidden layers, no activation on output
- Loss: CrossEntropyLoss
- Optimizer: Adam, lr=0.001
- Train for 10 epochs, batch size 64

Report the final test accuracy (it should be ~97%).

**(b)** Extract intermediate representations from the trained network. For a random subset of 3000 test images, compute:
- The raw 784-dimensional pixel input
- The 256-dimensional output of the first hidden layer (after ReLU)
- The 64-dimensional output of the second hidden layer (after ReLU)
- The 10-dimensional output (logits) of the final layer

**(c)** Apply t-SNE to each of these four representations (784-dim, 256-dim, 64-dim, and 10-dim) and create four scatter plots, coloring by digit label. Arrange them in a 2x2 grid with appropriate titles.

**(d)** Answer these questions in detail:
- How does cluster separation change as you go deeper into the network? Describe the progression qualitatively.
- At which layer do the digit classes first become clearly separable?
- Are the clusters in the 10-dimensional logit space nearly linearly separable? (They should be — that is what the final softmax classification requires.)
- How do the learned 256-dimensional representations compare to the raw pixel t-SNE from Problem 4?
- Do any digit classes remain close together even in the deeper layers? Which ones?

**(e)** **Quantitative measure (linear probes):** For each of the four representations (raw pixels, 256-dim, 64-dim, logits), train a simple logistic regression classifier (`sklearn.linear_model.LogisticRegression(max_iter=1000)`) on the training set representations and report test accuracy. Plot linear probe accuracy vs. layer depth. How does this quantitative measure align with the visual impression from t-SNE?

---

## Problem 6: The Manifold Hypothesis (Conceptual)

**(a)** A 256x256 RGB image has dimension $d = 256 \times 256 \times 3 = 196{,}608$. Suppose the space of "natural photographs" forms a manifold of intrinsic dimension $k \approx 100$ within this ambient space. What fraction of the ambient dimensions does the manifold occupy? Express this as a ratio and comment on its magnitude.

**(b)** If we were to uniformly sample random vectors in $\mathbb{R}^{196{,}608}$ (each pixel independently uniform in $[0, 255]$), what would these "images" look like? Would any of them resemble natural photographs? Explain why or why not, and estimate (roughly, order-of-magnitude) how many random samples you would need before seeing something that resembles a natural image.

**(c)** PCA on a dataset of face images finds that 50 components explain 90% of the variance. Does this prove the manifold hypothesis for faces? What does it confirm, and what does it leave open? (Hint: PCA captures linear structure, but the face manifold might be curved.)

**(d)** Consider a dataset of 100 points sampled uniformly on a circle of radius 1 in $\mathbb{R}^2$. The circle is a 1-dimensional manifold.
- If you apply PCA, how many components will have significant eigenvalues?
- Will PCA correctly identify the intrinsic dimensionality as 1?
- Explain your answer geometrically.

**(e)** Now consider the Swiss roll: a 2D surface rolled up in 3D. Data is sampled uniformly on this surface.
- What is the intrinsic dimensionality?
- How many significant principal components will PCA find?
- Why does PCA overestimate the intrinsic dimensionality in this case?
- How would t-SNE or UMAP handle this dataset differently?

**(f)** In the context of MNIST, the data lives in $\mathbb{R}^{784}$ but (from Problem 3) roughly 50-100 PCA components capture most of the variance. Does this mean the digit manifold is approximately 50-100 dimensional? Or could the true intrinsic dimensionality be much lower? What is the distinction between the number of significant PCA components and the intrinsic dimension of the manifold?

---

## Submission Checklist

- [ ] Problem 1: PCA derived from maximum variance with Lagrange multipliers, including reconstruction error equivalence
- [ ] Problem 2: Complete guided proof that linear AE = PCA, including uniqueness discussion
- [ ] Problem 3: PCA from scratch with scree plot, component visualization, reconstructions, and MSE curve
- [ ] Problem 4: t-SNE visualizations with perplexity comparison, PCA comparison, and PCA-then-t-SNE
- [ ] Problem 5: Layer-by-layer representation visualization with linear probe accuracy analysis
- [ ] Problem 6: Manifold hypothesis conceptual analysis with Swiss roll and MNIST discussion
