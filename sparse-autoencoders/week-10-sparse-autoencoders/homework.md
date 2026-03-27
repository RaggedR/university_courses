# Week 10: Sparse Autoencoders -- Homework

## Overview

This problem set has you build sparse autoencoders from scratch, compare two different sparsity penalties, and develop the practical intuitions that separate a working SAE from a broken one. By the end, you should be able to train an SAE, diagnose its failure modes, and evaluate its features.

**Tools:** Python, PyTorch, matplotlib, torchvision (for MNIST), scikit-learn (for PCA and dictionary learning baselines).

**Estimated time:** 8-12 hours.

---

## Problem 1: SAE with L1 Penalty (Implementation)

Implement a sparse autoencoder with L1 penalty and train it on MNIST.

### 1a. Architecture

Implement the following SAE class in PyTorch:

- **Encoder:** Linear layer $784 \to 2000$, followed by ReLU activation.
- **Decoder:** Linear layer $2000 \to 784$, no activation (linear output).
- **Pre-encoder bias:** Subtract the dataset mean from inputs before encoding; add it back after decoding.

Your `forward` method should return both the reconstruction $\hat{\mathbf{x}}$ and the hidden activations $\mathbf{z}$ (you will need $\mathbf{z}$ for the sparsity penalty).

### 1b. Loss Function

Implement the L1-penalized loss:

$$
\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2 + \lambda \frac{1}{m} \sum_{i=1}^{m} \|\mathbf{z}_i\|_1
$$

Use $\lambda = 0.005$ initially.

### 1c. Decoder Norm Constraint

After each optimizer step, project each column of the decoder weight matrix to unit L2 norm. Implement this as a separate function that modifies the weights in-place using `torch.no_grad()`.

### 1d. Training

Train for 50 epochs on MNIST using Adam with learning rate $3 \times 10^{-4}$ and batch size 256. Normalize pixel values to $[0, 1]$.

Log and plot (per epoch):
- Total loss
- Reconstruction loss (MSE component only)
- Average L0 (number of non-zero activations per input)
- Average L1 (sum of activations per input)

### 1e. Visualization

After training:
1. **Decoder columns as features:** Reshape each of the 2000 decoder columns to $28 \times 28$ and display the first 100 as a $10 \times 10$ grid of small images. Use a diverging colormap (e.g., `RdBu_r`).
2. **Reconstructions:** Show 10 original MNIST digits and their SAE reconstructions side by side.
3. **Activation histogram:** For a batch of 1000 inputs, plot a histogram of all activation values $z\_j$. Comment on the distribution -- how many are exactly zero?

---

## Problem 2: SAE with KL Divergence Penalty (Implementation)

Implement a sparse autoencoder with the KL divergence penalty and compare it to the L1 version.

### 2a. Architecture

Use the same encoder-decoder architecture as Problem 1.

### 2b. KL Loss

Implement the KL divergence sparsity loss. For each minibatch:

1. Compute the average activation of each neuron across the batch:
   $$
   \hat{\rho}_j = \frac{1}{m} \sum_{i=1}^m z_j(\mathbf{x}_i)
   $$

2. Compute the Bernoulli KL divergence for each neuron:
   $$
   \text{KL}_j = \rho \log \frac{\rho}{\hat{\rho}_j} + (1 - \rho) \log \frac{1 - \rho}{1 - \hat{\rho}_j}
   $$

3. Sum over all neurons and add to the reconstruction loss:
   $$
   \mathcal{L} = \text{MSE} + \beta \sum_{j=1}^{d} \text{KL}_j
   $$

Use target sparsity $\rho = 0.05$ and weight $\beta = 3.0$ as starting values.

**Implementation note:** You will need to clamp $\hat{\rho}\_j$ to avoid $\log(0)$. Use `torch.clamp(rho_hat, min=1e-6, max=1-1e-6)`.

### 2c. Training

Train with the same setup as Problem 1 (50 epochs, same optimizer and batch size).

### 2d. Comparison

Create a side-by-side comparison of the L1 and KL SAEs:

1. Feature visualizations (decoder columns) -- do they look qualitatively different?
2. Activation distributions -- how does the distribution of non-zero activations compare?
3. Average L0 -- which method produces sparser representations?
4. Reconstruction MSE -- which reconstructs better?

Write a paragraph summarizing the differences you observe.

---

## Problem 3: KL Divergence Derivation (Theory)

### 3a. Derivation

Starting from the definition of KL divergence for discrete distributions:

$$
\text{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

derive the formula for the KL divergence between $\text{Bernoulli}(\rho)$ and $\text{Bernoulli}(\hat{\rho})$:

$$
\text{KL}(\rho \| \hat{\rho}) = \rho \log \frac{\rho}{\hat{\rho}} + (1 - \rho) \log \frac{1 - \rho}{1 - \hat{\rho}}
$$

Show each step explicitly.

### 3b. Gradient

Compute $\frac{\partial}{\partial \hat{\rho}} \text{KL}(\rho \Vert  \hat{\rho})$.

Verify that this gradient:
- Equals zero when $\hat{\rho} = \rho$
- Is negative when $\hat{\rho} < \rho$ (the penalty pushes the neuron to be more active)
- Is positive when $\hat{\rho} > \rho$ (the penalty pushes the neuron to be less active)

### 3c. Second Derivative

Compute $\frac{\partial^2}{\partial \hat{\rho}^2} \text{KL}(\rho \Vert  \hat{\rho})$ and show that it is always positive for $\hat{\rho} \in (0, 1)$. What does this tell you about the shape of the KL penalty as a function of $\hat{\rho}$?

### 3d. Numerical Verification

Write a short Python script that:
1. Computes the KL divergence numerically for $\rho = 0.05$ and $\hat{\rho} \in [0.001, 0.999]$.
2. Plots $\text{KL}(\rho \Vert  \hat{\rho})$ as a function of $\hat{\rho}$.
3. Overlays the analytical gradient on the same plot (as a second y-axis or a separate subplot).

Verify that the minimum is at $\hat{\rho} = \rho = 0.05$ and that the gradient changes sign there.

---

## Problem 4: The Sparsity-Reconstruction Trade-off (Experimental)

### 4a. Lambda Sweep

Train L1 SAEs (same architecture as Problem 1) with the following values of $\lambda$:

$$
\lambda \in \{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1\}
$$

For each, record:
- Final reconstruction MSE (on the test set)
- Average L0 (on the test set)
- Number of dead neurons (neurons that never activate on 10,000 test examples)

### 4b. Pareto Plot

Create a scatter plot of **reconstruction MSE (y-axis) vs. average L0 (x-axis)**, with each point labeled by its $\lambda$ value. This is the Pareto frontier of your SAE family.

### 4c. Feature Quality

For $\lambda \in \lbrace 0.0005, 0.005, 0.05\rbrace $ (representing low, medium, and high sparsity), visualize the top 25 decoder column features (by decoder column norm or by average activation frequency).

Comment on how feature quality changes with $\lambda$:
- At low $\lambda$, are features interpretable?
- At high $\lambda$, are features interpretable? What happens differently?
- Where is the "sweet spot"?

### 4d. Discussion

In 3-5 sentences, explain the fundamental tension between reconstruction quality and sparsity. Why can we not have both? Under what conditions might it be possible to have very high sparsity with very low reconstruction error?

---

## Problem 5: Dead Neuron Detection and Resampling (Implementation)

### 5a. Detection

Using the best SAE from Problem 1 (or retrain if needed), implement a function `count_dead_neurons(model, dataloader, threshold=0)` that:
1. Passes the entire dataset through the encoder.
2. For each hidden neuron, checks whether it activated (value > threshold) on *any* input.
3. Returns the number and indices of dead neurons.

Report: how many neurons are dead in your trained SAE?

### 5b. Resampling

Implement a neuron resampling function. The algorithm:

1. **Identify dead neurons:** neurons with zero activations on a large batch (e.g., 10,000 examples).
2. **Compute reconstruction errors** $e\_i = \Vert \mathbf{x}\_i - \hat{\mathbf{x}}\_i\Vert ^2$ for each example.
3. **Sample a data point** $\mathbf{x}\_i$ with probability proportional to $e\_i$.
4. **Reinitialize the dead neuron's encoder weights:** set them to a normalized version of the residual: $\mathbf{w}\_j \leftarrow c \cdot (\mathbf{x}\_i - \hat{\mathbf{x}}\_i) / \Vert \mathbf{x}\_i - \hat{\mathbf{x}}\_i\Vert $, where $c$ is the average encoder weight norm of alive neurons.
5. **Reinitialize the decoder column** to match (also normalized).
6. **Set the encoder bias** $b\_j$ to $0$ or a small negative value.

### 5c. Training with Resampling

Retrain the SAE from scratch, but now apply resampling every 5 epochs. Compare:
- Number of dead neurons at the end of training (with vs. without resampling)
- Reconstruction MSE
- Average L0
- A feature visualization grid

Does resampling improve feature utilization without hurting reconstruction?

---

## Problem 6: SAE vs. PCA vs. Dictionary Learning (Comparative)

### 6a. Baselines

On MNIST, train/compute the following:
1. **PCA** with 50 components (using scikit-learn).
2. **Dictionary Learning** with 2000 atoms and sparsity parameter similar to your SAE (using `sklearn.decomposition.MiniBatchDictionaryLearning`).
3. **Your SAE** from Problem 1.

### 6b. Feature Visualization

Visualize 25 features (components/atoms/decoder columns) from each method as $28 \times 28$ images. Arrange them in a $3 \times 25$ or similar grid to facilitate comparison.

### 6c. Reconstruction

For 10 test digits, show the reconstruction from each method. Quantify reconstruction quality (MSE) for each.

### 6d. Downstream Classification

Use the features from each method as input to a logistic regression classifier (scikit-learn) predicting digit labels. Report test accuracy for each method.

### 6e. Analysis

Write a comparative analysis (1-2 paragraphs) addressing:
- Which method produces the most visually interpretable features?
- Which method achieves the best reconstruction?
- Which method produces the best features for classification?
- Are the answers to these three questions the same? Why or why not?

---

## Problem 7: The Decoder Norm Exploit (Theory + Experiment)

### 7a. Theoretical Argument

Prove the following claim: if the SAE loss is

$$
\mathcal{L} = \|\mathbf{x} - \mathbf{W}_d \mathbf{z}\|_2^2 + \lambda \|\mathbf{z}\|_1
$$

and there is no constraint on $\mathbf{W}\_d$, then for any encoding $\mathbf{z}$ with $\mathbf{z} \neq \mathbf{0}$, we can find $\mathbf{W}\_d'$ and $\mathbf{z}'$ such that:
- $\mathbf{W}\_d' \mathbf{z}' = \mathbf{W}\_d \mathbf{z}$ (same reconstruction)
- $\Vert \mathbf{z}'\Vert \_1 < \Vert \mathbf{z}\Vert \_1$ (lower sparsity penalty)

Conclude that without a decoder norm constraint, the optimal strategy drives $\Vert \mathbf{z}\Vert \_1 \to 0$, making the sparsity penalty meaningless.

*Hint:* Consider the scaling transformation $\mathbf{z}' = \mathbf{z}/\alpha$, $\mathbf{W}\_d' = \alpha \mathbf{W}\_d$ for $\alpha > 1$.

### 7b. Experimental Verification

Train two SAEs:
1. **With** decoder norm constraint (column normalization after each step).
2. **Without** decoder norm constraint.

Use the same architecture and $\lambda$ for both. After training, report:
- The distribution of decoder column norms for each model.
- The distribution of hidden activations for each model.
- Reconstruction MSE for each model.
- Average L0 for each model.

Does the unconstrained model learn features with inflated decoder norms and shrunken activations, as the theory predicts?

### 7c. Feature Comparison

Visualize 25 decoder columns from each model. For the unconstrained model, normalize each column to unit norm before visualizing (so you can see the feature directions regardless of magnitude). Do the underlying feature *directions* differ between the two models, or is the main difference just the scaling?

---

## Submission Checklist

- [ ] Problem 1: SAE with L1 penalty -- code, training curves, feature visualizations, reconstructions, activation histogram
- [ ] Problem 2: SAE with KL penalty -- code, training curves, comparison with L1
- [ ] Problem 3: KL derivation -- written derivation, gradient computation, numerical verification plot
- [ ] Problem 4: Lambda sweep -- Pareto plot, feature visualizations at three sparsity levels, discussion
- [ ] Problem 5: Dead neuron resampling -- detection code, resampling code, comparison with/without resampling
- [ ] Problem 6: Comparative study -- feature visualizations, reconstructions, classification accuracies, analysis
- [ ] Problem 7: Decoder norm exploit -- proof, experimental comparison, feature visualizations
