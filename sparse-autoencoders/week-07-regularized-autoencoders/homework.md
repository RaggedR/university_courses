# Week 7: Regularized Autoencoders -- Homework

**Estimated time:** 8-10 hours
**Prerequisites:** Week 6 autoencoder implementation, PyTorch basics from Weeks 3-4

For all implementation problems, use MNIST (available via `torchvision.datasets.MNIST`). Unless stated otherwise, flatten images to vectors of dimension 784.

---

## Problem 1: The Identity Mapping in Overcomplete Autoencoders (Theory)

### Part (a)

Prove that a linear autoencoder with encoder $f(x) = Wx + b$ (where $W \in \mathbb{R}^{d_z \times d_x}$) and decoder $g(z) = W'z + b'$ (where $W' \in \mathbb{R}^{d_x \times d_z}$) can represent the identity function whenever $d_z \geq d_x$.

Specifically, construct explicit matrices $W, W'$ and biases $b, b'$ such that $g(f(x)) = x$ for all $x \in \mathbb{R}^{d_x}$.

### Part (b)

Now consider a nonlinear autoencoder where the encoder has one hidden layer:

$$f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

with $W_1 \in \mathbb{R}^{m \times d_x}$, $W_2 \in \mathbb{R}^{d_z \times m}$, and the decoder is linear: $g(z) = W'z + b'$. Suppose $d_z < d_x$ but $m$ is very large (say, $m \gg d_x$).

Argue informally (no rigorous proof needed) that this network can still approximate the identity function on a finite training set. What role does $m$ play?

*Hint: Think about what happens when $m$ is large enough that each training example can be "routed" through its own set of ReLU neurons.*

### Part (c)

Train a linear overcomplete autoencoder ($d_z = 1000$, $d_x = 784$) on MNIST with no regularization. Report the training loss after convergence. Compute the product $W' W$ and compare it to the identity matrix $I_{784}$. What do you observe?

---

## Problem 2: Denoising Autoencoder (Implementation)

Build a denoising autoencoder for MNIST with the following architecture:
- Encoder: $784 \to 512 \to 256$ (ReLU activations)
- Latent dimension: $d_z = 256$
- Decoder: $256 \to 512 \to 784$ (ReLU activations, sigmoid on final layer)

### Part (a): Gaussian Noise DAE

Train the autoencoder with Gaussian noise corruption: $\tilde{x} = x + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, clipping $\tilde{x}$ to $[0, 1]$.

Train three models with $\sigma \in \{0.1, 0.3, 0.5\}$. For each:
1. Report the final test reconstruction MSE (computed on *clean* inputs, not corrupted).
2. Display a figure showing: (row 1) original images, (row 2) corrupted images, (row 3) reconstructed images. Show 10 examples.

### Part (b): Masking Noise DAE

Train the same architecture with masking noise: each pixel is independently set to 0 with probability $p$.

Train three models with $p \in \{0.1, 0.3, 0.5\}$. Display the same kind of figure as Part (a).

### Part (c): Comparison

Which noise type produces better reconstructions when the corruption is moderate ($\sigma = 0.3$ vs. $p = 0.3$)? Which produces a better *representation* (see Problem 4 for evaluation)? Discuss why the results might differ.

---

## Problem 3: Contractive Autoencoder (Implementation)

### Part (a): Jacobian Penalty -- Analytical Derivation

Consider a single-layer encoder $f(x) = \sigma(Wx + b)$ where $\sigma$ is the sigmoid function.

1. Derive the expression for the Jacobian $J_f(x)$ in terms of $W$ and the activations $f(x)$.
2. Show that $\|J_f(x)\|_F^2 = \sum_{i=1}^{d_z} [f_i(x)(1 - f_i(x))]^2 \|w_i\|^2$.
3. For ReLU activation $h(a) = \max(0, a)$, what is $h'(a)$? What does the Jacobian penalty become? Why might this be less interesting than the sigmoid case?

### Part (b): Implementation

Implement a contractive autoencoder with:
- Encoder: single layer, $784 \to 256$, sigmoid activation
- Decoder: single layer, $256 \to 784$, sigmoid activation
- Contractive penalty: $\lambda \|J_f(x)\|_F^2$ using the analytical expression from Part (a)

Train with $\lambda \in \{0, 0.01, 0.1, 1.0\}$.

For each value of $\lambda$:
1. Report the reconstruction MSE on the test set.
2. Compute the average $\|J_f(x)\|_F^2$ over the test set.
3. Visualize 10 reconstructed test images.

Plot reconstruction MSE vs. average Jacobian norm across the four $\lambda$ values. What trade-off do you observe?

### Part (c): Verifying Contractiveness

For the model trained with $\lambda = 0.1$, select 5 test images. For each image:
1. Add small random perturbation $\delta \sim \mathcal{N}(0, 0.01 \cdot I)$
2. Compute $\|f(x + \delta) - f(x)\| / \|\delta\|$ (the ratio of output change to input change)

Compare this ratio for the $\lambda = 0$ model vs. $\lambda = 0.1$ model. Verify that the contractive model has a smaller ratio.

---

## Problem 4: Representation Quality Comparison (Implementation)

A good representation is not just one that reconstructs well -- it should capture meaningful structure. One way to test this: train a simple linear classifier on the latent codes and see how well it classifies digits.

### Setup

Train four autoencoders on MNIST, all with $d_z = 128$:
1. **Vanilla AE** (no regularization)
2. **Gaussian DAE** ($\sigma = 0.3$)
3. **Masking DAE** ($p = 0.3$)
4. **Contractive AE** ($\lambda = 0.1$)

Use the same encoder-decoder architecture for all: $784 \to 512 \to 128$ (encoder), $128 \to 512 \to 784$ (decoder), ReLU activations. For the CAE, you may approximate the Jacobian penalty using only the first encoder layer to keep computation manageable.

### Evaluation

For each trained autoencoder:
1. Encode all training and test images to get latent representations $z = f(x)$
2. Train a **logistic regression** classifier (from scikit-learn, `LogisticRegression`) on the training latent codes to predict digit labels
3. Report test classification accuracy

### Questions

1. Which autoencoder produces the best classification accuracy?
2. Which produces the best reconstruction MSE?
3. Are these the same model? If not, what does this tell you about the relationship between reconstruction quality and representation quality?
4. Visualize the latent space using t-SNE (from Week 5) for each model. Which model produces the most separated clusters?

---

## Problem 5: DAE Approximates CAE (Theory)

This problem walks you through the proof that denoising autoencoders implicitly perform contractive regularization.

Let $r(x) = g(f(x))$ be the reconstruction function of an autoencoder. Consider a DAE with Gaussian noise: $\tilde{x} = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.

### Part (a)

Write the Taylor expansion of $r(\tilde{x}) = r(x + \epsilon)$ around $x$, keeping terms up to first order in $\epsilon$:

$$r(x + \epsilon) \approx \; ???$$

### Part (b)

Substitute your Taylor expansion into the reconstruction loss $\|x - r(\tilde{x})\|^2$ and expand. You should get three terms.

### Part (c)

Take the expectation $\mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)}$ of each term. Use these facts:
- $\mathbb{E}[\epsilon] = 0$
- $\mathbb{E}[\epsilon \epsilon^\top] = \sigma^2 I$
- For any matrix $A$: $\mathbb{E}[\|A\epsilon\|^2] = \sigma^2 \|A\|_F^2$

Show that:

$$\mathbb{E}_\epsilon \left[ \|x - r(x + \epsilon)\|^2 \right] = \|x - r(x)\|^2 + \sigma^2 \|J_r(x)\|_F^2 + O(\sigma^4)$$

### Part (d)

Interpret this result. What are the two terms? How does this connect DAEs to CAEs?

### Part (e)

Under what conditions is this approximation good? When might it break down? (Think about the size of $\sigma$ relative to the curvature of $r$.)

---

## Problem 6: Overcomplete DAE Learns Useful Features (Implementation)

This problem demonstrates the power of denoising regularization: an overcomplete DAE learns useful features despite having more latent dimensions than input dimensions.

### Setup

Train two overcomplete autoencoders on MNIST with $d_z = 1000$ (recall $d_x = 784$):
1. **Vanilla overcomplete AE** (no regularization)
2. **Overcomplete DAE** (masking noise with $p = 0.5$)

Use architecture: $784 \to 1000$ (encoder, ReLU), $1000 \to 784$ (decoder, sigmoid). Single-layer encoder and decoder.

### Part (a): Reconstruction Quality

Report test MSE for both. The vanilla AE should achieve near-zero training loss. Does it also have low test loss? Compare to the DAE.

### Part (b): Feature Visualization

For the single-layer decoder $g(z) = \sigma(W'z + b')$, each column of $W'$ is a "feature" -- the pattern in input space that a single latent unit contributes to the reconstruction.

Visualize the first 100 columns of $W'$ (reshaped to 28x28) for both models, displayed as a 10x10 grid.

For the vanilla AE: do the features look like meaningful patterns, or noise?
For the DAE: what do the features look like? Do they resemble parts of digits (strokes, curves, edges)?

### Part (c): Sparsity of Activations

For each model, compute the average activation of each latent unit over the test set: $\bar{a}_i = \frac{1}{N} \sum_{n=1}^{N} f_i(x_n)$.

Plot a histogram of $\bar{a}_i$ for both models. Is the DAE's activation pattern sparser than the vanilla AE's?

### Part (d): Discussion

Based on your observations, explain why:
1. The vanilla overcomplete AE fails to learn useful features even though it achieves low training loss
2. The DAE succeeds despite having more latent dimensions than needed
3. How does this relate to the idea that "the autoencoder should learn about the data distribution, not just memorize inputs"?

---

## Submission Checklist

- [ ] Problem 1: Proof (a), informal argument (b), empirical verification (c)
- [ ] Problem 2: DAE with Gaussian and masking noise, reconstructions for 6 models
- [ ] Problem 3: Jacobian derivation, CAE implementation, contractiveness verification
- [ ] Problem 4: Four autoencoders trained, classification accuracies, t-SNE visualizations
- [ ] Problem 5: Complete DAE-CAE equivalence proof
- [ ] Problem 6: Overcomplete vanilla AE vs. DAE comparison, feature visualizations

All code should be in Python using PyTorch. Submit as a Jupyter notebook or Python scripts with clearly labeled outputs.
