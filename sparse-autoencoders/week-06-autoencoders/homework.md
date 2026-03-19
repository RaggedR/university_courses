# Week 6: The Autoencoder — Homework

## Problem Set Overview

This problem set takes you from implementing your first autoencoder to understanding its latent space. You will build fully-connected and convolutional autoencoders, explore how bottleneck size affects reconstruction, visualize and interpolate in latent space, implement tied weights, and prove that overcomplete linear autoencoders can learn the identity.

**Estimated time:** 8-10 hours

**Submission:** A Jupyter notebook containing your solutions, code, plots, and written explanations.

---

## Problem 1: Your First Autoencoder

Implement a fully-connected autoencoder in PyTorch and train it on MNIST.

**(a)** Implement the following architecture:

```
Encoder: 784 -> 256 -> 64 -> k
Decoder: k -> 64 -> 256 -> 784
```

Use ReLU activations on all hidden layers. Use sigmoid on the decoder output (since MNIST pixels are in $[0,1]$). The bottleneck dimension $k$ should be a parameter you can vary.

Your implementation should define a class `Autoencoder(nn.Module)` with:
- `encode(self, x)` — returns the latent code $\mathbf{z}$
- `decode(self, z)` — returns the reconstruction $\hat{\mathbf{x}}$
- `forward(self, x)` — returns `self.decode(self.encode(x))`

**(b)** Train the autoencoder with $k = 32$ for 20 epochs using:
- Optimizer: Adam, lr = $10^{-3}$
- Loss: BCE (binary cross-entropy) — use `nn.BCELoss()`
- Batch size: 128
- Normalize MNIST pixel values to $[0, 1]$

Plot the training loss curve (loss per epoch).

**(c)** Visualize 10 random test images alongside their reconstructions. Display them in a 2-row grid: original on top, reconstruction on bottom.

**(d)** Compute the average reconstruction loss on the entire test set. Compare it to a PCA baseline: apply PCA with $k = 32$ components to the same test data and compute the MSE of the PCA reconstruction. Which method gives better reconstruction? (Note: you will need to convert BCE loss to MSE for a fair comparison, or compute MSE for both.)

---

## Problem 2: Bottleneck Size Experiment

**(a)** Train autoencoders with bottleneck sizes $k \in \{2, 5, 10, 20, 50, 100, 200\}$. Use the same architecture and training setup as Problem 1 (but vary $k$).

**(b)** For each value of $k$, record:
- Final test reconstruction loss (MSE)
- Train a logistic regression classifier on the latent codes and record test accuracy

Plot both metrics as functions of $k$ (two subplots).

**(c)** For each value of $k$, show the reconstruction of the *same* 5 test digits. Display as a grid: rows = values of $k$, columns = the 5 digits. This visualization should clearly show how reconstruction quality improves with $k$.

**(d)** Based on your plots, what is the "sweet spot" bottleneck size — the smallest $k$ that gives good reconstructions? How does this compare to the intrinsic dimensionality suggested by the PCA scree plot from Week 5?

**(e)** For $k = 2$: is the autoencoder reconstruction better or worse than PCA with 2 components? Why? (The nonlinear encoder should have an advantage over linear PCA — verify this.)

---

## Problem 3: Visualizing the Latent Space

**(a)** Train an autoencoder with $k = 2$ (2-dimensional latent space). Use the architecture from Problem 1 but with the bottleneck reduced to 2.

**(b)** Encode the entire MNIST test set and create a scatter plot of the 2D latent codes, coloring by digit label. Include a legend. Do the digit classes form recognizable clusters?

**(c)** Create a **latent space grid visualization:** sample a uniform grid of points in the 2D latent space (covering the range of the encoded test data). Decode each grid point and display the resulting images in a grid. This shows you what the decoder "imagines" at each point in latent space.

For example: if the encoded test data spans roughly $[-3, 3]$ in both dimensions, create a 20x20 grid spanning $[-3, 3] \times [-3, 3]$, decode each point, and display the 400 resulting images as a large grid.

**(d)** Do you observe smooth transitions between digit classes in the grid? Are there "holes" — regions where the decoder produces unrealistic or noisy images? Where are these holes relative to the encoded data?

**(e)** Repeat (b) using an autoencoder with $k = 32$, but apply t-SNE to reduce the 32-dimensional latent codes to 2D for visualization. Compare the cluster quality to the $k = 2$ scatter plot and to the raw pixel t-SNE from Week 5.

---

## Problem 4: Interpolation in Latent Space

**(a)** Using your trained autoencoder from Problem 1 ($k = 32$), select pairs of test images:
- A "3" and an "8"
- A "1" and a "7"
- A "4" and a "9"
- A "0" and a "6"

**(b)** For each pair $(\mathbf{x}_A, \mathbf{x}_B)$:
1. Encode: $\mathbf{z}_A = f(\mathbf{x}_A)$, $\mathbf{z}_B = f(\mathbf{x}_B)$
2. Interpolate: $\mathbf{z}_\alpha = (1 - \alpha)\mathbf{z}_A + \alpha\mathbf{z}_B$ for $\alpha \in \{0, 0.1, 0.2, \ldots, 0.9, 1.0\}$
3. Decode: $\hat{\mathbf{x}}_\alpha = g(\mathbf{z}_\alpha)$

Display each interpolation as a row of 11 images ($\alpha$ from 0 to 1).

**(c)** Are the interpolations smooth and realistic? Do they pass through recognizable digit shapes, or do they become blurry/noisy in the middle?

**(d)** Compare with **pixel-space interpolation**: for the same pairs, compute $\hat{\mathbf{x}}_\alpha = (1 - \alpha)\mathbf{x}_A + \alpha\mathbf{x}_B$ (interpolating directly in pixel space). Display the results. Which interpolation — latent space or pixel space — produces more realistic intermediate images?

**(e)** In a few sentences, explain why latent space interpolation is typically better than pixel space interpolation. What does this tell us about the structure of the latent space?

---

## Problem 5: Tied Weights

**(a)** Implement a tied-weight autoencoder. Use a single hidden layer for simplicity:
- Encoder: $\mathbf{z} = \text{ReLU}(W_e \mathbf{x} + \mathbf{b}_e)$, where $W_e \in \mathbb{R}^{k \times 784}$
- Decoder: $\hat{\mathbf{x}} = \sigma(W_e^\top \mathbf{z} + \mathbf{b}_d)$

Note: the decoder weight matrix is $W_e^\top$ — the transpose of the encoder weight. Only the biases $\mathbf{b}_e$ and $\mathbf{b}_d$ are independent.

**(b)** Also implement the untied version: same architecture but with an independent decoder weight matrix $W_d$.

**(c)** Train both models on MNIST with $k = 64$, using the same hyperparameters:
- Adam, lr = $10^{-3}$, 20 epochs, batch size 128, BCE loss

Report:
- Number of trainable parameters for each model
- Final test reconstruction loss for each model
- Train a linear classifier on the latent codes of each model and report test accuracy

**(d)** Visualize the rows of $W_e$ as $28 \times 28$ images for both models (these are the "encoder filters" — patterns that each latent dimension responds to). Do the tied and untied models learn similar or different filters?

**(e)** In 3-4 sentences, discuss the trade-offs of tied weights. When might tied weights be preferable? When might they hurt?

---

## Problem 6: Convolutional Autoencoder

**(a)** Implement a convolutional autoencoder for MNIST:

```
Encoder:
  Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  -> 16x14x14
  ReLU
  Conv2d(16, 32, kernel_size=3, stride=2, padding=1) -> 32x7x7
  ReLU
  Flatten -> 1568
  Linear(1568, k) -> k

Decoder:
  Linear(k, 1568) -> 1568
  Reshape -> 32x7x7
  ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1) -> 16x14x14
  ReLU
  ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1) -> 1x28x28
  Sigmoid
```

**(b)** Train with $k = 32$, Adam lr=$10^{-3}$, 20 epochs, batch size 128, BCE loss.

**(c)** Compare with the fully-connected autoencoder from Problem 1 ($k = 32$):
- Report parameter count for each model
- Report test reconstruction loss (MSE) for each
- Show side-by-side reconstructions of 10 test images: original / FC reconstruction / Conv reconstruction

**(d)** Which model gives better reconstructions? Which has fewer parameters? Discuss the advantages of the convolutional architecture.

**(e)** **Bonus:** Try removing the fully-connected bottleneck layer entirely — let the 32x7x7 = 1568-dimensional feature maps serve as the latent representation (with no compression to $k$ dimensions). How does reconstruction quality change? Is this still a useful autoencoder? Why or why not?

---

## Problem 7: Overcomplete Linear Autoencoders (Theory)

**(a)** Consider a linear autoencoder with $d = 3$ (input dimension) and $k = 5$ (latent dimension). The encoder is $W_e \in \mathbb{R}^{5 \times 3}$ and the decoder is $W_d \in \mathbb{R}^{3 \times 5}$.

Construct specific matrices $W_e$ and $W_d$ such that $W_d W_e = I_3$ (the $3 \times 3$ identity matrix). Verify by computing the product.

**(b)** With your construction from (a), what is the reconstruction error? What is the latent code $\mathbf{z} = W_e \mathbf{x}$ for an arbitrary input $\mathbf{x}$? Is $\mathbf{z}$ a "useful" representation?

**(c)** Prove that for any $k \geq d$, there exist $W_e \in \mathbb{R}^{k \times d}$ and $W_d \in \mathbb{R}^{d \times k}$ such that $W_d W_e = I_d$. (Hint: use the fact that a $d \times k$ matrix with $k \geq d$ can have rank $d$.)

**(d)** Now consider a *nonlinear* overcomplete autoencoder with ReLU activations and $k > d$. Can it still learn the identity function? Explain why or why not. (Hint: think about what happens if the encoder maps all data to the positive orthant, where ReLU is the identity.)

**(e)** In 3-4 sentences, explain why the ability of overcomplete autoencoders to learn the identity motivates the regularization techniques we will study next week. What property should a regularizer enforce to prevent the identity solution?

---

## Submission Checklist

- [ ] Problem 1: Working autoencoder with training curve, reconstructions, and PCA comparison
- [ ] Problem 2: Bottleneck size sweep with reconstruction loss and accuracy plots
- [ ] Problem 3: 2D latent space visualization with scatter plot and grid decode
- [ ] Problem 4: Latent space interpolation for 4 digit pairs, comparison with pixel interpolation
- [ ] Problem 5: Tied vs. untied weights comparison with filter visualization
- [ ] Problem 6: Convolutional autoencoder with FC comparison
- [ ] Problem 7: Overcomplete identity construction and theoretical analysis
