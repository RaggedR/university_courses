# Week 4: Training Neural Networks — Homework

## Problem Set Overview

This problem set covers backpropagation, optimization algorithms, and regularization. Problems 1-2 are pen-and-paper derivations. Problems 3-6 are implementation exercises in Python/PyTorch. Problem 7 is conceptual.

**Estimated time:** 6-8 hours

**Submission:** A Jupyter notebook containing your solutions, code, plots, and written explanations.

---

## Problem 1: Backpropagation by Hand (Pen and Paper)

Consider a neural network with the following architecture:
- Input: $\mathbf{x} \in \mathbb{R}^2$
- Hidden layer: 3 units with ReLU activation
- Output: 1 unit (no activation, i.e., regression)
- Loss: MSE, $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$

The parameters are:

$$W^{(1)} = \begin{pmatrix} 0.5 & -0.3 \\ 0.2 & 0.8 \\ -0.1 & 0.4 \end{pmatrix}, \quad \mathbf{b}^{(1)} = \begin{pmatrix} 0.1 \\ -0.2 \\ 0.0 \end{pmatrix}$$

$$W^{(2)} = \begin{pmatrix} 0.6 & -0.4 & 0.3 \end{pmatrix}, \quad b^{(2)} = 0.1$$

The input is $\mathbf{x} = (1, -1)^\top$ and the target is $y = 0.5$.

**(a)** Perform the forward pass. Compute $\mathbf{a}^{(1)}$, $\mathbf{h}$, $a^{(2)}$, $\hat{y}$, and $\mathcal{L}$. Show every intermediate value.

**(b)** Perform the backward pass. Compute:
- $\delta^{(2)} = \frac{\partial \mathcal{L}}{\partial a^{(2)}}$
- $\frac{\partial \mathcal{L}}{\partial W^{(2)}}$ and $\frac{\partial \mathcal{L}}{\partial b^{(2)}}$
- $\frac{\partial \mathcal{L}}{\partial \mathbf{h}}$
- $\boldsymbol{\delta}^{(1)} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}}$ (be careful with ReLU — which units have zero gradient?)
- $\frac{\partial \mathcal{L}}{\partial W^{(1)}}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}}$

**(c)** Apply one step of gradient descent with learning rate $\eta = 0.1$. Write the updated parameters.

**(d)** Perform a second forward pass with the updated parameters. Verify that the loss decreased.

---

## Problem 2: Deriving Adam (Pen and Paper)

**(a)** Write out the complete Adam algorithm, starting from the initialization of $m_0 = 0$, $v_0 = 0$, through the bias-corrected update at step $t$.

**(b)** Show that at step $t=1$ with $\beta_1 = 0.9$ and $\beta_2 = 0.999$, the uncorrected first moment $m_1$ underestimates the gradient by a factor of 10. What is the bias correction factor at $t=1$?

**(c)** What happens to Adam when $\beta_1 = 0$ (no momentum)? Write the simplified update rule and identify which optimizer it reduces to.

**(d)** What happens to Adam when $\beta_2 = 0$ (no second moment tracking, i.e., $v_t = g_t^2$ at each step)? Write the simplified update rule. How does this compare to standard SGD?

**(e)** Explain in 3-4 sentences why Adam's update is approximately scale-invariant: if you multiply all the gradients by a constant $c$, the step direction remains approximately the same. (Hint: consider the ratio $\hat{m} / \sqrt{\hat{v}}$.)

---

## Problem 3: Backpropagation from Scratch

Extend the simple neural network from Week 3 (NumPy-only implementation) by implementing backpropagation from scratch.

**Requirements:**

1. Implement a `class TwoLayerNet` with the following methods:
   - `__init__(self, d_in, d_hidden, d_out)` — initialize weights using He initialization
   - `forward(self, X)` — forward pass, storing intermediate values needed for backprop
   - `backward(self, y)` — backward pass, computing gradients for all parameters
   - `update(self, lr)` — apply one step of gradient descent

2. Use ReLU activation for the hidden layer and no activation for the output layer (regression) or softmax for classification.

3. Train the network on a simple 2D classification problem:
   - Generate 500 points from `sklearn.datasets.make_moons(n_samples=500, noise=0.2)`
   - Split into 400 train / 100 test
   - Train for 1000 steps with learning rate 0.01
   - Plot the training loss curve
   - Plot the decision boundary

4. **Verification:** After implementing your own backward pass, verify that your gradients match PyTorch's autograd. For a single mini-batch, compute gradients both ways and print the maximum absolute difference. It should be $< 10^{-6}$.

---

## Problem 4: Comparing Optimizers on MNIST

Train a 2-hidden-layer MLP (784 -> 256 -> 128 -> 10) on MNIST using three different optimizers. For each optimizer, use the best commonly-recommended hyperparameters:

1. **SGD** with learning rate 0.01
2. **SGD + Momentum** with learning rate 0.01, momentum 0.9
3. **Adam** with learning rate 0.001

**Requirements:**

**(a)** Use the same network architecture and initialization for all three (set a random seed). Use cross-entropy loss, batch size 64, and train for 20 epochs.

**(b)** For each optimizer, record and plot:
- Training loss per mini-batch (or per 100 mini-batches)
- Validation accuracy per epoch

All three curves should appear on the same plot for easy comparison.

**(c)** Report the final test accuracy for each optimizer.

**(d)** Answer these questions in a few sentences each:
- Which optimizer converges fastest in terms of number of epochs?
- Which achieves the best final accuracy?
- How does the training loss curve for SGD (no momentum) differ qualitatively from Adam?

**(e)** **Bonus:** Add a fourth curve for SGD + Nesterov momentum. Does it differ noticeably from standard momentum?

---

## Problem 5: Implementing Dropout from Scratch

**(a)** Implement dropout as a function `dropout(x, p, training)` in NumPy:
- During training: generate a binary mask from $\text{Bernoulli}(1-p)$, multiply element-wise, scale by $1/(1-p)$
- During evaluation: return `x` unchanged

**(b)** Verify your implementation:
- For a random input tensor of shape (1000,), apply dropout with $p=0.5$ during training
- Compute the mean of the output. It should be approximately equal to the mean of the input (within ~5%)
- Compute the fraction of zeros. It should be approximately $p$ (within ~5%)

**(c)** Train two networks on MNIST, identical except that one uses dropout ($p = 0.5$ on hidden layers):
- Architecture: 784 -> 512 -> 512 -> 10
- Use Adam with lr=0.001, batch size 64, train for 30 epochs
- Record training loss and validation accuracy for both networks
- Plot both on the same axes

**(d)** Answer: At what point in training does dropout start to help? How large is the gap between training accuracy (no dropout) and validation accuracy (no dropout)? How does dropout affect this gap?

---

## Problem 6: Training MNIST to >97% Accuracy

Your goal: achieve **>97% test accuracy** on MNIST. This is an exercise in practical training — you must make choices and document them.

**Constraints:**
- Use a feedforward network (no convolutions — we have not covered them yet)
- Use PyTorch
- You may use any optimizer, learning rate schedule, regularization, and architecture
- You must train from scratch (no pre-trained weights)

**Deliverables:**

**(a)** Your final model architecture (layer sizes, activations, regularization).

**(b)** Your training configuration (optimizer, learning rate, schedule, batch size, number of epochs).

**(c)** A training log showing:
- Final training loss and accuracy
- Final validation accuracy
- Final test accuracy

**(d)** A brief write-up (200-400 words) documenting your tuning process:
- What did you try first?
- What did not work and why?
- What made the biggest difference?
- Did you use the "overfit one batch" heuristic? What did it reveal?

**Hints:**
- 97% is achievable with a 3-layer MLP using ~256-512 hidden units per layer
- BatchNorm + Dropout + Adam + a learning rate schedule should get you there
- If you are stuck at 96%, try increasing the hidden dimension or adding another layer
- Data normalization matters: normalize inputs to mean 0, std 1

---

## Problem 7: Loss Landscapes and Saddle Points (Conceptual)

**(a)** Consider a function $f(x, y) = x^2 - y^2$. Compute the gradient and Hessian at the origin. Show that the origin is a saddle point. What are the eigenvalues of the Hessian? In which direction does the function decrease, and in which does it increase?

**(b)** Generalize: consider $f(\mathbf{x}) = \sum_{i=1}^d \lambda_i x_i^2$ where each $\lambda_i$ is independently $+1$ or $-1$ with equal probability. What is the probability that the origin is a local minimum (all $\lambda_i > 0$)? Compute this for $d = 10$, $d = 100$, and $d = 10^6$.

**(c)** This is a toy model, but it illustrates a real phenomenon. In a neural network with $10^6$ parameters, which type of critical point dominates — local minima or saddle points? Why does this make optimization easier than you might expect?

**(d)** Consider two minima of a loss function:
- Minimum A: loss = 0.01, but the Hessian has eigenvalues in the range [100, 1000] (sharp minimum)
- Minimum B: loss = 0.02, but the Hessian has eigenvalues in the range [0.1, 1.0] (flat minimum)

Which minimum would you expect to generalize better to test data, and why? Which minimum would SGD (with noise) be more likely to find, and why?

**(e)** Explain in 3-4 sentences how gradient clipping helps with exploding gradients. Why do we clip the *norm* of the gradient (preserving direction) rather than clipping each component independently?

---

## Submission Checklist

- [ ] Problem 1: All intermediate values and gradients computed, loss decreased after update
- [ ] Problem 2: Complete Adam derivation with bias correction analysis
- [ ] Problem 3: Working backprop implementation with gradient verification
- [ ] Problem 4: Training curves for 3+ optimizers on one plot, accuracy comparison
- [ ] Problem 5: Dropout implementation with verification, train/val curves with and without dropout
- [ ] Problem 6: >97% test accuracy with documented tuning process
- [ ] Problem 7: Saddle point analysis with eigenvalue computation
