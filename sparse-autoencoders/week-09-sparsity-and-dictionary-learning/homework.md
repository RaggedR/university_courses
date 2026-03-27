# Week 9: Sparsity and Dictionary Learning -- Homework

**Estimated time:** 10-12 hours
**Prerequisites:** L1/L2 norms (Week 1), optimization (Weeks 1, 4), PyTorch

---

## Problem 1: Why L1 Produces Sparse Solutions (Theory)

### Part (a): The Diamond Argument

Consider a simple 2D problem: minimize $f(\beta\_1, \beta\_2) = (\beta\_1 - 3)^2 + (\beta\_2 - 4)^2$ subject to:
- **L1 constraint:** $|\beta\_1| + |\beta\_2| \leq t$
- **L2 constraint:** $\beta\_1^2 + \beta\_2^2 \leq t$

The unconstrained minimum is at $\hat{\beta} = (3, 4)$.

1. Sketch (by hand or using matplotlib) the level curves of $f$ (concentric circles centered at $(3,4)$) along with:
   - The L1 constraint set (a diamond) for $t = 4$
   - The L2 constraint set (a circle) for $t = 4$

2. For each constraint, identify where the smallest level curve touches the constraint set. For L1, show that the contact point has $\beta\_1 = 0$ (i.e., the solution is sparse). For L2, show that neither coordinate is zero.

3. Explain in 2-3 sentences why this happens geometrically. What property of the diamond makes it likely that level curves first touch at a vertex?

### Part (b): Higher Dimensions

Now consider $\beta \in \mathbb{R}^{10}$ with the unconstrained minimum at $\hat{\beta} = (3, 0.5, 0.1, 0.8, 4, 0.2, 0.05, 1.5, 0.3, 2)$ (some large, some small values).

Solve the LASSO problem:

$$
\min_\beta \frac{1}{2}\Vert \hat{\beta} - \beta\Vert ^2 + \lambda \Vert \beta\Vert _1
$$

for $\lambda \in \lbrace 0.01, 0.1, 0.5, 1.0, 2.0, 5.0\rbrace$.

*Hint: The solution to this simplified problem (identity design matrix) is given by soft-thresholding: $\beta^*\_j = S\_\lambda(\hat{\beta}\_j)$.*

For each $\lambda$, report:
1. The solution $\beta^*$
2. The number of nonzero entries
3. Which entries survive?

Plot the number of nonzero entries vs. $\lambda$. How does sparsity increase with $\lambda$?

---

## Problem 2: Derive the Soft-Thresholding Operator (Theory)

### Part (a)

The proximal operator for $h(z) = \lambda |z|$ (scalar case) is defined as:

$$
\text{prox}_h(v) = \arg\min_z \left\lbrace \frac{1}{2}(z - v)^2 + \lambda |z|\right\rbrace 
$$

Derive the solution by considering three cases: $z > 0$, $z < 0$, and $z = 0$.

For each case:
1. Write the objective function with $|z|$ replaced by $z$ or $-z$
2. Take the derivative (where it exists) and set to zero
3. Check when the case condition is satisfied

Combine the cases to show:

$$
\text{prox}_h(v) = S_\lambda(v) = \text{sign}(v) \max(|v| - \lambda, 0)
$$

### Part (b): Subdifferential Approach

The function $h(z) = \lambda|z|$ is not differentiable at $z = 0$. Its subdifferential at $z = 0$ is the interval $[-\lambda, \lambda]$.

The optimality condition for $\min\_z g(z) + h(z)$ (where $g(z) = \frac{1}{2}(z-v)^2$) is:

$$
0 \in \nabla g(z) + \partial h(z)
$$

Use this condition to re-derive the soft-thresholding operator. Show that when $|v| \leq \lambda$, the optimality condition is satisfied at $z = 0$.

---

## Problem 3: Implement ISTA (Implementation)

Implement the Iterative Shrinkage-Thresholding Algorithm from scratch in PyTorch.

### Part (a): Basic ISTA

Write a function `ista(x, D, lambda_, alpha, num_iters)` that solves:

$$
\min_z \frac{1}{2}\Vert x - Dz\Vert ^2 + \lambda \Vert z\Vert _1
$$

Your implementation should:
1. Initialize $z = 0$
2. Iterate: gradient step followed by soft-thresholding
3. Return the final $z$ and the loss history (objective value at each iteration)

### Part (b): Verify Convergence

Create a synthetic test problem:
- $d\_x = 50$, $d\_z = 100$ (overcomplete)
- Generate a random dictionary $D$ with normalized columns
- Generate a ground truth sparse code $z^*$ with only 5 nonzero entries
- Create the signal $x = Dz^* + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 0.01 \cdot I)$

Run ISTA for 500 iterations. Plot:
1. The objective value vs. iteration number (should decrease monotonically)
2. The recovered $z$ vs. the true $z^*$ (scatter plot). How well does ISTA recover the sparse code?
3. The reconstruction error $\Vert x - Dz^{(t)}\Vert ^2$ vs. iteration

### Part (c): FISTA

Implement FISTA (the accelerated version) by adding the momentum step:

$$
y^{(t+1)} = z^{(t)} + \frac{t-1}{t+2}(z^{(t)} - z^{(t-1)})
$$

Run FISTA on the same problem from Part (b). Plot the objective value of ISTA and FISTA on the same graph. How many iterations does each need to reach objective value within 1% of the final ISTA value?

---

## Problem 4: Sparse Coding on Image Patches (Implementation)

This problem reproduces (in miniature) the famous result of Olshausen and Field (1996).

### Part (a): Data Preparation

1. Load 5-10 natural images (use any convenient source: images from the internet, `torchvision.datasets` that contain natural photos, or the scikit-image sample images like `skimage.data.astronaut()`).
2. Convert to grayscale and normalize to $[0, 1]$.
3. Extract 10000 random patches of size 8x8 (or 12x12 if computationally feasible).
4. Flatten each patch into a vector ($d\_x = 64$ for 8x8 patches).
5. Subtract the mean from each patch (mean-centering).

### Part (b): Dictionary Learning

Implement the alternating optimization algorithm:

```
Initialize D randomly (d_x x d_z), normalize columns
For each epoch:
    For each (mini-batch of) patch(es) x:
        1. Sparse coding: z = ISTA(x, D, lambda, alpha, num_iters)
        2. Dictionary update: D += eta * (x - Dz) @ z.T  (gradient step)
        3. Normalize: d_j = d_j / max(1, ||d_j||)
```

Use $d\_z = 256$ (4x overcomplete for 8x8 patches).

Train for 50-100 epochs. Experiment with $\lambda$ to get 5-15% nonzero entries in $z$.

### Part (c): Visualize the Dictionary

Display all 256 dictionary atoms as a 16x16 grid of small images (each atom reshaped to 8x8).

Do the atoms look like Gabor-like filters (oriented edges and bars at various positions and scales)? Compare your results to Figure 3 of Olshausen and Field (1996).

### Part (d): Analysis

1. For 10 random test patches, display: (i) the original patch, (ii) the reconstruction $Dz^*$, (iii) the active dictionary atoms (i.e., the atoms $d\_j$ for which $z\_j \neq 0$). How many atoms are typically active?

2. Compute the average sparsity (fraction of nonzero entries in $z$) across 1000 test patches.

---

## Problem 5: L1 vs. L2 Regularization Comparison (Implementation)

### Part (a): Regression Setup

Generate a regression problem where sparsity matters:

```python
import numpy as np

np.random.seed(42)
n, p = 100, 50
X = np.random.randn(n, p)

# True coefficients: only 5 are nonzero
beta_true = np.zeros(p)
beta_true[0] = 3.0
beta_true[5] = -2.0
beta_true[10] = 1.5
beta_true[25] = -4.0
beta_true[40] = 2.5

y = X @ beta_true + 0.5 * np.random.randn(n)
```

### Part (b): Solve with L1 and L2

Solve the regularized regression problem for a range of $\lambda$ values:

$$
\hat{\beta}_{\text{L1}} = \arg\min_\beta \frac{1}{2n}\Vert y - X\beta\Vert ^2 + \lambda \Vert \beta\Vert _1
$$

$$
\hat{\beta}_{\text{L2}} = \arg\min_\beta \frac{1}{2n}\Vert y - X\beta\Vert ^2 + \lambda \Vert \beta\Vert _2^2
$$

For the L1 problem, use your ISTA implementation (with appropriate modifications for the regression setting). For the L2 problem, use the closed-form solution $\hat{\beta}\_{\text{L2}} = (X^\top X + 2n\lambda I)^{-1} X^\top y$.

Use $\lambda \in \lbrace 0.001, 0.01, 0.05, 0.1, 0.5, 1.0\rbrace$.

### Part (c): Compare

For each $\lambda$, report:
1. Number of coefficients with $|\hat{\beta}\_j| > 0.01$ (approximately nonzero)
2. Test MSE on a held-out test set (generate 50 test samples from the same model)
3. $\Vert \hat{\beta} - \beta\_{\text{true}}\Vert \_2$ (parameter estimation error)

Create two plots:
1. **Regularization paths:** Plot $\hat{\beta}\_j$ vs. $\lambda$ for all 50 coefficients (one line per coefficient). Do this for both L1 and L2. The L1 plot should show coefficients "entering" the model one by one as $\lambda$ decreases.
2. **Sparsity:** Plot number of nonzero coefficients vs. $\lambda$ for both L1 and L2.

---

## Problem 6: Dictionary Learning Implementation (Implementation)

Implement a complete dictionary learning pipeline and apply it to MNIST.

### Part (a): Implementation

Write a `DictionaryLearning` class with:
- `__init__(self, d_x, d_z, lambda_)`: Initialize the dictionary randomly with normalized columns.
- `sparse_code(self, X)`: Given a batch of inputs $X$, return the sparse codes $Z$ using ISTA.
- `update_dictionary(self, X, Z, lr)`: Update the dictionary using the gradient: $\nabla\_D = -(X - DZ)Z^\top$.
- `fit(self, X, num_epochs, lr)`: Full training loop with alternating optimization.

### Part (b): MNIST Dictionary

Apply your dictionary learning to MNIST:
1. Take the first 10000 training images (flattened to 784-dim vectors, normalized to $[0,1]$)
2. Learn a dictionary with $d\_z = 512$ (overcomplete)
3. Use $\lambda$ chosen so that each image uses roughly 20-50 active dictionary atoms

Visualize:
1. A 16x32 grid showing all 512 dictionary atoms (reshaped to 28x28)
2. For 5 test digits, show the digit, its reconstruction, and the top-5 most active dictionary atoms

### Part (c): Comparison

Compare the dictionary atoms to:
1. The decoder weights of a vanilla autoencoder (from Week 6) with $d\_z = 512$
2. PCA components (from Week 5)

Which representation looks most interpretable? Which produces the sparsest codes?

---

## Problem 7: From Sparse Coding to Sparse Autoencoders (Conceptual)

This problem is mostly theoretical and asks you to connect the ideas of this week to the sparse autoencoders we will build next week.

### Part (a): Amortization Speed Test

Using the dictionary learned in Problem 6:
1. Time how long it takes to sparse-code 1000 MNIST images using ISTA (100 iterations each).
2. Now train a simple encoder network: $f\_\phi(x) = \text{ReLU}(Wx + b)$ with $W \in \mathbb{R}^{512 \times 784}$.
   - Train $f\_\phi$ to minimize $\sum\_n \Vert z\_n^{\text{ISTA}} - f\_\phi(x\_n)\Vert ^2$ where $z\_n^{\text{ISTA}}$ are the ISTA codes.
3. Time how long it takes to encode the same 1000 images using the trained encoder (single forward pass).

Report the speedup factor.

### Part (b): Amortization Gap

For the 1000 test images, compare:
1. The sparse coding objective $\frac{1}{2}\Vert x - Dz^{\text{ISTA}}\Vert ^2 + \lambda\Vert z^{\text{ISTA}}\Vert \_1$ (using ISTA codes)
2. The same objective using the amortized encoder's codes: $\frac{1}{2}\Vert x - Df\_\phi(x)\Vert ^2 + \lambda\Vert f\_\phi(x)\Vert \_1$

The gap between these two values is the **amortization gap** -- the price we pay for speed. How large is it?

### Part (c): Discussion

Answer the following questions (2-3 sentences each):

1. Why is the amortized encoder's code typically not as good as the ISTA code? What would make it better? (Think about encoder capacity and training data.)

2. A sparse autoencoder jointly trains the encoder and decoder (dictionary). In Problem 6, we trained the dictionary first and the encoder second. What advantage does joint training have?

3. In the mechanistic interpretability setting (Weeks 11-12), we will apply sparse autoencoders to the internal activations of a language model. The "signal" $x$ will be a residual stream vector of dimension $d\_x = 768$ (for GPT-2 small), and the dictionary will have $d\_z = 768 \times 16 = 12288$ atoms. Why is amortization especially important in this setting?

4. ISTA uses soft-thresholding, which is like a ReLU shifted by $\lambda$. Many sparse autoencoders use ReLU as their encoder activation. Is this a coincidence? Explain the connection.

---

## Submission Checklist

- [ ] Problem 1: Geometric argument for L1 sparsity with sketch, LASSO sweep over $\lambda$
- [ ] Problem 2: Soft-thresholding derivation (both approaches)
- [ ] Problem 3: ISTA and FISTA implementations, convergence plots, comparison
- [ ] Problem 4: Dictionary learning on natural image patches, Gabor-like filter visualization
- [ ] Problem 5: L1 vs. L2 comparison, regularization paths, sparsity plots
- [ ] Problem 6: Dictionary learning on MNIST, dictionary visualization
- [ ] Problem 7: Amortization speed comparison, gap analysis, conceptual questions

All code should be in Python using PyTorch (and optionally NumPy/scikit-learn for comparison). Submit as a Jupyter notebook or Python scripts with clearly labeled outputs.
