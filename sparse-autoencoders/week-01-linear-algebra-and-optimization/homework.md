# Week 1: Homework — Linear Algebra and Optimization

**Instructions:** This problem set has 7 problems mixing theory (pen-and-paper derivations) and implementation (Python/NumPy). For implementation problems, write your solutions in a Jupyter notebook or `.py` file. Show your work for theoretical problems — a correct answer with no justification receives no credit.

**Estimated time:** 4-6 hours

---

## Problem 1: Eigendecomposition by Hand

Consider the matrix:

$$A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}$$

**(a)** Find the eigenvalues of $A$ by solving the characteristic equation $\det(A - \lambda I) = 0$.

**(b)** For each eigenvalue, find the corresponding eigenvector.

**(c)** Write the eigendecomposition $A = PDP^{-1}$ explicitly. Verify your answer by computing $PDP^{-1}$ and checking it equals $A$.

**(d)** Use your eigendecomposition to compute $A^5$ without multiplying $A$ by itself five times. (Hint: $A^5 = PD^5P^{-1}$, and $D^5$ is trivial to compute.)

---

## Problem 2: The Spectral Theorem in Action

Consider the symmetric matrix:

$$S = \begin{pmatrix} 5 & 2 \\ 2 & 2 \end{pmatrix}$$

**(a)** Find the eigenvalues and eigenvectors of $S$. Verify that the eigenvectors are orthogonal (as the spectral theorem guarantees).

**(b)** Normalize the eigenvectors to unit length and write the spectral decomposition $S = Q\Lambda Q^T$ explicitly.

**(c)** **Proof.** Prove the following: if $A$ is a real symmetric matrix and $\mathbf{v}_1, \mathbf{v}_2$ are eigenvectors corresponding to *distinct* eigenvalues $\lambda_1 \neq \lambda_2$, then $\mathbf{v}_1 \perp \mathbf{v}_2$.

*Hint:* Start with $\langle A\mathbf{v}_1, \mathbf{v}_2 \rangle$ and compute it two ways — once using $A\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$, and once by exploiting the symmetry of $A$.

---

## Problem 3: SVD and Low-Rank Approximation (Theory)

Consider the matrix:

$$B = \begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}$$

**(a)** Compute $B^TB$ and $BB^T$.

**(b)** Find the singular values of $B$, and the right singular vectors (from the eigenvectors of $B^TB$) and left singular vectors (from the eigenvectors of $BB^T$).

**(c)** Write out the full SVD $B = U\Sigma V^T$. Verify by multiplying $U\Sigma V^T$.

**(d)** What is the best rank-1 approximation $B_1$ of $B$? What is the Frobenius norm of the error $\|B - B_1\|_F$?

---

## Problem 4: Image Compression with SVD (Implementation)

In this problem, you'll implement SVD-based image compression from scratch using NumPy.

**(a)** Load a grayscale image as a NumPy array. You can use any image you like, or generate a test image:
```python
import numpy as np
from PIL import Image
# Option 1: Load your own image
# img = np.array(Image.open('photo.jpg').convert('L'), dtype=float)
# Option 2: Generate a test image (gradient with some features)
x = np.linspace(-3, 3, 256)
y = np.linspace(-3, 3, 256)
X, Y = np.meshgrid(x, y)
img = np.sin(X) * np.cos(Y) + 0.5 * np.exp(-(X**2 + Y**2))
```

**(b)** Compute the SVD of the image matrix using `np.linalg.svd`. Plot the singular values on a log scale. What do you observe about how quickly they decay?

**(c)** Write a function `compress(image, k)` that returns the rank-$k$ approximation of the image. Reconstruct the image using $k = 1, 5, 10, 20, 50, 100$. Display the reconstructions side by side (use `matplotlib`).

**(d)** For each value of $k$, compute:
- The compression ratio: $\frac{mn}{k(m + n + 1)}$
- The relative error: $\frac{\|A - A_k\|_F}{\|A\|_F}$

Plot both as functions of $k$. At what rank does the image become "good enough" (say, < 5% relative error)?

**(e)** If you used a real photograph, you should observe that the singular values decay *much* faster for structured images (faces, buildings) than for random noise. Explain intuitively why this is the case. (What does it mean about the "true dimensionality" of the image?)

---

## Problem 5: Gradient Descent from Scratch (Implementation)

Implement gradient descent in Python/NumPy to minimize the Rosenbrock function:

$$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$$

This is a classic test function for optimization algorithms. The global minimum is at $(1, 1)$ where $f = 0$, but the function has a narrow curved valley that makes optimization challenging.

**(a)** Write a function `rosenbrock(x, y)` that evaluates $f$, and a function `rosenbrock_grad(x, y)` that returns the gradient $\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)$. Derive the gradient by hand first, then implement it.

**(b)** Implement gradient descent:
```python
def gradient_descent(grad_fn, x0, y0, learning_rate, num_steps):
    """
    Returns: list of (x, y, f(x,y)) at each step
    """
    # Your implementation here
```

**(c)** Run gradient descent from the starting point $(-1, 1)$ with several learning rates: $\eta = 0.001, 0.0005, 0.0001$. For each:
- Plot the trajectory $(x_t, y_t)$ overlaid on a contour plot of $f$
- Plot $f(x_t, y_t)$ vs. step number

What happens with $\eta = 0.01$? (Try it and explain.)

**(d)** The Rosenbrock function is a good illustration of why vanilla gradient descent can be slow. In 2-3 sentences, explain what geometric property of the Rosenbrock function's level sets causes the difficulty, and connect this to the condition number of the Hessian (you don't need to compute the Hessian, just reason about it).

---

## Problem 6: $L_1$ vs $L_2$ Geometry (Theory + Implementation)

This problem makes the sparsity argument from the notes concrete.

**(a)** Consider the optimization problem: minimize $f(\mathbf{x}) = \|\mathbf{x} - \mathbf{b}\|_2^2$ subject to $\|\mathbf{x}\|_1 \leq 1$, where $\mathbf{b} = (0.8, 0.6)^T$.

Solve this graphically: sketch the $L_1$ unit ball and the level sets of $f$ (circles centered at $\mathbf{b}$), and identify the point where they first touch. What is the solution?

**(b)** Now solve the same problem with an $L_2$ constraint: minimize $\|\mathbf{x} - \mathbf{b}\|_2^2$ subject to $\|\mathbf{x}\|_2 \leq 1$.

This has a closed-form solution. Find it. (Hint: the solution is $\mathbf{b}/\|\mathbf{b}\|_2$ when $\|\mathbf{b}\|_2 > 1$.) Is the solution sparse?

**(c) (Implementation)** Write a Python script that:
1. Generates 1000 random vectors $\mathbf{b} \in \mathbb{R}^{10}$, each with entries drawn from $\mathcal{N}(0, 1)$.
2. For each $\mathbf{b}$, solves both the $L_1$-constrained and $L_2$-constrained problems (you may use `scipy.optimize.minimize` with appropriate constraints, or solve the $L_2$ version analytically).
3. Counts the average number of zero entries (entries with $|x_i| < 10^{-6}$) in each solution.

Report the average sparsity for $L_1$ vs $L_2$. Does the result match the geometric argument from the notes?

---

## Problem 7: Connecting the Dots

This is a conceptual problem — no computation required, but think carefully.

**(a)** A dataset of $N$ images, each with $d$ pixels, can be stored as an $N \times d$ matrix $X$. The SVD of $X$ gives $X = U\Sigma V^T$. Explain what the columns of $V$ represent, what the columns of $U$ represent, and what the singular values tell you about the dataset. (Connect this to the idea of "principal components" if you know it, or reason from the geometry.)

**(b)** Suppose you have a weight matrix $W \in \mathbb{R}^{100 \times 784}$ in a neural network that processes 28x28 images. The SVD reveals that only 15 singular values are significantly non-zero (the rest are $< 10^{-6}$). What does this tell you about what the layer is doing? Could you replace this layer with something simpler? What would you lose?

**(c)** We said that the $L_1$ penalty encourages sparsity by creating "corners" in the constraint set. In a sparse autoencoder, the sparsity penalty is applied to the *activations* (hidden layer outputs), not the *weights*. Explain in your own words why we want sparse activations rather than sparse weights, and what interpretive advantage this gives us. (Think about what "a few neurons activate for each input" means vs. "each neuron has few non-zero weights.")

**(d)** If a linear autoencoder with a $k$-dimensional bottleneck learns the same representation as rank-$k$ SVD, why do we bother with nonlinear autoencoders? What can they capture that SVD cannot? Give an intuitive example of data that lies on a nonlinear manifold where SVD would fail.

---

## Submission Checklist

- [ ] Problem 1: eigenvalues, eigenvectors, decomposition, and $A^5$
- [ ] Problem 2: eigenvalues, eigenvectors, spectral decomposition, and proof
- [ ] Problem 3: SVD computation and rank-1 approximation
- [ ] Problem 4: Jupyter notebook with image compression code and plots
- [ ] Problem 5: Jupyter notebook with gradient descent code, trajectories, and analysis
- [ ] Problem 6: graphical solution, closed-form solution, and sparsity experiment code
- [ ] Problem 7: written answers (a paragraph each)
