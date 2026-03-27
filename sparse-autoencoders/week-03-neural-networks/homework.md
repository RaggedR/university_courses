# Week 3: Homework — Neural Networks from First Principles

**Instructions:** This problem set has 7 problems mixing theory (pen-and-paper proofs and derivations) and implementation (Python/NumPy and PyTorch). Problems 3 and 4 are substantial implementation exercises — start early. Show your work for theoretical problems.

**Estimated time:** 5-7 hours

---

## Problem 1: XOR Impossibility (Theory)

**(a)** Prove that the XOR function cannot be computed by a single perceptron (a single neuron with a step activation function). Use the approach from the notes: set up the four inequalities corresponding to the four input-output pairs and derive a contradiction.

**(b)** Now prove a more general result: a single perceptron with $d$ inputs computes a function of the form $f(\mathbf{x}) = \mathbf{1}[\mathbf{w}^T\mathbf{x} + b > 0]$, which classifies points based on which side of a hyperplane they fall on. Argue (you don't need a fully formal proof) that any function computed by a single perceptron must be a **linearly separable** function — there must exist a hyperplane separating the positive from the negative examples.

**(c)** Construct a two-layer network (with step activations) that computes XOR. Specify the weights and biases for each neuron explicitly. Verify your network produces the correct output for all four inputs.

**(d)** (Bonus) Can you construct a network that computes XOR using exactly 2 hidden neurons and 1 output neuron, all with ReLU activations instead of step functions? If yes, give the weights. If no, explain why not and state the minimum number of hidden neurons needed.

---

## Problem 2: Forward Pass by Hand (Theory)

Consider a neural network with the following architecture:
- Input: $\mathbf{x} \in \mathbb{R}^2$
- Hidden layer 1: 3 neurons, ReLU activation
- Hidden layer 2: 2 neurons, ReLU activation
- Output: 1 neuron, sigmoid activation

The parameters are:

$$
W_1 = \begin{pmatrix} 1 & -1 \\\\ -1 & 1 \\\\ 0.5 & 0.5 \end{pmatrix}, \quad \mathbf{b}_1 = \begin{pmatrix} 0 \\\\ 0 \\\\ -0.5 \end{pmatrix}
$$

$$
W_2 = \begin{pmatrix} 1 & -1 & 0.5 \\\\ 0.5 & 1 & -0.5 \end{pmatrix}, \quad \mathbf{b}_2 = \begin{pmatrix} 0 \\\\ 0 \end{pmatrix}
$$

$$
W_3 = \begin{pmatrix} 1 & -1 \end{pmatrix}, \quad b_3 = 0
$$

**(a)** Compute the forward pass for input $\mathbf{x} = (1, 0)^T$. Show each intermediate result:
- Pre-activation $\mathbf{z}\_1 = W\_1\mathbf{x} + \mathbf{b}\_1$ and post-activation $\mathbf{h}\_1 = \text{ReLU}(\mathbf{z}\_1)$
- Pre-activation $\mathbf{z}\_2 = W\_2\mathbf{h}\_1 + \mathbf{b}\_2$ and post-activation $\mathbf{h}\_2 = \text{ReLU}(\mathbf{z}\_2)$
- Pre-activation $z\_3 = W\_3\mathbf{h}\_2 + b\_3$ and output $y = \sigma(z\_3)$

**(b)** Repeat for input $\mathbf{x} = (0, 1)^T$.

**(c)** Repeat for input $\mathbf{x} = (1, 1)^T$.

**(d)** How many neurons are "dead" (output exactly 0) for each input? Which neurons are consistently dead across all three inputs? What does this tell you about the effective capacity of the network for these inputs?

**(e)** Count the total number of parameters in this network. Then count how many parameters actually influence the output for input $(1, 0)^T$ (i.e., parameters multiplied by a non-zero value during the forward pass). What fraction of parameters are "active"?

---

## Problem 3: MLP from Scratch in NumPy (Implementation)

Implement a two-layer MLP (one hidden layer) **using only NumPy** — no PyTorch, no scikit-learn for the model itself. You'll train it to classify the "two moons" dataset.

**(a)** Generate the dataset:
```python
from sklearn.datasets import make_moons
import numpy as np

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
# Split into train/test
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]
```

**(b)** Implement the following functions:

```python
def sigmoid(z):
    """Sigmoid activation function."""
    # Your implementation

def relu(z):
    """ReLU activation function."""
    # Your implementation

def relu_derivative(z):
    """Derivative of ReLU (needed for backprop)."""
    # Your implementation

def initialize_parameters(input_dim, hidden_dim, output_dim):
    """Initialize weights and biases.
    Use He initialization for weights: W ~ N(0, 2/fan_in)
    Initialize biases to zero.
    Returns: dict with keys 'W1', 'b1', 'W2', 'b2'
    """
    # Your implementation

def forward(X, params):
    """Forward pass.
    Args:
        X: input data, shape (N, input_dim)
        params: dict with W1, b1, W2, b2
    Returns:
        y_hat: predictions, shape (N, 1)
        cache: dict with intermediate values needed for backprop
               (z1, h1, z2, h2)
    """
    # Your implementation

def compute_loss(y_hat, y_true):
    """Binary cross-entropy loss.
    Returns: scalar loss value
    """
    # Your implementation

def backward(y_hat, y_true, cache, params):
    """Backward pass (backpropagation).
    Compute gradients of the loss with respect to all parameters.
    Returns: dict with keys 'dW1', 'db1', 'dW2', 'db2'
    """
    # Your implementation

def update_parameters(params, grads, learning_rate):
    """Gradient descent update."""
    # Your implementation

def train(X, y, hidden_dim=32, learning_rate=0.1, num_epochs=1000):
    """Full training loop.
    Returns: trained params, list of losses
    """
    # Your implementation
```

**(c)** Train your network with `hidden_dim=32` and `learning_rate=0.1` for 1000 epochs. Plot:
1. The training loss vs. epoch number
2. The decision boundary (create a mesh grid over the input space, run forward pass on each point, color by predicted class)

**(d)** Report the training and test accuracy. Try different hidden dimensions (4, 16, 32, 64, 128) and report how accuracy changes.

**(e)** Note: You'll need to implement backpropagation for the backward pass. Here are the key gradient formulas for binary cross-entropy with sigmoid output:

For the output layer (sigmoid + BCE):
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}_2} = \hat{\mathbf{y}} - \mathbf{y}
$$

This remarkably simple formula (prediction minus truth) is one reason cross-entropy pairs so well with sigmoid. Derive the gradients for $W\_2$, $\mathbf{b}\_2$, and then propagate back through the hidden layer to get gradients for $W\_1$, $\mathbf{b}\_1$.

---

## Problem 4: The Same Network in PyTorch (Implementation)

Now reimplement the same two-layer MLP using PyTorch.

**(a)** Define the model:
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Define layers

    def forward(self, x):
        # Define forward pass
        pass
```

**(b)** Write a training loop using `nn.BCELoss()` and `torch.optim.Adam`. Train on the same data with `hidden_dim=32`.

**(c)** Compare:
1. **Code complexity.** How many lines of code is the PyTorch version vs. the NumPy version?
2. **Performance.** Compare training time, final loss, and test accuracy.
3. **Decision boundaries.** Plot the decision boundaries from both models side by side. Are they similar?

**(d)** Verify that the gradients computed by PyTorch match yours from Problem 3. After one forward pass on a small batch (say, the first 5 training examples):
1. Run `loss.backward()` in PyTorch
2. Manually compute gradients using your NumPy code on the same data
3. Compare the gradient values (they should match up to floating-point precision)

**Hint:** To make this comparison work, initialize the PyTorch model with the same weights as the NumPy model. You can do this by setting `model.layer1.weight.data = torch.FloatTensor(params['W1'])`, etc.

---

## Problem 5: Activation Function Exploration (Implementation)

**(a)** Implement a flexible MLP in PyTorch that lets you swap activation functions:
```python
class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        super().__init__()
        # Support: 'relu', 'sigmoid', 'tanh', 'leaky_relu', 'gelu'
        # Use the same architecture: 2 hidden layers of hidden_dim neurons
        pass
```

**(b)** Train this model on the two moons dataset with each of the five activation functions. Use the same hyperparameters (learning rate, epochs, hidden dimension) for all. For each activation function, record:
1. Final training loss
2. Final test accuracy
3. Training time (epochs to reach 95% accuracy, or total epochs if never reached)

**(c)** For each trained model, plot the decision boundary. Arrange all five plots in a row for easy comparison. Are there visible differences?

**(d)** Create a deeper version (5 hidden layers of 32 neurons each) and repeat the experiment. Now which activation functions perform best? Does any activation function fail to train at all? If so, explain why in terms of the vanishing gradient problem.

**(e)** For the deep network experiment, plot the distribution of activations (histogram of the output values) at each layer for each activation function. This visualization shows the "health" of gradient flow: are activations saturating? Are they collapsing to zero?

---

## Problem 6: Universal Approximation (Conceptual)

This problem builds intuition for what the universal approximation theorem means and doesn't mean.

**(a)** Consider the 1D function $f(x) = \sin(2\pi x)$ on $[0, 1]$. Explain how a single-hidden-layer network with ReLU activations could approximate this function. (Hint: a ReLU neuron $\max(0, wx + b)$ computes a ramp. How can you combine ramps to approximate a sine curve?)

**(b)** Roughly how many ReLU neurons would you need in a single hidden layer to approximate $\sin(2\pi x)$ to within error $\epsilon = 0.01$ on $[0, 1]$? Give a rough estimate with reasoning, not a precise bound. (Hint: think about how many piecewise-linear segments you need.)

**(c)** Now consider the 2D function $f(x\_1, x\_2) = \sin(2\pi x\_1)\sin(2\pi x\_2)$ on $[0,1]^2$. How does the number of required neurons scale compared to the 1D case? This is a taste of the **curse of dimensionality** — explain in your own words what this means for neural network width.

**(d)** The universal approximation theorem says a single wide layer suffices. But in practice, deep narrow networks often work better than shallow wide ones. Give an intuitive explanation using the example of learning to recognize a handwritten digit. What kind of hierarchical features might a deep network learn that a single wide layer would have to represent "flatly"?

---

## Problem 7: From Classification to Reconstruction (Conceptual Bridge to Week 6)

This problem previews the autoencoder idea.

**(a)** In a classifier, the network maps input $\mathbf{x} \in \mathbb{R}^d$ to a label $y \in \lbrace 1, \ldots, K\rbrace$, compressing the input down to one of $K$ categories. The hidden layers extract features useful for classification. Explain why features useful for classification might NOT be useful for reconstruction (recovering the original input). Give a concrete example.

**(b)** Imagine a network architecture that looks like this:

$$
\mathbf{x} \in \mathbb{R}^{784} \xrightarrow{} \mathbf{h}_1 \in \mathbb{R}^{256} \xrightarrow{} \mathbf{h}_2 \in \mathbb{R}^{32} \xrightarrow{} \mathbf{h}_3 \in \mathbb{R}^{256} \xrightarrow{} \hat{\mathbf{x}} \in \mathbb{R}^{784}
$$

This is an autoencoder. The loss is $\Vert \mathbf{x} - \hat{\mathbf{x}}\Vert ^2$. In this architecture, what is the "bottleneck"? Why does it force the network to learn a compressed representation? What information must be preserved in $\mathbf{h}\_2$, and what information can be discarded?

**(c)** In the autoencoder above, $\mathbf{h}\_2 \in \mathbb{R}^{32}$ is the compressed representation of a $784$-dimensional input (e.g., a $28 \times 28$ image). The **encoder** is the first half of the network ($\mathbf{x} \to \mathbf{h}\_2$) and the **decoder** is the second half ($\mathbf{h}\_2 \to \hat{\mathbf{x}}$).

Now suppose we remove all activation functions. From Week 1, what theorem tells us what the optimal linear encoder+decoder will learn? What specific representation will $\mathbf{h}\_2$ converge to?

**(d)** This is why we need nonlinear activations in autoencoders. Explain, in 2-3 sentences, what advantage a nonlinear autoencoder has over a linear one. Use the concept of a "manifold" if you can: real data (like images of faces) doesn't fill the full $784$-dimensional space uniformly. It lies on or near a lower-dimensional curved surface. What can a nonlinear autoencoder capture that a linear one (i.e., PCA/SVD) cannot?

---

## Submission Checklist

- [ ] Problem 1: XOR impossibility proof, two-layer XOR construction
- [ ] Problem 2: Forward pass computations (all four inputs), dead neuron analysis
- [ ] Problem 3: NumPy MLP — complete implementation, training curve, decision boundary, accuracy report
- [ ] Problem 4: PyTorch MLP — implementation, comparison with NumPy version, gradient verification
- [ ] Problem 5: Activation function experiments with plots and analysis
- [ ] Problem 6: Universal approximation intuition (written answers)
- [ ] Problem 7: Autoencoder preview (written answers connecting to Weeks 1 and 6)
