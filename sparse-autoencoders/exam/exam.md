---
title: "CS 371: Sparse Autoencoders and Neural Feature Extraction"
---

# CS 371: Sparse Autoencoders and Neural Feature Extraction

# Final Examination

**Duration:** 3 hours
**Total marks:** 120
**Instructions:**
- This exam has 6 sections. Attempt ALL questions.
- Marks for each question are indicated in brackets.
- You may use a single A4 sheet of handwritten notes (both sides).
- Show all working for derivation questions. Correct answers without justification receive no credit.
- For code-reading questions, you may assume standard PyTorch semantics.
- Where you are asked to "explain," a precise 2-4 sentence answer is expected unless otherwise stated.

---

## Section 1: Foundations (20 marks)

*Covers Weeks 1-4: Linear algebra, optimization, probability, neural networks.*

### Question 1.1 [6 marks]

Let $\mathbf{X} \in \mathbb{R}^{n \times d}$ be a data matrix where each row is a data point (zero-mean). The sample covariance matrix is $\mathbf{C} = \frac{1}{n}\mathbf{X}^\top \mathbf{X}$.

**(a)** [2 marks] What do the eigenvectors and eigenvalues of $\mathbf{C}$ represent geometrically?

**(b)** [4 marks] Prove that the top-$k$ eigenvectors of $\mathbf{C}$ solve the following optimization problem:

$$
\max_{\mathbf{W} \in \mathbb{R}^{d \times k}, \; \mathbf{W}^\top \mathbf{W} = \mathbf{I}} \text{tr}(\mathbf{W}^\top \mathbf{C} \mathbf{W})
$$

*Hint: Use the eigendecomposition of $\mathbf{C}$ and the constraint that $\mathbf{W}$ has orthonormal columns.*

### Question 1.2 [6 marks]

Consider gradient descent on a function $f(\mathbf{w})$ with update rule $\mathbf{w}\_{t+1} = \mathbf{w}\_t - \eta \nabla f(\mathbf{w}\_t)$.

**(a)** [3 marks] Suppose $f$ is $L$-smooth (i.e., $\Vert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\Vert \leq L \Vert \mathbf{x} - \mathbf{y}\Vert$ for all $\mathbf{x}, \mathbf{y}$). What is the maximum learning rate $\eta$ that guarantees $f(\mathbf{w}\_{t+1}) \leq f(\mathbf{w}\_t)$ at every step? Justify your answer in 2-3 sentences.

**(b)** [3 marks] Adam uses adaptive per-parameter learning rates. Explain the role of the first moment estimate $\mathbf{m}\_t$ and the second moment estimate $\mathbf{v}\_t$. Why does Adam often converge faster than vanilla SGD on loss surfaces with different curvatures along different dimensions?

### Question 1.3 [8 marks]

**(a)** [3 marks] Define the KL divergence $D\_{\text{KL}}(p \Vert q)$ for two continuous distributions $p$ and $q$. State two properties of KL divergence (with brief justification for each).

**(b)** [5 marks] Given a dataset $\lbrace x\_1, \ldots, x\_n\rbrace$ drawn i.i.d. from an unknown distribution $p^*$, and a parametric family $p\_\theta$, show that maximizing the log-likelihood $\sum\_{i=1}^n \log p\_\theta(x\_i)$ is equivalent to minimizing $D\_{\text{KL}}(\hat{p}\_{\text{data}} \Vert p\_\theta)$, where $\hat{p}\_{\text{data}}$ is the empirical distribution.

---

## Section 2: Autoencoders and Representation Learning (20 marks)

*Covers Weeks 5-8: PCA, autoencoders, VAEs.*

### Question 2.1 [8 marks]

Consider a linear autoencoder with encoder $\mathbf{z} = \mathbf{W}\_e \mathbf{x}$ and decoder $\hat{\mathbf{x}} = \mathbf{W}\_d \mathbf{z}$, where $\mathbf{x} \in \mathbb{R}^d$, $\mathbf{z} \in \mathbb{R}^k$ ($k < d$), and the reconstruction loss is $\mathcal{L} = \mathbb{E}[\Vert \mathbf{x} - \mathbf{W}\_d \mathbf{W}\_e \mathbf{x}\Vert ^2]$ (data is zero-mean).

**(a)** [5 marks] Prove that at the global optimum of $\mathcal{L}$, the columns of $\mathbf{W}\_d$ span the same subspace as the top $k$ eigenvectors of the data covariance matrix $\mathbf{C}$. You may assume $\mathbf{C}$ has distinct eigenvalues.

*Hint: Consider what $\mathbf{P} = \mathbf{W}\_d \mathbf{W}\_e$ must be at the optimum.*

**(b)** [3 marks] Why does this equivalence break down for nonlinear autoencoders? Give a concrete example of a 2D dataset where a nonlinear autoencoder with a 1D bottleneck would outperform PCA.

### Question 2.2 [12 marks]

This question concerns variational autoencoders (VAEs).

**(a)** [6 marks] Starting from the marginal log-likelihood $\log p\_\theta(\mathbf{x})$, derive the Evidence Lower Bound (ELBO):

$$
\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \Vert p(\mathbf{z}))
$$

Show each step clearly.

**(b)** [3 marks] Explain the reparameterization trick. Why is it necessary? What would go wrong if we tried to compute $\nabla\_\phi \mathbb{E}\_{q\_\phi(\mathbf{z}|\mathbf{x})}[\log p\_\theta(\mathbf{x}|\mathbf{z})]$ by sampling directly from $q\_\phi$?

**(c)** [3 marks] In a $\beta$-VAE, the objective is:

$$
\mathcal{L}_{\beta} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \Vert p(\mathbf{z}))
$$

What happens when $\beta \gg 1$? What happens when $\beta \to 0$? Explain the tradeoff in terms of reconstruction quality and latent space structure.

---

## Section 3: Sparsity and Sparse Autoencoders (25 marks)

*Covers Weeks 9-10: Sparse coding, LASSO, ISTA, sparse autoencoders.*

### Question 3.1 [8 marks]

Consider the LASSO problem: $\min\_{\mathbf{z}} \frac{1}{2}\Vert \mathbf{x} - \mathbf{D}\mathbf{z}\Vert \_2^2 + \lambda \Vert \mathbf{z}\Vert \_1$, where $\mathbf{D} \in \mathbb{R}^{n \times m}$ is a dictionary with $m > n$.

**(a)** [5 marks] For the special case $\mathbf{D} = \mathbf{I}$ (the identity), derive the closed-form solution. Show that the optimal $z\_i^*$ is given by the **soft-thresholding operator**:

$$
z_i^* = \text{sign}(x_i) \max(|x_i| - \lambda, 0)
$$

**(b)** [3 marks] Sketch the soft-thresholding operator as a function of $x\_i$ for a fixed $\lambda > 0$. How does this operator differ from hard thresholding (i.e., setting $z\_i = 0$ if $|x\_i| < \lambda$, else $z\_i = x\_i$)? Which introduces shrinkage bias, and why?

### Question 3.2 [7 marks]

The ISTA (Iterative Shrinkage-Thresholding Algorithm) solves the general LASSO problem by iterating:

$$
\mathbf{z}^{(t+1)} = S_{\lambda/L}\left(\mathbf{z}^{(t)} + \frac{1}{L}\mathbf{D}^\top(\mathbf{x} - \mathbf{D}\mathbf{z}^{(t)})\right)
$$

where $S\_\tau$ applies soft-thresholding with threshold $\tau$ and $L$ is the Lipschitz constant of the gradient.

**(a)** [4 marks] Explain the structure of one ISTA iteration. Identify the gradient step and the proximal step. Why can't we simply use gradient descent on $\frac{1}{2}\Vert \mathbf{x} - \mathbf{D}\mathbf{z}\Vert \_2^2 + \lambda \Vert \mathbf{z}\Vert \_1$ directly?

**(b)** [3 marks] A sparse autoencoder with ReLU activation and L1 penalty can be seen as performing *amortized* sparse coding. In what sense does a single forward pass of the SAE encoder approximate the iterative ISTA procedure? What does the SAE sacrifice by using a single step?

### Question 3.3 [10 marks]

Read the following PyTorch code for training a sparse autoencoder:

```python
class SAE(nn.Module):
    def __init__(self, d_model, n_features):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

# Training loop (excerpt)
sae = SAE(d_model=512, n_features=512 * 8)
optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)

for batch in dataloader:
    x = batch  # shape: (B, 512)
    x_hat, z = sae(x)

    mse_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
    l1_loss = z.abs().sum(dim=-1).mean()
    loss = mse_loss + 1e-3 * l1_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**(a)** [2 marks] What is the expansion factor of this SAE? Is it undercomplete or overcomplete?

**(b)** [3 marks] Identify **two** potential issues with this training code that could lead to poor SAE quality. For each, explain the problem and suggest a fix.

**(c)** [2 marks] After training, you observe that 60% of the features have zero activation across the entire test set. Diagnose the likely cause and propose two different strategies to address it.

**(d)** [3 marks] You decide to evaluate this SAE by computing the CE loss recovered metric. Describe precisely how you would compute this, including what the baseline and reference values are. If the CE loss recovered is 0.85, what does this tell you?

---

## Section 4: Mechanistic Interpretability (25 marks)

*Covers Weeks 11-13: Superposition, polysemanticity, SAEs for LMs, advanced architectures.*

### Question 4.1 [8 marks]

**(a)** [4 marks] Explain the **superposition hypothesis** using a concrete example. Your answer should include: what superposition is, why neural networks might use it, and why it makes interpretability difficult.

**(b)** [4 marks] In Elhage et al.'s toy model of superposition, a network with $n$ neurons represents $m > n$ features. The model is:

$$
\hat{\mathbf{x}} = \mathbf{W}^\top \text{ReLU}(\mathbf{W} \mathbf{x})
$$

where $\mathbf{W} \in \mathbb{R}^{n \times m}$, $\mathbf{x} \in \mathbb{R}^m$ is a sparse input, and the loss is $\Vert \mathbf{x} - \hat{\mathbf{x}}\Vert ^2$.

Explain why, when features are sparse enough, the optimal $\mathbf{W}$ will represent more than $n$ features. What geometric structure do the columns of $\mathbf{W}$ form, and why? What role does feature sparsity play in determining how many features can be superimposed?

### Question 4.2 [7 marks]

You have trained an SAE on the residual stream of layer 8 of GPT-2 Small (hidden dimension 768, SAE dictionary size 768 * 16 = 12288). You examine a particular feature $f\_{3421}$ and find that its top-10 activating examples all contain references to the Golden Gate Bridge.

**(a)** [3 marks] Describe a **causal intervention** experiment you would perform to test whether $f\_{3421}$ causally influences the model's behavior related to the Golden Gate Bridge. Be precise about what you would clamp, what you would measure, and what result would support a causal role.

**(b)** [4 marks] A colleague argues: "The feature activates on Golden Gate Bridge, but that doesn't mean the model *uses* this feature for anything. It could be a correlate, not a cause." Design an experiment that would distinguish between these two possibilities. What would you measure, and what pattern of results would support each interpretation?

### Question 4.3 [10 marks]

This question compares advanced SAE architectures.

**(a)** [4 marks] Explain **shrinkage bias** in L1-penalized SAEs. Then explain, with equations, how the **Gated SAE** architecture eliminates shrinkage bias. Your answer should clearly show how the gating path and magnitude path work together.

**(b)** [3 marks] In a **TopK SAE**, the sparsity constraint is $\Vert \mathbf{z}\Vert \_0 = K$ exactly. Explain how gradients are computed through the TopK operation. What is the key assumption that makes the straight-through estimator reasonable?

**(c)** [3 marks] A **JumpReLU SAE** uses learned per-feature thresholds $\theta\_i$. Give one advantage of learned thresholds over both fixed-threshold ReLU (as in vanilla SAEs) and fixed-K sparsity (as in TopK SAEs). In what situation would per-feature thresholds be most beneficial?

---

## Section 5: Code Analysis and Design (15 marks)

*Covers all weeks. Tests practical understanding.*

### Question 5.1 [8 marks]

Consider the following PyTorch implementation of a TopK SAE:

```python
class TopKSAE(nn.Module):
    def __init__(self, d_model, n_features, k):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_model, n_features) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.randn(n_features, d_model) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.k = k

    def forward(self, x):
        h = x @ self.W_enc + self.b_enc             # (B, n_features)
        topk_vals, topk_idx = torch.topk(h, self.k)  # (B, k)
        z = torch.zeros_like(h)
        z.scatter_(1, topk_idx, topk_vals)            # (B, n_features)
        x_hat = z @ self.W_dec + self.b_dec           # (B, d_model)
        return x_hat, z, h
```

**(a)** [2 marks] There is a subtle but important bug in this implementation that breaks gradient flow. Identify it and explain why it prevents learning.

*Hint: Think about what `torch.zeros_like` and `scatter_` do to the computational graph.*

**(b)** [3 marks] Write a corrected version of the `forward` method that properly allows gradients to flow through the top-K selection. You may use any standard PyTorch operations.

**(c)** [3 marks] You want to add decoder column normalization (constraining each column of $\mathbf{W}\_d$ to unit norm). Should this be done inside `forward()`, as a post-processing step after `optimizer.step()`, or as a constraint in the loss function? Justify your choice and explain the tradeoffs.

### Question 5.2 [7 marks]

You are tasked with designing an SAE training pipeline for a new language model. The model has a hidden dimension of 2048, and you have a budget of 1 billion tokens of training data.

**(a)** [3 marks] What dictionary size would you start with? Justify your choice using the scaling laws discussed in the course. What expansion factor would you use, and why?

**(b)** [4 marks] Describe your complete evaluation protocol. For each metric you would compute, explain: (i) what it measures, (ii) how to compute it, and (iii) what a "good" value looks like. Include at least three different metrics.

---

## Section 6: Integration and Synthesis (15 marks)

*Spans all weeks. Requires connecting ideas across the course.*

### Question 6.1 [8 marks]

Trace the intellectual arc from PCA to sparse autoencoders for mechanistic interpretability. Your answer should address:

**(a)** [2 marks] What problem does PCA solve, and what are its key limitations for understanding neural network representations?

**(b)** [2 marks] How do (nonlinear) autoencoders generalize PCA? What new capabilities do they provide, and what new problems do they introduce?

**(c)** [2 marks] Why is sparsity important? Connect the concept of sparsity to both the statistics of natural data (Olshausen & Field) and the superposition hypothesis.

**(d)** [2 marks] How do sparse autoencoders combine the ideas from (a)-(c) to address the interpretability problem? What is the key insight that connects dictionary learning to mechanistic interpretability?

### Question 6.2 [7 marks]

**Open-ended question.** The current SAE paradigm assumes that neural network features are best described as *directions* in activation space, and that a *linear* dictionary suffices to decompose them.

**(a)** [3 marks] Present one argument *for* the linear representation hypothesis (features as directions). What evidence supports it?

**(b)** [2 marks] Present one argument *against* it. What phenomena might it fail to capture?

**(c)** [2 marks] Propose an alternative approach or modification to SAEs that could address the limitation you identified in (b). You do not need to work out the details — a clear description of the idea and its motivation is sufficient.

---

**END OF EXAMINATION**

*Marks: Section 1 (20) + Section 2 (20) + Section 3 (25) + Section 4 (25) + Section 5 (15) + Section 6 (15) = 120 total*
