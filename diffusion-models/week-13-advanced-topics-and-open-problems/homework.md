---
title: "Week 13: Advanced Topics and Open Problems -- Homework"
---

# Week 13: Advanced Topics and Open Problems -- Homework

**Estimated time:** 12-15 hours
**Prerequisites:** All previous weeks, especially Weeks 5 (DDPM), 11 (flow matching), and 12 (distillation)

---

## Problem 1: Discrete Diffusion for Binary Sequences (Implementation)

Implement a simple discrete diffusion model for binary sequences, demonstrating the core ideas of D3PM.

### Part (a): Forward Process

Consider binary sequences $x \in \lbrace 0, 1\rbrace ^L$ of length $L = 16$.

Define an absorbing-state forward process where each bit independently transitions to a [MASK] state (represented as 2) with probability $\beta\_t$ at each step:

$$
q(x_t^{(i)} | x_{t-1}^{(i)}) = \begin{cases} 1 - \beta_t & \text{if } x_t^{(i)} = x_{t-1}^{(i)} \text{ and } x_{t-1}^{(i)} \neq [\text{MASK}] \\\\ \beta_t & \text{if } x_t^{(i)} = [\text{MASK}] \text{ and } x_{t-1}^{(i)} \neq [\text{MASK}] \\\\ 1 & \text{if } x_t^{(i)} = x_{t-1}^{(i)} = [\text{MASK}] \end{cases}
$$

Use a linear schedule $\beta\_t = t/T$ for $T = 100$ steps.

Implement the forward process. Starting from the sequence $x\_0 = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1]$, plot the sequence at $t = 0, 10, 25, 50, 75, 100$, visualizing which positions are 0, 1, or [MASK]. Compute the expected fraction of masked positions as a function of $t$.

### Part (b): Reverse Process Model

Define a data distribution: binary sequences encoding the 4-bit representations of even numbers (i.e., sequences of the form $[b\_3, b\_2, b\_1, 0, b\_3', b\_2', b\_1', 0, \ldots]$ where the last bit of each 4-bit block is 0).

Build a small transformer (or MLP) that takes a partially masked sequence $x\_t$ and time $t$ as input and predicts the clean sequence $\hat{x}\_0$:

```python
class DiscreteDenoisingModel(nn.Module):
    def __init__(self, seq_len=16, vocab_size=3, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.time_embed = ...
        # A few transformer layers or an MLP
        self.head = nn.Linear(hidden, 2)  # predict logits for {0, 1}
```

### Part (c): Training

Train the model using the discrete diffusion ELBO. For the absorbing-state case, the loss simplifies to:

$$
\mathcal{L} = \mathbb{E}_{t, x_0}\left[\sum_{i : x_t^{(i)} = [\text{MASK}]} -\log p_\theta(x_0^{(i)} | x_t, t)\right]
$$

This is a cross-entropy loss, summed only over masked positions -- exactly a masked language model loss!

Train for 5000-10000 steps. Plot the training loss.

### Part (d): Generation

Generate sequences by iteratively unmasking:

1. Start with all [MASK]: $x\_T = [\text{MASK}, \text{MASK}, \ldots, \text{MASK}]$
2. For $t = T, T-1, \ldots, 1$:
   - Predict $p\_\theta(x\_0 | x\_t, t)$ for all masked positions
   - For each masked position, sample: unmask with probability $1/(t)$, keep masked otherwise
   - For unmasked positions, sample from $p\_\theta(x\_0^{(i)} | x\_t, t)$

Generate 1000 sequences and compute:
1. The fraction of generated sequences that are valid (all 4-bit blocks represent even numbers)
2. The diversity of generated sequences (number of unique sequences)

Compare to the true data distribution.

---

## Problem 2: Score Distillation in 2D (Implementation)

Implement score distillation sampling to optimize a parameterized distribution using a pre-trained generative model.

### Part (a): The Diffusion Prior

Use a pre-trained flow matching model on the 8-Gaussians dataset (from Week 11) as your "diffusion prior." This model represents $p\_{\text{data}}$.

### Part (b): Optimizing a Single Point

Start with a learnable 2D point $\theta = (\theta\_1, \theta\_2)$, initialized at the origin. Implement the SDS gradient:

$$
\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon}\left[w(t) \left(v_\phi(x_t, t) - (x_1 - x_0)\right)\right]
$$

where $x\_0 \sim \mathcal{N}(0, I)$, $x\_1 = \theta$ (the current point), $x\_t = (1-t)x\_0 + tx\_1$, and $v\_\phi$ is the pre-trained velocity model.

Run SDS optimization for 500 steps with $w(t) = 1$. Plot the trajectory of $\theta$ over the data distribution.

### Part (c): Optimizing a Gaussian

Now parameterize a full Gaussian distribution $q\_\psi = \mathcal{N}(\mu, \sigma^2 I)$ with learnable $\mu \in \mathbb{R}^2$ and $\sigma > 0$.

Implement the **variational score distillation (VSD)** gradient, which minimizes $D\_{\text{KL}}(q\_\psi \Vert p\_{\text{data}})$:

1. Sample $x\_1 \sim q\_\psi$ (reparameterization trick: $x\_1 = \mu + \sigma \cdot \xi$, $\xi \sim \mathcal{N}(0, I)$)
2. Compute the flow matching SDS gradient as in Part (b)
3. Backpropagate through the reparameterization to update $\mu$ and $\sigma$

Run for 2000 steps. Does the Gaussian converge to cover one mode or multiple modes? How does this depend on the initial $\sigma$?

### Part (d): Mode Coverage Analysis

Run 50 independent SDS optimizations (single points, as in Part b) from random initializations $\theta\_0 \sim \mathcal{N}(0, 4I)$. Plot the final points.

1. How many of the 8 Gaussian modes are covered?
2. Is coverage uniform across modes?
3. Explain why SDS tends to find high-density modes. *Hint: SDS minimizes $D\_{\text{KL}}(\delta\_\theta \Vert p)$, which is the "reverse KL." Which modes does reverse KL prefer?*

---

## Problem 3: Discrete Diffusion and Masked Language Models (Theory)

### Part (a): The Absorbing-State Transition Matrix

For a vocabulary of size $K$ plus one absorbing state [MASK], write the $(K+1) \times (K+1)$ transition matrix $Q\_t$ for the absorbing-state forward process with corruption rate $\beta\_t$.

Show that the cumulative transition matrix $\bar{Q}\_t = Q\_1 Q\_2 \cdots Q\_t$ has a simple form:

$$
[\bar{Q}_t]_{ij} = \begin{cases} \bar{\alpha}_t & \text{if } i = j \neq [\text{MASK}] \\\\ 1 - \bar{\alpha}_t & \text{if } j = [\text{MASK}], i \neq [\text{MASK}] \\\\ 1 & \text{if } i = j = [\text{MASK}] \end{cases}
$$

where $\bar{\alpha}\_t = \prod\_{s=1}^t (1 - \beta\_s)$.

### Part (b): The ELBO

The discrete diffusion ELBO is:

$$
\log p(x_0) \geq \mathbb{E}\left[-D_{\text{KL}}(q(x_T | x_0) \Vert p(x_T)) - \sum_{t=1}^T D_{\text{KL}}(q(x_{t-1} | x_t, x_0) \Vert p_\theta(x_{t-1} | x_t))\right]
$$

For the absorbing-state process, show that the posterior $q(x\_{t-1} | x\_t, x\_0)$ has a simple form:
- If $x\_t^{(i)} \neq [\text{MASK}]$: $x\_{t-1}^{(i)} = x\_t^{(i)}$ with probability 1 (unmasked tokens stay unmasked going backward)
- If $x\_t^{(i)} = [\text{MASK}]$: $x\_{t-1}^{(i)} = x\_0^{(i)}$ with probability $\frac{\beta\_t \bar{\alpha}\_{t-1}}{1 - \bar{\alpha}\_t}$, and $x\_{t-1}^{(i)} = [\text{MASK}]$ otherwise.

*Hint: Apply Bayes' rule to $q(x\_{t-1} | x\_t, x\_0) \propto q(x\_t | x\_{t-1}) q(x\_{t-1} | x\_0)$.*

### Part (c): Connection to MLM

Show that when we parameterize $p\_\theta(x\_{t-1} | x\_t)$ by first predicting $p\_\theta(x\_0 | x\_t)$ and then computing the posterior, the training loss reduces to:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, x_t}\left[\sum_{i: x_t^{(i)} = [\text{MASK}]} w(t) \cdot \text{CE}(x_0^{(i)}, p_\theta(x_0^{(i)} | x_t))\right]
$$

where $\text{CE}$ is the cross-entropy loss and $w(t)$ is a time-dependent weight.

Compare this to the BERT masked language model objective. What is the key difference? (Consider the masking rate and the weighting.)

---

## Problem 4: Paper Summary (Reading + Writing)

Read one recent (2024-2025) paper on diffusion models that was not assigned as reading in this course. Write a structured summary.

### Paper Selection

Choose a paper from one of the following areas (or propose your own -- any diffusion/flow matching paper from 2024-2025 is acceptable):

- Video generation (e.g., CogVideoX, Open-Sora, MovieGen)
- 3D generation (e.g., InstantMesh, SV3D, Unique3D)
- Discrete diffusion for language (e.g., MDLM, SEDD, Simple Diffusion for Language)
- Improved consistency models (e.g., improved CT, sCM)
- Architecture innovations (e.g., SiT, large-scale DiT studies)
- Applications (e.g., protein design, drug discovery, weather prediction)

### Summary Structure

Write a 1.5-2 page summary covering:

1. **Problem** (2-3 sentences): What problem does the paper address? Why is it important?

2. **Key idea** (1 paragraph): What is the main technical contribution? Explain it clearly enough that a fellow student who has not read the paper could understand the approach.

3. **Method** (1-2 paragraphs): How does the method work? Include the key equations or algorithms. Relate it to concepts from this course (which weeks' material does it build on?).

4. **Results** (1 paragraph): What are the main experimental findings? What baselines does it beat? What are the limitations?

5. **Your assessment** (1 paragraph): What do you think of the paper? Is the contribution significant? What would you do differently? What follow-up experiments would you run?

---

## Problem 5: Research Proposal (Writing)

Write a 1-page research proposal for an open problem in diffusion models.

### Structure

1. **Title**: A concise, descriptive title.

2. **Problem statement** (2-3 sentences): What is the specific problem you want to address? Why is it important? What is the current state of the art?

3. **Proposed approach** (1-2 paragraphs): What is your idea? Be specific -- describe the method, architecture, loss function, or experiment you would design. Ground your proposal in the mathematical framework we have developed (score functions, SDEs, flow matching, distillation, etc.).

4. **Expected outcome** (2-3 sentences): What would success look like? What metric would you use to evaluate your approach? What would a positive result demonstrate?

5. **Risks and alternatives** (2-3 sentences): What could go wrong? What would you try if your first approach does not work?

### Topic Suggestions (Choose One or Propose Your Own)

- **Optimal noise process**: Design a noise process that is adapted to the data distribution (not fixed Gaussian). Can you learn the forward process jointly with the reverse process?
- **Few-step consistency models for video**: Extend consistency training to temporal data. How do you enforce consistency across both time steps and frames?
- **Evaluation beyond FID**: Propose a new metric for evaluating generative models that addresses FID's known shortcomings.
- **Discrete flow matching**: Can you apply the flow matching framework (continuous velocity fields, linear interpolation) to discrete data by embedding tokens in continuous space?
- **Diffusion model memorization**: Design experiments to precisely characterize when a diffusion model memorizes vs. generalizes. What factors determine the transition?
- **Physics-aware diffusion**: Incorporate physical constraints (conservation laws, symmetries) directly into the diffusion model architecture or training procedure.

---

## Problem 6: Course Reflection (Writing)

This is a short, ungraded exercise to consolidate what you have learned.

### Part (a): The Key Ideas

List what you consider the 5 most important ideas from this course, in order of importance. For each, write 2-3 sentences explaining why it is important and how it connects to other ideas in the course.

### Part (b): Connections

Draw a concept map (by hand or using any tool) showing the relationships between the major topics:
- Score functions and Langevin dynamics
- Denoising score matching
- DDPM and the ELBO
- SDE/ODE unification
- Flow matching
- Distillation and consistency models

For each connection, label the edge with a brief description of the relationship (e.g., "probability flow ODE converts SDE to ODE," "flow matching generalizes the probability flow ODE").

### Part (c): What Would You Change?

If you were to redesign this course, what would you add, remove, or change? Are there topics that deserved more time? Topics that could have been compressed? Prerequisites that should have been covered differently?

This feedback will be used to improve the course for future offerings.

---

## Submission Checklist

- [ ] Problem 1: Discrete diffusion implementation, generation quality analysis
- [ ] Problem 2: SDS implementation in 2D, mode coverage analysis
- [ ] Problem 3: Discrete diffusion theory (pen and paper)
- [ ] Problem 4: Paper summary (1.5-2 pages)
- [ ] Problem 5: Research proposal (1 page)
- [ ] Problem 6: Course reflection (ungraded)

Implementation problems should be submitted as Jupyter notebooks or Python scripts with clearly labeled outputs. Writing problems should be submitted as PDF.
