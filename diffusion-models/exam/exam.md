# CS 372: Diffusion Models — Final Exam

**Duration:** 3 hours
**Total marks:** 120
**Instructions:**
- This exam has 6 sections. Attempt ALL questions.
- Marks for each question are indicated in brackets.
- You may use a single two-sided A4 formula sheet (handwritten or typed).
- Show all working for derivation questions. Correct answers without justification receive no credit.
- For code-reading questions, you may assume standard PyTorch semantics.
- Where you are asked to "explain," a precise 2-4 sentence answer is expected unless otherwise stated.

---

## Section 1: Foundations — Probability and Stochastic Processes (20 marks)

*Covers Weeks 1-3: Probability, stochastic processes, Markov chains, score functions, Langevin dynamics, SDEs.*

### Question 1.1 [8 marks]

Consider the DDPM forward process defined by the Markov chain:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I})
$$

where $\beta\_1, \ldots, \beta\_T$ is a fixed variance schedule with $\beta\_t \in (0, 1)$.

**(a)** [5 marks] Derive the closed-form expression for $q(\mathbf{x}\_t | \mathbf{x}\_0)$. Define $\alpha\_t = 1 - \beta\_t$ and $\bar{\alpha}\_t = \prod\_{s=1}^t \alpha\_s$. Show that:

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1 - \bar{\alpha}_t)\mathbf{I})
$$

*Hint: Use the reparameterization trick at each step and combine the noise terms.*

**(b)** [3 marks] What happens to $q(\mathbf{x}\_T | \mathbf{x}\_0)$ as $T \to \infty$ (assuming $\bar{\alpha}\_T \to 0$)? Why is this property essential for the generative model to work? What would go wrong if $\bar{\alpha}\_T$ remained bounded away from zero?

### Question 1.2 [6 marks]

Let $p(\mathbf{x})$ be a differentiable probability density on $\mathbb{R}^d$.

**(a)** [2 marks] Define the score function $\nabla\_{\mathbf{x}} \log p(\mathbf{x})$. Give a geometric interpretation of what the score vector points toward at any point $\mathbf{x}$.

**(b)** [4 marks] Langevin dynamics generates samples from $p(\mathbf{x})$ via the iteration:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\eta}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_k) + \sqrt{\eta}\,\boldsymbol{\epsilon}_k, \quad \boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Explain the role of each of the three terms. What happens if we remove the noise term $\sqrt{\eta}\,\boldsymbol{\epsilon}\_k$? Why is the noise necessary for sampling (as opposed to optimization)?

### Question 1.3 [6 marks]

Consider the Ornstein-Uhlenbeck (OU) process:

$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\,\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{w}
$$

where $\mathbf{w}$ is a standard Wiener process and $\beta(t) > 0$.

**(a)** [3 marks] Identify the drift coefficient $\mathbf{f}(\mathbf{x}, t)$ and the diffusion coefficient $g(t)$. Explain intuitively why this SDE drives any initial distribution toward $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

**(b)** [3 marks] Write down the reverse-time SDE (Anderson, 1982) for this process. Your answer should explicitly involve the score $\nabla\_{\mathbf{x}} \log p\_t(\mathbf{x})$.

---

## Section 2: Denoising, Score Matching, and DDPM (25 marks)

*Covers Weeks 4-6: Denoising score matching, DDPM, NCSN.*

### Question 2.1 [8 marks]

Tweedie's formula states that for the Gaussian perturbation $\mathbf{x} = \boldsymbol{\mu} + \sigma \boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $\boldsymbol{\mu} \sim p(\boldsymbol{\mu})$, the posterior mean satisfies:

$$
\mathbb{E}[\boldsymbol{\mu} | \mathbf{x}] = \mathbf{x} + \sigma^2 \nabla_{\mathbf{x}} \log p_\sigma(\mathbf{x})
$$

where $p\_\sigma(\mathbf{x}) = \int \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I})\, p(\boldsymbol{\mu})\, d\boldsymbol{\mu}$ is the marginal density of the noisy observation.

**(a)** [4 marks] Prove Tweedie's formula. Start from the definition of $p\_\sigma(\mathbf{x})$, compute the score $\nabla\_{\mathbf{x}} \log p\_\sigma(\mathbf{x})$, and show it equals $\frac{1}{\sigma^2}(\mathbb{E}[\boldsymbol{\mu}|\mathbf{x}] - \mathbf{x})$.

**(b)** [4 marks] In the DDPM setting, we have $\mathbf{x}\_t = \sqrt{\bar{\alpha}\_t}\,\mathbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_t}\,\boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Apply Tweedie's formula to express the score $\nabla\_{\mathbf{x}\_t} \log q(\mathbf{x}\_t)$ in terms of the noise $\boldsymbol{\epsilon}$. Use this to explain why learning to predict $\boldsymbol{\epsilon}$ is equivalent to learning the score.

### Question 2.2 [10 marks]

This question asks you to derive the DDPM training loss from the variational bound.

The DDPM generative model defines:

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2 \mathbf{I})
$$

**(a)** [3 marks] The variational lower bound (VLB) on $-\log p\_\theta(\mathbf{x}\_0)$ decomposes as:

$$
\mathcal{L}_{\text{VLB}} = D_{\text{KL}}(q(\mathbf{x}_T | \mathbf{x}_0) \Vert p(\mathbf{x}_T)) + \sum_{t=2}^{T} D_{\text{KL}}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \Vert p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)) + \mathcal{L}_0
$$

The key term is $q(\mathbf{x}\_{t-1} | \mathbf{x}\_t, \mathbf{x}\_0)$, the forward process posterior. Show that this is Gaussian and derive its mean $\tilde{\boldsymbol{\mu}}\_t(\mathbf{x}\_t, \mathbf{x}\_0)$ and variance $\tilde{\beta}\_t$.

**(b)** [4 marks] Ho et al. (2020) parameterize the model mean as:

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)
$$

Show that minimizing the KL term $D\_{\text{KL}}(q(\mathbf{x}\_{t-1} | \mathbf{x}\_t, \mathbf{x}\_0) \Vert p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t))$ reduces to minimizing $\Vert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t)\Vert ^2$ (up to a time-dependent weighting factor).

**(c)** [3 marks] Ho et al. found that dropping the time-dependent weighting and using the simplified loss $\mathcal{L}\_{\text{simple}} = \mathbb{E}\_{t, \mathbf{x}\_0, \boldsymbol{\epsilon}}[\Vert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t)\Vert ^2]$ works better in practice. Why might equal weighting across timesteps improve sample quality, even though it no longer corresponds to the exact variational bound?

### Question 2.3 [7 marks]

The Noise Conditional Score Network (NCSN, Song & Ermon 2019) trains a score network $\mathbf{s}\_\theta(\mathbf{x}, \sigma)$ at multiple noise levels $\sigma\_1 > \sigma\_2 > \cdots > \sigma\_L$.

**(a)** [3 marks] The NCSN training objective is denoising score matching:

$$
\mathcal{L}_{\text{DSM}} = \frac{1}{L}\sum_{i=1}^{L} \lambda(\sigma_i)\,\mathbb{E}_{\mathbf{x}_0, \tilde{\mathbf{x}}}\left[\left\Vert \mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma_i) + \frac{\tilde{\mathbf{x}} - \mathbf{x}_0}{\sigma_i^2}\right\Vert ^2\right]
$$

Explain the term $-\frac{\tilde{\mathbf{x}} - \mathbf{x}\_0}{\sigma\_i^2}$. What is this quantity, and why is it the target for the score network?

**(b)** [4 marks] Show that $\boldsymbol{\epsilon}$-prediction in DDPM and score estimation in NCSN are equivalent, up to a known scaling factor. Specifically, if $\boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t)$ is the DDPM noise predictor and $\mathbf{s}\_\theta(\mathbf{x}\_t, t)$ is the score estimator, express one in terms of the other.

---

## Section 3: The SDE Framework and Samplers (20 marks)

*Covers Weeks 7-8: SDE unification, DDIM, DPM-Solver.*

### Question 3.1 [8 marks]

Song et al. (2021) unify DDPM and NCSN under the SDE framework. A general forward SDE is:

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}
$$

**(a)** [3 marks] Write down the corresponding reverse-time SDE. Explain why knowledge of the score function $\nabla\_{\mathbf{x}} \log p\_t(\mathbf{x})$ at all times $t$ is sufficient to reverse the process.

**(b)** [5 marks] Song et al. also derive the **probability flow ODE**:

$$
d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt
$$

Explain why this ODE defines the same marginal distributions $p\_t(\mathbf{x})$ as the forward SDE. What is the practical advantage of sampling via the ODE instead of the reverse SDE? What is the potential disadvantage?

### Question 3.2 [7 marks]

DDIM (Song et al. 2020) defines a non-Markovian forward process that preserves the same marginals $q(\mathbf{x}\_t | \mathbf{x}\_0)$ as DDPM.

**(a)** [4 marks] In DDIM, the generative update from $\mathbf{x}\_t$ to $\mathbf{x}\_{t-1}$ is:

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \sigma_t \boldsymbol{\epsilon}

$$

Identify and interpret each of the three terms. What is the "predicted $\mathbf{x}\_0$" in this expression?

**(b)** [3 marks] When $\sigma\_t = 0$ for all $t$, DDIM becomes fully deterministic. Explain why this allows DDIM to skip timesteps (e.g., going from $t=1000$ to $t=900$ directly) while DDPM cannot. What property of Markov chains prevents DDPM from skipping steps?

### Question 3.3 [5 marks]

You are given a trained score network $\mathbf{s}\_\theta(\mathbf{x}, t) \approx \nabla\_{\mathbf{x}} \log p\_t(\mathbf{x})$ and need to generate samples.

**(a)** [2 marks] Compare Euler-Maruyama discretization of the reverse SDE with the probability flow ODE in terms of: (i) stochasticity, (ii) number of function evaluations (NFE) needed for good quality.

**(b)** [3 marks] DPM-Solver achieves high-quality samples in 10-20 NFE by treating the probability flow ODE as a semi-linear ODE. Briefly explain the key insight: what is the "linear part" and what is the "nonlinear part" that is approximated with high-order methods?

---

## Section 4: Architecture, Conditioning, and Guidance (15 marks)

*Covers Weeks 9-10: Latent diffusion, U-Net, classifier guidance, classifier-free guidance.*

### Question 4.1 [6 marks]

Latent Diffusion Models (Rombach et al. 2022) run the diffusion process in the latent space of a pretrained autoencoder rather than pixel space.

**(a)** [3 marks] Let $\mathcal{E}$ be the encoder and $\mathcal{D}$ the decoder of a pretrained autoencoder, so $\mathbf{z} = \mathcal{E}(\mathbf{x})$ and $\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z})$. The diffusion model is trained to denoise $\mathbf{z}\_t$ rather than $\mathbf{x}\_t$. Give two concrete advantages of running diffusion in latent space rather than pixel space.

**(b)** [3 marks] In the U-Net architecture used by most diffusion models, conditioning information (e.g., text embeddings) is injected via cross-attention layers. Write the cross-attention operation, clearly labeling which quantities come from the noisy image features and which come from the conditioning signal.

### Question 4.2 [9 marks]

**(a)** [3 marks] In **classifier guidance** (Dhariwal & Nichol 2021), a pretrained classifier $p\_\phi(y | \mathbf{x}\_t)$ is used to steer generation toward class $y$. The guided score is:

$$
\tilde{\nabla}_{\mathbf{x}_t} \log p(\mathbf{x}_t | y) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) + \gamma \nabla_{\mathbf{x}_t} \log p_\phi(y | \mathbf{x}_t)
$$

Derive this expression from Bayes' rule (for $\gamma = 1$) and explain why $\gamma > 1$ is used in practice.

**(b)** [6 marks] **Classifier-free guidance** (Ho & Salimans 2022) avoids the need for a separate classifier. During training, the conditioning signal $\mathbf{c}$ is randomly replaced with a null token $\varnothing$ with probability $p\_{\text{uncond}}$.

At inference, the guided noise prediction is:

$$
\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing) + w\left(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)\right)
$$

(i) [3 marks] Show that this corresponds to an implicit classifier whose log-probability gradient is proportional to $\boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, \mathbf{c}) - \boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, \varnothing)$. Connect this to the classifier guidance formula from part (a).

(ii) [3 marks] As the guidance scale $w$ increases, what happens to sample quality (e.g., FID) and sample diversity? Explain the mechanism behind this tradeoff.

---

## Section 5: Modern Directions (20 marks)

*Covers Weeks 11-13: Flow matching, consistency models, advanced topics.*

### Question 5.1 [8 marks]

Flow matching (Lipman et al. 2023) defines a time-dependent velocity field $\mathbf{v}\_t(\mathbf{x})$ that transports a source distribution $p\_0$ to a target distribution $p\_1$ via the ODE:

$$
\frac{d\mathbf{x}}{dt} = \mathbf{v}_t(\mathbf{x}), \quad t \in [0, 1]
$$

**(a)** [3 marks] In conditional flow matching, the conditional probability path from noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ to data $\mathbf{x}\_1$ is defined as:

$$
\mathbf{x}_t = (1 - t)\boldsymbol{\epsilon} + t\,\mathbf{x}_1
$$

Derive the conditional velocity field $\mathbf{v}\_t(\mathbf{x}\_t | \mathbf{x}\_1)$ for this path. Compare this to the paths used by DDPM's forward process.

**(b)** [5 marks] The flow matching training loss is:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_1, \boldsymbol{\epsilon}}\left[\left\Vert \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t | \mathbf{x}_1)\right\Vert ^2\right]
$$

Compare this to the DDPM loss $\mathcal{L}\_{\text{simple}} = \mathbb{E}\_{t, \mathbf{x}\_0, \boldsymbol{\epsilon}}[\Vert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t)\Vert ^2]$ along three dimensions: (i) what is being predicted, (ii) the geometry of the transport paths, and (iii) one practical advantage of flow matching.

### Question 5.2 [7 marks]

Consistency models (Song et al. 2023) are defined by a function $f\_\theta(\mathbf{x}\_t, t)$ that satisfies the **self-consistency property**: for any $t, t' \in [\epsilon, T]$ lying on the same ODE trajectory,

$$
f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_{t'}, t')
$$

**(a)** [3 marks] Explain what the self-consistency property means geometrically. What does $f\_\theta(\mathbf{x}\_t, t)$ map to, and why does self-consistency enable single-step generation?

**(b)** [4 marks] There are two ways to train a consistency model: **consistency distillation** (from a pretrained diffusion model) and **consistency training** (from scratch). For consistency distillation, the training loss is:

$$
\mathcal{L}_{\text{CD}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[d\left(f_\theta(\mathbf{x}_{t+\Delta t}, t+\Delta t),\; f_{\theta^-}(\hat{\mathbf{x}}_t, t)\right)\right]
$$

where $\hat{\mathbf{x}}\_t$ is obtained by running one step of the ODE solver from $\mathbf{x}\_{t+\Delta t}$ using the pretrained model, and $\theta^-$ is an exponential moving average of $\theta$. Explain: (i) why $\theta^-$ (the EMA target) is needed instead of using $\theta$ directly, and (ii) the role of $\Delta t$ and what happens as $\Delta t \to 0$.

### Question 5.3 [5 marks]

**(a)** [3 marks] Adapting diffusion models to **video** generation introduces a new challenge beyond image generation: temporal consistency. Describe two architectural or algorithmic modifications that video diffusion models use to maintain coherence across frames. For each, explain why naive frame-by-frame generation would fail.

**(b)** [2 marks] Diffusion models for **discrete** data (e.g., text) cannot use the standard Gaussian noise process. Briefly describe one approach to defining a "forward noising process" for discrete tokens, and state what plays the role of the score function in the discrete setting.

---

## Section 6: Code Reading and Synthesis (20 marks)

*Spans all weeks. Tests practical understanding and the ability to connect ideas.*

### Question 6.1 [10 marks]

Read the following PyTorch code for a DDPM training loop:

```python
import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    """Assume this is a correctly implemented U-Net that takes
    (x_t, t_embedding) and outputs a tensor of the same shape as x_t."""
    ...

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def train_ddpm(model, dataloader, T=1000, epochs=100, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    betas = linear_beta_schedule(T).cuda()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for epoch in range(epochs):
        for x_0 in dataloader:
            x_0 = x_0.cuda()
            B = x_0.shape[0]

            # Sample random timesteps
            t = torch.randint(0, T, (B,)).cuda()

            # Sample noise
            epsilon = torch.randn_like(x_0)

            # Compute x_t
            sqrt_alpha_bar = alphas_cumprod[t].sqrt()
            sqrt_one_minus_alpha_bar = (1 - alphas_cumprod[t]).sqrt()
            x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon

            # Predict noise
            epsilon_pred = model(x_t, t)

            # Compute loss
            loss = nn.MSELoss()(epsilon_pred, epsilon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**(a)** [4 marks] There are **two bugs** in this code. Identify each bug, explain why it causes incorrect behavior, and provide the fix.

*Hint: One bug is a shape/broadcasting issue. The other is a subtle problem with the loss function.*

**(b)** [3 marks] Suppose we modify the code to predict $\mathbf{x}\_0$ directly instead of $\boldsymbol{\epsilon}$. Rewrite the three lines that compute `x_t`, `epsilon_pred`, and `loss` for $\mathbf{x}\_0$-prediction. What is one advantage and one disadvantage of $\mathbf{x}\_0$-prediction compared to $\boldsymbol{\epsilon}$-prediction?

**(c)** [3 marks] After training, you write a sampling loop using 1000 steps and find that images look good but generation takes 45 seconds per image. Name two methods from the course that would reduce generation time and briefly explain the mechanism of each.

### Question 6.2 [10 marks]

**Synthesis question.** Trace the evolution of the core generative mechanism from DDPM to consistency models, showing how each advance solves a specific limitation of its predecessor.

Your answer should address the following progression and, for each transition, identify (i) the limitation being addressed and (ii) the key insight:

**(a)** [2 marks] **DDPM** $\to$ **Score SDE framework:** What does the SDE perspective reveal that the discrete Markov chain formulation obscures?

**(b)** [2 marks] **Score SDE** $\to$ **Probability flow ODE:** Why move from an SDE to an ODE? What new capabilities does this unlock?

**(c)** [2 marks] **Probability flow ODE** $\to$ **DDIM:** How does DDIM exploit the ODE perspective, and what is the practical payoff?

**(d)** [2 marks] **DDIM/ODE solvers** $\to$ **Flow matching:** What changes in the design philosophy, and what advantage does this bring?

**(e)** [2 marks] **Flow matching / Diffusion ODE** $\to$ **Consistency models:** What is the final bottleneck being removed, and how does the self-consistency idea bypass it?

---

**END OF EXAMINATION**

*Marks: Section 1 (20) + Section 2 (25) + Section 3 (20) + Section 4 (15) + Section 5 (20) + Section 6 (20) = 120 total*
