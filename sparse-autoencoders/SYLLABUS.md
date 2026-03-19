# CS 371: Sparse Autoencoders and Neural Feature Extraction

## Course Overview

This course provides a rigorous, implementation-driven introduction to sparse autoencoders (SAEs) and the broader landscape of neural feature extraction. Starting from mathematical foundations, we build toward the cutting-edge application of SAEs in mechanistic interpretability — understanding what neural networks have learned by extracting human-interpretable features from their internal representations.

The course is designed for learners with an undergraduate mathematics background (linear algebra, calculus, probability) and software engineering experience who want to develop both theoretical understanding and practical implementation skills.

**Format:** Self-study, one semester (13 weeks), ~8-10 hours/week
**Prerequisites:** Undergraduate linear algebra, multivariable calculus, basic probability. Programming experience in any language (we will use Python + PyTorch).
**Assessment:** Weekly problem sets (50%), final exam (30%), course project (20%)

---

## Learning Objectives

By the end of this course, you will be able to:

1. **Explain** the mathematical foundations of autoencoders, including the relationship between PCA, autoencoders, and sparse coding
2. **Implement** from scratch: feedforward networks, autoencoders, variational autoencoders, and sparse autoencoders using PyTorch
3. **Derive** the training objectives for each autoencoder variant, including the ELBO for VAEs and sparsity penalties for SAEs
4. **Analyze** learned representations using visualization techniques and quantitative metrics
5. **Apply** sparse autoencoders to extract interpretable features from neural network activations
6. **Critically evaluate** current research in mechanistic interpretability, including Anthropic's and OpenAI's work on SAEs
7. **Design** experiments to validate whether extracted features are meaningful and causally relevant

---

## Required Software Setup

```bash
# Python 3.10+ recommended
python3 -m venv sae-course
source sae-course/bin/activate

# Core dependencies
pip install torch torchvision numpy matplotlib jupyter
pip install scikit-learn pandas seaborn

# For later weeks (mechanistic interpretability)
pip install transformer-lens einops fancy-einsum
pip install sae-lens  # Anthropic's SAE library
```

---

## Weekly Schedule

### Part I: Mathematical Foundations (Weeks 1-4)

The first four weeks rebuild your mathematical muscles and establish the neural network foundations everything else rests on. If your linear algebra is rusty, spend extra time here — it pays dividends throughout the course.

| Week | Topic | Key Concepts |
|------|-------|-------------|
| 1 | Linear Algebra & Optimization | Vector spaces, eigendecomposition, SVD, gradient descent, convexity |
| 2 | Probability & Information Theory | Distributions, Bayes' theorem, KL divergence, entropy, MLE |
| 3 | Neural Networks from First Principles | Perceptrons, MLPs, activation functions, universal approximation |
| 4 | Training Neural Networks | Backpropagation, SGD, Adam, loss functions, regularization |

### Part II: Autoencoders (Weeks 5-8)

We build from the idea of representation learning through progressively more sophisticated autoencoder architectures. Each variant addresses a specific limitation of the previous one.

| Week | Topic | Key Concepts |
|------|-------|-------------|
| 5 | Representation Learning & Dimensionality Reduction | Features, manifold hypothesis, PCA, t-SNE |
| 6 | The Autoencoder | Encoder-decoder architecture, bottleneck, reconstruction loss |
| 7 | Regularized Autoencoders | Denoising AE, contractive AE, overcomplete representations |
| 8 | Variational Autoencoders | Latent variables, ELBO, reparameterization trick, generative models |

### Part III: Sparsity (Weeks 9-10)

The mathematical heart of the course. We study why sparsity is a powerful inductive bias — from biological neural coding to compressed sensing — and how to enforce it in autoencoders.

| Week | Topic | Key Concepts |
|------|-------|-------------|
| 9 | Sparsity & Dictionary Learning | L1 regularization, LASSO, sparse coding, ISTA, dictionary learning |
| 10 | Sparse Autoencoders | SAE architecture, KL divergence penalty, L1 penalty, dead neurons |

### Part IV: Interpretability & Applications (Weeks 11-13)

We connect everything to the modern motivation for SAEs: understanding what neural networks have learned. This section engages directly with recent research papers.

| Week | Topic | Key Concepts |
|------|-------|-------------|
| 11 | Mechanistic Interpretability | Superposition hypothesis, features as directions, toy models |
| 12 | SAEs for Language Models | Extracting features from transformers, Anthropic's work, feature visualization |
| 13 | Advanced Architectures & Open Problems | TopK SAEs, Gated SAEs, JumpReLU, scaling laws, open questions |

---

## Reading List

### Textbooks and References

- **Goodfellow, Bengio, Courville** — *Deep Learning* (MIT Press, 2016). Free online at deeplearningbook.org. Chapters 5, 6, 7, 14 are most relevant.
- **3Blue1Brown** — *Neural Networks* video series. Excellent visual intuition for backpropagation and linear algebra.
- **Gilbert Strang** — *Linear Algebra and Its Applications*. For the linear algebra refresher.

### Key Papers (by week)

| Week | Paper |
|------|-------|
| 5 | Bengio et al., "Representation Learning: A Review and New Perspectives" (2013) |
| 6 | Hinton & Salakhutdinov, "Reducing the Dimensionality of Data with Neural Networks" (2006) |
| 7 | Vincent et al., "Extracting and Composing Robust Features with Denoising Autoencoders" (2008) |
| 8 | Kingma & Welling, "Auto-Encoding Variational Bayes" (2014) |
| 9 | Olshausen & Field, "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images" (1996) |
| 10 | Makhzani & Frey, "k-Sparse Autoencoders" (2013); Andrew Ng, "Sparse Autoencoder" (CS294A notes) |
| 11 | Elhage et al., "Toy Models of Superposition" (Anthropic, 2022) |
| 12 | Bricken et al., "Towards Monosemanticity" (Anthropic, 2023); Templeton et al., "Scaling Monosemanticity" (Anthropic, 2024) |
| 13 | Gao et al., "Scaling and Evaluating Sparse Autoencoders" (OpenAI, 2024); Rajamanoharan et al., "Improving Dictionary Learning with Gated Sparse Autoencoders" (2024) |

---

## Assessment

### Weekly Problem Sets (50%)

Each week includes a problem set with both theoretical (pen-and-paper) and implementation (Python/PyTorch) components. The theoretical problems build mathematical maturity; the implementation problems build engineering intuition. Both are essential.

- **Weeks 1-4**: Emphasis on mathematical derivations and implementing algorithms from scratch
- **Weeks 5-8**: Building and training autoencoder variants, analyzing learned representations
- **Weeks 9-10**: Implementing sparse coding and sparse autoencoders, comparing approaches
- **Weeks 11-13**: Reproducing results from research papers, interpreting extracted features

### Final Exam (30%)

A comprehensive exam covering all course material. Format: a mix of short-answer conceptual questions, mathematical derivations, and code reading/analysis. The exam tests understanding, not memorization — you should be able to derive results from first principles, not recite formulas.

### Course Project (20%)

**Option A:** Train a sparse autoencoder on a small language model (e.g., GPT-2 small) and produce an interpretability report analyzing 10+ features you discover.

**Option B:** Reproduce a key result from one of the course papers, extending it in a novel direction of your choosing.

**Option C:** Design and implement a comparison study between two SAE architectures (e.g., vanilla L1 vs. TopK vs. Gated) on the same model, with quantitative evaluation.

---

## Suggested Weekly Study Plan

| Activity | Hours/Week |
|----------|-----------|
| Read lecture notes | 2-3 |
| Read assigned paper(s) | 1-2 |
| Work on problem set (theory) | 2-3 |
| Work on problem set (implementation) | 2-3 |
| **Total** | **7-11** |

---

## A Note on Mathematical Maturity

This course assumes you once knew linear algebra and probability well but may be rusty. Weeks 1-2 are genuine refreshers, not just hand-waving — work through them carefully even if the material feels familiar. The effort pays compound interest: when we derive the VAE's ELBO in Week 8, you'll need comfortable fluency with KL divergence; when we analyze superposition in Week 11, you'll need geometric intuition about high-dimensional vector spaces.

If you find weeks 1-2 genuinely difficult (not just rusty), consider supplementing with:
- 3Blue1Brown's *Essence of Linear Algebra* video series
- Khan Academy's probability and statistics courses
- Gilbert Strang's MIT OpenCourseWare lectures on Linear Algebra

---

## A Note on Python for Non-Python Developers

If you're coming from Flutter/Dart or another ecosystem, you'll need basic Python fluency. We recommend:
- Official Python tutorial (the first 10 chapters)
- NumPy quickstart tutorial
- PyTorch's "60 Minute Blitz" tutorial

You do NOT need to be a Python expert. The implementation exercises are focused on mathematical logic, not software engineering patterns. Write clear, correct code — don't worry about being "Pythonic."
