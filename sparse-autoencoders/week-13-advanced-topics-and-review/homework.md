# Week 13 Homework: Advanced SAE Architectures

**CS 371: Sparse Autoencoders and Neural Feature Extraction**

This is the final homework of the course. It is intentionally lighter than previous weeks to give you time to review for the exam. Focus on depth of understanding rather than volume of work.

---

## Problem 1: TopK SAE Implementation (50 points)

Implement a TopK Sparse Autoencoder and compare it against the L1 SAE you built in Week 10.

### 1a. Implementation (30 points)

Implement a `TopKSAE` class in PyTorch with the following specification:

```python
class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k: int):
        """
        Args:
            input_dim: dimension of input activations
            hidden_dim: number of dictionary features (overcomplete)
            k: number of features to keep active
        """
        ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_hat: reconstructed input
            z: sparse feature activations (exactly k nonzero per input)
        """
        ...
```

Requirements:
- The encoder should compute pre-activations $\mathbf{h} = \mathbf{W}\_e \mathbf{x} + \mathbf{b}\_e$.
- Apply the TopK operation: keep only the top $K$ values, zero the rest.
- Use `torch.topk` for the forward pass. For the backward pass, PyTorch's autograd handles the straight-through behavior automatically when you index into the pre-activations.
- The loss should be reconstruction-only: $\mathcal{L} = \Vert \mathbf{x} - \hat{\mathbf{x}} \Vert \_2^2$.
- Normalize decoder columns to unit norm after each gradient step (as in Week 10).

### 1b. Comparison Experiment (20 points)

Using the same dataset and model activations as Week 10 (or MNIST if you prefer a simpler testbed):

1. Train both an L1 SAE (with your best $\lambda$ from Week 10) and a TopK SAE with several values of $K$.
2. For each trained model, report:
   - Average L0 (number of active features per input)
   - Reconstruction MSE
   - Fraction of dead features (features that never activate over the test set)
3. Plot the **Pareto frontier**: L0 (x-axis) vs. MSE (y-axis) for both architectures. Vary $\lambda$ for the L1 SAE and $K$ for the TopK SAE.
4. In 3-5 sentences, discuss: Which architecture gives better control over sparsity? Which achieves better reconstruction at the same sparsity level? Why?

---

## Problem 2: Paper Reading (30 points)

Read **one** of the following papers:

- **Option A:** Gao et al., "Scaling and Evaluating Sparse Autoencoders" (OpenAI, 2024)
- **Option B:** Rajamanoharan et al., "Improving Dictionary Learning with Gated Sparse Autoencoders" (DeepMind, 2024)
- **Option C:** Rajamanoharan et al., "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders" (DeepMind, 2024)

Write a **one-page summary** (400-600 words, no more) covering:

1. **Problem.** What specific limitation of existing SAEs does this paper address?
2. **Method.** What is the key architectural or methodological innovation? Describe it precisely enough that someone who has taken this course could implement it.
3. **Results.** What are the main empirical findings? Be specific: cite numbers, plots, or comparisons from the paper.
4. **Limitations.** What does the paper *not* address? What questions does it leave open? Identify at least one limitation the authors acknowledge and one they do not.

**Grading criteria:** Precision and conciseness. We are looking for evidence that you read and understood the paper, not that you can paraphrase the abstract.

---

## Problem 3: Experimental Design — Feature Faithfulness (20 points)

A central open question in SAE research is **faithfulness**: are the features that SAEs find the "true" features the network uses in its computation?

Design a hypothetical experiment to test faithfulness. Your proposal should include:

### 3a. Setup (5 points)
- What model would you study? What layer?
- What SAE architecture would you use?
- What dataset?

### 3b. Methodology (10 points)
Describe a concrete experimental procedure. You must specify:
- What you would measure (be precise about the metric).
- What intervention you would perform (e.g., clamping features, ablating features, steering).
- What control conditions you would include.
- What comparison would distinguish "faithful features" from "SAE artifacts."

### 3c. Interpretation (5 points)
- What result would **convince** you that the features are faithful?
- What result would make you **doubt** faithfulness?
- What are the limitations of your experiment — i.e., what could go wrong even if the results look positive?

**Note:** You do not need to run this experiment. This is a thought exercise in experimental design. We are evaluating the quality of your reasoning, not the results.

---

## Submission Guidelines

- **Problem 1:** Submit your `topk_sae.py` implementation, training script, and a short report (plots + discussion) as a Jupyter notebook or PDF.
- **Problem 2:** Submit your one-page summary as a PDF. Strict one-page limit.
- **Problem 3:** Submit your experimental proposal as a PDF (1-2 pages).

**Due date:** One week from today, but prioritize exam preparation. This homework is weighted to reflect the lighter workload.

---

## Study Tips for the Final Exam

As you work through this homework, use it as a springboard for exam review:

- **Problem 1** connects to Weeks 9-10 (sparsity, SAE architecture, training).
- **Problem 2** connects to Weeks 11-13 (interpretability, advanced architectures).
- **Problem 3** connects to Weeks 11-12 (causal interpretability, experimental methodology).

For the exam, make sure you can:
- Derive key results from first principles (ELBO, soft-thresholding, PCA = linear AE).
- Explain the motivation behind each autoencoder variant (what problem does it solve?).
- Trace the conceptual arc from PCA through SAEs to interpretability.
- Read and reason about PyTorch code for autoencoders and SAEs.
- Design experiments and reason about evaluation metrics.

Good luck with both the homework and the exam.
