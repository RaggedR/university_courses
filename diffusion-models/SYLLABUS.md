---
title: "CS 372: Diffusion Models and Generative Modeling"
---

# CS 372: Diffusion Models and Generative Modeling

## Course Overview

This course provides a rigorous, implementation-driven introduction to diffusion models — the family of generative models that has transformed image synthesis, video generation, and beyond. We start from the mathematical foundations (stochastic processes, score functions, SDEs), build through the core theory of denoising diffusion and score matching, and arrive at the modern engineering that makes these models practical: fast samplers, latent-space compression, conditioning mechanisms, flow matching, and distillation.

The emphasis throughout is on understanding *why* diffusion models work, not just *how* to use them. Every key result is derived from first principles before being implemented. By the end of the course, you will be able to read current research papers in this area fluently and implement the core algorithms yourself.

**Format:** Self-study, one semester (13 weeks), ~8-10 hours/week
**Prerequisites:** Undergraduate linear algebra, multivariable calculus, basic probability. Programming experience in any language (we will use Python + PyTorch).
**Assessment:** Weekly problem sets (50%), final exam (30%), course project (20%)

---

## Learning Objectives

By the end of this course, you will be able to:

1. **Derive** the forward and reverse diffusion processes from first principles, including the variational bound and its connection to denoising score matching
2. **Implement** from scratch: DDPM training and sampling, score-based models, DDIM, and classifier-free guidance using PyTorch
3. **Explain** the SDE unification framework (Song et al. 2021) and how DDPM, NCSN, and probability flow ODEs are special cases of a single theory
4. **Analyze** the trade-offs between sample quality, diversity, and generation speed across different samplers and acceleration techniques
5. **Apply** conditioning and guidance methods (classifier guidance, classifier-free guidance, text conditioning) to control generation
6. **Evaluate** modern architectural choices — U-Nets, attention mechanisms, VAE compression, latent diffusion — and explain why each design decision was made
7. **Critically engage** with the current research frontier: flow matching, consistency models, distillation, and applications beyond images
8. **Design** experiments to compare generative models using appropriate metrics (FID, CLIP score, human evaluation)

---

## Required Software Setup

```bash
# Python 3.10+ recommended
python3 -m venv diffusion-course
source diffusion-course/bin/activate

# Core dependencies
pip install torch torchvision numpy matplotlib jupyter
pip install scikit-learn pandas seaborn

# Diffusion-specific libraries
pip install diffusers accelerate transformers  # Hugging Face ecosystem
pip install einops tqdm pillow

# For later weeks (latent diffusion, CLIP conditioning)
pip install open-clip-torch safetensors
```

---

## Weekly Schedule

### Part I: Mathematical Foundations (Weeks 1-3)

The first three weeks build the mathematical machinery that diffusion models rest on. If your probability and stochastic processes background is thin, invest extra time here — every derivation in the course flows from these foundations. Week 3 on SDEs is the most demanding mathematical week; it is also the week that makes the unification in Part II possible.

| Week | Topic | Key Concepts |
|------|-------|--------------|
| 1 | Probability, Stochastic Processes, and Markov Chains | Gaussian distributions, multivariate normals, Markov chains, transition kernels, stationary distributions, detailed balance, ergodicity |
| 2 | Score Functions and Langevin Dynamics | The score function ∇ₓ log p(x), score estimation, Fisher divergence, Langevin Monte Carlo sampling, annealed Langevin dynamics, mixing time |
| 3 | Stochastic Differential Equations | Brownian motion, Wiener process, Itô calculus, Itô's lemma, Fokker-Planck equation, Ornstein-Uhlenbeck process, reverse-time SDEs (Anderson 1982) |

### Part II: Core Diffusion Theory (Weeks 4-7)

The theoretical heart of the course. We develop the two originally separate threads — denoising diffusion (Ho et al.) and score-based models (Song & Ermon) — then show how Song et al. 2021 unified them through the SDE framework. By the end of Week 7, you will understand that DDPM, NCSN, and probability flow ODEs are all views of the same underlying process.

| Week | Topic | Key Concepts |
|------|-------|--------------|
| 4 | Denoising and Score Matching | Denoising score matching, the Vincent (2011) connection between denoising and score estimation, noise conditional score networks, sliced score matching |
| 5 | Denoising Diffusion Probabilistic Models (DDPM) | Forward diffusion process, reverse denoising process, variational lower bound, noise prediction, the ε-prediction reparameterization, variance schedules (linear, cosine) |
| 6 | Score-Based Generative Models | Noise Conditional Score Networks (NCSN), multi-scale noise perturbation, annealed score matching, geometric noise schedules, connections between NCSN and DDPM |
| 7 | The SDE Unification | Variance Preserving SDE (VP-SDE), Variance Exploding SDE (VE-SDE), sub-VP-SDE, probability flow ODE, likelihood computation via the instantaneous change-of-variables formula, the Song et al. 2021 framework |

### Part III: Making It Practical (Weeks 8-10)

Theory meets engineering. We address the three problems that stood between diffusion models and real-world deployment: sampling speed (Week 8), computational cost of operating in pixel space (Week 9), and controllability (Week 10). This is where diffusion models went from impressive research to the engine behind Stable Diffusion, DALL-E, and Midjourney.

| Week | Topic | Key Concepts |
|------|-------|--------------|
| 8 | Samplers and Acceleration | DDIM (deterministic sampling), DPM-Solver (high-order ODE solvers), exponential integrators, predictor-corrector methods, adaptive step sizes, trading off steps vs. quality |
| 9 | Latent Diffusion and Architecture | U-Net architecture for diffusion, self-attention and cross-attention in U-Nets, VAE compression to latent space, the Stable Diffusion architecture, timestep embeddings, downsampling/upsampling |
| 10 | Conditioning and Guidance | Classifier guidance (Dhariwal & Nichol 2021), classifier-free guidance (Ho & Salimans 2022), guidance scale, text conditioning via CLIP and T5 encoders, ControlNet, IP-Adapter, image-to-image and inpainting |

### Part IV: Frontiers (Weeks 11-13)

The field is moving fast. These weeks cover the most important post-2022 developments: flow matching as a cleaner theoretical alternative to diffusion, distillation methods that reduce generation from 50 steps to 1-4, and the expansion of diffusion beyond static images. Engage critically — some of this work is still being refined.

| Week | Topic | Key Concepts |
|------|-------|--------------|
| 11 | Flow Matching and Rectified Flows | Continuous normalizing flows (CNFs), simulation-free training, conditional flow matching (Lipman et al. 2023), optimal transport paths, rectified flows (Liu et al. 2023), linear interpolation, relationship to diffusion SDEs |
| 12 | Distillation and Fast Generation | Progressive distillation (Salimans & Ho 2022), consistency models (Song et al. 2023), consistency training vs. consistency distillation, latent consistency models, adversarial distillation (SDXL-Turbo), one-step and few-step generation |
| 13 | Advanced Topics and Open Problems | Video diffusion models (Ho et al. 2022), 3D generation and DreamFusion (Poole et al. 2023), Score Distillation Sampling, discrete diffusion for text, diffusion for audio and music, connections to optimal transport, open research questions |

---

## Reading List

### Textbooks and References

- **Prince** — *Understanding Deep Learning* (2023). Free online at udlbook.github.io. Chapters 18 (diffusion models) and surrounding material on generative models. The clearest textbook treatment currently available.
- **Song** — *Diffusion and Score-Based Generative Models*. Online course notes from Stanford CS 236. The definitive lecture notes by one of the field's creators.
- **Bishop & Bishop** — *Deep Learning: Foundations and Concepts* (2024). Chapter on generative models provides a complementary perspective.

### Key Papers (by week)

| Week | Paper |
|------|-------|
| 2 | Hyvärinen, "Estimation of Non-Normalized Statistical Models by Score Matching" (2005) |
| 3 | Anderson, "Reverse-Time Diffusion Equation Models" (1982) |
| 4 | Vincent, "A Connection Between Score Matching and Denoising Autoencoders" (2011) |
| 5 | Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015); Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models" (2020) |
| 6 | Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution" (NeurIPS 2019); Song & Ermon, "Improved Techniques for Training Score-Based Generative Models" (NeurIPS 2020) |
| 7 | Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole, "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021) |
| 8 | Song, Meng & Ermon, "Denoising Diffusion Implicit Models" (ICLR 2021); Lu, Zhou, Bao, Chen, Li & Zhu, "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling" (NeurIPS 2022) |
| 9 | Rombach, Blattmann, Lorenz, Esser & Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022); Ronneberger, Fischer & Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015) |
| 10 | Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (NeurIPS 2021); Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022) |
| 11 | Lipman, Chen, Ben-Hamu, Nickel & Le, "Flow Matching for Generative Modeling" (ICLR 2023); Liu, Gong & Liu, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR 2023) |
| 12 | Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models" (ICLR 2022); Song, Dhariwal, Chen & Sutskever, "Consistency Models" (ICML 2023) |
| 13 | Ho, Salimans, Gritsenko, Chan, Norouzi & Fleet, "Video Diffusion Models" (NeurIPS 2022); Poole, Jain, Barron & Mildenhall, "DreamFusion: Text-to-3D using 2D Diffusion" (ICLR 2023) |

---

## Assessment

### Weekly Problem Sets (50%)

Each week includes a problem set with both theoretical (pen-and-paper) and implementation (Python/PyTorch) components. The theoretical problems build mathematical fluency; the implementation problems build engineering intuition. Both are essential.

- **Weeks 1-3**: Mathematical derivations (Fokker-Planck from Itô, Langevin dynamics convergence, reverse-time SDE derivation) and implementing basic samplers
- **Weeks 4-7**: Implementing denoising score matching, DDPM training loop, noise schedules, connecting NCSN and DDPM empirically
- **Weeks 8-10**: Building DDIM from a trained DDPM, comparing sampler quality vs. speed, implementing classifier-free guidance
- **Weeks 11-13**: Implementing flow matching, reproducing consistency model results, reading and summarizing a frontier paper

### Final Exam (30%)

A comprehensive exam covering all course material. Format: a mix of short-answer conceptual questions, mathematical derivations, and code reading/analysis. The exam tests understanding, not memorization — you should be able to derive the DDPM variational bound from scratch, explain why classifier-free guidance works, and trace through the SDE unification without notes.

### Course Project (20%)

**Option A:** Implement DDPM from scratch (no library code for the diffusion logic) and train it on CIFAR-10 or CelebA-64. Report FID scores, show generated samples at various training stages, and analyze failure modes.

**Option B:** Reproduce a key result from one of the course papers, with a clearly documented methodology. Good candidates: the NCSN→DDPM connection (Week 6), the probability flow ODE likelihood computation (Week 7), or classifier-free guidance ablation (Week 10).

**Option C:** Implement and compare three samplers — DDIM, DPM-Solver, and Euler-Maruyama — on the same pretrained diffusion model. Measure FID vs. number of function evaluations (NFE) and analyze when each method is preferable.

---

## Suggested Weekly Study Plan

| Activity | Hours/Week |
|----------|-----------|
| Read lecture notes / textbook chapters | 2-3 |
| Read assigned paper(s) | 1-2 |
| Work on problem set (theory) | 2-3 |
| Work on problem set (implementation) | 2-3 |
| **Total** | **7-11** |

---

## A Note on Mathematical Maturity

This course leans more heavily on stochastic processes and differential equations than a typical deep learning course. Weeks 1-3 are genuine mathematical foundations, not hand-waving — work through them carefully even if you have seen some of this material before. The effort pays compound interest: when we derive the reverse-time SDE in Week 7, you will need comfortable fluency with Itô calculus; when we compare VP-SDE and VE-SDE, you will need intuition about how different drift and diffusion coefficients shape the forward process.

If you find weeks 1-3 genuinely difficult (not just rusty), consider supplementing with:
- Øksendal, *Stochastic Differential Equations* (chapters 1-5) for the SDE material
- Pavliotis, *Stochastic Processes and Applications* for a physics-flavored introduction
- 3Blue1Brown's *Essence of Linear Algebra* and *Essence of Calculus* for geometric intuition

---

## A Note on Python for Non-Python Developers

If you are coming from another ecosystem, you will need basic Python fluency. We recommend:
- Official Python tutorial (the first 10 chapters)
- NumPy quickstart tutorial
- PyTorch's "60 Minute Blitz" tutorial

You do NOT need to be a Python expert. The implementation exercises are focused on mathematical logic, not software engineering patterns. Write clear, correct code — don't worry about being "Pythonic."
