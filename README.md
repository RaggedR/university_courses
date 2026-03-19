# University Courses

Self-study courses and guides covering deep learning, interpretability, and software engineering. Each full course is a 13-week semester with lecture notes, problem sets, and a final exam.

## Full Courses

### [CS 371: Sparse Autoencoders and Neural Feature Extraction](sparse-autoencoders/)
A rigorous, implementation-driven introduction to sparse autoencoders and mechanistic interpretability. Starting from linear algebra and probability, builds through autoencoders and variational autoencoders to the cutting edge of neural feature extraction — understanding what neural networks have learned by extracting human-interpretable features from their internal representations.

**Topics:** Linear algebra, probability, neural networks, autoencoders, VAEs, sparsity, dictionary learning, sparse autoencoders, mechanistic interpretability, superposition, Anthropic's monosemanticity work.

### [CS 372: Diffusion Models and Generative Modeling](diffusion-models/)
A ground-up treatment of diffusion models, from stochastic calculus through DDPM and score-based models to modern frontiers like flow matching and consistency models. Emphasises the deep connections: denoising is score estimation (Tweedie's formula), DDPM and NCSN are discrete cases of a unified SDE framework, and flow matching is the clean successor.

**Topics:** Score functions, Langevin dynamics, SDEs, denoising score matching, DDPM, NCSN, the SDE unification, samplers (DDIM, DPM-Solver), latent diffusion, classifier-free guidance, flow matching, rectified flows, distillation, consistency models, video/3D/discrete diffusion.

## Standalone Guides

| Guide | Description |
|-------|-------------|
| [Sparse Autoencoders Guide](sparse_autoencoders_guide.md) | Accessible essay on SAEs for developers — no homework, just understanding |
| [Diffusion Models Guide](diffusion_models_guide.md) | Standalone guide to diffusion models |
| [CS Foundations Guide](cs_foundations_guide.md) | Computer science fundamentals |
| [Fullstack Guide](fullstack_guide.md) | Full-stack software engineering |
| [Claude Code Guide](claude_code_guide.md) | Guide to using Claude Code |

## Course Format

Each full course follows the same structure:
- `SYLLABUS.md` — overview, schedule, reading list, assessment
- `week-XX-topic/notes.md` — detailed lecture notes with mathematical derivations
- `week-XX-topic/homework.md` — theory (pen-and-paper) + implementation (PyTorch) problems
- `exam/exam.md` + `exam/solutions.md` — comprehensive final exam

**Prerequisites:** Undergraduate linear algebra, multivariable calculus, basic probability. Programming experience in any language (courses use Python + PyTorch).
