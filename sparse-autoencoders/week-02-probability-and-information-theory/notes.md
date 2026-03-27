# Week 2: Probability and Information Theory

> *"Information is the resolution of uncertainty."*
> — Claude Shannon

Last week we built the geometric and algebraic machinery for working with vectors and matrices. This week we add the second pillar: probability and information. These aren't separate subjects — they're deeply intertwined with linear algebra. The covariance matrix is both a linear algebra object (symmetric, positive semi-definite) and a probability object (it encodes the shape of a distribution). Maximum likelihood estimation is both an optimization problem (Week 1) and a probabilistic inference (this week). And information theory — particularly KL divergence — will turn out to be the mathematical core of variational autoencoders (Week 8) and a key sparsity penalty for sparse autoencoders (Week 10).

If your probability is rusty, work through this week carefully. The payoff comes in Week 8, when we need to manipulate expectations, KL divergences, and log-likelihoods fluently to derive the ELBO.

---

## 1. Probability Foundations

### 1.1 Sample Spaces and Events

A **probability space** consists of three things:

1. A **sample space** $\Omega$ — the set of all possible outcomes
2. An **event space** $\mathcal{F}$ — a collection of subsets of $\Omega$ (the things we can assign probabilities to)
3. A **probability measure** $P: \mathcal{F} \to [0, 1]$ satisfying the axioms:
   - $P(\Omega) = 1$
   - $P(A) \geq 0$ for all events $A$
   - For disjoint events $A\_1, A\_2, \ldots$: $P(\bigcup\_i A\_i) = \sum\_i P(A\_i)$ (countable additivity)

From these three axioms, all of probability theory follows. This is Kolmogorov's formalization, and it's worth pausing to appreciate: these axioms say nothing about "randomness" or "chance." They're about assigning consistent numbers to sets. The philosophical interpretation — frequentist, Bayesian, or otherwise — is a separate matter.

**Key derived properties:**
- $P(\emptyset) = 0$
- $P(A^c) = 1 - P(A)$
- If $A \subseteq B$, then $P(A) \leq P(B)$
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (inclusion-exclusion)

### 1.2 Conditional Probability and Bayes' Theorem

The **conditional probability** of $A$ given $B$ is:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
$$

This is the probability of $A$ in the "restricted universe" where $B$ is known to have occurred.

**Bayes' theorem** follows immediately:

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

In the Bayesian interpretation: $P(A)$ is the **prior** (what you believed before seeing data), $P(B|A)$ is the **likelihood** (how probable the data is under hypothesis $A$), and $P(A|B)$ is the **posterior** (what you believe after seeing data). The denominator $P(B)$ is the **evidence** or **marginal likelihood** — a normalizing constant that's often intractable to compute.

**Concrete example.** A medical test for a rare disease (prevalence 1%) has 95% sensitivity (true positive rate) and 90% specificity (true negative rate). If you test positive, what's the probability you have the disease?

$$
P(\text{disease}|\text{positive}) = \frac{P(\text{positive}|\text{disease}) \cdot P(\text{disease})}{P(\text{positive})} = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.10 \times 0.99} = \frac{0.0095}{0.1085} \approx 8.8\%
$$

The base rate (1%) dominates: even with a positive test, you probably don't have the disease. This is a vivid illustration of why priors matter — and why Bayesian reasoning is essential for making sense of evidence.

**Why this matters for ML:** Bayesian reasoning underpins regularization (priors = soft constraints on parameters), variational inference (approximating intractable posteriors), and the entire framework of generative models. The VAE (Week 8) is fundamentally a Bayesian model.

### 1.3 Random Variables

A **random variable** $X$ is a function from outcomes to numbers: $X: \Omega \to \mathbb{R}$. We never work with sample spaces directly — random variables are how we translate probability into calculus.

**Discrete random variables** take countably many values. Their distribution is described by a **probability mass function (PMF)**:

$$
p_X(x) = P(X = x)
$$

**Continuous random variables** take values in intervals. Their distribution is described by a **probability density function (PDF)** $f\_X(x)$ such that:

$$
P(a \leq X \leq b) = \int_a^b f_X(x) \, dx
$$

A subtlety worth noting: $f\_X(x)$ is NOT a probability. It's a density. It can be greater than 1. What must integrate to 1 is $\int\_{-\infty}^{\infty} f\_X(x) \, dx = 1$.

### 1.4 Joint, Marginal, and Conditional Distributions

For two random variables $X, Y$:

- **Joint distribution:** $p(x, y) = P(X = x, Y = y)$ (or joint density for continuous)
- **Marginal distribution:** $p(x) = \sum\_y p(x, y)$ (or $\int p(x, y) \, dy$ for continuous)
- **Conditional distribution:** $p(y|x) = \frac{p(x, y)}{p(x)}$

The relationship $p(x, y) = p(y|x) \cdot p(x) = p(x|y) \cdot p(y)$ is the **chain rule of probability**, and it generalizes to any number of variables:

$$
p(x_1, x_2, \ldots, x_n) = p(x_1) \cdot p(x_2|x_1) \cdot p(x_3|x_1, x_2) \cdots p(x_n|x_1, \ldots, x_{n-1})
$$

This factorization is the backbone of autoregressive models (which generate data one element at a time, each conditioned on the previous ones).

**Independence:** $X$ and $Y$ are independent if $p(x, y) = p(x) \cdot p(y)$ for all $x, y$. Equivalently, knowing $X$ tells you nothing about $Y$. The i.i.d. assumption (independent and identically distributed) in most ML is that the training data points are drawn independently from the same distribution.

---

## 2. Key Distributions

### 2.1 Bernoulli and Binomial

The **Bernoulli distribution** models a single coin flip with success probability $p$:

$$
X \sim \text{Bernoulli}(p): \quad P(X = 1) = p, \quad P(X = 0) = 1 - p
$$

Properties: $\mathbb{E}[X] = p$, $\text{Var}(X) = p(1-p)$.

The **Binomial distribution** counts successes in $n$ independent Bernoulli trials:

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

These appear in binary classification: the probability a neural network assigns to the positive class is a Bernoulli parameter, and the cross-entropy loss (Section 4.3) is the negative log-likelihood of a Bernoulli model.

### 2.2 The Gaussian Distribution

The **univariate Gaussian** (normal) distribution with mean $\mu$ and variance $\sigma^2$:

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

We write $X \sim \mathcal{N}(\mu, \sigma^2)$. The standard normal has $\mu = 0$, $\sigma^2 = 1$.

**Why Gaussians are everywhere:** The **Central Limit Theorem** says that the sum of many independent random variables (with finite variance) converges to a Gaussian, regardless of their individual distributions. Formally: if $X\_1, \ldots, X\_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$, then:

$$
\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

Since many natural quantities are the result of many small additive effects, Gaussian distributions arise constantly. The CLT also explains why batch averages of stochastic gradients (Week 1, Section 5.4) become more Gaussian as batch size increases.

### 2.3 The Multivariate Gaussian

This is the distribution you need to understand deeply for this course. The **multivariate Gaussian** in $\mathbb{R}^d$ with mean vector $\boldsymbol{\mu} \in \mathbb{R}^d$ and covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$:

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

We write $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$.

Let's unpack the exponent. The quantity $(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})$ is called the **Mahalanobis distance** from $\mathbf{x}$ to $\boldsymbol{\mu}$. It generalizes Euclidean distance by accounting for the covariance structure: directions with high variance contribute less to the distance (because points naturally spread more in those directions). Contours of constant density (constant Mahalanobis distance) are ellipsoids.

**The covariance matrix** $\Sigma$ is symmetric and positive semi-definite. Its entries are $\Sigma\_{ij} = \text{Cov}(X\_i, X\_j)$. The diagonal entries are the variances; the off-diagonal entries measure how pairs of variables co-vary.

**Geometric interpretation:** The covariance matrix defines an ellipsoid. Its eigenvectors point along the principal axes of the ellipsoid, and its eigenvalues are the squared lengths of those axes. Here's the connection to Week 1: the spectral decomposition $\Sigma = Q\Lambda Q^T$ gives you the natural coordinate system for the distribution. In the eigenbasis, the variables are uncorrelated and the Gaussian factors into independent 1D Gaussians.

**Concrete example in $\mathbb{R}^2$:**

$$
\boldsymbol{\mu} = \begin{pmatrix} 1 \\\\ 2 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} 4 & 1.5 \\\\ 1.5 & 1 \end{pmatrix}
$$

The eigenvalues of $\Sigma$ are approximately $\lambda\_1 \approx 4.37$ and $\lambda\_2 \approx 0.63$. The corresponding eigenvectors point roughly along the directions $(0.97, 0.24)$ and $(-0.24, 0.97)$. This means the distribution is stretched along a direction close to the $x$-axis (high variance) and compressed nearly along the $y$-axis (low variance). The contours of constant probability are tilted ellipses.

**The precision matrix** $\Lambda = \Sigma^{-1}$ is sometimes more natural to work with. It appears directly in the exponent of the Gaussian PDF. Zero entries in $\Lambda$ indicate **conditional** independence between variables — a fact that's important for graphical models but won't be central to this course.

### 2.4 Properties of the Multivariate Gaussian

The multivariate Gaussian has remarkable algebraic closure properties:

1. **Marginals are Gaussian.** If $(X, Y)^T \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$, then $X$ alone is Gaussian with mean $\mu\_X$ and variance $\Sigma\_{XX}$.
2. **Conditionals are Gaussian.** $X | Y = y$ is also Gaussian (with a mean that's a linear function of $y$). Specifically, $\mathbb{E}[X|Y=y] = \mu\_X + \Sigma\_{XY}\Sigma\_{YY}^{-1}(y - \mu\_Y)$.
3. **Linear transformations preserve Gaussianity.** If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ and $A$ is a matrix, then $A\mathbf{X} \sim \mathcal{N}(A\boldsymbol{\mu}, A\Sigma A^T)$.
4. **Sums of independent Gaussians are Gaussian.** If $X \sim \mathcal{N}(\mu\_1, \sigma\_1^2)$ and $Y \sim \mathcal{N}(\mu\_2, \sigma\_2^2)$ independently, then $X + Y \sim \mathcal{N}(\mu\_1 + \mu\_2, \sigma\_1^2 + \sigma\_2^2)$.

Property 3 is why Gaussians and linear algebra are so tightly linked. A neural network layer $\mathbf{h} = W\mathbf{x} + \mathbf{b}$ with Gaussian input produces Gaussian output (before the activation function). The activation function is precisely what breaks this Gaussianity and allows the network to learn non-Gaussian representations.

### 2.5 Other Important Distributions

A few other distributions that appear in this course:

**Categorical distribution:** Generalization of Bernoulli to $K$ outcomes. $P(X = k) = \pi\_k$ with $\sum\_k \pi\_k = 1$. This is the distribution over class labels in classification.

**Uniform distribution:** $f(x) = \frac{1}{b-a}$ for $x \in [a, b]$. Maximum entropy distribution on a bounded interval (see Section 5.1). Used for initialization and as a "non-informative" prior.

**Laplace distribution:** $f(x) = \frac{1}{2b}\exp(-|x-\mu|/b)$. Like a Gaussian but with heavier tails and a sharp peak at $\mu$. The connection to $L\_1$ regularization: a Laplace prior on model weights corresponds to $L\_1$ penalty (Section 6.2).

---

## 3. Expectation and Variance

### 3.1 Expectation

The **expected value** of a random variable is its probability-weighted average:

$$
\mathbb{E}[X] = \sum_x x \cdot p(x) \quad \text{(discrete)}, \qquad \mathbb{E}[X] = \int x \cdot f(x) \, dx \quad \text{(continuous)}
$$

More generally, for a function $g(X)$:

$$
\mathbb{E}[g(X)] = \sum_x g(x) \cdot p(x) \quad \text{(LOTUS: Law of the Unconscious Statistician)}
$$

The crucial property is **linearity of expectation**:

$$
\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]
$$

This holds even when $X$ and $Y$ are *not* independent. It's one of the most powerful tools in probability, and we'll use it constantly.

**Concrete example.** Roll two dice and let $S$ be their sum. By linearity: $\mathbb{E}[S] = \mathbb{E}[D\_1] + \mathbb{E}[D\_2] = 3.5 + 3.5 = 7$. We didn't need to enumerate all 36 outcomes and their probabilities. This simplicity is why linearity of expectation matters.

### 3.2 Variance and Covariance

The **variance** measures spread:

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

The second form is usually easier to compute. Note that $\text{Var}(aX + b) = a^2\text{Var}(X)$ — variance scales with the square of the coefficient (and is unaffected by additive constants).

The **covariance** measures how two variables move together:

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

If $X$ and $Y$ are independent, $\text{Cov}(X, Y) = 0$. The converse is false: zero covariance does not imply independence (it only means no *linear* relationship).

The **correlation** normalizes covariance to $[-1, 1]$:

$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}
$$

For a random vector $\mathbf{X} = (X\_1, \ldots, X\_d)^T$, the **covariance matrix** is the $d \times d$ matrix:

$$
\Sigma = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T], \quad \Sigma_{ij} = \text{Cov}(X_i, X_j)
$$

This is the same covariance matrix that parameterizes the multivariate Gaussian. Even for non-Gaussian random vectors, the covariance matrix captures the second-order statistical structure of the data.

### 3.3 The Law of Large Numbers

**Theorem (Weak Law of Large Numbers).** If $X\_1, X\_2, \ldots$ are i.i.d. with mean $\mu$, then the sample mean converges in probability to $\mu$:

$$
\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{p} \mu \quad \text{as } n \to \infty
$$

This is why empirical averages work: if you compute the average loss over a large batch of data, it approximates the expected loss. SGD works because of the law of large numbers — even a mini-batch average is close to the true gradient in expectation.

**Quantitative version:** Chebyshev's inequality gives $P(|\bar{X}\_n - \mu| \geq \epsilon) \leq \frac{\sigma^2}{n\epsilon^2}$. The variance of the sample mean decreases as $1/n$, which is why larger batches give more accurate gradient estimates.

### 3.4 Jensen's Inequality

**Theorem (Jensen's Inequality).** If $\phi$ is a convex function, then:

$$
\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)]
$$

For concave $\psi$: $\psi(\mathbb{E}[X]) \geq \mathbb{E}[\psi(X)]$.

This inequality is extremely useful in information theory and machine learning. We'll use it in Section 5.3 to prove that KL divergence is non-negative, and in Week 8 to derive the ELBO for VAEs.

**Concrete example.** Since $\phi(x) = x^2$ is convex: $(\mathbb{E}[X])^2 \leq \mathbb{E}[X^2]$. This is exactly the statement that $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \geq 0$.

Since $\psi(x) = \log(x)$ is concave: $\log(\mathbb{E}[X]) \geq \mathbb{E}[\log(X)]$. This shows up when deriving the EM algorithm.

---

## 4. Maximum Likelihood Estimation

### 4.1 The Principle

You have data $\mathbf{x}\_1, \ldots, \mathbf{x}\_N$ that you believe came from some distribution $p(\mathbf{x}; \theta)$ parameterized by $\theta$. What's the "best" value of $\theta$?

The **maximum likelihood estimator (MLE)** says: choose the $\theta$ that makes the observed data most probable.

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^N p(\mathbf{x}_i; \theta)
$$

The product assumes the data points are independent. Taking the log (which doesn't change the argmax because log is monotonic):

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log p(\mathbf{x}_i; \theta)
$$

This is the **log-likelihood**. We maximize the log-likelihood, or equivalently minimize the **negative log-likelihood (NLL)**:

$$
\hat{\theta}_{\text{MLE}} = \arg\min_\theta \left[-\sum_{i=1}^N \log p(\mathbf{x}_i; \theta)\right]
$$

The log transform has a practical benefit: products of small probabilities quickly underflow to zero in floating-point arithmetic. Log-probabilities are sums, which are numerically stable. This is why we always work with log-likelihoods in practice.

### 4.2 MLE for the Gaussian

Suppose $x\_1, \ldots, x\_N \sim \mathcal{N}(\mu, \sigma^2)$ i.i.d. The log-likelihood is:

$$
\ell(\mu, \sigma^2) = -\frac{N}{2}\log(2\pi) - \frac{N}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (x_i - \mu)^2
$$

Setting $\frac{\partial \ell}{\partial \mu} = 0$:

$$
\hat{\mu} = \frac{1}{N}\sum_{i=1}^N x_i = \bar{x}
$$

Setting $\frac{\partial \ell}{\partial \sigma^2} = 0$:

$$
\hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^2
$$

The MLE for the mean is the sample mean; the MLE for the variance is the (biased) sample variance. Satisfying and unsurprising — but the derivation method generalizes to far more complex models.

### 4.3 MLE and Loss Functions

Here's one of the most important connections in machine learning:

> **Minimizing a loss function is (often) maximizing a likelihood.**

**Example: MSE loss is Gaussian MLE.** Suppose we model $y\_i = f\_\theta(\mathbf{x}\_i) + \epsilon\_i$ where $\epsilon\_i \sim \mathcal{N}(0, \sigma^2)$. Then $p(y\_i | \mathbf{x}\_i; \theta) = \mathcal{N}(f\_\theta(\mathbf{x}\_i), \sigma^2)$. The negative log-likelihood is:

$$
-\ell(\theta) = \frac{N}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - f_\theta(\mathbf{x}_i))^2
$$

Minimizing over $\theta$ (with $\sigma$ fixed) is equivalent to minimizing $\sum\_i (y\_i - f\_\theta(\mathbf{x}\_i))^2$ — the **mean squared error**. So when you train a neural network with MSE loss, you're implicitly assuming Gaussian noise.

**Example: Cross-entropy loss is Bernoulli MLE.** For binary classification, $y\_i \in \lbrace 0, 1\rbrace $ and the model outputs $\hat{p}\_i = \sigma(f\_\theta(\mathbf{x}\_i))$ (sigmoid of the logit). Then $p(y\_i | \mathbf{x}\_i; \theta) = \hat{p}\_i^{y\_i}(1-\hat{p}\_i)^{1-y\_i}$, and the negative log-likelihood is:

$$
-\ell(\theta) = -\sum_{i=1}^N \left[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right]
$$

This is the **binary cross-entropy loss**. When you train a classifier with cross-entropy, you're doing maximum likelihood.

**Example: Categorical cross-entropy is Categorical MLE.** For multi-class classification with $K$ classes, the model outputs $\hat{\mathbf{p}}\_i = \text{softmax}(f\_\theta(\mathbf{x}\_i))$ and the negative log-likelihood is:

$$
-\ell(\theta) = -\sum_{i=1}^N \sum_{k=1}^K y_{ik}\log\hat{p}_{ik}
$$

where $y\_{ik} = 1$ if sample $i$ belongs to class $k$. This is categorical cross-entropy.

### 4.4 Properties of MLE

**Consistency:** As $N \to \infty$, the MLE converges to the true parameter value (under regularity conditions). This is reassuring: with enough data, MLE finds the truth.

**Asymptotic efficiency:** The MLE achieves the lowest possible variance among consistent estimators (the Cramer-Rao bound), at least asymptotically. No estimator can do consistently better.

**Potential overfitting:** With finite data, the MLE can overfit — it finds the parameters that best explain the training data, which may not generalize. This motivates regularization (Section 6).

---

## 5. Information Theory

### 5.1 Entropy: Measuring Surprise

The **entropy** of a discrete random variable $X$ with PMF $p$ is:

$$
H(X) = -\sum_x p(x) \log p(x)
$$

(We use natural logarithm unless stated otherwise; with $\log\_2$, entropy is in bits.)

The quantity $-\log p(x)$ is the **surprise** or **self-information** of outcome $x$: rare events are more surprising than common ones. Entropy is the **expected surprise** — the average amount of "new information" you get from observing $X$.

**Concrete examples:**

- A fair coin: $H = -\frac{1}{2}\log\frac{1}{2} - \frac{1}{2}\log\frac{1}{2} = \log 2 \approx 0.693$ nats
- A biased coin ($p = 0.99$): $H = -0.99\log(0.99) - 0.01\log(0.01) \approx 0.056$ nats
- A certain outcome ($p = 1$): $H = 0$

Entropy is maximized when the distribution is uniform (maximum uncertainty) and minimized when it's deterministic (no uncertainty). For $n$ outcomes, $0 \leq H(X) \leq \log n$.

**Connection to coding theory:** Shannon's source coding theorem says that the entropy $H(X)$ is the minimum average number of bits (or nats) needed to encode samples from $X$ losslessly. You can't compress below the entropy — it's a fundamental limit. This is why entropy measures "irreducible randomness."

**For continuous random variables**, we use **differential entropy**:

$$
h(X) = -\int f(x) \log f(x) \, dx
$$

The differential entropy of a Gaussian $\mathcal{N}(\mu, \sigma^2)$ is $h(X) = \frac{1}{2}\log(2\pi e \sigma^2)$. Among all continuous distributions with a given variance, the Gaussian has the *maximum* entropy. This is another reason Gaussians are natural defaults — they make the fewest assumptions beyond the specified mean and variance.

Note: differential entropy can be negative (e.g., for a uniform distribution on $[0, 0.1]$: $h = \log(0.1) < 0$). This is unlike discrete entropy, which is always non-negative.

### 5.2 Cross-Entropy

The **cross-entropy** between distributions $p$ (the "true" distribution) and $q$ (our model) is:

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

or in the continuous case: $H(p, q) = -\int p(x) \log q(x) \, dx$.

Cross-entropy measures the average number of nats needed to encode samples from $p$ using a code optimized for $q$. If $q = p$, this reduces to the entropy $H(p)$ — the optimal code. If $q \neq p$, you need more nats: $H(p, q) \geq H(p)$.

**This is the cross-entropy loss in machine learning.** When we minimize cross-entropy between the true labels $p$ and our model's predictions $q$, we're finding the model that most efficiently encodes the training data. The model that assigns highest probability to the data is the same model that requires the fewest nats to encode it.

### 5.3 KL Divergence

The **Kullback-Leibler divergence** from $q$ to $p$ (or "KL divergence of $q$ from $p$") is:

$$
D_{\text{KL}}(p \Vert  q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]
$$

Equivalently: $D\_{\text{KL}}(p \Vert  q) = H(p, q) - H(p)$. It's the "extra cost" of using $q$ instead of $p$.

**Key properties:**

1. **Non-negativity (Gibbs' inequality):** $D\_{\text{KL}}(p \Vert  q) \geq 0$, with equality iff $p = q$ a.e.
2. **Asymmetry:** $D\_{\text{KL}}(p \Vert  q) \neq D\_{\text{KL}}(q \Vert  p)$ in general. KL divergence is NOT a distance metric.
3. **Not a metric:** It violates both symmetry and the triangle inequality.

**The non-negativity proof** is elegant and worth knowing:

$$
D_{\text{KL}}(p \Vert  q) = -\mathbb{E}_p\left[\log\frac{q(x)}{p(x)}\right] \geq -\log\mathbb{E}_p\left[\frac{q(x)}{p(x)}\right] = -\log\sum_x p(x)\frac{q(x)}{p(x)} = -\log\sum_x q(x) = -\log 1 = 0
$$

The inequality is Jensen's inequality applied to the concave function $\log$. Equality holds iff $q(x)/p(x)$ is constant, i.e., $p = q$.

**The asymmetry matters.** $D\_{\text{KL}}(p \Vert  q)$ penalizes $q$ heavily where $p$ has mass but $q$ doesn't (because $\log(p/q)$ is large when $q$ is small but $p$ is not). The reverse $D\_{\text{KL}}(q \Vert  p)$ penalizes $q$ for having mass where $p$ doesn't. In variational inference:
- Minimizing $D\_{\text{KL}}(q \Vert  p)$ (the "reverse KL") tends to make $q$ zero-seeking — it avoids placing mass where $p$ is small, even if this means missing some modes of $p$.
- Minimizing $D\_{\text{KL}}(p \Vert  q)$ (the "forward KL") tends to make $q$ mass-covering — it tries to have mass everywhere $p$ does, even if this means placing mass where $p$ doesn't.

This distinction is important for understanding why VAEs sometimes produce "blurry" outputs (they use reverse KL, which mode-covers).

Despite not being a distance, KL divergence is the workhorse of probabilistic machine learning. It appears in:
- **Variational inference** (Week 8): We minimize $D\_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \Vert  p(\mathbf{z}|\mathbf{x}))$ to approximate the posterior
- **Sparse autoencoders** (Week 10): The KL divergence between the actual activation distribution and a target sparse distribution serves as a sparsity penalty

### 5.4 KL Divergence Between Gaussians

A result we'll use repeatedly. For two univariate Gaussians $p = \mathcal{N}(\mu\_1, \sigma\_1^2)$ and $q = \mathcal{N}(\mu\_2, \sigma\_2^2)$:

$$
D_{\text{KL}}(p \Vert  q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

For multivariate Gaussians $p = \mathcal{N}(\boldsymbol{\mu}\_1, \Sigma\_1)$ and $q = \mathcal{N}(\boldsymbol{\mu}\_2, \Sigma\_2)$:

$$
D_{\text{KL}}(p \Vert  q) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T\Sigma_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)\right]
$$

When $q = \mathcal{N}(\mathbf{0}, I)$ (the standard normal), this simplifies beautifully:

$$
D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu}, \Sigma) \Vert  \mathcal{N}(\mathbf{0}, I)) = \frac{1}{2}\left[-\log|\Sigma| - d + \text{tr}(\Sigma) + \boldsymbol{\mu}^T\boldsymbol{\mu}\right]
$$

For diagonal $\Sigma = \text{diag}(\sigma\_1^2, \ldots, \sigma\_d^2)$, this becomes:

$$
D_{\text{KL}} = \frac{1}{2}\sum_{j=1}^d \left[-\log\sigma_j^2 - 1 + \sigma_j^2 + \mu_j^2\right]
$$

This exact formula appears in the VAE loss function (Week 8). Memorize it — or better, derive it yourself so you can re-derive it when you need it.

### 5.5 Mutual Information

The **mutual information** between $X$ and $Y$ is:

$$
I(X; Y) = D_{\text{KL}}(p(x, y) \Vert  p(x)p(y)) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$

Equivalently: $I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X, Y)$. Mutual information measures how much knowing one variable tells you about the other. It's zero iff $X$ and $Y$ are independent.

Unlike correlation, mutual information captures *any* kind of dependency — not just linear. Unlike KL divergence, mutual information is symmetric: $I(X; Y) = I(Y; X)$.

**Connection to representation learning:** A good representation $\mathbf{Z}$ of input $\mathbf{X}$ should have high mutual information $I(\mathbf{X}; \mathbf{Z})$ — the representation should preserve information about the input. But we also want the representation to be compact. This trade-off — maximize information while minimizing complexity — is formalized by the **information bottleneck** principle, which provides a theoretical foundation for autoencoders.

### 5.6 The Relationship Between Entropy, Cross-Entropy, and KL Divergence

These three quantities are tightly linked:

$$
H(p, q) = H(p) + D_{\text{KL}}(p \Vert  q)
$$

Since $H(p)$ is fixed (it depends only on the true distribution, not on our model), minimizing cross-entropy $H(p, q)$ over $q$ is equivalent to minimizing $D\_{\text{KL}}(p \Vert  q)$ over $q$. This is why cross-entropy and KL divergence are nearly interchangeable as loss functions.

**Diagram of relationships:**

```
Cross-Entropy H(p,q) = Entropy H(p) + KL Divergence D_KL(p||q)
       |                    |                      |
  "total cost of         "inherent           "extra cost from
   encoding with q"      randomness"          using wrong model"
```

**Venn diagram interpretation of mutual information:**

```
  H(X)     H(Y)
 /    \   /    \
|      | |      |
|  H(X|Y) I(X;Y) H(Y|X) |
|      | |      |
 \    /   \    /
    H(X,Y)
```

$I(X;Y)$ is the overlap between $H(X)$ and $H(Y)$. The joint entropy $H(X,Y) = H(X) + H(Y) - I(X;Y)$.

---

## 6. Connecting to Machine Learning

### 6.1 The Big Picture: Training = Minimizing Divergence

When we train a model with maximum likelihood, we're solving:

$$
\hat{\theta} = \arg\min_\theta D_{\text{KL}}(p_{\text{data}} \Vert  p_\theta)
$$

where $p\_{\text{data}}$ is the empirical distribution of the training data and $p\_\theta$ is the model distribution. This is equivalent to minimizing cross-entropy, which is equivalent to minimizing negative log-likelihood.

All three perspectives — MLE, cross-entropy minimization, KL minimization — are the same optimization problem. But having all three perspectives is valuable because they suggest different generalizations:

- The MLE perspective suggests looking at MAP estimation (add a prior = regularize)
- The cross-entropy perspective connects to coding theory (efficiency of representation)
- The KL perspective connects to variational inference (approximate intractable posteriors)

### 6.2 MAP Estimation: A Bridge to Bayesian Thinking

**Maximum a posteriori (MAP) estimation** adds a prior $p(\theta)$ to the MLE objective:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\log p(\theta) + \sum_{i=1}^N \log p(\mathbf{x}_i; \theta)\right]
$$

The prior $\log p(\theta)$ is a regularization term:

- **Gaussian prior** on $\theta$ ($\theta \sim \mathcal{N}(0, \tau^2 I)$) gives $\log p(\theta) = -\frac{\Vert \theta\Vert \_2^2}{2\tau^2} + \text{const}$, which is **$L\_2$ regularization** (weight decay).
- **Laplace prior** on $\theta$ ($\theta \sim \text{Laplace}(0, b)$) gives $\log p(\theta) = -\frac{\Vert \theta\Vert \_1}{b} + \text{const}$, which is **$L\_1$ regularization** (LASSO).

This is a beautiful connection: the geometry of the $L\_1$ ball from Week 1 has a probabilistic interpretation. The "diamond constraint" is the Laplace prior. Sparsity promotion via $L\_1$ regularization is Bayesian inference with a prior that concentrates mass at zero. The $\lambda$ hyperparameter in regularization corresponds to the precision (inverse scale) of the prior.

### 6.3 A Taste of Variational Inference

In many models (including VAEs), the posterior $p(\theta | \mathbf{x})$ is intractable — we can't compute it exactly because the marginal likelihood $p(\mathbf{x}) = \int p(\mathbf{x}|\theta)p(\theta)d\theta$ involves an intractable integral. **Variational inference** approximates the posterior with a simpler distribution $q(\theta)$ by minimizing:

$$
D_{\text{KL}}(q(\theta) \Vert  p(\theta | \mathbf{x}))
$$

This KL divergence is also intractable (it involves the posterior we can't compute). But through algebraic manipulation, we can derive a tractable lower bound on the log-likelihood — the **ELBO (Evidence Lower Bound)**:

$$
\log p(\mathbf{x}) \geq \mathbb{E}_{q(\theta)}[\log p(\mathbf{x} | \theta)] - D_{\text{KL}}(q(\theta) \Vert  p(\theta))
$$

We'll derive this properly in Week 8 when we study VAEs. For now, just note the structure: it's a **reconstruction term** (how well the model explains the data) minus a **KL term** (how much the approximate posterior deviates from the prior). This trade-off between reconstruction and regularization is the fundamental tension in autoencoder design.

### 6.4 The KL Divergence as a Sparsity Penalty

Here's a preview of Week 10. In a sparse autoencoder, we want each hidden neuron to activate infrequently. Let $\hat{\rho}\_j$ be the average activation of neuron $j$ over the training data, and let $\rho$ be a small target activation level (e.g., $\rho = 0.05$).

We can penalize non-sparse activations using the KL divergence between two Bernoulli distributions:

$$
\sum_j D_{\text{KL}}(\text{Bernoulli}(\rho) \Vert  \text{Bernoulli}(\hat{\rho}_j))
$$

$$
= \sum_j \left[\rho \log\frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j}\right]
$$

This is zero when $\hat{\rho}\_j = \rho$ and increases as $\hat{\rho}\_j$ deviates from $\rho$ in either direction. It's a smooth, differentiable sparsity penalty that we can add to the reconstruction loss and optimize with gradient descent. The entire information-theoretic framework of this week is what makes this penalty principled rather than ad hoc.

**Why KL rather than just $(hat{\rho}\_j - \rho)^2$?** The KL penalty is asymmetric in the right way: it penalizes a neuron that's always active ($\hat{\rho}\_j = 0.95$) much more heavily than one that's just slightly too active ($\hat{\rho}\_j = 0.07$). This matches our intuition that widespread activation is a bigger problem than marginal over-activation.

---

## Summary

| Concept | Key Idea | Where It Shows Up Later |
|---------|----------|------------------------|
| Bayes' theorem | Prior + likelihood $\to$ posterior | Bayesian perspective on regularization, VAEs (Week 8) |
| Multivariate Gaussian | Covariance = ellipsoid; eigenvalues = axis lengths | PCA (Week 5), VAE latent space (Week 8) |
| Maximum likelihood | Choose parameters that make data most probable | All training losses are (disguised) NLLs |
| Entropy | Expected surprise; maximum for uniform distributions | Information-theoretic sparsity measures |
| KL divergence | "Extra cost" of using $q$ instead of $p$; non-negative, asymmetric | VAE loss (Week 8), sparsity penalty (Week 10) |
| Cross-entropy | $H(p) + D\_{\text{KL}}(p\Vert q)$; equals NLL | The standard classification loss function |
| Mutual information | Shared information between variables; captures nonlinear dependence | Information bottleneck, representation quality |
| MAP = regularized MLE | Gaussian prior = $L\_2$; Laplace prior = $L\_1$ | Connecting Bayesian and optimization perspectives |

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $P(A)$ | Probability of event $A$ |
| $p(x)$, $f(x)$ | PMF or PDF |
| $\mathbb{E}[X]$ | Expected value of $X$ |
| $\text{Var}(X)$ | Variance of $X$ |
| $\text{Cov}(X, Y)$ | Covariance of $X$ and $Y$ |
| $\Sigma$ | Covariance matrix |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian distribution |
| $H(X)$ | Entropy of $X$ |
| $H(p, q)$ | Cross-entropy between $p$ and $q$ |
| $D\_{\text{KL}}(p \Vert  q)$ | KL divergence from $q$ to $p$ |
| $I(X; Y)$ | Mutual information between $X$ and $Y$ |
| $\hat{\theta}\_{\text{MLE}}$ | Maximum likelihood estimator |

---

## Further Reading

- **Goodfellow et al.** *Deep Learning*, Chapter 3 (Probability and Information Theory). Available free at deeplearningbook.org. Concise and machine-learning-focused.
- **Cover, T. and Thomas, J.** *Elements of Information Theory*, 2nd ed. The definitive reference on entropy, KL divergence, and mutual information. Chapters 2-4 cover everything in Section 5 of these notes.
- **Bishop, C.** *Pattern Recognition and Machine Learning*, Chapters 1-2. Excellent treatment of MLE, Bayesian inference, and the Gaussian distribution.
- **MacKay, D.** *Information Theory, Inference, and Learning Algorithms*. Free online. Beautifully written, with great exercises. Chapters 2-4 for probability, Chapters 28-33 for information theory connections to learning.
- **3Blue1Brown.** "But what is a convolution?" and "Bayes theorem, the geometry of changing beliefs" (YouTube). Good visual introductions.
