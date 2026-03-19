# Week 2: Homework — Probability and Information Theory

**Instructions:** This problem set has 7 problems mixing theory (pen-and-paper derivations) and implementation (Python/NumPy). For implementation problems, write your solutions in a Jupyter notebook or `.py` file. Show your work for theoretical problems.

**Estimated time:** 4-6 hours

---

## Problem 1: MLE for the Gaussian (Theory)

Suppose you observe data $x_1, x_2, \ldots, x_N$ drawn i.i.d. from $\mathcal{N}(\mu, \sigma^2)$.

**(a)** Write down the log-likelihood $\ell(\mu, \sigma^2) = \log \prod_{i=1}^N p(x_i; \mu, \sigma^2)$.

**(b)** Derive the MLE for $\mu$ by taking $\frac{\partial \ell}{\partial \mu} = 0$ and solving. Show all steps.

**(c)** Derive the MLE for $\sigma^2$ by taking $\frac{\partial \ell}{\partial \sigma^2} = 0$ and solving. Show all steps.

**(d)** The MLE for $\sigma^2$ is $\hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^2$, which is the *biased* sample variance (divides by $N$ instead of $N-1$). Show that this estimator is biased by computing $\mathbb{E}[\hat{\sigma}^2]$ and showing it equals $\frac{N-1}{N}\sigma^2$, not $\sigma^2$.

*Hint for (d):* Use the identity $\sum_i (x_i - \bar{x})^2 = \sum_i x_i^2 - N\bar{x}^2$, then take expectations using $\mathbb{E}[x_i^2] = \sigma^2 + \mu^2$ and $\mathbb{E}[\bar{x}^2] = \frac{\sigma^2}{N} + \mu^2$.

---

## Problem 2: KL Divergence by Hand (Theory)

Consider two discrete distributions over $\{1, 2, 3, 4\}$:

| $x$ | $p(x)$ | $q(x)$ |
|-----|---------|---------|
| 1   | 0.4     | 0.25    |
| 2   | 0.3     | 0.25    |
| 3   | 0.2     | 0.25    |
| 4   | 0.1     | 0.25    |

**(a)** Compute $H(p)$, the entropy of $p$.

**(b)** Compute $H(p, q)$, the cross-entropy from $p$ to $q$.

**(c)** Compute $D_{\text{KL}}(p \| q)$ directly from the definition, and verify that it equals $H(p, q) - H(p)$.

**(d)** Compute $D_{\text{KL}}(q \| p)$. Is it equal to $D_{\text{KL}}(p \| q)$? What does this tell you about KL divergence as a "distance"?

**(e)** Now let $q$ be the uniform distribution. Without computing, argue why $D_{\text{KL}}(p \| q_{\text{uniform}})$ gives a measure of how "non-uniform" $p$ is. What value of $p$ would minimize this KL divergence?

---

## Problem 3: KL Divergence is Non-Negative (Theory — Proof)

Prove that $D_{\text{KL}}(p \| q) \geq 0$ for any distributions $p, q$, with equality iff $p = q$ almost everywhere.

**Approach:** Use **Jensen's inequality**, which states that for a convex function $\phi$:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)]$$

Or equivalently, for a concave function $\psi$:

$$\psi(\mathbb{E}[X]) \geq \mathbb{E}[\psi(X)]$$

**(a)** Write $D_{\text{KL}}(p \| q) = -\mathbb{E}_{x \sim p}\left[\log\frac{q(x)}{p(x)}\right]$.

**(b)** Apply Jensen's inequality using the fact that $\log$ is concave:

$$\mathbb{E}_{x \sim p}\left[\log\frac{q(x)}{p(x)}\right] \leq \log\left(\mathbb{E}_{x \sim p}\left[\frac{q(x)}{p(x)}\right]\right)$$

**(c)** Show that $\mathbb{E}_{x \sim p}\left[\frac{q(x)}{p(x)}\right] = 1$.

**(d)** Combine (a), (b), and (c) to conclude that $D_{\text{KL}}(p \| q) \geq 0$.

**(e)** When does equality hold in Jensen's inequality? Use this to argue that $D_{\text{KL}}(p \| q) = 0$ iff $p = q$ a.e.

---

## Problem 4: Cross-Entropy and MLE (Theory)

This problem makes the connection between cross-entropy loss and maximum likelihood concrete.

**(a)** Suppose we have a classification problem with $K$ classes. The true label for sample $i$ is a one-hot vector $\mathbf{y}_i \in \{0, 1\}^K$ (with $y_{ik} = 1$ for the true class $k$). Our model outputs a probability vector $\hat{\mathbf{y}}_i = \text{softmax}(f_\theta(\mathbf{x}_i))$. Write down the negative log-likelihood of the data under this model.

**(b)** Show that the negative log-likelihood from (a) equals:

$$-\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log \hat{y}_{ik}$$

which is the sum of cross-entropies $H(\mathbf{y}_i, \hat{\mathbf{y}}_i)$ over the training set.

**(c)** Explain in 2-3 sentences why cross-entropy loss gives much larger gradients than MSE loss when the model's predicted probability is very wrong (e.g., the model assigns probability 0.01 to the correct class). Hint: compare $-\log(0.01)$ with $(1 - 0.01)^2$.

---

## Problem 5: Entropy and Mutual Information (Theory + Computation)

Consider two jointly distributed binary random variables $X$ and $Y$ with the following joint PMF:

| | $Y = 0$ | $Y = 1$ |
|---|---------|---------|
| $X = 0$ | 0.3 | 0.1 |
| $X = 1$ | 0.2 | 0.4 |

**(a)** Compute the marginal distributions $p(x)$ and $p(y)$.

**(b)** Compute $H(X)$, $H(Y)$, and $H(X, Y)$.

**(c)** Compute the conditional entropy $H(Y|X) = \sum_x p(x) H(Y|X=x)$. Verify that $H(X, Y) = H(X) + H(Y|X)$ (chain rule for entropy).

**(d)** Compute the mutual information $I(X; Y)$ using:
1. $I(X; Y) = H(X) - H(X|Y)$
2. $I(X; Y) = H(X) + H(Y) - H(X, Y)$

Verify both give the same answer.

**(e)** Are $X$ and $Y$ independent? How does the mutual information confirm your answer?

---

## Problem 6: KL Divergence Between Gaussians (Theory + Implementation)

**(a)** Derive the KL divergence between two univariate Gaussians $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$.

Start from the definition:

$$D_{\text{KL}}(p \| q) = \mathbb{E}_{x \sim p}\left[\log\frac{p(x)}{q(x)}\right]$$

Substitute the Gaussian PDFs, expand the logarithm, and use the facts $\mathbb{E}_p[x] = \mu_1$, $\mathbb{E}_p[x^2] = \sigma_1^2 + \mu_1^2$. You should arrive at:

$$D_{\text{KL}}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

**(b) (Implementation)** Write a Python function `kl_gaussian(mu1, sigma1, mu2, sigma2)` that computes this formula. Verify your implementation by:
1. Checking that $D_{\text{KL}}(p \| p) = 0$ for several values of $\mu$ and $\sigma$.
2. Checking that $D_{\text{KL}}(p \| q) \geq 0$ for many random pairs.
3. Comparing against a numerical estimate: sample 100,000 points from $p$, compute $\frac{1}{N}\sum_{i=1}^N \log\frac{p(x_i)}{q(x_i)}$, and verify it's close to the formula.

**(c) (Implementation)** Fix $q = \mathcal{N}(0, 1)$ and plot $D_{\text{KL}}(p \| q)$ as a function of $\mu_1$ for $\sigma_1 = 1$, and as a function of $\sigma_1$ for $\mu_1 = 0$. Interpret the plots: what does the KL divergence "penalize" when comparing to a standard normal?

**(d)** In the VAE (Week 8), the loss includes $D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1))$ where $\mu$ and $\sigma$ are outputs of the encoder. Using your formula from (a), show this simplifies to:

$$D_{\text{KL}} = -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

---

## Problem 7: MLE for a Mixture of Gaussians (Implementation)

A **Gaussian mixture model (GMM)** with $K$ components has the density:

$$p(x; \theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \sigma_k^2)$$

where $\pi_k \geq 0$, $\sum_k \pi_k = 1$ are the mixing weights.

Direct MLE for a GMM doesn't have a closed-form solution (the log of a sum is intractable). The **Expectation-Maximization (EM) algorithm** provides an iterative solution.

**(a)** Generate synthetic data from a mixture of 3 Gaussians:
```python
import numpy as np

np.random.seed(42)
N = 1000
# True parameters
true_means = [-3.0, 0.0, 4.0]
true_stds = [0.8, 1.2, 0.5]
true_weights = [0.3, 0.5, 0.2]

# Generate data
components = np.random.choice(3, size=N, p=true_weights)
data = np.array([np.random.normal(true_means[c], true_stds[c]) for c in components])
```

Plot a histogram of the data.

**(b)** Implement the EM algorithm for a 1D GMM with $K$ components:

**E-step:** For each data point $x_i$ and component $k$, compute the "responsibility":
$$r_{ik} = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i; \mu_j, \sigma_j^2)}$$

**M-step:** Update parameters:
$$N_k = \sum_{i=1}^N r_{ik}, \quad \mu_k = \frac{1}{N_k}\sum_i r_{ik} x_i, \quad \sigma_k^2 = \frac{1}{N_k}\sum_i r_{ik}(x_i - \mu_k)^2, \quad \pi_k = \frac{N_k}{N}$$

Implement this as a function `em_gmm(data, K, num_iterations)` that returns the fitted parameters and the log-likelihood at each iteration.

**(c)** Run your EM algorithm on the data from (a) with $K = 3$. Initialize the means randomly. Plot:
1. The log-likelihood vs. iteration number (it should monotonically increase — this is a theorem, not a hope).
2. The fitted Gaussian components overlaid on the histogram.

How close are the recovered parameters to the true parameters?

**(d)** Run EM with $K = 2$ and $K = 5$ on the same data. How does the log-likelihood compare? What happens with $K = 5$ — do you see any degenerate components?

**(e)** (Conceptual) The EM algorithm is guaranteed to converge to a local maximum of the likelihood, but not necessarily the global maximum. Run your $K = 3$ fit 10 times with different random initializations. How much do the final log-likelihoods vary? What does this suggest about the importance of initialization?

---

## Submission Checklist

- [ ] Problem 1: MLE derivation for Gaussian (all four parts)
- [ ] Problem 2: KL divergence computation by hand
- [ ] Problem 3: Proof of non-negativity using Jensen's inequality
- [ ] Problem 4: Cross-entropy and MLE connection
- [ ] Problem 5: Entropy and mutual information computations
- [ ] Problem 6: Gaussian KL derivation, implementation, and plots
- [ ] Problem 7: EM algorithm implementation with plots and analysis
