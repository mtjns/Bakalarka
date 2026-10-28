# Study Roadmap: From Least Squares to Prior-Fitted Networks

The goal is to understand how PFNs relate to classical statistical inference.
Each module has a short reading pointer, the one equation you must understand,
and a small task to build intuition.

---

## Module 1 — Ordinary Least Squares

Fitting a line to data is the oldest problem in statistics, yet it contains
nearly every difficulty that appears in modern machine learning: what to
optimise, how to handle noise, and when the solution breaks down numerically.
Understanding OLS deeply — including its failure modes — is essential before
any probabilistic generalisation makes sense.

**Reading:** Murphy *Probabilistic Machine Learning* Book 1, Ch. 11.2–11.3.
Free at [probml.ai](https://probml.ai). Focus on the geometric interpretation
and the connection between ridge regression and regularisation.

**Core equation**

$$\hat{\beta} = (X^\top X)^{-1} X^\top y$$

Understand what goes wrong when $X^\top X$ is ill-conditioned: small eigenvalues
amplify noise in $\hat{\beta}$ because inversion magnifies directions with low
variance in the data. Ridge regression adds $\lambda I$, which lifts all
eigenvalues and stabilises the inversion at the cost of bias.

**Tasks**

1. Generate $n=20$ points in two scenarios: (a) $x$ uniformly spread over
   $[-1, 1]$; (b) $x$ clustered in a narrow band with one near-duplicate,
   introducing near-collinearity. For each, compute $\hat{\beta}$ and plot
   the eigenvalues of $X^\top X$. How does the condition number differ?
2. Apply ridge regression to the ill-conditioned case from Task 1.
   Plot the eigenvalues of $(X^\top X + \lambda I)$ as $\lambda$ increases.
   At what $\lambda$ is the condition number acceptable?
   - plot loss surface, i.e. on a grid of possible valies of w, plot loss $L=r^Tr$ of residues $r=y-Xw$ for every grid point
   - plot loss surface for the colinear $X$ and for $L=r^Tr + \lambda w^Tw$
   - see how it changes if you replave L2 norm ($L_2 = w^Tw$) by a general $L_p$ norm for $p=0.5,1,2,3$
3. *Preview question:* OLS is an algorithm that maps $(X, y) \to \hat{\beta}$.
   If you ran it on 1000 datasets from the same model, could a neural network
   learn to approximate this map without ever computing a matrix inverse?
   Write down what its inputs and outputs would be.

---

## Module 2 — Bayesian Linear Regression

OLS gives a point estimate. Bayesian regression gives a full distribution over
parameters — and consequently a distribution over predictions. This is important
practically because it quantifies uncertainty, and theoretically because it
introduces the prior as a tool for encoding beliefs before seeing data.
The posterior also reveals that ridge regression is not an ad-hoc trick but
the MAP estimate under a specific prior.

**Reading:** Rasmussen & Williams *Gaussian Processes for Machine Learning*,
Ch. 2.1. Free at [gaussianprocess.org](http://www.gaussianprocess.org/gpml).
This book will be the main reference for Modules 2–4.

**Core equation** (posterior mean)

$$\mu_{\ast} = \left(\Sigma_0^{-1} + \frac{X^\top X}{\sigma^2}\right)^{-1}
          \frac{X^\top y}{\sigma^2}$$

where $\Sigma_0$ is the prior covariance. A hierarchical prior places a
distribution over $\Sigma_0$ itself, marginalising out the regularisation
strength instead of setting it by cross-validation.

**Tasks**

1. Derive the posterior $p(\beta \mid X, y)$ for a Gaussian prior and likelihood.
   Verify that the MAP coincides with ridge regression.
2. Plot the posterior predictive $p(y_{\ast} \mid x_{\ast}, X, y)$ for several values of
   the prior variance $\Sigma_0$. What happens in the limit $\Sigma_0 \to \infty$?
3. *Question:* The posterior mean is a linear function of $y$:
   $\mu_{\ast} = A(X) y$ where $A$ depends on the data locations but not on $y$.
   If you wanted to learn this map from examples, what class of models would
   be sufficient? Does a linear model suffice, or do you need something
   nonlinear? What is the role of $X$ in the learned map?

---

## Module 3 — From Weight Space to Function Space

Bayesian linear regression operates on parameters $\beta$. But we often care
about the function $f(x)$, not the weights that produced it. Switching to
function space is not just a mathematical convenience — it allows the model
complexity to grow with the data and leads directly to Gaussian processes.
This module is the conceptual hinge of the entire roadmap.

**Reading:** R&W Ch. 2.1 (weight-space view) and the opening of Ch. 4 (kernels).
Pay close attention to how the kernel emerges from marginalising out $\beta$.

**Core equation**

$$f(x) = \phi(x)^\top \beta, \quad \beta \sim \mathcal{N}(0, \Sigma_0)
\;\Longrightarrow\;
\mathrm{Cov}[f(x), f(x')] = \phi(x)^\top \Sigma_0 \phi(x') =: k(x, x')$$

The kernel $k$ fully specifies the prior over functions — no explicit $\beta$ is needed.

**Tasks**

1. Show that the RBF kernel $k(x,x') = \exp(-\|x-x'\|^2 / 2\ell^2)$
   corresponds to an infinite-dimensional feature map.
   (Hint: Taylor expansion of the exponential or Bochner's theorem.)
2. *Question:* Why does moving to function space remove the need to ever
   specify or invert a weight-space covariance matrix?

---

## Module 4 — Gaussian Processes

A GP is a prior over functions that is closed under conditioning: given
observations, the posterior is also a GP with an analytic formula. This makes
GPs an exact, principled Bayesian model for regression — and a practical
baseline that any learned model should be compared against. The $O(n^3)$
cost of exact inference is the key limitation that motivates approximate
methods, including PFNs.

**Reading:** R&W Ch. 2.2–2.3 (posterior) and Ch. 5.4 (marginal likelihood
and hyperparameter optimisation). Implement before reading the theory — the
equations become much clearer once you have run them.

**Core equations** (posterior mean and variance)

$$\mu(x_{\ast}) = k(x_{\ast}, X) [K(X,X) + \sigma^2 I]^{-1} y$$

$$\sigma^2(x_{\ast}) = k(x_{\ast},x_{\ast}) - k(x_{\ast}, X) [K(X,X) + \sigma^2 I]^{-1} k(X, x_{\ast})$$

Use Cholesky decomposition in code; add a small jitter ($10^{-6}$) to the
diagonal for numerical stability.

**Tasks**

1. Implement GP regression from scratch (no GPy/sklearn) for 1-D data.
   Use the RBF kernel. Plot the posterior mean and ±2σ band.
2. Optimise the lengthscale $\ell$ by maximising the log marginal likelihood.
   How does $\ell$ change with denser training data?
3. *Question:* The GP posterior mean $\mu(x_{\ast})$ is a weighted sum of training
   labels: $\mu(x_{\ast}) = \sum_j \alpha_j y_j$ where $\alpha_j$ depends on
   $x_{\ast}$ and all training inputs but not on $y$. Write out $\alpha_j$
   explicitly. What property of the kernel determines how quickly $\alpha_j$
   decays as $X_j$ moves away from $x_{\ast}$?

---

## Module 5 — Attention as a Learned Kernel

Transformers are the architecture underlying PFNs, and their key operation —
attention — is structurally a kernel method. Understanding this connection
explains *why* a transformer can approximate Bayesian inference: it computes
a weighted combination of training outputs, just as a GP does, but with
weights that are learned rather than fixed by a kernel formula.

**Reading:** Vaswani et al. 2017 *Attention Is All You Need*, Sections 3.1–3.2.
Available on [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
Read only the attention mechanism — skip the translation model details.

**Core equation**

$$\hat{y}_{\ast} = \sum_j \alpha_j(x_{\ast}, X) y_j, \qquad
\alpha_j \propto \exp\!\left(\frac{q^\top k_j}{\sqrt{d}}\right)$$

GP weights are fixed by the kernel; attention weights are learned end-to-end.
Both are convex combinations of training labels.

**Tasks**

1. Write a single-head attention layer in PyTorch (~15 lines).
   Verify that the output is always a convex combination of the values $V$.
2. *Question:* Compare the GP weight $\alpha_j \propto k(x_{\ast}, X_j)$ from
   Module 4 to the attention weight $\alpha_j \propto \exp(q^\top k_j / \sqrt{d})$.
   What would $Q$ and $K$ need to learn for attention to recover the GP kernel?

---

## Module 6 — Prior-Fitted Networks

GPs are exact but $O(n^3)$. What if you could train a model once — on
samples from the prior — so that at test time inference costs a single
forward pass? This is the PFN idea: *learn the algorithm, not the answer.*
The model never sees the same dataset twice during training; instead it
learns the general structure of Bayesian posterior inference from the prior
distribution over datasets.

**Reading:** Müller et al. 2022 *Transformers Can Do Bayesian Inference*,
NeurIPS 2022. Available at [arxiv.org/abs/2112.10510](https://arxiv.org/abs/2112.10510).
Read the introduction and Section 3 first.

**Core equation** (training objective)

$$\mathcal{L}(\theta) = \mathbb{E}_{(\mathcal{D}, y_{\ast}) \sim p}
\bigl[-\log p_\theta(y_{\ast} \mid x_{\ast}, \mathcal{D})\bigr]$$

At test time $\mathcal{D} = (X, y)$ is passed as context tokens; the model
produces $p(y_{\ast} \mid x_{\ast}, \mathcal{D})$ in a single forward pass — no matrix
inversion, no retraining.

**Tasks**

1. Return to Module 1 Task 3. Train a small transformer on 10 000 synthetic
   linear datasets. At test time pass $(X, y)$ as context and $x_{\ast}$ as query.
   Check that the output matches the OLS / Bayesian posterior mean.
2. Repeat with data drawn from a GP prior with RBF kernel.
   Compare the PFN output to the exact GP posterior on held-out test points.
3. *Question:* The PFN training loss resembles a cross-entropy loss.
   What is the connection to the GP log marginal likelihood
   $\log p(y \mid X) = \log \mathcal{N}(y; 0, K + \sigma^2 I)$?
   Under what conditions are they equivalent?

---

## Key References

| Reference | Where |
|-----------|-------|
| Murphy, *Probabilistic Machine Learning* Book 1 (2022) | [probml.ai](https://probml.ai) |
| Rasmussen & Williams, *Gaussian Processes for ML* (2006) | [gaussianprocess.org](http://www.gaussianprocess.org/gpml) |
| Vaswani et al., *Attention Is All You Need* (2017) | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| Müller et al., *Transformers Can Do Bayesian Inference* (NeurIPS 2022) | [arXiv:2112.10510](https://arxiv.org/abs/2112.10510) |
