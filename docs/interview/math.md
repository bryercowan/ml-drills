# The Math They Might Ask

You don't need to derive anything. You need intuition for why things work.

SOFTMAX: exp(x_i) / sum(exp(x_j)). Maps any vector to a probability
distribution. Numerical trick: subtract max(x) first to prevent overflow.
Gradient: softmax * (1 - softmax) for the correct class — saturates
when confident, which can slow learning.

MATRIX MULTIPLICATION AS LINEAR TRANSFORMATION: Wx takes input x and
maps it to a new space. The columns of W define the basis of the output
space. Rank(W) = dimensionality of that output space. LoRA says: the
UPDATE to W is low-rank, meaning the adaptation lives in a small subspace.

EIGENVALUES: For square matrix A, Ax = lambda*x means x is stretched by
factor lambda in direction x. The eigenvalue spectrum tells you about
conditioning — if max(|lambda|)/min(|lambda|) is large, optimization is
hard (the loss landscape is an elongated valley, gradient descent
oscillates).

SVD: Any matrix W = U * Sigma * V^T where Sigma is diagonal with singular
values. Keep only the top-r singular values → best rank-r approximation.
THIS IS LORA: the weight update delta_W ≈ B * A is a rank-r matrix.

LOG PROBABILITIES: We use log P instead of P because: (1) products become
sums (numerically stable), (2) maximizing log P = maximizing P (log is
monotonic), (3) cross-entropy is naturally in log space. Your SFT loss IS
negative log probability of the correct action.

BAYES' THEOREM: P(A|B) = P(B|A)*P(A)/P(B). Your model computes P(action|screenshot).
The pretrained weights encode P(action) — the prior. Fine-tuning adjusts
the likelihood P(screenshot|action) fit to your specific task.

INFORMATION THEORY INTUITION: Entropy H(P) = expected surprise = -sum P*log(P).
Cross-entropy H(P,Q) >= H(P). KL divergence = H(P,Q) - H(P) = the extra
surprise from using Q when the true distribution is P. Your SFT loss IS
cross-entropy. Your GRPO KL penalty IS KL divergence.
