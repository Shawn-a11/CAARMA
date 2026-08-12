# Title

Real-Gradient-Aligned Reuse of Persistent Synthetic Speaker Classes

## Motivation

Persistent CAARMA currently decides which synthetic speaker class to revisit using visit counts or proxy utility signals such as uncertainty and prototype geometry. The completed Boundary Utility and Fisher-UCB experiments show that geometrically ambiguous or classifier-uncertain synthetic classes are not necessarily useful for real speaker verification. The missing quantity is the effect of a synthetic-class update on the real-speaker objective.

## Method

1. Retain E3 candidate construction: natural-cluster plus within-cluster cosine top-4, parent-SLERP initialization, persistent `W_syn[c]`, joint `L_syn`, and Projection-D.
2. Keep the CRP probability of creating a new class unchanged.
3. In a small shared parameter subspace, compute the real-speaker gradient `g_real` and synthetic-class gradient `g_c`.
4. Define class utility by normalized gradient alignment:

   $$
   a_c=\frac{\langle g_{\mathrm{real}},g_c\rangle}
   {\lVert g_{\mathrm{real}}\rVert_2\lVert g_c\rVert_2+\epsilon}.
   $$

5. Maintain the mean positive alignment `bar_a_c` and reuse existing class `c` with probability proportional to `n_c bar_a_c`.
6. Validate the mechanism with shuffled-utility and bottom-alignment controls under matched class creation, ratio, and compute.

## Falsifiable Claim

If synthetic-class gradient alignment is a valid utility signal, it should predict one-step reduction of real-speaker loss, and benefit-weighted reuse should outperform popularity reuse while shuffled or anti-aligned reuse should not.

