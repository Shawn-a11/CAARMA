# Paper-Aligned HuBERT Arm

This arm applies the CAARMA paper's explicit optimization statements: fixed
discriminator LR 2e-4, D-step then M-step for every batch, and unscaled BCE
sums from Equations 1-2. HuBERT is trainable because the paper describes the
discriminator as concurrently optimized and the public model leaves it
trainable.

Unresolved paper details remain explicit: the exact lambda-adv control law and
whether batch size 50 is global or per GPU. This arm retains the public control
thresholds and uses 50 samples per DDP rank. It must be compared against the
source-state-machine control rather than presented as a uniquely recoverable
3.09 recipe.
