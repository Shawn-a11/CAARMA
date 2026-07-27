# User Requirements

- Use an isolated branch based on the `3.61` joint-Lsyn MLP-D baseline.
- Do not mix CRP, Projection-D, gender, clustering, or other proposed methods
  into baseline hyperparameter tuning.
- Do not commit checkpoints, logs, generated trial configurations, or results.
- The server workflow must support four-GPU DDP training over SSH.
- Do not require rerunning an experiment that is already recorded.
- Keep the original baseline behaviour as the default configuration.
- Freeze MLP-D capacity and synthetic-pair construction during tuning.
- Restrict autonomous changes to an explicit, reviewable hyperparameter
  whitelist.
- Select configurations on development trials; final test trials must require an
  explicit override.
- Remove every development-trial utterance from the tuning-time training CSV and
  keep the split fixed across all compared trials.
