# User Requirements

## Prototype-Only Virtual Speakers

- Base the experiment on the existing 3.61 joint-L_syn branch.
- Implement virtual classifier prototypes with no corresponding synthetic
  recording or synthetic embedding.
- A virtual prototype must be used only as a negative AM-Softmax column.
- Keep the first experiment isolated from Projection-D, boundary utility,
  gender constraints, and a persistent CRP registry.
- Run fast local correctness tests only; full DDP training runs on the remote
  GPU server.

### Document Preferences

- Use Chinese explanations unless an English version is requested.
- Keep mathematical notation renderable rather than converting it to ASCII.
