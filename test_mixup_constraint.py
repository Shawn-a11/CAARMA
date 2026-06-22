"""Dry-run test for attribute-constrained mixup — NO training, NO GPU needed.

Run on the server (where voxceleb_full.csv + vox1_meta.csv live):
    python test_mixup_constraint.py

Part A — metadata wiring: builds the label->attribute map from the real CSVs
         and reports coverage. This is the thing that can actually break on the
         server (path id-regex / meta path). Want "mapped 1211/1211".
Part B — constraint logic: pure synthetic batch, no data. Proves that turning
         the constraint on flips a cross-gender nearest neighbour to a
         same-gender one.
"""
import sys
import yaml
import torch

sys.path.insert(0, ".")
from helper.speaker_meta import build_attr_map
from helper.mixup_avg import mixup_data_euc_avg


def _partner_of(x_mix_row, eye_dim):
    """Given x_mix[i] = 0.5*(onehot(i_pos)+onehot(partner_pos)), return the two
    active positions."""
    active = (x_mix_row.abs() > 1e-6).nonzero().flatten().tolist()
    return active


# ── Part A — metadata wiring (needs the real CSVs) ──────────────────────────
print("=" * 70)
print("PART A — metadata wiring")
print("=" * 70)
try:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    train_csv = cfg.get("dataset")
    meta_csv = cfg.get("meta_csv")
    num_spk = int(cfg.get("num_spk", 1211))
    constraint = cfg.get("mixup_constraint", "gender")
    print(f"train_csv = {train_csv}")
    print(f"meta_csv  = {meta_csv}")
    print(f"constraint= {constraint}\n")

    attr = build_attr_map(train_csv, meta_csv, num_spk, attr=constraint)
    n_ok = int((attr >= 0).sum())
    print(f"\n=> mapped {n_ok}/{num_spk}", end="  ")
    if n_ok >= 0.95 * num_spk:
        print("OK ✓  (metadata wired correctly)")
    else:
        print("WARNING ✗  too few mapped — check utt_paths id-regex / meta path")
except FileNotFoundError as e:
    print(f"SKIP Part A (CSV not found here — run on the server): {e}")
except Exception as e:
    print(f"Part A error: {e}")


# ── Part B — constraint logic (pure, no data) ───────────────────────────────
print("\n" + "=" * 70)
print("PART B — constraint logic (synthetic, no data needed)")
print("=" * 70)

# 4 speakers, labels 0..3.  gender A = {0,1}, gender B = {2,3}
# Prototypes on a line so spk0's UNCONSTRAINED nearest neighbour is spk2
# (cross-gender, dist 1) but its same-gender option is spk1 (dist 10).
emb_dim = 4
W = torch.zeros(emb_dim, 4)
positions = [0.0, 10.0, 1.0, 11.0]   # spk0=0, spk1=10, spk2=1, spk3=11
for k, p in enumerate(positions):
    W[0, k] = p
spk_attr = torch.tensor([0, 0, 1, 1])   # genders A,A,B,B

labels = torch.tensor([0, 1, 2, 3])
x = torch.eye(4)                        # row i = onehot(i) -> reveals partner

uncon = mixup_data_euc_avg(x, W, labels, spk_attr=None)[0]
con = mixup_data_euc_avg(x, W, labels, spk_attr=spk_attr)[0]

p0_uncon = _partner_of(uncon[0], 4)     # row for label 0
p0_con = _partner_of(con[0], 4)
print(f"spk0 active positions  unconstrained: {p0_uncon}  (expect [0,2] = paired w/ spk2, cross-gender)")
print(f"spk0 active positions  constrained  : {p0_con}  (expect [0,1] = paired w/ spk1, same-gender)")

ok_uncon = set(p0_uncon) == {0, 2}
ok_con = set(p0_con) == {0, 1}
print()
print(f"unconstrained picks cross-gender spk2 : {'✓' if ok_uncon else '✗'}")
print(f"constrained flips to same-gender spk1 : {'✓' if ok_con else '✗'}")
print()
if ok_uncon and ok_con:
    print("PART B PASS ✓  — gender constraint changes the pairing as intended")
else:
    print("PART B FAIL ✗  — constraint not behaving as expected")
