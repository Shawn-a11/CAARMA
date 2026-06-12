"""Build a per-speaker attribute lookup (gender / nationality) for
attribute-constrained mixup.

The training manifest (voxceleb_full.csv) maps each utterance to an integer
speaker label (utt_spk_int_labels) and a file path (utt_paths) that contains
the VoxCeleb1 ID, e.g. .../id10001/video/00001.wav. vox1_meta.csv maps the
VoxCeleb1 ID to Gender / Nationality. Composing the two gives
    mapped_id (0..num_spk-1)  ->  attribute code (int)
which mixup uses to restrict synthetic-speaker pairing to same-attribute pairs
(a male+female blend, or a cross-language blend, is not a meaningful "speaker").
"""
import re
import pandas as pd
import torch

_VOXID_RE = re.compile(r"id\d+")


def build_attr_map(train_csv, meta_csv, num_spk, attr="gender", verbose=True):
    """Return a LongTensor of shape (num_spk,) mapping speaker label -> attr code.

    Unknown / unmatched labels are left as -1 (mixup treats -1 as "no
    constraint" so they fall back to the unconstrained nearest neighbour).

    attr: "gender" or "nationality".
    """
    col = {"gender": "Gender", "nationality": "Nationality"}[attr]

    # ── label -> VoxCeleb1 ID (first occurrence per label is enough) ──
    df = pd.read_csv(train_csv)
    labels = df["utt_spk_int_labels"].values
    paths = df["utt_paths"].values
    label2vox = {}
    for lab, p in zip(labels, paths):
        lab = int(lab)
        if lab in label2vox:
            continue
        m = _VOXID_RE.search(str(p))
        if m:
            label2vox[lab] = m.group(0)

    # ── VoxCeleb1 ID -> attribute string ──
    meta = pd.read_csv(meta_csv, sep="\t")
    meta.columns = [c.strip() for c in meta.columns]
    id_col = [c for c in meta.columns if "VoxCeleb1" in c][0]
    vox2attr = {
        str(r[id_col]).strip(): str(r[col]).strip()
        for _, r in meta.iterrows()
    }

    # ── encode attribute strings to integer codes ──
    present = sorted({vox2attr[v] for v in label2vox.values() if v in vox2attr})
    attr2code = {a: i for i, a in enumerate(present)}

    arr = torch.full((num_spk,), -1, dtype=torch.long)
    n_mapped = 0
    for lab, vox in label2vox.items():
        a = vox2attr.get(vox)
        if a is not None and 0 <= lab < num_spk:
            arr[lab] = attr2code[a]
            n_mapped += 1

    if verbose:
        # distribution over the training speakers we actually mapped
        counts = {a: int((arr == c).sum()) for a, c in attr2code.items()}
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])
        print(f"[speaker_meta] attr='{attr}'  mapped {n_mapped}/{num_spk} speakers "
              f"({len(present)} distinct values)")
        print(f"[speaker_meta] top groups: "
              + ", ".join(f"{a}:{n}" for a, n in ordered[:8]))
        n_unknown = int((arr < 0).sum())
        if n_unknown:
            print(f"[speaker_meta] WARNING: {n_unknown} labels unmatched "
                  f"-> unconstrained (check path id-regex / meta coverage)")

    return arr
