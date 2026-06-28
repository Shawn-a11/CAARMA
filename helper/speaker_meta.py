"""Speaker attribute lookup for attribute-constrained synthetic pairing."""

import re

import pandas as pd
import torch


_VOXID_RE = re.compile(r"id\d+")


def build_attr_map(train_csv, meta_csv, num_spk, attr="gender", verbose=True):
    """Return speaker-label -> attribute-code tensor.

    The training CSV maps each utterance to an integer speaker label and a path
    containing a VoxCeleb ID. The metadata CSV maps VoxCeleb IDs to attributes
    such as gender or nationality.
    """
    col = {"gender": "Gender", "nationality": "Nationality"}[attr]

    df = pd.read_csv(train_csv)
    labels = df["utt_spk_int_labels"].values
    paths = df["utt_paths"].values
    label2vox = {}
    for lab, path in zip(labels, paths):
        lab = int(lab)
        if lab in label2vox:
            continue
        match = _VOXID_RE.search(str(path))
        if match:
            label2vox[lab] = match.group(0)

    meta = pd.read_csv(meta_csv, sep="\t")
    meta.columns = [c.strip() for c in meta.columns]
    id_col = [c for c in meta.columns if "VoxCeleb1" in c][0]
    vox2attr = {
        str(row[id_col]).strip(): str(row[col]).strip()
        for _, row in meta.iterrows()
    }

    present = sorted({vox2attr[v] for v in label2vox.values() if v in vox2attr})
    attr2code = {a: i for i, a in enumerate(present)}

    out = torch.full((num_spk,), -1, dtype=torch.long)
    mapped = 0
    for lab, vox in label2vox.items():
        value = vox2attr.get(vox)
        if value is not None and 0 <= lab < num_spk:
            out[lab] = attr2code[value]
            mapped += 1

    if verbose:
        counts = {a: int((out == c).sum()) for a, c in attr2code.items()}
        top = sorted(counts.items(), key=lambda item: -item[1])
        print(
            "[speaker_meta] attr='{}' mapped {}/{} speakers ({} groups)".format(
                attr, mapped, num_spk, len(present)
            )
        )
        print("[speaker_meta] top groups: " + ", ".join(
            "{}:{}".format(a, n) for a, n in top[:8]
        ))
        unknown = int((out < 0).sum())
        if unknown:
            print(
                "[speaker_meta] WARNING: {} labels unmatched; they fall back "
                "to unconstrained pairing.".format(unknown)
            )

    return out
