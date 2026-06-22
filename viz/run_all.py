"""One-shot generator for CAARMA Discriminator visualization assets — slide 6.

Outputs (all in ./viz/):
  d_arch.png            ← torchview hierarchical forward graph
  d_arch.gv             ← graphviz source (editable)
  d_summary.txt         ← torchinfo layer-by-layer shape + param table
  d_backward.pdf/.png   ← torchviz autograd backward graph
  d.onnx                ← ONNX for Netron interactive viewer

Run from repo root:
  python viz/run_all.py

CPU is fine; first run downloads the SSL backbone (HuBERT-Large ~1.2GB or
WavLM-Large) into HF cache. Subsequent runs are fast.

Requires:
  pip install torchview torchinfo torchviz graphviz onnx
  brew install graphviz   # macOS; or `apt install graphviz` on Ubuntu
"""
import os
import sys
import torch

# Make repo root importable when running from `viz/` or `./`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model.discriminator_mix import MixupDiscriminator   # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "viz", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

device = "cpu"
print(f"[init] loading MixupDiscriminator on {device} ...")

# The branches have different default `*_model_name` paths. On the server the
# default is a local mirror like /root/autodl-tmp/hubert-large; on a laptop
# that won't exist. We detect that and fall back to the HF hub identifier.
import inspect
_sig = inspect.signature(MixupDiscriminator.__init__)
_default_path = None
_path_arg = None
for arg in ("hubert_model_name", "wavlm_model_name"):
    if arg in _sig.parameters:
        _default_path = _sig.parameters[arg].default
        _path_arg = arg
        break

_HF_FALLBACK = {
    "hubert_model_name": "facebook/hubert-large-ls960-ft",
    "wavlm_model_name":  "microsoft/wavlm-large",
}
_kwargs = {"cache_dir": os.path.join(REPO_ROOT, "cache_dir/")}
if _default_path and isinstance(_default_path, str) and _default_path.startswith("/"):
    if not os.path.exists(_default_path):
        _kwargs[_path_arg] = _HF_FALLBACK[_path_arg]
        print(f"[init] local path {_default_path} missing → using HF: {_HF_FALLBACK[_path_arg]}")

D = MixupDiscriminator(**_kwargs).to(device).eval()
print(f"[init] total params: {sum(p.numel() for p in D.parameters()):,}")

B = 4
EMB_DIM = 192
emb = torch.randn(B, EMB_DIM, device=device, requires_grad=True)

# ───────────────────────────────────────────────────────────────────────────
# 1. torchview — hierarchical forward graph (best for slide 6.2)
# ───────────────────────────────────────────────────────────────────────────
print("\n[1/4] torchview — drawing forward graph ...")
try:
    from torchview import draw_graph

    # COMPACT — depth=2, HuBERT collapsed into ONE box; best for slide 6.2
    draw_graph(
        D,
        input_data=emb,
        expand_nested=False,                 # ← keep submodules folded
        depth=2,
        graph_name="MixupDiscriminator",
        save_graph=True,
        filename="d_arch_compact",
        directory=OUT_DIR,
        hide_inner_tensors=True,
        hide_module_functions=True,
        roll=True,
    )
    print(f"       → {OUT_DIR}/d_arch_compact.png  (slide 6.2 main figure)")

    # DETAILED — depth=3, EnhancedAdapter and classifier head expanded but
    # HuBERT still collapsed. Useful for showing layer counts inside D head.
    draw_graph(
        D,
        input_data=emb,
        expand_nested=False,
        depth=3,
        graph_name="MixupDiscriminator (detailed)",
        save_graph=True,
        filename="d_arch_detailed",
        directory=OUT_DIR,
        hide_inner_tensors=True,
        hide_module_functions=True,
        roll=True,
    )
    print(f"       → {OUT_DIR}/d_arch_detailed.png  (backup, more layers visible)")
except ImportError:
    print("       SKIP (pip install torchview)")
except Exception as e:
    print(f"       FAILED: {e}")

# ───────────────────────────────────────────────────────────────────────────
# 2. torchinfo — text summary with input/output shapes and trainable column
# ───────────────────────────────────────────────────────────────────────────
print("\n[2/4] torchinfo — layer summary ...")
try:
    from torchinfo import summary

    s = summary(
        D,
        input_size=(B, EMB_DIM),
        depth=4,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=0,
    )
    out_path = os.path.join(OUT_DIR, "d_summary.txt")
    with open(out_path, "w") as f:
        f.write(str(s))
    print(f"       → {out_path}")
except ImportError:
    print("       SKIP (pip install torchinfo)")
except Exception as e:
    print(f"       FAILED: {e}")

# ───────────────────────────────────────────────────────────────────────────
# 3. torchviz — autograd backward graph (best for slide 6.4)
# ───────────────────────────────────────────────────────────────────────────
print("\n[3/4] torchviz — backward computation graph ...")
try:
    from torchviz import make_dot

    logit = D(emb)
    dot = make_dot(
        logit,
        params=dict(D.named_parameters()),
        show_attrs=False,
        show_saved=False,
    )
    dot.format = "png"
    dot.render(os.path.join(OUT_DIR, "d_backward"), cleanup=True)
    dot.format = "pdf"
    dot.render(os.path.join(OUT_DIR, "d_backward"), cleanup=True)
    print(f"       → {OUT_DIR}/d_backward.{{png,pdf}}")
except ImportError:
    print("       SKIP (pip install torchviz)")
except Exception as e:
    print(f"       FAILED: {e}")

# ───────────────────────────────────────────────────────────────────────────
# 4. ONNX export — for Netron interactive viewer
# ───────────────────────────────────────────────────────────────────────────
print("\n[4/4] ONNX export for Netron ...")
try:
    # spectral_norm and other weight reparam modules can't be deep-copied or
    # traced as-is by torch.onnx. We mutate D in place — this is the last step
    # of the script so the side effect doesn't affect anything else.
    for name, mod in list(D.named_modules()):
        # 1. spectral_norm
        if hasattr(mod, "weight_orig"):
            try:
                torch.nn.utils.remove_spectral_norm(mod)
            except Exception:
                pass
        # 2. torch.nn.utils.parametrize style reparam (newer transformers
        # use this for HuBERT's positional conv embedding etc.)
        if hasattr(mod, "parametrizations"):
            try:
                import torch.nn.utils.parametrize as _para
                # remove every parametrization registered on this module
                for pname in list(mod.parametrizations.keys()):
                    _para.remove_parametrizations(mod, pname, leave_parametrized=True)
            except Exception:
                pass

    onnx_path = os.path.join(REPO_ROOT, "viz", "models", "d.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        D,
        torch.randn(B, EMB_DIM),
        onnx_path,
        input_names=["embedding_192d"],
        output_names=["real_fake_logit"],
        dynamic_axes={"embedding_192d": {0: "batch"}},
        opset_version=14,
    )
    print(f"       → {onnx_path}  (open with `netron {onnx_path}`)")
except Exception as e:
    print(f"       FAILED: {e}")
    print(f"       (ONNX export sometimes fails on transformers — torchview/torchviz output is sufficient)")

print(f"\n[done] all assets in {OUT_DIR}/")
print("[done] slide 6 mapping:")
print("       6.1 input contradiction  →  use Netron screenshot of d.onnx input node")
print("       6.2 D architecture       →  d_arch.png + d_summary.txt caption")
print("       6.3 training mechanism   →  manual diagram (no tool covers dual-optimizer toggling)")
print("       6.4 gradient flow        →  d_backward.pdf + manual annotation of frozen layers")
