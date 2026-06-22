"""Build slide deck explaining the CAARMA discriminator.

Five questions structured around how a pretrained SSL backbone is used as
the adversarial critic in CAARMA, followed by a backbone-alternative
analysis and a summary.

  1. Title
  2. Five questions about D (agenda)
  3. GAN recap (anchor in familiar ground)
  4. Q1 — Input contradiction (embedding vs speech)
  5. Q2 — D architecture (where is HuBERT)
  6. Q3 — Training mechanism (alternating updates)
  7. Q4 — Gradient flow (frozen but grad flows)
  8. Q5 — Adversarial dynamics across 8 runs (empirical)
  9. Ext — Backbone alternative: WavLM analysis
 10. Summary
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

VIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "CAARMA_discriminator_deck.pptx")

# ── Charcoal Minimal palette (matches our figure colors) ──
C_BG_DARK   = RGBColor(0x1E, 0x2A, 0x33)   # near-black charcoal
C_BG_LIGHT  = RGBColor(0xFA, 0xFA, 0xF5)   # off-white
C_TITLE     = RGBColor(0x21, 0x21, 0x21)
C_BODY      = RGBColor(0x36, 0x45, 0x4F)
C_ACCENT    = RGBColor(0xE7, 0x6F, 0x51)   # coral (key callout)
C_MUTED     = RGBColor(0x6B, 0x72, 0x80)
C_GREEN     = RGBColor(0x2A, 0x7C, 0x4F)
C_RED       = RGBColor(0xC6, 0x28, 0x28)
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height

BLANK = prs.slide_layouts[6]


def add_bg(slide, color):
    """Solid background rectangle covering the whole slide."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    bg.fill.solid(); bg.fill.fore_color.rgb = color
    bg.line.fill.background()


def add_text(slide, text, x, y, w, h, *,
             font="Calibri", size=14, bold=False, italic=False,
             color=None, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             line_spacing=1.15):
    """Convenience textbox helper. `text` may contain \\n for multi-line."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        f = r.font
        f.name = font
        f.size = Pt(size)
        f.bold = bold
        f.italic = italic
        if color is not None:
            f.color.rgb = color
    return tb


def add_accent_chip(slide, label, x, y, w=Inches(1.3), h=Inches(0.32),
                    fill=None, font_color=None):
    """Small pill-shaped label for question tags (Q1, Q2 …)."""
    fill = fill or C_ACCENT
    font_color = font_color or C_WHITE
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    pill.fill.solid(); pill.fill.fore_color.rgb = fill
    pill.line.fill.background()
    pill.adjustments[0] = 0.5
    tf = pill.text_frame; tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = label
    r.font.name = "Calibri"; r.font.size = Pt(12); r.font.bold = True
    r.font.color.rgb = font_color


def add_image_centered(slide, img_path, top, max_w, max_h):
    """Add image scaled to fit within max_w × max_h, horizontally centered."""
    from PIL import Image
    with Image.open(img_path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w = int(iw * scale); h = int(ih * scale)
    x = (SW - w) // 2
    slide.shapes.add_picture(img_path, x, top, width=w, height=h)


# ────────────────────────────────────────────────────────────────────────
# SLIDE 1 — Title (dark)
# ────────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_bg(s, C_BG_DARK)

# Coral accent bar on left
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.4), Inches(0.18), Inches(2.7))
bar.fill.solid(); bar.fill.fore_color.rgb = C_ACCENT; bar.line.fill.background()

add_text(s, "How HuBERT Serves as the\nCAARMA Discriminator",
         Inches(0.6), Inches(2.3), Inches(12), Inches(1.8),
         font="Calibri", size=44, bold=True, color=C_WHITE,
         line_spacing=1.05)

add_text(s, "Architecture, training mechanism, and gradient flow",
         Inches(0.6), Inches(4.4), Inches(12), Inches(0.5),
         font="Calibri", size=18, color=C_ACCENT, italic=True)

add_text(s, "CAARMA Reproduction & Extension Study",
         Inches(0.6), Inches(6.7), Inches(12), Inches(0.4),
         font="Calibri", size=12, color=C_MUTED)


# ────────────────────────────────────────────────────────────────────────
# SLIDE 2 — Agenda (the 5 questions, in his words)
# ────────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_bg(s, C_BG_LIGHT)

add_text(s, "Five questions about the discriminator",
         Inches(0.6), Inches(0.4), Inches(12), Inches(0.7),
         font="Calibri", size=30, bold=True, color=C_TITLE)
add_text(s, "How a pretrained SSL backbone functions as the adversarial critic in CAARMA.",
         Inches(0.6), Inches(1.05), Inches(12), Inches(0.4),
         font="Calibri", size=14, italic=True, color=C_MUTED)

questions = [
    ("Q1", "The input-type contradiction",
     "HuBERT was built to consume speech, but CAARMA feeds it an embedding."),
    ("Q2", "The discriminator's architecture",
     "What is the network? Where exactly does HuBERT sit inside it?"),
    ("Q3", "The training mechanism",
     "How does standard GAN alternation work when D is a pretrained SSL model?"),
    ("Q4", "Gradient backpropagation",
     "How does HuBERT send gradients back to the encoder?"),
    ("Q5", "Empirical adversarial dynamics",
     "How the BCE losses evolve across 8 training runs — what does the data say?"),
]
y0 = Inches(1.7)
row_h = Inches(1.0)
for i, (tag, head, body) in enumerate(questions):
    y = y0 + row_h * i
    # Q-tag pill
    add_accent_chip(s, tag, Inches(0.6), y + Inches(0.18),
                    w=Inches(0.7), h=Inches(0.42))
    add_text(s, head, Inches(1.5), y + Inches(0.05), Inches(11), Inches(0.45),
             font="Calibri", size=18, bold=True, color=C_TITLE)
    add_text(s, body, Inches(1.5), y + Inches(0.5), Inches(11.2), Inches(0.45),
             font="Calibri", size=13, color=C_BODY, italic=True)


# ────────────────────────────────────────────────────────────────────────
# Helper: standard content slide layout (used 6.0 – 6.7)
# ────────────────────────────────────────────────────────────────────────
def content_slide(tag, title, subtitle, img_path, bullets):
    s = prs.slides.add_slide(BLANK)
    add_bg(s, C_BG_LIGHT)
    # Tag chip top-left
    add_accent_chip(s, tag, Inches(0.5), Inches(0.45),
                    w=Inches(0.85), h=Inches(0.38))
    # Title
    add_text(s, title, Inches(1.5), Inches(0.4), Inches(11.5), Inches(0.6),
             font="Calibri", size=24, bold=True, color=C_TITLE)
    # Subtitle
    add_text(s, subtitle, Inches(1.5), Inches(0.95), Inches(11.5), Inches(0.4),
             font="Calibri", size=12, italic=True, color=C_MUTED)
    # Bullets area — tighter so the figure can grow vertically
    bullet_y = Inches(6.25)
    bullet_h = Inches(1.15)
    bullet_w = Inches(12.3)
    add_text(s, bullets, Inches(0.5), bullet_y, bullet_w, bullet_h,
             font="Calibri", size=12, color=C_BODY, line_spacing=1.25)
    # Figure occupies middle — 5.0" tall vs previous 4.0" → 25% more vertical
    # space lets portrait-aspect figures (6.2, 6.4, 6.5) scale up and fill the
    # slide horizontally instead of being squeezed in the middle.
    add_image_centered(s, img_path,
                       top=Inches(1.40),
                       max_w=Inches(12.3),
                       max_h=Inches(4.75))
    return s


# ────────────────────────────────────────────────────────────────────────
# SLIDE 3 — GAN background (anchor in familiar GAN terms)
# ────────────────────────────────────────────────────────────────────────
content_slide(
    "Setup",
    "Anchoring in standard GAN terms",
    "Same alternating game, different inputs — D's role is conceptually identical.",
    os.path.join(VIZ, "fig_6_0_gan_recap.png"),
    "•  Standard GAN: G generates from noise, D distinguishes real vs fake samples.\n"
    "•  CAARMA: encoder M produces e from audio, mixup produces e_syn from pairs.\n"
    "•  D in both cases is a binary classifier — and is discarded at inference."
)


# ────────────────────────────────────────────────────────────────────────
# SLIDE 4 — Q1 input contradiction
# ────────────────────────────────────────────────────────────────────────
content_slide(
    "Q1",
    "Resolving the input-type contradiction",
    "HuBERT was designed for speech, but CAARMA feeds it an embedding. Here is how.",
    os.path.join(VIZ, "fig_6_1_input_contradiction.png"),
    "•  Standard HuBERT path uses a conv feature extractor on raw waveforms — CAARMA bypasses it.\n"
    "•  An EnhancedAdapter projects the 192-d speaker embedding into HuBERT's 1024-d hidden-state space.\n"
    "•  The 24-layer Transformer encoder is borrowed for its pretrained representations, not its speech I/O."
)


# ────────────────────────────────────────────────────────────────────────
# SLIDE 5 — Q2 D architecture
# ────────────────────────────────────────────────────────────────────────
content_slide(
    "Q2",
    "Where HuBERT sits inside the discriminator",
    "D is a small classifier wrapped around a borrowed 24-layer Transformer stack.",
    os.path.join(VIZ, "fig_6_2_d_architecture.png"),
    "•  D total: 326M parameters — 315M HuBERT backbone + ~11M trainable head.\n"
    "•  Adapter sits between input embedding and HuBERT; layer 7/9/11/12 hidden states are pooled.\n"
    "•  HuBERT is frozen in our DDP setup (deadlock fix); trainable in the source repo."
)


# ────────────────────────────────────────────────────────────────────────
# SLIDE 6 — Q3 training mechanism
# ────────────────────────────────────────────────────────────────────────
content_slide(
    "Q3",
    "How D is trained — standard GAN alternation",
    "Each batch: update D, then update M. Never both in the same backward pass.",
    os.path.join(VIZ, "fig_6_3_training_mechanism.png"),
    "•  Step 1 (D): encoder runs inside torch.no_grad(); only D's parameters receive gradients.\n"
    "•  Step 2 (M): D is frozen via toggle_optimizer; encoder + AM-Softmax W receive gradients.\n"
    "•  This is exactly the textbook GAN cycle — the SSL backbone simply lives inside D."
)


# ────────────────────────────────────────────────────────────────────────
# SLIDE 7 — Q4 gradient flow
# ────────────────────────────────────────────────────────────────────────
content_slide(
    "Q4",
    "How D propagates gradients back to the encoder",
    "Frozen does NOT mean blocked — `requires_grad=False` only prevents parameter updates.",
    os.path.join(VIZ, "fig_6_4_gradient_flow.png"),
    "•  L_G's gradient propagates through every layer of D, including the frozen HuBERT Transformer.\n"
    "•  The gradient arrives at e and e_syn, then flows back through mixup (SLERP is differentiable).\n"
    "•  This is precisely how a frozen pretrained backbone still trains the encoder through D."
)


# ────────────────────────────────────────────────────────────────────────
# SLIDE 8 — Q5 empirical diagnosis (the rebuttal slide)
# ────────────────────────────────────────────────────────────────────────
content_slide(
    "Q5",
    "Adversarial dynamics across 8 training runs",
    "How the BCE losses evolve, and what that tells us about the discriminator's behaviour.",
    os.path.join(VIZ, "fig_6_6_empirical_diagnosis.png"),
    "•  D's outputs settle at σ(real) ≈ 0.65  vs  σ(fake) ≈ 0.35 — well above the 0.5 random baseline.\n"
    "•  d_loss stays near 0.85 while g_loss climbs to 3–5 — the discriminator wins consistently.\n"
    "•  The open question is how this signal is weighted in the total loss, not the discriminator's capacity."
)


# ────────────────────────────────────────────────────────────────────────
# SLIDE 9 — WavLM backbone analysis (replaces the previous λ_adv slides)
# ────────────────────────────────────────────────────────────────────────
content_slide(
    "Ext",
    "Backbone alternative: why we tried WavLM (and what we found)",
    "Same 24-layer Transformer architecture as HuBERT; different pretraining objective.",
    os.path.join(VIZ, "fig_6_5_wavlm_vs_hubert.png"),
    "•  WavLM (Chen et al. 2022): masked prediction + utterance MIXING + denoising — designed for speaker tasks.\n"
    "•  Empirical: WavLM gives EER 3.71%  vs  HuBERT 3.48% — WavLM is actually WORSE in our setup.\n"
    "•  Cause: seq_len=1 input collapses self-attention to identity, erasing WavLM's sequence-level advantages."
)


# ────────────────────────────────────────────────────────────────────────
# SLIDE 11 — Closing
# ────────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_bg(s, C_BG_DARK)

add_text(s, "Summary",
         Inches(0.6), Inches(0.4), Inches(12), Inches(0.6),
         font="Calibri", size=18, italic=True, color=C_ACCENT)

add_text(s, "Five mechanistic answers — one architectural finding",
         Inches(0.6), Inches(0.85), Inches(12), Inches(0.7),
         font="Calibri", size=28, bold=True, color=C_WHITE)

bullets_summary = [
    ("Q1",  "Adapter projects embedding into HuBERT's hidden-state space — no conv extractor used."),
    ("Q2",  "D = adapter + 24-L Transformer + multi-layer pool + classifier head, 326M total."),
    ("Q3",  "Standard GAN alternation: opt_D step, then opt_M step, every batch."),
    ("Q4",  "Frozen HuBERT still passes gradient through to the encoder — only updates are blocked."),
    ("Q5",  "D works (σ 0.65 vs 0.35).  Discrimination is happening — capacity is not the bottleneck."),
    ("Ext", "WavLM tried; underperformed HuBERT (3.71% vs 3.48%) — seq_len=1 collapses its advantage."),
]
y0 = Inches(2.0); row_h = Inches(0.70)
for i, (tag, body) in enumerate(bullets_summary):
    y = y0 + row_h * i
    add_accent_chip(s, tag, Inches(0.7), y + Inches(0.10),
                    w=Inches(0.8), h=Inches(0.36),
                    fill=C_ACCENT, font_color=C_WHITE)
    add_text(s, body, Inches(1.75), y + Inches(0.08), Inches(11), Inches(0.55),
             font="Calibri", size=14, color=C_WHITE)

# Bottom strip — takeaway
strip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                           Inches(0), Inches(6.5), SW, Inches(1.0))
strip.fill.solid(); strip.fill.fore_color.rgb = C_ACCENT
strip.line.fill.background()
add_text(s,
         "Open question:  how should the adversarial signal be weighted in the total loss?",
         Inches(0.6), Inches(6.75), Inches(12.1), Inches(0.5),
         font="Calibri", size=16, bold=True, color=C_WHITE,
         align=PP_ALIGN.CENTER)


# Save
prs.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"Slides: {len(prs.slides)}")
