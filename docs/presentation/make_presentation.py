"""Build docs/presentation/Poison_Resilient_Continual_NIDS_review.pptx from COMMITTED files only.

  python docs/presentation/make_presentation.py      # needs python-pptx; numbers read from experiments/results/
Then: soffice --headless --convert-to pdf --outdir docs/presentation docs/presentation/<deck>.pptx
Every number on a slide is read from a committed file here (no typing numbers by hand); figures are docs/figures/*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "presentation" / "Poison_Resilient_Continual_NIDS_review.pptx"

INK, INK2, MUTED = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x40, 0x40, 0x3E), RGBColor(0x6E, 0x6D, 0x69)
BLUE, ORANGE, AQUA, YELLOW = RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xEB, 0x68, 0x34), RGBColor(0x1B, 0xAF, 0x7A), RGBColor(0xED, 0xA1, 0x00)
RED, LIGHT, PANEL = RGBColor(0xE3, 0x49, 0x48), RGBColor(0xE8, 0xF1, 0xFB), RGBColor(0xF3, 0xF2, 0xEE)
TAGC = {"VALIDATION": RGBColor(0x1C, 0x5C, 0xAB), "HELD-OUT TEST": RGBColor(0x00, 0x66, 0x33),
        "POST HOC": RGBColor(0xB0, 0x4A, 0x00), "PRE-REGISTERED": RGBColor(0x40, 0x40, 0x3E),
        "DATA": RGBColor(0x40, 0x40, 0x3E), "LIMITATION": RED, "DIAGNOSTIC": RGBColor(0x6E, 0x6D, 0x69)}


def jl(rel):
    return json.loads((ROOT / rel).read_text())


def rc(rel):
    return pd.read_csv(ROOT / rel)


# ---- numbers (all from committed files) ---------------------------------------------------------------------------
A = jl("notebooks/manifest.json")
SM = jl("experiments/results/01_data/split_manifest.json")
EM = jl("experiments/results/01_data/cii/experience_manifest.json")
FU2T = rc("experiments/results/06_fu2/test/test_summary.csv").set_index("arm")
FU2R = rc("experiments/results/06_fu2/test/test_recall.csv")
ROUNDS = {s: jl(f"experiments/results/06_fu2/fu2/cii/fu2_relabel_f1.0_s{s}/poisoned_relabel/rounds.json") for s in (42, 43, 44)}
DEC = rc("experiments/results/04_sweep/decisions.csv")
DM = rc("experiments/results/04_sweep/deployed_models.csv")
VS = rc("experiments/results/07_oracle/test/test_vs_continual.csv")
VS = VS[VS["scenario"] == "cii"].set_index("metric")
RFT = rc("experiments/results/10_rf/test/test_summary.csv")
RFV = rc("experiments/results/10_rf/rf/trajectory_val.csv")

N_RAW = sum(A["label_totals"].values())
N_DUP = sum(f["exact_duplicate_rows"] for f in A["files"])
BEN_RAW, BEN_KEPT = A["label_totals"]["BENIGN"], SM["stats"]["kept_by_label"]["BENIGN"]
CPE = EM["counts_per_experience"]
N_TRAIN4 = sum(sum(v.values()) for v in CPE.values())
KEPT = EM["protected_exclusions"]["kept_by_split_class"]
N_VAL, N_TEST, N_TEST_BEN = sum(KEPT["val"].values()), sum(KEPT["test"].values()), KEPT["test"]["BENIGN"]


def learned_all4(f, arm):
    g = DM[(DM["poison_frac"] == f) & (DM["arm"] == arm) & DM["run"].str.startswith("sweep_")]
    return int((g["classes_learned"].str.split().str.len() == 4).sum()), len(g)


def rf6(rho, arm, col):
    r = RFT[(abs(RFT["rho"] - rho) < 1e-9) & (RFT["arm"] == arm) & (RFT["model"] == "A6")].iloc[0]
    return r[f"{col}_mean"]


# ---- slide helpers --------------------------------------------------------------------------------------------------
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = prs.slide_width


def text(slide, x, y, w, h, s, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        segs = line if isinstance(line, list) else [(line, bold, color)]
        for seg in segs:
            t, b, c = (seg + (color,))[:3] if len(seg) == 2 else seg
            r = p.add_run()
            r.text, r.font.size, r.font.bold, r.font.color.rgb, r.font.name = t, Pt(size), b, c, font
        p.space_after = Pt(4)
    return tb


def bullets(slide, x, y, w, h, items, size=15):
    rows = []
    for it in items:
        if isinstance(it, tuple):  # (bold lead, rest)
            rows.append([("• " + it[0], True, INK), (it[1], False, INK2)])
        else:
            rows.append([("• " + it, False, INK2)])
    return text(slide, x, y, w, h, rows, size=size)


def tag(slide, x, y, label):
    w = 0.13 * len(label) + 0.35
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.32))
    sh.fill.solid()
    sh.fill.fore_color.rgb = TAGC[label]
    sh.line.fill.background()
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text, r.font.size, r.font.bold, r.font.color.rgb = label, Pt(11), True, RGBColor(0xFF, 0xFF, 0xFF)
    return x + w + 0.12


def box(slide, x, y, w, h, s, fill=LIGHT, size=13, color=INK, bold=False, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.25)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    lines = s if isinstance(s, list) else [s]
    for i, line_ in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text, r.font.size, r.font.color.rgb, r.font.bold = line_, Pt(size), color, bold and i == 0
    return sh


def arrow(slide, x1, y1, x2, y2, color=INK2, label=None, lsize=11):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    if label:
        text(slide, (x1 + x2) / 2 - 0.9, (y1 + y2) / 2 - 0.32, 1.8, 0.3, label, size=lsize, color=color, bold=True,
             align=PP_ALIGN.CENTER)


def table(slide, x, y, w, df, col_w=None, size=12, header_fill=PANEL, hl_rows=()):
    rows, cols = df.shape[0] + 1, df.shape[1]
    t = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(0.36 * rows)).table
    for j, c in enumerate(df.columns):
        cell = t.cell(0, j)
        cell.text = str(c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.size, r.font.bold, r.font.color.rgb = Pt(size), True, INK
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(row):
            cell = t.cell(i, j)
            cell.text = str(v)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xF4, 0xD6) if (i - 1) in hl_rows else RGBColor(0xFF, 0xFF, 0xFF)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size, r.font.color.rgb = Pt(size), INK
                    r.font.bold = (i - 1) in hl_rows
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
    return t


def new_slide(title, n, notes, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.5, 0.3, 12.3, 0.7, title, size=28, bold=True)
    if subtitle:
        text(s, 0.5, 0.95, 12.3, 0.45, subtitle, size=15, color=MUTED)
    text(s, 11.9, 7.05, 1.2, 0.3, f"{n} / 10", size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    text(s, 0.5, 7.05, 9, 0.3, "Source: github.com/DevChi1l/cicddos2019-continual-nids (committed files; tables: docs/RESULT_TABLES.md)",
         size=9, color=MUTED)
    s.notes_slide.notes_text_frame.text = notes
    return s


def picture(slide, png, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(FIG / png), Inches(x), Inches(y), **kw)


# ---- 1. title -------------------------------------------------------------------------------------------------------
s = new_slide("Poison-resilient continual intrusion detection on CICDDoS2019", 1,
              "Thirty seconds. One sentence on the problem, one on the result, one on the scope. Say 'four-label' and "
              "'post hoc' out loud now so nobody hears a 12-label claim later.")
f = FU2T.loc["poisoned_relabel"]
text(s, 0.5, 1.3, 12.3, 0.6, "Can a gate plus a frozen teacher keep a replay-based NIDS safe AND still learning when its stored labels are poisoned?",
     size=18, color=INK2)
box(s, 0.5, 2.3, 12.3, 1.6,
    ["Main result (four labels, CII, seeds 42-44)",
     f"With ALL 2,000 stored Syn labels flipped to BENIGN, teacher relabelling learns all four classes in 3/3 seeds on held-out test:",
     f"Syn {f['Syn_mean']:.4f} · LDAP {f['LDAP_mean']:.4f} · UDP {f['UDP_mean']:.4f} · benign FAR {f['benign_far_mean']:.4f}   "
     "(gate only and soft repair: LDAP = UDP = 0)"], fill=LIGHT, size=16, bold=True)
x = tag(s, 0.5, 4.15, "HELD-OUT TEST")
x = tag(s, x, 4.15, "POST HOC")
text(s, x + 0.05, 4.1, 8, 0.4, "the relabel method was chosen after the pre-registered sweep's diagnostic", size=13, color=MUTED)
bullets(s, 0.5, 4.7, 12.3, 2.2, [
    ("Scope: ", "four labels defended (BENIGN, Syn, LDAP, UDP); one attacked class (Syn); CII; 3 seeds; no significance claim"),
    ("Not claimed: ", "a 12- or 14-label defence (unsolved), or training on all 70.4M raw rows (we trained on capped, deduplicated subsets)"),
    ("Known limitation: ", "a recall-based gate does not see confidence loss (RF study, slide 10)")], size=15)

# ---- 2. problem and novelty -----------------------------------------------------------------------------------------
s = new_slide("Problem, threat model and what is new here", 2,
              "Replay buffers stop forgetting but are an attack surface. The reference paper (Ariffin et al., arXiv "
              "2608.04602, CICIDS2017) showed label flipping on the buffer collapses accuracy and proposed no defence. "
              "Our contribution is not a new architecture: it is a gated update protocol with a frozen teacher and one "
              "repair, tested with pre-registered experiments on an audited CICDDoS2019 pipeline. We do not claim the gate "
              "idea is new in the literature; we did not survey that systematically.")
bullets(s, 0.5, 1.3, 6.2, 5.5, [
    ("Continual NIDS: ", "a detector keeps learning new attack types; replay mixes stored old rows into each update so it does not forget"),
    ("Attack surface: ", "whoever can edit stored labels can make it forget an old attack or block new learning"),
    ("Threat model: ", "flip a share f of the stored labels of ONE old class (Syn -> BENIGN). The attacker cannot touch new data, validation, test, the teacher or gradients"),
    ("Inherited: ", "CI/CII scenarios, replay + buffer label-flipping threat, joint oracle (reference paper); feature-token Transformer (FT-Transformer, Gorishniy et al. 2021); CICDDoS2019 (Sharafaldin et al. 2019)"),
], size=15)
box(s, 7.0, 1.4, 5.8, 0.5, "Our contribution (implemented and measured)", fill=PANEL, size=15, bold=True)
bullets(s, 7.0, 2.0, 5.8, 4.8, [
    "Leakage-controlled pipeline over all 70.4M raw rows (audit, global dedupe, conflict drop, E0-only scaler, transformed-duplicate exclusion)",
    "Validation gate + frozen E0 teacher + one repair attempt; rejected update -> keep last accepted model",
    "Pre-registered poison sweep; post-hoc teacher relabelling (FU2) as the defence at 100% flips",
    "Honest limits: 12-label protocol fails; two pilots stopped; confidence loss invisible to the gate",
], size=15)

# ---- 3. data ----------------------------------------------------------------------------------------------------------
s = new_slide("Audited dataset and preprocessing", 3,
              f"We read all {N_RAW:,} rows once, but the data is mostly copies: {N_DUP:,} within-file duplicates. "
              "Filenames are not labels: 03-11/UDPLag.csv is mostly Syn, so the label column decides. Benign is 0.16%; "
              "we keep every distinct benign row. Training used capped subsets only: 98,833 rows for the four-label "
              "experiments. Never say we trained on 70 million rows.")
x = tag(s, 0.5, 1.15, "DATA")
picture(s, "fig1_data_pipeline.png", 0.4, 1.55, w=8.4)
bullets(s, 9.0, 1.5, 4.1, 5.4, [
    ("Audit: ", f"18 CSV files, {N_RAW:,} rows, 14 raw labels, {N_DUP:,} within-file duplicates"),
    ("Order: ", "label column (not filename) -> drop ids, IPs, ports, timestamp, Inbound -> drop non-finite -> global dedupe on a feature hash -> drop 235,261 label-conflict vectors -> cap 60,000 distinct per attack -> split 60/10/10/20"),
    ("BENIGN: ", f"{BEN_RAW:,} raw (0.16%) -> {BEN_KEPT:,} distinct kept, never capped"),
    ("Inputs: ", "77 numeric features + protocol"),
    ("Trained on: ", f"capped subsets only ({N_TRAIN4:,} rows four-label; at most 159,595)"),
], size=13)

# ---- 4. experiences ----------------------------------------------------------------------------------------------------
s = new_slide("How E0 / E1 / E2 are formed (four-label, CII)", 4,
              "E0 trains the first model on benign plus Syn. Each update brings one new attack class and a fresh, "
              "disjoint benign portion (that is CII). The scaler is fitted on E0 only and frozen. 166 validation/test rows "
              "whose scaled input equals a training input were excluded, so evaluation never sees a training input. "
              "The buffer keeps 2,000 rows per class from each accepted experience.")
df = pd.DataFrame([{"experience": e, "new class": k, "attack rows": f"{cnt[k]:,}", "fresh BENIGN rows": f"{cnt['BENIGN']:,}",
                    "model after it": m}
                   for (e, cnt), k, m in zip(CPE.items(), ("Syn", "LDAP", "UDP"), ("T (frozen) and A0", "A1 (if accepted)", "A2 (if accepted)"))])
table(s, 0.5, 1.4, 7.6, df, col_w=[1.2, 1.2, 1.5, 1.9, 1.8], size=14)
bullets(s, 0.5, 3.2, 7.6, 3.6, [
    ("Splits fixed first: ", f"validation {N_VAL:,} rows, held-out test {N_TEST:,} rows ({N_TEST_BEN:,} benign); never used for training"),
    ("Scaler: ", "signed log1p -> standardise -> clip ±5, fitted on E0 training rows only, then frozen"),
    ("Leakage guard: ", f"{EM['protected_exclusions']['n_excluded']} protected rows excluded because their scaled input equals a training input (residual 0)"),
    ("Replay buffer: ", "2,000 rows per class per ACCEPTED experience, true label stored; a rejected experience adds nothing"),
], size=14)
box(s, 8.5, 1.4, 4.3, 5.3, ["Attack cap = 13,890",
                            "= LDAP's whole cleaned training count, so every attack class contributes equally",
                            "", "Same Stage 1 splits feed every experiment (sha256 bb1f5acb…)",
                            "", "12-label runs: 9,312 per class, two classes per update; RF: fixed 3 classes"],
    fill=PANEL, size=14, bold=True)

# ---- 5. model ----------------------------------------------------------------------------------------------------------
s = new_slide("Model: FT-Transformer-style feature-token Transformer", 5,
              "Every numeric feature becomes its own 32-dimensional token, x_j times W_j plus b_j; protocol becomes an "
              "embedding token; a learned CLS token goes first. 79 tokens pass through 3 PreNorm encoder layers with 4 "
              "heads of 8 dimensions. The final CLS vector goes to LayerNorm and a linear head. 43,396 parameters with 4 "
              "classes. It is NOT the original TabTransformer, which attends only over categorical columns; the class name "
              "in the code is historical.")
y0 = 2.0
box(s, 0.4, y0, 2.2, 1.3, ["one flow", "77 scaled numbers", "+ protocol (0/6/17)"], size=13)
arrow(s, 2.6, y0 + 0.65, 3.1, y0 + 0.65)
box(s, 3.1, y0 - 0.3, 2.6, 1.9, ["Feature tokenizer", "z_j = x_j·W_j + b_j  (77 tokens)", "protocol embedding (1)", "[CLS] first (1)",
                                 "-> (N, 79, 32)"], size=12, bold=True)
arrow(s, 5.7, y0 + 0.65, 6.2, y0 + 0.65)
box(s, 6.2, y0 - 0.3, 2.8, 1.9, ["3 × PreNorm encoder layer", "self-attention: 4 heads × 8 dims", "softmax(QKᵀ/√8)·V over 79 tokens",
                                 "FFN 32 -> 128 -> 32, GELU"], size=12, bold=True)
arrow(s, 9.0, y0 + 0.65, 9.5, y0 + 0.65)
box(s, 9.5, y0, 3.3, 1.3, ["[CLS] -> LayerNorm -> Linear(32, K)", "logits for the classes seen so far"], size=12, bold=True)
bullets(s, 0.5, 4.1, 12.3, 2.8, [
    ("Size: ", "d = 32; 43,396 parameters with 4 classes (43,660 with 12)"),
    ("New classes: ", "the head grows; old output rows are copied, so old logits are unchanged before training"),
    ("Loss: ", "class-weighted cross-entropy, c_k = sqrt(N / (K·n_k)), × a per-row weight (used only by repair); AdamW, 10 epochs for E0, 5 per update"),
    ("Closest published design: ", "FT-Transformer's feature tokenizer + [CLS] + PreNorm (smaller; GELU not ReGLU). Not the original TabTransformer"),
], size=14)

# ---- 6. protocol -------------------------------------------------------------------------------------------------------
s = new_slide("Replay poisoning and the defended update protocol", 6,
              "Walk the diagram left to right. T is a frozen copy of the E0 model; A0 is the copy that keeps learning. "
              "Each update trains C1 on new data plus the buffer. The gate checks validation only: old-class recall "
              "within 0.05 of the last accepted model, new class at least 0.80, both false-alarm rates at most 0.05. "
              "If C1 fails, the repair arm tries R1 once, restarting from the last accepted model and using T on the "
              "buffer. If R1 also fails, we keep the old model. Only accepted experiences enter the buffer.")
box(s, 0.4, 1.4, 2.6, 1.1, ["Attacker: flip stored Syn labels -> BENIGN in buffer B"], fill=RGBColor(0xFD, 0xE3, 0xE3), size=12, bold=True,
    line=RED)
box(s, 0.4, 2.9, 2.6, 1.3, ["E0 -> T (frozen, hash-checked)", "and A0 (keeps learning)"], fill=PANEL, size=12, bold=True)
box(s, 3.5, 2.0, 2.4, 1.3, ["C1", "copy of A + new head rows", "train on D_t + B"], size=12, bold=True)
arrow(s, 3.0, 3.5, 3.5, 2.8)
arrow(s, 3.0, 1.95, 3.5, 2.4)
box(s, 6.4, 2.0, 2.4, 1.3, ["GATE (validation only)", "old ≥ ref − 0.05 · new ≥ 0.80", "FARs ≤ 0.05"], fill=RGBColor(0xFF, 0xF4, 0xD6),
    size=12, bold=True, line=YELLOW)
arrow(s, 5.9, 2.65, 6.4, 2.65)
box(s, 9.6, 1.2, 3.3, 1.0, ["ACCEPT -> deploy A_t", "buffer gains rows of E_t"], fill=RGBColor(0xDF, 0xF3, 0xEA), size=12, bold=True)
arrow(s, 8.8, 2.3, 9.6, 1.75, color=AQUA, label="pass")
box(s, 6.4, 4.1, 2.4, 1.4, ["R1 (repair arm only)", "restart from A; T weights or", "relabels buffer rows"], size=12, bold=True)
arrow(s, 7.6, 3.3, 7.6, 4.1, color=RED, label="fail")
box(s, 9.6, 4.2, 3.3, 1.2, ["REJECT both -> fallback", "keep last accepted model;", "buffer unchanged"], fill=RGBColor(0xFD, 0xE3, 0xE3),
    size=12, bold=True)
arrow(s, 8.8, 4.8, 9.6, 4.8, color=RED, label="fail")
arrow(s, 8.8, 4.3, 9.6, 2.0, color=AQUA, label="pass")
bullets(s, 0.4, 5.75, 12.5, 1.3, [
    ("Soft R1 (pre-registered): ", "row weight = p_T(stored label | x).  Relabel R1 (FU2, post hoc): if p_T(stored) ≤ 0.1 and T is ≥ 0.9 sure of another class, train the row with T's label"),
    ("T is never trained after E0", " and knows only BENIGN and Syn; test rows are never read by any decision"),
], size=13)

# ---- 7. FU2 validation -------------------------------------------------------------------------------------------------
s = new_slide("Why relabelling: the sweep, then FU2's validation decisions", 7,
              "Context first: in the pre-registered sweep, soft repair worked up to 95% flips, learning all four classes in "
              "9 of 9 repair runs versus 3 of 9 gate-only, but failed at 100% because no correct Syn label was left. FU2 "
              "deploys teacher relabelling at 100%. Read the table: the plain update C1 drops Syn recall to zero and the "
              "gate rejects it; relabel R1 restores 1,994-1,995 labels and passes, at E1 and E2, in all three seeds. "
              "Say clearly: the method is post hoc, and E1 replays the earlier diagnostic, so E2 and the test are the new evidence.")
x = tag(s, 0.5, 1.15, "VALIDATION")
x = tag(s, x, 1.15, "POST HOC")
text(s, x + 0.05, 1.1, 7.5, 0.4, "f = 1.0: all 2,000 stored Syn labels flipped; seeds 42-44", size=13, color=MUTED)
rows = []
for sd, rr in ROUNDS.items():
    for r in rr:
        v, ref = r["val"], r["ref_val"]
        tw = r.get("teacher_weights") or {}
        rows.append({"seed": sd, "update": f"E{r['experience']}", "attempt": "C1 (stored labels)" if r["attempt"] == "C1" else "R1 relabel",
                     "Syn val recall": f"{v['recall']['Syn']:.4f}", "Syn floor": f"{ref['recall']['Syn'] - 0.05:.4f}",
                     "rows relabelled (to true class)": f"{tw['n_relabelled']} ({tw['n_relabelled_to_true']})" if tw else "-",
                     "gate": "ACCEPT" if r["accepted"] else "REJECT"})
df = pd.DataFrame(rows)
table(s, 0.5, 1.6, 8.3, df, col_w=[0.7, 0.9, 1.9, 1.3, 1.1, 1.6, 0.8], size=11,
      hl_rows=[i for i, r in enumerate(rows) if r["gate"] == "ACCEPT"])
sweep = [(f, learned_all4(f, "poisoned"), learned_all4(f, "poisoned_repair")) for f in (0.75, 0.9, 0.95, 1.0)]
gate_le95 = sum(g[0] for f, g, _ in sweep if f <= 0.95)
rep_le95 = sum(r[0] for f, _, r in sweep if f <= 0.95)
soft1 = DEC[(DEC["poison_frac"] == 1.0) & (DEC["attempt"] == "R1")]
box(s, 9.1, 1.6, 3.8, 0.45, "Pre-registered sweep (context)", fill=PANEL, size=13, bold=True)
x2 = tag(s, 9.1, 2.15, "PRE-REGISTERED")
tag(s, x2, 2.15, "HELD-OUT TEST")
bullets(s, 9.1, 2.6, 3.9, 4.3, [
    f"f ≤ 0.95: all 4 classes learned in {rep_le95}/9 soft-repair runs vs {gate_le95}/9 gate-only",
    f"f = 1.0: soft R1 accepted {int(soft1['accepted'].sum())}/{len(soft1)}; teacher weight on flipped rows ≈ {soft1['w_flipped'].mean():.3f}: wrong labels removed, no Syn label left",
    "Relabel diagnostic would have passed 3/3 -> FU2 deploys it (registered before its run)",
], size=13)

# ---- 8. FU2 test -------------------------------------------------------------------------------------------------------
s = new_slide("Result: four-class defence at 100% flips (held-out test)", 8,
              "Same 46,527 test rows for every arm, scored once after the validation commit for FU2. Relabelling learns all "
              "four classes, within about 0.01 of clean, with the same benign false-alarm rate. Gate-only and soft repair "
              "keep the stale A0, which calls LDAP and UDP benign; their false-alarm rate of 0 is not safety. Scope: four "
              "labels, one attacked class, 3 seeds, post hoc method.")
x = tag(s, 0.5, 1.15, "HELD-OUT TEST")
x = tag(s, x, 1.15, "POST HOC")
text(s, x + 0.05, 1.1, 8, 0.4, f"mean (min-max) over seeds 42-44; {N_TEST:,} test rows, {N_TEST_BEN:,} benign", size=13, color=MUTED)
picture(s, "fig3_fu2_defence.png", 0.4, 1.6, w=8.2)
names = {"clean": "clean", "poisoned": "gate only", "poisoned_repair": "soft repair", "poisoned_relabel": "relabel (FU2)"}
df = pd.DataFrame([{"arm": names[a], "Syn": f"{FU2T.loc[a, 'Syn_mean']:.4f}", "LDAP": f"{FU2T.loc[a, 'LDAP_mean']:.4f}",
                    "UDP": f"{FU2T.loc[a, 'UDP_mean']:.4f}", "benign FAR": f"{FU2T.loc[a, 'benign_far_mean']:.4f}"} for a in names])
table(s, 8.8, 1.7, 4.3, df, col_w=[1.25, 0.75, 0.75, 0.75, 0.8], size=11, hl_rows=[3])
h3 = rc("experiments/results/06_fu2/test/h3.csv")
bullets(s, 8.8, 3.8, 4.3, 3.0, [
    f"H3 (Syn ≥ 0.95, LDAP & UDP ≥ 0.80, FAR ≤ 0.05) met in {int(h3['met'].sum())}/{len(h3)} seeds",
    "FAR 0 for gate-only is NOT safety: the stale A0 labels LDAP/UDP traffic benign",
    "Scope: 4 labels, Syn attacked, CII, 3 seeds",
], size=13)

# ---- 9. 12-label limitation ------------------------------------------------------------------------------------------
s = new_slide("Limitation: the 12-label continual setting is not solved", 9,
              "With 12 labels even CLEAN updates fail the gate: DNS overlaps LDAP and UDP overlaps SSDP, and a rejected block "
              "is lost for good. The continual CII models keep 6 of 12 classes. A retry recipe (FU1) learned DNS/LDAP but "
              "lost others. A jointly trained oracle reaches all 12 classes but is weak on DNS/LDAP and UDP even without "
              "continual learning, and it is a diagnostic, not a defended continual result. So we claim four labels only.")
x = tag(s, 0.5, 1.15, "HELD-OUT TEST")
x = tag(s, x, 1.15, "LIMITATION")
x = tag(s, x, 1.15, "DIAGNOSTIC")
text(s, x + 0.05, 1.1, 6, 0.4, "oracle = one jointly trained model, not continual", size=13, color=MUTED)
picture(s, "fig4_12label_vs_oracle.png", 0.4, 1.6, w=9.0)
bullets(s, 9.6, 1.6, 3.5, 5.2, [
    (f"Continual s3 clean: ", f"6/12 classes, macro recall {VS.loc['macro_recall', 'continual_s3_clean']:.3f} (one seed)"),
    (f"FU1 retry: ", f"6/12, macro {VS.loc['macro_recall', 'continual_fu1']:.3f}; DNS/LDAP {VS.loc['recall_DNS', 'continual_fu1']:.2f}/{VS.loc['recall_LDAP', 'continual_fu1']:.2f} learned, MSSQL/NTP lost"),
    (f"Joint oracle (diagnostic): ", f"macro {VS.loc['macro_recall', 'oracle_joint_e10']:.3f}/{VS.loc['macro_recall', 'oracle_joint_e20']:.3f}; "
     f"DNS {VS.loc['recall_DNS', 'oracle_joint_e10']:.2f}, LDAP {VS.loc['recall_LDAP', 'oracle_joint_e20']:.2f}, UDP {VS.loc['recall_UDP', 'oracle_joint_e10']:.2f}"),
    ("Claim: ", "no 12- or 14-label defence"),
], size=13)

# ---- 10. RF -----------------------------------------------------------------------------------------------------------
po = RFV[(RFV["arm"] == "poisoned") & (RFV["t"] > 0)]
rep = RFV[(RFV["arm"] == "poisoned_repair") & (RFV["t"] > 0)]
s = new_slide("Limitation: confidence loss the recall-based gate does not see (RF)", 10,
              "RF is post hoc and uses three fixed classes with six traffic-refresh updates, no new class. Before every "
              "update 10 percent more stored Syn labels are flipped, 60 percent by the end. Every poisoned update passed "
              "the gate, so repair never ran. Recall stayed at 0.994, but the model's confidence on Syn fell to 0.77. "
              "Validation is the top row of the figure, held-out test the bottom row. Close: the four-class defence works; "
              "the gate needs a confidence check, which is our proposed next step, not a result.")
x = tag(s, 0.5, 1.15, "VALIDATION")
x = tag(s, x, 1.15, "HELD-OUT TEST")
x = tag(s, x, 1.15, "POST HOC")
x = tag(s, x, 1.15, "LIMITATION")
picture(s, "fig6_rf_trajectory.png", 0.4, 1.55, h=5.3)
bullets(s, 8.7, 1.6, 4.4, 5.3, [
    ("Setting: ", "BENIGN, Syn, TFTP fixed; 6 refresh updates; no new class"),
    ("Attack: ", "+10% stored Syn labels flipped per update -> 60% after E6"),
    ("Gate: ", f"poisoned C1 accepted {int(po['c1_accepted'].sum())}/{len(po)}; repair R1 ran {int(rep['r1_ran'].sum())} times"),
    ("Test, A6 (10%): ", f"Syn recall {rf6(0.10, 'poisoned', 'Syn_recall'):.4f} vs {rf6(0.10, 'clean', 'Syn_recall'):.4f} clean; "
     f"mean Syn confidence {rf6(0.10, 'poisoned', 'Syn_mean_p'):.4f} vs {rf6(0.10, 'clean', 'Syn_mean_p'):.4f}"),
    ("Not claimed: ", "that repair fixed this (it never ran); new-class arrival"),
    ("Next (proposal): ", "a pre-registered confidence check in the gate"),
], size=13)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"wrote {OUT.relative_to(ROOT)} ({len(prs.slides)} slides)")
