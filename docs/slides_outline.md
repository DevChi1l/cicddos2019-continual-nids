# Slide outline with speaker notes (about 15 minutes)

Numbers marked **[main]** come from the completed run (`nids_results.zip`, `runs/cii/main`).
Slots marked **[fill: file]** are filled from `deliverables/` (made by `s7_deliverables.py`) only after
those runs exist. If a file does not exist, the slide says "not run". Never fill a slot from memory.

---

## 1. Title: Poison-resilient continual intrusion detection on CICDDoS2019
Notes: one sentence on the problem. A detector that keeps learning new attacks replays stored examples,
and an attacker who can edit those stored labels can make it forget an old attack.

## 2. Question and threat model
- Continual updates: E0 (BENIGN + Syn) -> E1 (+ LDAP) -> E2 (+ UDP), scenario CII (fresh benign every update).
- Attacker: flips the stored label of a fraction f of ONE old class (Syn -> BENIGN) in the replay buffer only.
Notes: say what the attacker cannot touch: training data of the new experience, validation, test, the teacher.

## 3. Data pipeline (leakage first)
- 77 numeric features + protocol; IDs, IPs, ports, timestamp, SimillarHTTP and Inbound removed.
- Global dedupe before capping; label-conflict vectors dropped; per-class 60/10/10/20 split.
- 166 protected rows excluded because their *model input* equals a training input (clip, rounding, float32); residual 0.
Notes: LDAP limits the design. All 38,144 distinct LDAP rows were read; 39% conflict with other labels,
which leaves 13,890 training rows. That number is why every class contributes 13,890 rows.

## 4. Model: FT-Transformer-style tabular Transformer
- Per-feature tokens x_j W_j + b_j, protocol embedding, [CLS]; 3 PreNorm blocks, d = 32, 4 heads; 43,396 parameters.
Notes: not the original TabTransformer (which only attends over categorical columns). The class name in
the code is historical.

## 5. Method: frozen teacher, gate, repair
- T = E0 model, frozen (sha256 unchanged after every run).
- Gate on val: old-class recall >= ref - 0.05, new recall >= 0.80, benign FAR <= 0.05, new-class FAR <= 0.05.
- If C1 is rejected, R1 weights each replay row by p_T(stored label | x); if R1 also fails, keep the old model.

## 6. Result 1 [main]: the gate never fired
- Every C1 accepted; R1 never ran. Final test Syn recall 0.99540 (clean) vs 0.99498 (poisoned).
Notes: say plainly that R1 is untested on real data in this run.

## 7. Result 2 [main]: the poison was silent, not harmless
- Mean p(Syn) on true-Syn buffer rows: about 0.994 (clean A1) vs about 0.79 (poisoned A1). Argmax still Syn.
- Weighted-CE optimum predicts 0.824: the class weights keep Syn above 0.5.
- T gave the flipped rows p 0.004 against about 0.997-0.999 on clean rows (diagnostic: R1 never ran).
Notes: the gate measures recall only, so it cannot see falling confidence.

## 8. Pre-registered sweep (design)
- f in {0.75, 0.9, 0.95, 1.0} x seeds {42, 43, 44}; predicted p*_Syn 0.691 / 0.543 / 0.444 / 0.
- Decisions on val, committed to git before any test number.
Notes: show the commit hash of `experiments/prereg_sweep.md`.

## 9. Sweep: decisions [fill: deliverables/sweep_hypotheses.md, figures/sweep_decisions.png]
Notes: read the H1/H2 rows as "k of 3 seeds". If no C1 was rejected, say so.

## 10. Sweep: repair [fill: deliverables/sweep_decisions.md (R1 rows), sweep/diagnostics.csv]
Notes: separate the gate-triggered R1 (a decision) from the forced-R1 rows ("diagnostic, not the decision rule").

## 11. Sweep: test [fill: deliverables/sweep_test_summary.md, figures/sweep_test_syn.png]

## 12. All-labels run [fill: deliverables/all_labels_final_test.md, or "not run"]
Notes: 12 of 14 labels by a fixed rule (Portmap and WebDDoS have too few rows). Its scaler and scored rows
differ from main's, so it is not row-comparable to main.

## 13. Limits
- One 3-experience schedule with 13,890 rows per class; random flow-level splits (near-duplicates can cross splits).
- T knows only BENIGN vs Syn: it labels 99.7% of LDAP and 100% of UDP test rows BENIGN [main, A0 test confusion].
- Header-length artefacts (duplicate column; negative values in some classes): no ablation run.
- Base paper (CICIDS2017, different model, scaler and attack): numbers are not comparable.

## 14. Demo (2-3 minutes; `demo.py`, saved outputs only)
1. `python demo.py flow --label Syn` shows one test flow, its scaled input, and the predictions of T, clean and poisoned.
2. `python demo.py poison` shows the flipped buffer and T's probabilities.
3. `python demo.py gate --arm poisoned` shows every gate inequality with its margin.

## 15. Conclusion
Notes: claim only what the claims table in `deliverables/claims.md` marks as completed.
