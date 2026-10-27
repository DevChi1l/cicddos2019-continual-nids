# Slide outline with speaker notes (about 15-18 minutes)

**Sources.**
- Every number comes from `docs/RESULT_TABLES.md` (generated; verify with `python make_report_tables.py --check`) or
  from a committed README.
- **T#** points to a table in that file. **H§#** points to a section of `docs/PROJECT_TECHNICAL_HANDBOOK.md`.
- Never fill a slot from memory. Mark post hoc and diagnostic material as such on the slide itself.

---

## 1. Title: Poison-resilient continual intrusion detection on CICDDoS2019
Notes: a detector that keeps learning new attacks replays stored examples. An attacker who edits those stored labels
can make it forget, or never learn. Our question: can a validation gate plus a frozen-teacher repair keep it safe
and still learning?

## 2. Threat model and setting (H§1, H§5)
- CII: fresh BENIGN in every experience. E0 = BENIGN + Syn, E1 + LDAP, E2 + UDP.
- The attacker flips the stored label of a fraction f of the stored Syn rows to BENIGN, in the replay buffer only.
Notes: say what the attacker cannot touch: the new data, validation, test, the teacher, weights.

## 3. Data: audit first (T1, T2, H§3)
- 18 files, 70,427,637 rows, 14 labels; 59.9M within-file duplicates; BENIGN 0.16%.
- Filenames lie: 03-11 `UDPLag.csv` is mostly Syn, so the label column decides.
- Global dedupe before cap; 235,261 conflicting vectors dropped; E0-only frozen scaler; transformed duplicates
  excluded (166 rows).
Notes: we read all 70M rows once but trained on capped subsets: 98,833 rows for the four-label setting. No
full-data training claim.

## 4. Model (H§4, `docs/ARCHITECTURE.md`)
- An FT-Transformer-style tabular Transformer: 77 numeric tokens + protocol + [CLS] = 79 tokens of width 32.
- 3 PreNorm layers, 4 heads, 43,396 parameters (4 classes); the head grows when classes arrive.
Notes: not the original TabTransformer, which attends only over categorical columns. The code's class name is
historical.

## 5. Method: frozen teacher, gate, one repair (H§5-6)
- T = the E0 model, frozen and hash-checked.
- Gate on validation: old recall >= ref - 0.05, new >= 0.80, benign FAR <= 0.05, new-class FAR <= 0.05.
- If C1 is rejected, R1 restarts from the last accepted model with T-derived row weights (soft) or labels (relabel).
  If R1 also fails, keep the old model.

## 6. First run `main`: the poison was silent (T4)
- f = 0.5: every update accepted; test Syn 0.99498 (poisoned) vs 0.99540 (clean).
- Notebook 03 [diagnostic]: confidence fell (about 0.79) but the argmax stayed Syn. The weighted-CE optimum predicted
  0.824.
Notes: a recall gate cannot see falling confidence. This returns at slide 12.

## 7. Pre-registered sweep (T5)
- f in {0.75, 0.9, 0.95, 1.0} x seeds 42-44; decisions on validation, committed before the test tables.
- H1 not supported (rejections already at f = 0.75); H2, H4 and H6 supported.
- Test, f <= 0.95: repair learns all 4 classes in 9/9 runs vs 3/9 for the gate only.
- f = 1.0: soft repair fails (0/6); weighting can delete wrong labels but cannot supply correct ones.
Notes: disclose D1 (test computed inside the runs before the decisions commit, never shown or used) and D2.

## 8. PRIMARY RESULT: FU2 teacher relabelling at f = 1.0 (T6)
- All 3 seeds learn all four classes. Test means: Syn 0.9952, LDAP 0.9825, UDP 0.9983, benign FAR 0.0011.
- The gate alone and soft repair: LDAP = UDP = 0 (stale A0).
Notes: say "post hoc method, chosen after the sweep's diagnostic". E1 replays that diagnostic; the new evidence is
E2 and the test. Four labels, one attacked class, 3 seeds.

## 9. Scaling to 12 labels fails, and why (T7, T8)
- Clean 12-label updates are rejected too (DNS vs LDAP, SSDP vs UDP, MSSQL near its floor): 6 / 8 of 12 labels.
- FU1 retries learn DNS/LDAP but lose others: still 6 of 12.
- Joint oracle [diagnostic]: macro recall 0.82-0.83, but DNS/LDAP about 0.6 and UDP about 0.45 even jointly.
Notes: "feature limit for this model/recipe", not proof that the features are inseparable (1-NN reaches 0.84 on
DNS/LDAP). The 12-label continual goal is unsolved.

## 10. Repeated small corruption: two registered stops (T9)
- RC and RCS needed a clean sequence that passes the gate before poisoning: 0/3 seeds, for three schedules.
- Every clean rejection was a non-Syn class forgotten when an overlapping class arrived; Syn never moved.
Notes: we honoured the stopping rules and did not keep searching schedules.

## 11. RF: fixed-class traffic refresh (T9, T10)
- Post hoc: E0 = BENIGN + Syn + TFTP, six refresh updates, no new class. The strict clean pilot passed 3/3.
- 10% more flips per update gives 60% flipped stored Syn labels after six updates.

## 12. LIMITATION: confidence loss the gate does not see (T10)
- Test Syn recall 0.9943 (clean 0.9952), but mean Syn confidence 0.7707 (clean 0.9922).
- No update rejected, so no repair deployed; the confidence loss was **not** repaired.
Notes: RF1 (recall ratchet) not supported; RF2 (confidence loss) supported. Three labels, fixed classes.

## 13. Limits (H§1.4)
- Defended result = four labels, CII, one attacked class, 3 seeds, no significance claim; relabel is post hoc.
- No 12- or 14-label defence; RF is a three-label limitation study.
- Random flow-level splits; header-length and `Inbound` effects not ablated; the teacher knows only BENIGN/Syn.
- Reference paper: different dataset (CICIDS2017), model and attack, so its numbers are not comparable.

## 14. Demo (2-3 minutes; committed files only, CPU)
1. `python demo.py fu2 --seed 42`:
   - the attack (2,000/2,000 flipped);
   - every gate inequality, with C1 failing by 0.9453 and R1 passing;
   - 1,995 rows relabelled to their true class;
   - the test table against the controls.
2. Optional: `python demo.py rf` shows the confidence trajectory falling while recall and the gate do not react.
3. Show `python make_report_tables.py --check` printing `OK`.

## 15. Conclusion
Notes: claim only what the claims table in `experiments/results/SUMMARY.md` supports.
- A gate plus teacher repair defends a four-label continual NIDS against targeted replay-label flipping. Soft repair
  works up to 95%; post-hoc relabelling works at 100%.
- The gate is blind to confidence loss.
- The 12-label continual setting is unsolved.
