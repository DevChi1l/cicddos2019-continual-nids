# Five-minute speaking script and likely questions

Deck: `Poison_Resilient_Continual_NIDS_review.pptx`, with a PDF backup alongside. It is built from committed files by
`python docs/presentation/make_presentation.py`. The same notes are in each slide's speaker notes.

| Slide | Time | Say |
|---|---|---|
| 1. Title | 0:00-0:25 | "Can a gate plus a frozen teacher keep a replay-based intrusion detector safe *and* still learning when someone poisons its stored labels? Our main result covers four labels. With every stored Syn label flipped, teacher relabelling still learns all four classes on held-out test. The method is post hoc, and we claim four labels only." |
| 2. Problem | 0:25-0:55 | "Replay stops forgetting, but the buffer is an attack surface. The attacker flips stored labels of one old class, Syn to BENIGN, and touches nothing else. Inherited: CI/CII and the buffer-flipping threat from the reference paper, and the FT-Transformer tokenizer. Ours: the audited pipeline, the gate, the frozen teacher with one repair, and pre-registered tests." |
| 3. Data | 0:55-1:30 | "We audited all 70.4 million rows, but 59.9 million are within-file duplicates, and filenames lie, so the label column decides. After global dedupe and conflict removal, BENIGN is 0.16% and kept whole. We trained on capped subsets only: 98,833 rows in the four-label setting." |
| 4. Experiences | 1:30-1:55 | "E0 is BENIGN plus Syn. E1 adds LDAP, E2 adds UDP, each with fresh BENIGN rows: that is CII. Splits are fixed first, and the scaler is fitted on E0 only. 166 evaluation rows identical to a training input after scaling were excluded. The buffer keeps 2,000 rows per class from accepted experiences." |
| 5. Model | 1:55-2:20 | "Every numeric feature becomes a 32-dimensional token, plus a protocol token and a CLS token: 79 tokens. That is three PreNorm layers, four heads, and 43 thousand parameters. It is FT-Transformer-style, not the original TabTransformer." |
| 6. Protocol | 2:20-2:55 | "T is a frozen copy of the E0 model. C1 trains on new data plus the buffer. The gate checks validation only. If C1 fails, R1 restarts from the last accepted model and uses T on the buffer. If R1 fails too, we keep the old model. Only accepted data enters the buffer." |
| 7. FU2 validation | 2:55-3:35 | "In the pre-registered sweep, soft repair worked up to 95% flips, 9 of 9 runs against 3 of 9 for the gate alone, but failed at 100%. At 100%, plain C1 drops Syn to zero and is rejected in every seed. Relabel R1 restores 1,994-1,995 labels and passes at E1 and E2. E1 replays the earlier diagnostic, so E2 and the test are the new evidence." |
| 8. FU2 test | 3:35-4:05 | "On held-out test, relabelling gets Syn 0.995, LDAP 0.983, UDP 0.998 and a false-alarm rate of 0.0011, within about 0.01 of clean. The gate alone and soft repair get 0 on LDAP and UDP, and their zero false-alarm rate is not safety." |
| 9. 12 labels | 4:05-4:30 | "With 12 labels even clean updates fail the gate. DNS overlaps LDAP and UDP overlaps SSDP. The continual models keep 6 of 12 classes. A joint oracle is a diagnostic only. So there is no 12-label claim." |
| 10. RF | 4:30-5:00 | "In a post-hoc fixed-class study we flipped 10% more each update, 60% in total. Every update passed and repair never ran. Recall stayed at 0.994, but confidence fell to 0.77. Next step, as a proposal: a pre-registered confidence check in the gate." |

## Five likely questions

1. **Did you train on all 70 million rows?**
   No. All rows were read once for the audit and deduplication. Training used capped, deduplicated subsets: 98,833
   rows (four-label), at most 159,595.
2. **Isn't the relabel result cherry-picked, since you chose the method after seeing results?**
   The method is post hoc, and we say so. It was chosen from a validation diagnostic, then registered before its own
   run. E1 replays that diagnostic, so the new evidence is E2's validation decision and a once-only held-out test,
   scored after the validation commit.
3. **Why is a false-alarm rate of 0 bad for gate-only?**
   The gate rejected every update, so the stale E0 model stayed deployed. It calls LDAP and UDP traffic benign: zero
   false alarms, zero detection of the new attacks.
4. **Why not 12 labels?**
   Clean 12-label updates fail the gate. DNS/LDAP and SSDP/UDP overlap, and a rejected block is never retried. A retry
   recipe (FU1) did not grow coverage. A joint oracle with the same model also struggles on those pairs. So the
   defence is shown for four labels only.
5. **What is your novelty versus the reference paper?**
   The reference paper (CICIDS2017) showed that buffer label-flipping breaks replay, and proposed no defence. We add a
   validation gate, a frozen-teacher repair with a deployed relabelling variant, and pre-registered evaluation on an
   audited CICDDoS2019 pipeline. We do not claim a new architecture, and we do not claim the gate idea is new in the
   literature.

## Live demo (saved evidence, not fresh inference)

`python demo.py walkthrough` (or `python demo.py fu2`) reads committed gate decisions, teacher actions and test
tables from `experiments/results/`. It does not run the model on new flows. It needs only Python 3 with pandas and
numpy.

Fresh per-flow inference is `python demo.py flow`. That needs the `main` run's checkpoints and the Stage 1 parquet
files, which are not in git, plus torch.
