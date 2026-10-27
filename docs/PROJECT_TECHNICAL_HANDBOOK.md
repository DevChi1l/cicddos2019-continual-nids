# Project technical handbook: poison-resilient continual intrusion detection on CICDDoS2019

This is the authoritative walkthrough of the repository, from the raw CICDDoS2019 CSV files to the final claims. It
is written so that a reader new to the topic can follow it, and so that every number can be traced to a file in
git. Other documents:

| Document | Role |
|---|---|
| `experiments/results/SUMMARY.md` | the results summary and the "what we can and cannot claim" table |
| `docs/RESULT_TABLES.md` | generated tables; `python make_report_tables.py --check` verifies them |
| `docs/ARCHITECTURE.md` | the one-page model and update-loop diagram |
| `experiments/prereg_*.md` | the pre-registrations, each committed before its code or run |
| `experiments/results/*/README.md` | per-experiment provenance, checks and tables |

**How to read the evidence labels used below**

| Label | Meaning |
|---|---|
| **[val]** | a validation-set number; gate decisions and every selection used only these |
| **[test]** | a held-out test number, scored once on the deployed model; never used for any decision |
| **[pre-registered]** | design and predictions committed in git before the code or run |
| **[post hoc]** | chosen after seeing earlier results; reported as such |
| **[diagnostic]** | computed for understanding, never deployed and never a decision |
| **[stopped]** | the registered stopping rule ended the experiment before its main runs |
| **[our inference]** | an interpretation, not a measured fact |
| **[reported by paper X]** | a number or claim taken from someone else's paper, never our result |

There is no `AGENTS.md` in this repository; the project rules live in `CLAUDE.md`. Git times are commit times in
IST (UTC+05:30); Kaggle times in result READMEs are UTC.

---

## Contents

1. Research question, what we inherited, what we contributed, limits
2. Chronological history of notebooks 00-10
3. Raw data audit and exact preprocessing
4. The implemented Transformer
5. Glossary and the update protocol (T, A0, A, C1, R1, D, B, V)
6. The gate, the teacher and the three kinds of R1, with code locations and a worked example
7. Results matrix, and why accuracy alone misleads
8. Problems faced and fixes (timeline)
9. Related work
10. Viva questions and answers
11. What we would change next (proposal only)
12. Reproduce and verify: exact commands and their recorded output

---

## 1. Research question, inheritance, contribution, limits

### 1.1 The question

A network intrusion detection system (NIDS) that keeps learning new attack types must not forget the old ones. The
standard fix is **experience replay**: keep a small buffer of stored past examples and mix them into every update.
That buffer is an attack surface. Someone who can edit stored labels can teach the model that an old attack is
benign, without touching the new training data, the validation or test data, or the model's weights.

Our question: **if an attacker flips the stored labels of one old attack class (Syn -> BENIGN) in the replay
buffer, can a validation gate plus a frozen-teacher repair keep the detector both safe (old attack still detected)
and still learning (new attacks still learned)?**

### 1.2 What we inherited (not our contribution)

| Inherited | From | How we use it |
|---|---|---|
| Dataset | CICDDoS2019 (Sharafaldin et al., 2019), a Kaggle CSV copy of 18 files | the only data source |
| CI / CII scenarios | reference paper (Ariffin et al., arXiv 2608.04602) | CI: BENIGN only in E0; CII: fresh BENIGN in every experience. We use CII for every defended result |
| Replay, label flipping on the buffer, a joint oracle as an upper reference | the same reference paper | the threat model and the oracle idea; our attack, budget and target differ (section 9) |
| Feature-tokenizer Transformer with a [CLS] token | FT-Transformer (Gorishniy et al., 2021) | the model family (section 4) |
| Transformer self-attention | Vaswani et al., 2017 | the encoder layers (PyTorch `nn.TransformerEncoderLayer`) |
| Starter scripts for Stages 0-2 | a Claude website session (`nids_starter.zip`) | rewritten after auditing the real data; the website's claims were checked on synthetic data only |

### 1.3 What this project actually built and measured (our contribution)

1. **A leakage-controlled CICDDoS2019 pipeline:**
   - one streaming pass over all 70,427,637 rows;
   - global deduplication on model features, then label-conflict removal, then capping;
   - fixed splits;
   - an E0-only frozen scaler;
   - exclusion of evaluation rows whose scaled model input equals a training input.
2. **A gated continual-update protocol** with a **frozen E0 teacher** and **one repair attempt**, implemented and
   hash-audited (`s3_continual.py`):
   - the gate decides on validation only;
   - a rejected update keeps the last accepted model;
   - the buffer grows only after acceptance.
3. **A pre-registered poison-fraction sweep** (four labels, 3 seeds, f in {0.75, 0.9, 0.95, 1.0}) with held-out test
   scored after the decisions.
4. **A post-hoc follow-up (FU2): teacher relabelling** at f = 1.0. It is the **primary positive finding**. At f = 1.0,
   the gate alone and soft repair learn 0 of the 2 new classes in 3/3 seeds, while relabel repair learns all four
   classes in 3/3 seeds within about 0.01 of the clean arm [test].
5. **Honest negative and limiting results:**
   - the 12-label protocol does not scale under this gate (clean updates rejected);
   - a retry recipe does not fix it (FU1);
   - a joint oracle shows that the model/recipe itself struggles with DNS/LDAP and SSDP/UDP;
   - two repeated-corruption designs stopped at their clean pilots (RC, RCS);
   - a post-hoc fixed-class refresh study (RF) shows a **confidence loss the recall-based gate never sees**.

### 1.4 Scope limits (read before quoting any result)

- **Defended continual result = four labels only** (BENIGN, Syn, LDAP, UDP), CII, one attacked class (Syn), seeds
  42-44, no significance test. The relabel method is post hoc.
- **The 12-label continual goal is unsolved.** No configuration we tried learned all 12 labels continually, and no
  12-label or 14-label defence is claimed.
- **RF is a three-label, fixed-class refresh study, not a defence.** Repair never ran, so the confidence loss was not
  repaired.
- **No full-data training.** We read all 70,427,637 raw rows once for the audit and Stage 1, but training used capped,
  deduplicated subsets (section 3): at most 159,595 training rows (the 12-label oracle) and 98,833 for the four-label
  runs.
- **Random flow-level splits.** Exact duplicates cannot cross splits, but near-identical neighbouring flows can. This
  is not a chronological deployment test.

---

## 2. Chronological history of notebooks 00-10

**Change types:**
- a **design change** alters what is trained or how it is decided;
- a **data correction** alters which rows are used;
- a **reporting change** alters only how results are recorded or presented.

The table lists commits as `sha` and the result folder under `experiments/results/`.

| # | Notebook (code) | Date | What changed and why | Validation [val] | Held-out test [test] | Status |
|---|---|---|---|---|---|---|
| 00 | `00_audit` (`s0_audit.py`) | 10-20 | **Data audit** of the raw files before any modelling, because the starter's assumptions were untested | n/a | n/a | completed; `notebooks/manifest.json` |
| 01 | `01_splits_experiences` (`s1_preprocess.py`, `s1b_experiences.py`) | 10-20 | **Data corrections**: label column instead of filename, global dedupe before cap, conflict drop, non-finite drop, protected exclusions of transformed duplicates | n/a | n/a | completed; `01_data/` |
| 02 | `02_train_evaluate` (`s3_continual.py`), run `main` | 10-20 | **Initial design**: E0 Syn -> E1 LDAP -> E2 UDP, CII, poison f = 0.5 once before E1, gate + soft R1, seed 42 | every C1 accepted in all arms; R1 never ran | Syn 0.99540 clean vs 0.99498 poisoned | completed, single seed, not pre-registered; `02_main/` |
| 03 | `03_understand_our_model` (`s5_explain.py`) | 10-20 | **Reporting / diagnosis** only, read-only: why `main`'s poison was silent | confidence fell (about 0.79 on true-Syn buffer rows) while argmax stayed Syn; T gave flipped rows p about 0.004 [diagnostic] | n/a | completed; numbers in `docs/slides_outline.md` |
| 04A | `04_poison_sweep` (`s6_sweep.py`) | registered `eb6729f`, run 10-21 | **Design change**: stronger poison (f = 0.75-1.0), 3 seeds, forced-R1 diagnostics, decisions before test | H1 not supported; H2, H4, H6 supported; soft R1 passed 11/17 when it ran, 0/6 at f = 1.0 | f <= 0.95: repair learns all 4 classes in 9/9 runs vs 3/9 without; f = 1.0: 0/3 for both | **pre-registered**, completed; deviations D1, D2 (**reporting**); `04_sweep/` |
| 04B | same notebook, part B | 10-21 | **Design change**: 12 labels (two per update), one seed, f = 0.5 | clean updates rejected (CII E1/E3/E5, CI E1/E5); soft R1 rescued 0/6 | 6 (CII) or 8 (CI) of 12 labels learned | exploratory (not pre-registered), completed; `04b_all12/` |
| 05 | `05_followup_fu1` (`s9_followup.py`) | registered `66c88d4`, val `5307b58`, test `fc164b0` | **Design change**: retry rejected clean blocks with a validation-selected recipe (C2) and one deferred retry (D1) | `ep20` passes E1 5/6 probe cells; no recipe passes E5; deployed models know 6/12 labels | DNS/LDAP 0.83 learned, but MSSQL/NTP (and CI NetBIOS/SNMP) lost; macro recall lower | **pre-registered**, completed, negative; `05_fu1/` |
| 06 | `06_followup_fu2_relabel` (`s10_relabel.py`) | registered `0d9a807`, val `db4ca21`, test `12ac151` | **Design change [post hoc]**: deploy teacher relabelling as R1 at f = 1.0, seeds 42-44 | E1 (a replay of the sweep diagnostic) and E2: C1 rejected, relabel R1 accepted, 3/3 | all 4 classes in 3/3 seeds; LDAP 0.9825, UDP 0.9983, FAR 0.0011 (means) | registration committed before the run, **method chosen post hoc**; completed; **primary positive result**; `06_fu2/` |
| 07 | `07_oracle_joint` (`s11_oracle.py`) | registered `d69666d`, val `ba3a940`, test `7f09f11` | **Diagnostic design**: one model trained on all 12 labels at once (no updates, no gate) | DNS/LDAP and SSDP/UDP weak even jointly ("feature limit" for this model/recipe) | macro recall 0.816 / 0.829 (e10 / e20) vs continual 0.47-0.65 | **pre-registered**, completed; `07_oracle/` |
| 08 | `08_repeated_corruption` (`s12_repeated.py`) | registered `6e2de78`, code `0d2c31c` | **Design change**: 5% / 10% more Syn flips before each of 6 class-arrival updates | clean pilot: 0/3 seeds accepted all 6 updates (LDAP forgotten at SNMP/MSSQL) | none | **pre-registered, stopped** at the clean pilot; `08_rc/` |
| 09 | `09_repeated_corruption_stable` (`s13_rcs.py`) | registered `5e4ada3`, code `c33e961` | **Design change**: screen two fixed schedules for clean stability before any poison | both candidates stable in 0/3 seeds | none | **pre-registered, stopped** at the clean screen; `09_rcs/` |
| 10 | `10_refresh_corruption` (`s14_refresh.py`) | registered `b254211`, val `b0d6b63`, test `c7db851` | **Design change [post hoc]**: fixed classes (BENIGN, Syn, TFTP) with 6 refresh updates; 5% / 10% more flips per update | strict clean pilot 3/3; all 36 poisoned updates accepted; RF1 not supported, RF2 supported, RF3 supported | rho 0.10: Syn recall 0.9943 vs 0.9952 clean; Syn mean p 0.7707 vs 0.9922 clean | **post hoc**, registered before code, completed; **limitation**; `10_rf/` |

### 2.1 The story in words

1. **Audit before modelling (00).** The starter code had only been run on synthetic data, so the first step was to
   scan every real file.
   - The audit found 18 untruncated files, 70,427,637 rows and 14 labels.
   - It found 59,924,164 within-file exact duplicate rows.
   - BENIGN was only 113,828 rows (0.16%).
   - In the 03-11 day folder, filenames disagree with the label column (section 3.2).
2. **Clean, split, build experiences (01).** Stage 1 kept every distinct BENIGN row, capped each attack label at
   60,000 distinct rows, dropped label-conflicting vectors and split 60/10/10/20. Two points about Stage 1b:
   - **It first refused.** In CI, transformed duplicates (protected rows whose scaled input equals a training input)
     exceeded its 0.10% guard: 0.19% of calibration, 0.17% of validation and 0.18% of test rows. These counts were
     reported in the session and are not in a committed file.
   - **The fix was not to raise the guard.** We excluded exactly those rows, the same ids for CI and CII: 166 rows,
     with residual overlap 0 (`01_data/protected_exclusions.json`).
   - **The guard changed shape.** The original guard tolerated up to 0.10% overlap. The current code requires zero
     residual overlap after exclusion, and separately refuses if exclusion would remove more than 5% of any split and
     class (section 3.4).
3. **The first defended run, `main` (02).** E0 = BENIGN + Syn, then LDAP, then UDP; 13,890 attack rows per
   experience (LDAP's whole cleaned training count). Half the stored Syn labels were flipped.
   - **Initial hypothesis:** the gate would reject the poisoned update and R1 would fix it.
   - **What happened:** every C1 passed and R1 never ran. Test Syn recall was 0.99498 poisoned vs 0.99540 clean.
   - **Notebook 03 explained why** [diagnostic]. With class-weighted cross-entropy, the 1,000 surviving Syn labels
     still outweigh the flipped ones; the predicted optimum for true-Syn inputs is p = 0.824. So the poison lowered
     confidence (about 0.79) but not the argmax.
   - **Consequence:** the attack was "silent", and `main` did not test repair.
4. **Pre-registered sweep (04A).** To make the gate fire, f was raised to 0.75-1.0 over 3 seeds, with the hypotheses
   and thresholds committed first (`eb6729f`).
   - **Rejections came earlier than predicted.** At f = 0.75 one seed already rejected, so H1 was not supported.
   - **Repair worked for f <= 0.95**, learning all four classes in 9/9 runs [test].
   - **Repair failed at f = 1.0.** With every stored Syn label flipped, soft weighting can delete wrong labels but has
     no correct Syn label left to learn from (0/6 R1 accepted).
   - **The relabel diagnostic would have passed 3/3** [diagnostic]. Its thresholds were chosen after `main`.
   - **Two reporting deviations were disclosed.** D1: test was scored inside the runs before the decisions commit
     (never displayed or used). D2: the test tables come from a bit-identical re-execution.
5. **Scale-up to 12 labels (04B, exploratory).** One seed, 9,312 rows per attack label, two labels per update.
   - **Clean updates were rejected too**, so these rejections are not caused by the poison: DNS/LDAP and SSDP/UDP
     confusion, and an old MSSQL recall 0.0006 below its floor (CII clean).
   - s3 never retries a rejected block, so its labels are lost.
6. **FU1 (05, pre-registered).** Can a retry with a changed recipe recover the rejected clean blocks?
   - **E1 can be recovered.** 20 update epochs let DNS/LDAP pass.
   - **But the total does not grow.** Once DNS/LDAP were accepted, their own recall floors blocked the overlapping
     later blocks (MSSQL/NTP, NetBIOS/SNMP). The models still knew 6 of 12 labels, with lower macro recall [test].
     This is a negative result.
   - **An analysis correction.** A float32 1-NN "separability" file from 10-21 was found to be a numerical artifact
     and marked invalid. An exact float64 version replaced it (descriptive only).
7. **FU2 (06, post hoc, primary result).** Deploy relabelling as R1 at f = 1.0.
   - **E1 is not independent evidence**: it replays the sweep diagnostic exactly.
   - **The new evidence is E2 and the test.** Relabel R1 passed again at E2, and all four classes were learned in 3/3
     seeds on test [test].
8. **Joint oracle (07, pre-registered diagnostic).** Is the 12-label failure only forgetting?
   - **Partly not.** Trained jointly, the same model/recipe still gets DNS/LDAP about 0.6 and UDP about 0.45 [test].
     The registered rule calls this "feature limit for this model/recipe".
   - **Not proof of inseparable features.** Exact 1-NN reaches 0.84 on DNS/LDAP, and FU1 reached 0.83.
   - MSSQL, NTP, NetBIOS, SNMP and UDPLag are learnable jointly but were lost by the continual protocol.
9. **Repeated small corruption, RC and RCS (08, 09; both pre-registered, both stopped).** The aim was to test whether
   many small poisonings, each inside the gate's 0.05 allowance, ratchet Syn down.
   - **Both need a clean sequence that passes the gate first, and neither had one.** In 9 clean runs (13 rejected
     updates), every rejection was a non-Syn amplification class forgotten when an overlapping class arrived. Syn never
     moved by more than 0.001. Only three registered schedules were tried, so this is not a claim about every class
     order.
   - **Both stopped as registered.** No poisoned run was made, and no schedule search continued.
10. **RF (10, post hoc).** Because class arrival looked like the unstable part [our inference from those three
    schedules], RF fixes the class set (BENIGN, Syn, TFTP) and refreshes traffic six times.
    - **Strict clean pilot:** passed 3/3 seeds.
    - **At 10% more flips per update** (60% after six updates), every update passed and repair never ran. Syn recall
      stayed at 0.9943 [test], but mean Syn confidence fell to 0.7707 against 0.9922 clean [test].
    - **What it is:** a limitation finding, not a defence.

### 2.2 Design changes vs data corrections vs reporting changes (consolidated)

| Type | Items |
|---|---|
| Design changes | f grid and seeds (04A); two-label 12-label schedule (04B); retry policy C2/D1 (05); relabel R1 deployed (06, post hoc); joint oracle (07); per-update corruption (08); stability screen (09); fixed-class refresh and `s1b --allow-repeat` (10, post hoc) |
| Data corrections | label from the label column; global dedupe before cap; label-conflict drop; non-finite rows dropped; ids, ports, timestamp and Inbound dropped; transformed-duplicate exclusion (166 rows four-label; 444 all12; 70 RF) |
| Analysis corrections | FU1 float32 separability marked INVALID, exact float64 recomputed (decided nothing) |
| Reporting changes | sweep D1 (test computed before the decisions commit, disclosed) and D2 (bit-identical re-execution); later experiments defer test to a separate once-only, commit-checked command; result zips committed unchanged; the generated tables and demo in this commit |

No held-out test data was ever altered. No gate threshold, training recipe or repair method was tuned on a test
number: the relabel thresholds came from `main`'s validation diagnostic, and FU1's recipe from validation probes.

**One disclosed exception concerns class selection.** RC's class set excluded DNS, UDP, UDPLag and NetBIOS. That
decision used the committed 12-label, FU1 and oracle results, test tables included
(`experiments/prereg_repeated_corruption.md`, "Class choice"). RCS inherited the class set, and RF chose TFTP from
the RC/RCS clean validation runs and those same earlier tables (`experiments/prereg_refresh.md`). Each of these
experiments then scored its own test once, after its own validation commit.

---

## 3. Raw data audit and exact preprocessing

All counts in this section come from `notebooks/manifest.json` (the audit), `experiments/results/01_data/` (Stage 1
and the four-label Stage 1b), `experiments/results/04b_all12/schedule.json` (12-label) and
`experiments/results/10_rf/ref6/cii/experience_manifest.json` (RF). `docs/RESULT_TABLES.md` tables T1-T3 regenerate
them.

### 3.1 Five different label sets (do not confuse them)

| Name | Labels | Where used |
|---|---|---|
| **18 source files** | two day folders: 01-12 (11 files) and 03-11 (7 files) | the raw input; filenames are not trusted |
| **14 raw labels** | BENIGN, DNS, LDAP, MSSQL, NTP, NetBIOS, Portmap, SNMP, SSDP, Syn, TFTP, UDP, UDPLag, WebDDoS | audit, Stage 1, `class_map.json` |
| **12-label experimental subset** | the 14 minus Portmap (363 train / 60 val rows) and WebDDoS (132 / 22), excluded by a fixed rule (min 1,000 train, 200 val) | 04B all12 runs, FU1, the joint oracle |
| **four-label main defence** | BENIGN, Syn, LDAP, UDP (E0 BENIGN+Syn, E1 +LDAP, E2 +UDP) | `main`, the sweep, **FU2 (primary result)** |
| **three-label RF study** | BENIGN, Syn, TFTP, with no new class after E0 | RF only |

### 3.2 Stage 0: the audit (`s0_audit.py`, notebook 00)

- **Size.** 18 CSV files, 31.06 GB, **70,427,637 rows**, 88 columns. Every file was scanned in full
  (`scan_truncated: false`).
- **Labels (label column, canonicalised by `common.canon_label`, e.g. `DrDoS_DNS` -> `DNS`).**

  | Label | Raw rows | Label | Raw rows |
  |---|---|---|---|
  | TFTP | 20,082,580 | DNS | 5,071,011 |
  | MSSQL | 10,309,945 | LDAP | 4,095,052 |
  | NetBIOS | 7,750,776 | SSDP | 2,610,611 |
  | UDP | 7,001,800 | NTP | 1,202,642 |
  | Syn | 6,473,789 | UDPLag | 368,334 |
  | SNMP | 5,159,870 | Portmap | 186,960 |
  | **BENIGN** | **113,828** | WebDDoS | 439 |

  BENIGN is 0.16% of the rows: 56,863 in 01-12 and 56,965 in 03-11.
- **Within-file exact duplicate rows: 59,924,164.** The data is mostly copies.
- **Non-finite values** occur only in `flow bytes/s` and `flow packets/s`: 2,170,750 rows (1,097 of them BENIGN).
- **Filenames are not labels.** In the 03-11 folder:
  - `UDPLag.csv` holds Syn 606,749, UDP 112,475, BENIGN 4,068 and UDPLag only 1,873;
  - `LDAP.csv` holds NetBIOS 202,919;
  - `MSSQL.csv` holds LDAP 9,931;
  - `UDP.csv` holds MSSQL 24,392.

  Every file also contains some BENIGN rows (table T2). **Decision:** the label column decides the class (data
  correction). We cannot tell from the files whether the filename or the label is "wrong"; trusting the label column
  is a documented assumption.
- **Leakage-prone columns.** The audit found 6 id-like columns: `flow id`, `source ip`, `destination ip`,
  `timestamp`, `unnamed: 0`, `simillarhttp`. There are also source and destination ports, and `Inbound`. For
  `Inbound`, 21,331 of 113,828 BENIGN rows have value 1, while almost every attack row has 1. That makes it a likely
  label giveaway, so it is dropped (no ablation was run).
- **The dataset's own documentation is ambiguous** about which capture day is "training" and which is "testing". The
  UNB page's text and its table disagree, so we do not use the day split.

### 3.3 Stage 1: one streaming pass (`s1_preprocess.py:45 stream_dedupe_cap`, notebook 01)

Arguments (`01_data/split_manifest.json`): `cap 60000`, `split random`, `nonfinite drop`, `chunksize 500000`,
seed 42. The 30 GB were never loaded into memory at once. Order of operations:

1. **Read in 500,000-row chunks**, normalise column names, and map labels from the label column.
2. **Fix the feature set from the first chunk.** Drop the id columns, ports, `Inbound` and the label, keep `protocol`
   as categorical, and keep every other numeric column. That gives **77 numeric + 1 categorical** model inputs
   (`01_data/features.json`).
3. **Non-finite to NaN, and drop the row** (`nonfinite drop`). The ReplayIDS release zero-fills instead; ours does
   not.
4. **Hash the model features.** `h = hash(77 numeric + protocol, cast to float64)`
   (`s1_preprocess.py:41 feature_hash`). Equal h means an identical model input, in any file.
5. **Global deduplication on (h, label)**, across all files: 59.9M within-file copies collapse. Distinct finite
   counts include, for example, BENIGN 95,381, LDAP 38,144 and NetBIOS 25,511 (table T1).
6. **Label conflicts.** 235,261 feature vectors appear under two or more labels; **all are dropped** (we cannot know
   which label is right). The largest are UDP 191,822 and SSDP 190,604; LDAP has 14,994.
7. **Cap by bottom-k.** Per attack label, keep the 60,000 distinct rows with the smallest h: a deterministic,
   file-order-independent uniform sample of distinct rows. **BENIGN is never capped.**
8. **What survives** (after the conflict drop): BENIGN **95,272**; LDAP 23,150; NetBIOS 15,520; most other attack
   labels 43,723-59,950; Portmap 606; WebDDoS 220. **Cross-split identical inputs: 0** (asserted).
9. **Split per class** with a seeded permutation: 60% train / 10% calibration / 10% validation / 20% test
   (`split_indices`, line 126). The splits are random at flow level, not time-based.

   | Split | BENIGN | Syn | LDAP | UDP | TFTP |
   |---|---|---|---|---|---|
   | train | 57,163 | 35,838 | 13,890 | 32,929 | 35,970 |
   | val | 9,527 | 5,973 | 2,315 | 5,488 | 5,995 |
   | test | 19,055 | 11,946 | 4,630 | 10,978 | 11,990 |

10. **Saved unscaled**, with a row id `_rid` and the source file for provenance. The calibration split is created and
    protected, but no code uses it (all thresholds are fixed constants).

### 3.4 Stage 1b: experiences, scaler, protected exclusions (`s1b_experiences.py`)

1. **Experiences** (`build_manifest`, line 58, seed 42).
   - Each attack class in an experience gets at most `attack_cap` training rows, chosen without replacement.
   - **CII**: BENIGN training rows are permuted and split into one disjoint, near-equal portion per experience.
   - **CI**: all BENIGN goes to E0.
   - With `--allow-repeat` (RF only, off by default), a class may appear in several experiences. Its rows are
     permuted once and split into disjoint portions.
2. **Scaler fitted on E0 training rows only, then frozen** (`fit_scaler`, line 104):
   - signed log1p, `slog(x) = sign(x) log(1 + |x|)`;
   - then `(x - mean) / std`;
   - then clip to [-5, 5];
   - a std below 1e-3 is floored to 1.0, which happens for 12 features constant in E0, such as `fin flag count`;
   - the protocol vocabulary comes from E0 as well: {0, 6, 17} map to ids 1, 2, 3, and anything else to 0 ("unseen").

   Later experiences are transformed with this frozen scaler. That is realistic (a deployed detector cannot refit on
   future data), and it is the reason clipping matters.
3. **Protected exclusions** (`protected_exclusions`, line 275), a data correction. A calibration, validation or test
   row is excluded if its **model input after the transform** equals any training row's input, under either
   scenario's scaler.
   - Clipping, float32 rounding and -0.0 == 0.0 can make distinct raw rows identical after scaling.
   - The same row ids are removed for CI and CII, so both scenarios score identical rows.
   - The code refuses if any split or class would lose more than 5%, or fewer than 5,000 benign validation or test
     rows would remain. A residual check must find 0 overlaps.
   - Excluded: 166 rows (four-label), 444 (all12), 70 (RF).
4. **The four-label experiences** (`01_data/cii/experience_manifest.json`). `attack_cap` = 13,890, the full cleaned
   LDAP training count, so every attack class contributes the same number of rows.

   | Experience | Training rows |
   |---|---|
   | E0 | BENIGN 19,055 + Syn 13,890 |
   | E1 | BENIGN 19,054 + LDAP 13,890 |
   | E2 | BENIGN 19,054 + UDP 13,890 |

   - **Total: 98,833 training rows.**
   - **Scored validation rows:** BENIGN 9,507, Syn 5,973, LDAP 2,296, UDP 5,487 (23,263).
   - **Scored test rows:** 46,527, of which 19,028 are BENIGN.
5. **12-label (all12).** `attack_cap` = 9,312 (NetBIOS's training count).
   - CII benign is split into 6 portions of 9,527-9,528 rows.
   - **Total: 159,595 training rows; 123,281 test rows.**
6. **RF (`ref6`).**
   - E0: BENIGN 8,167 + Syn 9,312 + TFTP 5,139.
   - E1..E6: BENIGN 8,166 + TFTP 5,138-5,139 each. TFTP's 35,970 training rows are split into 7 disjoint portions.
   - **Scored test rows: 42,962.**

### 3.5 Replay buffer sampling (`s3_continual.py:263 buffer_add`)

- After an experience is **accepted**, up to `mem_per_class` = **2,000** of its training rows per class are added.
  They are sampled without replacement (generator seed + 1, the same draws in every arm), with the **true** label
  stored.
- After E0 the buffer holds 2,000 BENIGN + 2,000 Syn. In CII each accepted update adds 2,000 BENIGN and 2,000 of each
  new class.
- **A rejected experience adds nothing.**
- The attacker (`poison`, line 277) flips the stored label of `round(f * 2000)` Syn rows to BENIGN, once before E1.
  In RC and RF (`poison_more`, line 289), it flips `rho * 2000` more not-yet-flipped rows before every update.
- Flips persist: repair never writes back to the buffer.

---

## 4. The implemented Transformer (`s2_model.py`)

One-page version: `docs/ARCHITECTURE.md`. Dimensions are code-derived. They come from building the model with the
committed scaler (77 numeric features, protocol vocabulary of 3 values + "unseen") and the `s3_continual.py`
defaults: `d` 32, `heads` 4, `layers` 3, `dropout` 0.1.

### 4.1 Forward pass with tensor shapes (batch of N flows)

| Step | Code | Shape |
|---|---|---|
| Numeric input after the frozen scaler | `x_num` | (N, 77), float32 |
| Protocol id | `x_cat` | (N, 1), int64, values 0-3 |
| Numeric tokens: `z_j = x_j * W_j + b_j`, one learned (W_j, b_j) per feature | `FeatureTokenizer.forward`, line 42 | (N, 77, 32) |
| Protocol token: `nn.Embedding(4, 32)[id]` | line 45 | (N, 1, 32) |
| Learned [CLS] token, prepended | line 43 | (N, 1, 32) |
| Token sequence | `torch.cat` | **(N, 79, 32)** |
| 3 encoder layers, PreNorm (`norm_first=True`) | `nn.TransformerEncoder`, lines 55-58 | (N, 79, 32) |
| Flow embedding = final [CLS] position, then LayerNorm | `embed`, line 62 | (N, 32) |
| Classifier `Linear(32, K)` | `head` | (N, K) logits |

**Inside one encoder layer** (PyTorch `TransformerEncoderLayer`, `norm_first=True`, GELU):
- the layer computes `h = h + Dropout(MHSA(LayerNorm(h)))`, then `h = h + Dropout(FFN(LayerNorm(h)))`;
- the FFN is `Linear(32, 128) -> GELU -> Dropout -> Linear(128, 32)`;
- self-attention uses 4 heads of 8 dimensions each. The fused input projection `in_proj_weight` has shape (96, 32):
  32 rows each for Q, K and V;
- for each head, Q, K and V are (N, 79, 8), and the attention matrix `softmax(Q K^T / sqrt(8))` is (N, 79, 79);
- the outputs of the 4 heads are concatenated to (N, 79, 32) and projected by `out_proj` (32 x 32).

**Parameters** (counted from the code):
- 43,363 with K = 3 (RF), 43,396 with K = 4 (four-label), 43,660 with K = 12;
- the tokenizer has 77 x 32 x 2 + 32 + 4 x 32 = 5,088;
- each encoder layer has 12,704;
- the final LayerNorm has 64, and the head has 33 K.

**Head expansion** (`expand_head`, line 71). When new classes arrive, a larger `Linear(32, K_new)` is created and
the old rows' weights and biases are copied, so the old logits are unchanged before training. The order of output
units is arrival order, stored in every checkpoint.

**Loss and training** (`s3_continual.py:150 train`):
- weighted cross-entropy, with per-sample weight = class weight x row weight, normalised by the weight sum in each
  batch;
- class weight `c_k = sqrt(N / (K n_k))`, computed from the stored training labels, so scarce classes count more;
- row weight `w` = 1, except in R1 (section 6);
- AdamW, lr 1e-3, weight decay 1e-4, batch 512; fp16 autocast on GPU; 10 epochs for E0 and 5 per update.

### 4.2 Comparison with the published models

| Aspect | Ours (code) | FT-Transformer (Gorishniy et al., 2021; verified from the paper) | Original TabTransformer (Huang et al., 2020; verified from the paper) | Reference paper's model (Ariffin et al., 2026; as described in its text) |
|---|---|---|---|---|
| Numeric features | each becomes a token `x_j W_j + b_j` | **same form**: `T_j = b_j + x_j W_j` | not tokenised: concatenated with the Transformer output, then fed to an MLP | "concatenated with the numerical features after layer normalisation"; exact numeric embedding not specified |
| Categorical features | `nn.Embedding` per column (no separate bias) | lookup table + per-feature bias | column embedding with a shared column identifier (concatenated) | embedded into learned vectors |
| Pooling | **[CLS] token first** | [CLS] token (stacked first) | none (concatenate all column outputs) | "pooled transformer output", method not stated |
| Block | PreNorm, GELU, FFN 4x | PreNorm, ReGLU, FFN 4/3; first LayerNorm of the first layer removed | post-norm ("after each layer") | 6 blocks; heads and dropout listed in its Table 5 |
| Head | LayerNorm -> Linear | Linear(ReLU(LayerNorm(CLS))) | MLP {4l, 2l} | MLP head (one hidden layer in Section 3.3; 4x and 2x in Section 3.5) |
| Width / depth | d 32, 3 layers, 4 heads | default d 192, 3 layers, 8 heads | d 32, 6 layers, 8 heads | embedding dimension 32 |
| New classes | append output rows, copy old weights | n/a | n/a | append output units, copy existing weights (Section 3.3) |

**Verified similarities** (the papers' equations were checked against our code):
- our numeric tokenizer and [CLS] pooling have the form of FT-Transformer's Feature Tokenizer;
- our PreNorm block matches FT-Transformer's choice of PreNorm.

**Differences**:
- GELU instead of ReGLU;
- no removal of the first-layer LayerNorm;
- no ReLU before the head;
- a much smaller width;
- signed-log + standardisation + clip preprocessing instead of FT-Transformer's default quantile transform.

**Conceptual inspiration only, not code-level**:
- the name `TabTransformer` and the general "Transformer for tabular IDS" framing come from the reference paper;
- the head-expansion idea matches its text.

We did not inspect or run the reference code's model, and we claim no reproduction of it. The accurate name for our
model is **"an FT-Transformer-style tabular Transformer (small, custom configuration)"**.

---

## 5. Glossary and the update protocol

### 5.1 Basic terms

| Term | Meaning in this repository |
|---|---|
| **Flow** | one row: CICFlowMeter's summary statistics of one bidirectional network connection (packet counts, sizes, timings, flags), plus a label |
| **Experience E_t** | the t-th block of training rows that arrives over time. E0 is the initial block; E1, E2, ... are updates. The number t is the **experience number** |
| **CI / CII** | class-incremental: BENIGN only in E0. Class-instance-incremental: fresh, disjoint BENIGN rows in every experience. All defended results use CII |
| **Arm** | one parallel version of the same run under one condition: `clean`, `poisoned`, `poisoned_repair` (FU2 adds `poisoned_relabel`). All arms share T, A0 and the buffer draws |
| **Recall of class k** | the share of class-k rows predicted as k |
| **Benign FAR** | false-alarm rate: the share of BENIGN rows predicted as any attack |
| **New-class FAR** | the share of rows that are not of the new classes but are predicted as a new class |
| **Mean p(k)** | the model's average softmax probability of the true class k on class-k rows: a confidence measure |

### 5.2 The protocol objects

| Symbol | What it is | Where |
|---|---|---|
| **D** (D_t) | the new training rows of experience t: its new classes plus, in CII, a fresh BENIGN portion. The attacker cannot touch D | `ctx["exp"][t]` |
| **B** | the replay buffer: 2,000 stored rows per class per accepted experience, with stored label, true label and a `flipped` flag. The flag is used for diagnostics only, never for training | `buffer_add`, line 263 |
| **V** | the protected validation rows (after exclusions). The gate reads only V. Test rows are never read by any decision | `load_context` |
| **T** | the frozen teacher: the E0 model saved as `T.pt`. **Never trained again.** Its parameters have `requires_grad=False` (line 378), and its state hash is checked equal before, after and on disk (lines 611-624). It knows only BENIGN and Syn | `initial_round`, line 356 |
| **A0** | the first accepted model: the **same weights** as T at the start, saved separately as `A0.pt`. A0 is the starting point that gets updated; T is the copy that is never updated | line 371 |
| **A** (A_t) | the **current deployed model**: the last accepted model after experience t. If an update is rejected, A_t = A_{t-1} | `run_arm` |
| **C1** | the first **candidate** for an update: a copy of A with new head rows, trained on D_t + B with plain stored labels | `attempt(None)`, line 419 |
| **R1** | the repair candidate, only in a repair arm and only after C1 is rejected. It **restarts from A, not from C1**, is trained on D_t + B with teacher-derived row weights or labels, and is gated again | `attempt(repair_method)` |
| **Candidate number** | C1, R1 (and FU1's C2, D1) name *attempts* within one experience. They are not experiences and not extra deployed models. Experience E2's C1 is a different object from E1's C1 | |

### 5.3 What happens at each outcome

```
experience t arrives (D_t)       (repair arms only: if C1 is rejected, R1 is tried)

C1 = copy(A) + new head rows, train on D_t + B
   |
   gate(C1) on V passes?
      |-- yes -> DEPLOY C1 as A_t; B += 2,000 rows/class of E_t (true labels)
      |-- no, arm has no repair -> KEEP A (A_t = A_{t-1}); B unchanged; E_t's classes are never learned (s3 never retries)
      |-- no, repair arm ---------> R1 = copy(A) + new head rows, train on D_t + B with T's weights/labels
                                       gate(R1) passes?
                                         |-- yes -> DEPLOY R1 as A_t; B += rows of E_t (true labels; R1's relabels are NOT written to B)
                                         |-- no  -> KEEP A; B unchanged
```

- **Deployed:** after every experience exactly one model is deployed, either the accepted candidate or the previous
  model.
- **Entering the buffer:** only rows of *accepted* experiences, always with their true label. The attacker's flips
  already in B stay there.
- **Experience vs candidate number:** "E2 R1" means the repair attempt for experience 2.

### 5.4 FU1's C2 and D1 are retry recipes, not extra models

FU1 (`s9_followup.py`, docstring and `policy`, line 188) replaces "a rejected block is lost for good" with:
- **C2** is one retry of the *same* block, from the same A and B, with a recipe selected on validation (FU1 chose
  `ep20`, 20 update epochs, for new-class failures).
- **D1** is one deferred retry of a still-rejected block, made right after the next accepted block (or at the end).

Both are gated candidates like C1. If accepted, the candidate becomes the next A; if not, it is discarded. At most
three attempts per block, and **there is never more than one deployed model**. FU1 is a separate experiment; s3
itself never retries.

---

## 6. The gate, the teacher and the three kinds of R1

### 6.1 Gate inequalities (`s3_continual.py:230 gate`)

The reference `ref` is the **last accepted model** evaluated on the same validation rows.
1. For every old attack class k (BENIGN excluded): `recall_C(k) >= recall_ref(k) - 0.05`.
2. For every new class n: `recall_C(n) >= 0.80`.
3. `benign FAR(C) <= 0.05`.
4. `new-class FAR(C) <= 0.05`.
5. Optional, **off in every reported run**: `mean p_C(k) >= mean p_ref(k) - d` (`--gate-conf-drop`). Mean p is
   always logged.

The candidate is accepted only if all conditions hold. The margin of each condition is its slack (value minus bound
for ">=", bound minus value for "<="). `s5_explain.gate_check` (line 139) recomputes every inequality from the saved
metrics, and asserts that the result equals the saved decision.

**Worked example** (real numbers; FU2 seed 42, from `06_fu2/fu2/.../poisoned_relabel/rounds.json`, printed by
`python demo.py fu2`):

| E1 attempt | old Syn: recall >= 0.9953 - 0.05 = 0.9453 | new LDAP >= 0.80 | benign FAR <= 0.05 | new-class FAR <= 0.05 | decision |
|---|---|---|---|---|---|
| C1 (plain stored labels) | 0.0000: **fails by 0.9453** | 0.9869 | 0.0007 | 0.0006 | rejected |
| R1 (teacher relabel) | 0.9953: passes by 0.0500 | 0.9874 | 0.0003 | 0.0002 | **accepted** |

With all 2,000 stored Syn labels flipped and no Syn rows in D1, plain training drives Syn recall to 0. The gate
catches this, because Syn's floor is relative to the last accepted model (0.9953).

### 6.2 Why a gate can reject a *clean* update

The gate does not know whether an update is poisoned. It only compares candidate and reference, so it rejects a clean
update whenever learning the new class costs an old class more than 0.05 of recall, or the new class cannot reach
0.80. Examples:
- **In the 12-label runs:** new DNS reached only 0.58-0.73 at E1 (all arms, both scenarios; < 0.80), because 26-41%
  of DNS rows were predicted LDAP;
  and adding UDP made 98-99% of old SSDP rows predicted UDP.
- **In RC and RCS:** adding SNMP sent 39-47% of old LDAP rows to SNMP.

These are forgetting and class-overlap failures, not attacks. In the 12-label runs a clean MSSQL update was rejected
by a margin of 0.0006 (0.9390 vs a floor of 0.9396). The gate is a hard threshold, so near-threshold decisions are
sensitive to training noise [our inference].

### 6.3 The teacher and how R1 uses it

- **T is updated never.** It is trained once on E0 (`initial_round`). Every later use only reads it with
  `predict_proba`, and the run asserts that its hash is unchanged (`T_unchanged: true` in every `run_manifest.json`).
  It knows only BENIGN and Syn, so it can judge only buffer rows whose stored label is BENIGN or Syn. All other rows
  keep weight 1 and their label.
- **Soft R1 (registered;** `teacher_weights`, line 300; `repair_rows` with method `soft`): row weight
  `w = p_T(stored label | x)`.
- **Three-way R1** (registered as a diagnostic option; `repair_rows`, line 319), with `ps = p_T(stored label | x)`:
  - `ps >= 0.9` -> weight 1;
  - `ps <= 0.1` -> weight 0;
  - otherwise weight 0.5.
- **Relabel R1** (`threeway+relabel`; **FU2's deployed method, post hoc**): for a T-known row with `ps <= 0.1` and
  `max_k p_T(k | x) >= 0.9`, the *training* label becomes `argmax_k p_T(k | x)` with weight 1. The other rows follow
  the three-way rule. The relabel is applied only to that R1's training batch, never written to B, so the flipped
  labels remain in the buffer for the next update.

**Small numerical example** (illustrative arithmetic using the sweep's *reported mean* weights, H4: flipped rows
about 0.003, clean Syn rows about 0.998):
- **At f = 0.75**, B holds 1,500 flipped rows (stored BENIGN, truly Syn) and 500 correct Syn rows.
  - Soft R1's effective weights: 1,500 x 0.003 = about 4.5 "BENIGN" votes from the flipped rows, against
    500 x 0.998 = about 499 Syn votes.
  - Class weighting then up-weights the small Syn class. The wrong labels are effectively deleted, and Syn survives.
  - Observed [val]: Syn recall back above its floor in all 11 sweep R1 attempts at f <= 0.95.
- **At f = 1.0** there are 0 correct Syn rows. Soft R1 removes the 2,000 wrong labels (2,000 x 0.003 = about 6
  votes), leaving essentially **no Syn label at all**.
  - Observed [val]: R1 Syn recall 0.000-0.0002 in 6/6 attempts, all rejected.
  - Weighting can delete wrong labels but cannot supply correct ones.
- **Relabel R1 at f = 1.0.** T predicts Syn with p >= 0.9 for 1,994-1,995 of the 2,000 flipped rows, so those rows
  train as Syn.
  - Observed [val]: the FU2 summary reports 1,994-1,995 relabelled rows per R1 across the three seeds, all to their
    true class; seed 42 relabelled 1,995 at both E1 and E2.
  - Result: acceptance at E1 and E2 in 3/3 seeds (seed 42: Syn recall 0.9953 at both).

**Why class weighting hid the f = 0.5 attack** (pre-registered prediction; `prereg_sweep.md`). Assume the model cannot
tell flipped rows from unflipped ones. The weighted cross-entropy optimum for true-Syn inputs at E1 is then
`p*_Syn = c_S n_S / (c_S n_S + c_B n_flipped)`.
- At f = 0.5, with c_S = 3.509, n_S = 1,000, c_B = 0.747 and n_flipped = 1,000, this is 0.824 > 0.5. The argmax stays
  Syn and recall does not move, while confidence falls (notebook 03 measured about 0.79).
- The registered predictions were 0.691, 0.543 and 0.444 at f = 0.75, 0.9 and 0.95, and 0 at 1.0.

### 6.4 Ordinary R1, forced diagnostic R1 and FU2's relabel R1 (do not mix them up)

| | Ordinary R1 | Forced diagnostic R1 | FU2 relabel R1 |
|---|---|---|---|
| Arm | `poisoned_repair` | `clean` and `poisoned` (or `poisoned` only in RC and RF) | `poisoned_relabel` |
| When | only after C1 is rejected | at every update, regardless of C1 | only after C1 is rejected |
| Method | soft (registered) | soft, threeway, threeway+relabel | threeway+relabel |
| Deployed? | yes, if it passes the gate | **never**; logged to `r1_diag.json` | yes, if it passes the gate |
| Status | registered decision rule | [diagnostic] | **[post hoc]** method, registered as FU2 before its run |
| Code | `run_arm` | `--forced-r1`, `diag_methods` | `s10_relabel.py` via `S3.run_arm` |

---

## 7. Results matrix

Full numbers: `docs/RESULT_TABLES.md` (T4-T10) and `experiments/results/SUMMARY.md`. "Learned" means test recall
> 0 for that class in the deployed model.

| Experiment | Labels | Scenario | Seeds | Poison schedule | Val / test status | Strongest supported claim | Failure or limitation | Source artifact |
|---|---|---|---|---|---|---|---|---|
| `main` (02) | 4 | CII | 42 | Syn -> BENIGN, f = 0.5, once before E1 | val decisions; test scored inside the run | at f = 0.5 every update passes and the poison does not change Syn recall (test 0.99498 vs 0.99540 clean) | silent: confidence fell (notebook 03); repair never exercised; one seed | `02_main/` |
| Sweep (04A) | 4 | CII | 42-44 | f in {0.75, 0.9, 0.95, 1.0}, once | **pre-registered**; val committed `eebc990`, then test (D1, D2 disclosed) | soft R1 restores learning when some correct Syn labels survive: all 4 classes in 9/9 repair runs vs 3/9 gate-only at f <= 0.95 [test] | soft R1 fails at f = 1.0 (0/6); the gate alone protects Syn but stops learning | `04_sweep/` |
| **FU2 (06)** | **4** | CII | 42-44 | f = 1.0, once | registered `0d9a807`; val `db4ca21`, then test once | **primary positive result**: relabel R1 learns all 4 classes in 3/3 seeds at f = 1.0, Syn 0.9952 / LDAP 0.9825 / UDP 0.9983 / benign FAR 0.0011 (means), within about 0.01 of clean; gate-only and soft R1: LDAP = UDP = 0 [test] | **post hoc** method (from the sweep diagnostic); E1 replays that diagnostic; teacher knows only BENIGN/Syn; one attacked class; 3 seeds | `06_fu2/` |
| all12 (04B) | 12 | CII and CI | 42 | f = 0.5, once | exploratory; test inside the run | none positive: clean updates are rejected too | 6 (CII) / 8 (CI) of 12 labels learned; soft R1 rescues 0/6; one seed | `04b_all12/` |
| FU1 (05) | 12 | CII and CI | 42 (start) | clean arm only | **pre-registered**; val `5307b58`, then test once | DNS/LDAP can be learned (test 0.83) with 20 update epochs | label count does not grow (6/12); macro recall falls (0.47); SSDP/UDP never passes | `05_fu1/` |
| Oracle (07) | 12 | CII (3 seeds), CI (1) | 42-44 | none | **pre-registered**; val `ba3a940`, then test once | joint training reaches all 12 labels with macro recall 0.816-0.829 [test]; MSSQL, NTP, NetBIOS, SNMP and UDPLag are learnable jointly | DNS/LDAP about 0.6 and UDP about 0.45 even jointly ("feature limit" for this model/recipe, not proof of inseparability); no gate, so not a method | `07_oracle/` |
| RC (08) | 7 (BENIGN, Syn + 6) | CII | 42-44 | planned: 5% / 10% more per update | **pre-registered, stopped** at the clean pilot; no test | none (RC1-RC4 untested) | the clean schedule was not stable: 0/3 seeds | `08_rc/` |
| RCS (09) | 6-7 | CII | 42-44 (screen) | planned, as RC | **pre-registered, stopped** at the clean screen; no test | none | both candidate schedules stable in 0/3 seeds | `09_rcs/` |
| RF (10) | 3 | CII, fixed classes | 42-44 | 5% / 10% more of the stored Syn labels before each of 6 refresh updates (30% / 60% total) | registered `b254211` (**post hoc**); val `b0d6b63`, then test once | **limitation**: 60% flipped Syn labels left test recall at 0.9943 (clean 0.9952), but mean Syn confidence fell to 0.7707 (clean 0.9922); RF2 supported | no rejection, so no repair deployed; the confidence loss was **not** repaired; RF1 (recall ratchet) not supported; fixed-class refresh, not new-class arrival | `10_rf/` |

### 7.1 Why overall accuracy alone is misleading here

Four-label test rows, f = 1.0, seed 42. Accuracy is computed from the per-class recalls in
`06_fu2/test/test_recall.csv` and the test supports in `01_data/cii/experience_manifest.json` (BENIGN 19,028, Syn
11,946, LDAP 4,576, UDP 10,977).

| Deployed model | Accuracy | Macro recall | LDAP | UDP | Benign FAR |
|---|---|---|---|---|---|
| clean | 0.9968 | 0.9953 | 0.9886 | 0.9992 | 0.0018 |
| poisoned, gate only (A0 kept) | **0.6645** | 0.4988 | **0** | **0** | **0.0000** |
| FU2 relabel | 0.9965 | 0.9948 | 0.9878 | 0.9979 | 0.0015 |

- **Accuracy depends on class shares.** BENIGN is 40.9% of these test rows (19,028 of 46,527). A model that never
  learned LDAP or UDP still scores 0.66 accuracy, with a "perfect" benign FAR of 0: it calls every LDAP and UDP flow
  benign. Only per-class recall exposes this.
- **Detecting "an attack" is not naming it.** In the 12-label runs, the s3 clean CII model has accuracy 0.604 and
  macro recall 0.492, yet it flags 98.4% of attack rows as *some* attack (`05_fu1/test/test_summary.csv`,
  `attack_not_benign`). Binary detection looks good while 6 of 12 classes are never named.
- **Recall can hide a weakening.** In RF, Syn recall stays at 0.994 while confidence falls to 0.77. A threshold or
  calibration that uses probabilities would see the damage; argmax accuracy and recall do not.
- **What we report instead:** per-class recall, macro recall, benign FAR with the number of benign rows, new-class
  FAR, and mean p for the attacked class.

---

## 8. Problems faced and fixes (timeline)

Each entry reads: observation -> evidence -> diagnosis (or uncertainty) -> action -> outcome.

1. **Starter claims untested on real data** (10-20).
   - The website starter had passed only synthetic checks. **Diagnosis:** unknown schema and counts.
   - **Action:** audit every file before modelling (`s0_audit.py`).
   - **Outcome:** `notebooks/manifest.json`, 18/18 files untruncated.
2. **BENIGN scarcity.**
   - **Evidence:** 113,828 BENIGN of 70,427,637 rows (0.16%); 95,272 distinct after cleaning.
   - **Diagnosis:** CII needs fresh BENIGN in each experience; capping BENIGN would starve the false-alarm
     measurement.
   - **Action:** never cap BENIGN; split CII BENIGN into disjoint portions; report benign counts next to every FAR;
     protect at least 5,000 benign validation and test rows.
   - **Outcome:** 19,055 / 19,054 / 19,054 BENIGN per four-label experience; 9,507 validation and 19,028 test BENIGN
     rows scored.
3. **Massive duplication.**
   - **Evidence:** 59,924,164 within-file exact duplicates.
   - **Diagnosis:** capping before deduplication would sample copies, and identical inputs could fall into both train
     and test.
   - **Action:** global dedupe on a float64 hash of the model features, before a bottom-k cap.
   - **Outcome:** cross-split identical inputs = 0 (asserted).
4. **Label conflicts.**
   - **Evidence:** 235,261 feature vectors appear under 2 or more labels (UDP 191,822; SSDP 190,604).
   - **Diagnosis:** the correct label cannot be known [our inference: reflection-attack flows can be indistinguishable
     at this feature level].
   - **Action:** drop all of them.
   - **Outcome:** LDAP is left with 13,890 training rows, which fixed the four-label attack cap at 13,890.
5. **Filenames contradict the label column (03-11).**
   - **Evidence:** `UDPLag.csv` holds 606,749 Syn rows and only 1,873 UDPLag; other files are mixed too (T2).
   - **Diagnosis, uncertain:** we do not know whether the file naming or the labelling is wrong.
   - **Action:** use the label column everywhere, and record the source file per row.
   - **Outcome:** documented assumption; per-file label mix kept in the audit.
6. **Non-finite values.**
   - **Evidence:** `flow bytes/s` and `flow packets/s` are inf/NaN in 2,170,750 rows.
   - **Action:** drop those rows (the reference release zero-fills instead). Not ablated.
7. **Leakage-prone columns.**
   - **Evidence:** ids, IPs, ports, timestamp, `SimillarHTTP`; `Inbound` is 1 for almost all attacks but only 21,331
     of 113,828 BENIGN.
   - **Action:** drop them all by default (`common.py`); flipping them back is reserved for labelled ablations.
   - **Outcome:** 77 numeric + protocol. No ablation was run.
8. **Transformed duplicates after scaling.**
   - **Evidence:** the first Stage 1b run refused in CI (reported in the session, not in a committed file): 0.19%
     calibration, 0.17% validation and 0.18% test overlap, against a 0.10% guard.
   - **Diagnosis** (from Stage 1b's per-step diagnosis): clipping to ±5, float32 rounding and -0.0 make distinct raw
     rows identical after the transform.
   - **Action:** do not raise the guard; exclude exactly those protected rows (the same ids for CI and CII), and
     record counts per split and class.
   - **Outcome:** 166 rows excluded, residual 0.
9. **Kaggle input mounting.**
   - **Observation:** dataset versions mount at different paths.
   - **Action:** notebooks find folders by marker files (`s5_explain.find_dir`) and hash-check every input
     (`config.json` `inputs`); `--resume` refuses if an input changed.
   - **Outcome:** every later run records the same Stage 1 hashes (`bb1f5acb…` and others).
10. **Test scored before the decisions commit (sweep D1).**
    - **Evidence:** s3 wrote `final_test.json` inside each run, about 1.5 h before the validation commit.
    - **Diagnosis:** the code computed test eagerly; no test number was displayed or used.
    - **Action:** disclosed as D1; from FU1 on, test is scored by a separate once-only command. That command refuses
      unless the validation files are byte-identical to a named git commit (`score-test --decisions-commit`); s3 got
      `--defer-test`.
    - **Outcome:** every later experiment has a val-commit-then-test record.
11. **Sessions closing between validation and test.**
    - **Action:** `bundle` / `restore` zips with sha256 manifests; a restored model must match the validated hash.
    - **Outcome:** used for FU1; available in every later notebook.
12. **Invalid separability numbers (FU1).**
    - **Evidence:** float32 `torch.cdist` reported impossible "identical input" shares.
    - **Diagnosis:** large squared norms (protocol x100) round small distances to 0.
    - **Action:** file marked INVALID; exact float64 recomputed (descriptive, after the test, decided nothing).
13. **Clean gate rejections and class overlap (12-label).**
    - **Evidence:** clean DNS < 0.80 (DNS predicted LDAP); old SSDP collapsed when UDP arrived; MSSQL 0.0006 below its
      floor.
    - **Diagnosis:** forgetting and class overlap, not poisoning.
    - **Action:** FU1 retries; joint oracle diagnosis.
    - **Outcome:** neither fixed it; the 12-label goal remains unsolved.
14. **Forgetting under the gate (FU1, RC, RCS).**
    - **Evidence:** a newly learned class's own floor blocks later overlapping classes (FU1); LDAP/SNMP/MSSQL and
      NTP/SSDP forgetting in the 9 clean RC/RCS runs (13 rejected updates).
    - **Action:** registered stops honoured; no schedule search continued.
15. **Soft repair at 100% flips.**
    - **Evidence:** R1 Syn recall 0.000-0.0002 in 6/6 attempts at f = 1.0.
    - **Diagnosis:** weighting removes wrong labels but leaves no Syn label to learn from (with no fresh Syn in D).
    - **Action:** post-hoc relabel R1 (FU2), registered before its run, with the provenance disclosed.
    - **Outcome:** 3/3 seeds learn all 4 classes [test].
16. **Confidence loss without recall loss** (`main`, RF).
    - **Evidence:** `main` about 0.79 vs 0.994 [diagnostic]; RF test mean p 0.7707 vs 0.9922 with recall unchanged.
    - **Diagnosis:** the gate checks recall only; class weighting keeps the argmax.
    - **Action:** reported as a limitation. A confidence gate exists in code (`--gate-conf-drop`) but was never
      registered or evaluated.
    - **Outcome:** open problem.
17. **Stopped pilots (RC, RCS).**
    - **Evidence:** pilot or screen `PROCEED = False` / `SELECTED = None`.
    - **Action:** files committed unchanged and the experiments closed. The next design (RF) was registered as a new,
      explicitly post-hoc experiment.

---

## 9. Related work (links checked on 2026-10-28)

Citations give authors, title, year and venue as shown on the primary page. We reproduce no other paper's numbers as
ours; a number quoted from a paper is marked [reported by paper].

### 9.1 The reference continual-IDS work

- **A. Ariffin, A. Haris, F. Zaki, H. Hanif, N. B. Anuar.** "Adaptive Intrusion Detection System using
  Transformer-Based Neural Networks and Continual Learning Approach with Adversarial Investigation." arXiv:2608.04602
  (cs.CR), 5 Aug 2026. https://arxiv.org/abs/2608.04602
  - **Code:** https://github.com/um-csnet/ReplayIDS
  - **What it does** [reported by paper]:
    - a TabTransformer-based IDS on **CICIDS2017** (8 classes), with CI and CII schedules of four experiences;
    - replay variants (ER-Stratified, ER-Balanced with benign anchoring), iCaRL, LwF and EWC baselines, and a joint
      oracle;
    - two buffer attacks: label flipping of 100% of stored exemplars at buffer budgets of 1-10% (the flipped label is
      drawn uniformly from the other classes), and an inter-arrival-time backdoor.
    - It reports accuracy collapsing to about 0.005 under label flipping at p = 1% (its Table 10), and implements **no
      defence**.
  - **How our experiment differs:**
    - dataset CICDDoS2019 (not CICIDS2017), audited and deduplicated globally;
    - our model is FT-Transformer-style (section 4.2);
    - our buffer stores 2,000 rows per class per experience (theirs is a percentage budget);
    - our attack is **targeted and bounded**: one old class, Syn -> BENIGN, a fraction f of its stored rows;
    - **we add a validation gate and a frozen-teacher repair** and test whether they defend.

    We did not run their code or reproduce their numbers. We did not survey the literature systematically for gated
    update acceptance, so we claim it only as our implementation choice, not as a new idea.

### 9.2 Tabular Transformers

- **Y. Gorishniy, I. Rubachev, V. Khrulkov, A. Babenko.** "Revisiting Deep Learning Models for Tabular Data."
  NeurIPS 2021. arXiv:2106.11959. https://arxiv.org/abs/2106.11959
  - The FT-Transformer: Feature Tokenizer `T_j = b_j + x_j W_j`, a [CLS] token and PreNorm blocks. Closest to our
    model.
- **X. Huang, A. Khetan, M. Cvitkovic, Z. Karnin.** "TabTransformer: Tabular Data Modeling Using Contextual
  Embeddings." arXiv:2012.06678, 2020. https://arxiv.org/abs/2012.06678
  - The Transformer runs only over categorical column embeddings; continuous features join after it. Our model is
    **not** this design.
- **A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin.** "Attention
  Is All You Need." NeurIPS 2017. arXiv:1706.03762. https://arxiv.org/abs/1706.03762
  - Scaled dot-product multi-head attention, used here through PyTorch.

### 9.3 Continual learning with replay, distillation and regularisation

- **D. Rolnick, A. Ahuja, J. Schwarz, T. P. Lillicrap, G. Wayne.** "Experience Replay for Continual Learning."
  NeurIPS 2019. arXiv:1811.11682. https://arxiv.org/abs/1811.11682
- **A. Chaudhry, M. Rohrbach, M. Elhoseiny, T. Ajanthan, P. K. Dokania, P. H. S. Torr, M. Ranzato.** "On Tiny
  Episodic Memories in Continual Learning." arXiv:1902.10486, 2019. https://arxiv.org/abs/1902.10486
  - Small replay buffers are effective. Our 2,000-row-per-class buffer is in this family.
- **S.-A. Rebuffi, A. Kolesnikov, G. Sperl, C. H. Lampert.** "iCaRL: Incremental Classifier and Representation
  Learning." CVPR 2017. arXiv:1611.07725. https://arxiv.org/abs/1611.07725
- **Z. Li, D. Hoiem.** "Learning without Forgetting." ECCV 2016. arXiv:1606.09282. https://arxiv.org/abs/1606.09282
  - Distillation from the previous model. FU1's `distill` recipe is LwF-style.
- **J. Kirkpatrick et al.** "Overcoming catastrophic forgetting in neural networks." PNAS 2017
  (doi:10.1073/pnas.1611835114). arXiv:1612.00796. https://arxiv.org/abs/1612.00796
- **A. Chaudhry, P. K. Dokania, T. Ajanthan, P. H. S. Torr.** "Riemannian Walk for Incremental Learning:
  Understanding Forgetting and Intransigence." ECCV 2018. arXiv:1801.10112. https://arxiv.org/abs/1801.10112
  - The forgetting and intransigence measures used by the reference paper. We report per-class recall instead.
- **G. Hinton, O. Vinyals, J. Dean.** "Distilling the Knowledge in a Neural Network." arXiv:1503.02531, 2015.
  https://arxiv.org/abs/1503.02531
  - The "teacher" vocabulary. Our teacher is **not** used for distillation: it only scores or relabels replay rows.

### 9.4 Poisoning and defences

- **B. Biggio, B. Nelson, P. Laskov.** "Poisoning Attacks against Support Vector Machines." ICML 2012.
  arXiv:1206.6389. https://arxiv.org/abs/1206.6389
  - Training-data poisoning as an attack on a learner.
- **M. Umer, G. Dawson, R. Polikar.** "Targeted Forgetting and False Memory Formation in Continual Learners through
  Adversarial Backdoor Attacks." arXiv:2002.07111, 2020. https://arxiv.org/abs/2002.07111
  - Poisoning aimed at continual learners (EWC), causing targeted forgetting [reported by paper]. Our attack edits
    stored replay labels rather than inserting backdoor samples.
- **J. Steinhardt, P. W. Koh, P. Liang.** "Certified Defenses for Data Poisoning Attacks." NIPS 2017.
  arXiv:1706.03691. https://arxiv.org/abs/1706.03691
  - Defences that bound poisoning damage. Our teacher repair is an empirical, not certified, defence.

### 9.5 Dataset

- **I. Sharafaldin, A. H. Lashkari, S. Hakak, A. A. Ghorbani.** "Developing Realistic Distributed Denial of Service
  (DDoS) Attack Dataset and Taxonomy." IEEE 53rd International Carnahan Conference on Security Technology (ICCST),
  Chennai, 2019. doi:10.1109/CCST.2019.8888419.
  - Dataset page: https://www.unb.ca/cic/datasets/ddos-2019.html
  - We used a Kaggle CSV copy (`rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files`). Counts in this handbook
    are from our audit of that copy, not from the paper.

---

## 10. Viva questions and answers

**Basics**

1. **What is a flow?** One bidirectional connection summarised by CICFlowMeter into numeric statistics: durations,
   packet counts and sizes, inter-arrival times, flag counts. One CSV row is one flow with one label. We use 77
   numeric features plus `protocol`.
2. **What is an experience?** A block of training rows that arrives at one time step. E0 = BENIGN + Syn; in the
   four-label setting E1 adds LDAP and E2 adds UDP (with fresh BENIGN in CII).
3. **CI vs CII?** In CI, BENIGN appears only in E0. In CII, every experience carries a fresh, disjoint BENIGN portion,
   which is closer to real traffic. All defended results are CII.
4. **What is a replay buffer and why use it?** A store of past training rows (2,000 per class per accepted
   experience) mixed into every update, so old classes are rehearsed and not forgotten.
5. **What exactly does the attacker do?** Before E1, it flips the stored label of a fraction f of the stored Syn rows
   to BENIGN (RC and RF: more flips before every update). It cannot touch the new data D, validation, test, the
   teacher, weights or gradients.
6. **What is catastrophic forgetting?** Updating on new classes overwrites what was learned for old ones. Seen here
   as old-class recall dropping when an overlapping new class arrives (e.g. LDAP rows predicted SNMP).

**Model**

7. **What is the model?** An FT-Transformer-style tabular Transformer: each numeric feature becomes a 32-dimensional
   token `x_j W_j + b_j`, protocol becomes an embedding token, and a [CLS] token is prepended. The 79 tokens pass
   through 3 PreNorm encoder layers with 4 heads; the final [CLS] vector goes to LayerNorm and a linear classifier.
   It has 43,396 parameters with 4 classes.
8. **Explain Q, K and V here.** For each token h (32 dims) and each of the 4 heads:
   - the model computes a query q = h W_Q, a key k = h W_K and a value v = h W_V, each 8-dimensional;
   - the weight of token j in token i's update is softmax over j of `q_i . k_j / sqrt(8)`;
   - the head output for token i is the weighted sum of the v_j.

   So the [CLS] token can learn which features matter for this flow; for example it can attend to the flag-count
   tokens. Over 79 tokens each head forms a 79 x 79 attention matrix.
9. **Why [CLS] instead of averaging?** It gives the network a dedicated learned summary position, as FT-Transformer
   does. We did not test pooling alternatives.
10. **Is it the original TabTransformer?** No. TabTransformer runs attention only over categorical columns and
    concatenates continuous features afterwards. Ours tokenises every numeric feature. The class name
    `TabTransformer` in the code is historical.
11. **How are new classes added?** `expand_head` builds a larger output layer and copies the old rows, so old logits
    are unchanged until training continues.

**Protocol**

12. **What are T and A0, and why two copies?** Both come from the same E0 training. **T** is frozen forever and only
    read (hash-checked); **A0** is the model that gets updated. Keeping T separate means later (possibly poisoned)
    updates can never corrupt the reference that judges the buffer.
13. **When is T updated?** Never. Its parameters have `requires_grad=False`, and every run asserts that its hash is
    unchanged.
14. **What are C1 and R1?** C1 is the plain update candidate. R1 is the repair candidate, tried only if C1 is rejected
    in a repair arm. R1 restarts from the last accepted model and uses teacher-derived weights or labels on buffer
    rows. Both must pass the same gate.
15. **What is deployed after a rejection?** The previous accepted model. The buffer does not grow, and in s3 that
    experience's classes are never retried.
16. **What enters the buffer?** Only rows of accepted experiences, with their true labels. R1's relabelling affects
    only its own training batch and is never written back.
17. **What are FU1's C2 and D1?** Retry *attempts* for a rejected block: C2 immediately, with a validation-selected
    recipe; D1 later, after the next accepted block. They are not extra models. Only one model is ever deployed.
18. **Experience number vs candidate number?** E_t numbers the data block. C1/R1/C2/D1 number the attempts within it.
    "E2 R1" is the repair attempt of the second update.

**Gate, repair and results**

19. **What does the gate check?**
    - old attack recall >= last accepted recall - 0.05;
    - new-class recall >= 0.80;
    - benign FAR <= 0.05;
    - new-class FAR <= 0.05.

    All on protected validation rows, never test.
20. **Why can the gate reject a clean update?** It cannot tell poison from ordinary forgetting. If learning a new
    class costs an overlapping old class more than 0.05 of recall, or the new class stays below 0.80, the update is
    rejected even without poison. Examples: DNS vs LDAP and SSDP vs UDP (12-label), LDAP vs SNMP (RC/RCS).
21. **Why does soft repair work at f <= 0.95 and fail at f = 1.0?** T gives flipped rows a weight of about 0.003 and
    correct rows about 0.998, so soft R1 deletes the wrong labels. At f <= 0.95 some correct Syn labels survive and
    class weighting amplifies them. At f = 1.0 none survive and D has no Syn, so there is nothing to learn Syn from.
22. **What does relabelling add?** If T is confident that a stored BENIGN row is really Syn (p_T(BENIGN) <= 0.1 and
    max p >= 0.9), that row is trained as Syn. This restores correct labels, not just removes wrong ones. It works
    here because T is very confident on these rows, and it is a post-hoc method.
23. **Why is FU2's E1 not independent evidence?** The relabel method and thresholds were chosen after the sweep's
    forced diagnostic, and FU2's E1 is exactly that diagnostic, replayed with identical weights. The new evidence is
    the E2 decision and the held-out test.
24. **Confidence vs recall?** Recall counts argmax hits. Confidence (mean p of the true class) measures how sure the
    model is. Poison can lower confidence (RF: 0.99 -> 0.77) while the argmax, and so recall (0.994), does not move.
    A recall-only gate is blind to that.
25. **Did RF repair the confidence loss?** No. No update was rejected, so repair never ran, and the repair arm
    deployed exactly the poisoned arm's models. The forced R1 diagnostic suggests repair *could* have helped
    [val, diagnostic], but it was never deployed or tested.
26. **Why is accuracy alone misleading?** Class shares dominate it. A model that never learned LDAP or UDP still gets
    0.66 accuracy and a benign FAR of 0 on the four-label test (section 7.1).

**Data and leakage**

27. **How many rows did you train on? Did you use all 70 million?** No. We read all 70,427,637 rows once to audit,
    deduplicate and sample. Training used capped, deduplicated subsets: 98,833 rows (four-label), 159,595 (12-label
    oracle), and 102,445 distinct rows in RF (E0 22,618; each refresh update 13,304-13,305 new rows plus the replay buffer).
28. **How many classes?** 18 files, 14 raw labels, a 12-label subset (Portmap and WebDDoS excluded for too few rows),
    the four-label defence setting and the three-label RF study (section 3.1).
29. **How do you prevent leakage?**
    - Labels come from the label column.
    - ids, IPs, ports, timestamps and `Inbound` are dropped.
    - Deduplication is global, before capping, and conflicting vectors are dropped.
    - Splits are made before anything is fitted; the scaler and vocabulary are fitted on E0 training rows only.
    - Evaluation rows whose scaled input equals a training input are excluded.
    - Validation and test never enter training or the buffer; every input is hash-checked.
30. **What is still leaky?** Random flow-level splits can put near-identical neighbouring flows on both sides, and the
    header-length artefacts were not ablated. This is not a time-ordered deployment test.
31. **Why is BENIGN scarce, and what did you do?** CICDDoS2019 is attack-heavy (0.16% BENIGN). We kept every distinct
    BENIGN row (95,272), split it disjointly across CII experiences, and report benign counts with every FAR.

**Method and integrity**

32. **What is pre-registration and why use it?** Writing the design, thresholds and predictions into git *before*
    running, so results cannot quietly shape the hypotheses. The commit is the timestamp; changes go under
    "Deviations".
33. **How do you know test did not influence decisions?**
    - Decisions read only validation.
    - Since FU1, test is scored by a separate once-only command. It refuses unless the validation files are
      byte-identical to a named commit.
    - In the sweep, test was computed early inside the runs; this was disclosed as D1 and never displayed or used.
34. **Why don't you claim a 12- or 14-label defence?** Clean 12-label updates fail the gate, retries did not fix it
    (FU1), and the joint oracle shows the model/recipe struggles with two class pairs even without continual
    learning. Portmap and WebDDoS have too few rows. So no defence beyond four labels is supported.
35. **Is the result statistically significant?** We claim no significance. There are 3 seeds per setting (1 for the
    12-label runs), and we report counts and ranges.
36. **What would you change next?** See section 11. Those are proposals, not results.

---

## 11. What we would change next (proposal only; nothing here has been run)

1. **A confidence-aware gate.** Register `--gate-conf-drop` (or a calibration check) in advance and re-run the RF
   design, to test whether the confidence loss would then trigger repair. The registration would have to fix the
   threshold on validation before any run.
2. **A teacher that grows.** T knows only BENIGN and Syn, so it cannot protect LDAP or later classes. One option is a
   teacher per accepted experience, frozen once accepted. This needs its own threat analysis, because a poisoned
   accepted update would freeze a poisoned teacher.
3. **Scaling beyond four labels.** For example: gate thresholds per class pair, buffer budgets that favour
   overlapping classes, or a joint-training checkpoint as E0. Each would be a new pre-registered study, judged first
   on a clean pilot.
4. **Time-ordered splits and ablations.** `--split time`, `Inbound` and ports as labelled ablations, and the
   header-length features removed, to bound the leakage risks in section 10 Q30.
5. **More seeds and attacked classes.** At least 5 seeds and a second attacked class, so a significance statement
   becomes possible.

---

## 12. Reproduce and verify (exact commands and recorded output)

Environment for the recorded runs below: a local Linux CPU, Python 3.14.6, torch 2.14.1+cpu, pandas 2.3.3. pandas 3.x
breaks parts of the older smoke test, so use pandas < 3. **No training, no test scoring and no data download happened
for this section**: everything reads committed files.

### 12.1 Regenerate and verify the result tables

```
$ python make_report_tables.py --check
OK: docs/RESULT_TABLES.md matches a fresh build from committed files; consistency checks passed
(exit code 0)
```

The check fails when a table does not match its sources. Recorded once with a deliberately hand-edited number, then
restored:

```
$ python make_report_tables.py --check   # after hand-editing one number in docs/RESULT_TABLES.md
--- committed
+++ regenerated
@@ -226,3 +226,3 @@
 | 0.1 | clean | 0% | 0.9952 (0.9949-0.9954) | 0.9922 (0.9904-0.9933) | 0.9974 | 0.0007 |
-| 0.1 | poisoned | 60% | 0.9999 (0.9936-0.9950) | 0.7707 (0.7565-0.7859) | 0.9974 | 0.0007 |
+| 0.1 | poisoned | 60% | 0.9943 (0.9936-0.9950) | 0.7707 (0.7565-0.7859) | 0.9974 | 0.0007 |
 | 0.1 | poisoned_repair | 60% | 0.9943 (0.9936-0.9950) | 0.7707 (0.7565-0.7859) | 0.9974 | 0.0007 |
FAIL: docs/RESULT_TABLES.md differs from a fresh build
(exit code 1)
```

Before writing, the script also recomputes aggregates from the per-seed files and asserts equality with the
committed summaries:
- sweep, FU2 and RF means / min / max;
- the oracle's CII medians;
- a single test-row hash per test set (`09245b03b63d` for four labels; `1eeeed6f1923` for 12 labels).

### 12.2 The small CPU demo of the primary result (committed FU2 evidence only)

```
$ python demo.py fu2 --seed 42
FU2, seed 42: four labels (E0 BENIGN+Syn, E1 +LDAP, E2 +UDP), CII. Before E1 the attacker flipped 2000 of the 2,000 stored Syn labels to BENIGN (f = 1.0); flips persist. R1 = threeway+relabel (post hoc method). Teacher T unchanged: True.

1) Every gate inequality on validation, recomputed from rounds.json (matches the saved decisions):
experience attempt                         check  value  bound  margin  pass
        E1      C1  old Syn: r_C >= r_ref - 0.05 0.0000 0.9453 -0.9453 False
        E1      C1          new LDAP: r_C >= 0.8 0.9869 0.8000  0.1869  True
        E1      C1            benign FAR <= 0.05 0.0007 0.0500  0.0493  True
        E1      C1         new-class FAR <= 0.05 0.0006 0.0500  0.0494  True
        E1      R1  old Syn: r_C >= r_ref - 0.05 0.9953 0.9453  0.0500  True
        E1      R1          new LDAP: r_C >= 0.8 0.9874 0.8000  0.1874  True
        E1      R1            benign FAR <= 0.05 0.0003 0.0500  0.0497  True
        E1      R1         new-class FAR <= 0.05 0.0002 0.0500  0.0498  True
        E2      C1  old Syn: r_C >= r_ref - 0.05 0.0000 0.9453 -0.9453 False
        E2      C1 old LDAP: r_C >= r_ref - 0.05 0.9887 0.9374  0.0513  True
        E2      C1           new UDP: r_C >= 0.8 0.9980 0.8000  0.1980  True
        E2      C1            benign FAR <= 0.05 0.0014 0.0500  0.0486  True
        E2      C1         new-class FAR <= 0.05 0.0007 0.0500  0.0493  True
        E2      R1  old Syn: r_C >= r_ref - 0.05 0.9953 0.9453  0.0500  True
        E2      R1 old LDAP: r_C >= r_ref - 0.05 0.9887 0.9374  0.0513  True
        E2      R1           new UDP: r_C >= 0.8 0.9985 0.8000  0.1985  True
        E2      R1            benign FAR <= 0.05 0.0017 0.0500  0.0483  True
        E2      R1         new-class FAR <= 0.05 0.0005 0.0500  0.0495  True

2) What the relabel R1 did to the replay buffer (T's prediction replaces the stored label):
   E1 R1: relabelled 1995 rows, 1995 to their true class; 1995 of them were attacked rows -> accepted: True
   E2 R1: relabelled 1995 rows, 1995 to their true class; 1995 of them were attacked rows -> accepted: True

3) Held-out test, seed 42 (46,527 rows, the same rows for every arm). poisoned_relabel was scored once after FU2's validation commit; clean / poisoned / poisoned_repair are the sweep's f = 1.0 controls:
                     Syn    LDAP     UDP  macro_recall  benign_far
arm                                                               
clean             0.9954  0.9886  0.9992        0.9953      0.0018
poisoned          0.9951  0.0000  0.0000        0.4988      0.0000
poisoned_repair   0.9951  0.0000  0.0000        0.4988      0.0000
poisoned_relabel  0.9951  0.9878  0.9979        0.9948      0.0015

   mean over seeds 42-44:
                     Syn    LDAP     UDP  benign_far
arm                                                 
clean             0.9954  0.9851  0.9988      0.0011
poisoned          0.9953  0.0000  0.0000      0.0000
poisoned_relabel  0.9952  0.9825  0.9983      0.0011
poisoned_repair   0.9953  0.0000  0.0000      0.0000
```

What to say while it runs (2-3 minutes):
1. The attacker flipped all 2,000 stored Syn labels.
2. Plain C1 drives Syn recall to 0, and the gate rejects it (margin -0.9453).
3. Relabel R1 restores 1,995 true labels, passes every inequality, and is deployed.
4. On held-out test, relabelling learns all four classes, while the gate alone and soft repair keep the stale A0
   (LDAP = UDP = 0).
5. State the scope: post hoc method, four labels, one attacked class, 3 seeds.

### 12.3 Optional demo: the RF limitation

```
$ python demo.py rf
RF (post hoc, fixed-class traffic refresh: E0 BENIGN+Syn+TFTP, six updates of fresh BENIGN+TFTP, no new class). rho = 0.10: 200 more stored Syn labels flipped before every update (60% after E6).

Validation trajectory, mean over seeds 42-44 (t = update):
    Syn_mean_p          Syn_recall         
arm      clean poisoned      clean poisoned
t                                          
0       0.9929   0.9929     0.9952   0.9952
1       0.9919   0.9523     0.9943   0.9951
2       0.9914   0.9062     0.9949   0.9950
3       0.9905   0.8562     0.9944   0.9911
4       0.9929   0.8626     0.9954   0.9951
5       0.9911   0.8125     0.9927   0.9915
6       0.9923   0.7705     0.9955   0.9943
C1 accepted: {'clean': '18/18', 'poisoned': '18/18'}; R1 ran: 0 times

Held-out test after E6 (mean over seeds; poisoned_repair is identical to poisoned):
 rho      arm  flipped_share  Syn_recall_mean  Syn_mean_p_mean  benign_far_mean
0.05    clean            0.0           0.9952           0.9922           0.0007
0.05 poisoned            0.3           0.9943           0.9027           0.0007
0.10    clean            0.0           0.9952           0.9922           0.0007
0.10 poisoned            0.6           0.9943           0.7707           0.0007

Reading: recall held and no update was rejected, so no repair was deployed; the confidence loss stayed.
```

### 12.4 Other checks and commands

| Purpose | Command | Needs |
|---|---|---|
| Model shape and head-expansion self-test (toy sizes: 54 features, 2 layers) | `python s2_model.py` | nothing; prints `old-class logits preserved after head expansion: OK` |
| Plumbing smoke test on synthetic data: every stage, all experiment scripts, refusal paths, and 30 checkpoint hashes against the older code | `python smoke_test.py --work /tmp/smoke [--golden golden.json]` | CPU, about 6-7 min; synthetic numbers are **not** results |
| `main` run demos (a test flow, the poisoned buffer, the gate) | `python demo.py flow` / `poison` / `gate --arm poisoned` | the `main` results zip and the Stage 1 parquet (not in git) |
| Rebuild an experiment's committed test tables from its zip | `make_result_tables.py`, `make_fu1_tables.py`, `make_fu2_tables.py`, `make_oracle_tables.py`, `make_rf_tables.py` | the Kaggle output zips (not in git) |

### 12.5 Storyboard figures and the guided demo (added after the handbook's first version)

| Purpose | Command |
|---|---|
| Write the 7 storyboard figures and their plotted-data CSVs to `docs/figures/` | `python make_figures.py` |
| Rebuild in a temporary folder; fail (exit 1) if any plotted number differs from `docs/figures/*_data.csv` or from the committed result tables | `python make_figures.py --check` |
| Execute `notebooks/11_results_storyboard.ipynb` end to end, with a temporary kernel spec for the current interpreter | `python make_figures.py --execute-notebook` |
| Guided 3-minute walkthrough of the FU2 defence; switch to the RF limitation; pause between steps | `python demo.py walkthrough` / `--story rf` / `--pause` |

`docs/figures/README.md` lists each figure's claim, limitation and sources.

### 12.6 Where the slides come from

`docs/slides_outline.md` lists each slide's source. Every number on a slide must come from `docs/RESULT_TABLES.md` or a
committed README, and the live demo is `python demo.py fu2` (with `python demo.py rf` optional).

---

## Appendix A. Code map

| File | Stage | Key functions (line) |
|---|---|---|
| `common.py` | shared | column rules (`ID_COLS`, `DROP_PORTS`, `DROP_INBOUND`), `canon_label` |
| `s0_audit.py` | 0 | `audit_file`: per-file labels, duplicates, non-finite values, leakage hints |
| `s1_preprocess.py` | 1 | `stream_dedupe_cap` (45), `feature_hash` (41), `split_indices` (126) |
| `s1b_experiences.py` | 1b | `build_manifest` (58), `fit_scaler` (104), `protected_exclusions` (275) |
| `s2_model.py` | 2 | `FeatureTokenizer` (31), `TabTransformer` (49), `expand_head` (71) |
| `s3_continual.py` | 3 | `train` (150), `evaluate` (201), `gate` (230), `buffer_add` (263), `poison` (277), `poison_more` (289), `teacher_weights` (300), `repair_rows` (319), `initial_round` (356), `run_arm` (382) |
| `s5_explain.py` | 03 | read-only diagnostics, `gate_check` (139), `find_dir` |
| `s6_sweep.py` | 04 | sweep decision and test tables |
| `s9_followup.py` | 05 | FU1 probes, selection rule, retry `policy` (188), `score-test` |
| `s10_relabel.py` | 06 | FU2 relabel arm, `bundle` / `restore` / `score-test` |
| `s11_oracle.py` | 07 | joint oracle training and verdict |
| `s12_repeated.py`, `s13_rcs.py`, `s14_refresh.py` | 08, 09, 10 | RC, RCS, RF stages, and once-only commit-checked tests |
| `make_report_tables.py` | report | `docs/RESULT_TABLES.md` from committed files; `--check` |
| `demo.py` | demo | `fu2`, `rf` (committed files); `flow`, `poison`, `gate` (`main` outputs) |
