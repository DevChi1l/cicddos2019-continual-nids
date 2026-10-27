# Pre-registration: joint-training oracle on the capped 12-label splits (diagnostic)

Written 2026-10-27, **before any oracle run**. The git commit that adds this file is its timestamp. Code:
`s11_oracle.py`, notebook `07_oracle_joint.ipynb`. It reads the all12 inputs and runs read only, never writes into
them, and changes no gate, threshold or split.

## Question

In the 12-label continual runs (`results/04b_all12/`) and FU1 (`results/05_fu1/`), the deployed models learned only
6 to 8 of 12 labels. Is that mainly **continual forgetting / interference** under the update protocol, or can these
features (with this model and recipe) **not separate** DNS/LDAP and SSDP/UDP well enough even when every class is
trained together?

Already known (all post hoc, listed so the predictions below are honest):
- Clean continual updates were rejected at E1 (DNS -> LDAP) and E5 (SSDP -> UDP). FU1 learned DNS/LDAP at test
  recall about 0.83 with 20 update epochs, then lost MSSQL/NTP/NetBIOS/SNMP.
- Exact 1-NN recall on validation (computed 2026-10-27, after FU1's test): DNS 0.84 / LDAP 0.85, SSDP 0.58 / UDP
  0.55. Stage 1 dropped many label-conflict vectors for UDP, SSDP, DNS and LDAP.

## What is fixed

| Item | Value |
|---|---|
| Rows | training = the union of a scenario's all12 experiences: 159,595 rows (BENIGN 57,163 + 9,312 per attack class, 12 labels); the same 61,608 protected val rows and 123,281 test rows as the continual runs (row hashes asserted) |
| Inputs | must hash the same Stage 1/1b files as `runs/<sc>/all12_<sc>_s42/config.json` (s11 refuses otherwise) |
| Scaler | the scenario's all12 scaler (fit on its E0 train rows), unchanged |
| Model and loss | architecture, lr, batch, dropout and class weights (inverse sqrt frequency) from the all12 run config; seed -> init and batch order, as s3's E0 |
| Settings | `e10` = the continual runs' E0 recipe (10 epochs); `e20` = 20 epochs, the only sensitivity check (FU1 showed more epochs helped DNS) |
| Cells | CII: e10 and e20 x seeds 42, 43, 44 (primary). CI: e10 and e20 x seed 42 (scaler check). 8 trainings |
| Test | not scored by `train`/`decide`. `score-test` scores every model once, after the validation files are committed under `experiments/results/07_oracle/`; it checks them byte for byte and each model's sha256 |

## Verdict rule (validation only; `s11_oracle.py decide`)

- **Primary data:** CII, median over seeds 42-44 of oracle val recall.
- **Setting:** per pair, the setting (e10 or e20) with the higher median recall of the pair's weaker class (an
  optimistic "best of two", chosen on validation).
- **Pair verdict:**
  - both classes >= 0.80 and median benign FAR <= 0.05 -> **separable for the gate**: joint training clears the
    gate's new-class bar, so this pair's 12-label failure is primarily continual;
  - weaker class < 0.70 -> **feature limit**: joint training also fails;
  - otherwise -> **borderline**.
- **Overall:** both pairs separable -> "primarily continual"; both feature limit -> "primarily features";
  otherwise "mixed".
- **Other classes:** for MSSQL, NTP, NetBIOS, SNMP, TFTP and UDPLag, the oracle's best median val recall is reported.
  A class at >= 0.80 that a continual model lost (test recall 0 in `04b_all12` clean or FU1) is listed in the
  results README as "lost by the protocol, not by the features".
- CI cells are reported next to CII. If CI gives a different pair verdict, we say so; it does not change the
  primary verdict.

## Predictions (low confidence)

- **O1.** SSDP/UDP: feature limit (weaker class < 0.70).
- **O2.** DNS/LDAP: borderline or separable (FU1 reached about 0.83 on test).
- **O3.** Overall "mixed". MSSQL, NTP, SNMP and TFTP >= 0.80 in joint training, so their loss in FU1 is caused
  by the protocol.

## Limits

- The oracle is one model family with two recipes, not a Bayes ceiling. "Feature limit" means "for this model and
  recipe on these features"; a stronger model might do better.
- Capped data (9,312 training rows per attack class). One split.
- No significance claims (3 seeds, plus 1 for CI).

## Runtime

On a T4, from the all12 E0 timing (66,475 rows x 10 epochs = 33 s): about 80 s per e10 and 160 s per e20 training.
8 trainings take about 16 min; about 20-25 min with loading and evaluation. A local CPU epoch took 393 s, so this
does not run locally.

## Deviations

(none yet)
