# Joint-training oracle: validation results (notebook 07)

Pre-registration: `experiments/prereg_oracle.md` (commit `d69666d`). Code `s11_oracle.py`, unchanged since then.
**Validation only. Test is not scored.** `s11_oracle.py score-test` scores it once, with `--decisions-commit` set to
the commit that adds these files.

## Files

The 10 files under `oracle/` are committed unchanged from `oracle_val.zip` (sha256
`cdb90d6c024bf7fdad47d9ce9e63f3976f75a9583bfabb27bcac46632e61c246`):
- `oracle/{cii,ci}/{e10,e20}_s{seed}/val.json`: 8 cells;
- `oracle/oracle_val.csv`;
- `oracle/verdict.json`.

Run on Kaggle (T4, `cuda`, torch 2.11.0+cu128) at code commit `12ac151`. Each training took 73-76 s (e10) or
146-147 s (e20). Checkpoints are not committed. `oracle_restore.zip` (sha256 `0886ce3c...`, kept by the user) holds
the 8 models.

## Checks done on 2026-10-27 (local, read only)

- **Validation files.** Each of the 10 files is byte-identical across `oracle_val.zip`, the run's `nids_out/oracle/`,
  and `oracle_restore.zip`. Each also matches its sha256 in `RESTORE_MANIFEST.json`.
- **Models.** For each of the 8 models, the sha256 in `nids_out` equals the one in `oracle_restore.zip`, the
  `model_sha256` in its `val.json`, and its manifest entry. The manifest lists exactly those 18 files.
- **Consistency.** `oracle_val.csv` agrees with the 8 `val.json` files. Recomputing the registered rule by hand gives
  the same `verdict.json`.
- **Rows.** Every cell trained on 159,595 rows and was scored on the same 61,608 validation rows (sha256 `357e24fd...`).
- **No test scored.** No `test.json` exists in the run output.

## Validation recall (61,608 rows: 9,507 benign)

| sc | setting | seed | benign FAR | Syn | **DNS** | **LDAP** | MSSQL | NTP | NetBIOS | SNMP | **SSDP** | TFTP | **UDP** | UDPLag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cii | e10 | 42 | 0.005 | 0.960 | 0.652 | 0.676 | 0.874 | 0.962 | 0.919 | 0.830 | 0.800 | 0.995 | 0.385 | 0.865 |
| cii | e10 | 43 | 0.003 | 0.973 | 0.641 | 0.550 | 0.902 | 0.955 | 0.928 | 0.842 | 0.273 | 0.994 | 0.843 | 0.825 |
| cii | e10 | 44 | 0.003 | 0.952 | 0.544 | 0.584 | 0.918 | 0.983 | 0.941 | 0.849 | 0.737 | 0.995 | 0.478 | 0.838 |
| cii | e20 | 42 | 0.002 | 0.962 | 0.598 | 0.604 | 0.867 | 0.978 | 0.926 | 0.926 | 0.770 | 0.996 | 0.426 | 0.870 |
| cii | e20 | 43 | 0.003 | 0.955 | 0.636 | 0.666 | 0.886 | 0.985 | 0.910 | 0.876 | 0.789 | 0.994 | 0.401 | 0.861 |
| cii | e20 | 44 | 0.001 | 0.982 | 0.599 | 0.639 | 0.929 | 0.987 | 0.937 | 0.905 | 0.751 | 0.995 | 0.453 | 0.844 |
| ci | e10 | 42 | 0.003 | 0.949 | 0.672 | 0.750 | 0.882 | 0.969 | 0.896 | 0.715 | 0.827 | 0.995 | 0.348 | 0.872 |
| ci | e20 | 42 | 0.002 | 0.960 | 0.650 | 0.618 | 0.918 | 0.982 | 0.935 | 0.908 | 0.792 | 0.995 | 0.412 | 0.857 |

BENIGN recall is 0.995-0.999 in every cell.

Where the errors go: 10-15% of DNS validation rows are predicted LDAP, and 7-15% of LDAP rows are predicted DNS. 52-62%
of UDP rows are predicted SSDP, except in CII e10 s43, where 70% of SSDP rows are predicted UDP. The boundary between
the two classes moves with the seed instead of separating them. Per-cell shares are in `oracle_val.csv`.

## Registered verdict (`verdict.json`, rule fixed in `d69666d`)

| pair | setting chosen (val) | median val recall (CII, 3 seeds) | median benign FAR | rule outcome |
|---|---|---|---|---|
| DNS / LDAP | e20 | 0.599 / 0.639 | 0.0024 | "feature limit" (weaker class < 0.70) |
| SSDP / UDP | e10 | 0.737 / 0.478 | 0.0033 | "feature limit" |
| overall | | | | "primarily features" |

The single CI seed gives the same outcome for both pairs: DNS/LDAP best weaker class 0.672 (e10), SSDP/UDP 0.412 (e20).

## How to read the verdict

The labels "feature limit" and "primarily features" are the registered names of the rule's outcomes. Their meaning
was fixed in the pre-registration ("Limits"): **for this model and recipe on these features**. Read precisely:

- **What it shows:**
  - The tested joint learner is one FT-Transformer-style model with the all12 recipe (10 or 20 epochs,
    inverse-sqrt class weights). It sees all 12 labels at once, with no updates, gate or replay.
  - It still does not separate DNS from LDAP or SSDP from UDP well enough to clear the gate's 0.80 new-class bar,
    and stays below 0.70 for the weaker class of each pair.
  - So the continual protocol's rejections at these two pairs are **not explained mainly by forgetting**: the
    same learner struggles with them even without continual learning.
- **What it does not show:** that the raw features are inherently inseparable for these pairs. The evidence against
  that reading, all post hoc:
  - **DNS/LDAP is separated better by other setups on the same features.** Exact 1-NN recall on validation is
    0.84 / 0.85 (`05_fu1/separability_exact_local/`, pair-only problem, computed after FU1's test). FU1's continual
    model, with 20 update epochs on a block containing only DNS and LDAP as new classes, reached test recall
    0.83 / 0.83. Both differ in task, split or training from the oracle, so neither is a like-for-like ceiling.
    They do show that this joint recipe, not the features alone, is what stays near 0.6.
  - **SSDP/UDP is weak for every method tried.** Oracle 0.74 / 0.48; exact 1-NN 0.58 / 0.55, near chance for a
    pair; no FU1 recipe passed. Stage 1 also dropped many label-conflict vectors for these classes. This is
    consistent with strong overlap in these features, but one model family and a 1-NN probe are not a Bayes bound.
  - **Narrow test.** One model family, two recipes, capped data (9,312 training rows per attack class), one split,
    and no tuning. A different architecture, loss or feature set might do better.
- **The other classes are learnable jointly.** The best median validation recall over the two settings (an
  optimistic, descriptive pick) is: MSSQL 0.90, NTP 0.98, NetBIOS 0.93, SNMP 0.91, TFTP 0.99, UDPLag 0.86, Syn 0.96.
  - Per the registered rule, these classes are **lost by the protocol, not by the features**: their oracle recall
    is >= 0.80, but a continual model's committed test recall was 0. The comparison is oracle validation against
    continual test, so it is not on the same rows.

    | class | continual models with test recall 0 |
    |---|---|
    | MSSQL | FU1 CII, FU1 CI |
    | NTP | FU1 CII, FU1 CI |
    | NetBIOS | all12 clean CII, FU1 CII, FU1 CI |
    | SNMP | all12 clean CII, FU1 CII, FU1 CI |
    | UDPLag | all12 clean CII and CI, FU1 CII and CI |

  - TFTP was never lost.
  - Joint Syn recall (0.95-0.98) is below the continual models' (about 0.995). Here Syn competes with 11 other
    classes, UDPLag among them.

**Conclusion (validation, one recipe family).** The 12-label failure is mixed in cause:
- At DNS/LDAP and SSDP/UDP, the tested learner fails even when trained jointly. The continual protocol's rejections
  there come mainly from these pairs being hard for this model and recipe, not from forgetting.
- The later losses of MSSQL, NTP, NetBIOS, SNMP and UDPLag come from the protocol. Once overlapping classes are
  accepted, their recall floors block the following blocks (FU1), and rejected blocks are never retried.

## Predictions (`prereg_oracle.md`)

- **O1** (SSDP/UDP: feature limit): **met**.
- **O2** (DNS/LDAP: borderline or separable): **not met**. The rule outcome is "feature limit" (weaker class 0.60).
- **O3** (overall "mixed"; MSSQL, NTP, SNMP and TFTP >= 0.80 jointly): **overall not met**, because the rule gives
  "primarily features". The class part is **met**: 0.90 / 0.98 / 0.91 / 0.99.

No significance claim (3 seeds for CII, 1 for CI). Validation rows only. The test is scored later, once, and is
descriptive.
