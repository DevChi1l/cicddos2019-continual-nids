# FU1 validation outcome (notebook 05, Kaggle, code `8986446`)

`followup/` is the content of `fu1_val.zip`, committed **unchanged**: 59 JSON/CSV files, no checkpoint, no test file.
Pre-registration: [`../../prereg_followup_fu1.md`](../../prereg_followup_fu1.md) (`66c88d4`, handoff `8986446`).
Nothing here is a test result. Test is scored once, later, by `s9_followup.py score-test` with this commit's sha.
That command checks that `followup/decisions.json` and each run's `summary.json` / `rounds.json` are byte-identical
to these files.

| File | sha256 |
|---|---|
| `fu1_val.zip` (source of `followup/`) | `e25c0061a604ed87995eda0ab13d3f9d38580e40d4bee2b7007f13a92c823e9d` |
| `fu1_restore.zip` (not committed: holds `final.pt`; manifest created 2026-10-21 12:56 UTC) | `5d5d7cbb4000f47c8bb10ecb04520ac51124e67b2b5c04c12c305be3911b1a01` |

Checks before this commit:
- the zip lists exactly the JSON/CSV files of the session's `nids_out/followup/`, byte for byte;
- the only test-related key is the input hash of `clean_test.parquet`, and no `final_test.json` exists anywhere;
- `fu1_restore.zip` passes `s9_followup.py restore`, and its `decisions.json`, `summary.json` and `rounds.json` are
  byte-identical to these files;
- its `final.pt` equals the session's file and the sha256 recorded in `summary.json`.

## Validation outcome (decisions only)

- **P0 holds.** `base` with seed 42 reproduced the logged s3 C1 exactly in all 4 (scenario, block) cells
  (`reproduces_logged_C1` = 0.0).
- **E1 probe (new-class failure):** `ep20` passed the gate in 5 of 6 cells (median margin +0.023), so it was
  selected. `ep10` passed 1 of 6; base, lr3e-4 and balanced passed 0 of 6. DNS val recall with ep20 was
  0.80-0.86, LDAP 0.82-0.88. Prediction P1 ("no E1 recipe eligible") is **not supported**.
- **E5 probe (old-class failure):** no recipe passed (0 of 6 each), so no recipe was selected; P2 holds.
  `balanced` moves SSDP val recall up (0.11-0.98) only by giving up UDP (0.006-0.93), and it never meets both.
  `distill` changed almost nothing (SSDP <= 0.04).
- **Retry policy (both scenarios, identical block outcome):**

  | Block | Outcome |
  |---|---|
  | E1 (DNS, LDAP) | accepted at C2 (ep20) |
  | E2 (MSSQL, NTP) | rejected at C1 and D1 |
  | E3 (NetBIOS, SNMP) | rejected at C1 and D1 |
  | E4 (SSDP, TFTP) | accepted at C1 |
  | E5 (UDP, UDPLag) | rejected at C1 and D1 |

  E2 and E3 were accepted (CI) or partly accepted (CII E2) by s3. Under FU1 they now fail because DNS, learned just
  above 0.80, drops below its own floor (old DNS 0.49-0.68 vs floor about 0.77). At E3, LDAP and new-class FAR also
  fail.
- **The deployed models know 6 of 12 labels:** BENIGN, Syn, DNS, LDAP, SSDP, TFTP. By validation decisions, the
  s3 clean arm ended with 6 (CII: BENIGN, Syn, MSSQL, NTP, SSDP, TFTP) and 8 (CI) labels. FU1 swaps MSSQL/NTP (and
  CI's NetBIOS/SNMP) for DNS/LDAP; it does not add labels.
- **Policy weakness found:** the end-of-schedule D1 for E5 starts from the same model and buffer, with the same
  recipe and seed, as its C1. So it repeats C1 exactly (identical metrics) and is not a real second chance.

## Known problem: `separability.csv` is invalid (descriptive only; it decided nothing)

Its `share_identical_input` (about 0.6-0.93) and 1-NN recalls are a **measurement artifact**. `torch.cdist` in
float32 uses the matrix-product formula; with the protocol column scaled x100 (squared norms about 1e5), small
distances round to 0. Ties then go to whichever class comes first in the training index.

Local re-check on CPU, CII validation rows, exact float64 distances (no training, no test):
- no validation row has an identical training input (min distance > 0), as Stage 1b guarantees;
- 1-NN recall is SSDP 0.58 / UDP 0.55 (near chance, i.e. strong overlap) and DNS 0.84 / LDAP 0.85.

The file is kept unchanged as the record of what ran. Do not cite its numbers.
