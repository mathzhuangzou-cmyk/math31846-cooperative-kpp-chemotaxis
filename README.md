# Math-31846: numerical code and reproducibility evidence

Manuscript: **Exact local stability threshold and pattern selection at onset in a cooperative Fisher–KPP–chemotaxis system**.
Revision snapshot date: **11 November 2026**. This date is not a GitHub release tag.
Repository: [https://github.com/mathzhuangzou-cmyk/math31846-cooperative-kpp-chemotaxis](https://github.com/mathzhuangzou-cmyk/math31846-cooperative-kpp-chemotaxis).
For a fixed version, cite the full commit SHA and use its commit permalink. No release tag or DOI is assigned. Uploading this repository does not constitute submission to the journal.
Chinese instructions: [README_CN.md](README_CN.md).

## Scope and status

The bundle contains all 12 supplied original Python scripts, their 12 audited counterparts, the independent reproduction and cross-check scripts, the nonlinear long-time checks, the revised Figure 2 plotting script, and saved numerical evidence. Original and corrected files intentionally coexist. These are scientific computation codes, not the document-production or storage-service automation.

The original Table 3 values and their quantitative spatial-convergence interpretation have been withdrawn from the revised manuscript. Their historical generating computation remains unidentified. This bundle does not reconstruct or justify those old values. The current replacement Table 3 uses `evidence/table3_offset_recomputed.csv`.

Table 2 and Figure 2 are finite-time consistency checks. Their initialization uses the theoretical leading amplitude. Smaller terminal residuals do not by themselves identify the asymptotic coefficient. The longer-time coefficient deviation is also retained and reported.

## Contents

| Directory | Role |
|---|---|
| `evidence/` | Current independent reproduction, full-FFT cross-check, nonlinear diagnostics, revised Figure 2 script and manuscript-facing CSV/JSON |
| `evidence/final_results/` | Saved finite-time states, full-precision tables and audit output |
| `evidence/extended_results/` | Saved four-job long-time results, states and summary |
| `patched_project/` | All 12 audited project scripts, preserving numerical filenames and imports; includes their saved outputs |
| `original_code/` | All 12 uploaded scripts, byte-for-byte snapshots; known historical problems are retained for audit |
| `archived_results/` | The user's original saved numerical outputs |
| `audit_workflow/` | Original/corrected solver checks and the original independent long-time workflow |
| `audit_evidence/` | Saved audit logs and checks; these describe the earlier numerical audit |
| `patched_quick_results/` | Short quick-preset results, not manuscript Table 3 |
| `figures/` | The two research images used in the current revised manuscript |
| `docs/` | Publication wording, code provenance and current revision trace snapshot |

`audit_workflow/independent_solver.py` and `audit_workflow/extended_checks.py` are retained at their earlier paths. Their current copies also appear in `evidence/` so both workflows remain self-contained.

## Environment

Python 3.12.14 was used for the audit. Install the independent numerical dependencies with:

```bash
python -m pip install -r requirements.txt
```

The 12-script original/corrected project additionally needs PyTorch. Its tested CPU environment is recorded in `requirements_torch_tested.txt`; the audit used PyTorch 2.15.0+cpu. The user's historical GPU outputs came from RTX 4060 / PyTorch 2.4.1+cu121. CUDA was not rerun during this audit. Install a PyTorch build appropriate for the intended platform; the independent `evidence/` workflow does not need PyTorch.

## Verify the delivered snapshot

From the repository root (after cloning or extracting a downloaded snapshot):

```bash
python verify_package.py
```

This verifies SHA-256 checksums, Python syntax and source counts. It is not a PDE correctness proof. Run it before regenerating bundled outputs; regenerated files may differ in metadata or final floating-point digits. The checksum manifest records this delivered snapshot.

## Reproduce the current numerical evidence

Run from the bundle root unless a `cd` command says otherwise. Commands work in a terminal or the PyCharm terminal. Keep the folder layout and numerical filenames intact.

Finite-time results for Tables 1–4, both initialization conventions, and saved states:

```bash
python evidence/reproduce_concern1.py --output rerun_outputs/final_results
cd rerun_outputs
python ../evidence/crosscheck_full_fft.py
cd ..
```

The cross-check reads `final_results/audit_results.json` relative to its working directory, hence the explicit `cd` above. Full grid counts are on `[0, 2L)` without a duplicated endpoint. The manuscript's Table 2 and replacement Table 3 use `offset` initialization; `critical` is the reviewer-comparison convention.

Additional nonlinear jobs used in Table 5:

```bash
python evidence/extended_checks.py --job 0 --output rerun_outputs/extended_results
python evidence/extended_checks.py --job 1 --output rerun_outputs/extended_results
python evidence/extended_checks.py --job 2 --output rerun_outputs/extended_results
python evidence/extended_checks.py --job 3 --output rerun_outputs/extended_results
```

Jobs contain 800,000–1,600,000 time steps. Saved complete results are included so they can be inspected without rerunning. The commands produce per-job JSON and NPZ files. The delivered `extended_checks.csv` is the concatenation of each job's `rows` with its `job` identifier; Table 5 uses job 0 at T=500 followed by the terminal row of each job. No standalone aggregate-CSV generator was part of the original run.

## Regenerate figures

Current Figure 2, from the saved Table 2 CSV, with the revised finite-time interpretation:

```bash
python evidence/regenerate_branch_figure.py
```

It writes `figures/figure4_bifurcation_branch.png` and `.pdf`. The filename is historical; this is Figure 2 in the manuscript. It does not rerun the PDE.

Figure 1:

```bash
cd patched_project
python 6.1.py
cd ..
```

It writes `patched_project/figures_task19/figure1_dispersion_real_high.png`.
Earlier plotting scripts remain in the archive but may carry old labels. Use the two commands above for the current manuscript figures.

## Original/corrected project checks

With PyTorch available, from the bundle root:

```bash
python audit_workflow/check_patched_solver.py
python audit_workflow/check_original_solver.py
```

These write diagnostics under `audit_evidence/rechecked/`. Original-project checks deliberately expose historical diagnostic problems.

The earlier project workflow can be replayed from `patched_project` with `python 3.1.py`, `python 2.py`, `python 5.py`, `python 4.1.py`, `python 4.2.py`, `python 6.1.py`, and `python 7.1.py --preset quick --device cpu`. Its quick preset is not the current replacement Table 3 experiment; the paper preset in `7.1.py` also uses different settings. `2.py`'s spatial section records grid configurations, not spatial errors. Do not mix these outputs with current tables.

## Evidence boundaries and publication

Terminal residuals refer to the spatially discretized system, not rigorous continuum error bounds. Floating-point-level grid differences are not used to infer a convergence order. Long-time checks at a single offset do not establish the asymptotic coefficient as the offset tends to zero.

`docs/GITHUB_PUBLICATION_NOTES_CN.txt` gives the repository URL and instructions for citing a verified commit in the manuscript and response. `docs/PACKAGE_VALIDATION.json` is the historical validation report for the local ZIP before GitHub upload; its `github_published: false` field describes that earlier packaging stage. Scientific source files and numerical results are unchanged from that validated ZIP. Repository documentation and the checksum manifest have been updated for this upload. No software license has been selected; public visibility does not itself grant an open-source license.
