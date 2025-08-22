# Brain Extraction Analysis — Code for "Comparison of Automated Brain Extraction Methods in over 5000 CT Scans"

This repository contains code and analysis assets used to compare multiple automated CT brain extraction/skull-stripping methods at scale. It accompanies the paper “Comparison of Automated Brain Extraction Methods in over 5000 CT Scans” and provides:
- Quantitative rim-intensity metrics to assess mask quality (e.g., rim volume, HU tail metrics, bone/air fractions)
- Qualitative exclusion-rate figures overall, by task (Registration, Volumetrics, Deep-Learning), and by subgroups (artifact, craniotomy, CTA)

Methods covered include: CTBET (v1), Robust-CT-BET, SynthStrip, HD-CTBET, CT_BET, Brainchop, and CTbet-Docker.

## Repository layout

- `data/` — Input images and QC CSVs (structure reference; actual datasets are not included)
  - `images/` — Per-method PNGs (often large; typically ignored in Git)
  - `qc/` — Method-level QC CSVs used for figure generation
- `python/`
  - `quantitative/get_rim_metrics.py` — Computes rim metrics from CT + mask NIfTI pairs
  - `quantitative/analyze_rim_metrics.ipynb` — Loads `results/quantitative/rim_metrics_r1.csv` and compares methods
  - `qualitative/` — Notebooks/utilities for qualitative summaries
- `R/` — R scripts to generate publication-style figures
- `results/` — Derived outputs (CSV summaries and PDFs)

## How this maps to the paper

- Quantitative (Rim metrics, Python)
  - Input: CT volume and corresponding brain masks per method
  - Outputs: `results/quantitative/rim_metrics_r1.csv` (per-scan metrics)
  - Notebook: `python/quantitative/analyze_rim_metrics.ipynb` compares SynthStrip vs Robust-CT-BET using paired scans and reports deltas and improvement rates

- Qualitative (Exclusion rates and subgroups, R)
  - Scripts: `R/get_plots_scan.R` (scan-level figures); similar scripts under `R/` for additional plots
  - Outputs (examples):
    - `results/qualitative/scan_lvl/figureA_overall_exclusion_rates.pdf`
    - `results/qualitative/scan_lvl/figureB_task_exclusion_rates.pdf`
    - `results/qualitative/scan_lvl/figureC_multiple_failures_rates.pdf`
    - `results/qualitative/scan_lvl/figureD_subgroup_artifact.pdf`
    - `results/qualitative/scan_lvl/figureE_subgroup_craniotomy.pdf`
    - `results/qualitative/scan_lvl/figureF_subgroup_cta.pdf`

## Rim metrics (what they measure)

Given a CT and its brain mask, the rim is defined as the outer one-voxel shell of the mask. On rim voxels, we compute:
- `rim_vol_ml`: physical rim volume (mL)
- `p95`, `p99`: 95th and 99th percentiles of HU within the rim
- `vol_bone_ml`, `vol_air_ml`: mL within rim above bone threshold (e.g., ≥70 HU) or below air threshold (e.g., ≤−200 HU)
- `frac_bone`, `frac_air`: fractions of the rim volume meeting those criteria

Higher bone/air fractions in the rim can indicate mask leakage or erosion issues. The notebook computes paired deltas (method A − method B) and the percentage of scans where one method is better.

## Requirements

Python (3.9+ recommended):
- numpy, pandas, nibabel, scipy

R:
- dplyr, tidyr, ggplot2, scales, showtext, sysfonts

Optional: Cairo PDF for high-quality PDF export (`ggsave(..., device = cairo_pdf)`).

## Reproducibility quick start

1) Update paths (important)
- `python/quantitative/get_rim_metrics.py`: set `ct_dir`, `synth_dir`, `robust_dir`, and `out_dir` for your environment.
- `R/get_plots_scan.R`: update font paths (or comment out `font_add`) and input CSV paths.

2) Quantitative (Python)
- If `results/quantitative/rim_metrics_r1.csv` exists, open `python/quantitative/analyze_rim_metrics.ipynb` to reproduce comparisons.
- To regenerate metrics from NIfTI data, run `get_rim_metrics.py` after setting the directories above.

3) Qualitative (R)
- Run `R/get_plots_scan.R` to produce the scan-level figures listed above.

## Outputs

- Quantitative metrics: `results/quantitative/rim_metrics_r1.csv`
- Figures (PDF): saved under `results/qualitative/scan_lvl/` and `results/qualitative/subj_lvl/`

## Data and versioning

- This repository does not include original CT data or masks.
- `.gitignore` is configured to ignore `*.csv`, `*.png`, and `*.pdf` by default (adjust if you wish to track outputs).

## Citation

If you use this code or figures, please cite the paper:

> “Comparison of Automated Brain Extraction Methods in over 5000 CT Scans,” authors, venue, year. [Add DOI or URL]

You can also cite specific tooling (e.g., SynthStrip, Robust-CT-BET) as appropriate.

## Acknowledgments

We acknowledge the authors and maintainers of the brain extraction methods evaluated (CTBET, Robust-CT-BET, SynthStrip, HD-CTBET, CT_BET, Brainchop, CTbet-Docker). This repo provides analysis code and derived results/figures only; obtain original tools/data from their respective sources.