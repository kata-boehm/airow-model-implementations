# Heart Rate Interval Segmentation — Notebook Guide

This repository contains implementations of six models for automatically detecting workout interval boundaries from heart rate time series data. Each model is a standalone Jupyter notebook that can be run top-to-bottom.

---

## Notebooks

| Notebook | Model | Features |
|---|---|---|
| `BiLSTM.ipynb` | Bidirectional LSTM | 39 (full set) |
| `BiLSTM_ReducedFeatures.ipynb` | Bidirectional LSTM | 7 (ablation study) |
| `CNN_Improved.ipynb` | Multi-scale Residual CNN | 39 (full set) |
| `CNN_Improved_ReducedFeatures.ipynb` | Multi-scale Residual CNN | 7 (ablation study) |
| `TCN_Dilated.ipynb` | Temporal Convolutional Network | 39 (full set) |
| `TCN_Dilated_ReducedFeatures.ipynb` | Temporal Convolutional Network | 7 (ablation study) |
| `UNet_1D.ipynb` | 1D U-Net | 39 (full set) |
| `UNet_1D_ReducedFeatures.ipynb` | 1D U-Net | 7 (ablation study) |
| `Per session number of intervals features_VAE_V4_Optimized_OG.ipynb` | VAE-based Neural Network | 39 (full set) |
| `XGBoost gradient boosting_per session number of intervals features_FINAL.ipynb` | XGBoost | 39 (full set) |
| `Model Comparison Final.ipynb` | Cross-model comparison | — |

---

To compare results across all models, run `Model Comparison Final.ipynb` after all individual model notebooks have been executed and saved their results to `results/`.

## How to Run

1. **Update Section 3 (Data Configuration)** with your local data paths if they differ from the defaults.
2. **Adjust Section 2 (Configuration)** if you want to change hyperparameters (e.g., window sizes, model depth, epochs).
3. Run all cells from top to bottom.

---

## Input Data Format

### File naming convention
Each session must be a CSV file named:
```
{session_name}_with_manual_labels.csv
```

### Required columns

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime (UTC) | One row per second (1 Hz signal) |
| `heart_rate` | integer (bpm) | Raw heart rate values; NaN allowed (forward-filled) |
| `power` | float (watts) | Power output; used only for visualization, NaN allowed |
| `Manual_Timestamps` | bool / 0-1 | Ground truth boundary labels: `True` or `1` at each interval start, `False` or `0` elsewhere |

### Notes
- Sessions are assumed to be **1 Hz** (one row per second). Timestamps must be sortable and unique.
- The first row of each session is always treated as a boundary (index 0 forced to 1).
- The last 5 rows are forced to 0 (suppresses spurious end-of-session detections).

---

## Folder Structure

All paths are relative to the notebook working directory.

```
AthleteDataCoding/
  Athlete12/
    OGs/       <- Raw .fit files (115 files; not loaded by notebooks)
    GTs/       <- Labeled CSVs (66 files; loaded by notebooks)
  Athlete2/
    OGs/       <- Raw .fit files (10 files; not loaded by notebooks)
    GTs/       <- Labeled CSVs (10 files; loaded by notebooks)
  SpecialBike_Labeled_NotSorted_Biking/
    OGs/       <- Raw .fit files (20 files; not loaded by notebooks)
    GTs/       <- Labeled CSVs (20 files; loaded by notebooks)
  SpecialBike_Labeled_NotSorted_Rowing/
    OGs/       <- Raw .fit files (2 files; not loaded by notebooks)
    GTs/       <- Labeled CSVs (2 files; loaded by notebooks)

```

Only the `GTs/` folders are read by the notebooks. The raw `.fit` files are defined in the data config but not used during training or evaluation.

### Fit file availability

| Athlete group | GTs | Fit files | Notes |
|---|---|---|---|
| Athlete12 | 66 | 115 | 2 GTs have no matching fit: `Rowing_indoor2`, `Rowing_indoor3` (not used in training) |
| Athlete2 | 10 | 10 | Full match |
| SpecialBike_Labeled_NotSorted_Biking | 20 | 20 | Full match |
| SpecialBike_Labeled_NotSorted_Rowing | 2 | 2 | Full match |

---

## Dataset Split

- **Training sessions:** 75 (51 rowing, 24 biking)
- **Test sessions:** 18 (12 rowing, 6 biking) — held out, defined in Section 3

---

## Output

Each notebook saves results to `results/` as a `.pkl` file (e.g., `results/bilstm_results.pkl`). The file contains per-session metrics: F-beta score (2:1 recall bias), precision, recall, and mean timing error (seconds).
