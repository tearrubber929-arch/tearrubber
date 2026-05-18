# Drilling Dysfunction Detection Using Spectral Feature Extraction and Machine Learning

## Overview

This notebook implements an automated pipeline for detecting five types of drilling dysfunctions from real-time sensor data. The approach combines **Fast Fourier Transform (FFT)** and **Wavelet Packet Transform (WPT)** feature extraction with rule-based labelling and multi-output machine learning classifiers.

The workflow is designed around a carbonate formation drilling dataset and analyzes seven drilling signals to identify abnormal spectral patterns associated with known downhole problems.

---

## Drilling Signals Analyzed

| Signal | Unit | Description |
|---|---|---|
| `RPM` | 1/min | Rotary Speed |
| `SPP` | psi | Standpipe Pressure |
| `TORQUE` | kft·lbf | Surface Torque |
| `WOB` | klbf | Weight on Bit |
| `ROP` | ft/h | Rate of Penetration |
| `Flow Rate` | gpm | Mud Flow Rate |
| `UCS` | psi | Unconfined Compressive Strength |

---

## Dysfunctions Detected

| Dysfunction | Primary Spectral Signature |
|---|---|
| **Stick-Slip** | Elevated low-frequency FFT energy |
| **Whirl** | Elevated mid-frequency FFT energy |
| **Bit Bounce** | Elevated high-frequency FFT energy |
| **Differential Sticking** | Collapse in WPT total energy and entropy |
| **Pack-Off** | High low-frequency FFT energy + suppressed mid-frequency energy |

---

## Methodology

### 1. Data Loading and EDA
- Load drilling data from `data.csv`
- Inspect dataset shape, column types, and basic statistics
- Visualize raw signal time series

### 2. Signal Denoising
- Apply a **Savitzky-Golay filter** (`window_length=11`, `polyorder=3`) to each signal
- Preserves signal shape while removing high-frequency noise

### 3. Feature Extraction

#### Time-Domain Features
Computed for each signal:
- Mean, standard deviation, variance, RMS, peak-to-peak, range
- Skewness, kurtosis, energy, power
- Zero-crossing rate, peak count, crest factor, shape factor, impulse factor, clearance factor

#### Frequency-Domain Features (FFT)
- **Spectral centroid, spread, skewness, kurtosis**
- **Spectral energy and entropy**
- **Peak frequency and mean frequency**
- **Band power ratios**: low (0–25%), mid (25–75%), high (75–100%) of the Nyquist range

#### Time-Frequency Features (WPT)
- Decompose each signal using **Daubechies-4 wavelet** at **level 3** (8 sub-bands)
- Extract **sub-band energies**, **total energy**, and **Shannon entropy**
- Applied via rolling windows of 128 samples

### 4. Dysfunction Labelling (Rule-Based)

Spectral features are smoothed (5-sample rolling mean) and compared against a lagged rolling baseline (200-sample window) to generate binary dysfunction labels for each depth sample.

**FFT-based rules:**
- Stick-Slip: `fft_low > 1.8 × baseline`
- Whirl: `fft_mid > 2.0 × baseline`
- Bit Bounce: `fft_high > 2.5 × baseline`
- Pack-Off: `fft_low > 1.5 × baseline` AND `fft_mid < 0.8 × baseline`

**WPT-based rules:**
- Differential Sticking: `wpt_total_energy < 0.8 × baseline` AND `wpt_entropy < 0.85 × baseline`

Both FFT and WPT rules are combined in the final `detect_dysfunctions` function.

Labelled ML datasets are exported as CSV files per signal (e.g., `spectral_ml_TORQUEkft.lbf_FFT_only.csv`).

### 5. Machine Learning Classification

Features and labels from all signals are assembled into a multi-output classification task.

**Classifiers evaluated (via `MultiOutputClassifier`):**
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Support Vector Machine (RBF kernel)
- Logistic Regression
- Decision Tree
- Gaussian Naive Bayes

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score (weighted average per dysfunction type)

Train/test split: 80/20, with `StandardScaler` normalization applied.

### 6. Feature Importance Analysis

- Extract feature importances from the **Random Forest** estimator
- Visualize the top 15 and top 20 most important features per signal and globally across all signals
- Global analysis concatenates features from all signals with signal-prefixed column names

### 7. Overfitting Analysis

Training vs. test metrics are compared side-by-side for all classifiers to identify overfitting behaviour.

### 8. Visualization

- Raw vs. denoised signal plots
- FFT magnitude and power spectra
- WPT sub-band reconstructed signals
- Spectral dysfunction detection plots per signal (depth-indexed, inverted x-axis)
- Depth density distributions for each dysfunction (KDE plots)
- Pearson correlation heatmaps between features and labels
- Confusion matrices for all classifiers
- Dispersion plot of dysfunction occurrences across depth
- Depth interval summary table per dysfunction type
- Heatmap of relative FFT/WPT energy changes per dysfunction type

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy pywt scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling and numerical operations |
| `matplotlib`, `seaborn` | Visualization |
| `scipy` | FFT, Savitzky-Golay filter, signal analysis |
| `pywt` | Wavelet Packet Transform |
| `scikit-learn` | Machine learning classifiers and evaluation |

---

## Input Data

- **File:** `data.csv`
- **Expected columns:** A depth column (containing "depth", "md", or "measured" in the name) and one or more drilling signal columns
- If no depth column is found, a synthetic index-based depth column is created automatically

---

## Output Files

One labelled ML dataset CSV is generated per signal per detection method:

```
spectral_ml_<SignalName>_FFT_only.csv
spectral_ml_<SignalName>_WPT_only.csv
```

Each file contains the smoothed spectral features and binary dysfunction labels indexed by depth.

---

## Key Results

- FFT-derived features — especially **total energy**, **low-frequency energy**, and **mid-frequency energy** — dominate global feature importance rankings across all signals
- **Flow Rate**, **RPM**, and **UCS** contribute the highest-importance features in the global analysis
- WPT features provide complementary information, particularly for Differential Sticking and Pack-Off detection

---

## Notes

- The `detect_dysfunctions` function uses both FFT and WPT features internally; exported files are labelled `_FFT_only` as a naming convention from earlier iterations
- Detection thresholds (e.g., `1.8×`, `2.5×`) are empirically set and may require tuning for different formations or drilling programmes
- The `detect_dysfunctions_wpt_only` variant uses WPT features exclusively and applies a separate set of detection rules
