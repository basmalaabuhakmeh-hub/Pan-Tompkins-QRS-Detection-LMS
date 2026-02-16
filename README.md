# Pan-Tompkins QRS Detection — Reproduction & LMS Enhancement
  
Reproduction and enhancement of the Pan-Tompkins QRS detection algorithm for ECG signals, with LMS-based adaptive thresholding and full DSP analysis (frequency response, pole-zero, group delay).


## Overview

This project reproduces the classic **Pan-Tompkins algorithm** (1985) for real-time QRS complex detection in ECG signals and extends it with an **LMS-based adaptive thresholding** stage to improve robustness in noisy conditions.

**Main components:**

1. **Five-stage Pan-Tompkins pipeline**  
   Bandpass filter (5–15 Hz), derivative filter, squaring, moving-window integration, and adaptive thresholding — implemented and applied to MIT-BIH Arrhythmia Database (record 100).

2. **DSP analysis of filters**  
   For bandpass, derivative, and integration stages: magnitude/phase response, pole-zero plots, and group delay (via `plot_filter_analysis`, `plot_combined_response`, `plot_individual_filter_analysis`).

3. **LMS-enhanced detection**  
   Adaptive thresholding using a Least Mean Squares–style update for the detection threshold, improving sensitivity and PPV under moderate and high noise (AWGN at various SNR levels).

4. **Evaluation**  
   Sensitivity, PPV, and F1 score vs. reference annotations; comparison between original and LMS methods.

All processing is in a single module: **`Main.py`** (class `PanTompkinsQRS`, MIT-BIH loading, AWGN, and `main()` demo).

---

## Project structure

```
.
├── README.md
├── 1220724_1220871_1220184.pdf   # Full project report
└── Main.py                       # Pan-Tompkins + LMS implementation and analysis
```

---

## Requirements

- **Python 3.x**
- **NumPy**, **Matplotlib**, **SciPy** (signal processing and plotting).
- **wfdb** — for reading MIT-BIH records: `pip install wfdb`
- **scikit-learn** — for confusion matrix (evaluation): `pip install scikit-learn`

MIT-BIH data is loaded via `wfdb` from the PhysioNet `mitdb` database (automatic download when using `pn_dir='mitdb'`). If the database is unavailable, the code falls back to synthetic ECG for demonstration.

---

## Usage

1. Install dependencies:

```bash
pip install numpy matplotlib scipy wfdb scikit-learn
```

2. Run the main script (loads record 100, 15 s, adds AWGN at 15 dB SNR, runs original + LMS detection, plots filter analyses and detection results):

```bash
python Main.py
```

**What the script does:**

- Loads MIT-BIH record `100` for 15 seconds (or synthetic ECG if loading fails).
- Adds AWGN at a configurable SNR (default 15 dB; change `snr_level` in `main()` for 20, 15, 10, 5 dB).
- Runs the full Pan-Tompkins pipeline (bandpass → derivative → squaring → integration).
- Plots individual filter analyses (bandpass, derivative, integration): magnitude, phase, pole-zero, group delay.
- Detects QRS with **original** and **LMS** methods and plots detection results (ECG + detected peaks, optional adaptive threshold).
- Computes and prints sensitivity, PPV, F1, and TP/FP/FN for both methods.

To use only the detector in your own code: instantiate `PanTompkinsQRS(fs=360)`, call `detect_qrs(ecg_signal, method='original')` or `method='lms'`, and use `evaluate_performance(detected_peaks, reference_peaks)` for metrics.

---

## Algorithm summary

| Stage              | Description |
|--------------------|-------------|
| **Bandpass**       | 2nd-order Butterworth 5–15 Hz; reduces baseline drift and high-frequency noise. |
| **Derivative**     | 5-point filter to emphasize QRS slopes. |
| **Squaring**       | Point-wise square; positive values, emphasizes large slopes. |
| **Integration**    | Moving window (e.g. 150 ms); smooths and extracts QRS energy. |
| **Thresholding**   | Original: adaptive dual threshold + refractory period. LMS: threshold updated with local statistics and LMS-style rule. |

---

## Report

Detailed methodology, literature review, DSP analysis, LMS design, and evaluation (including sensitivity/PPV/F1 under clean, moderate, and high noise) are in:  
**`1220724_1220871_1220184.pdf`**


