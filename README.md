# Linear Regression from Scratch — Assignment 5

A from-scratch (NumPy-only) implementation of Linear Regression, trained with manually
implemented Batch Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient
Descent — no `sklearn.linear_model.LinearRegression`, `SGDRegressor`, or any library that
trains the model for you.

## Project structure

```
assignment_linear_regression/
├── data/                     # place downloaded datasets here (see data/README.md)
├── handwritten/              # scanned handwritten Part 2 manual gradient-descent calculation
├── src/                      # all core classes (model, trainer, preprocessing, evaluator, ...)
├── experiments/              # one script per assignment sub-part (2, 3a, 3b, 3c, 4, 7)
├── models/                   # saved trained model(s) (.pkl)
├── figures/                  # all generated plots
├── main.py                   # end-to-end pipeline on the California housing dataset
├── report.md / report.pdf    # full written report (analysis, tables, answers)
└── README.md
```

## How to run

```bash
pip install numpy scikit-learn matplotlib pandas librosa pillow

python main.py                                        # full pipeline, California housing
python experiments/part2_manual_gd_verification.py     # Part 2: hand-calc vs code
python experiments/part3a_learning_rate_experiment.py  # Part 3a: learning-rate sweep
python experiments/part3b_normalization_experiment.py  # Part 3b: raw vs standardized
python experiments/part3c_optimizer_comparison.py      # Part 3c: Batch vs SGD vs Mini-batch
python experiments/part4_test_evaluation.py            # Part 4: test-set metrics + plot
python experiments/part7_coefficient_interpretation.py # Part 7: coefficient interpretation
python experiments/part3_image_real_data.py            # Part 3: real face-age images
python experiments/part3_audio_real_data.py             # Part 3: real COUGHVID audio (full set, ~1hr)
```

`sklearn` is used ONLY for: loading `fetch_california_housing`, and `train_test_split`.
It is never used to fit the regression model itself.

## Datasets

| Dataset | Modality | Status |
|---|---|---|
| `sklearn.datasets.fetch_california_housing` | Numeric | Fully run — all results in `report.md` |
| Kaggle `frabbisw/facial-age` | Image (age regression) | Fully run on all 9,778 real images — test MSE 251.8 vs. baseline 621.4. See `report.md` Section 9.1 |
| COUGHVID (Zenodo record 4498364) | Audio (`cough_detected` regression) | Fully run on all 27,550 real clips — test MSE 0.0416 vs. baseline 0.1526. See `report.md` Section 9.2 |

`data/README.md` documents where each dataset lives locally (not committed to git — see
`.gitignore` — since `data/coughvid/` alone is ~1.3GB) and how to re-download them.

## What's implemented

- `src/Data_Handler.py` — load / 3-way split / feature names (California housing)
- `src/Data_Handler_Image.py` — same interface, for face-age images (inherits `DataHandler`)
- `src/Data_Handler_Audio.py` — same interface, for COUGHVID audio (inherits `DataHandler`)
- `src/Data_Preprocessor.py` — standardization, fit ONLY on training data
- `src/LinearRegression_Model.py` — predict / MSE loss / gradient / parameter update
- `src/LinearRegression_Trainer.py` — Batch GD / SGD / Mini-Batch GD training loops
- `src/Regression_Evaluator.py` — MSE / RMSE / MAE
- `src/Regression_Visualizer.py` — loss curves, multi-curve comparison, actual-vs-predicted
- `src/Model_Persistence.py` — save/load via pickle

## Known limitations (see report.md for full discussion)

1. The image/audio dataset handlers were validated on synthetic stand-in data only —
   real accuracy numbers require running them against the actual downloaded datasets.
2. Part 2's manual gradient-descent calculation was verified by code
   (`experiments/part2_manual_gd_verification.py`) and cross-checked against the handwritten
   copy at `handwritten/part2_manual_gradient_descent.pdf` — both match to numerical
   precision.
