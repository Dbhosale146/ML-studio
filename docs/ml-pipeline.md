# ML pipeline

This is every stage the agent runs, with the rules and thresholds it uses. All of it lives in `services/api/src/datalyon/ml/`. The agent stages in `agent/stages/` only call these functions and narrate the results.

## 1. Ingest (`ingest.py`)

- Accepts `.xlsx`, `.xls` and `.csv`, sniffed by content rather than by extension. Size, row and column limits come from settings.
- Per sheet, the header row is the first row that looks like a header (mostly text, followed by data rows), so title rows above it are skipped. Merged header gaps are filled, and duplicate or blank names are made unique.
- Cleaning: empty rows and columns are dropped. `#N/A`, `n/a`, `-` and similar become missing. Currency and thousands separators are stripped from numeric-looking text. Excel serial dates and `dd/mm` text dates are converted in a separate pass.
- A workbook with several usable sheets asks the user which one to use; the largest is suggested.

## 2. Profiling (`profiling.py`)

Each column gets a kind (numeric, categorical, boolean, datetime, id, text, constant, empty), its missing share, unique count, top values or a histogram, and quality warnings. The dataset gets a 0–100 quality score.

## 3. Detection (`detection.py`)

- **Target:** the column named or implied by the goal, scored against all columns by name similarity and suitability. ID-like, constant and free-text columns are excluded. The user confirms it ✋.
- **Problem type:** forecasting if the goal talks about the future and there is a usable time column. Otherwise binary for boolean or 2-value targets, multiclass for categorical targets (or small integer codes named like `grade`, `tier`, `level`), and regression for everything else.
- **Forecasting:** frequency is inferred from the median gap, series keys from low-cardinality columns that repeat per date. If the goal does not state a horizon, the agent asks ✋.
- **Leakage by name:** a column is suspect only if it pairs with the target and carries an after-the-fact qualifier (for example `cancellation_reason`, `refund_date`). Names like `sale_date` next to `sale_price` are not flagged.

## 4. Strategy (`strategy.py`)

| Situation | Validation | Primary metric |
|---|---|---|
| Binary, positive share < 20% | Stratified k-fold + 20% holdout, class weighting | PR-AUC |
| Binary, balanced | Stratified k-fold + 20% holdout | ROC-AUC |
| Multiclass | Stratified k-fold + 20% holdout | Macro-F1 |
| Regression | k-fold + 20% holdout | RMSE; MAE when the target is heavily skewed (R² reported) |
| Has a time column (tabular) | Time-ordered CV on the oldest 80% + newest 20% holdout | as above |
| Forecasting | Rolling-origin backtest, up to 4 windows of one horizon | MAPE; WAPE when the series contain zeros |

The fold count shrinks for small data. Class weighting also turns on probability calibration for binary.

## 5. Features and cleaning (`features.py`, `preprocessing.py`)

The cleaning plan is shown to the user before training ✋; optional steps can be switched off. Everything is fitted inside the sklearn `Pipeline`, so it learns on training folds only.

- Type coercion and label normalisation (`" north"`, `"NORTH"` → `north`).
- Rare categories: columns with at least 10 values where some are below 1% get those grouped into `Other`.
- Numeric outliers are capped at the 1st and 99th percentiles.
- Dates become year, month, day of week and day of year with sin/cos encodings.
- Missing-value flags are added for columns with at least 5% missing.
- Derived features (ratio, diff, sum, product, log1p) are validated: each is kept only if it improves CV.
- Linear models get median imputation, scaling and one-hot encoding. Tree models get ordinal encoding with native missing-value handling.

## 6. Training (`candidates.py`, `training.py`)

- Candidates: a majority or mean baseline, Logistic or Ridge regression, Random Forest, Hist Gradient Boosting, LightGBM and XGBoost. Random Forest is skipped on very large data and boosting on very small data, and the reason is shown.
- **Round 1:** cross-validate every candidate, then Optuna-tune the top ones within the remaining budget. Each round always trains at least one real model, even on a tiny budget.
- Every fold reports progress, and every run becomes an `experiment` event.

## 7. Critic (`critic.py`)

| Check | Rule | Fix in round 2 |
|---|---|---|
| Leakage (single feature) | One feature alone reaches AUC > 0.97 or \|corr\| > 0.95 with the target | Exclude the column |
| Leakage (missingness) | Whether a value is missing predicts the target with AUC ≥ 0.95 | Exclude the column |
| Leakage (dominance) | Score ≥ 0.97 and one feature drives ≥ 50% of permutation importance | Exclude the column |
| Leakage (name) | Suspect names from detection | Exclude the column |
| Baseline | Lift over the baseline is within 1.5 × fold std (noise); PR-AUC is judged relative to the positive rate | Stop honestly with `no_signal` if no fix helps |
| Overfitting | Train − validation gap > 0.08 (error ratio > 1.6 for losses) | Stronger regularisation |
| Stability | Fold std > 0.04 (relative std > 15% for losses) | Reported |
| Imbalance | Binary minority < 20%; multiclass smallest class < 40% / k | Class weights |
| Calibration | Expected calibration error > 5 points | Isotonic or Platt calibration |
| Heteroscedasticity | Positive skewed target whose error grows with its size | Log-transform the target |
| Worst segment | The categorical slice with the worst error | Reported |

**Round 2** re-runs all candidates with the fixes. A changed exclusion set means round 1 scores are no longer comparable.

## 8. Final choice and evaluation (`evaluation.py`, `explain.py`)

- The top 3 from round 2 are refit on all CV data and scored on the untouched holdout. Ties within noise go to the lower-std model, then the faster one.
- The user accepts the final model ✋.
- Reported: every metric against the baseline, ROC and PR curves, a confusion matrix at the F1-optimal threshold, a quantile-binned calibration curve, score distributions, predicted-vs-actual and residuals for regression, and per-class tables for multiclass.
- Permutation importance on the holdout. Per-prediction drivers use SHAP (tree or linear explainer), mapped back to raw columns, with an occlusion fallback.

## 9. Forecasting (`forecasting.py`)

- One global LightGBM or XGBoost model across all series, with a series code as a feature.
- Lags are at least the horizon, plus the seasonal lag. Rolling means and std are computed on shifted values, plus calendar features. Nothing from the future can leak in.
- The seasonal naive model is the baseline. Prediction intervals come from backtest residual quantiles (80%).
- The critic can add differencing or calendar dummies for round 2.

## 10. Publish and inference (`registry.py`, `inference.py`)

- `artifacts/models/{id}/vN/` holds `model.joblib`, the model card (data hash, library versions, seed, decisions) and the input schema: typed fields with ranges, options, importance and examples.
- Single prediction validates every field and warns about out-of-range numbers, unseen categories, empty important fields and unknown fields.
- Batch scoring picks the sheet that best matches the training columns, fills missing columns with typical values and appends prediction (and probability) columns.
- Retraining with the agent creates version N+1 of the same model.
