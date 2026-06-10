# Delhi NCR — Air Quality Index (AQI) Prediction

End-to-end ML pipeline for hourly AQI forecasting across Delhi NCR cities (Delhi, Noida, Gurugram, Faridabad, Ghaziabad). Fuses CPCB pollutant data with Open-Meteo meteorological data, trains a Random Forest + XGBoost ensemble, provides SHAP-based explainability, and exposes an interactive ipywidgets UI for 7-day autoregressive forecasting.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Data Sources](#data-sources)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Feature Engineering](#feature-engineering)
- [Models & Hyperparameters](#models--hyperparameters)
- [Explainability](#explainability)
- [Interactive Forecasting Widget](#interactive-forecasting-widget)
- [Saved Artifacts](#saved-artifacts)
- [Reproducing Results](#reproducing-results)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture Overview

```
Raw CSVs (AQI + Meteorology)
        │
        ▼
  Data Ingestion & Type Coercion
        │
        ▼
  Outlier Handling (IQR winsorization @ 1–99%)
        │
        ▼
  Hourly Left-Join Merge (AQI ← Meteo on datetime_hour)
        │
        ▼
  Feature Engineering
  ├── Cyclic time encodings (hour/month/dow sin-cos)
  ├── Lag features: pm25/pm10/no2/aqi @ [1,2,3,6,12,24]h
  ├── Rolling means: 3h / 6h / 24h windows
  └── Derived: temp×humidity, pm_ratio, nox_approx, dew_depression
        │
        ▼
  Temporal Train / Val / Test Split (70/15/15, no shuffle)
        │
        ├──► RandomForestRegressor  ──┐
        └──► XGBRegressor            ├──► Ensemble average
                                     │
                                     ▼
                          RMSE / MAE / R² / MAPE
                          SHAP global + local explanations
                          What-if scenario analysis
                          7-day autoregressive widget
```

---

## Data Sources

| File | Description | Notes |
|------|-------------|-------|
| `delhi_ncr_aqi_dataset.csv` | Hourly pollutant readings from CPCB monitoring stations | Columns: `pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `aqi`, `aqi_category`, `city`, `station`, `datetime` |
| `open-meteo-28.58N77.19E214m.csv` | Hourly meteorological data from Open-Meteo API (28.58°N, 77.19°E, 214 m) | `skiprows=3`; columns renamed to `met_temp`, `met_humidity`, `met_wind_speed`, `met_wind_dir`, `met_precip`, `met_pressure`, `met_cloud_cover`, `met_dew_point` |

Both files should be placed at the Colab `/content/` path or updated in the ingestion cell if running locally. Raw data is excluded from version control — see `data/README.md`.

---

## Repository Structure

```
Air_Quality/
│
├── AQI_Prediction.ipynb        # Main notebook — full end-to-end pipeline
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # This file
│
├── data/
│   ├── .gitignore              # Excludes raw CSVs from version control
│   └── README.md               # Data acquisition instructions
│
└── src/                        # Optional — refactored modules (not yet extracted)
```

Serialized model artifacts are written to the working directory by the notebook's save cell:

```
rf_aqi_model.pkl          Random Forest regressor
xgb_aqi_model.pkl         XGBoost regressor
feature_scaler.pkl        StandardScaler (fitted on train set)
feature_columns.pkl       Ordered feature list (required for inference)
aqi_label_encoder.pkl     AQI category LabelEncoder
model_summary.json        Performance summary + dataset metadata
```

---

## Requirements

Python 3.8+ (3.10 recommended).

```bash
pip install -r requirements.txt
```

Core dependencies:

```
numpy
pandas
scikit-learn
matplotlib
seaborn
plotly
xgboost
shap
lime
joblib
ipywidgets
jupyterlab
```

---

## Getting Started

```bash
git clone https://github.com/MisterStranger03/Air_Quality.git
cd Air_Quality
python -m venv .venv && source .venv/bin/activate   # or conda create -n aqi python=3.10
pip install -r requirements.txt
jupyter lab
```

Open `AQI_Prediction.ipynb` and run cells sequentially. The notebook is designed for Google Colab but works locally — update the two CSV paths in the **Data Ingestion** cell if running outside Colab.

---

## Pipeline Walkthrough

### 1 — Data Ingestion & Coercion

Both CSVs are loaded with `dtype=str` to handle mixed-type columns safely, then numeric columns are cast via `pd.to_numeric(errors='coerce')`. Quote characters are stripped from all fields before conversion. Datetimes are parsed with `format='mixed'`.

### 2 — EDA & Outlier Detection

IQR-based outlier report is generated across all pollutants and meteorological variables. Outliers are handled via **winsorization at the 1st and 99th percentile** (not dropped), preserving record count while bounding extreme values.

### 3 — Dataset Merge

The two DataFrames are joined on `datetime_hour` (floored to the hour) via a left join. Meteorological nulls after the join are filled first from the AQI dataset's own weather columns, then from column medians. Only the overlapping datetime window is retained for modelling.

### 4 — Feature Engineering

See [Feature Engineering](#feature-engineering) section below.

### 5 — Train / Val / Test Split

Temporal split with `shuffle=False` to prevent data leakage:

| Split | Size | Purpose |
|-------|------|---------|
| Train | ~70% | Model fitting |
| Validation | ~15% | Early stopping (XGBoost), intermediate evaluation |
| Test | 15% | Final held-out evaluation |

Features are standardised via `StandardScaler` fitted only on the train set.

### 6 — Model Training

Two regressors and two classifiers (AQI category) are trained independently. See [Models & Hyperparameters](#models--hyperparameters).

### 7 — Evaluation

- Regression metrics: RMSE, MAE, R², MAPE on both validation and test sets
- Residual diagnostics: distribution (skewness), residuals-vs-fitted, MAE broken down by AQI severity band
- City-wise breakdown: per-city RMSE and R² for both models
- Temporal plot: actual vs predicted over the first 500 test hours

### 8 — Explainability

SHAP TreeExplainer on XGBoost, permutation importance, and seasonal feature importance breakdowns. See [Explainability](#explainability).

### 9 — Interactive Widget

Autoregressive 7-day forecast via ipywidgets. See [Interactive Forecasting Widget](#interactive-forecasting-widget).

---

## Feature Engineering

| Group | Features | Count |
|-------|----------|-------|
| Raw pollutants | `pm25`, `pm10`, `no2`, `so2`, `co`, `o3` | 6 |
| Meteorological | `met_temp`, `met_humidity`, `met_wind_speed`, `met_wind_dir`, `met_pressure`, `met_precip`, `met_cloud_cover`, `met_dew_point`, `visibility` | 9 |
| Cyclic time encodings | `hour_sin/cos`, `month_sin/cos`, `dow_sin/cos` | 6 |
| Lag features (1/2/3/6/12/24h) | `{pm25,pm10,no2,aqi}_lag{N}h` | 24 |
| Rolling averages (3/6/24h) | `{pm25,pm10,no2,aqi}_roll{N}h` | 12 |
| Interaction / derived | `temp_humidity_interaction`, `wind_precip_interaction`, `dew_depression`, `pm_ratio`, `nox_approx` | 5 |
| Categorical | `city_code`, `station_code`, `season_code`, `is_weekend`, `aqi_cat_code` | 5 |

Rows with any null in a lag feature column are dropped after the lag construction step to avoid feeding NaN into models.

---

## Models & Hyperparameters

### Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators     = 300,
    max_depth        = 20,
    min_samples_split= 5,
    min_samples_leaf = 2,
    max_features     = 'sqrt',
    oob_score        = True,
    n_jobs           = -1,
    random_state     = 42
)
```

### XGBoost Regressor

```python
XGBRegressor(
    n_estimators         = 500,
    learning_rate        = 0.05,
    max_depth            = 8,
    min_child_weight     = 3,
    subsample            = 0.8,
    colsample_bytree     = 0.8,
    gamma                = 0.1,
    reg_alpha            = 0.1,
    reg_lambda           = 1.0,
    early_stopping_rounds= 30,   # monitored on validation RMSE
    eval_metric          = 'rmse',
    n_jobs               = -1,
    random_state         = 42
)
```

Classifier variants (`RandomForestClassifier`, `XGBClassifier`) use the same depth/estimator settings with `class_weight='balanced'` for the RF and `eval_metric='mlogloss'` for XGBoost.

### Ensemble Inference

Final AQI prediction is the simple average of RF and XGBoost outputs:

```python
pred_ensemble = (rf_model.predict(X) + xgb_model.predict(X)) / 2
```

### Results

> Run the notebook to populate this table. The saved `model_summary.json` contains the exact values from your run.

| Model | Split | RMSE | MAE | R² | MAPE |
|-------|-------|------|-----|----|------|
| Random Forest | Validation | — | — | — | — |
| Random Forest | Test | — | — | — | — |
| XGBoost | Validation | — | — | — | — |
| XGBoost | Test | — | — | — | — |

---

## Explainability

Three complementary explainability layers are implemented:

**SHAP (TreeExplainer on XGBoost)**
- Global summary bar chart — mean |SHAP| per feature, top 20
- Beeswarm plot — feature impact direction and magnitude across all samples
- Dependence plots — SHAP value vs feature value for the top 6 features
- Per-prediction waterfall charts for a high-AQI (>300) and a low-AQI (<100) example

**Permutation Importance**
- `n_repeats=10` on the test set, scored by R²
- Compares tree-based gain importance vs actual predictive impact; highlights redundant features

**Seasonal Feature Importance**
- Separate XGBoost model trained per season (winter / spring / summer / monsoon / post-monsoon)
- Shows which features dominate under different atmospheric conditions

**Feature Group Contribution Summary**

The notebook prints and visualises how much total importance (%) each feature group (lag, rolling, raw pollutants, meteorological, cyclical time, interactions) contributes — for both RF and XGBoost.

---

## Interactive Forecasting Widget

The notebook includes an `ipywidgets`-based UI (requires JupyterLab or Jupyter Notebook — not rendered on GitHub) for interactive, autoregressive 7-day AQI forecasting.

**Inputs**

| Group | Controls |
|-------|----------|
| Pollutants | PM2.5, PM10, NO₂, SO₂, CO, O₃ |
| Meteorology | Temperature, Humidity, Wind speed, Precipitation, Dew point, Pressure, Visibility |
| Date / Location | Hour, Month, City (Delhi/Noida/Gurugram/Faridabad/Ghaziabad), Day type |

**Autoregressive logic**

Each day's prediction is fed back into the lag history for the next day. The seed is the observed pollutant inputs provided by the user.

```python
# simplified loop
for d in range(7):
    build feature vector from inputs + rolling history
    pred = (rf_model.predict(X) + xgb_model.predict(X)) / 2
    aqi_history = [pred] + aqi_history[:-1]   # shift window
```

**Output** — colour-coded day cards (Good → Severe) plus a tabular RF / XGBoost / ensemble breakdown per day.

---

## Saved Artifacts

The final notebook cell serialises everything needed for standalone inference:

| File | Contents |
|------|----------|
| `rf_aqi_model.pkl` | Fitted `RandomForestRegressor` |
| `xgb_aqi_model.pkl` | Fitted `XGBRegressor` (best iteration preserved) |
| `feature_scaler.pkl` | `StandardScaler` fitted on train set |
| `feature_columns.pkl` | Python list — column order for `X` construction |
| `aqi_label_encoder.pkl` | `LabelEncoder` mapping category strings ↔ integers |
| `model_summary.json` | RMSE/MAE/R²/MAPE per model, dataset shape, city list, date range |

To load and run inference outside the notebook:

```python
import joblib, pandas as pd

rf_model      = joblib.load('rf_aqi_model.pkl')
xgb_model     = joblib.load('xgb_aqi_model.pkl')
feature_cols  = joblib.load('feature_columns.pkl')

X_new = pd.DataFrame([your_feature_dict])[feature_cols]
pred  = (rf_model.predict(X_new) + xgb_model.predict(X_new)) / 2
```

---

## Reproducing Results

- Set `random_state=42` everywhere (already done in all model calls).
- Do **not** shuffle the train/test split — temporal ordering is critical for valid lag features and to prevent leakage.
- Lag feature rows with nulls are dropped before splitting; the exact row count depends on your input CSV. Expect a small variation if the CSV differs.
- `early_stopping_rounds=30` in XGBoost means the best iteration may shift slightly with a different dataset — check `xgb_model.best_iteration` after training.
- For quick iteration, reduce `n_estimators` to 50–100 and re-run; restore for final results.

---

## Known Limitations & Future Work

**Current limitations**

- Lag features assume continuous hourly data; gaps in the CSV will introduce NaNs that are silently filled with medians during inference.
- The autoregressive widget holds pollutant inputs constant across all 7 days (no weather forecast integration).
- No explicit uncertainty quantification — RF OOB score is reported but prediction intervals are not exposed.

**Planned improvements**

- [ ] Replace constant pollutant assumption in the widget with Open-Meteo forecast API integration
- [ ] Add prediction intervals (RF quantile regression or XGBoost `objective='reg:quantileerror'`)
- [ ] Benchmark against LSTM / Temporal Fusion Transformer for sequence-aware forecasting
- [ ] Refactor `predict_aqi_custom` and `quick_predict` into a shared `src/inference.py` module
- [ ] Add a `src/train.py` CLI script (`argparse`) for training outside the notebook
- [ ] Dockerise for portable deployment
- [ ] Add unit tests for preprocessing and feature construction

---

## Contributing

1. Open an issue to discuss the proposed change before starting work.
2. Fork and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Follow PEP 8. If adding notebook cells, label them clearly and ensure they run sequentially from a clean kernel.
4. Submit a pull request with a description of changes and, where applicable, before/after metric comparisons.

---

## License

This project is licensed under the [MIT License](LICENSE).