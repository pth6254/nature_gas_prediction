# Project: Henry Hub Natural Gas Forecast

## Objective

Build a reproducible forecasting pipeline that predicts Henry Hub natural gas prices for 1, 2, and 3 months ahead.

Target column:

`Natural_Gas_US_Henry_Hub_Gas`

Prediction outputs:

- `pred_1`: t+1 month price
- `pred_2`: t+2 month price
- `pred_3`: t+3 month price

The official primary metric is MAPE.

The project should also analyze directional changes and turning points.

---

# Core Modeling Principles

## 1. Prevent time-series leakage

Never use information that would not have been available at the forecast origin.

For a forecast made at time `t`, only information available at or before `t` may be used.

Do not use random train/test split.

Use rolling-origin or expanding-window validation.

Be especially careful with variables whose publication is delayed.

---

## 2. Baseline first

The primary baseline is Last Value Naive:

`pred(t+h) = HenryHub(t)`

for h = 1, 2, 3.

Every ML model must be compared against this baseline under the exact same validation periods.

Also optionally evaluate Seasonal Naive as a secondary baseline.

---

# Forecasting Strategy

Use direct multi-horizon forecasting.

Train separate models for:

- H1: t → t+1
- H2: t → t+2
- H3: t → t+3

Do not use recursive forecasting as the primary approach.

Prefer predicting future log returns rather than raw price:

`target_h = log(price[t+h] / price[t])`

Recover the final price using:

`predicted_price = price[t] * exp(predicted_log_return)`

Compare this approach against direct raw-price prediction during experiments.

---

# Fundamental Market Structure

Do not blindly use all input columns.

The main domain hypothesis is that Henry Hub prices are driven first by natural-gas fundamentals.

Create three interpretable components.

## Production / Supply

Candidate variables:

- `Total_Natural_Gas_Marketed_Production_USA`
- `Natural_Gas_Imports_USA`
- `Natural_Gas_Imports_From_Canada_USA`
- `Natural_Gas_Rotary_Rig_Count_USA`

Rig count should be treated as a potential leading supply indicator rather than immediate production.

Candidate derived features:

- production change
- production YoY
- imports change
- imports YoY
- rig count change
- lagged rig count

---

## Storage

Primary variable:

- `Total_Natural_Gas_Underground_Storage_Volume_USA`

Treat storage as a stock variable, not as the same type of variable as production or consumption.

Candidate derived features:

- storage monthly change
- storage YoY
- storage relative to 12-month rolling average
- storage deviation from historical seasonal level

---

## Consumption / Demand

Primary variable:

- `Natural_Gas_Total_Consumption_USA`

Candidate demand-related variables:

- `high_temperature`
- `low_temperature`

Candidate derived features:

- consumption monthly change
- consumption YoY
- temperature deviation
- temperature-related demand proxy

Do not assume temperature is useful. Validate it empirically.

---

# Fundamental Balance

Do not directly add raw production, storage, and consumption values because their meanings and scales differ.

First create standardized sub-indices.

Conceptually:

`ProductionIndex = weighted standardized supply variables`

`StorageIndex = weighted standardized storage variables`

`ConsumptionIndex = weighted standardized demand variables`

Then define:

`FundamentalBalance = ProductionIndex + StorageIndex - ConsumptionIndex`

For the initial interpretable baseline, use equal weights within each component.

Then compare against weights learned using Ridge/ElasticNet.

Do not choose arbitrary manual weights without validation.

Also create a simpler physical balance proxy:

`SupplyDemandProxy = Production + Imports - Consumption`

Because export data is missing, clearly label this as a proxy rather than a true material balance.

---

# External Variables

External variables must not be mixed directly into the physical supply-demand equation.

Treat them as separate market or external-pressure layers.

## Market signals

Priority candidates:

- `BCOMNG_INDX`
- `WTI_Crude_Oil`
- `Brent_Crude_Oil`
- `BCOMCL_INDX`
- `DXY_INDX`

Prefer returns or changes over raw levels when appropriate.

---

## Global / macro signals

Candidate variables:

- `Kilian_Global_Economy_Index_WORLD`
- `Fed_Fund_Effective_USA`
- selected Composite Leading Indicators

Avoid adding many highly correlated macro variables simultaneously.

---

## Global gas-market variables

Candidate provided variables:

- `EU_Aggregated_Gas_Storage_Inventory`
- `EU_Aggregated_LNG_System_Inventory`

These should be treated as global gas-market proxies.

Do not claim direct causal effects without supporting evidence.

Potential external datasets may later include:

- European TTF natural gas price
- US LNG export volume
- JKM LNG price

Do not add external datasets until the provided dataset baseline has been completed.

---

# Feature Selection

Use the following order:

1. Data availability at forecast time
2. Missing-value structure
3. Domain plausibility
4. Redundancy / multicollinearity
5. Lagged predictive relationship
6. Rolling-origin backtest performance
7. Ablation study
8. Final-model importance / SHAP for interpretation

Do not select features solely from correlation or LightGBM feature importance.

Distinguish:

- level correlation
- change/return correlation
- lagged relationship

because non-stationary levels may create spurious correlations.

---

# Missing Data

Do not automatically apply global mean or median imputation.

First classify missingness where possible:

- structurally unavailable
- late-starting series
- seasonal observation
- publication delay
- recent unpublished observation
- accidental missing value

Use conservative forward-looking-safe imputation only when justified.

Never use future values for interpolation.

---

# Models

Start simple.

Required baseline/candidates:

1. Last Value Naive
2. Ridge
3. ElasticNet
4. LightGBM
5. CatBoost as challenger if useful

Do not introduce deep-learning models unless simpler models clearly fail and there is a defensible reason.

The training sample is small relative to the number of candidate features, so control model complexity.

---

# Validation

Use expanding-window or rolling-origin validation.

Evaluate each horizon independently.

Primary metric:

- MAPE

Secondary diagnostic metrics:

- MAE
- Direction Accuracy

Turning-point performance should also be inspected separately.

Every experiment must use identical validation origins when comparing models.

Save experiment metrics in a machine-readable format.

---

# Experiment Design

Run feature-group ablation.

Suggested sequence:

E0:
Last Value Naive

E1:
Henry Hub history only

E2:
E1 + natural gas fundamentals

E3:
E2 + energy-market variables

E4:
E3 + macro variables

E5:
E4 + weather / EU gas variables

Keep a feature group only if it provides reproducible out-of-sample improvement or meaningful turning-point information.

Do not keep features simply because training performance improves.

---

# Interpretation

For final models, analyze:

- SHAP values
- successful turning-point predictions
- failed turning-point predictions
- relationship between model signals and domain mechanisms

Do not describe SHAP as proof of causality.

Use language such as:

"the variable contributed to the model's forecast"

rather than:

"the variable caused the price movement"

For causal/leading relationship analysis, optionally use:

- lagged correlation
- distributed-lag regression
- Granger causality
- small VAR / impulse-response analysis

Treat these as evidence of temporal predictive relationships unless stronger causal identification exists.

---

# Code Structure

Keep the project simple and reproducible.

Recommended structure:

`notebooks/01_eda.ipynb`
- exploratory analysis only

`src/features.py`
- feature engineering

`src/validation.py`
- rolling-origin splits and horizon dataset creation

`src/train.py`
- model training, comparison and submission generation

`src/evaluate.py`
- metrics and turning-point analysis

Do not over-engineer the project with unnecessary classes, factories, services, or config frameworks.

Prefer clear functions and readable Python.

---

# Required Outputs

Generate:

- `submission.csv`
- experiment metrics CSV
- figures used in the report
- reproducible training code
- README instructions

The final submission CSV must preserve the provided submission format:

`date,pred_1,pred_2,pred_3`

---

# Working Style

Before implementing a major modeling decision:

1. inspect the relevant data,
2. state the hypothesis,
3. implement the experiment,
4. report the validation result,
5. explain the next decision.

Do not silently make major modeling assumptions.

When an assumption affects leakage, feature availability, or evaluation fairness, explicitly document it.