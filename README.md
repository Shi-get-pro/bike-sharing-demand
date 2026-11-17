# 🚲 Bike Sharing Demand Prediction

Predicting hourly bike rental demand from time and weather data, using linear models with polynomial features and regularization (Ridge and Lasso).

Built as part of my M.Tech in Artificial Intelligence & Machine Learning.

## Problem

Bike sharing systems need to know how many bikes will be rented each hour so they can place bikes and plan staffing. The goal is to predict the hourly rental `count` from the date, time, season and weather.

## Dataset

| Item | Detail |
|---|---|
| Source | Bike Sharing Demand dataset (Kaggle / UCI) |
| Size | 10,450 rows × 12 columns, no missing values |
| Target | `count`: total bikes rented in the hour |
| Categorical | `season` (1–4), `holiday` (0/1), `workingday` (0/1), `weather` (1–4) |
| Continuous | `temp`, `atemp`, `humidity`, `windspeed` |
| Dropped | `casual`, `registered`: they add up to `count`, so using them would leak the answer |

## What the data shows

- **Hour of day is the strongest signal.** On working days there are two commute peaks, at 8 AM and 5–6 PM. Non-working days have one broad midday peak instead.
- **Temperature** raises demand up to about 30 °C, then demand dips slightly in extreme heat.
- **Season and month:** Fall and June–September are busiest. Spring and winter months are quietest.
- **Weather:** clear days bring far more riders than rainy or stormy ones.
- **Year:** ridership grew a lot from 2011 to 2012.

## Feature engineering

- Pulled `hour`, `weekday`, `month` and `year` out of `datetime`
- `is_peak`: a flag for rush hours (7–9 AM, 5–7 PM) on working days
- `temp_diff = atemp − temp`: captures the wind-chill and heat-index effect
- Log transform of `windspeed`, which is right-skewed
- Dropped the leaky columns `casual` and `registered`

## Models and results

80/20 train/validation split (8,360 / 2,090 rows), `random_state=42`. Metric: **RMSLE**.

| Model | Best setting | Validation RMSLE |
|---|---|---|
| Linear Regression + degree-2 polynomial features (135 features) | none | 1.2293 |
| Ridge Regression + degree-2 polynomial features | α = 500 | 1.2038 |
| **Lasso Regression + degree-2 polynomial features** | **α = 10** | **1.1119** |

### Key takeaways

1. **A straight line can't model the commute peaks.** Linear regression gives `hour` one coefficient, so it can only say "later means more" or "later means fewer." It can't represent high demand at 8 AM, low at noon and high again at 6 PM.
2. **Polynomial features add the curves,** and **regularization keeps them in check.** With 135 features, Ridge and Lasso both beat plain polynomial regression. Lasso did best here because it sets unhelpful terms to zero.
3. **RMSLE fits count data better than RMSE.** It measures *relative* error, so being 2× off counts the same at 10 bikes as at 1,000.
4. **The residuals fan out at high predictions.** The model under-predicts at peak hours, which is a known limit of linear models on count data.

## Next steps

- Train on `log1p(count)` so the model optimizes directly for RMSLE
- One-hot encode `hour`, or use cyclic sin/cos features
- Try tree-based models (Random Forest, Gradient Boosting) that can split the hour axis anywhere

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter / Google Colab

## How to run

```bash
pip install -r requirements.txt
jupyter notebook bike_sharing.ipynb
```

Or open the notebook in Google Colab.

*Personal academic project using a public dataset.*
