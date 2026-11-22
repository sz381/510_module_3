# Seattle Airbnb Price Prediction

AI PI 510 — Module 3, Project 3

## What this is

We predict the nightly price of an Airbnb listing in Seattle.

Input is a property description: room type, neighbourhood, how many people it
sleeps, bedrooms, bathrooms, minimum nights, cleaning fee, and whether the host
is a superhost. Output is a nightly price in US dollars, with a range around it.

The point of the project is not model accuracy. The point is the full path:
cloud data, a reproducible pipeline driven by `config.yaml`, experiment
tracking, a containerised API, and a frontend that calls it.

## Data

Three CSVs from Inside Airbnb, Seattle snapshot of 2016-01-04. Also published
on Kaggle as `airbnb/seattle` under CC0.

| File | Rows | What one row is |
| --- | --- | --- |
| `listings.csv` | 3,818 | One listing: the property, the host, the price |
| `calendar.csv` | 1,393,570 | One listing on one day: bookable or not, and the quoted price |
| `reviews.csv` | 84,849 | One guest review: who, when, what they wrote |

`listings` is the main table. The other two are one-to-many and get aggregated
back into it, so the final dataset stays at 3,818 rows.

## What we have done

### EDA

`notebooks/eda.ipynb`. Findings that changed the plan:

- Price is right-skewed (median $100, max $1,000), so the model trains on `log1p(price)`
- `room_type` is the strongest single feature: median $126 entire home, $69 private room, $40 shared room
- `calendar.price` is only quoted for bookable nights, never impute it for unavailable dates
- `weekly_price` and `monthly_price` are derived from `price` and would leak the target
- `review_scores_rating` is 0-100 while the other six scores are 0-10

### Cleaning

- `$1,200.00` strings parsed to floats, `96%` parsed to 0.96, `t`/`f` parsed to 1/0
- `amenities` is a brace format, not JSON: `{TV,"Cable TV",Internet}` parsed by hand
- `property_type` grouped from 16 values to 7, rare types folded into `Other`
- `minimum_nights` capped at 30 (the raw column reaches 1,000)
- Dropped all-null and zero-variance columns, plus every id, url and free-text column
- Three assertions run on every execution: row count stays at 3,818, listing id stays unique, and calendar-derived availability matches `availability_30`

### Features

107 numeric features, built in `src/preprocess.py`:

- Property: accommodates, bedrooms, bathrooms, beds, cleaning fee, deposit, minimum nights
- Host: response rate, superhost flag, tenure in days, listing count
- Reviews: seven score columns, review count, days since last review, mean comment length
- Calendar aggregates: occupancy over 30 / 90 / 365 days, weekend-to-weekday price ratio
- Location: distance to downtown Seattle, 17 neighbourhood groups
- Amenities: count plus the 20 most common ones as 0/1 columns

### Models

Both are sklearn `Pipeline` objects with the same target, `log1p(price)`.
Metrics are converted back to dollars before scoring.

| Model | Validation MAE | Test MAE | Test R² |
| --- | --- | --- | --- |
| HistGradientBoosting | 26.78 | 27.24 | 0.668 |
| Ridge (imputer + scaler) | 30.00 | 28.75 | 0.633 |

A 24-combination grid search (`script/tune.py`) put every run between 26.5 and
27.6 validation MAE. The best combination on validation scored 27.81 on test,
so the differences are noise. Hyperparameters are not the bottleneck here.

Every run is logged to MLflow, including the grid search.

## How to run

```bash
pip install -r requirements.txt

python src/preprocess.py     # raw CSVs -> data/processed splits
python src/train.py          # trains both models, logs to MLflow
python src/evaluate.py       # re-scores the saved model, writes reports/
python script/tune.py        # optional grid search

mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Repo layout

```
config.yaml          every path and parameter the pipeline uses
src/preprocess.py    read, clean, merge, engineer features, split
src/train.py         train, score, log to MLflow, save the model
src/evaluate.py      reload the model, rebuild the metrics and plots
script/tune.py       grid search over the boosting parameters
notebooks/eda.ipynb  the exploration behind every decision above
data/                raw, interim and processed (not committed)
models/              trained model and model_card.json
interface/           Next.js frontend
api/                 FastAPI backend (empty for now)
```

## Status

Done: data download, EDA, cleaning, feature engineering, training,
evaluation, MLflow tracking, `config.yaml`.

Not done yet:

- `api/` — FastAPI service with `/predict`, `/metadata` and `/health`
- `Dockerfile` — one image for both training and serving
- Deployment of the API to a cloud platform
- The frontend itself — Next.js scaffold and UI library are installed, no pages yet
- Hosting the data in cloud storage and reading it from there

## Next

1. FastAPI backend
2. Docker image with a `train` and a `serve` entry point
3. Deploy the API and confirm `/predict` responds publicly
4. Build the Next.js page and point it at the deployed API
5. Move the raw data to cloud storage and have the pipeline pull from there
