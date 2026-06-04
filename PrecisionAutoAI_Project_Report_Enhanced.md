# Turkish Car Price Prediction Using Machine Learning
**Enhanced Project Report** | Muhammed Yusa Duymaz

---

## Abstract

Predicting used-car prices is a regression problem, but predicting them for the Turkish market requires more than a generic machine learning workflow. Turkey's automotive pricing is strongly shaped by currency, tax burden, engine displacement and local buyer expectations. A model trained directly on UK prices would not be useful unless the data were localized into a Turkey-oriented target.

This project builds an end-to-end pipeline for used-car price prediction. It starts from the Kaggle UK used-car dataset, merges manufacturer CSV files, cleans noisy records, localizes price and mileage values, trains multiple regression models, tunes XGBoost, saves the final model artifacts, and exposes the prediction workflow through a Flask web interface named **PrecisionAutoAI**.

The final tuned XGBoost model achieves **R2 = 0.9682** on the held-out test set, with **MAE = 265,370 TRY** and **RMSE = 417,255 TRY**. The project therefore demonstrates both a high-performing ML model and a usable deployment layer.

---

## 1. Project Objective

The goal of this project is to estimate the Turkish Lira value of a used car from practical vehicle attributes:

- Brand
- Model
- Year
- Mileage in kilometers
- Transmission type
- Fuel type
- Engine size

The project was not limited to notebook experimentation. After model training, the preprocessing pipeline and trained model were exported as artifacts and integrated into a Flask UI so that users can enter car details and receive a TRY price estimate.

---

## 2. Dataset

**Dataset source:** Kaggle - Used Car Dataset: Ford and Mercedes  
**URL:** https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes  
**Raw size after merging:** 108,540 rows  
**Final size after cleaning:** 99,322 rows

The dataset consists of manufacturer-level CSV files including Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall/Opel and Volkswagen.

| Feature | Type | Description |
|---|---:|---|
| `brand` | Categorical | Added during CSV merge; manufacturer name |
| `model` | Categorical | Vehicle model |
| `year` | Numerical | Registration/manufacture year |
| `transmission` | Categorical | Manual, Automatic, Semi-Auto, Other |
| `mileage` / `km` | Numerical | Miles in raw data, converted to kilometers |
| `fuelType` | Categorical | Petrol, Diesel, Hybrid, Electric, Other |
| `engineSize` | Numerical | Engine displacement in litres |
| `price` / `price_tl` | Numerical | GBP in raw data, converted to localized TRY target |

![Dataset loading and CSV structure](<image/Ekran görüntüsü 2026-06-03 190231.png>)

---

## 3. Data Preparation and Cleaning

The raw dataset required several transformations before it could be used reliably for model training.

### 3.1 CSV Merge

Eleven CSV files were read and appended into a single dataframe. Because the original files were split by manufacturer, a `brand` column was manually added before concatenation.

### 3.2 Column Standardization

The Hyundai file used `tax(£)` while other files used `tax`. This inconsistency was normalized. The project later dropped `tax` and `mpg` because the final Turkish-market price target was built with a custom localization layer.

### 3.3 Invalid Value Removal

Rows with `engineSize = 0` were removed because a zero-litre engine is not meaningful for the target use case.

The year range was also restricted to realistic values:

```text
2000 <= year < 2026
```

### 3.4 High-Cardinality Model Handling

The raw dataset contained **193 unique models**. Rare models with 100 or fewer records were grouped into `Other`, reducing the model cardinality to **113**. This improved generalization and also made the UI fallback behavior safer.

### 3.5 IQR Outlier Removal

The IQR method was applied to:

- `price`
- `mileage`
- `year`
- `engineSize`

This removed **8,907 rows**, leaving **99,322 rows** for modeling.

![Outlier analysis before cleaning](<image/Ekran görüntüsü 2026-06-03 190443.png>)

---

## 4. Exploratory Data Analysis

EDA was used to understand feature behavior before choosing the model family.

### Key Observations

- Newer vehicles generally have higher prices.
- Engine size is a strong signal because it affects both market value and tax burden.
- Transmission type changes the price distribution; automatic and semi-automatic vehicles often command a premium.
- Mileage has the expected depreciation effect, but after Turkish localization it becomes less dominant than engine size and vehicle age.
- The target is nonlinear, making ensemble tree models more suitable than pure linear models.

![Relationship between vehicle year and price](<image/Ekran görüntüsü 2026-06-03 190348.png>)

![Price distribution by transmission type](<image/Ekran görüntüsü 2026-06-03 190359.png>)

![Correlation heatmap](<image/Ekran görüntüsü 2026-06-03 190427.png>)

---

## 5. Turkish Market Localization

The original dataset contains UK prices in GBP and mileage in miles. For the target application, both had to be transformed.

### 5.1 Currency Conversion

```text
base_tl = gbp_price * 58
```

The notebook used a fixed exchange rate of **1 GBP = 58 TRY**.

### 5.2 Mileage Conversion

```text
km = mileage * 1.609
```

This matches Turkish listing conventions, where mileage is shown in kilometers.

### 5.3 Engine-Size Tax Multiplier

The project applied a simplified Turkish-market tax multiplier:

| Engine size | Multiplier |
|---|---:|
| `<= 1.6L` | `3.0x` |
| `<= 2.0L` | `4.5x` |
| `> 2.0L` | `6.0x` |

This step transformed the raw target from `price` to `price_tl`.

```python
def calculate_tr_price(row):
    gbp_price = row["price"]
    engine = row["engineSize"]

    base_tl = gbp_price * exchange_rate

    if engine <= 1.6:
        tax_multiplier = 3.0
    elif engine <= 2.0:
        tax_multiplier = 4.5
    else:
        tax_multiplier = 6.0

    return base_tl * tax_multiplier
```

This is the most domain-specific feature engineering step in the project. Without it, the model would learn UK-market pricing instead of the intended Turkey-oriented pricing behavior.

---

## 6. Feature Engineering and Preprocessing Pipeline

The dataset was split before encoding and scaling to avoid data leakage.

```text
Train: 80%
Test: 20%
random_state = 42
```

### Encoding Strategy

| Column | Method | Reason |
|---|---|---|
| `brand` | LabelEncoder | Small categorical set, useful for tree models |
| `model` | LabelEncoder | High cardinality; avoids hundreds of sparse one-hot columns |
| `transmission` | BinaryEncoder | Non-ordinal category with few values |
| `fuelType` | BinaryEncoder | Compact representation without false ordering |

### Scaling Strategy

`StandardScaler` was applied only to:

- `km`
- `engineSize`
- `year`

Scaling matters for KNN and linear models because these algorithms are sensitive to feature magnitude. The scaler was fitted only on the training set and reused on test data and deployment inputs.

![Encoding and scaling workflow](<image/Ekran görüntüsü 2026-06-03 190527.png>)

---

## 7. Model Training and Comparison

The project compared several regression models:

- Linear Regression
- Lasso
- Ridge
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

### Evaluation Metrics

| Metric | Meaning |
|---|---|
| RMSE | Penalizes large prediction errors |
| MAE | Average absolute error, easy to interpret in TRY |
| R2 | Percentage of variance explained by the model |

### Model Results

| Model | Train R2 | Test R2 | Test RMSE (TRY) | Test MAE (TRY) |
|---|---:|---:|---:|---:|
| Linear Regression | 0.8036 | 0.7998 | 1,046,985 | 816,700 |
| KNN | 0.9771 | 0.9632 | 449,198 | 282,683 |
| Decision Tree | 0.9991 | 0.9487 | 530,187 | 306,079 |
| Random Forest | 0.9947 | 0.9657 | 433,243 | 269,697 |
| AdaBoost | 0.8393 | 0.8351 | 950,170 | 738,616 |
| Gradient Boosting | 0.9385 | 0.9362 | 591,074 | 411,198 |
| XGBoost | 0.9731 | 0.9662 | 430,215 | 280,101 |

The linear models were useful baselines but did not capture the nonlinear effects in vehicle pricing. KNN, Random Forest and XGBoost performed strongly. XGBoost was selected for tuning because it combines strong nonlinear performance with regularization and flexible hyperparameters.

![Model training and comparison notebook section](<image/Ekran görüntüsü 2026-06-03 190541.png>)

![Model evaluation output](<image/Ekran görüntüsü 2026-06-03 190633.png>)

---

## 8. Hyperparameter Tuning

`RandomizedSearchCV` was used instead of a full grid search because the search space was large and the dataset had nearly 100k rows after cleaning.

### Search Space

```text
learning_rate:    [0.1, 0.01]
max_depth:        [5, 8, 12, 20, 30]
n_estimators:     [100, 200, 300, 500]
colsample_bytree: [0.3, 0.4, 0.5, 0.7, 1]
```

### Best Parameters

```text
n_estimators:     100
max_depth:        12
learning_rate:    0.1
colsample_bytree: 0.7
```

The final model was trained as:

```python
XGBRegressor(
    colsample_bytree=0.7,
    learning_rate=0.1,
    max_depth=12,
    n_estimators=200
)
```

---

## 9. Final Model Performance

| Metric | Training Set | Test Set |
|---|---:|---:|
| RMSE | 278,707 TRY | 417,255 TRY |
| MAE | 182,636 TRY | 265,370 TRY |
| R2 | 0.9860 | **0.9682** |

### Interpretation

The model explains **96.82%** of the variance in unseen data. The average absolute error is approximately **265k TRY**, which is acceptable for a dataset whose localized prices can span millions of TRY after exchange-rate and tax transformations.

The train/test gap is present but not severe, suggesting the tuned model generalizes well enough for the project scope.

---

## 10. Feature Importance and Market Interpretation

The feature importance chart shows that the strongest signals are aligned with real car pricing logic:

- `engineSize` is dominant because it affects the localization multiplier.
- Transmission-related encoded columns carry significant signal.
- `year` reflects depreciation and vehicle recency.
- `brand` and `model` encode segment and market-position differences.
- `km` matters, but in this localized target it is less dominant than engine size.

![Top 10 feature importance values](<image/Ekran görüntüsü 2026-06-03 190639.png>)

This validates the main hypothesis of the project: a Turkey-oriented car price model must account for engine displacement and tax-like price jumps.

---

## 11. Model Export and Artifact Design

The project did not save only the trained model. It saved the full set of objects required to reproduce preprocessing during deployment:

```python
artifacts = {
    "model": model,
    "scaler": scaler,
    "label_encoders": encoders,
    "binary_encoders": binary_encoders,
    "feature_columns": X_train_scaled.columns.tolist(),
    "columns_to_scale": cols_to_scale,
}
```

Two output files were produced:

- `car_price_artifacts.pkl`
- `car_price_model.pkl`

The artifact bundle is important because the Flask app must apply the exact same transformations used during training. Without the saved encoders, scaler and feature order, the deployed prediction results would not match notebook evaluation behavior.

---

## 12. Flask UI Development

The original notebook workflow was extended into a working web application using Flask. This is a major project step because it turns the trained ML model into a user-facing product prototype.

![PrecisionAutoAI Flask UI](<image/flask-ui-price-predictor.png>)

### 12.1 Application Structure

| File | Role |
|---|---|
| `app.py` | Flask backend, artifact loading, input validation, prediction route |
| `templates/index.html` | User interface, form layout, responsive CSS and client-side model filtering |
| `car_price_artifacts.pkl` | Model plus preprocessing bundle |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container deployment definition |
| `Procfile` | Gunicorn command for platform deployment |

### 12.2 Backend Flow

When the Flask app starts, it loads:

- Trained XGBoost model
- StandardScaler
- Label encoders
- Binary encoders
- Feature column order
- Columns that must be scaled

The backend then constructs a `brand_model_map` from the CSV files. This lets the interface show only relevant models after a user selects a brand.

### 12.3 Prediction Flow

When the user submits the form:

1. The backend reads form inputs.
2. Brand and model are validated against the trained encoders.
3. Unsupported model values safely fall back to `Other`.
4. A one-row pandas DataFrame is created.
5. The same LabelEncoder and BinaryEncoder objects are applied.
6. Numerical columns are scaled with the saved StandardScaler.
7. Missing feature columns are added as zeros if needed.
8. The final row is reordered to match the training feature order.
9. The XGBoost model predicts the TRY price.
10. The UI displays the rounded prediction.

This flow prevents a common deployment bug: training with one feature order but predicting with a different feature order.

### 12.4 UI Design Decisions

The interface was designed around practical user input:

- Brand-first selection
- Dynamic model list filtered by brand
- Required input validation
- Mileage entered in kilometers
- Transmission and fuel type dropdowns
- Clear result card showing estimated TRY price
- Responsive layout for smaller screens

The UI also communicates the actual deployment stack through its metrics: **Brand-aware**, **TRY estimate**, and **Flask + Docker**.

### 12.5 Deployment Readiness

The project includes both Docker and Gunicorn configuration.

`Dockerfile`:

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

`Procfile`:

```text
web: gunicorn --bind 0.0.0.0:$PORT app:app
```

This means the project is not only a notebook experiment; it is prepared for containerized or platform-based deployment.

---

## 13. Updated Tech Stack

| Layer | Tool / Library |
|---|---|
| Language | Python |
| Notebook environment | Google Colab / Jupyter Notebook |
| Data processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML models | scikit-learn, XGBoost |
| Categorical encoding | category_encoders |
| Hyperparameter tuning | RandomizedSearchCV |
| Model persistence | joblib |
| Backend | Flask |
| Frontend | HTML, CSS, Jinja templates |
| Production server | Gunicorn |
| Deployment packaging | Dockerfile, Procfile |

---

## 14. Limitations

The project is strong as an academic ML + deployment prototype, but it still has realistic limitations:

- The dataset is UK-sourced, not scraped from Turkish listing platforms.
- Exchange rate is fixed at the notebook value and not updated dynamically.
- The tax multiplier is simplified and does not fully reproduce all Turkish tax rules.
- Rare models are grouped as `Other`, so niche vehicles may be less accurate.
- The model does not yet return uncertainty intervals.
- UI currently supports single prediction rather than batch upload.

---

## 15. Future Improvements

High-value next steps:

- Use real Turkish listing data from platforms such as sahibinden.com or arabam.com.
- Add dynamic exchange-rate updates.
- Add a more detailed tax calculation layer using actual ÖTV/KDV brackets.
- Add SHAP explanations to show why a price was predicted.
- Add confidence intervals or prediction ranges.
- Add batch prediction via CSV upload.
- Add model monitoring for drift as market prices change.
- Add authentication and a database if the app becomes a real product.

---

## 16. Conclusion

PrecisionAutoAI is an end-to-end machine learning project, not just a notebook model. It covers:

1. Dataset acquisition and merging
2. Data cleaning and outlier removal
3. Turkish-market localization
4. Categorical encoding and scaling
5. Multi-model benchmarking
6. XGBoost tuning
7. Final model evaluation
8. Feature importance interpretation
9. Artifact export
10. Flask UI development
11. Docker/Gunicorn deployment preparation

The final result is a high-performing used-car price prediction system with a working web interface and a clear path toward production deployment.

---

*Enhanced report prepared from the Colab notebook, local Flask application and project screenshots.*  
*Muhammed Yusa Duymaz - PrecisionAutoAI*
