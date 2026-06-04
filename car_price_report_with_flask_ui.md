# Turkish Car Price Prediction Using Machine Learning
**Project Report** | Muhammed Yuşa Duymaz

---

## Abstract

Predicting used car prices is a classic regression problem — but doing it accurately for the **Turkish market** requires more than a generic ML pipeline. Turkey's automotive pricing is heavily distorted by the ÖTV (Özel Tüketim Vergisi / Special Consumption Tax), a progressive tax tied directly to engine displacement. A 1.6L car and a 2.0L car of the same brand and year can differ by hundreds of thousands of TRY purely due to tax brackets — a nuance invisible to models trained on Western datasets.

This project builds an end-to-end ML pipeline that starts from a UK used car dataset, transforms it into Turkish market pricing through a domain-informed localization layer, and then trains and compares eight regression models. The final tuned XGBoost model achieves **R² = 0.9682** on the held-out test set, demonstrating that with proper feature engineering and market-specific preprocessing, highly accurate price estimation is achievable.

---

## 1. Problem Definition & Motivation

### Why this problem?
The Turkish second-hand car market is one of the most active in Europe, yet pricing is notoriously opaque. Buyers often overpay or sellers underprice because there is no reliable, data-driven reference. A machine learning model that can estimate fair market value — given vehicle specs — has real-world utility for:
- Individual buyers/sellers comparing prices
- Dealerships automating valuation
- Insurance companies estimating vehicle worth

### Why is Turkey different?
In most markets, car price correlates primarily with age and mileage. In Turkey, **engine size is the dominant factor** because of the ÖTV system:

| Engine Displacement | ÖTV Rate |
|---|---|
| ≤ 1.6L | ~45% (lower bracket) |
| 1.6L – 2.0L | ~80% (middle bracket) |
| > 2.0L | ~130%+ (upper bracket) |

This creates sharp pricing discontinuities at the 1.6L and 2.0L thresholds that a model must learn. Using raw foreign data without accounting for this would produce fundamentally wrong predictions.

---

## 2. Dataset

**Source:** UK used car listings (covers Audi, BMW, Ford, Toyota, Volkswagen, and others)  
**Raw size:** ~108,000+ rows  
**Final size after cleaning:** 99,322 rows

**Key features available:**
| Feature | Type | Description |
|---|---|---|
| `brand` | Categorical | Car manufacturer |
| `model` | Categorical | Specific model name |
| `year` | Numerical | Year of manufacture |
| `transmission` | Categorical | Manual / Automatic / Semi-Auto |
| `mileage` | Numerical | Distance driven (miles) |
| `fuelType` | Categorical | Petrol / Diesel / Hybrid / Electric |
| `engineSize` | Numerical | Engine displacement in litres |
| `price` | Numerical | Listed price in GBP (target) |

The dataset is UK-sourced, meaning prices are in GBP and distances are in miles. Both must be converted before any Turkish market modelling is meaningful.

---

## 3. Exploratory Data Analysis (EDA)

Before any modelling, understanding the data distribution is critical. Key observations that informed later decisions:

- **Price distribution** was heavily right-skewed — a small number of luxury/exotic vehicles had extremely high prices. This justified the outlier removal step.
- **Engine size** showed clear clustering around 1.0L, 1.6L, 2.0L — aligning with common market segments and validating the tax threshold design.
- **Year** correlated strongly with price, confirming vehicle age as a major value driver.
- **Mileage** showed expected negative correlation with price (higher km = lower value).
- **Categorical imbalance**: some transmission and fuel types were rare (e.g. electric), which influenced encoding choice.

---

## 4. Preprocessing Pipeline

### Why a pipeline approach?
Each preprocessing step was designed to solve a specific problem observed in the raw data. Skipping any step would degrade model performance in a predictable way.

### 4.1 Outlier Removal — IQR Method

**Why:** Extreme values in price, mileage, year, and engineSize would distort the model's learned relationships. A £200,000 Lamborghini in the training data teaches the model very little about the average £15,000 used Ford Focus — but it pulls regression coefficients significantly.

**How:** The Interquartile Range (IQR) method was applied to `price`, `mileage`, `year`, and `engineSize`. Any row where a value fell below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR` was dropped.

**Result:** 8,907 rows removed → **99,322 rows remaining**

This is a conservative clean — only genuine statistical outliers were removed, not "expensive" cars generally.

---

## 5. Localization — Making Data Turkish

This is the most domain-specific and arguably most important step in the pipeline.

### Why localization matters
Training a model on GBP prices and then asking it to predict TRY prices would be meaningless — not just because of currency, but because the **tax structure fundamentally changes relative pricing**. A direct GBP × exchange_rate conversion ignores ÖTV entirely, making predictions structurally wrong.

### 5.1 Currency Conversion
```
1 GBP = 58 TRY  (rate used at time of study)
```
Base TRY price = GBP price × 58

### 5.2 Distance Conversion
```
1 mile = 1.609 km
```
All mileage values converted to kilometres, as Turkish buyers and listings universally use km.

### 5.3 Engine-Displacement Tax Multiplier
The core localization logic. A custom function `calculate_tr_price()` applies a multiplier on top of the base TRY price:

| Engine Size | Multiplier | Rationale |
|---|---|---|
| ≤ 1.6L | 3.0× | Standard ÖTV bracket |
| 1.6L – 2.0L | 4.5× | Higher ÖTV bracket |
| > 2.0L | 6.0× | Premium ÖTV bracket |

This transforms the base converted price into a realistic Turkish market price, reflecting the tax burden that any buyer would actually face.

**Column renames:** `mileage → km`, `price → price_tl`

---

## 6. Data Splitting & Categorical Encoding

### 6.1 Train/Test Split
**Why first:** Encoding and scaling must be fit only on training data. Splitting before encoding prevents **data leakage** — where information from the test set inadvertently influences the model during training, leading to overly optimistic evaluation metrics.

- Train: 80% | Test: 20% | `random_state=42` for reproducibility

### 6.2 Label Encoding — `brand` & `model`
**Why Label Encoding here:** Brand and model have a natural ordinal-like structure in terms of market positioning (though not strictly ordinal). More importantly, tree-based models (which dominate this experiment) handle label-encoded categoricals well, and one-hot encoding `model` would produce hundreds of sparse binary columns, hurting performance.

### 6.3 Binary Encoding — `transmission` & `fuelType`
**Why Binary Encoding:** These columns have 3–4 unique values each. Binary encoding produces fewer columns than one-hot encoding while avoiding the false ordinality of label encoding. For `transmission` (Manual/Automatic/Semi-Auto), there is no meaningful order, so label encoding would be misleading.

---

## 7. Feature Scaling — StandardScaler

**Why scaling is necessary:** Algorithms like K-Nearest Neighbours and Linear Regression are distance-sensitive. Without scaling, `km` (values in tens of thousands) would dominate over `engineSize` (values like 1.0–3.0) purely due to magnitude, not actual predictive power.

**Why only these three columns:** `km`, `engineSize`, and `year` are the only continuous numerical features that weren't already encoded. Encoded binary/label columns don't require scaling.

**Why StandardScaler over MinMaxScaler:** StandardScaler (z-score normalization) is more robust to outliers that survived the IQR step. MinMaxScaler would compress those residual outliers into the 0–1 range but could distort the bulk of the distribution.

**Critical:** `scaler.fit_transform()` on train, `scaler.transform()` on test — never fit on test data.

---

## 8. Model Training & Comparison

### Why compare 8 models?
No single algorithm is universally best. The goal was to evaluate a spectrum:
- **Linear models** (Linear Regression, Lasso, Ridge): establish a baseline; check if the problem is linearly separable
- **Instance-based** (KNN): captures local pricing patterns
- **Tree-based** (Decision Tree, Random Forest): handle non-linear relationships and interactions
- **Ensemble/Boosting** (AdaBoost, Gradient Boosting, XGBoost): progressively more powerful, better at capturing complex patterns like the ÖTV tax discontinuities

### Evaluation Metrics
| Metric | What it measures |
|---|---|
| **RMSE** | Penalises large errors heavily — important for pricing (big misses are costly) |
| **MAE** | Average absolute error — more interpretable in TRY terms |
| **R²** | Proportion of variance explained — overall fit quality |

### Results

| Model | Train R² | Test R² | Test RMSE (TRY) | Test MAE (TRY) |
|---|---|---|---|---|
| Random Forest | 0.9947 | 0.9653 | 433,243 | 269,697 |
| AdaBoost | 0.8393 | 0.8352 | 950,170 | 738,616 |
| Gradient Boosting | 0.9385 | 0.9362 | 591,074 | 411,198 |
| XGBoost | 0.9730 | 0.9662 | 430,215 | 280,101 |

**Key observations:**
- AdaBoost underperformed significantly — it is better suited to classification and struggles with high-variance regression targets like car prices.
- Random Forest and XGBoost both achieved ~96.5% R² on test data, with XGBoost having a slightly lower RMSE.
- The gap between Random Forest's train R² (0.9947) and test R² (0.9653) suggests mild overfitting — XGBoost showed a healthier gap (0.9730 vs 0.9662).
- XGBoost was selected for tuning: faster inference, built-in regularisation, and more tunable hyperparameters than Random Forest.

---

## 9. Hyperparameter Tuning — XGBoost

### Why tune?
Default XGBoost settings are a reasonable starting point but not optimised for this specific dataset. Tuning can reduce overfitting (via regularisation params) and improve generalisation.

### Why RandomizedSearchCV over GridSearchCV?
With 5 hyperparameter dimensions and multiple values each, a full grid search would require training hundreds of models. RandomizedSearchCV samples a random subset of combinations — statistically, this finds near-optimal configurations in a fraction of the time, making it the practical choice for large datasets.

**Search space explored:**
```
learning_rate:    [0.1, 0.01]       — controls step size per tree
max_depth:        [5, 8, 12, 20, 30] — controls tree complexity
n_estimators:     [100, 200, 300, 500] — number of trees
colsample_bytree: [0.3, 0.4, 0.5, 0.7, 1] — feature subsampling per tree
```

**Best parameters found:**
```
n_estimators:     100
max_depth:        12
learning_rate:    0.1
colsample_bytree: 0.7
```

A `max_depth` of 12 is relatively deep, suggesting the model benefits from capturing complex multi-way interactions between features (e.g. brand × engineSize × year combinations).

---

## 10. Final Model Performance

**Final model:** `XGBRegressor(colsample_bytree=0.7, learning_rate=0.1, max_depth=12, n_estimators=200)`

*(n_estimators increased to 200 from the CV best of 100 for additional stability)*

| Metric | Training Set | Test Set |
|---|---|---|
| RMSE | 278,707 TRY | 417,255 TRY |
| MAE | 182,635 TRY | 265,370 TRY |
| R² | 0.9860 | **0.9682** |

**Interpretation:**
- The model explains **96.82%** of price variance in unseen data — a strong result for a noisy real-world pricing dataset.
- The average prediction error (MAE) is ~265,370 TRY. Given that the dataset spans vehicles from ~1,000,000 TRY to 10,000,000+ TRY after localization, this represents a reasonably tight margin.
- The train/test R² gap (0.9860 vs 0.9682) is acceptable — no severe overfitting.

---

## 11. Feature Importance & Market Interpretation

### Top 10 features (XGBoost importance scores):

| Rank | Feature | Importance Score | Market Interpretation |
|---|---|---|---|
| 1 | `engineSize` | ~0.47 | Dominant — directly drives ÖTV tax brackets |
| 2 | `transmission_2` | ~0.30 | Automatic/semi-auto commands premium in Turkish market |
| 3 | `fuelType_2` | ~0.09 | Hybrid/alternative fuel premium |
| 4 | `year` | ~0.06 | Depreciation — newer = more expensive |
| 5 | `brand` | ~0.05 | Brand premium (Audi vs Toyota etc.) |
| 6 | `model` | ~0.04 | Model-specific pricing |
| 7 | `fuelType_1` | ~0.02 | Diesel vs petrol differential |
| 8 | `transmission_1` | ~0.02 | Manual discount |
| 9 | `km` | ~0.02 | Mileage depreciation |
| 10 | `fuelType_0` | ~0.01 | Petrol baseline |

### Key insight
The fact that `engineSize` alone accounts for ~47% of the model's decision-making confirms the initial hypothesis: **Turkey's ÖTV structure is the single largest pricing driver in the used car market.** This also validates the localization step — without the tax multiplier, engineSize would likely have a much weaker signal.

The relatively low importance of `km` (mileage) is also notable — in Western markets, mileage is typically a top-3 predictor. In Turkey, the tax burden so significantly amplifies engine-size-based pricing that mileage becomes secondary.

---

## 12. Flask UI & Deployment

After the notebook phase, the trained model was converted into a small web application using **Flask**. This step is important because it turns the ML model from an offline experiment into a usable prediction tool.

![PrecisionAutoAI Flask UI](<image/flask-ui-price-predictor.png>)

### UI purpose
The interface allows a user to enter vehicle details and receive an estimated Turkish Lira price. The form includes:

- Brand
- Model
- Year
- Mileage in kilometers
- Transmission type
- Fuel type
- Engine size

The UI follows a brand-first flow: once the user selects a brand, the model dropdown is filtered to show only relevant models. This makes the form easier to use and reduces invalid input.

### Backend prediction flow
The Flask backend loads the saved `car_price_artifacts.pkl` file, which contains the trained XGBoost model and all preprocessing objects:

- Label encoders for `brand` and `model`
- Binary encoders for `transmission` and `fuelType`
- StandardScaler for numerical columns
- Final feature column order

When the form is submitted, `app.py` rebuilds the input row using the same preprocessing pipeline used during training. Unsupported model values safely fall back to `Other`, then the model predicts the final TRY price.

This prevents a common deployment mistake: training with one preprocessing flow but predicting with a different one.

### Deployment readiness
The project also includes a `Dockerfile` and `Procfile`, so the app can be served with **Gunicorn** and deployed more easily.

| File | Purpose |
|---|---|
| `app.py` | Flask backend and prediction route |
| `templates/index.html` | Web interface |
| `car_price_artifacts.pkl` | Model + preprocessing bundle |
| `Dockerfile` | Container setup |
| `Procfile` | Gunicorn deployment command |

---

## 13. Conclusions & Future Work

### What was achieved
- A complete, market-specific ML pipeline for Turkish used car price prediction
- Domain knowledge (ÖTV tax structure) successfully encoded as a feature engineering step
- 8 models benchmarked systematically with justified evaluation metrics
- Final model: **XGBoost with R² = 0.9682**, MAE ≈ 265,370 TRY
- A working Flask UI that uses the saved model artifacts for real-time price prediction

### Limitations
- Exchange rate and tax rates are hardcoded — the model would need retraining as economic conditions change
- Dataset is UK-sourced; Turkish-specific models, brands (e.g. TOGG), and regional pricing variations are not captured
- The tax multiplier is simplified — actual ÖTV calculation also depends on vehicle value and additional fees (MTV, KDV)
- The current UI supports single-vehicle prediction, not batch prediction

### Potential improvements
- Scrape real Turkish listing data (sahibinden.com, arabam.com) for a fully native dataset
- Add inflation adjustment as a time-aware feature
- Experiment with LightGBM and CatBoost as XGBoost alternatives
- Add SHAP explanations to the Flask UI so users can see why a price was predicted
- Add CSV batch prediction support

---

## 14. Tech Stack

| Component | Tool/Library |
|---|---|
| Language | Python 3.12 |
| Data processing | pandas, numpy |
| ML models | scikit-learn, xgboost |
| Categorical encoding | category_encoders |
| Hyperparameter tuning | RandomizedSearchCV |
| Visualization | matplotlib |
| Environment | Google Colab / Jupyter Notebook |
| Backend UI | Flask |
| Frontend | HTML, CSS, Jinja |
| Deployment | Gunicorn, Docker |

---

*Report prepared from project notebook | Turkish Automotive Market ML Study*  
*Muhammed Yuşa Duymaz — yusaduymaz.tech*
