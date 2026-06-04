import numpy as np # linear algebra
import pandas as pd # data processing, CSV I/O (e.g. pd.read_csv)

from pathlib import Path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

"""# 1. DATASETS IMPORT AND MERGING"""

import os

# 1. Base path tanımı
# Script, notebook veya local klasörden çalıştığında CSV'leri aynı klasörden okur.
base_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# 2. Dosya isimleri ve değişken eşleştirmesi
# Key: Dosya adı, Value: Senin istediğin değişken ismi
files = {
    "audi.csv": "df_audi",
    "bmw.csv": "df_bmw",
    "cclass.csv": "df_cclass",
    "focus.csv": "df_focus",
    "ford.csv": "df_ford",
    "hyundi.csv": "df_hyundi",
    "merc.csv": "df_merc",
    "skoda.csv": "df_skoda",
    "toyota.csv": "df_toyota",
    "vauxhall.csv": "df_opel",
    "vw.csv": "df_vw"
}

df_all = []
for file_name, var_name in files.items():

    full_path = base_path / file_name
    if not full_path.exists():
        raise FileNotFoundError(f"CSV file not found: {full_path}")

    df = pd.read_csv(full_path)

    globals()[var_name] = df

    df_all.append(df)

print(f"Total datasets loaded: {len(df_all)}")

"""# 2. DATA PREPROCESSING AND CLEANING"""

for df in df_all:
    print(df.columns)
    print("---------------------------------------------------------")

df_hyundi.head()

df_hyundi.rename(columns={'tax(£)': 'tax'}, inplace=True)

for df in df_all:
    df.drop(columns=['tax',"mpg"], inplace=True, errors="ignore")

df_audi['brand'] = 'Audi'
df_bmw['brand'] = 'BMW'
df_cclass['brand'] = 'Mercedes'
df_focus['brand'] = 'Ford'
df_ford['brand'] = 'Ford'
df_hyundi['brand'] = 'Hyundai'
df_merc['brand'] = 'Mercedes'
df_skoda['brand'] = 'Skoda'
df_toyota['brand'] = 'Toyota'
df_opel['brand'] = 'Opel'
df_vw['brand'] = 'Volkswagen'

df_merge = pd.concat([df_audi, df_bmw, df_cclass, df_focus, df_ford,
                      df_hyundi, df_merc, df_skoda, df_toyota, df_opel, df_vw],
                     ignore_index=True)

df_merge.head()

df_merge.shape

df_merge.info()

df_merge.isnull().sum()

df_merge.describe()

print(len(df_merge[df_merge["engineSize"] == 0]))

df_merge = df_merge[df_merge["engineSize"] != 0]

print(len(df_merge[df_merge["engineSize"] == 0]))

print(len(df_merge.value_counts()))

df_merge = df_merge[(df_merge["year"] >= 2000) & (df_merge["year"] < 2026)]

len(df_merge.value_counts())

df_merge.columns

"""# 3. HANDLING HIGH CARDINALITY (MODEL THRESHOLDING)"""

df_merge["model"].value_counts()

sns.boxplot(df_merge["model"].value_counts())
plt.title("Distribution of Vehicle Models")
plt.show()

counts = df_merge['model'].value_counts()
print(f"Unique models : {df_merge['model'].nunique()}")

threshold = 100

repl = counts[counts <= threshold].index

df_merge['model'] = df_merge['model'].apply(lambda x: 'Other' if x in repl else x)

print(f"Unique models remaining: {df_merge['model'].nunique()}")

df_merge.head()

plt.figure(figsize=(12, 7))

sns.scatterplot(data=df_merge, x="year", y="price", alpha=0.4, s=20, color="darkblue")
plt.title("Relationship Between Vehicle Year and Price", fontsize=15, pad=15)
plt.xlabel("Year of Registration", fontsize=12)
plt.ylabel("Price (GBP)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

plt.figure(figsize=(10, 6))

sns.boxplot(data=df_merge, x="transmission", y="price", hue="transmission", palette="Set2", legend=False)

plt.title("Vehicle Price Distribution by Transmission Type", fontsize=14, pad=15)
plt.xlabel("Transmission Type", fontsize=12)
plt.ylabel("Price (GBP)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

sns.pairplot(df_merge, diag_kind='kde',plot_kws={'alpha': 0.4, 's': 15, 'linewidth': 0})

plt.suptitle("Multivariate Analysis of Car Features", y=1.02, fontsize=16)
plt.show()

sns.heatmap(df_merge.corr(numeric_only= True),annot = True, fmt=".2f",linewidth=.10)
plt.show()

"""# 4. OUTLIER DETECTION AND REMOVAL (IQR METHOD)"""

def plot_outliers(df, columns):

    for col in columns:
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 2, 1)
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f'Box Plot of {col.capitalize()}')

        plt.subplot(1, 2, 2)
        sns.histplot(df[col], kde=True, color='salmon')
        plt.title(f'Distribution of {col.capitalize()}')

        plt.tight_layout()
        plt.show()

numerical_cols = ['price', 'mileage', 'year', 'engineSize']
plot_outliers(df_merge, numerical_cols)

def remove_outliers_iqr(df, columns):

    original_size = len(df)

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    new_size = len(df)
    print(f"Rows removed: {original_size - new_size}")
    print(f"Final dataset size: {new_size}")

    return df

numerical_features = ['price', 'mileage', 'year', 'engineSize']
df_merge = remove_outliers_iqr(df_merge, numerical_features)

plot_outliers(df_merge, numerical_cols)

"""# 5. LOCALIZATION (CURRENCY AND TAX ADJUSTMENTS)
In this section, we convert prices from GBP to TRY and mileage to kilometers. We also apply a dynamic tax multiplier based on engine size (1.6L and 2.0L thresholds) to accurately reflect the Turkish automotive market.
"""

exchange_rate = 58  # 1 GBP = 58 TRY
mile_to_km = 1.609

def calculate_tr_price(row):
    gbp_price = row['price']
    engine = row['engineSize']


    base_tl = gbp_price * exchange_rate


    if engine <= 1.6:
        tax_multiplier = 3.0
    elif engine <= 2.0:
        tax_multiplier = 4.5
    else:
        tax_multiplier = 6.0

    return base_tl * tax_multiplier


df_merge['mileage'] = df_merge['mileage'] * mile_to_km
df_merge['price'] = df_merge.apply(calculate_tr_price, axis=1)


df_merge.rename(columns={'mileage': 'km', 'price': 'price_tl'}, inplace=True)

"""# 6. DATA SPLITTING AND CATEGORICAL ENCODING"""

X = df_merge.drop("price_tl",axis = 1)
y = df_merge["price_tl"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


df_merge.head()

encoders = {}
label_cols = ["brand", "model"]

for col in label_cols:
    encoders[col] = LabelEncoder()
    X_train[col] = encoders[col].fit_transform(X_train[col])
    X_test[col] = encoders[col].transform(X_test[col])

binary_encoders = {}
binary_cols = ["transmission", "fuelType"]

for col in binary_cols:
    binary_encoders[col] = ce.BinaryEncoder(cols=[col])

    X_train = binary_encoders[col].fit_transform(X_train)
    X_test = binary_encoders[col].transform(X_test)

print("-X Train -",X_train.head())
print("-----------------------------------------------------------------")
print("-X Test -",X_test.head())

"""# 7. FEATURE SCALING (STANDARD SCALER)"""

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

scaler = StandardScaler()
cols_to_scale = ['km', 'engineSize', 'year']
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

X_train_scaled.head()

"""# 8. MODEL TRAINING AND COMPARISON"""

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(true,predicted):
    mae = mean_absolute_error(true,predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true,predicted))
    r2_square = r2_score(true,predicted)
    return mae,mse,rmse,r2_square

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "k Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Adaboost Regressor": AdaBoostRegressor(),
    "Gradient Bost Regressor": GradientBoostingRegressor(),
    "XGBoost Regressor": XGBRegressor()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train_scaled,y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    model_train_mae,model_train_mse,model_train_rmse,model_train_r2_square = evaluate_model(y_train,y_train_pred)
    model_test_mae,model_test_mse,model_test_rmse,model_test_r2_square = evaluate_model(y_test,y_test_pred)

    print(list(models.keys())[i])
    print("Model Performance for Training Set")
    print("Root Mean Squared Error: ", model_train_rmse)
    print("Mean Absolute Error: ",model_train_mae)
    print("R2 Score: ",model_train_r2_square)

    print("-----------------------------------------------")

    print("Model Performance for Test Set")
    print("Root Mean Squared Error: ", model_test_rmse)
    print("Mean Absolute Error: ",model_test_mae)
    print("R2 Score: ",model_test_r2_square)

    print("-----------------------------------------------")
    print("\n")

"""# 9. HYPERPARAMETER TUNING (XGBOOST OPTIMIZATION)"""

from sklearn.model_selection import RandomizedSearchCV

xgboost_params = {
    "learning_rate": [0.1,0.01],
    "max_depth": [5,8,12,20,30],
    "n_estimators": [100,200,300,500],
    "colsample_bytree": [0.3,0.4,0.5,0.7,1]
}

randomized_cv = RandomizedSearchCV(
    estimator=XGBRegressor(),
    param_distributions=xgboost_params,
    n_iter=5,
    cv=5,
    n_jobs=1,
    random_state=42,
)

randomized_cv.fit(X_train_scaled,y_train)

randomized_cv.best_params_

"""# 10. MODEL EVALUATION AND PERFORMANCE METRICS"""

model = XGBRegressor(colsample_bytree=0.7, learning_rate=0.1, max_depth=12, n_estimators=200)
model.fit(X_train_scaled,y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

model_train_mae,model_train_mse,model_train_rmse,model_train_r2_square = evaluate_model(y_train,y_train_pred)
model_test_mae,model_test_mse,model_test_rmse,model_test_r2_square = evaluate_model(y_test,y_test_pred)

print("XGBoost Regressor")
print("Model Performance for Training Set")
print("Root Mean Squared Error: ", model_train_rmse)
print("Mean Absolute Error: ",model_train_mae)
print("R2 Score: ",model_train_r2_square)

print("-----------------------------------------------")

print("Model Performance for Test Set")
print("Root Mean Squared Error: ", model_test_rmse)
print("Mean Absolute Error: ",model_test_mae)
print("R2 Score: ",model_test_r2_square)

"""# 10.1 MODEL INTERPRETATION (TECHNICAL INSIGHT)
As seen in the Feature Importance chart, 'Year' and 'EngineSize' are the dominant predictors. This aligns with the Turkish market where vehicle age and the progressive tax system based on engine displacement significantly dictate the final valuation.
"""

importances = pd.Series(model.feature_importances_, index=X_train_scaled.columns)

plt.figure(figsize=(10, 6))
importances.nlargest(10).plot(kind='barh', color='teal')
plt.title('Top 10 Factors Affecting Car Price (Turkey Market)')
plt.xlabel('Importance Score')
plt.show()

import joblib

# Deployment için model + preprocessing nesnelerini birlikte kaydet.
artifacts = {
    "model": model,
    "scaler": scaler,
    "label_encoders": encoders,
    "binary_encoders": binary_encoders,
    "feature_columns": X_train_scaled.columns.tolist(),
    "columns_to_scale": cols_to_scale,
}

joblib.dump(artifacts, 'car_price_artifacts.pkl')
joblib.dump(model, 'car_price_model.pkl')

