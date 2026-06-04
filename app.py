from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "car_price_artifacts.pkl"

if not ARTIFACT_PATH.exists():
    raise FileNotFoundError(
        f"Missing artifact file: {ARTIFACT_PATH}. Run car_price_prediction.py first."
    )

artifacts = joblib.load(ARTIFACT_PATH)
model = artifacts["model"]
scaler = artifacts["scaler"]
label_encoders = artifacts["label_encoders"]
binary_encoders = artifacts["binary_encoders"]
feature_columns = artifacts["feature_columns"]
columns_to_scale = artifacts["columns_to_scale"]

brand_options = list(label_encoders["brand"].classes_)
model_options = list(label_encoders["model"].classes_)
transmission_options = ["Automatic", "Manual", "Other", "Semi-Auto"]
fuel_options = ["Diesel", "Electric", "Hybrid", "Other", "Petrol"]

csv_files = {
    "audi.csv": "Audi",
    "bmw.csv": "BMW",
    "cclass.csv": "Mercedes",
    "focus.csv": "Ford",
    "ford.csv": "Ford",
    "hyundi.csv": "Hyundai",
    "merc.csv": "Mercedes",
    "skoda.csv": "Skoda",
    "toyota.csv": "Toyota",
    "vauxhall.csv": "Opel",
    "vw.csv": "Volkswagen",
}

brand_model_map = {brand: [] for brand in brand_options}
for file_name, brand in csv_files.items():
    csv_path = BASE_DIR / file_name
    if not csv_path.exists():
        continue

    data = pd.read_csv(csv_path)
    if "tax(£)" in data.columns:
        data = data.rename(columns={"tax(£)": "tax"})
    data = data.drop(columns=["tax", "mpg"], errors="ignore")

    seen_models = set()
    for value in data["model"].dropna().tolist():
        if value not in seen_models:
            seen_models.add(value)
            if value not in brand_model_map[brand]:
                brand_model_map[brand].append(value)

for brand in brand_model_map:
    brand_model_map[brand] = sorted(brand_model_map[brand], key=lambda value: value.strip().lower())
    if "Other" not in brand_model_map[brand]:
        brand_model_map[brand].append("Other")

app = Flask(__name__)


def parse_float(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a valid number.")


def parse_int(value, field_name):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a valid whole number.")


def build_feature_frame(form_data):
    brand = form_data["brand"]
    model_name = form_data["model"].strip() or "Other"
    transmission = form_data["transmission"]
    fuel_type = form_data["fuelType"]

    if brand not in label_encoders["brand"].classes_:
        raise ValueError("Selected brand is not supported by the trained model.")

    if model_name not in brand_model_map.get(brand, []):
        model_name = "Other"

    if model_name not in label_encoders["model"].classes_:
        model_name = "Other"

    row = pd.DataFrame(
        [
            {
                "brand": brand,
                "model": model_name,
                "year": parse_int(form_data["year"], "Year"),
                "km": parse_float(form_data["mileage"], "Mileage"),
                "transmission": transmission,
                "fuelType": fuel_type,
                "engineSize": parse_float(form_data["engineSize"], "Engine size"),
            }
        ]
    )

    row["brand"] = label_encoders["brand"].transform(row["brand"])
    row["model"] = label_encoders["model"].transform(row["model"])

    row = binary_encoders["transmission"].transform(row)
    row = binary_encoders["fuelType"].transform(row)

    row[columns_to_scale] = scaler.transform(row[columns_to_scale])

    for column in feature_columns:
        if column not in row.columns:
            row[column] = 0

    return row[feature_columns]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    submitted = {
        "brand": brand_options[0],
        "model": "",
        "year": 2020,
        "mileage": 50000,
        "transmission": transmission_options[0],
        "fuelType": fuel_options[-1],
        "engineSize": 1.6,
    }

    if request.method == "POST":
        submitted = {
            "brand": request.form.get("brand", brand_options[0]),
            "model": request.form.get("model", ""),
            "year": request.form.get("year", 2020),
            "mileage": request.form.get("mileage", 50000),
            "transmission": request.form.get("transmission", transmission_options[0]),
            "fuelType": request.form.get("fuelType", fuel_options[-1]),
            "engineSize": request.form.get("engineSize", 1.6),
        }

        try:
            features = build_feature_frame(submitted)
            predicted_price = model.predict(features)[0]
            prediction = round(float(predicted_price), 2)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        submitted=submitted,
        brand_options=brand_options,
        brand_model_map=brand_model_map,
        transmission_options=transmission_options,
        fuel_options=fuel_options,
    )


if __name__ == "__main__":
    app.run(debug=True)
