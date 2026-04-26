"""
ATM Cash Forecaster — Flask Web Application
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

# ── Load model at startup ──────────────────────────────────────────
MODEL_PATH    = 'atm_forecast_model.pkl'
METADATA_PATH = 'model_metadata.json'
DATA_PATH = "data/history.csv"

history_df = pd.read_csv(DATA_PATH)

history_df["date"] = pd.to_datetime(
    history_df["date"]
)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Please run the Jupyter notebook first to train and save the model."
        )
    with open(MODEL_PATH, 'rb') as f:
        pkg = pickle.load(f)
    return pkg

def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    return {}

try:
    model_package = load_model()
    model    = model_package['model']
    features = model_package['features']
    metadata = model_package.get('metadata', load_metadata())
    print(f"✅ Model loaded — {len(features)} features | R²={metadata.get('r2','N/A')}")
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    model, features, metadata = None, [], {}

HOLIDAYS = [
    "2021-01-01", "2021-01-14","2021-01-26","2021-02-11","2021-03-12","2021-03-29","2021-04-02",
    "2021-04-25","2021-01-26","2021-07-20","2021-08-15","2021-09-10","2021-10-15","2021-11-04","2021-12-25",

    "2022-01-01", "2022-01-14","2022-01-26","2022-03-01","2022-03-16","2022-03-17","2022-04-10","2022-04-15",
    "2022-05-03","2022-04-18","2022-07-09","2022-08-15","2022-09-01", "2022-10-04","2022-10-24", "2022-12-25",

    "2023-01-01","2023-01-14","2023-01-26","2023-02-18", "2023-03-06", "2023-03-07","2023-03-31","2023-04-07","2023-04-21","2023-05-05","2023-06-29",
    "2023-07-19","2023-08-15","2023-08-29","2023-09-06","2023-09-18","2023-10-02","2023-10-12","2023-10-31","2023-11-14","2023-12-25",

    "2024-01-01","2024-01-14","2024-01-26","2024-03-25","2024-03-29","2024-04-05","2024-04-09","2024-04-11","2024-04-14","2024-04-17","2024-04-21",
    "2024-05-01","2024-06-17","2024-07-17","2024-08-15","2024-08-26","2024-09-16","2024-10-02","2024-10-12","2024-10-31","2024-11-15","2024-12-25",
]

# ── Feature builder (mirrors notebook logic) ──────────────────────
def build_features(atm_id, selected_date, history_df):

    # Convert date
    selected_date = pd.to_datetime(selected_date)

    # Filter ATM history before selected date
    atm_df = history_df[
        (history_df["atm_id"] == atm_id) &
        (history_df["date"] < selected_date)
    ].copy()

    atm_df = atm_df.sort_values("date")

    if len(atm_df) < 30:
        raise ValueError("Not enough history for prediction")

    # ── Date Features ──

    day_of_week = selected_date.dayofweek
    month = selected_date.month
    day_of_month = selected_date.day

    is_weekend = 1 if day_of_week >= 5 else 0
    is_month_start = 1 if day_of_month <= 3 else 0
    is_month_end = 1 if day_of_month >= 28 else 0

    is_payday = 1 if day_of_month in [1, 5, 10, 15, 25] else 0

    # ── Lag Features ──

    withdrawals = atm_df["withdrawal"]

    lag_2 = withdrawals.iloc[-2]
    lag_3 = withdrawals.iloc[-3]
    lag_5 = withdrawals.iloc[-5]
    lag_7 = withdrawals.iloc[-7]
    lag_10 = withdrawals.iloc[-10]
    lag_14 = withdrawals.iloc[-14]
    lag_21 = withdrawals.iloc[-21]
    lag_28 = withdrawals.iloc[-28]

    # ── Rolling Features ──

    roll_mean_3 = withdrawals.tail(3).mean()
    roll_std_3 = withdrawals.tail(3).std()

    roll_mean_7 = withdrawals.tail(7).mean()
    roll_std_7 = withdrawals.tail(7).std()

    roll_mean_10 = withdrawals.tail(10).mean()
    roll_std_10 = withdrawals.tail(10).std()

    roll_mean_14 = withdrawals.tail(14).mean()
    roll_std_14 = withdrawals.tail(14).std()

    roll_mean_21 = withdrawals.tail(21).mean()
    roll_std_21 = withdrawals.tail(21).std()

    roll_mean_30 = withdrawals.tail(30).mean()
    roll_std_30 = withdrawals.tail(30).std()


    # ── Placeholder Flags (can improve later)

    is_holiday = 1 if str(selected_date.date()) in HOLIDAYS else 0
    nearby_event_flag = 0

    weekend_holiday = (
        1 if is_weekend and is_holiday else 0
    )

    event_holiday = (
        1 if nearby_event_flag and is_holiday else 0
    )

    high_transaction_day = (
        1 if is_weekend or is_payday else 0
    )

    # ── Final Feature Row ──

    features = pd.DataFrame([{

        'atm_id': atm_id,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'nearby_event_flag': nearby_event_flag,
        'month': month,
        'day_of_week': day_of_week,
        'day_of_month': day_of_month,
        'is_month_start': is_month_start,
        'is_month_end': is_month_end,
        'is_payday': is_payday,
        'weekend_holiday': weekend_holiday,
        'event_holiday': event_holiday,
        'high_transaction_day': high_transaction_day,

        'withdrawal_lag_2': lag_2,
        'withdrawal_lag_3': lag_3,
        'withdrawal_lag_5': lag_5,
        'withdrawal_lag_7': lag_7,
        'withdrawal_lag_10': lag_10,
        'withdrawal_lag_14': lag_14,
        'withdrawal_lag_21': lag_21,
        'withdrawal_lag_28': lag_28,

        'withdrawal_roll_mean_3': roll_mean_3,
        'withdrawal_roll_std_3': roll_std_3,
        'withdrawal_roll_mean_7': roll_mean_7,
        'withdrawal_roll_std_7': roll_std_7,
        'withdrawal_roll_mean_10': roll_mean_10,
        'withdrawal_roll_std_10': roll_std_10,
        'withdrawal_roll_mean_14': roll_mean_14,
        'withdrawal_roll_std_14': roll_std_14,
        'withdrawal_roll_mean_21': roll_mean_21,
        'withdrawal_roll_std_21': roll_std_21,
        'withdrawal_roll_mean_30': roll_mean_30,
        'withdrawal_roll_std_30': roll_std_30
    }])

    return features

FEATURE_ORDER = [
    'atm_id',
    'is_weekend',
    'is_holiday',
    'nearby_event_flag',
    'month',
    'day_of_week',
    'day_of_month',
    'is_month_start',
    'is_month_end',
    'is_payday',
    'weekend_holiday',
    'event_holiday',
    'high_transaction_day',
    'withdrawal_lag_2', 
    'withdrawal_lag_3', 
    'withdrawal_lag_5', 
    'withdrawal_lag_7', 
    'withdrawal_lag_10', 
    'withdrawal_lag_14', 
    'withdrawal_lag_21', 
    'withdrawal_lag_28', 
    'withdrawal_roll_mean_3', 
    'withdrawal_roll_std_3', 
    'withdrawal_roll_mean_7', 
    'withdrawal_roll_std_7', 
    'withdrawal_roll_mean_10', 
    'withdrawal_roll_std_10',
    'withdrawal_roll_mean_14', 
    'withdrawal_roll_std_14', 
    'withdrawal_roll_mean_21', 
    'withdrawal_roll_std_21', 
    'withdrawal_roll_mean_30', 
    'withdrawal_roll_std_30'
]

# ── Routes ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', metadata=metadata)

@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({
            'error': 'Model not loaded.'
        }), 503

    try:
        # Get JSON input
        data = request.get_json()

        # Expected inputs
        atm_id = int(data['atm_id'])
        selected_date = data['date']

        # Build features from history
        features_df = build_features(
            atm_id=atm_id,
            selected_date=selected_date,
            history_df=history_df
        )

        # Ensure correct feature order
        features_df = features_df[FEATURE_ORDER]

        # Model prediction
        prediction = float(
            model.predict(features_df)[0]
        )

        # Clean prediction
        prediction = max(0, round(prediction))

        # Simple level logic
        avg = history_df["withdrawal"].mean()

        diff_pct = (
            (prediction - avg) / avg
        ) * 100

        trend = (
            "above" if diff_pct > 0
            else "below"
        )

        insight = (
            f"{abs(diff_pct):.1f}% "
            f"{trend} historical average"
        )

        # Demand level
        MAX_CAPACITY = 200000
        if prediction > MAX_CAPACITY * 0.7:
            level = "⚠ Refill Soon"
            color = "#F35549"
        else:
            level = "✔ Safe Level"
            color = "#4CAF50"

        
        return jsonify({

            'prediction': prediction,

            'level': level,

            'level_color': color,

            'insight': insight

        })

    except (KeyError, ValueError, TypeError) as e:

        return jsonify({
            'error': f'Invalid input: {str(e)}'
        }), 400

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500
    

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'features': len(features),
        'metadata': metadata
    })


if __name__ == '__main__':
    print("\n ATM Cash Forecaster Web App")
    print("   URL: http://localhost:5000")
    print("   Press CTRL+C to quit\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
