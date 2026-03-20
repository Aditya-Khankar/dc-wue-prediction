"""
predict.py
===========
Quick inference script.
Load the trained model and predict
cooling water consumption for any data centre.

Usage:
    python predict.py

Or import in your own code:
    from predict import predict_wue
"""

import joblib
import numpy as np
import os

# ── Cooling system encoding ───────────────────────────────
COOLING_TYPES = {
    'air_side':     1,
    'evaporative':  2,
    'chilled_water':3,
    'seawater':     4,
    'liquid':       5,
}

# ── Climate zone encoding ─────────────────────────────────
CLIMATE_ZONES = {
    'tropical':     1,
    'subtropical':  2,
    'temperate':    3,
    'continental':  4,
    'oceanic':      5,
    'subarctic':    6,
    'mediterranean':7,
    'desert':       8,
}

def load_model(model_path: str = 'models/knn_wue_model.pkl'):
    """Load the trained KNN pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run knn_wue_pipeline.py first to train and save the model.")
    return joblib.load(model_path)


def predict_wue(
    cooling_type: str,
    ambient_temp_c: float,
    humidity_pct: float,
    it_load_mw: float,
    server_utilisation_pct: float,
    climate_zone: str,
    model=None,
    model_path: str = 'models/knn_wue_model.pkl',
) -> dict:
    """
    Predict cooling water consumption for a data centre.

    Parameters
    ----------
    cooling_type : str
        Cooling system: 'air_side', 'evaporative', 'chilled_water',
        'seawater', or 'liquid'
    ambient_temp_c : float
        Outside air temperature in degrees Celsius
    humidity_pct : float
        Outdoor relative humidity (0-100)
    it_load_mw : float
        Total IT equipment power draw in megawatts
    server_utilisation_pct : float
        Average server CPU/GPU utilisation (0-100)
    climate_zone : str
        Climate: 'tropical', 'subtropical', 'temperate',
        'continental', 'oceanic', 'subarctic',
        'mediterranean', or 'desert'
    model : sklearn Pipeline, optional
        Pre-loaded model. If None, loads from model_path.
    model_path : str
        Path to saved model pickle file.

    Returns
    -------
    dict with keys:
        wue          : float — litres per kWh
        litres_per_day : float — at given IT load
        compliant_bca : bool  — meets Singapore BCA Green Mark (< 0.4)
        compliant_best: bool  — meets best practice target (< 0.5)
        verdict       : str   — 'Excellent', 'Good', or 'Review'
    """
    # Validate inputs
    if cooling_type not in COOLING_TYPES:
        raise ValueError(
            f"cooling_type must be one of: {list(COOLING_TYPES.keys())}")
    if climate_zone not in CLIMATE_ZONES:
        raise ValueError(
            f"climate_zone must be one of: {list(CLIMATE_ZONES.keys())}")

    # Load model if not provided
    if model is None:
        model = load_model(model_path)

    # Build feature vector
    features = np.array([[
        COOLING_TYPES[cooling_type],
        ambient_temp_c,
        humidity_pct,
        it_load_mw,
        server_utilisation_pct,
        CLIMATE_ZONES[climate_zone],
    ]])

    # Predict
    wue = float(model.predict(features)[0])
    lpd = wue * it_load_mw * 1000 * 24  # litres/day

    verdict = ('Excellent' if wue < 0.5
               else 'Good' if wue < 1.0
               else 'Review required')

    return {
        'wue':              round(wue, 4),
        'litres_per_day':   round(lpd, 0),
        'compliant_bca':    wue < 0.4,
        'compliant_best':   wue < 0.5,
        'verdict':          verdict,
    }


# ── CLI demo ──────────────────────────────────────────────
if __name__ == '__main__':
    model = load_model()

    test_cases = [
        {
            'name':       'Singapore — Liquid Cooling',
            'cooling_type':        'liquid',
            'ambient_temp_c':      28,
            'humidity_pct':        82,
            'it_load_mw':         100,
            'server_utilisation_pct': 80,
            'climate_zone':        'tropical',
        },
        {
            'name':       'Stockholm — Air-side Cooling',
            'cooling_type':        'air_side',
            'ambient_temp_c':       5,
            'humidity_pct':        73,
            'it_load_mw':         200,
            'server_utilisation_pct': 75,
            'climate_zone':        'subarctic',
        },
        {
            'name':       'Houston — Evaporative Cooling',
            'cooling_type':        'evaporative',
            'ambient_temp_c':      28,
            'humidity_pct':        65,
            'it_load_mw':         300,
            'server_utilisation_pct': 85,
            'climate_zone':        'subtropical',
        },
        {
            'name':       'Dubai — Chilled Water',
            'cooling_type':        'chilled_water',
            'ambient_temp_c':      35,
            'humidity_pct':        40,
            'it_load_mw':          50,
            'server_utilisation_pct': 70,
            'climate_zone':        'desert',
        },
    ]

    print("=" * 65)
    print("  Cooling Water Consumption Prediction")
    print("=" * 65)
    print(f"\n  {'Location':<35} {'WUE':>6}  {'L/day':>10}  Verdict")
    print(f"  {'-'*62}")

    for tc in test_cases:
        name = tc.pop('name')
        result = predict_wue(**tc, model=model)
        bca = '✓' if result['compliant_bca'] else '✗'
        print(f"  {name:<35} {result['wue']:>6.3f}  "
              f"{result['litres_per_day']:>10,.0f}  "
              f"{result['verdict']} (BCA:{bca})")

    print(f"\n  BCA Green Mark target: < 0.4 L/kWh")
    print(f"  Best practice target : < 0.5 L/kWh")
