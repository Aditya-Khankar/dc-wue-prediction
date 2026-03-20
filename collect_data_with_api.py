"""
collect_data_with_api.py
=========================
Enriches the base dataset with REAL ambient temperature
and humidity from OpenWeatherMap API.

This upgrades the 30 company-report rows from
climatological averages to actual measured weather
at each data centre location.

Setup (free, 5 minutes):
    1. Sign up at openweathermap.org/api
    2. Copy your free API key (60 calls/min, 1000/day)
    3. Set it as environment variable:
          export OWM_API_KEY="your_key_here"
       OR paste directly into OWM_API_KEY below.

Run:
    python collect_data_with_api.py

Output:
    data/data_centre_wue_realweather.csv
    — identical structure to data_centre_wue.csv
    — company report rows now have real weather data
    — drop-in replacement for the model

Why this matters:
    The 30 company-report rows are from real facilities.
    Adding real current weather makes their features
    genuinely measured, not estimated. This strengthens
    the real-data component of the dataset.
"""

import os
import time
import requests
import pandas as pd
import numpy as np

# ── API Key ───────────────────────────────────────────────
# Option 1: Set environment variable OWM_API_KEY
# Option 2: Paste key directly here
OWM_API_KEY = os.environ.get('OWM_API_KEY', 'YOUR_API_KEY_HERE')

OWM_ENDPOINT = 'https://api.openweathermap.org/data/2.5/weather'


def get_weather(lat: float, lon: float,
                api_key: str,
                retries: int = 3) -> dict | None:
    """
    Fetch current weather from OpenWeatherMap API.

    Returns dict with:
        temp_c   : float — current temperature (°C)
        humidity : float — relative humidity (%)
        desc     : str   — weather description
    Returns None on failure.
    """
    params = {
        'lat':   lat,
        'lon':   lon,
        'appid': api_key,
        'units': 'metric',  # Celsius
    }
    for attempt in range(retries):
        try:
            r = requests.get(OWM_ENDPOINT, params=params,
                             timeout=10)
            if r.status_code == 200:
                d = r.json()
                return {
                    'temp_c':   round(d['main']['temp'],       1),
                    'humidity': round(d['main']['humidity'],   1),
                    'pressure': round(d['main']['pressure'],   1),
                    'desc':     d['weather'][0]['description'],
                }
            elif r.status_code == 401:
                print(f"  ✗ Invalid API key. "
                      f"Get one free at openweathermap.org/api")
                return None
            else:
                print(f"  ⚠ HTTP {r.status_code} for "
                      f"({lat},{lon}), attempt {attempt+1}")
        except requests.exceptions.Timeout:
            print(f"  ⚠ Timeout for ({lat},{lon}), "
                  f"attempt {attempt+1}")
        except Exception as e:
            print(f"  ⚠ Error: {e}")
        time.sleep(1.0)
    return None


def enrich_with_real_weather(
    input_path:  str = 'data/data_centre_wue.csv',
    output_path: str = 'data/data_centre_wue_realweather.csv',
    api_key:     str = OWM_API_KEY,
) -> pd.DataFrame:
    """
    Load base dataset, fetch real weather for company-report
    rows, and save enriched dataset.
    """
    if api_key == 'YOUR_API_KEY_HERE':
        print("=" * 60)
        print("  OpenWeatherMap API key not set.")
        print("  Get a free key at: openweathermap.org/api")
        print("  Then set: export OWM_API_KEY='your_key'")
        print("  Or paste it into OWM_API_KEY in this file.")
        print("=" * 60)
        return None

    df = pd.read_csv(input_path)
    real_mask = df['Data_Source'] == 'company_report'
    real_rows = df[real_mask].copy()

    print(f"Fetching real weather for "
          f"{len(real_rows)} company-report data centres...")
    print(f"(Rate: 0.5 second delay between calls)\n")

    updated = 0
    for idx, row in real_rows.iterrows():
        print(f"  [{updated+1}/{len(real_rows)}] "
              f"{row['Data_Centre_Name'][:40]}...", end=' ')

        weather = get_weather(
            row['Latitude'], row['Longitude'], api_key)

        if weather:
            df.at[idx, 'Ambient_Temperature_C'] = weather['temp_c']
            df.at[idx, 'Relative_Humidity_Pct'] = weather['humidity']
            # Update Data_Source to reflect real weather
            df.at[idx, 'Data_Source'] = 'company_report_realweather'
            df.at[idx, 'Citation'] = (
                row['Citation'] +
                f" | Weather: OpenWeatherMap API "
                f"({pd.Timestamp.now().strftime('%Y-%m-%d')})")
            print(f"✓ {weather['temp_c']}°C, "
                  f"{weather['humidity']}% humidity")
            updated += 1
        else:
            print("✗ Failed — keeping climatological average")

        time.sleep(0.5)  # Respect rate limit

    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  Enriched dataset saved: {output_path}")
    print(f"  Rows updated with real weather: {updated} / {len(real_rows)}")
    print(f"  Total rows: {len(df)}")
    print(f"\n  To use in model:")
    print(f"    Change DATA_PATH in knn_wue_pipeline.py to:")
    print(f"    '{output_path}'")
    print(f"{'='*60}")

    return df


if __name__ == '__main__':
    enrich_with_real_weather()
