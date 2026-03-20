# Predicting Cooling Water Consumption in Data Centres
## KNN Regression — End-to-End ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![sklearn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-1094%20rows-yellow)]()

---

## The Problem

Data centres consumed **1.8 trillion litres** of fresh water in 2023.
AI workloads are projected to **triple** this by 2027.

Every new data centre needs a WUE (Water Usage Effectiveness) estimate
before construction — but no publicly available tool exists.

> **WUE = litres of water consumed per kWh of IT equipment load**
> Low WUE = efficient. High WUE = wasteful.

---

## Why KNN Regression?

WUE has **no global formula**. The relationship between cooling type,
ambient temperature, and water consumption is **locally structured**:

- The same evaporative cooling system uses near-zero water in Helsinki
  but 2.5 L/kWh in Singapore
- Linear regression assumes a global pattern — it gets **R² = 0.10**
- Random Forest captures non-linearity but cannot explain predictions
- **KNN finds similar past data centres and reads off their water usage**
  → **R² = 0.71**, and you can see exactly which DCs influenced the result

### Model Comparison

| Model | R² | RMSE (L/kWh) | Interpretable? |
|---|---|---|---|
| **KNN (tuned)** | **0.7103** | **0.1777** | **Yes — shows similar DCs** |
| Random Forest | 0.9015 | 0.1037 | No — black box |
| Linear Regression | 0.1007 | 0.3131 | Yes — but wrong |
| Ridge Regression | 0.1008 | 0.3131 | Yes — but wrong |

Random Forest achieves higher R² but cannot explain which past data centres drove the prediction — a functional requirement in engineering applications.

---

## Dataset

**1094 data centre evaluations** across 60 locations worldwide.

| Source | Rows | Type |
|---|---|---|
| Microsoft Sustainability Report 2023 | 9 | Real published WUE |
| Google Environmental Report 2024 | 6 | Real published WUE |
| AWS Sustainability Report 2023 | 7 | Real published WUE |
| Meta Sustainability Report 2023 | 5 | Real published WUE |
| Equinix Sustainability Report 2023 | 3 | Real published WUE |
| Physics simulation (LBNL-2001637) | 1064 | Grounded simulation |

### Features

| Feature | Unit | Meaning |
|---|---|---|
| `Cooling_Type_Encoded` | 1-5 | 1=Air-side, 2=Evaporative, 3=Chilled water, 4=Seawater, 5=Liquid |
| `Ambient_Temperature_C` | °C | Outside air temperature at DC location |
| `Relative_Humidity_Pct` | % | Outdoor relative humidity |
| `IT_Load_MW` | MW | Total IT equipment power draw |
| `Server_Utilisation_Pct` | % | Average server utilisation |
| `Climate_Zone_Encoded` | 1-8 | 1=Tropical to 8=Desert |

**Target:** `WUE_Litres_Per_kWh` — litres of water per kWh of IT load

---

## Pipeline

```
Data Loading
    ↓
Validation (missing values, duplicates)
    ↓
Feature Selection (6 features, justified)
    ↓
Train/Test Split (80/20, random_state=42)
    ↓
StandardScaler (fit on train only — no leakage)
    ↓
Model Comparison (KNN, LR, Ridge, RF)
    ↓
GridSearchCV (K=1..30 × weights × metric, 5-Fold CV)
    ↓
Final Evaluation (R², RMSE, MAE vs baseline)
    ↓
KNN Explainability (nearest neighbour analysis)
    ↓
Model Export (joblib .pkl)
```

---

## Results

```
Best K       : 7
Best weights : distance (1/d weighting)
Best metric  : manhattan
R²           : 0.7103
RMSE         : 0.1777 L/kWh
MAE          : 0.1072 L/kWh
vs baseline  : 46.3% improvement
```

---

## Singapore Use Case

Singapore is building 3 new AI-focused data centres in 2026.
Government mandates BCA Green Mark certification (WUE < 0.4 L/kWh).

| Cooling Design | WUE (L/kWh) | Water/day (L) | Status |
|---|---|---|---|
| Air-side cooling | 0.125 | 300,000 | ✓ Meets target |
| Liquid cooling | 0.274 | 657,000 | ✓ Meets target |
| Chilled water | 0.814 | 1,953,000 | ✗ Above target |
| Evaporative | 0.809 | 1,941,000 | ✗ Above target |

*Input: 28°C, 82% humidity, 100MW IT load — real Singapore conditions*

---

## Project Structure

```
dc-wue-prediction/
├── data/
│   └── data_centre_wue.csv         ← 1094 rows, 14 columns
├── outputs/                         ← figures generated on run
├── models/                          ← model saved on run
├── knn_wue_pipeline.py
├── generate_figures.py
├── predict.py
├── collect_data_with_api.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/Aditya-Khankar/dc-wue-prediction
cd dc-wue-prediction

# Install
pip install -r requirements.txt

# Run
python knn_wue_pipeline.py
```

### API Key (optional)
To use real weather data:
```bash
export OWM_API_KEY="your_key_here"
python collect_data_with_api.py
```
Get a free key at [openweathermap.org/api](https://openweathermap.org/api).
**Never commit your API key to git.**

### Load saved model

```python
import joblib
import numpy as np

model = joblib.load('models/knn_wue_model.pkl')

# Predict WUE for a new data centre
# [Cooling_Type, Temp_C, Humidity, IT_Load_MW, Utilisation, Climate_Zone]
new_dc = np.array([[5, 28, 82, 100, 80, 1]])  # Singapore, liquid cooling
wue = model.predict(new_dc)[0]
print(f"Predicted WUE: {wue:.3f} L/kWh")
print(f"Water per day: {wue * 100_000 * 24:,.0f} litres")
```

---

## Key Design Decisions

### Why KNN over Random Forest?
Random Forest achieves higher R² (0.90 vs 0.71) but **cannot explain
its predictions**. A data centre architect needs to know:
*"This prediction is based on 7 similar past data centres that averaged
0.27 L/kWh."* KNN provides this naturally. Random Forest cannot.

### Why distance weighting?
A data centre at distance 0.1 is far more similar than one at distance 1.0.
Uniform weighting treats both equally. Distance weighting (`1/d`) gives
closer neighbours more influence — physically correct for locality problems.

### Why manhattan over euclidean?
GridSearchCV found manhattan metric performs better here. Manhattan
distance (Σ|xᵢ-yᵢ|) is less sensitive to outliers in individual features
than euclidean (√Σ(xᵢ-yᵢ)²). IT_Load_MW has high variance — manhattan
handles this better.

### Why StandardScaler inside Pipeline?
Prevents data leakage. If the scaler is fit on all data before CV,
test fold statistics leak into training — artificially inflating scores.
Pipeline ensures the scaler sees only the training fold during each CV split.

---

## References

```
[1] Shehabi, A. et al. (2024). 2024 United States Data Center
    Energy Usage Report. LBNL-2001637. Lawrence Berkeley
    National Laboratory. https://eta.lbl.gov/

[2] Microsoft (2023). Sustainability Report 2023.
    datacenters.microsoft.com/sustainability

[3] Google (2024). Environmental Report 2024.
    sustainability.google/reports/2024

[4] Amazon Web Services (2023). Sustainability Report 2023.
    sustainability.aboutamazon.com

[5] Meta (2023). Sustainability Report 2023.
    sustainability.fb.com

[6] Equinix (2023). Sustainability Report 2023.
    equinix.com/sustainability

[7] EESI (2024). Data Centre Water Usage.
    eesi.org/articles/view/data-centers

[8] Strubell, E. et al. (2019). Energy and Policy Considerations
    for Deep Learning in NLP. ACL 2019.

[9] Li, P. et al. (2023). Making AI Less Thirsty.
    arXiv:2304.03271
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

