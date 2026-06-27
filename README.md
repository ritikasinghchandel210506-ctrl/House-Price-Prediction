# House Price Prediction App

A full-stack Machine Learning web application built using Python, Flask, and Scikit-Learn. The system trains a Predictive Linear Regression model using the Kaggle Housing Dataset to forecast residential real estate pricing based on core features (structural area, rooms, furnishing configuration, and utilities).

## 🚀 System Architecture & Pipeline
1. **Data Layer (`data/`):** Houses the raw `Housing.csv` records.
2. **Training Engine (`train.py`):** - Handles cleaning and binary encoding of categorical variants.
   - Applies dummy structural mappings to tracking components like `furnishingstatus`.
   - Scales numerical weights using `StandardScaler`.
   - Exports compressed asset states via serialization (`model.pkl`, `scaler.pkl`, `features.pkl`).
3. **Web Infrastructure (`app.py`):** Runs a localized Flask environment rendering a clean frontend dashboard UI to serve inferences dynamically.

---

## 🛠️ Project Structure

```text
├── data/
│   └── Housing.csv         # Kaggle Dataset
├── static/
│   └── style.css           # Custom UI Styling
├── templates/
│   └── index.html          # Dynamic Web Input Form
├── app.py                  # Flask Application Server
├── train.py                # ML Model Training Pipeline
└── requirements.txt        # Production Dependencies