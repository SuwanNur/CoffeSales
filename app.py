# ===============================================================
# ☕ COFFEE SALES PREDICTION MODEL TRAINING
# ===============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from google.colab import files

# ===============================================================
# 📂 Load Dataset
# ===============================================================
url = "https://drive.google.com/uc?id=1cnzXzPrShReziNp7B1hQ9HURVu9drGGF"  # Ganti dengan link file Coffe_sales kamu
df = pd.read_csv(url)

print("🔹 5 Data Pertama:")
print(df.head())

print("\n📊 Informasi Dataset:")
print(df.info())

# ===============================================================
# 🧹 Pra-pemrosesan Data
# ===============================================================
# Fitur yang digunakan untuk prediksi
num_features = ['hour_of_day', 'Weekdaysort', 'Monthsort']

# Target variabel: nilai penjualan
target = 'money'

# Pisahkan fitur dan target
X = df[num_features]
y = df[target]

# ===============================================================
# ✂️ Split Data (Training & Testing)
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================================================
# ⚖️ Standarisasi Fitur Numerik
# ===============================================================
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
X_test_scaled[num_features] = scaler.transform(X_test[num_features])

# ===============================================================
# 🌲 Latih Model Random Forest
# ===============================================================
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# ===============================================================
# 💾 Simpan Model dan Scaler
# ===============================================================
joblib.dump(rf, "rf_model.joblib")
joblib.dump(scaler, "scaler_coffee.joblib")

print("✅ Model dan scaler berhasil disimpan!")

# ===============================================================
# 📈 Evaluasi Sederhana
# ===============================================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = rf.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Hasil Evaluasi Model:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ===============================================================
# ⬇️ Unduh File ke Komputer Lokal (untuk Streamlit)
# ===============================================================
files.download("rf_model.joblib")
files.download("scaler_coffee.joblib")

print("\n📥 File model dan scaler siap digunakan di Streamlit app!")
