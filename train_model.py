import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# 1. Create Synthetic Training Data (10,000 samples)
# Features: Eg, L, log_N, tau, T, G
n_samples = 10000
data = {
    'E_g': np.random.uniform(0.8, 2.0, n_samples),
    'L_absorber_um': np.random.uniform(0.5, 5.0, n_samples),
    'log_N': np.random.uniform(14.0, 18.0, n_samples),
    'tau_rec_us': np.random.uniform(0.1, 50.0, n_samples),
    'T_cell': np.random.uniform(273, 350, n_samples),
    'G_irradiance': np.random.uniform(200, 1000, n_samples)
}
X = pd.DataFrame(data)

# 2. Simulated Targets (Voc, Jsc, FF, PCE)
# In a real scenario, these come from your numerical SDM solver
y = pd.DataFrame({
    'Voc': X['E_g'] * 0.7,
    'Jsc': (2.0 - X['E_g']) * 25,
    'FF': np.full(n_samples, 0.8),
    'PCE': X['E_g'] * (2.0 - X['E_g']) * 20
})

# 3. Train the Model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X, y)

# 4. Save the Model
joblib.dump(model, 'solarsim_ml_model.joblib')
print("✅ Model trained and saved as solarsim_ml_model.joblib")
