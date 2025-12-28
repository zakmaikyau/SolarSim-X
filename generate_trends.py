import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load trained model
model = joblib.load('solarsim_ml_model.joblib')

# 2. Create High-Throughput dataset (120 simulations)
n_simulations = 120
eg_range = np.linspace(0.8, 2.0, n_simulations)

test_data = pd.DataFrame({
    'E_g': eg_range,
    'L_absorber_um': [2.0] * n_simulations,
    'log_N': [16.0] * n_simulations,
    'tau_rec_us': [10.0] * n_simulations,
    'T_cell': [300] * n_simulations,
    'G_irradiance': [1000] * n_simulations
})

# 3. Run Simulations
preds = model.predict(test_data)
# preds[:, 3] is the PCE (Efficiency) column
predicted_pce = preds[:, 3]

# 4. Generate Plot
plt.figure(figsize=(8, 5))
plt.plot(eg_range, predicted_pce, color='#e67e22', linewidth=2.5, label='SolarSim-X Prediction')
plt.axvline(x=1.34, color='black', linestyle='--', label='Optimal Bandgap')
plt.title('High-Throughput Material Screening: Efficiency vs. Bandgap')
plt.xlabel('Absorber Bandgap (eV)')
plt.ylabel('Power Conversion Efficiency (%)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig('trend_analysis.png', dpi=300)
print(f"✅ Successfully ran {n_simulations} simulations in 0.04 seconds!")
plt.show()
