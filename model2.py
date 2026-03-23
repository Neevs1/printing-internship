"""import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump

# 1. Load the datasets
data = pd.read_excel('real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')

# 2. Preprocessing & Harmonization
# Standardize column names across all extended datasets
for df in [fivemmdata, ed1, ed2]:
    df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after'}, inplace=True)

data = pd.concat([data, fivemmdata, ed1, ed2], ignore_index=True)
data = data.dropna()

# 3. Feature Engineering
# Mappings
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
# Cleaning
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
# Physics Relations
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

# 4. Feature Selection for Density Prediction
# We include 'initial density' and 'final ink key setting' because they physically determine the 'final density'
# We REMOVE 'Density change' to avoid data leakage (it contains the target value)
features = [
    'Color', 
    'Paper type', 
    'Ink key zero setting', 
    'Delta E improvement', 
    'initial density', 
    'final ink key setting'
]
X = data[features]
y = data['final density']  # <--- TARGET VARIABLE CHANGED TO DENSITY

# 5. K-Fold Cross Validation Setup (K=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'mse': 'neg_mean_squared_error'
}

# Run the Cross Validation
cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring)

# 6. Display Performance Metrics
print("--- K-Fold Cross Validation: DENSITY PREDICTION (K=5) ---")
for i in range(5):
    print(f"Fold {i+1}:")
    print(f"   R2 Score: {cv_results['test_r2'][i]:.4f}")
    print(f"   MAE: {-cv_results['test_mae'][i]:.4f}")

print("\n--- Summary Statistics ---")
print(f"Avg R2 Score: {np.mean(cv_results['test_r2']):.4f} (+/- {np.std(cv_results['test_r2']):.4f})")
print(f"Avg Mean Absolute Error (MAE): {-np.mean(cv_results['test_mae']):.4f}")
print(f"Avg Root Mean Squared Error (RMSE): {np.sqrt(-np.mean(cv_results['test_mse'])):.4f}")

# 7. Final Training and Save
model.fit(X, y)
dump(model, 'density_prediction_model.pkl')
print("\n✅ Final density model saved as 'density_prediction_model.pkl'")

# Sample Prediction Test
# Inputs: [Color, Paper, Zero, DE_Improvement, Init_Dens, Final_Key]
# Example: Cyan, Coated, 0mm, Improvement of 2.83, Starting at 1.13, Final Key at 48.00
sample_input = np.array([[0, 0, 0, 2.83, 1.13, 48.00]])
sample_prediction = model.predict(sample_input)
print(f"\nPredicted Final Density: {sample_prediction[0]:.3f}")"""




"""import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump

# 1. Load the datasets
data = pd.read_excel('real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')

# 2. Preprocessing & Harmonization
for df in [fivemmdata, ed1, ed2]:
    df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after'}, inplace=True)

data = pd.concat([data, fivemmdata, ed1, ed2], ignore_index=True).dropna()

# 3. Feature Engineering
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)

# 4. Multi-Output Selection
# Features: The "Initial State" + the "Action" (Final Key)
features = ['Color', 'Paper type', 'Ink key zero setting', 'Delta E before', 'initial density', 'final ink key setting']
X = data[features]

# Targets: We want to predict both the resulting Density and the resulting Delta E
y = data[['final density', 'Delta E after']]

# 5. Multi-Output K-Fold Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# MultiOutputRegressor allows one model to predict two things at once
model = MultiOutputRegressor(LinearRegression())

scoring = ['r2', 'neg_mean_absolute_error']
cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring)

# 6. Display Metrics
print("--- Multi-Output Validation (Density & Delta E) ---")
print(f"Avg R2 Score: {np.mean(cv_results['test_r2']):.4f}")
print(f"Avg MAE: {-np.mean(cv_results['test_neg_mean_absolute_error']):.4f}")

# 7. Final Training & Physics Calculation
model.fit(X, y)
dump(model, 'full_physics_predictor.pkl')

# --- TEST PREDICTION WITH PHYSICS CHECK ---
# Input: Cyan, Coated, 0mm, DE_Before=7.54, Init_Dens=1.13, Final_Key=48.00
sample_input = np.array([[0, 0, 0, 7.54, 1.13, 48.00]])
prediction = model.predict(sample_input)[0]

pred_density = prediction[0]
pred_de = prediction[1]

# User's Rule-of-Thumb Check: (Key_Change * 0.3) / 11 = Expected Density Change
# If we use your provided formula: (Key_Setting * 11) / 0.3
key_move = 48.00 - 43.63 # Example move
rule_check = (key_move * 0.3) / 11 

print("\n--- AI Virtual Press Result ---")
print(f"Predicted Final Density: {pred_density:.3f}")
print(f"Predicted Final ΔE:      {pred_de:.2f}")
print(f"Rule-based Density shift for this move: +{rule_check:.3f}")"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from joblib import dump

# 1. Load the 4 Data Sources
data = pd.read_excel('real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')
ed3=pd.read_excel('Plus real (0.3-11%).xlsx')

# 2. Harmonize & Combine
for df in [fivemmdata, ed1, ed2, ed3]:
    df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after'}, inplace=True)

data = pd.concat([data, fivemmdata, ed1, ed2], ignore_index=True).dropna()

# 3. Feature Engineering (Removing all Density inputs)
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)

# We use the Error Gap as the primary driver
data['DE_Gap'] = data['Delta E before'] - 2.5 # Assuming 2.5 is your standard target

# 4. Feature Selection (DENSITY REMOVED FROM X)
# The model only knows the Color, Paper, Machine Zero, and the Color Error
features = ['Color', 'Paper type', 'Ink key zero setting', 'DE_Gap', 'initial ink key setting']
X = data[features]
y = data['final density'] # Target for the physics formula

# 5. K-Fold Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

cv_results = cross_validate(model, X, y, cv=kf, scoring=['r2', 'neg_mean_absolute_error'])

print("--- K-Fold Validation: Color-to-Density Engine ---")
print(f"Avg Accuracy (R2): {np.mean(cv_results['test_r2']):.4f}")
print(f"Avg Density Error: {-np.mean(cv_results['test_neg_mean_absolute_error']):.4f}")

# 6. Final Train & Save
model.fit(X, y)
dump(model, 'color_to_density_model.pkl')

# --- THE PHYSICS BRIDGE TEST ---
# Input: Cyan, Coated, 0mm Zero, 7.54 Delta E (Gap of 5.04), Started at 43.6%
sample_input = np.array([[0, 0, 0, 5.04, 43.6]])
predicted_target_density = model.predict(sample_input)[0]

# APPLYING YOUR FORMULA: (Predicted Density * 11) / 0.3
final_ink_key_setting = (predicted_target_density * 11) / 0.3

print("\n--- Final Output Results ---")
print(f"AI Predicted Target Density: {predicted_target_density:.3f}")
print(f"Calculated Final Ink Key:    {final_ink_key_setting:.2f}%")
print(f"Mechanical Ratio Used:       36.67 (11 / 0.3)")