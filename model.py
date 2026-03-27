"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load


# Load the dataset
data = pd.read_excel('./real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('./Sample data for 5mm (34_).xlsx')
ed1=pd.read_excel('./Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2=pd.read_excel('./Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')
ed3=pd.read_excel('./Plus real (0.3-11%).xlsx')
data = pd.concat([data, fivemmdata,ed1,ed2,ed3], ignore_index=True)
print(data.head())
data = data.dropna()  # Drop rows with missing values
data['Density change'] = data['final density'] - data['initial density']
data['Color']= data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']
x = data[['Color', 'Paper type','Ink key zero setting','Delta E improvement', 'Density change', 'initial ink key setting']]
y = data['final ink key setting']
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Create and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)
print(f'Model score: {model.score(x_test, y_test)} ')
# Make predictions on the test set
y_pred = model.predict(x_test)
# Print the score of the model
print(f'Model score: {model.score(x_test, y_test)} ')
# Print the mean squared error
print(f'Mean squared error: {mean_squared_error(y_test, y_pred)} ')
# Print the R2 score
print(f'R2 score: {r2_score(y_test, y_pred)} ')

# sample prediction
sample_data = np.array([[0, 0,0, (7.54-4.71), (1.37-1.13), 43.63],[0, 0, 5, (5.2-2.4), (1.37-1.13), 76.84]])  # Replace with actual values
sample_prediction = model.predict(sample_data)
print(f'Sample prediction: {sample_prediction[0]}, 5mm Sample prediction: {sample_prediction[1]}')

# Save the model to a file
dump(model, 'ink_key_setting_model.pkl')"""
"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# 1. Load all 5 datasets
data = pd.read_excel('./real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('./Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('./Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('./Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')
ed3 = pd.read_excel('./Plus real (0.3-11%).xlsx')

# 2. Harmonization (Rename columns for consistency)
for df in [fivemmdata, ed1, ed2, ed3]:
    df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after'}, inplace=True)

# 3. Combine and Clean
data = pd.concat([data, fivemmdata, ed1, ed2, ed3], ignore_index=True)
data = data.dropna()

# 4. Feature Engineering (Keeping your original logic)
data['Density change'] = data['final density'] - data['initial density']
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

# 5. Feature Selection
# Inputs (x) remain the same as your original snippet
x = data[['Color', 'Paper type', 'Ink key zero setting', 'Delta E improvement', 'Density change', 'initial ink key setting']]

# Target variable (y) is now Final Density
y = data['final density']

# 6. Training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

# 7. Evaluation
y_pred = model.predict(x_test)
print(f'Model R2 Score: {r2_score(y_test, y_pred):.4f}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}')

# 8. Sample Prediction
# [Color, Paper, Zero, DE_Improvement, Density_Change, Init_Key]
sample_data = np.array([[0, 0, 0, (7.54-4.71), (1.37-1.13), 43.63]])
sample_prediction = model.predict(sample_data)
print(f'\nSample Prediction (Final Density): {sample_prediction[0]:.3f}')

# 9. Save
dump(model, 'final_density_model.pkl')"""
"""import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump

# 1. Load all 5 datasets
data = pd.read_excel('./real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('./Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('./Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('./Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')
ed3 = pd.read_excel('./Plus real (0.3-11%).xlsx')

# 2. Harmonization (Rename columns for consistency)
for df in [fivemmdata, ed1, ed2, ed3]:
    df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after'}, inplace=True)

# 3. Combine and Clean
data = pd.concat([data, fivemmdata, ed1, ed2, ed3], ignore_index=True)
data = data.dropna()

# 4. Feature Engineering
data['Density change'] = data['final density'] - data['initial density']
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

# 5. Feature Selection
# Features (x) and Target (y)
x = data[['Color', 'Paper type', 'Ink key zero setting', 'Delta E improvement', 'Density change', 'initial ink key setting']]
y = data['final density']

# 6. Training and Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 7. Initialize and Train XGBoost Regressor
# Hyperparameters tuned for small-to-medium tabular datasets
model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    objective='reg:squarederror',
    random_state=42
)

model.fit(x_train, y_train)

# 8. Evaluation
y_pred = model.predict(x_test)
print(f"--- XGBoost Model Performance ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# 9. Sample Prediction
# [Color, Paper, Zero, DE_Improvement, Density_Change, Init_Key]
sample_data = np.array([[0, 0, 0, (7.54-4.71), (1.37-1.13), 43.63]])
sample_prediction = model.predict(sample_data)
print(f"\nSample Prediction (Final Density): {sample_prediction[0]:.3f}")

# 10. Save the XGBoost model
dump(model, 'final_density_xgboost_model.pkl')
print("\n✅ XGBoost model saved successfully!")"""

"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump

# 1. Load all 5 datasets
data = pd.read_excel('./real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('./Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('./Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('./Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')
ed3 = pd.read_excel('./Plus real (0.3-11%).xlsx')

# 2. Harmonization (Standardize column names)
for df in [fivemmdata, ed1, ed2, ed3]:
    df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after'}, inplace=True)

# 3. Combine and Clean
data = pd.concat([data, fivemmdata, ed1, ed2, ed3], ignore_index=True)
data = data.dropna()

# 4. Feature Engineering
data['Density change'] = data['final density'] - data['initial density']
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

# 5. Feature Selection
# Features (x) and Target (y = Final Density)
x = data[['Color', 'Paper type', 'Ink key zero setting', 'Delta E improvement', 'Density change', 'initial ink key setting']]
y = data['final density']

# 6. Training and Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 7. Initialize and Train Gradient Boosting Regressor
# max_depth=3 is standard to prevent the model from memorizing the data (over-fitting)
model = GradientBoostingRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)

model.fit(x_train, y_train)

# 8. Evaluation
y_pred = model.predict(x_test)
print(f"--- Gradient Boosting Performance ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# 9. Sample Prediction
# Inputs: [Color, Paper, Zero, DE_Improvement, Density_Change, Init_Key]
sample_data = np.array([[0, 0, 0, (7.54-4.71), (1.37-1.13), 43.63]])
sample_prediction = model.predict(sample_data)
print(f"\nSample Prediction (Final Density): {sample_prediction[0]:.3f}")

# 10. Save the Gradient Boosting model
dump(model, 'final_density_gb_model.pkl')
print("\n✅ Gradient Boosting model saved successfully!")"""

"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump

# 1. Load all 5 datasets
data = pd.read_excel('./real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('./Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('./Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('./Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')
ed3 = pd.read_excel('./Plus real (0.3-11%).xlsx')

# 2. Harmonization
for df in [fivemmdata, ed1, ed2, ed3]:
    df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after'}, inplace=True)

# 3. Combine and Clean
data = pd.concat([data, fivemmdata, ed1, ed2, ed3], ignore_index=True)
data = data.dropna()

# 4. Feature Engineering
data['Density change'] = data['final density'] - data['initial density']
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

# 5. Feature Selection
x = data[['Color', 'Paper type', 'Ink key zero setting', 'Delta E improvement', 'Density change', 'initial ink key setting']]
y = data['final density']

# 6. Training and Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 7. Define the Parameter Grid
# This tells GridSearchCV which combinations to test
param_grid = {
    'n_estimators': [100, 200, 300],      # Number of trees
    'learning_rate': [0.01, 0.05, 0.1],   # Step size shrinkage
    'max_depth': [3, 4, 5],               # Depth of each tree
    'min_samples_split': [2, 5, 10]       # Min samples required to split a node
}

# 8. Initialize GridSearchCV
gb = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=gb, 
    param_grid=param_grid, 
    cv=5,                 # 5-Fold Cross Validation
    n_jobs=-1,            # Use all available CPU cores
    scoring='r2',
    verbose=1             # Shows progress during tuning
)

print("--- Starting Hyperparameter Tuning ---")
grid_search.fit(x_train, y_train)

# 9. Get the Best Model
best_model = grid_search.best_estimator_

print("\n--- Best Parameters Found ---")
print(grid_search.best_params_)

# 10. Evaluation
y_pred = best_model.predict(x_test)
print(f"\nTuned Model Performance:")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")

# 11. Save the Optimized Model
dump(best_model, 'optimized_density_gb_model.pkl')
print("\n✅ Optimized Gradient Boosting model saved!")"""

"""import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump

# 1. Load all 5 datasets
data = pd.read_excel('./real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('./Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('./Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('./Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')
ed3 = pd.read_excel('./Plus real (0.3-11%).xlsx')

# 2. Harmonization (Standardize column names)
for df in [fivemmdata, ed1, ed2, ed3]:
    df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after'}, inplace=True)

# 3. Combine and Clean
data = pd.concat([data, fivemmdata, ed1, ed2, ed3], ignore_index=True)
data = data.dropna()

# 4. Feature Engineering
data['Density change'] = data['final density'] - data['initial density']
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

# 5. Feature Selection
# X = Inputs, y = Final Ink Key Setting
features = ['Color', 'Paper type', 'Ink key zero setting', 'Delta E improvement', 'Density change', 'initial ink key setting']
X = data[features]
y = data['final ink key setting']  # <--- Target parameter set to Final Ink Key

# 6. Training and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Define the Parameter Grid for XGBoost
# Tuning these parameters helps prevent "overfitting" on specific press jobs
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 8. Initialize GridSearchCV
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5, 
    scoring='r2',
    n_jobs=-1, 
    verbose=1
)

print("--- Starting XGBoost Hyperparameter Tuning ---")
grid_search.fit(X_train, y_train)

# 9. Evaluate the Best Model
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)

print("\n--- Best XGBoost Parameters ---")
print(grid_search.best_params_)

print(f"\n--- Optimized Model Performance ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# 10. Sample Prediction
# [Color, Paper, Zero, DE_Improvement, Density_Change, Init_Key]
sample_input = np.array([[0, 0, 0, 2.83, 0.24, 43.63]])
prediction = best_xgb.predict(sample_input)
print(f"\nPredicted Final Ink Key Setting: {prediction[0]:.2f}%")

# 11. Save the model
dump(best_xgb, 'optimized_ink_key_xgb_model.pkl')"""
"""import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump

# 1. Load all 5 datasets
data = pd.read_excel('./real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('./Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('./Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('./Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')
ed3 = pd.read_excel('./Plus real (0.3-11%).xlsx')

# 2. Harmonization (Rename columns for consistency)
for df in [fivemmdata, ed1, ed2, ed3]:
    df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after'}, inplace=True)

# 3. Combine and Clean
data = pd.concat([data, fivemmdata, ed1, ed2, ed3], ignore_index=True)
data = data.dropna()

# 4. Feature Engineering
data['Density change'] = data['final density'] - data['initial density']
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

# 5. Feature Selection
# X = Inputs, y = Final Density
features = ['Color', 'Paper type', 'Ink key zero setting', 'Delta E improvement', 'Density change', 'initial ink key setting']
X = data[features]
y = data['final density']  # <--- Target parameter set to Final Density

# 6. Training and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Define the Parameter Grid for Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# 8. Initialize and Run GridSearchCV
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5, 
    scoring='r2',
    n_jobs=-1, 
    verbose=1
)

print("--- Tuning XGBoost for Density Prediction ---")
grid_search.fit(X_train, y_train)

# 9. Evaluate the Best Model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n--- Best Parameters Found ---")
print(grid_search.best_params_)

print(f"\n--- Optimized Model Performance ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# 10. Sample Prediction
# [Color, Paper, Zero, DE_Improvement, Density_Change, Init_Key]
sample_input = np.array([[3, 0, 0.0, 4.15, 48.03]])
prediction = best_model.predict(sample_input)
print(f"\nPredicted Final Density: {prediction[0]:.3f}")

# 11. Save the model
dump(best_model, 'optimized_density_xgboost_model.pkl')"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump

# 1. Load and Harmonize (Consolidated)
data_files = [
    './real job dataset ( pakka wala ).xlsx',
    './Sample data for 5mm (34_).xlsx',
    './Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx',
    './Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx',
    './Plus real (0.3-11%).xlsx'
]

dfs = []
for f in data_files:
    temp_df = pd.read_excel(f)
    temp_df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after'}, inplace=True)
    dfs.append(temp_df)

data = pd.concat(dfs, ignore_index=True).dropna()

# 2. Feature Engineering
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['DE_Improvement'] = data['Delta E before'] - data['Delta E after']

# 3. Corrected Feature Selection (Removing Leakage, Adding Initial Density)
# Inputs available BEFORE the move
features = ['Color', 'Paper type', 'Ink key zero setting', 'DE_Improvement', 'initial density', 'initial ink key setting']
X = data[features]
y = data['final density'] 

# 4. GridSearchCV Tuning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 5. Zone 19 Specific Prediction
# Mapping: Black(3), Coated(0), Zero(0.0), DE_Gap(4.15), Init_Dens(1.31), Init_Key(48.03)
zone_19_input = np.array([[3, 0, 0.0, 4.15, 1.31, 48.03]])
pred_density = best_model.predict(zone_19_input)[0]

# 6. Apply your mechanical formula to find the suggested key
# Formula: (Density * 11) / 0.3
final_key_suggestion = (pred_density * 11) / 0.3

print(f"--- Optimized Results for Zone 19 ---")
print(f"Predicted Final Density: {pred_density:.3f}")
print(f"Suggested Ink Key Setting: {final_key_suggestion:.2f}%")

dump(best_model, 'optimized_density_xgboost_model.pkl')