import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load


# Load the dataset
data = pd.read_excel('./real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('./Sample data for 5mm (34_).xlsx')
data = pd.concat([data, fivemmdata], ignore_index=True)
print(data.head())
data = data.dropna()  # Drop rows with missing values
data['Density change'] = data['final density'] - data['initial density']
data['Color']= data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].str.replace("mm","",regex=False).astype(int)
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
dump(model, 'ink_key_setting_model.pkl')