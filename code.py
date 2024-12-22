import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#pip install pandas numpy scikit-learn matplotlib 
# Load the dataset
data = pd.read_csv(r"C:\Users\tanu1\Desktop\Project learning\gold_price_prediction\test_cases.csv")

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature Engineering
data['Day'] = data.index.day
data['Month'] = data.index.month
data['Year'] = data.index.year
data['Day_of_Week'] = data.index.dayofweek

# Select features and target
features = data[['Day', 'Month', 'Year', 'Day_of_Week']]
target = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot predictions vs actual prices
plt.scatter(y_test, predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Gold Price Prediction')
plt.show()
