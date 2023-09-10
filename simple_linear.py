import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate or load your dataset (replace this with your own data)
data = {
    'Bedrooms': [2, 3, 4, 2, 3, 4, 3, 4, 5, 5],
    'SquareFootage': [1200, 1500, 1800, 1000, 1400, 1600, 1300, 1700, 2000, 2100],
    'AgeOfHouse': [5, 10, 15, 2, 8, 12, 6, 11, 16, 18],
    'Price': [220000, 300000, 350000, 200000, 280000, 320000, 290000, 340000, 400000, 420000]
}

df = pd.DataFrame(data)

# Split the data into training and testing sets
X = df[['Bedrooms', 'SquareFootage', 'AgeOfHouse']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Predict the price of a new house
new_house = np.array([[3, 1700, 7]])  # Replace with the features of the new house
predicted_price = model.predict(new_house)
print(f"Predicted Price for the New House: ${predicted_price[0]:,.2f}")
