# -PRODIGY_ML_01-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a synthetic dataset (you should replace this with your own dataset)
data = {
    'SquareFootage': [1000, 1500, 1200, 1700, 800],
    'Bedrooms': [2, 3, 2, 4, 1],
    'Bathrooms': [1, 2, 1, 3, 1],
    'Price': [200000, 300000, 240000, 350000, 180000]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Split the data into training and testing sets
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Predict the price of a new house
new_house = np.array([[1500, 3, 2]])  # Replace with the features of your new house
predicted_price = model.predict(new_house)
print(f'Predicted Price for New House: ${predicted_price[0]:,.2f}')

