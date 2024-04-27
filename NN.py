import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset
data = pd.read_csv("songs_processed.csv")

# Drop columns that are not useful for prediction
X = data.drop(columns=['popularity', 'artist'])  

# Target variable
y = data['popularity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that standardizes the data then creates a MLPRegressor
pipeline = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
)

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Plot actual vs predicted popularity
# Plot actual popularity in red
plt.scatter(y_test, y_test, color='red', alpha=0.5, label='Actual Popularity')
# Plot predicted popularity in blue
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted Popularity')

plt.xlabel("Actual Popularity")
plt.ylabel("Popularity Score")
plt.title("Comparison of Actual vs Predicted Popularity with Neural Network")
plt.legend()
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color='green')  # Line of best fit

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Output metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

plt.show()
