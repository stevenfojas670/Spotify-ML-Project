import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#Load the dataset
songs_data = pd.read_csv('songs_processed.csv')

# Select features and target variable
features = ['duration_ms', 'year', 'danceability', 'energy', 'loudness', 
            'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo']  
target = 'popularity'

# Prepare the data
X = songs_data[features]
y = songs_data[target]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=64)

# Create and train the Random Forest model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Predicting the test set results
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Output results
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)

# Feature Importance
feature_importance = pd.DataFrame({'Feature': features, 'Importance': random_forest_model.feature_importances_}).sort_values(by='Importance', ascending=False)
print(feature_importance)
