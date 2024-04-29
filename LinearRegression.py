import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
songs_data = pd.read_csv('songs_processed.csv')

# Selecting features for the model
continuous_features = ['duration_ms', 'year', 'danceability', 'energy', 'loudness', 
                       'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                       'valence', 'tempo']
target = 'popularity'

# Preparing the data
X = songs_data[continuous_features]
y = songs_data[target]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=62)

#Train the linear regression model
linear = LinearRegression()
linear.fit(X_train, y_train)

# Predicting the test set results
y_pred = linear.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Calculate Root Mean Squared Error
r2 = r2_score(y_test, y_pred)

# Output results
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)
