# Evaluates a neural network model using TensorFlow and Keras to predict the popularity of songs 

''' 
Requirements:
    pip install tensorflow
    pip install keras-tuner
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Load the dataset
data = pd.read_csv("songs_processed.csv")

# Define the  popularity threshold below are considered outliers
popularity_threshold = 10

# Filter out songs with low popularity
filtered_data = data[data['popularity'] >= popularity_threshold]

# Drop non-numeric columns and columns that are not useful for prediction
filtered_data['explicit'] = filtered_data['explicit'].astype(int)
X = filtered_data.drop(columns=['popularity', 'artist'])

# Target variable
y = filtered_data['popularity'].values.reshape(-1, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Define a neural network model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.1),
    Dense(1, activation='linear')
])

# Compile the model with mean squared error loss and the Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model on the test data
mse = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calculate the performance metrics
mse_value = mean_squared_error(y_test, y_pred)
r2_value = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse_value}")
print(f"R^2 Score: {r2_value}")

# Plotting actual vs predicted popularity
plt.scatter(y_test, y_pred, label='Data points')
plt.plot(y_test, y_test, color='red', label='Actual')  # The line shows perfect predictions
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("NN: Actual vs Predicted Popularity")
plt.legend()
plt.show()

# Plotting training and validation loss over epochs
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
