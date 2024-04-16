import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# If you want to print all columns of a dataset, uncomment this
# pd.set_option('display.max_columns', None)

# Preprocessing libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler

"""

    # Run this code if for some reason songs_processed.csv is not available
    # This code will process songs_normalize and prepare it for further standardization

df = pd.read_csv('songs_normalize.csv')
df = df.drop(df.columns[1], axis=1)

# Checking if drop was successful
print(df.head().all())

# Step 1: Split the 'genre' column into lists of genres
df['genre_split'] = df['genre'].str.split(', ')

# Step 2: Explode this list into a row per genre per song
exploded_df = df.explode('genre_split')

# Step 3: Create dummy variables for each genre
dummies = pd.get_dummies(exploded_df['genre_split'])

# Step 4: Sum these dummies back to the original song level
genre_dummies = dummies.groupby(exploded_df.index).sum()

# Step 5: Concatenate these new columns back to the original DataFrame
df_modified = pd.concat([df.drop(columns=['genre', 'genre_split']), genre_dummies], axis=1)

# Step 6: Dropping set()
df_modified = df_modified.drop(columns=['set()'])

# Show the updated DataFrame structure
print(df_modified.head())

# Recreating the CSV file

# Get the absolute path to the script file being executed
script_path = os.path.abspath(__file__)

# Extract the directory path
script_dir = os.path.dirname(script_path)

# Define the CSV file path in this script's directory
file_path = os.path.join(script_dir, 'songs_processed.csv')

# Save the DataFrame to this path
df_modified.to_csv(file_path, index=False)

# Confirm the path where the file has been saved
print("File saved at:", file_path)
print("File path:", file_path)

"""

p_data = pd.read_csv('songs_processed.csv')

numerical_data = p_data.select_dtypes(include=['int64', 'float64'])
categorical_data = p_data.select_dtypes(include=['bool', 'object'])

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the numerical data
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Create a DataFrame for the scaled numerical data
scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_data.columns)

# Display the scaled numerical data
# print(scaled_numerical_df.head())

final_data = pd.concat([categorical_data, scaled_numerical_df], axis=1)
# print(final_data.head())
