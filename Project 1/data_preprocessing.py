import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('C:/Users/Acer/Documents/ITDAA4/Assignments/Project 1/heart.csv', delimiter=';')

# Data preprocessing steps (e.g., handle missing values, encode categorical variables)
...

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data (optional)
X_train.to_csv('preprocessed_data.csv', index=False)
