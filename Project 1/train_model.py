# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the preprocessed dataset
file_path = 'C:/Users/Acer/Documents/ITDAA4/Assignments/Project 1/heart.csv'  # Update with your actual file path
df = pd.read_csv(file_path, sep=';')

# Perform any necessary preprocessing steps
# For example, converting categorical variables to numeric, handling missing values, scaling, etc.

# Split data into features and target
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (example with Random Forest Classifier)
model = RandomForestClassifier(random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model to a file
model_file_path = 'C:/Users/Acer/Documents/ITDAA4/Assignments/Project 1/heart_disease_model.pkl'
joblib.dump(model, model_file_path)

print(f"Model saved to {model_file_path}")
