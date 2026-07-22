


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data from the CSV file
try:
    df = pd.read_csv("dataset.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'dataset.csv' file not found. Please check the file path.")
    exit()

print("\nOriginal Dataset Preview:")
print(df.head())

# Drop the Roll column since it's just an identifier, not a feature
if "Roll" in df.columns:
    df = df.drop("Roll", axis=1)

# Use One-Hot Encoding for categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features and target based on your exact CSV column name
# pd.get_dummies appends '_Yes' to the target column 'Buys_Computer'
target_col = "Buys_Computer_Yes" 

X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

# Split data (test_size = 0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
plt.title("Naive Bayes - Buys Computer Dataset")
plt.show()

print("\nClass Probabilities:")
print(model.predict_proba(X))
