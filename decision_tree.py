# Decision Tree ML - Buys Computer Dataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# Create Dataset
data = {
    "AGE": ["y","y","M","S","S","S","M","Y","Y","S","Y","M","M","S"],
    "INCOME": ["H","H","H","M","L","L","L","M","L","M","M","M","H","M"],
    "STUDENT": ["NO","NO","NO","NO","YES","YES","YES","NO","YES","YES","YES","NO","YES","NO"],
    "CREDIT_RATING": [
        "FAIR","EXCELLENT","FAIR","FAIR",
        "FAIR","EXCELLENT","EXCELLENT","FAIR",
        "FAIR","FAIR","EXCELLENT","EXCELLENT",
        "FAIR","EXCELLENT"
    ],
    "CLASS_BUYSCOMPUTERS": [
        "NO","NO","YES","YES",
        "YES","NO","YES","NO",
        "YES","YES","YES","YES",
        "YES","NO"
    ]
}


# Convert to DataFrame
df = pd.DataFrame(data)

print("Original Dataset:")
print(df)


# Convert categorical data into numbers
encoder = LabelEncoder()

for col in df.columns:
    df[col] = encoder.fit_transform(df[col])


print("\nEncoded Dataset:")
print(df)


# Split Features and Target
X = df.drop("CLASS_BUYSCOMPUTERS", axis=1)
y = df["CLASS_BUYSCOMPUTERS"]


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)


# Create Decision Tree Model
model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=1
)


# Train model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Accuracy
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))


print("\nActual Values:")
print(list(y_test))

print("\nPredicted Values:")
print(list(y_pred))


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Display Decision Tree
plt.figure(figsize=(12,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["NO","YES"],
    filled=True
)

plt.title("Decision Tree - Buys Computer Dataset")
plt.show()
