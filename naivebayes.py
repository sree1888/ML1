import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_any_dataset():
    # 1. Ask the user for the dataset file path
    file_path = input("Enter the path/name of your dataset file (e.g., data.csv): ").strip()
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found in this folder!")
        return

    # 2. Load the dataset automatically based on extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file format. Please use a .csv or .xlsx file.")
        return

    print("\n--- Dataset Columns Loaded ---")
    print(list(df.columns))

    # 3. Ask the user which column they want to predict
    target_column = input("\nWhich column is your target (Y) variable?: ").strip()
    
    if target_column not in df.columns:
        print(f"Error: '{target_column}' is not a valid column in this dataset!")
        return

    # 4. Separate Features (X) and Target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 5. Automatically encode target variable if it contains text
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    # 6. Automatically detect and handle data types
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # 7. Fill missing values automatically (Mean for numbers, Mode for text)
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].mean())
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")

    # 8. Encode all text columns to numbers dynamically
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))

    # 9. Scale numeric columns if they exist
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

    # 10. Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42
    )

    # 11. Train Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 12. Print results to terminal
    print("\n================== RESULTS ==================")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 13. Plot Confusion Matrix
    print("\nDisplaying Confusion Matrix... (Close the chart window to finish the script)")
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    plt.title(f"Naive Bayes - {target_column}")
    plt.show()

# This triggers the prompt when you run the file
if __name__ == "__main__":
    evaluate_any_dataset()
