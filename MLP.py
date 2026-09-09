from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
data = load_breast_cancer()

# 1. Separate features and target variable
X = data.data
y = data.target

# 2. Divide dataset into training and testing (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Perform scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Construct an MLP (Input -> 64 hidden -> 32 hidden -> Output) & Train it
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), 
    activation='relu', 
    solver='adam', 
    max_iter=500, 
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# 5. Predict classes for testing data
y_pred = mlp.predict(X_test_scaled)

# 6. Evaluate the trained model & display classification results
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=data.target_names)

print("=== MODEL EVALUATION RESULTS ===")
print(f"Overall Accuracy: {accuracy:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
