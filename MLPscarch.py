import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# 1. Setup Data
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = y.reshape(-1, 1)

# 2. Weight & Bias Initialization (30 inputs -> 10 hidden -> 1 output)
np.random.seed(42)
W1, b1 = np.random.randn(30, 10) * 0.1, np.zeros((1, 10))
W2, b2 = np.random.randn(10, 1) * 0.1, np.zeros((1, 1))

lr = 0.1
m = X.shape[0]

# 3. Iterative Training Loop (Forward, Loss, Backprop, Updates)
for epoch in range(300):
    # Forward Propagation & Activation Functions (ReLU & Sigmoid)
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)  # ReLU
    Z2 = np.dot(A1, W2) + b2
    A2 = 1 / (1 + np.exp(-np.clip(Z2, -500, 500)))  # Sigmoid

    # Loss Calculation (Binary Cross-Entropy)
    loss = -np.mean(y * np.log(A2 + 1e-15) + (1 - y) * np.log(1 - A2 + 1e-15))

    # Backpropagation (Gradient Calculation)
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0)  # Derivative of ReLU
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Weight & Bias Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 50 == 0:
        acc = np.mean((A2 >= 0.5) == y) * 100
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.2f}%")

# 4. Prediction & Final Evaluation
final_preds = (A2 >= 0.5).astype(int)
print(f"\nFinal Training Accuracy: {np.mean(final_preds == y) * 100:.2f}%")
