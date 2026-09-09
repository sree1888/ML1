import numpy as np
from sklearn.linear_model import Perceptron

# === AUTOMATIC FILE LOADING ===
data = np.loadtxt("dataset.txt", delimiter=",")
X = data[:, :-1]
y = data[:, -1]

# === TRAIN AND DISPLAY ===
model = Perceptron(max_iter=100)
model.fit(X, y)

print("=== BUILT-IN LIBRARY OUTPUT ===")
print(f"Weights: {model.coef_[0]} | Bias: {model.intercept_[0]}\n")
for x_sample, y_sample in zip(X, y):
    pred = model.predict([x_sample])[0]
    print(f"Input: {list(x_sample)} | Target: {int(y_sample):2d} | Predicted: {int(pred):2d} | {'✓' if pred == y_sample else '✗'}")
