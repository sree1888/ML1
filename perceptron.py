class PerceptronFromScratch:
    def __init__(self, lr=1.0, epochs=20):
        self.lr, self.epochs = lr, epochs
        self.w, self.b = [], 0.0

    def predict(self, x):
        total = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return 1 if total >= 0 else -1  # Sign function activation

    def fit(self, X, y):
        self.w = [0.0] * len(X[0])  # Dynamic size based on file inputs
        for _ in range(self.epochs):
            for x_i, y_i in zip(X, y):
                error = y_i - self.predict(x_i)
                if error != 0:
                    self.w = [wj + self.lr * error * xi for wj, xi in zip(self.w, x_i)]
                    self.b += self.lr * error

# === AUTOMATIC FILE LOADING ===
X, y = [], []
with open("dataset.txt", "r") as file:
    for line in file:
        cleaned_line = line.strip()
        if cleaned_line:
            row = [float(val) for val in cleaned_line.split(",")]
            X.append(row[:-1])      # Features
            y.append(int(row[-1]))   # Label (-1 or 1)

# === TRAIN AND DISPLAY ===
model = PerceptronFromScratch()
model.fit(X, y)

print("=== FROM SCRATCH OUTPUT ===")
print(f"Weights: {model.w} | Bias: {model.b}\n")
for x_i, y_i in zip(X, y):
    pred = model.predict(x_i)
    print(f"Input: {x_i} | Target: {y_i:2d} | Predicted: {pred:2d} | {'✓' if pred == y_i else '✗'}")
