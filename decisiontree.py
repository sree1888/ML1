import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Read dataset from CSV file
df = pd.read_csv("data.csv")

# Remove RID column (not needed)
df = df.drop("RID", axis=1)

# Encode categorical columns
le = LabelEncoder()
for c in df.columns:
    df[c] = le.fit_transform(df[c])

# Features and Target
X = df.drop("CLASS:BUYS_COMPUTER", axis=1)
y = df["CLASS:BUYS_COMPUTER"]

# Train Decision Tree
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# Plot Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["NO", "YES"],
    filled=True
)

plt.show()