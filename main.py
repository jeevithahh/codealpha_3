# Handwritten Character Recognition using Random Forest

# Import libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# Load the dataset
digits = load_digits()

# Features and target
X = digits.data
y = digits.target

print("Dataset Loaded Successfully!")
print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)

# Display a sample digit
plt.figure(figsize=(4,4))
plt.imshow(digits.images[0], cmap="gray")
plt.title(f"Actual Digit: {digits.target[0]}")
plt.axis("off")
plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "handwritten_model.pkl")

print("\nModel saved as handwritten_model.pkl")

# Test prediction
sample = X_test[10].reshape(1, -1)

prediction = model.predict(sample)

print("\nPredicted Digit:", prediction[0])

# Show predicted image
plt.figure(figsize=(4,4))
plt.imshow(sample.reshape(8,8), cmap="gray")
plt.title(f"Predicted Digit: {prediction[0]}")
plt.axis("off")
plt.show()