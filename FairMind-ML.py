from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset with gender bias (using lists instead of pandas)
data = [
    {'Gender': 'Male', 'Experience': 5, 'Hired': 1},
    {'Gender': 'Male', 'Experience': 7, 'Hired': 1},
    {'Gender': 'Male', 'Experience': 8, 'Hired': 1},
    {'Gender': 'Female', 'Experience': 6, 'Hired': 0},
    {'Gender': 'Female', 'Experience': 2, 'Hired': 0},
    {'Gender': 'Female', 'Experience': 1, 'Hired': 0},
    {'Gender': 'Female', 'Experience': 3, 'Hired': 0},
    {'Gender': 'Male', 'Experience': 10, 'Hired': 1}
]

# Convert gender to numeric (Male=0, Female=1)
for entry in data:
    entry['Gender'] = 0 if entry['Gender'] == 'Male' else 1

# Manually balancing the dataset (add one extra female with experience 6)
balanced_data = data + [{'Gender': 1, 'Experience': 6, 'Hired': 1}]

# Extract features (X) and target (y)
X = [[entry['Gender'], entry['Experience']] for entry in balanced_data]
y = [entry['Hired'] for entry in balanced_data]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Output results
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")