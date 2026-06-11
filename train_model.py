# train_model.py
# Trains the expiry risk classifier and saves it to model.pkl
# Run once with: python train_model.py

import random
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------
# STEP 1: Build the constructed sample dataset
# (section 1.5 of the document: testing used sample datasets)
#
# Each training example is one product, described by one feature:
# days_to_expiry. The label is the correct risk category.
# ---------------------------------------------------------------

def label_for(days_to_expiry):
    # The ground-truth rules the model will learn from.
    # Negative days means the expiry date is already in the past.
    if days_to_expiry < 0:
        return "expired"
    elif days_to_expiry <= 7:
        return "critical"
    elif days_to_expiry <= 30:
        return "warning"
    else:
        return "safe"

# Generate 2000 example products with days_to_expiry spread
# between 60 days already expired and one year of shelf life
features = []   # the inputs,  e.g. [[-12], [3], [25], [180], ...]
labels = []     # the answers, e.g. ["expired", "critical", ...]

for i in range(2000):
    days = random.randint(-60, 365)
    features.append([days])          # a list inside a list, scikit-learn requires it
    labels.append(label_for(days))

# ---------------------------------------------------------------
# STEP 2: Split the data, 80% for training, 20% for testing.
# The model never sees the test portion while learning, so the
# accuracy measured on it is honest.
# ---------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# STEP 3: Train the Decision Tree classifier
# ---------------------------------------------------------------

model = DecisionTreeClassifier()
model.fit(X_train, y_train)   # fit() means "learn from this data"

# ---------------------------------------------------------------
# STEP 4: Evaluate on the unseen 20%
# ---------------------------------------------------------------

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model trained. Accuracy on unseen test data: {accuracy * 100:.2f}%")

# ---------------------------------------------------------------
# STEP 5: Save the trained model to a file for the web service
# ---------------------------------------------------------------

joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")