
# ONLINE PAYMENT FRAUD DETECTION
# Algorithm: Logistic Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# 1. Load the dataset
df = pd.read_csv("paypal_fraud_synthetic.csv")

print("DATASET")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# 2. Select features and target
features = [
    "amount_usd",
    "payment_method",
    "login_country",
    "time_since_last_login_min",
    "device_type"
]

X = df[features]
y = df["is_fraud"]

print("\nTarget distribution:")
print(y.value_counts())

# 3. Separate numerical and categorical features
numeric_features = [
    "amount_usd",
    "time_since_last_login_min"
]

categorical_features = [
    "payment_method",
    "login_country",
    "device_type"
]

# 4. Preprocess the data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Create the Logistic Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# 6. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

# 8. Predict the test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 9. Evaluate the model
print("\nMODEL EVALUATION")
print("-----------------------------")

print("Accuracy:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("Precision:",
      round(precision_score(y_test, y_pred, zero_division=0) * 100, 2), "%")

print("Recall:",
      round(recall_score(y_test, y_pred, zero_division=0) * 100, 2), "%")

print("F1 Score:",
      round(f1_score(y_test, y_pred, zero_division=0) * 100, 2), "%")

print("ROC-AUC:",
      round(roc_auc_score(y_test, y_prob), 3))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

# 10. Display the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Confusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
).plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# 11. Display the ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)

plt.title("Logistic Regression - ROC Curve")
plt.show()

# 12. Predict a sample transaction
sample = pd.DataFrame([{
    "amount_usd": 250.0,
    "payment_method": "Credit Card",
    "login_country": "USA",
    "time_since_last_login_min": 30.0,
    "device_type": "Mobile"
}])

prediction = model.predict(sample)[0]
fraud_probability = model.predict_proba(sample)[0][1]

print("\nSAMPLE TRANSACTION RESULT")
print("-----------------------------")
print("Fraud probability:",
      round(fraud_probability * 100, 2), "%")

if prediction == 1:
    print("Prediction: Potential Fraud")
else:
    print("Prediction: Legitimate Transaction")

