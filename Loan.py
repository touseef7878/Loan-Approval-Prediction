# Task 4: Loan Approval Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# Step 1: Load Dataset
# Load the dataset
df = pd.read_csv("loan_data_set.csv")

# Step 2: Preprocessing
# Handle missing values
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

# Fill missing categorical values with mode
for col in ["Gender","Married","Dependents","Self_Employed"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical features
categorical_cols = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area","Loan_Status"]
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features & Target
X = df.drop(columns=["Loan_ID","Loan_Status"])
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Step 3: Train Models
# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Step 4: Evaluation
# Create images directory if it doesn't exist
# Ensure images folder exists
os.makedirs("images", exist_ok=True)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n📊 {model_name} Classification Report")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Approved","Approved"],
                yticklabels=["Not Approved","Approved"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_confusion_matrix.png")
    plt.close()

# Evaluate both models
evaluate_model(y_test, y_pred_log, "LogisticRegression")
evaluate_model(y_test, y_pred_tree, "DecisionTree")

# ROC Curve Comparison
plt.figure(figsize=(6,5))
RocCurveDisplay.from_estimator(log_model, X_test, y_test, name="Logistic Regression", ax=plt.gca())
RocCurveDisplay.from_estimator(tree_model, X_test, y_test, name="Decision Tree", ax=plt.gca())
plt.title("ROC Curve Comparison")
plt.savefig("images/ROC_Curve_Comparison.png")
plt.close()
