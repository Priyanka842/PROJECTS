# Import required libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("lfw_arnie_nonarnie.csv")

# Separate features and target
X = df.drop("Label", axis=1)
y = df["Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=21,
    stratify=y
)

# Build models
models = {
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),

    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "random_forest": Pipeline([
        ("model", RandomForestClassifier(random_state=21))
    ])
}

# Hyperparameter grids
param_grids = {
    "svm": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"]
    },

    "logistic_regression": {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100]
    },

    "random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10]
    }
}

# Cross-validation setup
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=21
)

# Find best model
best_model_name = None
best_model_info = None
best_model_cv_score = 0

for name, pipeline in models.items():

    print(f"Training {name}...")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print(f"Best CV Score: {grid.best_score_:.4f}")

    if grid.best_score_ > best_model_cv_score:
        best_model_name = name
        best_model_info = grid.best_params_
        best_model_cv_score = grid.best_score_
        best_model = grid.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Required project variable
score = accuracy

# Results
print("\n" + "=" * 50)
print("BEST MODEL RESULTS")
print("=" * 50)
print("Best Model:", best_model_name)
print("Best Parameters:", best_model_info)
print("Best CV Score:", round(best_model_cv_score, 4))
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

