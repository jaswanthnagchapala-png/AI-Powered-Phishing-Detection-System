
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DATA_PATH = Path("data/processed/cleaned_dataset.csv")
MODEL_PATH = Path("models/best_model.joblib")


def train_models():
    if not DATA_PATH.exists():
        print("Dataset not found:", DATA_PATH.resolve())
        return

    # Load processed dataset
    df = pd.read_csv(DATA_PATH)

    target = "is_phishing"
    X = df.drop(columns=[target])
    y = df[target]

    # Verify input features
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        print("Error: All features must be numeric.")
        return

    if X.isnull().any().any() or not X.map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all().all():
        print("Error: Missing or infinite feature values found.")
        return

    if set(y.unique()) != {0, 1}:
        print("Error: Expected labels 0 (legitimate) and 1 (phishing).")
        return

    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    best_model = None
    best_name = None
    best_f1 = -1

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4)
        })

        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Phishing Recall:", round(recall, 4))
        print("Phishing F1-Score:", round(f1, 4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(pd.DataFrame(results).to_string(index=False))

    # Save the best model and its feature order
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": best_model,
        "feature_names": X.columns.tolist(),
        "model_name": best_name,
        "target": target
    }, MODEL_PATH)

    print("\nBest model:", best_name)
    print("Saved to:", MODEL_PATH.resolve())

    print("\nDetailed classification report:")
    print(classification_report(
        y_test,
        best_model.predict(X_test),
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))


if __name__ == "__main__":
    train_models()
