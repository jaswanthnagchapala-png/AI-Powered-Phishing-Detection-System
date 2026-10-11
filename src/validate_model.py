
import pandas as pd
import numpy as np

from pathlib import Path
from urllib.parse import urlsplit

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from features import extract_features


DATA_PATH = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
RANDOM_STATE = 42


def get_hostname(url):
    """Extract a hostname without the www. prefix."""
    try:
        url = str(url).strip()

        if "://" not in url:
            url = "http://" + url

        hostname = urlsplit(url).hostname

        if not hostname:
            return None

        hostname = hostname.lower().rstrip(".")

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname or None

    except (ValueError, TypeError):
        return None


def evaluate_model(name, X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("-" * 45)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")
    print(f"Accuracy:      {accuracy_score(y_test, predictions):.4f}")
    print(
        "Precision:     "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        "Phishing recall: "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        "Phishing F1:   "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


def main():
    if not DATA_PATH.exists():
        print("Dataset not found:", DATA_PATH.resolve())
        print("Update DATA_PATH to your actual CSV location.")
        return

    df = pd.read_csv(DATA_PATH, usecols=["URL", "label"])
    df = df.dropna(subset=["URL", "label"]).copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""]

    # Remove URLs that have conflicting labels
    label_counts = df.groupby("URL")["label"].nunique()
    conflicting = label_counts[label_counts > 1].index
    df = df[~df["URL"].isin(conflicting)]

    # One record per exact URL
    df = df.drop_duplicates(subset="URL").copy()

    # Original dataset: 0 = phishing, 1 = legitimate
    df["is_phishing"] = (df["label"] == 0).astype(int)

    df["domain_group"] = df["URL"].map(get_hostname)
    df = df.dropna(subset=["domain_group"]).reset_index(drop=True)

    print("Unique URLs:", len(df))
    print("Unique hostname groups:", df["domain_group"].nunique())

    # Extract the same features used by the application
    records = []
    valid_indices = []

    for index, url in enumerate(df["URL"]):
        try:
            features = extract_features(url)

            if not isinstance(features, dict) or not features:
                continue

            values = list(features.values())

            if not all(
                isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value)
                for value in values
            ):
                continue

            records.append(features)
            valid_indices.append(index)

        except (ValueError, TypeError, KeyError):
            continue

    if not records:
        print("No valid feature records were generated.")
        return

    X = pd.DataFrame(records)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    y = valid_df["is_phishing"]
    groups = valid_df["domain_group"]

    if y.nunique() != 2:
        print("Error: Both target classes are required.")
        return

    print("Usable records:", len(X))
    print("Feature count:", X.shape[1])

    # Test A: ordinary random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    evaluate_model(
        "TEST A: Random URL split",
        X_train, X_test, y_train, y_test
    )

    # Test B: keep each hostname in only one split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    train_domains = set(groups.iloc[train_idx])
    test_domains = set(groups.iloc[test_idx])

    overlap = train_domains.intersection(test_domains)

    print("\nDomain leakage check:")
    print("Overlapping hostnames:", len(overlap))

    if overlap:
        print("ERROR: Hostnames overlap between splits.")
        return

    evaluate_model(
        "TEST B: Unseen-hostname split",
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx]
    )


if __name__ == "__main__":
    main()
