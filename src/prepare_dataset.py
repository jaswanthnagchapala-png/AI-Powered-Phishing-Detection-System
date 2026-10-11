
import pandas as pd
from pathlib import Path
from features import extract_features

DATASET_PATH = Path(
    "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
)
OUTPUT_PATH = Path("data/processed/cleaned_dataset.csv")


def prepare_dataset():
    if not DATASET_PATH.exists():
        print("Dataset not found:", DATASET_PATH.resolve())
        return

    df = pd.read_csv(DATASET_PATH)

    # Keep only the required columns
    df = df[["URL", "label"]].copy()

    # Remove missing and empty URLs
    df = df.dropna(subset=["URL", "label"])
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""]

    # Check for URLs assigned conflicting labels
    conflicts = df.groupby("URL")["label"].nunique()
    conflicting_urls = conflicts[conflicts > 1].index

    if len(conflicting_urls) > 0:
        print("Conflicting URL labels found:", len(conflicting_urls))
        df = df[~df["URL"].isin(conflicting_urls)]

    # Remove repeated URLs to reduce data leakage
    df = df.drop_duplicates(subset=["URL"], keep="first")

    # UCI: label 0 = phishing, label 1 = legitimate
    df["is_phishing"] = (df["label"] == 0).astype(int)

    # Extract features using the same function as the app
    records = []
    failed = 0

    for url, target in zip(df["URL"], df["is_phishing"]):
        try:
            features = extract_features(url)

            if not isinstance(features, dict) or not features:
                failed += 1
                continue

            records.append({
                **features,
                "is_phishing": target
            })

        except Exception as error:
            failed += 1

    if not records:
        print("No records were successfully processed.")
        return

    processed = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUTPUT_PATH, index=False)

    print("\nDATASET PREPARATION COMPLETE")
    print("Original rows:", len(pd.read_csv(DATASET_PATH, usecols=["URL"])))
    print("Unique URLs considered:", len(df))
    print("Processed rows:", len(processed))
    print("Failed rows:", failed)
    print("\nTarget distribution:")
    print(processed["is_phishing"].value_counts().rename(
        index={0: "Legitimate", 1: "Phishing"}
    ))
    print("\nSaved to:", OUTPUT_PATH.resolve())
    print("\nFeature columns:")
    print([c for c in processed.columns if c != "is_phishing"])


if __name__ == "__main__":
    prepare_dataset()
