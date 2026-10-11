
import pandas as pd
from pathlib import Path

# Dataset location
DATASET_PATH = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")


def check_dataset():
    if not DATASET_PATH.exists():
        print("ERROR: Dataset file not found!")
        print("Expected location:", DATASET_PATH.resolve())
        return

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print("=" * 50)
    print("PHISHING DATASET CHECK")
    print("=" * 50)

    # 1. Dataset size
    print("\n1. Dataset Shape")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # 2. Column names
    print("\n2. Column Names")
    print(df.columns.tolist())

    # 3. Missing values
    print("\n3. Missing Values")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any()
          else "No missing values found.")

    # 4. Duplicate rows
    print("\n4. Duplicate Rows")
    print("Count:", df.duplicated().sum())

    # 5. Duplicate URLs
    print("\n5. Duplicate URLs")
    if "URL" in df.columns:
        print("Count:", df["URL"].duplicated().sum())

        # Check whether identical URLs have different labels
        if "label" in df.columns:
            label_counts = df.groupby("URL")["label"].nunique()
            conflicts = label_counts[label_counts > 1]
            print("URLs with conflicting labels:", len(conflicts))
    else:
        print("URL column not found. Check the column names.")

    # 6. Label distribution
    print("\n6. Label Distribution")
    if "label" in df.columns:
        print(df["label"].value_counts(dropna=False))
    else:
        print("Label column not found. Check the column names.")

    # 7. Sample records
    print("\n7. Sample Records")
    print(df.head(5).to_string(index=False))

    print("\n" + "=" * 50)
    print("DATASET CHECK COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    check_dataset()
