
import pandas as pd
from pathlib import Path

DATASET_PATH = Path(
    "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
)

df = pd.read_csv(DATASET_PATH)

print("=" * 55)
print("DATASET INSPECTION")
print("=" * 55)

# 1. Check target labels
print("\n1. Label counts:")
print(df["label"].value_counts().sort_index())

print("\n2. Label percentages:")
print(
    (df["label"].value_counts(normalize=True) * 100)
    .round(2)
)

# 2. Check data types
print("\n3. Data types:")
print(df.dtypes.to_string())

# 3. Inspect duplicate URLs
print("\n4. Duplicate URL analysis:")
duplicate_urls = df[df.duplicated("URL", keep=False)]

print("Rows involved in repeated URLs:", len(duplicate_urls))
print("Unique repeated URLs:", duplicate_urls["URL"].nunique())

# 4. Check duplicate URLs with labels
print("\n5. Repeated URLs and labels:")
if not duplicate_urls.empty:
    print(
        duplicate_urls[["URL", "label"]]
        .sort_values("URL")
        .head(10)
        .to_string(index=False)
    )

# 5. List likely URL-only features
url_features = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]

available = [col for col in url_features if col in df.columns]

print("\n6. Candidate URL-only features:")
print(available)
print("Number of candidate features:", len(available))

# 6. Check invalid or infinite numeric values
numeric = df.select_dtypes(include="number")

print("\n7. Infinite values in numeric columns:")
print(
    numeric.columns[
        numeric.isin([float("inf"), float("-inf")]).any()
    ].tolist()
)

print("\nInspection completed.")
