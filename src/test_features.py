
from features import extract_features

test_urls = [
    "https://www.google.com",
    "http://192.168.1.10/login",
    "https://example.com/account/verify?id=123",
]

for url in test_urls:
    print("\nURL:", url)
    print(extract_features(url))
