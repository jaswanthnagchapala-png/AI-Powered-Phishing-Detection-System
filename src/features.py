
from urllib.parse import urlsplit
import ipaddress
import re


# Hostname-only features. The PhiUSIIL dataset's legitimate URLs are all
# "https://www.<domain>" homepages, so scheme, www, path, query and port
# are dataset artifacts, not real phishing signals, and are excluded.
FEATURE_NAMES = [
    "hostname_length",
    "num_dots",
    "num_subdomains",
    "num_digits",
    "num_letters",
    "num_hyphens",
    "tld_length",
    "has_ip_address",
    "has_suspicious_keyword",
]


SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "account",
    "update",
    "secure",
    "signin",
    "password",
    "banking",
]


def extract_features(url):
    url = str(url).strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    # Handle URLs with or without a protocol.
    if url.startswith("//"):
        url = "http:" + url
    elif not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url
    ):
        url = "http://" + url

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid URL.") from exc

    if not hostname:
        raise ValueError("URL must contain a hostname.")

    hostname = hostname.lower()

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    if hostname.startswith("www."):
        hostname = hostname[4:]

    hostname_parts = hostname.split(".")
    subdomain_count = max(0, len(hostname_parts) - 2)

    features = {
        "hostname_length": len(hostname),
        "num_dots": hostname.count("."),
        "num_subdomains": subdomain_count,
        "num_digits": sum(c.isdigit() for c in hostname),
        "num_letters": sum(c.isalpha() for c in hostname),
        "num_hyphens": hostname.count("-"),
        "tld_length": len(hostname_parts[-1]),
        "has_ip_address": has_ip,
        "has_suspicious_keyword": int(
            any(word in hostname for word in SUSPICIOUS_WORDS)
        ),
    }

    return {
        name: features[name]
        for name in FEATURE_NAMES
    }
