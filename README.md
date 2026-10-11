# AI-Powered Phishing Detection System

An AI-based cybersecurity project that aims to identify potentially phishing URLs using machine learning and provide users with a simple web interface for URL assessment.

## Project Overview

Phishing attacks use deceptive URLs to trick users into visiting fraudulent websites. This project aims to develop a machine learning-based system that analyzes URL characteristics and classifies submitted URLs as potentially phishing or legitimate.

**Current status:** Working prototype (hostname-based model, ~80% accuracy)

## Objectives

- Develop a URL-based phishing detection model using Python.
- Prepare and analyze a labelled phishing URL dataset.
- Extract meaningful features directly from URLs.
- Compare machine learning algorithms using appropriate evaluation metrics.
- Build a web interface using Flask, HTML, CSS, and JavaScript.
- Display prediction results and relevant URL indicators.

## Planned Features

- URL input and validation
- URL feature extraction
- Machine learning-based classification
- Phishing and legitimate prediction results
- Input error handling
- User-friendly web interface
- Model performance evaluation

## Technology Stack

- **Language:** Python
- **Data Processing:** pandas, NumPy
- **Machine Learning:** scikit-learn
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Development Tools:** VS Code, Git, GitHub

## Dataset

The project uses the PhiUSIIL Phishing URL Dataset from the UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset

The dataset will be inspected and prepared before model training. Dataset attribution and licensing requirements will be followed.

## Planned Project Structure

```text
AI-Powered-Phishing-Detection-System/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── best_model.joblib
├── src/
│   ├── inspect_dataset.py
│   ├── checkdatabase.py
│   ├── prepare_dataset.py
│   ├── validate_model.py
│   ├── features.py
│   ├── train_model.py
│   └── app.py
├── templates/
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

## Development Roadmap

1. Project requirements and planning
2. Dataset inspection and preparation
3. URL feature engineering
4. Machine learning model training and comparison
5. Model evaluation and selection
6. Flask backend development
7. Frontend development and integration
8. Testing, documentation, and final improvements

## Features and Validation Notes

The model uses hostname-only features (length, dots, subdomains, digits, letters, hyphens, TLD length, IP address, suspicious keywords). Scheme (HTTPS), `www.`, path, query and port were removed because in the PhiUSIIL dataset every legitimate URL is an `https://www.` homepage, so those features let the model reach ~99.5% accuracy by learning a dataset artifact, which failed on real sites (for example, any URL with a path was flagged as phishing). After removing them, accuracy is about 80% (phishing recall about 64%) on both random and unseen-hostname splits.

## Limitations

Hostnames with subdomains (for example `docs.python.org`) can be wrongly flagged. Adding legitimate URLs with paths and subdomains from another source (such as the Tranco list) would improve the model.

The system will provide predictions based on learned URL patterns. It cannot guarantee that a website is safe or detect every phishing attack. HTTPS does not necessarily mean a website is trustworthy.

## Disclaimer

This project is intended for educational and research purposes. Predictions should be treated as risk indicators rather than definitive security verdicts.

## License

See the [LICENSE](LICENSE) file for licensing information.
