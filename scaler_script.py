import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("Starting scaler generation for the UI...")
NUM_FEATURES = 50
NUM_SAMPLES = 20000
NUM_CLASSES = 5
SCALER_PATH = 'scaler.gz'

print("Generating synthetic dataset to match model training data...")
X, y = make_classification(
    n_samples=NUM_SAMPLES, n_features=NUM_FEATURES, n_informative=30, n_redundant=5,
    n_classes=NUM_CLASSES, n_clusters_per_class=2, flip_y=0.01, random_state=42,
)

X_train, _, _, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Fitting StandardScaler on the training data and saving to '{SCALER_PATH}'...")
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, SCALER_PATH)

print(f"\nSuccess! Scaler saved to '{SCALER_PATH}'. You can now run the Streamlit app.")
