# experiments/compare_lda_qda_rda_split_csv.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os
from sklearn.metrics import confusion_matrix


# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lda import LDA
from models.qda import QDA
from models.rda import RDA

# Argument parser
parser = argparse.ArgumentParser(description="Compare LDA, QDA and RDA with train/test split using a CSV file")
parser.add_argument('csv', type=str, help="Path to CSV file")
args = parser.parse_args()

import pandas as pd

# Load dataset
df = pd.read_csv(args.csv)
print(df["Disease"])

# Filter only the two classes of interest
df = df[df["Disease"].isin(["Asthma", "Bronchiectasis"])]
# Encode labels: Asthma -> 0, Bronchiectasis -> 1
df["label"] = df["Disease"].map({"Asthma": 0, "Bronchiectasis": 1})
print(df["label"] )

# Drop non-feature columns (string or irrelevant ones)
#feature_cols = ["met7874", "met19602", "met11006", "met7500", "met9231", "met2758", "met17100", "met7147", "met460"]
feature_cols = [
    "met7874", "met19602", "met11006", "met7500", "met9231", "met2758", "met17100", "met7147", "met460",  
    "met137353", "met141033", "met146102", "met160669", "met520958", "met537332", "met551986", "met557859", "met558880"  
]

X = df[feature_cols].values
y = df["label"].values

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(y_test)
# Initialize models
lda = LDA()
qda = QDA()
rda = RDA(alpha=0.5)

models = [lda, qda, rda]
model_names = ["LDA", "QDA", "RDA (alpha=0.5)"]

# Train
for model in models:
    model.fit(X_train, y_train)

# Evaluate
print(f"Classification accuracies on test set (CSV = {args.csv}):")
for name, model in zip(model_names, models):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")
    print(confusion_matrix(y_test, y_pred))


