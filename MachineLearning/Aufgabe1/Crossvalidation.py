# Sangeeths Chandrakuar
# 2023-09-20
# CDS Machine Learning

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


data = load_iris()
X, y = data.data, data.target #Features, Labels

# Modell mit Hyperparametern
clf = RandomForestClassifier(n_estimators=100, random_state=42) # random_state für Reproduzierbarkeit

# Cross-Validation
scores = cross_val_score(clf, X, y, cv=5)  # 5 Folds
print("Durchschnittliche Genauigkeit:", scores.mean())

