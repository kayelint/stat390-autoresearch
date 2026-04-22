"""
EDITABLE -- The agent modifies this file.

Define the classification pipeline for the manga → anime adaptation task.
build_model() must return an sklearn-compatible classifier with predict_proba().
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_model():
    """Return an sklearn Pipeline. This is what the agent improves."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
