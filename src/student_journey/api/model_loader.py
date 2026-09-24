"""Loads the trained model and SHAP explainer exactly once, at process
startup, and hands them to request handlers — never reloaded per-request.
"""
from __future__ import annotations

from student_journey.explainability.shap_utils import PersistenceExplainer
from student_journey.models.predict import PersistenceModel


class ModelBundle:
    def __init__(self):
        self.model = PersistenceModel()
        self.explainer = PersistenceExplainer()

    @property
    def model_version(self) -> str:
        return self.model.model_version
