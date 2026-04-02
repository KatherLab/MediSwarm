"""
This package initializes the necessary modules and classes for the project.
"""

from .base_model import VeryBasicModel, BasicModel, BasicClassifier
from .model_factory import model_factory

__all__ = ['VeryBasicModel', 'BasicModel', 'BasicClassifier', 'model_factory']
