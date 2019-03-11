"""
This module simply trains and tests a Random Forest classifier with the data from the specified dataset.
All features are used by default.

"""

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from model.feature_selection_complete import normalize, import_trained_model, export_trained_model
from model_config import *

