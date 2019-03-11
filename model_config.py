"""
Contains the global configurations for model related tasks and performs common imports.

Attributes
----------
DATA_PATH : str
    loading in the actual dataset for one sensor (Data under test)
NEW_DATASET : str
    Directory name for new data set which contains the training/testing data for the classifier
NEW_DATASET_PATH : str
    Directory path for new data set which contains the training/testing data for the classifier
"""

# Global imports
import os
import numpy as np
import pandas as pd
from time import time
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from config import new_sensor_paths, ROOT


# Model Configuration Variables
# list of all feature labels + StepLabel
cols = pd.read_csv(f'{new_sensor_paths[0]}\\{os.listdir(new_sensor_paths[0])[0]}', sep='\t', index_col=0).columns
# Setting numpy print precision
np.set_printoptions(precision=5)
# no. of rows of dataset to be used
row_count = 5000
# no. of Decision Trees per Random Forest
RF_ESTIMATORS = 100
# Test Data size (out of 1.0)
TEST_SIZE = 0.5
# Cross validation folds
K_FOLD = 2
# Performance metric to optimize the model for
SCORING = 'f1_weighted'
# Set to True if TESTING with console
TESTING = False
# If True, the dataset is normalized before training
DATA_NORMALIZATION = True
# If True, a selected portion of the entire dataset is used for training (# of rows = row_count)
DATA_REDUCE = False
# If True, generate a .csv file for the feature ranking
GEN_RANKING_FILE = True
# If True, a plot will be generated for the # of features used vs performance metric
PLOT = True
# If True, trained model is exported to TRAINED_MODEL_PATH
EXPORT_MODEL = True

# Paths
# loading in the actual dataset for the ML classifier
DATA_PATH = f"{ROOT}\\Features_Dataset\\ds_all.csv"
# Directory name for new data set which contains the training/testing data for the classifier
NEW_DATASET = "Features_Dataset"
# Directory path for new data set which contains the training/testing data for the classifier
NEW_DATASET_PATH = f'{ROOT}\\{NEW_DATASET}'
# Trained Model directory name
TRAINED_MODEL_DIR = 'Trained_Model'
# Trained Model directory path
TRAINED_MODEL_PATH = f'{ROOT}\\{TRAINED_MODEL_DIR}'
# Trained Model name
TRAINED_MODEL_NAME = 'step_detection_model_test.pkl'
# Trained Normalizer name
TRAINED_NORMALIZER_NAME = 'step_detection_min_max_norm_test.pkl'
