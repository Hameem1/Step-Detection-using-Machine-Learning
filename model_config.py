"""
Contains the global configurations for model related tasks and performs common imports.

Attributes
----------
cols : list of str
    List of all feature labels + StepLabel
row_count : int
    No. of rows of dataset to be used
RF_ESTIMATORS : int
    No. of Decision Trees per Random Forest
TEST_SIZE : float
    Test Data size (out of 1.0)
VERBOSE : int or bool
    Controls Model processing verbosity
N_JOBS : int
    Controls the no. of threads to use for computations (N_JOBS = -1 for auto)
K_FOLD : int
    No. of Cross validation folds
SCORING : str
    Performance metric to optimize the model for
TESTING : bool
    Set to True if testing with the Python CONSOLE
DATA_NORMALIZATION : bool
    If True, the dataset is normalized before training and testing
DATA_REDUCE : bool
    If True, a selected portion of the entire dataset is used for training+testing (# of rows = row_count)
GEN_RANKING_FILE : bool
    If True, generate a .csv file for the feature ranking
PLOT : bool
    If True, a plot will be generated for the # of features used vs performance metric
EXPORT_MODEL : bool
    If True, trained model is exported to TRAINED_MODEL_PATH

DATA_PATH : str
    loading in the actual dataset for one sensor (Data under test)
NEW_DATASET : str
    Directory name for new data set which contains the training/testing data for the classifier
NEW_DATASET_PATH : str
    Directory path for new data set which contains the training/testing data for the classifier
TRAINED_MODEL_DIR : str
    Trained Model directory name
TRAINED_MODEL_PATH : str
    Trained Model directory path
TRAINED_MODEL_NAME : str
    Trained Model name
TRAINED_NORMALIZER_NAME : str
    Trained Normalizer name

"""

# Global imports
import os
import locale
import numpy as np
import pandas as pd
from time import time
from datetime import datetime as dt
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from config import new_sensor_paths, ROOT

# Configuring locale for datetime purposes
lang = 'de_DE'
locale.setlocale(locale.LC_ALL, lang)

# Model Configuration Variables
# list of all feature labels + StepLabel
cols = pd.read_csv(f'{new_sensor_paths[0]}\\{os.listdir(new_sensor_paths[0])[0]}', sep='\t', index_col=0).columns
# Setting numpy print precision
np.set_printoptions(precision=5)
# no. of rows of dataset to be used
row_count = 50000
# no. of Decision Trees per Random Forest
RF_ESTIMATORS = 100
# Test Data size (out of 1.0)
TEST_SIZE = 0.5
# Controls Model processing verbosity
VERBOSE = True
# Controls the no. of threads to use for computations (N_JOBS = -1 for auto)
N_JOBS = -1
# Cross validation folds
K_FOLD = 2
# Performance metric to optimize the model for
SCORING = 'f1_weighted'
# Set to True if TESTING with the Python CONSOLE
TESTING = False
# If True, the dataset is normalized before training & testing
DATA_NORMALIZATION = True
# If True, a selected portion of the entire dataset is used for training+testing (# of rows = row_count)
DATA_REDUCE = False
# If True, generate a .csv file for the feature ranking
GEN_RANKING_FILE = False
# If True, a plot will be generated for the # of features used vs performance metric
PLOT = False
# If True, trained model is exported to TRAINED_MODEL_PATH
EXPORT_MODEL = False

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
