# Imports
from pathlib import Path

# Dataset configuration variables for analysis and processing
# Sampling frequency of the sensors
Fs = 100
# No. of points around step point to consider as step span
STEP_SIZE = 20
# Sensor type being used
SENSOR = 'acc'
# Class label being used
USED_CLASS_LABEL = 'valid'
# Type of window used for feature extraction
WINDOW_TYPE = 'sliding'
# Window size for feature extraction (Should be EVEN)
WINDOW_SIZE = 40
# Age groups to use for analysis
ageGroups = ['(1-13)', '(5-15)']

# Configuration variables
# Set to "True" for verbose status messages
FILE_STATUS_MESSAGES = False
# Allows forcing user input to 'y'
FORCE = False
# Storage selector for SUBJECTS_REPO (Dict/List)
STORAGE = "List"

# Paths and directories
# Raw Dataset directory name
FOLDER_NAME = "OU-InertGaitAction_wStepAnnotation"
# Feature Dataset directory name
DATASET_FOLDER = "Step_Detection_Dataset (w=40, sliding)"
# Root path (project directory)
ROOT = str(Path(__file__).parent)
# DATASETS directory path (Generated data sets)
DATASET_ROOT = f"{ROOT}\\..\\DATASETS"
# Data-files directory path
data_files_path = ROOT+'\\data-files'
# Sensor paths
sensors = ["Center", "Left", "Right"]
sensor_paths = [f"{ROOT}\\{FOLDER_NAME}\\{sensor}" for sensor in sensors]
