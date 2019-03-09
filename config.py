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
# Root path (project directory)
ROOT = str(Path(__file__).parent)
# Raw Dataset directory name
FOLDER_NAME = "OU-InertGaitAction_wStepAnnotation"
# Feature Dataset directory name
DATASET_FOLDER = "Step_Detection_Dataset (w=40, s=20, sliding)"
# Sensor paths in original dataset
sensors = ["Center", "Left", "Right"]
sensor_paths = [f"{ROOT}\\{FOLDER_NAME}\\{sensor}" for sensor in sensors]
# DATASETS directory path (Generated data sets)
DATASET_ROOT = f"{ROOT}\\..\\DATASETS"
# Paths to sensor directories in the NEW Dataset
new_sensor_paths = [f"{DATASET_ROOT}\\{DATASET_FOLDER}\\{sensor}" for sensor in sensors]
# Data-files directory path
data_files_path = ROOT + '\\data-files'
# Age sorted dataset directories
age_dirs = {"Age_" + dirName: f'{DATASET_ROOT}\\{DATASET_FOLDER}_Age_Sorted\\Age_{dirName}' for dirName in ageGroups}
# Paths to C, L and R in the age folders
sensor_dirs = {"Age_" + dirName: [f'{DATASET_ROOT}\\{DATASET_FOLDER}_Age_Sorted\\Age_{dirName}\\{sensor}'
                                  for sensor in sensors]
               for dirName in ageGroups}
# loading in the actual dataset for one sensor (Data under test)
DATA_PATH = f"{ROOT}\\Features_Dataset\\ds_right.csv"
# Directory name for new data set which contains the training/testing data for the classifier
NEW_DATASET = "Features_Dataset"
# Directory path for new data set which contains the training/testing data for the classifier
NEW_DATASET_PATH = f'{ROOT}\\{NEW_DATASET}'
