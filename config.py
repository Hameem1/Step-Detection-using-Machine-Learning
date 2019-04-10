"""
Contains the global configurations for the entire project.

Attributes
----------
Fs : int
    Sampling frequency of the sensors
SENSOR : {'acc', 'gyr'}
    Sensor being used
USED_CLASS_LABEL : {"valid", "invalid", "level", "upstairs", "downstairs", "incline", "decline", "complete"}
    Class label being used for analysis
WINDOW_TYPE : {'sliding', 'hopping'}
    Type of window used for feature extraction
WINDOW_SIZE : int
    Window size for feature extraction (must be EVEN)
STEP_SIZE : int
    No. of points around step point to consider as step span (must be EVEN)
ageGroups : list of str
    Age groups to use for analysis

DEBUGGER : bool
    Sets the Flask debugger mode
FILE_STATUS_MESSAGES : bool
    Set to "True" for verbose status messages
FORCE : bool
    Allows forcing user input to 'y'
STORAGE : {'list', 'dict'}
    Storage selector for SUBJECTS_REPO

ROOT : str
    Root path (project directory)
ORIGINAL_DATASET : str
    Raw Dataset directory name
FEATURES_DATASET : str
    Feature Dataset directory name
sensors : list of str
    List of sensor directory names
sensor_paths : list of str
    Sensor directory paths in original dataset_operations
DATASETS : str
    Data Sets directory path (Generated data sets)
new_sensor_paths : list of str
    Paths to sensor directories in the NEW Dataset
data_files_path : str
    Data-files directory path
age_dirs : dict of list
    Age sorted dataset_operations directories
sensor_dirs : dict of list
    Paths to sensor directories in the age folders

"""

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
# Set to "True" for development
DEBUGGER = True
# Set to "True" for verbose status messages
FILE_STATUS_MESSAGES = False
# Allows forcing user input to 'y'
FORCE = False
# Storage selector for SUBJECTS_REPO (dict/list)
STORAGE = "list"

# Paths and directories
# Root path (project directory)
ROOT = str(Path(__file__).parent)
# Raw Dataset directory name
ORIGINAL_DATASET = "OU-InertGaitAction_wStepAnnotation"
# Feature Dataset directory name
FEATURES_DATASET = "Step_Detection_Dataset (w=40, s=20, sliding)"
# Data Sets directory path (Generated data sets)
DATASETS = Path(f"{ROOT}/Data Sets")
# Sensor paths in original dataset_operations
sensors = ["Center", "Left", "Right"]
sensor_paths = [Path(f"{DATASETS}/{ORIGINAL_DATASET}/{sensor}") for sensor in sensors]
# Paths to sensor directories in the NEW Dataset
new_sensor_paths = [Path(f"{DATASETS}/{FEATURES_DATASET}/{sensor}") for sensor in sensors]
# Data-files directory path
data_files_path = Path(f'{ROOT}/Data Files')
# Age sorted dataset_operations directories
age_dirs = {"Age_" + dirName: f'{DATASETS}/{FEATURES_DATASET}_Age_Sorted/Age_{dirName}' for dirName in ageGroups}
# Paths to C, L and R in the age folders
sensor_dirs = {"Age_" + dirName: [f'{DATASETS}/{FEATURES_DATASET}_Age_Sorted/Age_{dirName}/{sensor}'
                                  for sensor in sensors]
               for dirName in ageGroups}
