# IMPORTS
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
from dataset.dataset_manipulator import ROOT, sensors

DATASET_FOLDER = "Step_Detection_Dataset (w=40, sliding)"
DATASET_ROOT = f"{ROOT}\\..\\DATASETS"
sensor_paths = [f"{DATASET_ROOT}\\{DATASET_FOLDER}\\{sensor}" for sensor in sensors]

# Loading the relevant data
col_names = list(pd.read_csv(f'{sensor_paths[0]}\\{os.listdir(sensor_paths[0])[0]}', sep='\t', index_col=0).columns)
df = pd.read_csv()
data_array = df.values
X = col_names[:-1]
y = col_names[-1]

