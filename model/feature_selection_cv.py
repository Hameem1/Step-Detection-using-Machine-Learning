# IMPORTS
import os
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from dataset.dataset_manipulator import ROOT, sensors
from sklearn.feature_selection import chi2, SelectKBest, RFECV

# Globals
# Directory paths
DATASET_FOLDER = "Step_Detection_Dataset (w=40, sliding)"
DATASET_ROOT = f"{ROOT}\\..\\DATASETS"
sensor_paths = [f"{DATASET_ROOT}\\{DATASET_FOLDER}\\{sensor}" for sensor in sensors]
# list of all feature labels + StepLabel
cols = pd.read_csv(f'{sensor_paths[0]}\\{os.listdir(sensor_paths[0])[0]}', sep='\t', index_col=0).columns
# Setting numpy print precision
np.set_printoptions(precision=5)
# Voting Box
nth_rank = {k: 0 for k in cols[0:-1]}
# Features obtained after feature selection
selected_features = []
# no. of rows of dataset to be used
row_count = 50000

# starting timer
start = time()
# loading in the actual dataset for one sensor
DATA_PATH = f"{ROOT}\\Features_Dataset\\ds_right.csv"
print(DATA_PATH)
DATA = pd.read_csv(DATA_PATH, sep='\t', index_col=0)
# Loading the relevant data
DATA = DATA.iloc[0:row_count, :]
print('>> Dataset loaded\n')
data_matrix = DATA.values
X = data_matrix[:, 0:len(DATA.columns)-1]
y = data_matrix[:, len(DATA.columns)-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
# Feature selection
# Recursive Feature Elimination/Model Based Feature Selection (with cross-validated selection of best # of features)
model = RandomForestClassifier(n_estimators=100)
rfecv = RFECV(estimator=model, cv=StratifiedKFold(2), scoring='accuracy', n_jobs=-1)
fit = rfecv.fit(X_train, y_train)

# summarize ranking
print(f"# of selected features:     {fit.n_features_}\n")
feature_names = list(DATA.columns[0:-1][fit.support_])
print(f"Selected Features:\n{fit.support_}\n{feature_names}\n")
ranking = pd.DataFrame({'rank': fit.ranking_, 'feature': DATA.columns[0:-1]})
ranking = ranking.sort_values(by=['rank']).reset_index(drop=True)
print(f"Feature Ranking:\n{fit.ranking_}\n{ranking}\n")
print(f"Optimal number of features : {rfecv.n_features_}")
# Plotting number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

print(f'Score of the classifier on test data = {rfecv.score(X_test, y_test):.3f}%\n')

duration = time() - start
print('Operation took:', f'{duration:.2f} seconds.' if duration < 60 else f'{duration/60:.2f} minutes.')
