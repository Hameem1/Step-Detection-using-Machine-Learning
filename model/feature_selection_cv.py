# IMPORTS
import os
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from dataset.dataset_manipulator import ROOT, sensors
from sklearn.feature_selection import chi2, SelectKBest, RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score


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
X = data_matrix[:, 0:-1]
y = data_matrix[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
# Feature selection
# Recursive Feature Elimination/Model Based Feature Selection (with cross-validated selection of best # of features)
model = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 100})
rfecv = RFECV(estimator=model, cv=StratifiedKFold(2), scoring='f1_weighted', n_jobs=-1)
print('>> Training model\n')
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

print('>> Testing model\n')
print(f'Score of the classifier on test data = {rfecv.score(X_test, y_test)*100:.3f}%\n')
y_pred = rfecv.predict(X_test)
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n')
print(f'Score of the classifier on test data:\n'
      f'Accuracy  = {accuracy_score(y_test, y_pred)*100:.3f}%\n'
      f'Precision = {precision_score(y_test, y_pred)*100:.3f}%\n'
      f'Recall    = {recall_score(y_test, y_pred)*100:.3f}%\n'
      f'F1 score  = {f1_score(y_test, y_pred)*100:.3f}%\n\n')

duration = time() - start
print('Operation took:', f'{duration:.2f} seconds.' if duration < 60 else f'{duration/60:.2f} minutes.')

# Todo: Make the steps to be a window of ones around the current 1 (super-sampling)
# Todo: Normalize the data before feeding it to the model (Train+Test)
# Todo: Adjust the parameters and test feature_selection_cv
