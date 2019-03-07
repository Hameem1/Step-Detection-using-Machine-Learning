# IMPORTS
import os
import operator
import numpy as np
import pandas as pd
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dataset.dataset_manipulator import ROOT, sensors
from sklearn.feature_selection import chi2, SelectKBest, RFE

# Globals
# Directory paths
DATASET_FOLDER = "Step_Detection_Dataset (w=40, sliding)"
DATASET_ROOT = f"{ROOT}\\..\\DATASETS"
sensor_paths = [f"{DATASET_ROOT}\\{DATASET_FOLDER}\\{sensor}" for sensor in sensors]
# list of all feature labels + StepLabel
cols = pd.read_csv(f'{sensor_paths[0]}\\{os.listdir(sensor_paths[0])[0]}', sep='\t', index_col=0).columns
# Setting numpy print precision
np.set_printoptions(precision=5)
# no. of best features to select
n_features = 15
# no. of files to test on
n_files = 10
# no. of different Random Forests participating in vote
r = 10
# Voting Box
nth_rank = {k: 0 for k in cols[0:-1]}
# Features obtained after feature selection
selected_features = []

# starting timer
start = time()
# loading in the actual dataset for one sensor
# DATA_PATH = f"{ROOT}\\'Features_Dataset'\\'ds_right.csv'"
# DATA = pd.read_csv(DATA_PATH, sep='\t', index_col=0)

for f in range(n_files):
    # Loading the relevant data
    test_file = f'{sensor_paths[0]}\\{os.listdir(sensor_paths[0])[f]}'
    print(f'\nFor file - {test_file[-12:]}')
    df = pd.read_csv(test_file, sep='\t', index_col=0)
    data_matrix = df.values
    X = data_matrix[:, 0:len(df.columns)-1]
    y = data_matrix[:, len(df.columns)-1]

    # Feature selection
    # Recursive Feature Elimination/Model Based Feature Selection (More powerful than Uni-variate selection)
    model = RandomForestClassifier(n_estimators=200)
    rfe = RFE(model, n_features)

    # Voting between multiple (r) Random forest based rankings
    for i in range(r):
        print(f'\nGenerating votes from RandomForest # {i+1}')
        fit = rfe.fit(X, y)
        # summarize ranking
        # print(f"# of selected features:     {fit.n_features_}\n")
        feature_names = list(df.columns[0:-1][fit.support_])
        # print(f"Selected Features:\n{fit.support_}\n{feature_names}\n")
        # ranking = pd.DataFrame({'rank': fit.ranking_, 'feature': df.columns[0:-1]})
        # ranking = ranking.sort_values(by=['rank']).reset_index(drop=True)
        # print(f"Feature Ranking:\n{fit.ranking_}\n{ranking}\n")
        for feature in feature_names:
            nth_rank[feature] += 1      # Casting Votes

sorted_nth_rank = sorted(nth_rank.items(), key=operator.itemgetter(1), reverse=True)
print(f'\n\nSelected {n_features} features:\n')
count = 1

for k, v in sorted_nth_rank[0:n_features]:
    print(f'#{count} -  {k} -   score = {v}/{r*n_files}')       # Printing Election Results
    selected_features.append(k)
    count += 1

print(f'Feature selection for "{n_files} files"" with "{r}" separate Random Forest Classifiers complete')
print(f'Operation took {time() - start:.2f} secs.')

# Uni-variate Selection (Unable to perform this because the data needs to be non-negative for this to work)
# test = SelectKBest(score_func=chi2, k=n_features)
# fit = test.fit(X, y)
# # summarizing scores
# print(f'SelectKBest Scores:         {fit.scores_}')
# features = fit.transform(X)
# # summarize selected features
# print(f'Selected features:\n{features[:, :]}')
