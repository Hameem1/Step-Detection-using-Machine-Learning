"""
This module implements Model Based Feature Selection using Random Forest classifiers.
Multiple random forests are independently trained and perform voting based feature selection.
After feature selection, a model is trained with the best features and tested with test data.

"""

# Imports
import os
import operator
import numpy as np
import pandas as pd
from time import time
from config import new_sensor_paths, DATA_PATH
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest, RFE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from model.feature_selection_cv import normalize

# Globals
# list of all feature labels + StepLabel
cols = pd.read_csv(f'{new_sensor_paths[0]}\\{os.listdir(new_sensor_paths[0])[0]}', sep='\t', index_col=0).columns
# Setting numpy print precision
np.set_printoptions(precision=5)
# no. of best features to select
n_features = 15
# no. of different Random Forests participating in vote
r = 5
# Voting Box
nth_rank = {k: 0 for k in cols[0:-1]}
# Features obtained after feature selection
selected_features = ['Ax_energy', 'Ay_skewness']

# no. of rows of dataset to be used
row_count = 50000
# no. of Decision Trees per Random Forest
RF_ESTIMATORS = 100
# If True, the dataset is normalized before training
DATA_NORMALIZATION = True
# If True, a selected portion of the entire dataset is used for training
DATA_REDUCE = True
# If True, Feature Selection is performed. If False, selected_features has to be manually initialized above
FEATURE_SELECTION = False

if FEATURE_SELECTION:
    selected_features = []

# starting timer
start = time()
# loading in the entire actual dataset for one sensor
print(DATA_PATH)
DATA = pd.read_csv(DATA_PATH, sep='\t', index_col=0)
# Loading the relevant data
if DATA_REDUCE:
    DATA = DATA.iloc[0:row_count, :]
print('>> Dataset loaded\n')
# Normalizing the dataset
if DATA_NORMALIZATION:
    DATA = normalize(DATA)
    print('>> Dataset normalized\n')
# Converting the data to numpy arrays
data_matrix = DATA.values
# separating the data into predictors and target
X = data_matrix[:, 0:-1]
y = data_matrix[:, -1]

if __name__ == '__main__':
    if FEATURE_SELECTION:
        # Splitting the data into training and testing splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

        # Feature selection
        # Recursive Feature Elimination/Model Based Feature Selection (More powerful than Uni-variate selection)
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rfe = RFE(model, n_features)

        # Voting between multiple (r) Random forest based rankings
        for i in range(r):
            print(f'\nGenerating votes from RandomForest # {i+1}')
            fit = rfe.fit(X_train, y_train)
            feature_names = list(DATA.columns[0:-1][fit.support_])
            for feature in feature_names:
                nth_rank[feature] += 1      # Casting Votes

        sorted_nth_rank = sorted(nth_rank.items(), key=operator.itemgetter(1), reverse=True)
        print(f'\n\nSelected {n_features} features:\n')

        # Printing Election Results
        count = 1
        for k, v in sorted_nth_rank[0:n_features]:
            print(f'#{count} -  {k} -   score = {v}/{r}')
            selected_features.append(k)
            count += 1

        print('>> Feature selection complete\n')
        print(f'Feature selection with "{r}" separate Random Forest Classifiers complete')
        duration = time() - start
        print('Operation took:', f'{duration:.2f} seconds.' if duration < 60 else f'{duration/60:.2f} minutes.\n\n')

    else:
        print(f'Features being used:\n{selected_features}\n')
    DATA_selected = DATA[selected_features]
    data_matrix = DATA_selected.values
    X_train, X_test, y_train, y_test = train_test_split(data_matrix, y, test_size=0.3, random_state=10)
    model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, n_jobs=-1)
    print('>> Training a Random Forest Classifier using the best features\n')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n')
    print(f'Score of the classifier on test data:\n'
          f'Accuracy  = {accuracy_score(y_test, y_pred)*100:.3f}%\n'
          f'Precision = {precision_score(y_test, y_pred)*100:.3f}%\n'
          f'Recall    = {recall_score(y_test, y_pred)*100:.3f}%\n'
          f'F1 score  = {f1_score(y_test, y_pred)*100:.3f}%\n\n')

else:
    print(f"\nModule imported : {__name__}")


# Uni-variate Selection (Unable to perform this because the data needs to be non-negative for this to work)
# test = SelectKBest(score_func=chi2, k=n_features)
# fit = test.fit(X, y)
# # summarizing scores
# print(f'SelectKBest Scores:         {fit.scores_}')
# features = fit.transform(X)
# # summarize selected features
# print(f'Selected features:\n{features[:, :]}')
