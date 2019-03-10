"""
This module implements Model Based Feature Selection (with cross-validated selection of best # of features).
The model used is a Random Forest classifier.
The model is trained during the feature selection phase and is tested at the end on test data from the best features.

"""

# Todo: Generate a csv file containing the feature ranking
# Todo: Combine the feature selection functionality of both models and rename feature_selection.py to model.py
# Todo: Use model.py for training the model with selected features (all for max performance)
# Todo: Generate the # of features vs score plot using plotly
# Todo: Use scikit-learn for normalization
# Todo: Test with disjoint datasets (e.g. train with ds_right and test on ds_left)
# Todo: Organize results


# Imports
import os
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from config import new_sensor_paths, DATA_PATH
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, SelectKBest, RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score


# Globals
# list of all feature labels + StepLabel
cols = pd.read_csv(f'{new_sensor_paths[0]}\\{os.listdir(new_sensor_paths[0])[0]}', sep='\t', index_col=0).columns
# Setting numpy print precision
np.set_printoptions(precision=5)
# no. of rows of dataset to be used
row_count = 50000
# no. of Decision Trees per Random Forest
RF_ESTIMATORS = 100
# Performance metric to optimize the model for
SCORING = 'f1_weighted'
# If True, the dataset is normalized before training
DATA_NORMALIZATION = True
# If True, a selected portion of the entire dataset is used for training (# of rows = row_count)
DATA_REDUCE = True
# Top t features to show
t = 30


def normalize(data):
    feature_dic = dict()
    # Performing Min-Max Scaling
    for feature in cols[:-1]:
        feature_dic[feature] = list((data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min()))
    feature_dic['StepLabel'] = data[cols[-1]]
    df_new = pd.DataFrame.from_dict(feature_dic)
    return df_new


if __name__ == '__main__':
    # starting timer
    start = time()
    # loading in the entire actual dataset for one sensor
    print(DATA_PATH)
    DATA = pd.read_csv(DATA_PATH, sep='\t', index_col=0)
    # Loading the relevant data (limiting # of rows)
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
    # Splitting the data into training and testing splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    # Feature selection
    # Recursive Feature Elimination/Model Based Feature Selection (with cross-validated selection of best # of features)
    model = RandomForestClassifier(n_estimators=RF_ESTIMATORS)
    rfecv = RFECV(estimator=model, cv=StratifiedKFold(2), scoring=SCORING, n_jobs=-1)
    print('>> Training the model\n')
    fit = rfecv.fit(X_train, y_train)

    # summarize ranking
    print(f"# of selected features:     {fit.n_features_}/{len(cols[:-1])}\n")
    feature_names = list(DATA.columns[0:-1][fit.support_])
    print(f"Selected Features:\n{fit.support_}\n{feature_names}\n")
    ranking = pd.DataFrame({'rank': fit.ranking_, 'feature': DATA.columns[0:-1]})
    ranking = ranking.sort_values(by=['rank']).reset_index(drop=True)
    top_t_features = list(ranking.feature[0:t])
    print(f'Top {t} features list:\n{top_t_features}\n')
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

else:
    print(f"\nModule imported : {__name__}")
