"""
This module implements Model Based Feature Selection (with cross-validated selection of best # of features).
The model used is a Random Forest classifier.
The model is trained during the feature selection phase and is tested at the end on test data using the best features.

"""

# Todo: Combine the feature selection functionality of both models and rename feature_selection.py to model.py
# Todo: Use model.py for training the model with selected features (all for max performance)
# Todo: Organize results
# Todo: Add docstrings for the entire "model" package
# Todo: Refactor the variable names involving paths, used in config.py

# Imports
import os
import numpy as np
import pandas as pd
from time import time
import plotly.offline as pyo
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from config import new_sensor_paths, DATA_PATH, data_files_path, ROOT, TRAINED_MODEL_PATH, \
    TRAINED_MODEL_NAME, TRAINED_MODEL_DIR, TRAINED_NORMALIZER_NAME


# Configuration Variables
# list of all feature labels + StepLabel
cols = pd.read_csv(f'{new_sensor_paths[0]}\\{os.listdir(new_sensor_paths[0])[0]}', sep='\t', index_col=0).columns
# Setting numpy print precision
np.set_printoptions(precision=5)
# no. of rows of dataset to be used
row_count = 5000
# no. of Decision Trees per Random Forest
RF_ESTIMATORS = 100
# Test Data size (out of 1.0)
TEST_SIZE = 0.5
# Cross validation folds
K_FOLD = 2
# Performance metric to optimize the model for
SCORING = 'f1_weighted'
# If True, the dataset is normalized before training
DATA_NORMALIZATION = True
# If True, a selected portion of the entire dataset is used for training (# of rows = row_count)
DATA_REDUCE = False
# If True, generate a .csv file for the feature ranking
GEN_RANKING_FILE = True
# If True, a plot will be generated for the # of features used vs performance metric
PLOT = True
# Test on a separate dataset
DISJOINT_TESTING = False
# Path for disjoint test dataset
TEST_DATA_PATH = f"{ROOT}\\Features_Dataset\\ds_left.csv"
# If True, trained model is exported to TRAINED_MODEL_PATH
EXPORT_MODEL = True


def plot_n_features_vs_score(grid_scores, mp_lib=False):
    x = list(range(1, len(grid_scores) + 1))
    y = grid_scores
    # Generating the plot trace
    data = [go.Scatter(x=x, y=y, text=[f'{i * 100:.2f}%' for i in y],
                       hoverinfo='text + x',
                       marker=dict(color='green'),
                       mode='lines',
                       showlegend=False)]
    # Defining the layout for the plot
    layout = go.Layout(title='No. of Features used vs Model Score',
                       xaxis=dict(title='# of features used'),
                       yaxis=dict(title=SCORING),
                       font=dict(family='arial', size=18, color='#000000'))

    # Plotting the figure
    fig = dict(data=data, layout=layout)
    pyo.plot(fig, filename=f'{data_files_path}\\Number of features vs Model score' + '.html', auto_open=False)
    print(f'>> File Generated : Number of features vs Model score' + '.html\n')

    if mp_lib:
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()


def get_selected_features():
    path = f'{data_files_path}\\features selected.csv'
    f_sel = pd.read_csv(path, sep='\t', index_col=0)
    return list(f_sel['selected features'])


def import_trained_model(dir_path, name):
    if os.path.exists(f"{dir_path}\\{name}"):
        path = f'{dir_path}\\{name}'
        ret_model = joblib.load(path)
        print('>> Model Imported.\n')
        return ret_model
    else:
        print(f'No .pkl file found in the directory : "{TRAINED_MODEL_DIR}"\n')
        return None


def export_trained_model(model, dir_path, name):
    # Creating the directory for the trained model
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f'WARNING: The path does not exist. Creating new directory...\n{TRAINED_MODEL_PATH}\n')
        os.mkdir(TRAINED_MODEL_PATH)
    else:
        print(f"Path for '{TRAINED_MODEL_DIR}' already exists!\n")

    # Saving the model externally in TRAINED_MODEL_PATH
    path = f'{dir_path}\\{name}'
    joblib.dump(model, path)
    print(f'>> Model stored externally as "{name}"\n')


if __name__ == '__main__':
    # starting timer
    start = time()
    # loading in the entire actual dataset
    print(f'{DATA_PATH}\n')
    DATA = pd.read_csv(DATA_PATH, sep='\t', index_col=0)
    # Loading the relevant data (limiting # of rows)
    if DATA_REDUCE:
        DATA = DATA.iloc[0:row_count, :]
    print('>> Dataset loaded\n')
    # Converting the data to numpy arrays
    data_matrix = DATA.values
    # separating the data into predictors and target
    X = data_matrix[:, 0:-1]
    y = data_matrix[:, -1]
    # Splitting the data into training and testing splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=101)

    # Data Normalization
    if DATA_NORMALIZATION:
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Training the normalizer
        normalizer = scaler.fit(X_train)
        # Normalizing the training data
        X_train = normalizer.transform(X_train)
        print('>> Training set normalized.\n')

    # x = pd.DataFrame(X_train, columns=cols[:-1])  # To convert from np.array to DataFrame

    # Feature selection
    # Recursive Feature Elimination/Model Based Feature Selection (with cross-validated selection of best # of features)
    classifier = RandomForestClassifier(n_estimators=RF_ESTIMATORS)
    rfecv = RFECV(estimator=classifier, cv=StratifiedKFold(K_FOLD), scoring=SCORING, n_jobs=1)
    print('>> Training the model & Performing feature ranking simultaneously\n')
    fit = rfecv.fit(X_train, y_train)
    print('>> Model Trained!\n')
    print('>> Feature Ranking complete!\n')

    # Summarizing the ranking and generating a .csv file
    print(f"# of selected features:     {fit.n_features_}/{len(cols[:-1])}\n")
    feature_names = list(DATA.columns[0:-1][fit.support_])
    print(f"Selected Features:\n\n{feature_names}\n")
    ranking = pd.DataFrame({'Feature': DATA.columns[0:-1], 'Rank': fit.ranking_})
    grid_score = pd.DataFrame({'Cumulative Predicted F1-score': [f'{i * 100:.2f}%' for i in rfecv.grid_scores_]})
    ranking = ranking.sort_values(by=['Rank']).reset_index(drop=True)
    ranking = ranking.join(grid_score)
    ranking.index = np.arange(1, len(ranking) + 1)
    print(f"Optimal number of features : {rfecv.n_features_}\n")
    # print(f"Feature Ranking:\n{fit.ranking_}\n{ranking}\n")

    # Generating .csv file from feature ranking
    if GEN_RANKING_FILE:
        ranking.to_csv(f'{data_files_path}\\feature ranking' + ".csv", sep="\t",
                       index=True, index_label='No. of features')
        print(f'>> File generated : feature ranking.csv\n')
        features_sel_df = pd.DataFrame({'selected features': feature_names})
        features_sel_df.to_csv(f'{data_files_path}\\features selected' + ".csv", sep="\t", index=True)
        print(f'>> File generated : features selected.csv\n')

    # Plotting number of features VS. cross-validation scores
    if PLOT:
        plot_n_features_vs_score(rfecv.grid_scores_)

    # Testing the model
    print('>> Testing model\n')
    if DISJOINT_TESTING:
        TEST_DATA = pd.read_csv(TEST_DATA_PATH, sep='\t', index_col=0).values
        X_test = TEST_DATA[:, 0:-1]
        y_test = TEST_DATA[:, -1]

    # Normalizing the test data
    X_test = normalizer.transform(X_test)
    print(f'Cross validation : Stratified {K_FOLD}-Fold\n')
    print(f'Performance metric used for model optimization : "{SCORING}"\n')
    y_pred = rfecv.predict(X_test)
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n')
    print(f'Score of the classifier on test data:\n'
          f'Accuracy  = {accuracy_score(y_test, y_pred) * 100:.3f}%\n'
          f'Precision = {precision_score(y_test, y_pred) * 100:.3f}%\n'
          f'Recall    = {recall_score(y_test, y_pred) * 100:.3f}%\n'
          f'F1-score  = {f1_score(y_test, y_pred) * 100:.3f}%\n'
          f'ROC_AUC   = {roc_auc_score(y_test, y_pred) * 100:.3f}%\n\n')

    duration = time() - start
    print('Operation took:', f'{duration:.2f} seconds.\n' if duration < 60 else f'{duration / 60:.2f} minutes.\n')

    # Re-training the normalizer with updated features (because they may have reduced)
    if DATA_NORMALIZATION:
        # removing the previous normalizer
        print('>> Removing the previous Normalizer\n')
        del scaler, normalizer
    print('>> Re-training the Normalizer\n')
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Converting a selected section of the dataset to a numpy array (based on best features)
    data_matrix = DATA[get_selected_features()+['StepLabel']].values
    X = data_matrix[:, 0:-1]
    y = data_matrix[:, -1]
    # Splitting the data into training and testing splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=101)
    # Re-training
    normalizer = scaler.fit(X_train)
    print('>> Normalizer re-trained\n')

    # Exporting the trained classifier and normalizer
    if EXPORT_MODEL:
        export_trained_model(rfecv, TRAINED_MODEL_PATH, TRAINED_MODEL_NAME)
        export_trained_model(normalizer, TRAINED_MODEL_PATH, TRAINED_NORMALIZER_NAME)


else:
    print(f"\nModule imported : {__name__}\n")

    # Loading the trained classification model for testing
    classifier = import_trained_model(TRAINED_MODEL_PATH, TRAINED_MODEL_NAME)
    if classifier is None:
        del classifier
    else:
        print("The following model is now available for testing:\n\n"
              f"{classifier}\n\n"
              f">> This model was trained on {classifier.n_features_} features:\n{get_selected_features()}\n"
              f">> Classifier imported : {TRAINED_MODEL_NAME}\n")

    # Loading the trained normalizer for testing
    normalizer = import_trained_model(TRAINED_MODEL_PATH, TRAINED_NORMALIZER_NAME)
    if normalizer is None:
        del normalizer
    else:
        print("The following model is now available for testing:\n\n"
              f"{normalizer} - Type = {type(normalizer)}\n\n"
              f">> Normalizer imported : {TRAINED_NORMALIZER_NAME}\n")

    print(f'DataFrames must be converted to numpy arrays before passing them to the models. (i.e., df.values)\n')

    # Setting up variables for testing via console
    DATA = pd.read_csv(DATA_PATH, sep='\t', index_col=0)
    if DATA_REDUCE:
        DATA = DATA.iloc[0:row_count, :]
    data_matrix = DATA[get_selected_features()+['StepLabel']].values
    X = data_matrix[:, 0:-1]
    y = data_matrix[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=101)
