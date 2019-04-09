"""
This module implements Model Based Feature Selection (with cross-validated selection of best # of features).
The model used is a Random Forest classifier.
The model is trained during the feature selection phase and is tested at the end on test data using the best features.

Notes
-----
This module/algorithm provides:
1)  A ranking of all the features and the predicted model accuracy for using n-features - saved in the data_files_path.
2)  A plot of "# of features used" vs "Model Performance" - saved in the data_files_path.
3)  A .csv file containing a list of the features selected for optimal model performance - saved in the data_files_path.
4)  Two exported trained models, one classifier and one normalizer - saved in the Trained_Models directory.
5)  Option to import this module and begin testing the pre-trained models (set TESTING = True - in model_config.py)

- The results have proved that the model has the highest accuracy when trained on all 51 features.
- The model stats on training/testing with a ratio of 1:1 on the entire feature extracted dataset_operations yielded:
    Accuracy    = 92.865%
    Precision   = 90.040%
    Recall      = 92.205%
    F1-score    = 91.110%
    ROC_AUC     = 92.752%

"""

# Todo: Organize results
# Todo: Refactor the variable names involving paths, used in config.py and model.py

# Imports
import plotly.offline as pyo
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, \
    roc_auc_score, classification_report
from config import data_files_path, Path
from model_config import *

# Configuration Variables
# Test on a separate dataset_operations
DISJOINT_TESTING = False
# Path for disjoint test dataset_operations
TEST_DATA_PATH = Path(f"{ROOT}/Features_Dataset/ds_left.csv")


def create_dir(path, suppress_print=False):
    if not os.path.exists(path):
        if not suppress_print:
            print(f'WARNING: The path does not exist. Creating new directory...\n{path}\n')
        os.mkdir(path)
    else:
        if not suppress_print:
            print(f"Path for '{path}' already exists!\n")
        else:
            pass


def plot_n_features_vs_score(grid_scores, mp_lib=False):
    """
    Creates a plot for 'No. of features used' vs 'Model Performance'.
    It is saved in 'data_files_path'.

    Parameters
    ----------
    grid_scores
        Result of model.grid_scores_
    mp_lib : bool
        Set to True if also plotting with matplotlib (default = False)

    """
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
    create_dir(data_files_path, suppress_print=True)
    pyo.plot(fig, filename=f'{data_files_path}/Number of features vs Model score.html', auto_open=False)
    print(f'>> File Generated : Number of features vs Model score' + '.html\n')

    if mp_lib:
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()


def get_selected_features():
    """
    Returns a list of selected features read in from the 'features selected.csv' file.

    Returns
    -------
    list : selected features

    """
    path = f'{data_files_path}/features selected.csv'
    f_sel = pd.read_csv(path, sep='\t', index_col=0)
    return list(f_sel['selected features'])


def import_trained_model(dir_path, name):
    """
    Imports a trained model from the given directory.

    Parameters
    ----------
    dir_path : str
        Directory of the saved model.
    name : str
        Name of the saved model.

    Returns
    -------
    None or Model

    """
    if os.path.exists(f"{dir_path}/{name}"):
        path = f'{dir_path}/{name}'
        ret_model = joblib.load(path)
        print('>> Model Imported.\n')
        return ret_model
    else:
        print(f'The file {name} does not exist in the directory : "{TRAINED_MODEL_DIR}"\n')
        return None


def export_trained_model(model, dir_path, name):
    """
    Imports a trained model to the given directory.

    Parameters
    ----------
    model : The trained model to be exported
    dir_path : Target directory
    name : File name to save the model as

    """
    # Creating the directory for the trained model
    create_dir(TRAINED_MODEL_PATH)

    # Saving the model externally in TRAINED_MODEL_PATH
    path = f'{dir_path}/{name}'
    joblib.dump(model, path)
    print(f'>> Model stored externally as "{name}"\n')


def normalize(x_train):
    """
    Trains a normalizer with the given data and return the normalized data along with the normalizer model.

    Parameters
    ----------
    x_train : np.Array
        Data to train the model on

    Returns
    -------
    np.Array : Normalized data
    Model : Trained normalization model

    """
    # Training the normalizer
    norm_model = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    # Normalizing the training data
    x_train = norm_model.transform(x_train)
    print('>> Training set normalized.\n')
    return x_train, norm_model


def print_scores(y_test, y_pred):
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n')
    print(f'Score of the classifier on test data:\n'
          f'Accuracy  = {accuracy_score(y_test, y_pred) * 100:.3f}%\n'
          f'Precision = {precision_score(y_test, y_pred) * 100:.3f}%\n'
          f'Recall    = {recall_score(y_test, y_pred) * 100:.3f}%\n'
          f'F1-score  = {f1_score(y_test, y_pred) * 100:.3f}%\n'
          f'ROC_AUC   = {roc_auc_score(y_test, y_pred) * 100:.3f}%\n\n'
          f'Classification Report: \n{classification_report(y_test, y_pred)}\n\n')


if __name__ == '__main__':
    # Preparing the Data
    # starting timer
    start = time()
    print(f'\nProcess started at :\n\nDate  :  {dt.today().strftime("%x")}\nTime  :  {dt.today().strftime("%X")}\n')
    # loading in the entire actual dataset_operations
    print('>> Loading the dataset_operations\n')
    print(f'Location : {DATA_PATH}\n')
    DATA = pd.read_csv(DATA_PATH, sep='\t', index_col=0)
    # limiting the # of rows used
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

    # Normalizing the training data
    if DATA_NORMALIZATION:
        X_train, normalizer = normalize(X_train)

    # Feature selection
    # Recursive Feature Elimination/Model Based Feature Selection (with cross-validated selection of best # of features)
    classifier = RandomForestClassifier(n_estimators=RF_ESTIMATORS, n_jobs=N_JOBS, verbose=VERBOSE)
    rfecv = RFECV(estimator=classifier, cv=StratifiedKFold(K_FOLD), scoring=SCORING, n_jobs=N_JOBS, verbose=VERBOSE)
    print('>> Training the model & Performing feature ranking simultaneously\n')
    fit = rfecv.fit(X_train, y_train)
    print('>> Model Trained!\n')
    print('>> Feature Ranking complete!\n')

    # Summarizing the feature ranking
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

    # Generating .csv files from feature ranking and selected features
    if GEN_RANKING_FILE:
        # Creating the data files directory incase it doesn't exist already
        create_dir(data_files_path)
        ranking.to_csv(f'{data_files_path}/feature ranking.csv', sep="\t",
                       index=True, index_label='No. of features')
        print(f'>> File generated : feature ranking.csv\n')
        features_sel_df = pd.DataFrame({'selected features': feature_names})
        features_sel_df.to_csv(f'{data_files_path}/features selected.csv', sep="\t", index=True)
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
    # Testing the model with the test set
    y_pred = rfecv.predict(X_test)
    # Printing model scores
    print_scores(y_test, y_pred)
    # Stopping the timer
    duration = time() - start
    print('Operation took:', f'{duration:.2f} seconds.\n' if duration < 60 else f'{duration / 60:.2f} minutes.\n')
    print(f'\nProcess ended at :\n\nDate  :  {dt.today().strftime("%x")}\nTime  :  {dt.today().strftime("%X")}\n')

    # Converting a selected section of the dataset_operations to a numpy array (based on best features)
    data_matrix = DATA[get_selected_features()+['StepLabel']].values
    X = data_matrix[:, 0:-1]
    y = data_matrix[:, -1]
    # Splitting the data into training and testing splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=101)
    # Re-training the normalizer with updated features (because they may have been reduced)
    if DATA_NORMALIZATION:
        print('>> Re-training the Normalizer\n')
        X_train, normalizer = normalize(X_train)
        print('>> Normalizer re-trained\n')

    # Exporting the trained classifier and normalizer
    if EXPORT_MODEL:
        export_trained_model(rfecv, TRAINED_MODEL_PATH, TRAINED_MODEL_NAME)
        if DATA_NORMALIZATION:
            export_trained_model(normalizer, TRAINED_MODEL_PATH, TRAINED_NORMALIZER_NAME)


else:
    if TESTING:
        # To OVERRIDE the model names
        # TRAINED_MODEL_NAME = 'step_detection_model.pkl'
        # TRAINED_NORMALIZER_NAME = 'step_detection_min_max_norm.pkl'

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
        print(f'The variables : DATA, data_matrix, X, Y, X_train, X_test, y_train, y_test have been generated from:\n'
              f'Dataset = {DATA_PATH}\n\n'
              f'The following models are also available:\n'
              f'classifier (The trained classifier)\n'
              f'normalizer (The trained normalizer)\n')
        print()

    else:
        print(f"\nModule imported : {__name__}\n")
