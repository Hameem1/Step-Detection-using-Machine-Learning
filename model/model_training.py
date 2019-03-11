"""
This module simply trains and tests a Random Forest classifier with the data from the specified dataset.
N top features are used for the training.

"""

from model.feature_selection_complete import normalize, import_trained_model, export_trained_model, print_scores
from config import data_files_path
from model_config import *

N = 20

if __name__ == '__main__':
    # Preparing the Data
    # starting timer
    start = time()
    # loading in the entire actual dataset
    print(f'{DATA_PATH}\n')
    DATA = pd.read_csv(DATA_PATH, sep='\t', index_col=0)
    # Selecting best N features
    fr = pd.read_csv(data_files_path + '\\feature ranking.csv', sep='\t', index_col=0)
    top_N_features = list(fr['Feature'][:N])
    DATA = DATA[top_N_features]
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

    # Initializing the classifier
    model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, n_jobs=1, verbose=True)
    # Training the classifier
    print('>> Training the model\n')
    model.fit(X_train, y_train)
    # Normalizing the testing data
    X_test = normalizer.transform(X_test)
    # Testing the model
    print('>> Testing the model\n')
    y_pred = model.predict(X_test)
    # Summarizing test scores
    print_scores(y_test, y_pred)
    # Stopping the timer
    duration = time() - start
    print('Operation took:', f'{duration:.2f} seconds.\n' if duration < 60 else f'{duration / 60:.2f} minutes.\n')

else:
    print(f"\nModule imported : {__name__}\n")
