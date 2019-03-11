"""
This module implements Model Based Feature Selection using Random Forest classifiers.
Multiple random forests are independently trained and perform voting based feature selection.
The number of best features to select is user determined.

Warnings
--------
Deprecated: This is not as good as the method used in "feature_selection_complete.py". Use that instead.

"""

# Imports
import operator
from sklearn.feature_selection import RFE
from model.feature_selection_complete import normalize
from model_config import *

# Global configurations
# no. of best features to select
n_features = 15
# no. of different Random Forests participating in vote
r = 5
# Voting Box
nth_rank = {k: 0 for k in cols[0:-1]}
# Features obtained after feature selection
selected_features = []


if __name__ == '__main__':
    # Preparing the Data
    # starting timer
    start = time()
    # loading in the entire actual dataset for one sensor
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    if DATA_NORMALIZATION:
        X_train, normalizer = normalize(X_train)

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
    print()

    print('>> Feature selection complete\n')
    print(f'Feature selection with "{r}" separate Random Forest Classifiers complete\n')
    duration = time() - start
    print('Operation took:', f'{duration:.2f} seconds.' if duration < 60 else f'{duration/60:.2f} minutes.\n\n')

    print(f'Best {n_features} features selected:\n{selected_features}\n')

else:
    print(f"\nModule imported : {__name__}")

