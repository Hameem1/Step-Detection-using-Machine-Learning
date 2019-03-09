""" This module implements the high level logic for normalizing the dataset and data visualizations

Before running any of the code, copy the data set folder named
"OU-InertGaitAction_wStepAnnotation" into the project directory (app.py).

Then set FIX = True.
After the first run, set Fix = False.

This is required!

INFO: This module can also be used for testing the entire code. Import as: >>from app import *
"""

from threading import Thread
from graphing.data_plot import data_plot as dp
from graphing.feature_plot import feature_plot as fp
from data_generator.features import feature_extractor
from data_generator.age_comparison import gen_age_histogram
from dataset import data_structs as ds, dataset_manipulator as dm
from config import USED_CLASS_LABEL, SENSOR

# Configuration variables
# True if the Data set needs to be fixed, otherwise False
FIX = False
DATA_PLOT = False
DEVELOPER_MODE = True
DATA_VISUALIZATION = True
TEST_SUBJECT_ID = 1

if __name__ == '__main__':
    # Fixing the entire Data set
    if FIX:
        # Renaming the data set files to fix the naming scheme
        dm.dataset_rename()
        # Verifying that every subject has a data set in each sub-directory (Center, Right, Left)
        dm.dataset_analysis()

    # Demonstrating the use of the Subject class and data_plot function

    # Generating the subject list and subject data from the data set
    subs_list, subs_data = dm.generate_subjects_data(gen_csv=False)
    # Choosing a subject to get features and visualizations for
    sub = ds.Subject(subs_list[TEST_SUBJECT_ID])
    # Generating & Printing the features
    features_list, features, step_positions_actual, step_positions_updated, step_positions_updated_bool = \
        feature_extractor(sub, "right", SENSOR)

    # from data_generator.features import print_features
    # print_features(features)
    # import pandas as pd
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(features)

    # Plotting the subject data
    if DATA_VISUALIZATION:
        t1 = Thread(target=dp, args=(sub, step_positions_actual), kwargs={'sensor_axis': "all"})
        t1.start()
        # Plotting the feature data
        t2 = Thread(target=fp, args=(sub, features_list, features,
                                     step_positions_updated, step_positions_updated_bool,))
        t2.start()

    if DATA_PLOT:
        gen_age_histogram(open_plot=True)


else:
    print(f"\nModule imported : {__name__}\n")
    import pandas as pd
    from config import WINDOW_SIZE, WINDOW_TYPE

    res = str(input(f"Would you like to FIX the data set? (y/n)")).lower()
    if res == 'y':
        if DEVELOPER_MODE:
            res = str(input(f"Would you like to proceed with prompts enabled? (y/n)")).lower()
            if res == 'n':
                dm.FORCE = True
            elif res == 'y':
                dm.FORCE = False
            else:
                print("Invalid input! Please run the program again.")
        if dm.FORCE:
            res = 'y'
        else:
            res = str(input(f"Would you like to Fix the data set in folder '{dm.FOLDER_NAME}'?")).lower()
        if res == 'n':
            print("Program terminated by the user...")
        elif res == 'y':
            # Renaming the data set files to fix the naming scheme
            dm.dataset_rename()
            # Verifying that every subject has a data set in each sub-directory (Center, Right, Left)
            dm.dataset_analysis()
            # Generating the subject list (Global variable) for the data set
            subs_list, subs_data = dm.generate_subjects_data()
            # Generating a sample 'Subject' class object
            sub = ds.Subject(subs_list[TEST_SUBJECT_ID])
        else:
            print("Invalid input! Please run the program again.")

    elif res == 'n':
        # Generating the subject list (Global variable) for the data set
        subs_list, subs_data = dm.generate_subjects_data(gen_csv=False)
        # Generating a sample 'Subject' class object
        sub = ds.Subject(subs_list[TEST_SUBJECT_ID])
        print(f'\nThe "Subject" class object "sub" has been created for testing.\n')
        sensor_pos = 'right'
        sensor_type = SENSOR
        data = sub.sensor_pos[sensor_pos].label[USED_CLASS_LABEL]

        res = str(input(f"Which output type for feature_extractor()? (df/dict)")).lower()
        if res == 'df':
            col_names, df, step_positions_actual, step_positions_updated, step_positions_updated_bool \
                = feature_extractor(sub, sensor_pos, sensor_type, output_type='df')
            print(f'\n"col_names", "df", "step_positions_actual", "step_positions_updated" and '
                  f'"step_positions_updated_bool" have been returned after a call to feature_extractor()\n')
            print(f'\nRatio of no_step(0)/step(1) for subject - {sub.subject_id[-4]} = '
                  f'{len(df[df["StepLabel"]==0]) / len(df[df["StepLabel"]==1])}\n')

        elif res == 'dict':
            features_list, features, step_positions_actual, step_positions_updated, step_positions_updated_bool = \
                feature_extractor(sub, "right", SENSOR)
            print(f'\n"features_list", "features", "step_positions_actual", "step_positions_updated" and '
                  f'"step_positions_updated_bool" have been returned after a call to feature_extractor()\n')
        else:
            print("Invalid input! Please run the program again.")

    else:
        print("Invalid input! Please run the program again.")
