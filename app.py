# TODO: Perform feature ranking on the data (Try to take cues from the visualization)

# TODO: Implement a technique to put together all the data required for the ML model
#       Hint - processing for segments of the entire Dataset folder would be helpful
#       Hint - Using threads would be helpful


""" This module implements the high level logic

Before running any of the code, copy the data set folder named
"OU-InertGaitAction_wStepAnnotation" into the project directory (app.py).

Then set FIX = True.
After the first run, set Fix = False.

This is required!

"""
import dataset_manipulator as dm
import data_structs as ds
# import dataset_generator as dg
from data_plot import data_plot as dp
from feature_plot import feature_plot as fp
from features import feature_extractor, print_features
from threading import Thread

# Configuration variables
# True if the Data set needs to be fixed, otherwise False
FIX = False
DEVELOPER_MODE = True
DATA_PLOT = False

# Fixing the entire Data set
if FIX:
    # Renaming the data set files to fix the naming scheme
    dm.dataset_rename()
    # Verifying that every subject has a data set in each sub-directory (Center, Right, Left)
    dm.dataset_analysis()


if __name__ == '__main__':
    # Demonstrating the use of the Subject class and data_plot function

    # Generating the subject list and subject data from the data set
    subs_list, subs_data = dm.generate_subjects_data(gen_csv=False)
    sub = ds.Subject("Id000104.txt")
    # Generating & Printing the features
    features_list, features = feature_extractor(sub, "right", "acc")
    # print_features(features)
    # Plotting the subject data
    if DATA_PLOT:
        t1 = Thread(target=dp, args=(sub,), kwargs={'sensor_axis': "all"})
        t1.start()
        # Plotting the feature data
        t2 = Thread(target=fp, args=(sub, features_list, features,))
        t2.start()

    # dg.create_dataset(subs_list)

else:
    print(f"\nModule imported : {__name__}\n")

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
            sub = ds.Subject("Id000104.txt")
        else:
            print("Invalid input! Please run the program again.")

    elif res == 'n':
        # Generating the subject list (Global variable) for the data set
        subs_list, subs_data = dm.generate_subjects_data()
        # Generating a sample 'Subject' class object
        sub = ds.Subject("Id000104.txt")
        print(f'\nThe "Subject" class object "sub" has been created for testing.\n')

    else:
        print("Invalid input! Please run the program again.")

