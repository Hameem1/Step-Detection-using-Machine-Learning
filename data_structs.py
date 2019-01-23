"""This module implements the data structures used for this project"""

import os
import pandas as pd
from dataset_manipulator import get_subjects_list
# folder which contains the entire Data set
from dataset_manipulator import FOLDER_NAME

# Configuration variables
# set to "True" for verbose status messages
FILE_STATUS_MESSAGES = False
# storage selector for SUBJECTS_REPO (Dict/List)
STORAGE = "List"


# Every label in "labels" contains data which belongs to it's 'ClassLabel'
# and is not null valued (all zeros)
# The order of the elements is important
labels = ["invalid", "level", "upstairs", "downstairs", "incline", "decline"]
# Labels in "labels_extra" are classified as follows:
# "complete"  : The entire Data set, as it is.
# "valid"     : Consists of data which is not "invalid" or null valued.
# "null_data" : Consists of only the null values in the Data set.
labels_extra = ["complete", "valid", "null_data"]

cwd = os.getcwd()
sensor_paths = []
sensors = ["Center", "Left", "Right"]
for sensor in sensors:
    sensor_paths.append(f"{cwd}\\{FOLDER_NAME}\\{sensor}")


class Dataset:
    """
    This class contains the data imported from the file whose name/path
    is passed in as a parameter. The data is separated into multiple Pandas
    data frames according to their ClassLabel value.

    :param path: The path of the file to create the Dataset object from

    - the path must have .txt appended at the end of the file name

    """

    def __init__(self, path):
        self.fs = 100

        try:
            # Reading in a Single Dataset file
            # sep = "\t" because the dataset is delimited by a tab space
            # Header = 1 : because the first line is a header
            # index_col = False : because the dataset is malformed (delimiter at the end of each row)
            data_table = pd.read_csv(path, sep="\t", header=1, index_col=False)
        except FileNotFoundError:
            print("The file name could not be found, please make sure the file name/path is correct.")
        except:
            print("There is a problem with the 'Dataset' object instantiation!\n")
        else:
            # Filtering the Dataset into different sections contained in "labels" and "labels_extra"
            self.label = {}
            for label in labels:
                self.label.update({label: data_table[(data_table.ClassLabel == labels.index(label) - 1) &
                                                     (data_table.Gx != 0) & (data_table.Ax != 0) &
                                                     (data_table.Gy != 0) & (data_table.Ay != 0) &
                                                     (data_table.Gz != 0) & (data_table.Az != 0)]
                                  .reset_index(drop=True)})

            self.label.update({labels_extra[0]: data_table})

            df1 = data_table[(data_table.ClassLabel != -1) & (data_table.Gx != 0) & (data_table.Gy != 0) &
                             (data_table.Gz != 0) & (data_table.Ax != 0) & (data_table.Ay != 0) &
                             (data_table.Az != 0)].reset_index(drop=True)

            self.label.update({labels_extra[1]: df1})

            df2 = data_table[(data_table.Gx == 0) & (data_table.Gy == 0) & (data_table.Gz == 0) &
                             (data_table.Ax == 0) & (data_table.Ay == 0) & (data_table.Az == 0)].reset_index(drop=True)

            self.label.update({labels_extra[2]: df2})

            if FILE_STATUS_MESSAGES:
                print(f"'Dataset' object created for file :\n{path}\n")

    def __str__(self):
        label_name = str(input('Which "label" would you like to print for this dataset?')).lower()
        if label_name in labels or label_name in labels_extra:
            return self.label[label_name].to_string()
        else:
            return "Invalid Input!"


class Subject:
    """
    This class contains the entire data on one subject (from all three sensors).
    Each sensor's data is represented as a Dataset class object.

    :param subject_id: the subject id (e.g. 'Idxxxxxx.txt')

    """

    def __init__(self, subject_id):
        try:
            os.chdir(cwd)
            self.subject_id = subject_id
            self.sensor_pos = {"center": Dataset(f"{sensor_paths[0]}\\{subject_id}"),
                               "left": Dataset(f"{sensor_paths[1]}\\{subject_id}"),
                               "right": Dataset(f"{sensor_paths[2]}\\{subject_id}")}

            if FILE_STATUS_MESSAGES:
                print(f"'Subject' object created for file : {subject_id}\n")

        except:
            print("There is a problem with the 'Subject' object instantiation!\n")

    def __str__(self):
        return f"Subject # {self.subject_id[0:8]}"


# TODO : this function could get a speed boost if converted to a generator


def get_subjects_repo():
    """
    Returns a list or dict of Subject class objects for every file in the data set

    :returns subjects_repo: list or dict (configurable via the global STORAGE variable)

    """
    from dataset_manipulator import get_subjects_list
    subjects = get_subjects_list()

    # SUBJECTS_REPO contains a list (or dict) of Subject objects for every file in the data set
    if STORAGE == "List":
        SUBJECTS_REPO = []
    else:
        SUBJECTS_REPO = {}

    for sub in subjects:
        if STORAGE == "List":
            SUBJECTS_REPO.append(Subject(sub))
        else:
            SUBJECTS_REPO.update({sub: Subject(sub)})

    print(f"SUBJECTS_REPO returned as a {STORAGE}")
    return SUBJECTS_REPO


def print_selection(selection):
    """
    Prints an entire selection of the Dataset

    :param selection: The selection to print (e.g. sub.sensor_pos['left'].label['valid'])
    """

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(f"The selected data is :\n\n{selection}")


if __name__ == "__main__":

    # This example code shows how to interact with the classes

    # Getting the subject list and repo
    subject_list = get_subjects_list()
    subject_repo = get_subjects_repo()

    s = Subject("Id000000.txt")
    # s = subject_repo[0]   #this would also work

    sensor_pos = "center"
    motion_type = "valid"

    if STORAGE == "List":
        print(subject_repo[0].sensor_pos[sensor_pos].label[motion_type])
    else:
        print(subject_repo[subject_list[0]].sensor_pos[sensor_pos].label[motion_type])

else:
    print(f"\nModule imported : {__name__}\n")
