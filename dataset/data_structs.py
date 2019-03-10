"""
This module implements the data structures used to store the dataset.

"""

import pandas as pd
from dataset.dataset_manipulator import get_subjects_list, sensor_paths
from config import FILE_STATUS_MESSAGES, STORAGE, Fs

# Every label in "labels" contains data which belongs to it's 'ClassLabel'
# and is not null valued (all zeros)
# The order of the elements is important
labels = ["invalid", "level", "upstairs", "downstairs", "incline", "decline"]
# Labels in "labels_extra" are classified as follows:
# "complete"  : The entire Data set, as it is.
# "valid"     : Consists of data which is not "invalid" or null valued.
# "null_data" : Consists of only the null values in the Data set.
labels_extra = ["complete", "valid", "null_data"]


class Dataset:
    """
    This class contains the data imported from the given file.
    The data is separated into multiple Pandas data frames according to their ClassLabel value.

    Parameters
    ----------
    path : str
        The path of the file to create the Dataset object from (with .txt)

    Attributes
    ----------
    fs : int
        Sampling frequency for the data set
    label : {"valid", "invalid", "level", "upstairs", "downstairs", "incline", "decline", "complete", "null_data"}

    """

    def __init__(self, path):

        self.fs = Fs

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
                                                     ((data_table.Gx != 0) | (data_table.Ax != 0) |
                                                      (data_table.Gy != 0) | (data_table.Ay != 0) |
                                                      (data_table.Gz != 0) | (data_table.Az != 0))]
                                  .reset_index(drop=True)})

            self.label.update({labels_extra[0]: data_table})

            df1 = data_table[(data_table.ClassLabel != -1) & ((data_table.Gx != 0) | (data_table.Gy != 0) |
                                                              (data_table.Gz != 0) | (data_table.Ax != 0) |
                                                              (data_table.Ay != 0) | (data_table.Az != 0))]\
                .reset_index(drop=True)

            self.label.update({labels_extra[1]: df1})

            df2 = data_table[(data_table.Gx == 0) & (data_table.Gy == 0) & (data_table.Gz == 0) &
                             (data_table.Ax == 0) & (data_table.Ay == 0) & (data_table.Az == 0)]\
                .reset_index(drop=True)

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
    Each sensor's data is represented as a Dataset Class object.

    Parameters
    ----------
    filename : str
        The filename to create the Subject from (e.g. 'Idxxxxxx.txt')

    Attributes
    ----------
    subject_id : str
        Filename of the Subject (e.g. "Idxxxxxx.txt")
    sensor_pos : dict of Dataset
        'center', 'left' or 'right'

    """

    def __init__(self, filename):
        try:
            self.subject_id = filename
            self.sensor_pos = {"center": Dataset(f"{sensor_paths[0]}\\{filename}"),
                               "left": Dataset(f"{sensor_paths[1]}\\{filename}"),
                               "right": Dataset(f"{sensor_paths[2]}\\{filename}")}

            if FILE_STATUS_MESSAGES:
                print(f"'Subject' object created for file : {filename}\n")

        except:
            print("There is a problem with the 'Subject' object instantiation!\n")

    def __str__(self):
        return f"Subject # {self.subject_id[0:8]}"


def get_subjects_repo(subs_list, storage='list'):
    """
    Returns a list or dict of Subject class objects for every file in the data set.

    Parameters
    ----------
    subs_list : list
        List of subject filenames
    storage : {'list', 'dict'}

    Returns
    -------
    SUBJECTS_REPO : list or dict
        configurable via the global STORAGE variable

    """

    # SUBJECTS_REPO contains a list (or dict) of Subject objects for every file in the data set
    if storage == "list":
        SUBJECTS_REPO = []
    else:
        SUBJECTS_REPO = {}

    for sub in subs_list:
        if storage == "list":
            SUBJECTS_REPO.append(Subject(sub))
        else:
            SUBJECTS_REPO.update({sub: Subject(sub)})

    print(f"SUBJECTS_REPO returned as a {storage}")
    return SUBJECTS_REPO


if __name__ == "__main__":
    # This example code shows how to interact with the classes

    # Getting the subject list and repo
    subject_list = get_subjects_list()
    subject_repo = get_subjects_repo(subject_list, storage='list')

    s = Subject("Id000000.txt")
    # s = subject_repo[0]   #this would also work

    sensor_pos = "center"
    motion_type = "valid"

    if STORAGE == "List":
        print(subject_repo[0].sensor_pos[sensor_pos].label[motion_type])
    else:
        print(subject_repo[subject_list[0]].sensor_pos[sensor_pos].label[motion_type])

else:
    print(f"\nModule imported : {__name__}")
