"""
This module should not be imported, it is meant to be run directly after adjusting the Configuration variables.

"""

# Imports
import os
import re
from time import time
from shutil import copyfile
from dataset_operations.data_structs import Subject
from multiprocessing import Pool, current_process
from data_generator.features import feature_extractor
from dataset_operations.dataset_manipulator import read_csv, generate_subjects_data
from config import ageGroups, FEATURES_DATASET, DATASETS, age_dirs, sensor_dirs, data_files_path, sensors, Path

# Configuration Variables
# ------------------------
GENERATE_DATASET = True
SORT_BY_AGE = False
TESTING = True
TEST_COUNT = 8  # Should be >= 4
# ------------------------

if not TESTING:
    FEATURES_DATASET = FEATURES_DATASET
else:
    FEATURES_DATASET = FEATURES_DATASET + "_TEST"

new_sensor_paths = [Path(f"{DATASETS}/{FEATURES_DATASET}/{sensor}") for sensor in sensors]

if not os.path.exists(DATASETS):
    print(f'\nWARNING: The path does not exist. Creating new directory...\n{DATASETS}\n')
    os.mkdir(DATASETS)


def create_dataset_folder_structure():
    """
    Creates the folder structure for the new dataset_operations.

    """

    path = Path(f'{DATASETS}/{FEATURES_DATASET}')
    if not os.path.exists(path):
        print(f'\nWARNING: The path does not exist. Creating new directory...\n{path}\n')
        os.mkdir(path)

    try:
        for path in new_sensor_paths:
            if not os.path.exists(path):
                print(f'\nWARNING: The path does not exist. Creating new directory...\n{path}\n')
                os.mkdir(path)
            else:
                print("\nPath already exists!")
    except:
        return False
    else:
        return True


def create_age_folder_structure():
    """
    Creates the folder structure for the Age Sorted Dataset.

    """

    try:
        new_dataset_path = Path(f'{DATASETS}/{FEATURES_DATASET}_Age_Sorted')
        if not os.path.exists(new_dataset_path):
            print(f'\nWARNING: The path does not exist. Creating new directory...\n{new_dataset_path}\n')
            os.mkdir(new_dataset_path)
    except:
        print("ERROR in creating the sorted dataset_operations directory within folder /Data Sets")
        return False

    try:
        for folder, age_dir in age_dirs.items():
            if not os.path.exists(age_dir):
                os.mkdir(age_dir)
            else:
                print(f"The directory {folder} already exists.")
    except:
        print("ERROR in creating age based directories in /Data Sets/Dataset_Age_Sorted")
        return False

    try:
        for sub_folder, sensor_dir in sensor_dirs.items():
            for sub_path in sensor_dir:
                if not os.path.exists(sub_path):
                    os.mkdir(sub_path)
                else:
                    print(f"The directory {sub_path} already exists.")
        return True
    except:
        print("ERROR in creating sensor directories in /Data Sets/Dataset_Age_Sorted/[age_Groups]")
        return False


def get_limits(age_groups):
    """
    Generates numerical limits from string representations.

    Parameters
    ----------
    age_groups: list of str
        ['(2-3)','(6-7)', ...]

    Returns
    -------
    limits : dict
        {'Age(X-Y)': [min, max], ...}

    """

    limits = {}
    for data in age_groups:
        pattern = re.compile(r'([\d]+)-([\d]+)')
        match = pattern.search(data)
        age_min = int(match.group(1).strip())
        age_max = int(match.group(2).strip())
        # print(f'limits = {age_min} to {age_max}')
        limits[f'Age_{data}'] = [age_min, age_max]
    return limits


def sort_dataset_by_age():
    """
    Sorts the Dataset created by create_dataset() into a new Age sorted Dataset.

    """

    data = read_csv(Path(f'{data_files_path}/subject_data'))
    limits = get_limits(ageGroups)
    sortedCount = 0

    # For every age bin
    for target_folder, limit in limits.items():
        # Get the indexes of all files to be copied to the target folder
        index_list = list(data[(data['Age'] >= limit[0]) & (data['Age'] <= limit[1])].index)
        subjectCount = 0
        # For every file to be copied
        for i in index_list:
            filename = data.iloc[i]['Filename']
            temp = sortedCount
            # Get the source and destination file paths
            for src, dest in zip(new_sensor_paths, sensor_dirs[target_folder]):
                # if the file exists in the source directory
                if os.path.exists(Path(f'{src}/{filename[:-4]}.csv')):
                    # copy it to the destination directory
                    copyfile(Path(f'{src}/{filename[:-4]}.csv'), Path(f'{dest}/{filename[:-4]}.csv'))
                    if temp == sortedCount:
                        sortedCount += 1
                        subjectCount += 1
                        # print(f'src = {src}\ndest = {dest}\n\n')

        print(f'\n# of Subjects in "{target_folder}" = {subjectCount}')

    print(f'\nTotal subjects sorted = {sortedCount}  ({round((sortedCount / len(data)) * 100, 2)}% of total data)\n')


def create_dataset(subs_list, indexing=True):
    """
    Creates the New Dataset using features calculated from the base data.

    Parameters
    ----------
    subs_list : list
        list of subjects to create the new dataset_operations for
    indexing : bool, optional
        dataset_operations index column (default = True)

    """

    S = None
    print(f'\nProcess - {current_process().name} has {len(subs_list)} files to work on.\n')

    try:
        start = time()
        repo = (Subject(sub) for sub in subs_list)
        for sub in repo:
            S = sub
            for i in range(3):
                filePath = Path(f'{new_sensor_paths[i]}/{sub.subject_id[:-4]}.csv')
                if not os.path.exists(filePath):
                    # Most expensive line of code in the module (Takes hours)
                    col_names, df, _, _, _ = feature_extractor(sub, sensors[i].lower(), output_type='df')
                    df.to_csv(filePath, sep="\t", index=indexing)
                    print(f"File generated - '{sub.subject_id[:-4]}.csv' by process : {current_process().name}")
                else:
                    print(f'File "{sub.subject_id[:-4]}.csv" already exists!')

        print(f'\nTime taken by - {current_process().name} : {time() - start:.2f} secs')
    except Exception as e:
        print(f"Exception occurred in {current_process().name}\n")
        print(f'While working on this portion of the subs_list:\n'
              f'{subs_list}')
        print(f'Error occurred in FILE # {S.subject_id}\n')
        raise e


def file_exists(subs_list):
    """
    Checks to see if any previous files with feature extracted data exist in the Dataset and returns the
    updated list of files which don't exist in the Dataset.

    This is done because generating the files is expensive and this avoids having to start over from scratch.

    Parameters
    ----------
    subs_list : list
        Complete subjects list

    Returns
    -------
    updated_subs : list
        list of subject files which are not already in the new Dataset

    """
    updated_subs = []
    print(f'Checking for existing files in directories:\n')
    for dir in new_sensor_paths:
        print(f'{dir}')
        updated_subs += subs_list
    print()

    for sub in subs_list:
        for i in range(3):
            filePath = Path(f'{new_sensor_paths[i]}/{sub[:-4]}.csv')
            if not os.path.exists(filePath):
                pass
            else:
                updated_subs.pop(updated_subs.index(sub))
    updated_subs = list(sorted(set(updated_subs)))
    print(f'There were {len(subs_list) - len(updated_subs)} existing files!\n')
    print(f'The updated subjects list now contains {len(updated_subs)} entries.\n')
    return updated_subs


if __name__ == '__main__':
    # no. of parallel processes equals the available no. of CPU cores
    nProcesses = os.cpu_count()
    # Creating folder structure for new Dataset
    if create_dataset_folder_structure():
        if GENERATE_DATASET:
            subs_list, subs_data = generate_subjects_data(gen_csv=False)
            subs_list = list(file_exists(subs_list))
            if TESTING:
                subs_list = subs_list[0:TEST_COUNT]
            # Dividing up the subject list for each available process
            f = lambda A, n=int(len(subs_list) / nProcesses): [A[i:i + n] for i in range(0, len(A), n)]
            s_list = f(subs_list)

            print(f'Running multi-processing operation:\n\n'
                  f'Total # of subjects = {len(subs_list)}\n'
                  f'Subjects per process = {len(subs_list) / nProcesses}')

            start = time()
            # Generating processes from a pool
            pool = Pool(processes=nProcesses)
            # Each process works on creating the dataset_operations for it's own subset of the subs_list
            for output in pool.map(create_dataset, s_list):
                pass

            duration = time() - start
            print(f'\n\nTime taken for all {len(subs_list)} subjects = ',
                  f'{duration:.2f} seconds.'if duration < 60 else f'{duration/60:.2f} minutes.')
            per_sub = duration / len(subs_list)
            print(f'\nTime required per Subject = ',
                  f'{per_sub:.2f} seconds.' if per_sub < 60 else f'{per_sub/60:.2f} minutes.')

    else:
        print("\nERROR occurred while creating the new Dataset's directory structure!")

    if SORT_BY_AGE:
        # Creating folder structure for the Age sorted Dataset
        if create_age_folder_structure():
            print("\nFolder structure successfully created!\n")
            start = time()
            # Sorting the dataset_operations
            sort_dataset_by_age()
            duration = time() - start
            print(f'Dataset "{FEATURES_DATASET}" sorted by Age.\n',
                  'Operation took:', f'{duration:.2f} seconds.' if duration < 60 else f'{duration/60:.2f} minutes.')
