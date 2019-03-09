"""This module should not be imported, it is meant to be run directly after adjusting the Configuration variables"""

# Imports
import os
import re
from time import time
from shutil import copyfile
from dataset.data_structs import Subject
from multiprocessing import Pool, current_process
from data_generator.features import feature_extractor
from dataset.dataset_manipulator import read_csv, generate_subjects_data
from config import ageGroups, DATASET_FOLDER, DATASET_ROOT, age_dirs, sensor_dirs, data_files_path, sensors

# Configuration Variables
# ------------------------
GENERATE_DATASET = True
SORT_BY_AGE = False
TESTING = True
TEST_COUNT = 4  # Should be >= 4
# ------------------------

if not TESTING:
    DATASET_FOLDER = DATASET_FOLDER
else:
    DATASET_FOLDER = DATASET_FOLDER + "_TEST"

new_sensor_paths = [f"{DATASET_ROOT}\\{DATASET_FOLDER}\\{sensor}" for sensor in sensors]

if not os.path.exists(DATASET_ROOT):
    print(f'\nWARNING: The path does not exist. Creating new directory...\n{DATASET_ROOT}\n')
    os.mkdir(DATASET_ROOT)


def create_dataset_folder_structure():
    """Creates the folder structure for the new dataset"""

    path = f'{DATASET_ROOT}\\{DATASET_FOLDER}'
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
    """Creates the folder structure for the Age Sorted Dataset"""

    try:
        new_dataset_path = f'{DATASET_ROOT}\\{DATASET_FOLDER}_Age_Sorted'
        if not os.path.exists(new_dataset_path):
            print(f'\nWARNING: The path does not exist. Creating new directory...\n{new_dataset_path}\n')
            os.mkdir(new_dataset_path)
    except:
        print("ERROR in creating the sorted dataset directory within folder \\DATASETS")
        return False

    try:
        for folder, age_dir in age_dirs.items():
            if not os.path.exists(age_dir):
                os.mkdir(age_dir)
            else:
                print(f"The directory {folder} already exists.")
    except:
        print("ERROR in creating age based directories in \\DATASETS\\Dataset_Age_Sorted")
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
        print("ERROR in creating sensor directories in \\DATASETS\\Dataset_Age_Sorted\\[age_Groups]")
        return False


def get_limits(age_groups):
    """
    Generates numerical limits from string representations

    :param age_groups: list of strings ['(2-3)','(6-7)', ...]
    :return: dict('Age(X-Y)': [min, max], ...)
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
    """Sorts the Dataset created by create_dataset() into a new Age sorted Dataset"""

    data = read_csv(f'{data_files_path}\\subject_data')
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
                if os.path.exists(f'{src}\\{filename[:-4]}' + '.csv'):
                    # copy it to the destination directory
                    copyfile(f'{src}\\{filename[:-4]}' + '.csv', f'{dest}\\{filename[:-4]}' + '.csv')
                    if temp == sortedCount:
                        sortedCount += 1
                        subjectCount += 1
                        # print(f'src = {src}\ndest = {dest}\n\n')

        print(f'\n# of Subjects in "{target_folder}" = {subjectCount}')

    print(f'\nTotal subjects sorted = {sortedCount}  ({round((sortedCount / len(data)) * 100, 2)}% of total data)\n')


def create_dataset(subs_list, indexing=True):
    """
    Creates the New Dataset using features calculated from the base data

    :param subs_list: list of subjects to create the new dataset for
    :param indexing: dataset index column (default:True)
    """

    start = time()
    repo = (Subject(sub) for sub in subs_list)
    for sub in repo:
        for i in range(3):
            filePath = f'{new_sensor_paths[i]}\\' + sub.subject_id[:-4] + ".csv"
            if not os.path.exists(filePath):
                # Most expensive line of code in the module (Takes hours)
                col_names, df, _, _, _ = feature_extractor(sub, sensors[i].lower(), output_type='df')
                df.to_csv(filePath, sep="\t", index=indexing)
                print(f"File generated - '{sub.subject_id[:-4]}.csv' by process : {current_process().name}")
            else:
                print(f'File "{sub.subject_id[:-4]}.csv" already exists!')

    print(f'\nTime taken by - {current_process().name} : {time() - start:.2f} secs')


if __name__ == '__main__':
    # no. of parallel processes equals the available no. of CPU cores
    nProcesses = os.cpu_count()
    # Creating folder structure for new Dataset
    if create_dataset_folder_structure():
        if GENERATE_DATASET:
            subs_list, subs_data = generate_subjects_data(gen_csv=False)
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
            # Each process works on creating the dataset for it's own subset of the subs_list
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
            # Sorting the dataset
            sort_dataset_by_age()
            duration = time() - start
            print(f'Dataset "{DATASET_FOLDER}" sorted by Age.\n',
                  'Operation took:', f'{duration:.2f} seconds.' if duration < 60 else f'{duration/60:.2f} minutes.')
