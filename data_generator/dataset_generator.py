"""Do NOT import this module, Run this directly after adjusting the Configuration variables"""

# TODO : Add a step column to the features dataset and calculate again

import os
import re
from time import time
from shutil import copyfile
from dataset.data_structs import Subject
from data_generator.features import feature_extractor
from multiprocessing import Pool, current_process
from dataset.dataset_manipulator import ROOT, sensors, read_csv, generate_subjects_data, data_files_path


# Configuration Variables
# ------------------------
NEW_DATASET = "Step_Detection_Dataset"
GENERATE_DATASET = False
SORT_BY_AGE = False
TESTING = False
TEST_COUNT = 6  # Should be >= 4
ageGroups = ['(1-7)', '(8-13)', '(14-20)']
# ------------------------

if not TESTING:
    NEW_DATASET = NEW_DATASET
else:
    NEW_DATASET = NEW_DATASET+"_TEST"

datasets_dir = f"{ROOT}\\..\\DATASETS"
if not os.path.exists(datasets_dir):
    print(f'\nWARNING: The path does not exist. Creating new directory...\n{datasets_dir}\n')
    os.mkdir(datasets_dir)

# Paths to C, L and R in the NEW Dataset
new_sensor_paths = [f"{datasets_dir}\\{NEW_DATASET}\\{sensor}" for sensor in sensors]
age_dirs = {"Age_" + dirName: f'{datasets_dir}\\{NEW_DATASET}_Age_Sorted\\Age_{dirName}' for dirName in ageGroups}
# Paths to C, L and R in the age folders
sensor_dirs = {"Age_" + dirName
               : [f'{datasets_dir}\\{NEW_DATASET}_Age_Sorted\\Age_{dirName}\\{sensor}' for sensor in sensors]
               for dirName in ageGroups}


def create_dataset_folder_structure():
    path = f'{datasets_dir}\\{NEW_DATASET}'
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


def create_dataset(subs_list, indexing=True):
    start = time()
    repo = (Subject(sub) for sub in subs_list)
    for sub in repo:
        for i in range(3):
            filePath = f'{new_sensor_paths[i]}\\' + sub.subject_id[:-4] + ".csv"
            if not os.path.exists(filePath):
                # Most expensive line of code in the module
                features_list, features = feature_extractor(sub, sensors[i].lower(), "acc", output_type='df')
                features.to_csv(filePath, sep="\t", index=indexing)
                print(f"File generated - '{sub.subject_id[:-4]}.csv' by process : {current_process().name}")
            else:
                print(f'File "{sub.subject_id[:-4]}.csv" already exists!')

    print(f'\nTime taken by - {current_process().name} : {time()-start:.2f} secs')


def create_age_folder_structure():
    try:
        new_dataset_path = f'{datasets_dir}\\{NEW_DATASET}_Age_Sorted'
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
                if os.path.exists(f'{src}\\{filename[:-4]}'+'.csv'):
                    # copy it to the destination directory
                    copyfile(f'{src}\\{filename[:-4]}'+'.csv', f'{dest}\\{filename[:-4]}'+'.csv')
                    if temp == sortedCount:
                        sortedCount += 1
                        subjectCount += 1
                        # print(f'src = {src}\ndest = {dest}\n\n')

        print(f'\n# of Subjects in "{target_folder}" = {subjectCount}')

    print(f'\nTotal subjects sorted = {sortedCount}  ({round((sortedCount/len(data))*100, 2)}% of total data)\n')


if __name__ == '__main__':
    nProcesses = os.cpu_count()
    if create_dataset_folder_structure():
        if GENERATE_DATASET:
            subs_list, subs_data = generate_subjects_data(gen_csv=False)
            if TESTING:
                subs_list = subs_list[0:TEST_COUNT]
            f = lambda A, n=int(len(subs_list)/nProcesses): [A[i:i + n] for i in range(0, len(A), n)]
            s_list = f(subs_list)

            print(f'Running multi-processing operation:\n\n'
                  f'Total # of subjects = {len(subs_list)}\n'
                  f'Subjects per process = {len(subs_list)/nProcesses}')

            start = time()
            pool = Pool(processes=nProcesses)
            for output in pool.map(create_dataset, s_list):
                pass

            print(f'\n\nTime taken for all {len(subs_list)} subjects = {time()-start:.2f} secs')
            print(f'\nTime required per Subject = {(time()-start)/len(subs_list):.2f} secs')

    else:
        print("\nERROR occurred while creating the new Dataset's directory structure!")

    if SORT_BY_AGE:
        if create_age_folder_structure():
            print("\nFolder structure successfully created!\n")
            start = time()
            sort_dataset_by_age()
            print(f'Dataset "{NEW_DATASET}" sorted by Age. Operation took {time()-start:.2f} secs.')
