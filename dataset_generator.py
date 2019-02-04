import os
from time import time
from data_structs import Subject
from features import feature_extractor
import dataset_manipulator as dm
from multiprocessing import Pool, current_process

NEW_FOLDER_NAME = "Step_Detection_Dataset_TEST"
project_dir = os.getcwd()
os.chdir("..")
datasets_dir = f"{os.getcwd()}\\DATASETS"
os.chdir(project_dir)
if not os.path.exists(datasets_dir):
    os.mkdir(datasets_dir)
sensors = ["Center", "Left", "Right"]
new_sensor_paths = [f"{datasets_dir}\\{NEW_FOLDER_NAME}\\{sensor}" for sensor in sensors]


def create_folder_structure():
    path = f'{datasets_dir}\\{NEW_FOLDER_NAME}'
    if not os.path.exists(path):
        os.mkdir(path)

    try:
        for path in new_sensor_paths:
            if not os.path.exists(path):
                print(f'\nWARNING: The path {path} does not exist. Creating new directory...')
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

    print(f'\nTime taken by - {current_process().name} : {time()-start} secs')


if __name__ == '__main__':
    nProcesses = os.cpu_count()
    if create_folder_structure():
        subs_list, subs_data = dm.generate_subjects_data(gen_csv=False)
        subs_list = subs_list[0:4]  # Comment this line out if not Testing
        f = lambda A, n=int(len(subs_list)/nProcesses): [A[i:i + n] for i in range(0, len(A), n)]
        s_list = f(subs_list)

        print(f'Running multi-processing operation:\n\n'
              f'Total # of subjects = {len(subs_list)}\n'
              f'Subjects per process = {len(subs_list)/nProcesses}')

        start = time()

        pool = Pool(processes=nProcesses)
        for output in pool.map(create_dataset, s_list):
            pass

        print(f'\n\nTime taken for all {len(subs_list)} subjects = {time()-start}')
        print(f'\nTime required per Subject = {(time()-start)/len(subs_list)}')


    else:
        print("\nERROR occurred while creating the new Dataset's directory structure!")