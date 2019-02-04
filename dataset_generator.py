import os
from time import time
from data_structs import Subject
from features import feature_extractor
import dataset_manipulator as dm
from multiprocessing import Process, current_process

NEW_FOLDER_NAME = "Step_Detection_Dataset"
cwd = os.getcwd()
sensors = ["Center", "Left", "Right"]
new_sensor_paths = [f"{cwd}\\{NEW_FOLDER_NAME}\\{sensor}" for sensor in sensors]


def create_folder_structure():
    path = f'{cwd}\\{NEW_FOLDER_NAME}'
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
                features_list, features = feature_extractor(sub, sensors[i].lower(), "acc", output_type='df')
                features.to_csv(filePath, sep="\t", index=indexing)
                print(f"File generated - '{sub.subject_id[:-4]}.csv' by process {current_process().name}")
            else:
                print(f'File "{sub.subject_id[:-4]}.csv" already exists!')
        name = sub.subject_id[:-4]

    print(f'\nTime taken for Subject - {name} : {time()-start} secs')


if __name__ == '__main__':
    nProcesses = 8

    if create_folder_structure():
        subs_list, subs_data = dm.generate_subjects_data(gen_csv=False)
        # subs_list = subs_list[0:8]  # Testing
        f = lambda A, n=int(len(subs_list)/nProcesses): [A[i:i + n] for i in range(0, len(A), n)]
        s_list = f(subs_list)
        os.chdir(cwd)

        p1 = Process(target=create_dataset, args=(s_list[0],))
        p2 = Process(target=create_dataset, args=(s_list[1],))
        p3 = Process(target=create_dataset, args=(s_list[2],))
        p4 = Process(target=create_dataset, args=(s_list[3],))
        p5 = Process(target=create_dataset, args=(s_list[4],))
        p6 = Process(target=create_dataset, args=(s_list[5],))
        p7 = Process(target=create_dataset, args=(s_list[6],))
        p8 = Process(target=create_dataset, args=(s_list[7],))
        print(f'Running multi-processing operation:\n\n'
              f'Total # of subjects = {len(subs_list)}\n'
              f'Subjects per process = {len(subs_list)/nProcesses}')

        start = time()

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        p8.join()

        print(f'\n\nTime taken for all {len(subs_list)} subjects = {time()-start}')


    else:
        print("\nERROR occurred while creating the new Dataset's directory structure!")