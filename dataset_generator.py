import os
from time import time
from data_structs import Subject
from features import feature_extractor
from dataset_manipulator import FOLDER_NAME

NEW_FOLDER_NAME = "Step_Detection_Dataset"
cwd = os.getcwd()
new_sensor_paths = []
sensors = ["Center", "Left", "Right"]


def create_folder_structure():
    for sensor in sensors:
        new_sensor_paths.append(f"{cwd}\\{NEW_FOLDER_NAME}\\{sensor}")
        path = new_sensor_paths[-1]
        if not os.path.exists(path):
            os.mkdir(path)


def create_dataset(subs_list, indexing=True):
    create_folder_structure()
    repo = (Subject(sub) for sub in subs_list)
    for sub in repo:
        for i in range(3):
            os.chdir(new_sensor_paths[i])
            features_list, features = feature_extractor(sub, sensors[i].lower(), "acc")
            features.to_csv(sub.subject_id[:-4] + ".csv", sep="\t", index=indexing)
            # print(f"File generated - '{sub.subject_id[:-4]}.csv'")

