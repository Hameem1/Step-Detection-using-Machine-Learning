"""
This module provides functions and variables to manipulate the entire data set.

"""

import os
import re
import pandas as pd
from model_config import PROCESSED_DATASET_PATH
from config import FORCE, FILE_STATUS_MESSAGES, ORIGINAL_DATASET, data_files_path,\
    sensor_paths, sensors, new_sensor_paths, DATASETS


def dataset_rename():
    """
    Renames files in all three sub-directories of the Data set.

    Warnings
    --------
    This function renames files in the given data set folder (does NOT make copies).
    Place the entire data set folder in the same directory as this .py file.
    Will only process files which are named as in the original data set.

    """
    rename_count = -1
    unchanged_total = 0
    unchanged_list = []

    for path in sensor_paths:
        unchanged = 0
        if os.path.exists(path):
            print(f"\n-------------------")
            print(f"Entering directory : {path}")
            print(f"-------------------\n")
        else:
            print(f"\nError: The directory \"{path}\" does not exist.\n")
            print("Please make sure the path/name for the data set directory is correct.")
            print("Note: The data set folder should be in the same directory as the program's .py files.")
            break

        for f in os.listdir(path):
            f_name, f_ext = os.path.splitext(f)
            try:
                pre_text, name, post_text = f_name.split("_")
            except ValueError:
                if unchanged == 0:
                    print(f"\nSome or all files have already been renamed in directory:\n"
                          f"\"{path}\"\n\n"
                          f"Note: The files must have native names for renaming.")
                if FILE_STATUS_MESSAGES:
                    print(f"File \"{f}\" unchanged")
                unchanged += 1
            except:
                print(f"Unknown Error: Operation cannot be performed.\n"
                      f"Please check if the given directory name/path is valid.")
            else:
                if rename_count == -1:
                    if FORCE:
                        res = 'y'
                    else:
                        res = str(
                            input("Are you sure you want to rename all the files in the data set? (y/n)\n")).lower()
                    if res == 'y':
                        rename_count += 1
                if rename_count != -1:
                    name = name.strip()
                    os.rename(f'{path}/{f}', f"{path}/{name}{f_ext}")
                    if FILE_STATUS_MESSAGES:
                        print(f"File \"{f}\" renamed to :\"{name}{f_ext}\"")
                    rename_count += 1
                elif res == "n":
                    print("Operation cancelled.")
                    exit()
                else:
                    print("Invalid input! Please run the program again.")
                    exit()
        unchanged_total += unchanged
        unchanged_list.append(f"\n-------------------\n{unchanged} files unchanged in directory:\n\"{path}\"")

    print(f"\n-------------------")
    if rename_count == -1:
        print(f"-------------------")
        print("\nAll sub-folders contain files which are already renamed or are not in their native name format\n")
        print(f"This data set has already been renamed!")
        print(f"No changes required!")
    elif rename_count == 0:
        print(f"\nNo files were renamed!\n")
    else:
        print(f"\n{rename_count} files renamed!")
    print(f"\n-------------------")
    print(f"\n{unchanged_total} files unchanged!")
    for i in unchanged_list:
        print(i)


def dataset_analysis():
    """
    Analyzes and Normalizes the Data set.

    Not required if using get_subjects_list() or generate_subjects_data().

    Warnings
    --------
    Should only be executed once the files have been renamed (dataset_rename()).
    This function analyses and deletes files from the three
    subdirectories which are not common to all of them.

    Place the entire data set folder in the same directory as this .py file.

    """

    del_count = 0
    to_del_list = []

    loop = 0
    while loop <= 1:
        f_list = []
        try:
            for path in sensor_paths:
                f_list.append(os.listdir(path))
        except:
            print(f"\nError: The directory \"{DATASETS}/{ORIGINAL_DATASET}\" does not exist.\n")
            print("Please make sure the path for the data set directory is correct.")
            print("Note: The data set folder should be in the same directory as the program's .py files.")
            break

        if loop == 1:
            print(f"-------------------")
            print(f"\nRe-running analysis of subdirectories in \"{ORIGINAL_DATASET}\"")
        print(f"\nAnalyzing subdirectories in \"{ORIGINAL_DATASET}\"\n")
        print(f"Analysis complete!")
        print(f"\n# of files in \"Center\" = {len(f_list[0])}")
        print(f"# of files in \"Left\" = {len(f_list[1])}")
        print(f"# of files in \"Right\" = {len(f_list[2])}")

        universe = (set(f_list[0]).union(set(f_list[1]))).union(set(f_list[2]))
        print(f"\nTotal # of Unique files = {len(universe)}\n")
        common_files = set(f_list[0]) & set(f_list[1]) & set(f_list[2])
        print(f"# of files common to all three Folders = {len(common_files)}")
        uncommon_files = universe - common_files
        print(f"# of files Not common to all three Folders = {len(uncommon_files)}\n")

        print(f"List of files not common to all three folders:\n{sorted(list(uncommon_files))}")

        # if there are files to be deleted
        if len(common_files) != len(universe):
            print("\nWARNING : There are files to be deleted!\n"
                  "This is to make the data set uniform.")
            # prompt to delete extra files from each folder
            if FORCE:
                res = 'y'
            else:
                res = str(input("\nWould you like to fix this problem by keeping only"
                                " the common files in each folder? (y/n)\n")).lower()
            if res == "n":
                print("\nProcess terminated. 0 files deleted")
                break

            elif res == "y":
                # Making a list of files to delete from every Sensor folder
                print("Files to be deleted:\n")
                for i in range(len(sensors)):
                    to_del_list.append(list(uncommon_files.intersection(f_list[i])))
                    print(f"from {sensors[i]} : {to_del_list[i]}")

                print("\ndeleting extra files...\n")
                for i in range(len(to_del_list)):
                    for j in to_del_list[i]:
                        try:
                            os.unlink(f'{sensor_paths[i]}/{j}')
                        except:
                            if FILE_STATUS_MESSAGES:
                                print(f"file \"{j}\" does not exist in folder \"{sensors[i]}\"")
                        else:
                            if FILE_STATUS_MESSAGES:
                                print(f"file \"{j}\" deleted successfully!")
                            del_count += 1

                print("\nFiles deleted!!\n")

            else:
                print("Invalid input! Please run the program again.")
                loop += 1

        else:
            if loop == 0:
                print(f"\nAll subdirectories of {ORIGINAL_DATASET} contain:\n"
                      f"- an equal number of files\n"
                      f"- files with the same names as the other subdirectories\n"
                      f"\nNo changes required!\n")
                loop += 1
            else:
                print(f"\nDataset fixed!\n"
                      f"{del_count} files deleted.\n")
                del_count = 0

        loop += 1


def get_subjects_list():
    """
    Returns the subjects list for the Data set.

    Includes functionality of dataset_analysis()

    Warnings
    --------
    Use generate_subjects_data() instead!
    Should be executed once the data set is normalized (dataset_analysis()).
    Will also work directly on an un-normalized data set by providing the option to normalize.
    Place the entire data set folder in the same directory as this .py file.

    Returns
    -------
    SUBJECTS_LIST : list of str or None
        List of subject filenames (subject_ids)

    """

    temp_list = []

    def is_normalized():
        temp_list.clear()
        for path in sensor_paths:
            if not os.path.exists(path):
                print(f"\nError: The directory \"{path}\" does not exist.\n")
                print("Please make sure the path/name for the data set directory is correct.")
                print("Note: The data set folder should be in the same directory as the program's .py files.")
                break
            else:
                temp_list.append(os.listdir(path))
        if temp_list[0] == temp_list[1] == temp_list[2]:
            return 1
        else:
            return 0

    # if the data set is normalized
    if is_normalized():
        SUBJECTS_LIST = list(temp_list[0])
        print("\nSubjects list generated successfully")
        return SUBJECTS_LIST
    # if the data set is NOT normalized
    else:
        print("\nWARNING : The data set needs to be normalized first.\n")
        if FORCE:
            res = 'y'
        else:
            res = str(input("Would you like to proceed with normalizing the data set? (y/n)\n")).lower()
        if res == "y":
            # normalizing the data set
            dataset_analysis()
            if is_normalized():
                print("\nThe data set has been normalized.")
                SUBJECTS_LIST = list(temp_list[0])
                print("\nSubjects list generated successfully")
                return SUBJECTS_LIST
            else:
                print("\nSome error occurred")
                return None
        elif res == "n":

            print("\nProgram terminated by user.\n")
            return None
        else:
            print("\nInvalid input! Please run the program again.")
            return None


def generate_subjects_data(gen_csv=None, indexing=True):
    """
    Generates a subjects list and subjects data along with an optional
    .csv file in the current working directory. Also verifies with the IDGenderAgelist.csv file.

    This function should be used instead of get_subjects_list() when running the first time!

    Parameters
    ----------
    gen_csv : bool, optional
        Generates a .csv file from the data as well (default = 'None')
    indexing : bool, optional
        Implements an indexing column for the .csv file (default = 'True')

    Returns
    -------
    subject_list : list
        List of subject filenames (subject_ids)
    subject_data : DataFrame
        Information about Subjects

    """

    SUBJECTS_LIST = get_subjects_list()

    total = len(SUBJECTS_LIST)
    found = 0
    sub_id = []
    gender = []
    age = []
    filename = []
    files_not_found = []

    with open(f"{data_files_path}/IDGenderAgelist.csv", 'r') as myfile:
        lines = myfile.readlines()

    for subject in SUBJECTS_LIST:
        pattern = re.compile((re.escape(subject[2:8])) + r',([01]),(\d{1,2})')
        line_num = -1
        is_found = False
        for line in lines:
            line_num += 1
            match = pattern.search(line)
            if not match:
                continue
            else:
                is_found = True
                break

        if is_found:
            found += 1
            filename.append(subject)
            sub_id.append(match.group(0)[0:6])
            gender.append("Male" if match.group(1) == '1' else "Female")
            age.append(match.group(2))
            # print(f"Match found in line {line_num}")
        else:
            files_not_found.append(subject)

    df = pd.DataFrame(dict(Filename=filename, Id=sub_id, Gender=gender, Age=age))

    print(f"\nProcess Completed.\n\nTotal searches = {total}\nTotal matches found = {found}\n")
    not_found = total - found
    if not_found > 0:
        print(f"Data for the following {not_found} files was not found in IDGenderAgelist.csv:\n")
        for file_na in files_not_found:
            print(file_na)
        ret = str(input(f"Would you like to remove these files from the dataset_operations? (y/n)\n")).lower()

        if ret == 'y':
            print("\ndeleting extra files...\n")
            del_count = 0
            for file_na in files_not_found:
                for sensor in sensor_paths:
                    print(f"\n-------------------")
                    print(f"Entering directory : {sensor}")
                    print(f"-------------------\n")
                    try:
                        os.unlink(f'{sensor}/{file_na}')
                    except:
                        print(f"file \"{file_na}\" does not exist in folder \"{sensor}\"")
                    else:
                        print(f"file \"{file_na}\" deleted successfully from folder \"{sensor}\"!")
                        del_count += 1

            print(f'\nTotal files deleted from all folders = {del_count}\n')
            print(f'Removing the deleted entries from the SUBJECTS_LIST')
            try:
                for file_na in files_not_found:
                    i = SUBJECTS_LIST.index(file_na)
                    SUBJECTS_LIST.pop(i)
            except:
                print("An error occurred while deleting an entry from the SUBJECTS_LIST")

        elif ret == 'n':
            print("Files NOT deleted from the dataset_operations.\n")
        else:
            print("Invalid input! Please run the program again.")

    if gen_csv is not False:
        if gen_csv is None:
            if FORCE:
                res = 'y'
            else:
                res = str(input("Would you like to generate a csv file for all the Subject data? (y/n)")).lower()

            if res == 'n':
                name = ''
                print("Process terminated by user!\n")
            elif res == 'y':
                if FORCE:
                    name = 'subject_data'
                else:
                    name = str(input("Please enter a name for the new .csv file: ")).lower()
            else:
                print("Invalid input! Please run the program again.")

        else:
            if FORCE:
                name = 'subject_data'
            else:
                name = str(input("Please enter a name for the new .csv file. (y/n)")).lower()

        if name:
            try:
                df.to_csv(f'{data_files_path}/{name}.csv', sep="\t", index=indexing)
                print(f"File generated - '{name}.csv'")
            except:
                print("An unexpected error occurred while creating the .csv file!")
        else:
            print("INFO: .csv file not generated!.")
    else:
        pass

    return SUBJECTS_LIST, df


def read_csv(file_path):
    """
    Reads in a .csv file from the specified file_path and returns it as a DataFrame.

    Parameters
    ----------
    file_path : str
        path + file name (without file extension)

    Returns
    -------
    DataFrame : DataFrame

    """

    try:
        data = pd.read_csv(file_path + ".csv", sep='\t', index_col=0)
    except FileNotFoundError:
        print(f"\nError : File not found.\nThis file does not exist at location:\n{file_path}.csv")
    else:
        # print(f"\nPreview of the .csv file contents:\n\n{data.head()}")
        return data


def dataframe_concatenator(single_file=False):
    """
    Generates the 'Processed_Dataset' within the project directory which will be used for the classifier.
    Three separate files for each sensor will be generated, with each file containing the data for every subject.

    Parameters
    ----------
    single_file : bool, optional
        Merges the three files into a single data file if True (default = False)

    """

    datasets = []

    if not os.path.exists(PROCESSED_DATASET_PATH):
        print(f'\nWARNING: The path does not exist. Creating new directory...\n{PROCESSED_DATASET_PATH}\n')
        os.mkdir(f"{PROCESSED_DATASET_PATH}")

    for path in new_sensor_paths:
        dfs = []
        print(f'Folder => {path}\n')
        for file_name in os.listdir(path):
            df = pd.read_csv(f'{path}/{file_name}', sep='\t', index_col=0)
            dfs.append(df)
        datasets.append(pd.concat(dfs, ignore_index=True))

    print()
    print(f'Length of ds_center = {len(datasets[0])} rows')
    print(f'Length of ds_left = {len(datasets[1])} rows')
    print(f'Length of ds_right = {len(datasets[2])} rows')
    print()
    print(f'Generating 3 separate .csv files...\n') if not single_file else print(f'Generating 1 main .csv file...\n')
    if not single_file:
        datasets[0].to_csv(f'{PROCESSED_DATASET_PATH}/ds_center.csv', sep="\t", index=True, float_format='%g')
        print(f'File generated : "ds_center.csv"')
        datasets[1].to_csv(f'{PROCESSED_DATASET_PATH}/ds_left.csv', sep="\t", index=True, float_format='%g')
        print(f'File generated : "ds_left.csv"')
        datasets[2].to_csv(f'{PROCESSED_DATASET_PATH}/ds_right.csv', sep="\t", index=True, float_format='%g')
        print(f'File generated : "ds_right.csv"')
    else:
        data_file = pd.concat(datasets, ignore_index=True)
        data_file.to_csv(f'{PROCESSED_DATASET_PATH}/ds_all.csv', sep="\t", index=True, float_format='%g')
        print(f'File generated : "ds_all.csv"')
    print(f'\nProcess complete!\n')


def end_tab_remover(subs_list):
    """
    Removes a tab space from the end of each line of each file in the given subjects list.

    Parameters
    ----------
    subs_list : list of str
        Subjects list


    Warnings
    --------
    This function has been deprecated (Avoid this if using "index_col=False" in data_structs.py)

    """

    ret = str(input("\nWould you like to proceed with removing trailing tabs from the whole data set? (y/n)\n")).lower()
    if ret == "y":
        try:
            print("\nPlease wait. This may take a few seconds...\n")
            for path in sensor_paths:
                print(f"\nProcessing files in the directory:\n{path}")
                for sub in subs_list:
                    with open(f"{path}/{sub}", 'r') as fin:
                        lines = [line.rstrip() + "\n" for line in fin]
                    with open(f"{path}/{sub}", 'w') as fout:
                        fout.writelines(lines)
        except:
            print("An error occurred in function 'end_tab_remover()'")
            print("Please check that the given file name/path is correct.")
        else:
            print("\nAll files processed successfully!\n")

    elif ret == "n":
        print("\nProgram terminated unsuccessfully.\n")
    else:
        print("\nInvalid input! Please run the program again.")


if __name__ == '__main__':
    # This example code demonstrates the use of this module's functions

    # Renaming the data set files to fix the naming scheme
    dataset_rename()

    # Verifying that every subject has a data set in each sub-directory (Center, Right, Left)
    # Makes the data set uniform (optional if using get_subjects_list())
    dataset_analysis()

    # Generating a subject data (subject_list + subject_data) and a .csv file (optional) from the subject list
    subject_list, subject_data = generate_subjects_data()
    print(f"-------------------")
    print(f"Subjects List : {subject_list}\n"
          f"# of Subjects : {len(subject_list) if type(subject_list) != type(None) else 0}")
    print(f"-------------------")

    # Reading the .csv file
    sub_data_csv = read_csv(f'{data_files_path}/subject_data')

else:
    print(f"\nModule imported : {__name__}")
