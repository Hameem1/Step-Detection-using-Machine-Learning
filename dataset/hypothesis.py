from data_generator.dataset_generator import NEW_DATASET, datasets_dir, ageGroups, get_limits, read_csv


FEATURES_USED = ['mean']
LIMITS = get_limits(ageGroups)


def get_samples(n):
    """
    Gets n random samples (filenames) from each group in ageGroups

    :param n: int(sample size)
    :return: dict('ageGroup1' : [n_samples], 'ageGroup2' : [n_samples], ...)
    """
    pass


def get_features(sensor_pos, samples):
    """
    Calculates the mean (Ax, Ay, Az) of every feature in FEATURES_USED, for every file in each ageGroup.

    :param sensor_pos: str('center', 'left', 'right')
    :param samples: dict(returned from 'get_samples()')
    :return: dict('ageGroup1': {'feature1': [], 'feature2': []}, 'ageGroup2': {'feature1': [], 'feature2': []}, ...)
    """
    pass


# print('\n', NEW_DATASET, '\n')
# print(datasets_dir, '\n')
# print(ageGroups, '\n')
# print(get_limits(ageGroups), '\n')
# read_csv('subject_data')
