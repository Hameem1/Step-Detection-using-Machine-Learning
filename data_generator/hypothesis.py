import numpy as np
import statistics as stat
from data_generator.dataset_generator import NEW_DATASET, datasets_dir, ageGroups, get_limits
from dataset.dataset_manipulator import read_csv

ageGroups = ['(1-15)', '(16-50)']

FEATURES_USED = ['mean']
LIMITS = get_limits(ageGroups)


def get_samples(n):
    """
    Gets n random samples (filenames) from each group in ageGroups

    :param n: int(sample size)
    :return samples: dict('ageGroup1' : [n_samples], 'ageGroup2' : [n_samples], ...)
    """

    samples = {}
    data = read_csv(f'subject_data')
    for target_folder, limit in LIMITS.items():
        age_bin = data[(data['Age'] >= limit[0]) & (data['Age'] <= limit[1])]['Filename'].reset_index(drop=True)
        sample = list(age_bin.sample(10))
        samples[target_folder] = sample
    return samples


def get_feature_stats(samples, sensor_pos='right'):
    """
    Calculates the mean (Ax, Ay, Az) of every feature in FEATURES_USED, for every file in each ageGroup.

    :param sensor_pos: str('center', 'left', 'right')
    :param samples: dict(returned from 'get_samples()')
    :return: dict('ageGroup1': {'stat1':{'feature1': [], 'feature2': []}}, 'ageGroup2': {'stat2':{'feature1': [], 'feature2': []}}, ...)
    """

    path = f'{datasets_dir}\\{NEW_DATASET}\\{sensor_pos.capitalize()}'
    features_mean = {}

    for age_group, files in samples.items():
        features = {'avg_' + feature: [] for feature in FEATURES_USED}
        for file in files:
            df = read_csv(path + f'\\{file[:-4]}')
            for feature in FEATURES_USED:
                features['avg_' + feature].append([stat.mean(df['Ax_' + feature]),
                                                 stat.mean(df['Ay_' + feature]),
                                                 stat.mean(df['Az_' + feature])])

        for feature in FEATURES_USED:
            l = len(features['avg_' + feature])
            summed = np.array([sum(x) for x in zip(*features['avg_' + feature])])
            features['avg_' + feature] = list(summed/l)
        features_mean[age_group] = features

    return features_mean


def gen_box_plot():
    pass


if __name__ == '__main__':
    n=10
    samples = get_samples(n)
    f_mean = get_feature_stats(samples, sensor_pos='right')

    for age_group, features in f_mean.items():
        print(f'\nFor Group : {age_group} - with a sample size of {n}')
        for feature, values in features.items():
            print(f'{feature} in (Ax, Ay, Az) = {values}\n')




# print(f'\n{NEW_DATASET}\n')
# print(datasets_dir, '\n')
# print(ageGroups, '\n')
# print(get_limits(ageGroups), '\n')
# read_csv('subject_data')
