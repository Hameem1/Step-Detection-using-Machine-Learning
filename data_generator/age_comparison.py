import os
import numpy as np
import pandas as pd
from statistics import mean
import plotly.offline as pyo
import plotly.graph_objs as go
from data_generator.dataset_generator import NEW_DATASET, datasets_dir, ageGroups, get_limits

ageGroups = ['(1-13)', '(5-15)']

FEATURES_USED = ['mean']
LIMITS = get_limits(ageGroups)


def get_samples(n):
    """
    Gets n random samples (filenames) from each group in ageGroups

    :param n: int(sample size)
    :return samples: dict('ageGroup1' : [n_samples], 'ageGroup2' : [n_samples], ...)
    """

    samples = {}

    try:
        data = pd.read_csv('subject_data' + ".csv", sep='\t', index_col=0)
    except FileNotFoundError:
        print(f"\nError : File not found.\nThis file does not exist in the current working directory.\n{os.getcwd()}")

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
    :returns groups, raw_groups: dict('ageGroup1': {'stat1':{'feature1': [], 'feature2': []}}, 'ageGroup2': {...}, ...),
                                dict(raw_data)
    """

    path = f'{datasets_dir}\\{NEW_DATASET}\\{sensor_pos.capitalize()}'
    groups = {}
    raw_groups = {}

    for age_group, files in samples.items():
        stats = {'avg': {feature: [] for feature in FEATURES_USED},
                 'q1': {feature: [] for feature in FEATURES_USED},
                 'q3': {feature: [] for feature in FEATURES_USED},
                 'iqr': {feature: [] for feature in FEATURES_USED}}

        for file in files:
            try:
                df = pd.read_csv(path + f'\\{file[:-4]}' + ".csv", sep='\t', index_col=0)
            except FileNotFoundError:
                print(f"\nError : File not found.\nThis file does not exist in the current working directory."
                      f"\n{os.getcwd()}")

            for feature in FEATURES_USED:
                stats['avg'][feature].append([mean(df['Ax_' + feature]),
                                              mean(df['Ay_' + feature]),
                                              mean(df['Az_' + feature])])
                q75x, q25x = np.percentile(df['Ax_' + feature], [75, 25])
                q75y, q25y = np.percentile(df['Ay_' + feature], [75, 25])
                q75z, q25z = np.percentile(df['Az_' + feature], [75, 25])
                stats['q1'][feature].append([q25x, q25y, q25z])
                stats['q3'][feature].append([q75x, q75y, q75z])
                stats['iqr'][feature].append([q75x - q25x, q75y - q25y, q75z - q25z])

        raw_groups[age_group] = np.array([i for i in zip(*stats['avg']['mean'])])

        for stat in stats:
            for feature in FEATURES_USED:
                l = len(files)
                summed = np.array([sum(x) for x in zip(*stats[stat][feature])])
                stats[stat][feature] = list(summed / l)
        groups[age_group] = stats

    return groups, raw_groups


def print_statistics(group_stats):
    for age_group, stats in group_stats.items():
        print(f'\nFor Group : {age_group} - with a sample size of {n}\n')
        for stat, features in stats.items():
            for feature, values in features.items():
                print(f'{stat.capitalize()} of feature "{feature}" in (Ax, Ay, Az) = '
                      f'{values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}')
        print()


def sample_average(S):
    """
    Calculates the average of groups from different samples

    :param S: list(sample groups)
    :return G: dict(average of groups)
    """
    pass


def gen_box_plot(y):
    """
    Generates a box plot for the given data

    """
    names = list(y.keys())
    values = list(y.values())

    data = [go.Box(y=values[i], name=names[i], boxpoints=False) for i in range(len(y))]
    layout = go.Layout(title='Comparison of Step features between different Age groups',
                       font=dict(family='arial', size=16, color='#000000'))
    fig = go.Figure(data=data, layout=layout)
    pyo.plot(fig, filename='age_comparison.html')


if __name__ == '__main__':
    n = 50
    samples = get_samples(n)
    sensor_pos = 'right'
    groups, raw_groups = get_feature_stats(samples, sensor_pos=sensor_pos)
    print_statistics(groups)
    print(raw_groups)
    gen_box_plot({f'Age_{name}': raw_groups[f'Age_{name}'][0] for name in ageGroups})


else:
    n = 10
    samples = get_samples(n)
    sensor_pos = 'right'
    groups, raw_groups = get_feature_stats(samples, sensor_pos=sensor_pos)
    print_statistics(groups)
    print(raw_groups)
