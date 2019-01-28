"""This module calculates the features from the base data using a moving window and stores them"""

"""
Currently the following features are being calculated:

-> mean, variance, standard deviation
-> median, max, min, rms
-> index(min), index(max), IQR , skewness, kurtosis, signal magnitude area, energy, entropy, mean abs deviation
-> xy, xz, yz (Correlation between axes)
    
"""

# Imports
import statistics as stat
import numpy as np
from scipy.stats import kurtosis, skew, entropy, pearsonr
import pandas as pd


class Features:
    """This class contains the calculated features for the given
        dataset and provides the functions to calculate them
    """

    def __init__(self, data):

        # The data to be used
        self.data = data
        # Width of the moving window (in # of samples)
        self.window_size = 50

        # All the calculated features
        if isinstance(data, pd.Series):
            print(f'Calculating features for {data.name}')
            self.mean = self.window(self.mean)
            self.variance = self.window(self.variance)
            self.standard_deviation = self.window(self.standard_deviation)
            self.median = self.window(self.median)
            self.value_max = self.window(self.value_max)
            self.value_min = self.window(self.value_min)
            self.index_max = self.window(self.index_max)
            self.index_min = self.window(self.index_min)
            self.rms = self.window(self.rms)
            self.iqr = self.window(self.iqr)
            self.signal_magnitude_area = self.window(self.sma)
            self.energy = self.window(self.energy)
            self.entropy = self.window(self.data_entropy)
            self.skewness = self.window(self.skewness)
            self.kurtosis = self.window(self.kurtosis)
            self.mean_abs_deviation = self.window(self.mean_abs_deviation)

        else:
            # Cross correlations between variables
            print(f'Calculating correlation between {data.columns}')
            self.corr_xy = self.window(self.xy, 'Ax', 'Ay')
            self.corr_xz = self.window(self.xz, 'Ax', 'Az')
            self.corr_yz = self.window(self.yz, 'Ay', 'Az')

        print(f'Dimensions of self.data = {data.size}')
        # list of features
        self.features = []
        # List of lengths for all features
        # self.feature_length = []
        # Data loss due to windowing
        # self.data_loss = 0.0
        self.get_features_list()

    # Basic Calculations
    @staticmethod
    def mean(data):
        return stat.mean(data)

    @staticmethod
    def variance(data):
        return np.var(data)

    @staticmethod
    def standard_deviation(data):
        return np.var(data)

    @staticmethod
    def median(data):
        return stat.median(data)

    @staticmethod
    def value_max(data):
        return np.max(data)

    @staticmethod
    def value_min(data):
        return np.min(data)

    @staticmethod
    def index_max(data):
        data = list(data)
        return data.index(min(data))

    @staticmethod
    def index_min(data):
        data = list(data)
        return data.index(max(data))

    @staticmethod
    def rms(data):
        return np.sqrt(np.mean(np.array(data) ** 2))

    @staticmethod
    def iqr(data):
        q75, q25 = np.percentile(data, [75, 25])
        return q75 - q25

    @staticmethod
    def kurtosis(data):
        return kurtosis(data)

    @staticmethod
    def skewness(data):
        return skew(data)

    @staticmethod
    def mean_abs_deviation(data):
        return data.mad()

    @staticmethod
    def data_entropy(data):
        value, counts = np.unique(data, return_counts=True)
        return entropy(counts)

    @staticmethod
    def energy(data):
        squares = data**2
        return squares.sum()

    @staticmethod
    def sma(data):
        absolute = list(map(abs, data))
        return sum(absolute)

    @staticmethod
    def xy(data):
        x = data['Ax']
        y = data['Ay']
        return pearsonr(x, y)[0]

    @staticmethod
    def xz(data):
        x = data['Ax']
        z = data['Az']
        return pearsonr(x, z)[0]

    @staticmethod
    def yz(data):
        y = data['Ay']
        z = data['Az']
        return pearsonr(y, z)[0]

    # This window runs over every @staticmethod and performs the given function
    def window(self, func, *args):
        ret = []
        window_size = self.window_size
        w_start = 0
        w_stop = w_start + window_size
        if isinstance(self.data, pd.Series):
            data = self.data
        else:
            data = self.data[[args[0], args[1]]]
        while w_stop < len(data):
            window_data = data[w_start:w_stop]
            # print(window_data)
            temp = func(window_data)
            # If the data is a float (Store up to 5 decimal places)
            if not float(temp).is_integer():
                ret.append("{0:.5f}".format(temp))
            # if the data is an int
            else:
                ret.append(temp)
            w_start += 1
            w_stop = w_start + window_size
        return ret

    # Generates a list of available features from what has been calculated in the class
    def get_features_list(self):
        self.features = list(
            f for f in dir(self) if not f.startswith('__')
            and not callable(getattr(self, f))
            and f is not "features"
            and f is not "data"
            and f is not "window_size"
            and f is not "feature_length"
            and f is not "data_loss")


def print_features(features):
    """This function prints the given features dictionary

        :param features: dict(features)
    """
    total_features = 0
    for axis, features_data in features.items():
        print(f'\n--------------------'
              f'  Calculated Features  '
              f'--------------------\n')
        print(f'Base Data = {axis}\n'
              f'Sensor Position = Right\n'
              f'# of calculated features = {len(features_data)}\n')
        print(f'--------------------\n')

        if total_features != len(features_data):
            total_features += len(features_data)

        for feature, value in features_data.items():
            print(f"\n{feature} = {value}")
            # print(f"Length of {feature} = {len(value)}")

    print(f'\nTotal # of unique features calculated = {total_features}')


# TODO: Return features as a new DataFrame

# This is the exposed endpoint for usage via import
def feature_extractor(sub, sensor_pos, sensor_type):
    """This function returns the features dictionary for the requested data

        :param sub: A Subject class object
        :param sensor_pos: str('center', 'left', 'right')
        :param sensor_type: str('acc', 'gyr')
        :returns features_list, features: list(features_list), dict(features)
    """
    features = {}
    features_list = {}
    data = sub.sensor_pos[sensor_pos].label['valid']
    if sensor_type == "acc":
        base_data = [col for col in data.columns if col.startswith('A')]
        base_data.append('all')
    else:
        base_data = [col for col in data.columns if col.startswith('G')]
        base_data.append('all')

    for axis in base_data:
        if axis is not 'all':
            f = Features(data[axis])
        else:
            f = Features(data[base_data[0:3]])
        features_list[axis] = f.features
        features[axis] = {x: getattr(f, x) for x in f.features}

    return features_list, features


if __name__ == '__main__':
    print(f"In __main__ of features.py")
else:
    print(f"\nModule imported : {__name__}\n")
