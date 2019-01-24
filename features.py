"""This module calculates the features from the base data using a moving window and stores them"""

"""
Currently the following features have been calculated:

-> mean, variance, standard deviation
-> median, max, min, rms
-> signal magnitude area, index(min), index(max), power, energy, entropy, skewness, kurtosis, IQR , mean abs deviation
-> xy, xz, yz
    
"""

import statistics as stat
import numpy as np

# TODO: Create a decorator to implement a window over the existing functions


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
        self.mean = self.mean()
        self.variance = self.variance()
        self.standard_deviation = self.standard_deviation()
        self.median = self.median()
        self.value_max = self.max()
        self.value_min = self.min()
        self.index_max = self.index_max()
        self.index_min = self.index_min()
        self.rms = self.rms()
        # self.signal_magnitude_area = self.signal_magnitude_area
        # self.power = self.power()
        # self.energy = self.energy()
        # self.entropy = self.entropy()
        # self.skewness = self.skewness()
        # self.kurtosis = self.kurtosis()
        # self.iqr = self.iqr()
        # self.mean_abs_deviation = self.mean_abs_deviation()
        # self.xy = self.xy()
        # self.xz = self.xz()
        # self.yz = self.yz()

        # list of features
        self.features = []
        self.get_features_list()
        # List of lengths for all features
        self.feature_length = []
        self.data_loss = 0.0

    # Class member functions
    def mean(self):
        print(len(self.data))
        mean = stat.mean(self.data)
        return mean

    def variance(self):
        print(len(self.data))
        variance = np.var(self.data)
        return variance

    def standard_deviation(self):
        print(len(self.data))
        standard_deviation = np.var(self.data)
        return standard_deviation

    def median(self):
        print(len(self.data))
        median = stat.median(self.data)
        return median

    def max(self):
        print(len(self.data))
        value_max = np.max(self.data)
        return value_max

    def min(self):
        print(len(self.data))
        value_min = np.min(self.data)
        return value_min

    def index_max(self):
        print(len(self.data))
        index_max = self.data.idxmax()
        return index_max

    def index_min(self):
        print(len(self.data))
        index_min = self.data.idxmin()
        return index_min

    def rms(self):
        print(len(self.data))
        rms = np.sqrt(np.mean(np.array(self.data) ** 2))
        return rms

    def get_features_list(self):
        self.features = list(
            f for f in dir(self) if not f.startswith('__')
            and not callable(getattr(self, f))
            and f is not "features"
            and f is not "data"
            and f is not "window_size"
            and f is not "feature_length"
            and f is not "data_loss")


# This is the exposed endpoint for usage via import
def feature_extractor(sub, sensor_pos, base_data):
    """This function returns the features dictionary for the requested data

        :param sub: A Subject class object
        :param sensor_pos: str('center', 'left', 'right')
        :param base_data: str('Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz')
        :returns features_list, features: list(features_list), dict(features)
    """

    f = Features(sub.sensor_pos[sensor_pos].label['valid'][base_data])
    features_list = f.features
    features = {x: getattr(f, x) for x in features_list}
    return features_list, features


if __name__ == '__main__':
    print(f'Example Code to be added for {__name__}')
else:
    print(f"\nModule imported : {__name__}\n")
