"""This module calculates the features from the base data using a moving window and stores them"""

"""
Currently the following features are being calculated:

-> mean, variance, standard deviation
-> median, max, min, rms
-> index(min), index(max), IQR ,signal magnitude area, power, energy, entropy, skewness, kurtosis, mean abs deviation
-> xy, xz, yz
    
"""

# Imports
import statistics as stat
import numpy as np


class Features:
    """This class contains the calculated features for the given
        dataset and provides the functions to calculate them
    """

    def __init__(self, data):

        # TODO: data should be a dictionary with x_data, y_data & z_data
        # The data to be used
        self.data = data
        # Width of the moving window (in # of samples)
        self.window_size = 50

        # All the calculated features
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
        # self.signal_magnitude_area = self.window(self.signal_magnitude_area)
        # self.power = self.window(self.power)
        # self.energy = self.window(self.energy)
        # self.entropy = self.window(self.entropy)
        # self.skewness = self.window(self.skewness)
        # self.kurtosis = self.window(self.kurtosis)
        # self.mean_abs_deviation = self.window(self.mean_abs_deviation)
        # self.xy = self.window(self.xy)
        # self.xz = self.window(self.xz)
        # self.yz = self.window(self.yz)

        # list of features
        self.features = []
        self.get_features_list()
        # List of lengths for all features
        self.feature_length = []
        self.data_loss = 0.0

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

    # This window runs over every @staticmethod and calls every calculation
    def window(self, func):
        ret = []
        window_size = self.window_size
        w_start = 0
        w_stop = w_start + window_size
        while w_stop < len(self.data):
            window_data = self.data[w_start:w_stop]
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


# TODO: change base_data to sensor_type('acc' or 'gyr') and store as x_data, y_data & z_data in the class
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
