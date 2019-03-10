"""
This module calculates the features from the base data using a moving window and stores them.

"""

# Imports
import numpy as np
import pandas as pd
import statistics as stat
from scipy.fftpack import fft
from scipy.stats import kurtosis, skew, entropy, pearsonr
from config import WINDOW_SIZE, WINDOW_TYPE, USED_CLASS_LABEL, STEP_SIZE, SENSOR

"""
Currently the following features are being calculated:

-> mean, variance, standard deviation
-> median, max, min, rms
-> index(min), index(max), IQR , skewness, kurtosis, signal magnitude area, energy, entropy, mean abs deviation
-> xy, xz, yz (Correlation between axes)
    
Total = 19

"""


class Features:
    """
    This class contains the calculated features for the given data.

    The data should be Pandas Series or DataFrame objects, containing decimal numbers.

    Parameters
    ----------
    data : Series or DataFrame
        Data to calculate features for
    step_positions_actual : list
        List of x-coordinate step positions in the original data
    window_size : int, optional
        Window size for the window which runs over the data
    window_type : str, optional
        Window type to be used for the data calculations

    Attributes
    ----------
    data : Series or DataFrame
        Contains the data used in the calculations (time domain)
    data_freq : float
        Contains the data used in the calculations (frequency domain) - [Future implementation]
    window_size : int
    window_type : str
    step_positions_actual : list
        List of x-coordinate step positions in the original data
    mean, median, variance, standard_deviation, iqr : float
        Basic statistical features (time domain)
    value_max, value_min : float
        Min and max values
    index_max, index_min : int
        Indices of min and max values
    rms, signal_magnitude_area, energy, entropy, skewness, kurtosis, mean_abs_deviation : float
        Advanced statistical features (time domain)
    corr_xy, corr_xz, corr_yz : float
        Cross-correlation features (between the axes)
    features : list
        List of names of extracted features

    Methods
    -------
    get_features_list()
        Generates a list of available features from the class which have been calculated

    """

    def __init__(self, data, step_positions_actual, window_size=WINDOW_SIZE, window_type=WINDOW_TYPE):
        # -------
        # WARNING
        # -------
        # When adding a new class member (which is not a new feature), add its exception to get_features_list()

        # The data to be used (Time domain)
        self.data = data
        # The data converted to frequency domain
        if isinstance(data, pd.Series):
            self.data_freq = pd.Series(fft(self.data))
        else:
            self.data_freq = None

        # Width of the moving window (in # of samples)
        self.window_size = window_size
        # Type of moving window (sliding/hopping)
        self.window_type = window_type
        # Actual step positions
        self.step_positions_actual = step_positions_actual

        # All the calculated features
        if isinstance(data, pd.Series):
            self.mean = self.window(self.calc_mean)
            self.variance = self.window(self.calc_variance)
            self.standard_deviation = self.window(self.calc_std)
            self.median = self.window(self.calc_median)
            self.value_max = self.window(self.calc_value_max)
            self.value_min = self.window(self.calc_value_min)
            self.index_max = self.window(self.calc_index_max)
            self.index_min = self.window(self.calc_index_min)
            self.rms = self.window(self.calc_rms)
            self.iqr = self.window(self.calc_iqr)
            self.signal_magnitude_area = self.window(self.calc_sma)
            self.energy = self.window(self.calc_energy)
            self.entropy = self.window(self.calc_data_entropy)
            self.skewness = self.window(self.calc_skewness)
            self.kurtosis = self.window(self.calc_kurtosis)
            self.mean_abs_deviation = self.window(self.calc_mean_abs_deviation)

            # TODO: try implementing fft_avg_band_power as well
            # self.fft_energy = self.window(self.calc_fft_energy)
            # self.fft_magnitude = abs(self.data_freq)
            # self.fft_mean = self.window(self.calc_mean, domain='freq')
            # self.fft_value_max = self.window(self.calc_value_max, domain='freq')
            # self.fft_value_min = self.window(self.calc_value_min, domain='freq')

        else:
            # Cross correlations between variables
            self.corr_xy = self.window(self.xy, 'Ax', 'Ay')
            self.corr_xz = self.window(self.xz, 'Ax', 'Az')
            self.corr_yz = self.window(self.yz, 'Ay', 'Az')

        # list of features
        self.features = []
        # Populating features
        self.get_features_list()

    # Basic Calculations
    # Time domain features
    @staticmethod
    def calc_mean(data):
        return stat.mean(data)

    @staticmethod
    def calc_variance(data):
        return np.var(data)

    @staticmethod
    def calc_std(data):
        return np.var(data)

    @staticmethod
    def calc_median(data):
        return stat.median(data)

    @staticmethod
    def calc_value_max(data):
        return np.max(data)

    @staticmethod
    def calc_value_min(data):
        return np.min(data)

    @staticmethod
    def calc_index_max(data):
        data = list(data)
        return data.index(min(data))

    @staticmethod
    def calc_index_min(data):
        data = list(data)
        return data.index(max(data))

    @staticmethod
    def calc_rms(data):
        return np.sqrt(np.mean(np.array(data) ** 2))

    @staticmethod
    def calc_iqr(data):
        q75, q25 = np.percentile(data, [75, 25])
        return q75 - q25

    @staticmethod
    def calc_kurtosis(data):
        return kurtosis(data)

    @staticmethod
    def calc_skewness(data):
        return skew(data)

    @staticmethod
    def calc_mean_abs_deviation(data):
        return data.mad()

    @staticmethod
    def calc_data_entropy(data):
        value, counts = np.unique(data, return_counts=True)
        return entropy(counts)

    @staticmethod
    def calc_energy(data):
        squares = data ** 2
        return squares.sum()

    @staticmethod
    def calc_sma(data):
        absolute = list(map(abs, data))
        return sum(absolute)

    # Correlation Features
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

    # Frequency Domain Features
    # @staticmethod
    # def calc_fft_energy(data):
    #     return sum(abs(data) ** 2)

    # This window runs over every @staticmethod and performs the given function
    def window(self, func, *args, domain='time'):
        ret = []
        w_start = 0
        w_stop = w_start + self.window_size

        if isinstance(self.data, pd.Series):
            if domain == 'time':
                data = self.data
            else:
                data = abs(self.data_freq)
        else:
            if domain == 'time':
                data = self.data[[args[0], args[1]]]
            else:
                data = abs(self.data_freq[[args[0], args[1]]])

        if self.window_type == 'sliding':
            while w_stop < len(data):
                window_data = data[w_start:w_stop + 1]
                temp = func(window_data)
                # If the data is a float (Store up to 5 decimal places)
                if not float(temp).is_integer():
                    ret.append("{0:.5f}".format(temp))
                # if the data is an int
                else:
                    ret.append(temp)
                w_start += 1
                w_stop = w_start + self.window_size

        elif self.window_type == 'hopping':
            for i in self.step_positions_actual:
                span = [int(i - WINDOW_SIZE / 2), int(i + WINDOW_SIZE / 2)]
                window_data = data[span[0]:span[1] + 1]
                temp = func(window_data)
                # If the data is a float (Store up to 5 decimal places)
                if not float(temp).is_integer():
                    ret.append("{0:.5f}".format(temp))
                # if the data is an int
                else:
                    ret.append(temp)
        else:
            raise Exception(f'No such window function exists: {self.window_type}')

        return ret

    # Generates a list of available features from the class which have been calculated
    def get_features_list(self):
        self.features = list(
            f for f in dir(self) if not f.startswith('__')
            and not callable(getattr(self, f))
            and f is not "features"
            and f is not "data"
            and f is not "window_size"
            and f is not "window_type"
            and f is not "data_freq"
            and f is not "step_positions_actual")


def print_features(features):
    """
    This function prints the given features dictionary.

    Parameters
    ----------
    features : dict
        features = {Axes: {feature_names: [feature_values]}}

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


def update_step_positions(data):
    """
    This function returns lists of step positions according to the window type being used.

    Parameters
    ----------
    data : DataFrame(subject data)

    Returns
    -------
    step_positions_actual : list
        List of x-axis step coordinates in the original data, i.e., Subject().sensor_pos[SENSOR].label[USED_CLASS_LABEL]
    step_positions_updated : list
        List of x-axis step coordinates in the windowed data
    step_positions_updated_bool : list
        List containing a boolean mask implementing STEP_SIZE for x-axis step coordinates in the windowed data

    """

    step_positions_actual = []

    for i in range(1, len(data['StepLabel'])):
        if (data.loc[i, 'StepLabel']) > (data.loc[i - 1, 'StepLabel']):
            step_positions_actual.append(i)

    # For a "sliding" window
    if WINDOW_TYPE == 'sliding':
        # Shifting the step indices
        step_positions_updated = np.array(step_positions_actual) - int(WINDOW_SIZE / 2)
        # Eliminating step indices which don't have enough data around them for the window
        step_positions_updated = [x for x in step_positions_updated if (x >= 0) and
                                  (x < (len(data['StepLabel'][int(WINDOW_SIZE/2):-int(WINDOW_SIZE/2)]) - STEP_SIZE))]
        # Creating a boolean step list with 1s representing the step size (step duration)
        step_positions_updated_bool = [False] * len(data['StepLabel'][int(WINDOW_SIZE / 2):-int(WINDOW_SIZE / 2)])
        for x in range(len(step_positions_updated_bool)):
            if x in step_positions_updated:
                for i in range(int(STEP_SIZE / 2) + 1):
                    step_positions_updated_bool[x + i] = step_positions_updated_bool[x - i] = True

    # For a "hopping" window
    else:
        step_positions_updated = range(len([x for x in np.array(step_positions_actual)
                                            if
                                            (x - int(WINDOW_SIZE / 2) >= 0) and (x < (len(data) - WINDOW_SIZE / 2))]))

        step_positions_updated_bool = [1 for _ in range(len(step_positions_updated))]

    return step_positions_actual, step_positions_updated, step_positions_updated_bool


# This is the exposed endpoint for usage via import
def feature_extractor(sub, sensor_pos, sensor_type=SENSOR, output_type='dict'):
    """
    This function returns the extracted features data for for the given subject.

    Parameters
    ----------
    sub : Subject
    sensor_pos : {'center', 'left', 'right'}
    sensor_type : {'acc', 'gyr'}
    output_type : {'dict', 'df'}

    Returns
    -------
    features_list, features : dict
    step_positions_actual : int
        List of x-axis step coordinates in the original data
    step_positions_updated : int
        List of x-axis step coordinates in the windowed data
    step_positions_updated_bool : bool
        List containing a boolean mask implementing STEP_SIZE for x-axis step coordinates in the windowed data

    """
    features = {}
    features_list = {}
    data = sub.sensor_pos[sensor_pos].label[USED_CLASS_LABEL]

    step_positions_actual, step_positions_updated, step_positions_updated_bool = update_step_positions(data)

    if sensor_type == "acc":
        base_data = [col for col in data.columns if col.startswith('A')]
        base_data.append('all')
    else:
        base_data = [col for col in data.columns if col.startswith('G')]
        base_data.append('all')

    for axis in base_data:
        if axis is not 'all':
            f = Features(data[axis], step_positions_actual)
        else:
            f = Features(data[base_data[0:3]], step_positions_actual)
        features_list[axis] = f.features
        features[axis] = {x: getattr(f, x) for x in f.features}

    # For output_type = dictionary
    if output_type == 'dict':
        return features_list, features, step_positions_actual, step_positions_updated, step_positions_updated_bool

    # For output_type = data frame
    elif output_type == 'df':
        columns = {}
        for axis, feature in features.items():
            for feature_name, feature_value in feature.items():
                if axis == 'all':
                    columns[feature_name] = feature_value
                else:
                    columns[f'{axis}_{feature_name}'] = feature_value

        # Adding the StepLabel to the dataframe
        columns['StepLabel'] = [1 if x else 0 for x in step_positions_updated_bool]
        column_names = list(columns.keys())
        df = pd.DataFrame(columns)
        return column_names, df, step_positions_actual, step_positions_updated, step_positions_updated_bool

    else:
        print("Invalid value for parameter 'output_type'! Please run the program again.")


if __name__ == '__main__':
    print(f"In __main__ of features.py")
else:
    print(f"\nModule imported : {__name__}")
