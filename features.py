"""This module calculates the features from the base data using a moving window and stores them"""

"""
Currently the following features have been implemented:

->
->
->
->
->
    
"""


class Features:
    """This class contains the calculated features for the given
        dataset and provides the functions to calculate them
    """

    def __init__(self, data):
        # Width of the moving window (in # of samples)
        self.window_size = 50
        # All the calculated features
        self.f1 = self.calculate_f1()
        self.f2 = self.calculate_f2()
        self.f3 = self.calculate_f3()
        self.f4 = self.calculate_f4()
        self.f5 = self.calculate_f5()
        # list of features
        self.features = []
        self.get_features_list()
        # List of lengths for all features
        self.feature_length = []
        self.truncation = 0.0

    def calculate_f1(self):
        pass

    def calculate_f2(self):
        pass

    def calculate_f3(self):
        pass

    def calculate_f4(self):
        pass

    def calculate_f5(self):
        pass

    def get_features_list(self):
        self.features = list(
            f for f in dir(self) if not f.startswith('__') and not callable(getattr(self, f)) and f is not "features")


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
