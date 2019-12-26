
import numpy as np


def fold_time_series(time_point, period, div_period):
    return (time_point -
            (time_point // (period / div_period)) * period / div_period)


def get_bin_means(x, num_bins):
    period = x['period']
    div_period = x['div_period']
    real_period = period / div_period
    bins = [i * real_period / num_bins for i in range(num_bins + 1)]
    feature_array = np.empty(2 * num_bins)
    for band in ['b', 'r']:
        time_points = np.array(x['time_points_' + band])
        light_points = np.array(x['light_points_' + band])
        time_points_folded = \
            np.array([fold_time_series(time_point, period, div_period)
                      for time_point in time_points])
        time_points_folded_digitized = \
            np.digitize(time_points_folded, bins) - 1
        binned_means = \
            np.array([light_points[time_points_folded_digitized == i].mean()
                      for i in range(num_bins)])
        if band == 'b':
            feature_array[:num_bins] = binned_means
        else:
            feature_array[num_bins:] = binned_means
    return feature_array


class FeatureExtractor():

    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        cols = [
            'magnitude_b',
            'magnitude_r'
        ]
        X_array = \
            np.array([[x[col] for col in cols] for _, x in X_df.iterrows()])
        real_period = np.array([x['period'] / x['div_period']
                                for _, x in X_df.iterrows()])
        X_array = np.concatenate((X_array.T, [real_period])).T
        num_bins = 5
        X_array_variable_features = \
            np.array([get_bin_means(x, num_bins) for _, x in X_df.iterrows()])
        X_array = np.concatenate((X_array.T, X_array_variable_features.T)).T
        return X_array
