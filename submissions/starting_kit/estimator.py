
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


def fold_time_series(time_point, period, div_period):
    return (time_point -
            (time_point // (period / div_period)) * period / div_period)


def get_bin_means(X_df, num_bins, band):
    feature_array = np.empty((len(X_df), num_bins))

    for k, (_, x) in enumerate(X_df.iterrows()):
        period = x['period']
        div_period = x['div_period']
        real_period = period / div_period
        bins = [i * real_period / num_bins for i in range(num_bins + 1)]

        time_points = np.array(x['time_points_' + band])
        light_points = np.array(x['light_points_' + band])
        time_points_folded = \
            np.array([fold_time_series(time_point, period, div_period)
                      for time_point in time_points])
        time_points_folded_digitized = \
            np.digitize(time_points_folded, bins) - 1

        for i in range(num_bins):
            this_light_points = light_points[time_points_folded_digitized == i]
            if len(this_light_points) > 0:
                feature_array[k, i] = np.mean(this_light_points)
            else:
                feature_array[k, i] = np.nan  # missing

    return feature_array


transformer_r = FunctionTransformer(
    lambda X_df: get_bin_means(X_df, 5, 'r')
)

transformer_b = FunctionTransformer(
    lambda X_df: get_bin_means(X_df, 5, 'b')
)

cols = [
    'magnitude_b',
    'magnitude_r',
    'period',
    'asym_b',
    'asym_r',
    'log_p_not_variable',
    'sigma_flux_b',
    'sigma_flux_r',
    'quality',
    'div_period',
]

common = ['period', 'div_period']
transformer = make_column_transformer(
    (transformer_r, common + ['time_points_r', 'light_points_r']),
    (transformer_b, common + ['time_points_b', 'light_points_b']),
    ('passthrough', cols)
)

pipe = make_pipeline(
    transformer,
    SimpleImputer(strategy='most_frequent'),
    RandomForestClassifier(max_depth=5, n_estimators=10)
)


def get_estimator():
    return pipe
