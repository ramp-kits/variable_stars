import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Classification of variable stars from light curves'
_target_column_name = 'type'
_ignore_column_names = []
_prediction_label_names = [1.0, 2.0, 3.0, 4.0]

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.Accuracy(name='acc', precision=4),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


# READ DATA
def csv_array_to_float(csv_array_string):
    return list(map(float, csv_array_string[1:-1].split(',')))


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


def _read_data(path, df_filename, vf_filename):
    df = pd.read_csv(os.path.join(path, 'data', df_filename), index_col=0)
    y_array = df[_target_column_name].values.astype(int)
    X_dict = df.drop(_target_column_name, axis=1).to_dict(orient='records')
    vf_raw = pd.read_csv(os.path.join(path, 'data', vf_filename),
                         index_col=0, compression='gzip')
    vf_dict = vf_raw.applymap(csv_array_to_float).to_dict(orient='records')
    X_dict = [merge_two_dicts(d_inst, v_inst) for d_inst, v_inst
              in zip(X_dict, vf_dict)]
    return pd.DataFrame(X_dict), y_array


def get_train_data(path='.'):
    df_filename = 'train.csv'
    vf_filename = 'train_varlength_features.csv.gz'
    return _read_data(path, df_filename, vf_filename)


def get_test_data(path='.'):
    df_filename = 'test.csv'
    vf_filename = 'test_varlength_features.csv.gz'
    return _read_data(path, df_filename, vf_filename)
