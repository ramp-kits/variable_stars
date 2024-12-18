import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import problem

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

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


X_df, y = problem.get_train_data()
X_transform = transformer.fit_transform(X_df)
y = y - 1 # so that the classes are [0 1 2 3]

class FullyCompatibleXGBClassifier(XGBClassifier, BaseEstimator, ClassifierMixin):
    def __sklearn_tags__(self):
        return {
            "estimator_type": "classifier",
            "requires_y": True,
            "X_types": ["2darray"],  # Adjust based on your data
        }

    def fit(self, X, y, *args, **kwargs):
        return super().fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        return super().predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        return super().predict_proba(X, *args, **kwargs)


def manual_cv_score(estimator, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return np.mean(scores)


def xgb_evaluate(learning_rate, max_depth, subsample, colsample_bytree):
    # Ensure integer values for max_depth
    max_depth = int(max_depth)
    
    # Initialize the model with the current parameters
    model = FullyCompatibleXGBClassifier(
        use_label_encoder=False,  # Disable unnecessary warnings
        eval_metric="logloss",  # Evaluation metric
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=100,  # Fixed for now, you can also tune this
        random_state=42
    )
    
    # Cross-validation to evaluate the model
    scores = manual_cv_score(model, X_transform, y, cv=5)
    return scores.mean()  # Return the mean accuracy

# Define the parameter bounds for Bayesian Optimization
param_bounds = {
    "learning_rate": (0.01, 0.3),         # Lower learning rate often performs better
    "max_depth": (3, 10),                 # Typical range for tree depth
    "subsample": (0.5, 1.0),              # Fraction of data used for training each tree
    "colsample_bytree": (0.5, 1.0),       # Fraction of features used for training each tree
}

# Initialize Bayesian Optimizer
optimizer = BayesianOptimization(
    f=xgb_evaluate,         # Objective function to maximize
    pbounds=param_bounds,   # Parameter bounds
    random_state=42,
    verbose=2               # Show optimization progress
)

# Run optimization
optimizer.maximize(init_points=5, n_iter=25)

# Get the best parameters
print("Best Parameters:", optimizer.max["params"])
