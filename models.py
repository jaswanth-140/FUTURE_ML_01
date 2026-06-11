# file: models.py

import os
import joblib
import numpy as np
from sklearn.linear_model    import LinearRegression, RidgeCV
from sklearn.ensemble        import RandomForestRegressor, StackingRegressor
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, KFold
from xgboost                 import XGBRegressor


MODELS_DIR = 'saved_models'


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_models(baseline_model, stack_model, stack_log, base_year, features):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(baseline_model, os.path.join(MODELS_DIR, 'baseline_model.joblib'))
    joblib.dump(stack_model,    os.path.join(MODELS_DIR, 'stack_model.joblib'))
    joblib.dump(stack_log,      os.path.join(MODELS_DIR, 'stack_log_model.joblib'))
    joblib.dump(
        {'base_year': base_year, 'features': features},
        os.path.join(MODELS_DIR, 'metadata.joblib')
    )
    print(f"\nModels saved to '{MODELS_DIR}/':")
    print(f"  baseline_model.joblib")
    print(f"  stack_model.joblib")
    print(f"  stack_log_model.joblib")
    print(f"  metadata.joblib  (base_year={base_year}, features={features})\n")


def load_models():
    baseline_model = joblib.load(os.path.join(MODELS_DIR, 'baseline_model.joblib'))
    stack_model    = joblib.load(os.path.join(MODELS_DIR, 'stack_model.joblib'))
    stack_log      = joblib.load(os.path.join(MODELS_DIR, 'stack_log_model.joblib'))
    print("All models loaded from saved files — skipping training.\n")
    return baseline_model, stack_model, stack_log


def models_exist():
    required = [
        'baseline_model.joblib',
        'stack_model.joblib',
        'stack_log_model.joblib',
        'metadata.joblib'
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


# =============================================================================
# TRAINING
# =============================================================================

def train_baseline(X_train, y_train):
    baseline_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr',     LinearRegression())
    ])
    baseline_model.fit(X_train, y_train)
    return baseline_model


def make_estimators():
    lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    rf  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    return [('lr', lr_pipe), ('rf', rf), ('xgb', xgb)]


def train_stacked_ensemble(X_train, y_train):
    base_cv = KFold(n_splits=5, shuffle=False)
    meta_learner = RidgeCV(
        alphas=np.logspace(-3, 3, 20),
        cv=TimeSeriesSplit(3)
    )
    stack_model = StackingRegressor(
        estimators=make_estimators(),
        final_estimator=meta_learner,
        cv=base_cv,
        passthrough=False
    )
    stack_model.fit(X_train, y_train)

    meta = stack_model.final_estimator_
    print(f"RidgeCV selected α : {meta.alpha_:.6f}")
    print("Meta-Learner Blend Weights:")
    for name, coef in zip(['lr', 'rf', 'xgb'], meta.coef_):
        print(f"  {name:>4} : {coef:+.4f}")
    print(f"  intercept : {meta.intercept_:+.4f}\n")

    return stack_model


def train_log_transformed_ensemble(X_train, y_train):
    y_train_log = np.log1p(y_train)
    base_cv = KFold(n_splits=5, shuffle=False)
    stack_log = StackingRegressor(
        estimators=make_estimators(),
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 20), cv=TimeSeriesSplit(3)),
        cv=base_cv,
        passthrough=False
    )
    stack_log.fit(X_train, y_train_log)
    return stack_log