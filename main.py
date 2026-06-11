# file: main.py

import warnings
warnings.filterwarnings('ignore')

import numpy as np

from data_prep  import load_and_prep_data
from models     import (train_baseline, train_stacked_ensemble,
                        train_log_transformed_ensemble,
                        save_models, load_models, models_exist)
from evaluation import print_metrics, generate_all_plots


# ── Toggle this flag ─────────────────────────────────────────────────────────
#   False → load saved models if they exist, skip training
#   True  → always retrain from scratch and overwrite saved files
FORCE_RETRAIN = False
# ─────────────────────────────────────────────────────────────────────────────


def main():
    filepath = r'C:\Users\jashw\Desktop\Future Interns\FUTURE_ML_01\train.csv'

    print("=== 1. Data Preparation ===")
    X_train, X_test, y_train, y_test, test_data, features, base_year = load_and_prep_data(filepath)

    # ── Train or Load ─────────────────────────────────────────────────────────
    if not FORCE_RETRAIN and models_exist():
        print("=== Saved models found — loading from disk ===")
        baseline_model, stack_model, stack_log = load_models()

    else:
        print("=== 2. Training Baseline Model ===")
        baseline_model = train_baseline(X_train, y_train)

        print("=== 3. Training Main Stacked Ensemble ===")
        print("Fitting models (this may take 20-30 seconds)...")
        stack_model = train_stacked_ensemble(X_train, y_train)

        print("=== 4. Training Log-Transformed Ensemble ===")
        print("Fitting models with log1p target...")
        stack_log = train_log_transformed_ensemble(X_train, y_train)
        print("Done.\n")

        save_models(baseline_model, stack_model, stack_log, base_year, features)

    # ── Predictions (always runs, regardless of train/load path) ─────────────
    baseline_preds  = baseline_model.predict(X_test)
    stack_preds     = stack_model.predict(X_test)
    stack_preds_log = np.expm1(stack_log.predict(X_test))

    print("=== 5. Evaluation Metrics ===")
    print_metrics(y_test, baseline_preds,  "Baseline - Linear Regression")
    print_metrics(y_test, stack_preds,     "Stacked Ensemble (Original Scale)")
    print_metrics(y_test, stack_preds_log, "Stacked Ensemble (Log1p Transform)")

    print("=== 6. Generating Visualizations ===")
    generate_all_plots(
        test_data, y_test, baseline_preds, stack_preds,
        stack_preds_log, stack_model, features, base_year
    )

    print("\nProcess Complete! Check your project folder for the PNG images.")


if __name__ == "__main__":
    main()