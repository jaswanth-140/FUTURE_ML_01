import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape_score(actual, predicted, label=''):
    mask = actual != 0
    skipped = int((~mask).sum())
    if skipped > 0:
        print(f"  [MAPE - {label}] skipped {skipped} zero-sale record(s)")
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def print_metrics(actual, predicted, label):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mape_score(actual, predicted, label)
    print(f"{'-' * 44}")
    print(f"  {label}")
    print(f"{'-' * 44}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  MAPE : {mape:.3f} %")
    print(f"  RMSE : {rmse:.3f}\n")


def generate_all_plots(test_data, y_test, baseline_preds, stack_preds, stack_preds_log, stack_model, features,
                       base_year):
    dates_test = test_data['date'].values

    # Plot 1: Timeline
    plt.figure(figsize=(16, 5))
    plt.plot(dates_test, y_test, color='#1a1a1a', linewidth=1.3, label='Actual')
    plt.plot(dates_test, baseline_preds, color='steelblue', linewidth=1.0, linestyle='--', alpha=0.85,
             label='Baseline LR')
    plt.plot(dates_test, stack_preds, color='tomato', linewidth=1.0, linestyle='--', alpha=0.85,
             label='Stacked Ensemble')
    plt.title('Sales Forecast - 2017 Test Period', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('plot1_timeline.png', dpi=150)
    plt.close()

    # Plot 2: 3D Surface
    month_vals = np.arange(1, 13)
    dow_vals = np.arange(0, 7)
    month_grid, dow_grid = np.meshgrid(month_vals, dow_vals)
    is_wknd_flat = (dow_grid.ravel() >= 5).astype(int)
    year_idx_2017 = 2017 - base_year
    year_idx_flat = np.full(month_grid.ravel().shape[0], year_idx_2017)

    grid_X = np.column_stack([year_idx_flat, month_grid.ravel(), dow_grid.ravel(), is_wknd_flat])
    z_grid = stack_model.predict(grid_X).reshape(month_grid.shape)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(month_grid, dow_grid, z_grid, cmap='plasma', edgecolor='none', alpha=0.90)
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, label='Predicted Sales')
    ax.set_xlabel('Month', labelpad=10)
    ax.set_ylabel('Day of Week', labelpad=10)
    ax.set_zlabel('Predicted Sales', labelpad=10)
    ax.set_title('Prediction Curvature - Stacked Ensemble (Year 2017)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_yticks(range(7))
    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.tight_layout()
    plt.savefig('plot2_surface.png', dpi=150)
    plt.close()

    # Plot 3: Feature Importance Bar Chart
    rf_fitted = stack_model.named_estimators_['rf']
    importances = rf_fitted.feature_importances_
    sorted_idx = np.argsort(importances)
    sorted_feats = [features[i] for i in sorted_idx]
    sorted_imp = importances[sorted_idx]

    plt.figure(figsize=(8, 4))
    bars = plt.barh(sorted_feats, sorted_imp, color='steelblue', edgecolor='black', height=0.5)
    for bar, val in zip(bars, sorted_imp):
        plt.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=10)
    plt.xlabel('Importance Score')
    plt.title('Random Forest Feature Importances', fontsize=11, fontweight='bold')
    plt.xlim(0, max(sorted_imp) * 1.25)
    plt.tight_layout()
    plt.savefig('plot3_rf_importance.png', dpi=150)
    plt.close()

    # Plot 4: Residuals vs Time
    residuals_baseline = y_test - baseline_preds
    residuals_stack = y_test - stack_preds
    plt.figure(figsize=(16, 4))
    plt.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)
    plt.plot(dates_test, residuals_baseline, color='steelblue', linewidth=0.8, alpha=0.75,
             label='Baseline LR residuals')
    plt.plot(dates_test, residuals_stack, color='tomato', linewidth=0.8, alpha=0.75, label='Stacked Ensemble residuals')
    plt.title('Residuals Over Time - 2017 Test Period', fontsize=13, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Residual (Actual − Predicted)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('plot4_residuals.png', dpi=150)
    plt.close()

    print("Residual Statistics (Stacked):")
    print(f"  Mean  : {residuals_stack.mean():.3f}  (≈0 → unbiased)")
    print(f"  Std   : {residuals_stack.std():.3f}\n")

    # Plot 5: Log1p Target Transform Comparison
    plt.figure(figsize=(16, 5))
    plt.plot(dates_test, y_test, color='#1a1a1a', linewidth=1.3, label='Actual')
    plt.plot(dates_test, stack_preds, color='tomato', linewidth=1.0, linestyle='--', alpha=0.85,
             label='Stack (original)')
    plt.plot(dates_test, stack_preds_log, color='#2ca02c', linewidth=1.0, linestyle='--', alpha=0.85,
             label='Stack (log1p)')
    plt.title('Log1p Transform Comparison - 2017 Test Period', fontsize=13, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('plot5_log1p_comparison.png', dpi=150)
    plt.close()

    print("All 5 plots saved successfully.")