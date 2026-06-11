# 📈 Retail Sales Demand Forecasting (Future Interns Task 1)

**Status:** Completed  
**Goal:** Build a production-grade time-series forecasting model using historical business data to predict retail sales, accompanied by clear visuals and actionable business insights.

## 🚀 Project Overview
This repository contains a highly optimized **Stacked Ensemble** machine learning pipeline designed to forecast daily retail sales. Using 5 years of historical business data, the pipeline processes temporal features, prevents data leakage via chronological cross-validation, and applies a log-level target transformation to achieve a highly accurate predictive model. 

### 📊 Key Results
* **Baseline Model (Linear Regression) MAPE:** 29.34%
* **Final Stacked Ensemble (Log1p Transformed) MAPE:** **20.55%**
* **Business Impact:** Reduced forecasting error by nearly 9%, allowing for tighter inventory management and reduced overstock/stockout costs.

---

## 🏗️ Architectural Design Document (The Blueprint)
*This architecture was rigorously designed to prevent time-series data leakage and optimize the blending of linear and tree-based algorithms.*

### 1. Leakage Prevention Protocol
* **Chronological Split:** The training data is strictly bounded from 2013–2016. The test data is strictly 2017.
* **Out-of-Fold (OOF) Meta-Features:** The base models generate predictions for the meta-learner using `KFold(shuffle=False)` mapped to a chronological timeline. This prevents "time-traveling" data leakage where future sales accidentally inform past predictions.

### 2. Feature Engineering & Scaling Protocol
* **Time-Index Transformation:** Raw years (e.g., 2013) were transformed to a 0-based index (`year_idx`) to prevent linear algorithms from exploding on 4-digit magnitudes.
* **Isolated Scaling:** The `LinearRegression` base model receives features wrapped in a `StandardScaler` pipeline. The tree-based models (`RandomForest`, `XGBoost`) receive raw, unscaled features as they are scale-invariant and greedy algorithms perform best without arbitrary bounds.

### 3. Log1p Target Transformation
* Retail sales data is highly right-skewed (standard days vs. massive holiday spikes). The target variable ($y$) is transformed using $log(1 + y)$ prior to training. This stabilizes variance, forces the models to optimize for percentage errors rather than absolute magnitude, and is inverted (`expm1`) exactly at the prediction step.

---

## 🧠 Model Breakdown: Pros & Cons

### 1. Linear Regression (Base Model 1)
* **Pros:** Extremely fast to train; highly interpretable; identifies broad, overarching macroeconomic trends easily.
* **Cons:** Completely fails at capturing non-linear relationships (like sudden holiday spikes); prone to wildly extrapolating trends into the future. 

### 2. Random Forest Regressor (Base Model 2)
* **Pros:** Highly robust to outliers; inherently captures complex interactions between features (e.g., "Month = December" AND "Day = Saturday"); does not require feature scaling.
* **Cons:** Computationally heavy; strictly interpolative (it physically cannot predict a number higher than the maximum value it saw in its training data).

### 3. XGBoost Regressor (Base Model 3)
* **Pros:** Best-in-class accuracy for tabular data; sequentially corrects its own errors using gradient descent logic; handles complex seasonality beautifully. 
* **Cons:** "Black-box" algorithm that is difficult to interpret directly; prone to overfitting if hyperparameters (`max_depth`, `learning_rate`) are not carefully constrained.

### 4. RidgeCV Regression (The Meta-Learner)
* **Pros:** Perfectly blends the base models using an $L_2$ regularization penalty. `RidgeCV` automatically runs an inner cross-validation to find the exact mathematical optimum for its penalty ($\alpha$). It handles the collinearity between the base models perfectly.
* **Cons:** Assumes the relationship between the base model predictions is purely linear.

---

## 📂 Repository Structure

The codebase is modularized for production-readiness rather than relying on a single monolithic script:

* `data_prep.py` - Handles CSV ingestion, zero-based temporal indexing, and chronological train/test splitting.
* `models.py` - Houses the Stacking Regressor architecture, Scaling Pipelines, and Log1p wrappers.
* `evaluation.py` - Calculates MAE, RMSE, and MAPE (handling zero-sales edge cases) and generates all diagnostic visualizations.
* `main.py` - The central execution script that orchestrates the pipeline.
* `saved_models/` - Directory containing the serialized `.joblib` models ready for deployment/inference.

---

## 📈 Visual Insights & Diagnostics
The pipeline automatically generates a suite of diagnostic plots (saved in the root directory upon execution):
1. **`plot1_timeline.png`:** A 365-day tracking timeline proving the model catches the seasonal spikes.
2. **`plot2_surface.png`:** A 3D topographical map of sales velocity by Month and Day of the Week.
3. **`plot3_rf_importance.png`:** Bar chart identifying which temporal features drive the most volume.
4. **`plot4_residuals.png`:** A senior-level residual analysis proving the model's errors are centered near zero without severe systemic bias.
5. **`plot5_log1p_comparison.png`:** An A/B test visualization proving the log-transform's superiority over the standard scale.

---

## ⚙️ How to Run
1. Ensure `train.csv` (Kaggle Store Item Demand dataset) is in the root directory.
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost matplotlib joblib`
3. Execute the pipeline:
   ```bash
   python main.py
