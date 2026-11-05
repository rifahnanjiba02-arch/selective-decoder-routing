"""Week 2 Task 3: classical rescue-router training (scaffold only).

Implement a Dummy baseline, Logistic Regression, Random Forest, and XGBoost
using the same leakage-safe X columns and rescue target from route_to_correlated.
Fit preprocessing and models on train only, using explicit allowed columns and
remainder='drop' so unexpected fields cannot pass through. Encode basis, record
seeds and hyperparameters, and return one [0, 1] rescue probability per
validation row.
Do not claim probability calibration without checking it.

See configs/models.yaml, tests/test_classical.py, and
docs/WEEK_2_TASK_GUIDE.md. No model-training functions are implemented yet.
"""
