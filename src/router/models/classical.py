"""Future Dummy, Logistic Regression, Random Forest, and XGBoost routers.

All candidates predict the same rescue target. Fit on train only, choose
hyperparameters on validation, and expose calibrated probabilities for the
routing threshold study where appropriate.
"""
