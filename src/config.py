"""Central configuration and paper-reported tuned hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "diabetic_data.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET = "readmitted"

# Hyperparameters transcribed from Table II of the supplied manuscript.
# These are the paper's reported settings, not necessarily identical to
# every final-model cell in the exploratory notebook.
TUNED_PARAMS = {
    "catboost": {
        "iterations": 389, "depth": 10, "learning_rate": 0.096,
        "l2_leaf_reg": 0.025, "bagging_temperature": 2.23,
        "bootstrap_type": "Bayesian", "boosting_type": "Plain",
        "od_wait": 41, "min_data_in_leaf": 101,
    },
    "random_forest": {
        "n_estimators": 965, "max_depth": 30,
        "min_samples_split": 5, "min_samples_leaf": 1,
        "max_features": None, "class_weight": "balanced_subsample",
    },
    "xgboost": {
        "n_estimators": 316, "max_depth": 11, "learning_rate": 0.099,
        "subsample": 0.82, "colsample_bytree": 0.55, "gamma": 1.18,
        "min_child_weight": 5, "reg_alpha": 0.007, "reg_lambda": 0.0002,
    },
    "lightgbm": {
        "n_estimators": 922, "learning_rate": 0.072, "max_depth": 11,
        "num_leaves": 142, "min_child_samples": 95,
        "subsample": 0.77, "colsample_bytree": 0.83,
        "reg_alpha": 0.0, "reg_lambda": 1.55,
    },
    "tabnet": {
        "n_d": 40, "n_a": 40, "n_steps": 5, "gamma": 1.13,
        "lambda_sparse": 1.4e-5, "lr": 0.016, "mask_type": "entmax",
    },
}

# The paper's causal-actionability analysis includes HbA1c assessment.
CONFOUNDERS = [
    "race", "gender", "age", "admission_type_id",
    "discharge_disposition_id", "admission_source_id", "time_in_hospital",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult",
]
ACTIONABLE_TREATMENTS = [
    "diabetesMed", "insulin", "metformin", "glipizide",
    "glyburide", "change", "A1Cresult",
]
