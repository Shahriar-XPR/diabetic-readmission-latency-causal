"""Model factories and Optuna search-space definitions."""
from .config import RANDOM_STATE, TUNED_PARAMS


def make_model(name, tuned=True, device="cpu"):
    key = name.lower().replace(" ", "_")
    p = TUNED_PARAMS[key].copy() if tuned else {}
    if key == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**p, loss_function="Logloss", eval_metric="Logloss", random_seed=RANDOM_STATE, verbose=False, task_type="CPU")
    if key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**p, random_state=RANDOM_STATE, n_jobs=-1)
    if key == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**p, objective="binary:logistic", eval_metric="auc", tree_method="hist", device=device, random_state=RANDOM_STATE, n_jobs=-1)
    if key == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**p, objective="binary", random_state=RANDOM_STATE, verbosity=-1)
    if key == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier
        from pytorch_tabnet.metrics import AUC
        lr = p.pop("lr", 0.02)
        return TabNetClassifier(**p, optimizer_params={"lr": lr}, seed=RANDOM_STATE, verbose=0)
    raise ValueError(f"Unknown model: {name}")


def optuna_search_space(name, trial):
    """Notebook-inspired search spaces. Used by train.py's optional tuning."""
    key = name.lower().replace(" ", "_")
    if key == "random_forest":
        return dict(n_estimators=trial.suggest_int("n_estimators",100,1000), max_depth=trial.suggest_int("max_depth",3,30), min_samples_split=trial.suggest_int("min_samples_split",2,20), min_samples_leaf=trial.suggest_int("min_samples_leaf",1,20), max_features=trial.suggest_categorical("max_features",["sqrt","log2",None]), class_weight=trial.suggest_categorical("class_weight",["balanced","balanced_subsample"]))
    if key == "xgboost":
        return dict(n_estimators=trial.suggest_int("n_estimators",100,1000), max_depth=trial.suggest_int("max_depth",3,12), learning_rate=trial.suggest_float("learning_rate",0.001,0.3,log=True), subsample=trial.suggest_float("subsample",0.5,1.0), colsample_bytree=trial.suggest_float("colsample_bytree",0.5,1.0), gamma=trial.suggest_float("gamma",0,5), min_child_weight=trial.suggest_int("min_child_weight",1,10), reg_alpha=trial.suggest_float("reg_alpha",1e-8,10,log=True), reg_lambda=trial.suggest_float("reg_lambda",1e-8,10,log=True))
    if key == "lightgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators",100,1200), learning_rate=trial.suggest_float("learning_rate",0.005,0.2,log=True), max_depth=trial.suggest_int("max_depth",3,15), num_leaves=trial.suggest_int("num_leaves",20,180), min_child_samples=trial.suggest_int("min_child_samples",10,120), subsample=trial.suggest_float("subsample",0.5,1), colsample_bytree=trial.suggest_float("colsample_bytree",0.5,1), reg_alpha=trial.suggest_float("reg_alpha",1e-8,10,log=True), reg_lambda=trial.suggest_float("reg_lambda",1e-8,10,log=True))
    raise NotImplementedError(f"Search space not separately defined for {name}; use the notebook for its full study configuration.")
