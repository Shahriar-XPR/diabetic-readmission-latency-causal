"""SHAP and GEMEX explainability for the selected XGBoost reference model."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def shap_global_importance(model, X_test, feature_names=None, top_n=20):
    """Compute mean absolute SHAP contribution using XGBoost's pred_contribs."""
    import xgboost as xgb
    names = list(feature_names or getattr(X_test, "columns", range(X_test.shape[1])))
    dm = xgb.DMatrix(X_test, feature_names=names)
    contributions = model.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    importance = pd.Series(np.abs(contributions).mean(axis=0), index=names).sort_values(ascending=False)
    return importance.head(top_n)


def gemex_global_importance(model, X_train, X_test, feature_names=None, fast_instances=50):
    """Run the notebook's fast GEMEX pass and return its aggregated feature scores.

    Requires the external `gemex` package. A full-fidelity interaction pass is
    deliberately separate because it is substantially more expensive.
    """
    from gemex import Explainer, GemexConfig
    names = list(feature_names or getattr(X_train, "columns", range(X_train.shape[1])))
    class Wrapper:
        def __init__(self, fitted): self.fitted=fitted
        def predict_proba(self, values): return self.fitted.predict_proba(values)
    cfg=GemexConfig(n_geodesic_steps=10,n_reference_samples=20,interaction_order=1,model_type="tree")
    explainer=Explainer(Wrapper(model),data_type="tabular",feature_names=names,class_names=["NoReadmission","Readmission"],config=cfg)
    scores={}
    for i in range(min(fast_instances,len(X_test))):
        explanation=explainer.explain(np.asarray(X_test)[i],X_reference=np.asarray(X_train))
        if explanation.fim_quality == "poor": continue
        for feature, value in explanation.top_features(20): scores[feature]=scores.get(feature,0)+abs(value)
    return pd.Series(scores,dtype=float).sort_values(ascending=False)


def save_importance_plot(importance, path, title):
    ax=importance.sort_values().plot(kind="barh",figsize=(9,6)); ax.set_title(title); ax.set_xlabel("Mean absolute contribution / aggregated score")
    ax.figure.tight_layout(); ax.figure.savefig(path,dpi=200); plt.close(ax.figure)
