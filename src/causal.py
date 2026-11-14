"""DirectLiNGAM discovery and DR-Learner treatment-effect estimation."""
import numpy as np
import pandas as pd
import networkx as nx
from .config import CONFOUNDERS, ACTIONABLE_TREATMENTS, RANDOM_STATE


def direct_lingam_analysis(X, y, threshold=0.05):
    from lingam import DirectLiNGAM
    frame=X.copy(); frame["readmitted_binary"]=np.asarray(y)
    model=DirectLiNGAM(random_state=RANDOM_STATE).fit(frame.values)
    adjacency=pd.DataFrame(model.adjacency_matrix_,index=frame.columns,columns=frame.columns)
    graph=nx.DiGraph()
    for source in adjacency.index:
        for target in adjacency.columns:
            weight=adjacency.loc[source,target]
            if abs(weight)>threshold: graph.add_edge(source,target,weight=float(weight))
    strength={node:sum(abs(graph[node][v]["weight"]) for v in graph.successors(node))+sum(abs(graph[p][node]["weight"]) for p in graph.predecessors(node)) for node in graph.nodes}
    return model, adjacency, graph, pd.Series(strength).sort_values(ascending=False)


def drlearner_effects(X, y, treatments=None, confounders=None, n_bootstrap=1000):
    """Estimate DR-Learner ATEs and percentile bootstrap intervals.

    This implements the manuscript's treatment encoding: T=1 when the
    treatment differs from its modal value. Negative effects denote predicted
    reductions in the readmission outcome under this encoding.
    """
    from econml.dr import DRLearner
    from xgboost import XGBClassifier, XGBRegressor
    treatments=treatments or ACTIONABLE_TREATMENTS
    confounders=[c for c in (confounders or CONFOUNDERS) if c in X.columns]
    Xc=X[confounders].copy(); Y=np.asarray(y)
    output=[]
    for treatment in treatments:
        if treatment not in X: continue
        T=(X[treatment] != X[treatment].mode().iloc[0]).astype(int).to_numpy()
        def fit_effect(indices):
            estimator=DRLearner(
                model_propensity=XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,eval_metric="logloss",tree_method="hist",random_state=RANDOM_STATE),
                model_regression=XGBRegressor(n_estimators=200,max_depth=5,learning_rate=0.05,tree_method="hist",random_state=RANDOM_STATE),
                random_state=RANDOM_STATE)
            estimator.fit(Y=Y[indices],T=T[indices],X=Xc.iloc[indices])
            return np.asarray(estimator.effect(Xc.iloc[indices])).reshape(-1)
        idx=np.arange(len(Xc)); cate=fit_effect(idx); ate=float(cate.mean())
        rng=np.random.default_rng(RANDOM_STATE); boot=[]
        for _ in range(n_bootstrap):
            sample=rng.choice(idx,size=len(idx),replace=True)
            try: boot.append(float(fit_effect(sample).mean()))
            except Exception: continue
        lo,hi=np.percentile(boot,[2.5,97.5]) if boot else (np.nan,np.nan)
        output.append({"treatment":treatment,"ate":ate,"ci_lower":lo,"ci_upper":hi,"percent_predicted_benefit":float((cate<0).mean())})
    return pd.DataFrame(output).sort_values("ate")
