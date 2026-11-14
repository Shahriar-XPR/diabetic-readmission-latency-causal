"""Cross-method consensus ranking (Table IV)."""
import pandas as pd


def consensus_table(shap_features, gemex_features, causal_features, actionable_features):
    layers={"SHAP":set(shap_features),"GEMEX":set(gemex_features),"Causal":set(causal_features),"Actionable":set(actionable_features)}
    universe=sorted(set().union(*layers.values()))
    rows=[]
    for feature in universe:
        row={"Feature":feature,**{name:feature in values for name,values in layers.items()}}
        row["Score"]=sum(row[name] for name in layers)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Score","Feature"],ascending=[False,True])
