"""Create analysis artifacts from model and explanation result files."""
import argparse
import pandas as pd
from .config import RESULTS_DIR, FIGURES_DIR


def summarize_causal(causal_csv):
    effects=pd.read_csv(causal_csv)
    effects.to_csv(RESULTS_DIR/"table_v_drlearner_effects.csv",index=False)
    return effects


def plot_model_metrics(metrics_csv):
    import matplotlib.pyplot as plt
    data=pd.read_csv(metrics_csv)
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(data.inference_seconds,data.accuracy)
    for _,r in data.iterrows(): ax.annotate(r.model,(r.inference_seconds,r.accuracy))
    ax.set_xscale("log"); ax.set_xlabel("Inference time (s, log scale)"); ax.set_ylabel("Accuracy")
    fig.tight_layout(); fig.savefig(FIGURES_DIR/"fig2_accuracy_latency_reproduced.png",dpi=200)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--metrics",default=str(RESULTS_DIR/"reproduced_model_metrics.csv")); a=p.parse_args(); plot_model_metrics(a.metrics)
