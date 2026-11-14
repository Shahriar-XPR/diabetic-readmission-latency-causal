"""Classification metrics and wall-clock inference measurement."""
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def evaluate_model(model, X_test, y_test):
    start = time.perf_counter()
    prediction = np.asarray(model.predict(X_test)).reshape(-1).astype(int)
    probability = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - start
    return {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "auroc": roc_auc_score(y_test, probability),
        "pr_auc": average_precision_score(y_test, probability),
        "inference_seconds": elapsed,
        "predictions": prediction,
        "probabilities": probability,
    }
