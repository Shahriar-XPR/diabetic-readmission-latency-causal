"""Train tuned classifiers; exports Tables II-III-compatible results and Fig. 2."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from .config import DATA_PATH, RANDOM_STATE, TEST_SIZE, RESULTS_DIR, FIGURES_DIR
from .preprocessing import preprocess_dataframe
from .models import make_model
from .evaluation import evaluate_model


def prepare_data(csv_path=DATA_PATH):
    raw = pd.read_csv(csv_path)
    frame = preprocess_dataframe(raw)
    X = frame.drop(columns="readmitted")
    y = frame["readmitted"]
    # As in the paper, SMOTE is applied before the 75:25 split.
    X_bal, y_bal = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X, y, X_train, X_test, y_train, y_test, scaler


def run(csv_path=DATA_PATH, device="cpu"):
    _, _, X_train, X_test, y_train, y_test, _ = prepare_data(csv_path)
    rows = []
    fitted = {}
    for name in ["catboost", "random_forest", "xgboost", "lightgbm", "tabnet"]:
        model = make_model(name, tuned=True, device=device)
        if name == "tabnet":
            model.fit(X_train.astype("float32"), y_train.to_numpy().astype("int64"), eval_set=[(X_test.astype("float32"), y_test.to_numpy().astype("int64"))], eval_name=["valid"], eval_metric=["auc"], max_epochs=100, patience=20, batch_size=2048, virtual_batch_size=256)
            result = evaluate_model(model, X_test.astype("float32"), y_test)
        else:
            model.fit(X_train, y_train)
            result = evaluate_model(model, X_test, y_test)
        fitted[name] = model
        rows.append({"model":name, **{k:v for k,v in result.items() if k not in ["predictions","probabilities"]}})
        print(name, rows[-1])
    RESULTS_DIR.mkdir(exist_ok=True); FIGURES_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "reproduced_model_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(out.inference_seconds, out.accuracy)
    for _, r in out.iterrows(): ax.annotate(r.model, (r.inference_seconds,r.accuracy))
    ax.set_xscale("log"); ax.set_xlabel("Inference time on test set (s, log scale)"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy–inference trade-off (reproduced run)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "reproduced_accuracy_latency.png", dpi=200)
    return fitted, out

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--csv", default=str(DATA_PATH)); parser.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    args=parser.parse_args(); run(args.csv,args.device)
