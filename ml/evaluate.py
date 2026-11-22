"""
Landas model evaluation (Option B: Fleming VARK label from saved q1..q16 answers).

Reads students' 16 saved answers, derives each student's dominant Fleming VARK
label (V, A, R, K) from their responses,
evaluates a Decision Tree and a Neural Network via stratified cross-validation,
and writes:
  ../backend/model_metrics.json   -> shown on the website's "Model performance" page
  output/                         -> tables and the confusion-matrix image for your paper

Usage (run on your own computer):
  pip install pandas scikit-learn matplotlib psycopg2-binary python-dotenv
  python evaluate.py                    # reads DATABASE_URL from ../backend/.env
  python evaluate.py --csv data.csv     # or from a CSV (columns: q1..q16)
  python evaluate.py --export data.csv  # just export the dataset and stop

Then commit backend/model_metrics.json and redeploy so the website shows it.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

LABELS = ["V", "A", "R", "K"]
NAMES = {
    "V": "Visual",
    "A": "Auditory",
    "R": "Read/Write",
    "K": "Kinesthetic",
}
HERE = os.path.dirname(os.path.abspath(__file__))
QCOLS = [f"q{i}" for i in range(1, 17)]
MIN_LABELED = 10      # minimum records for meaningful evaluation metrics
RECOMMENDED = 100


def fleming_label(df):
    """Dominant VARK modality from q1..q16 (ties broken in V, A, R, K order)."""
    counts = np.stack([(df[QCOLS] == l).sum(axis=1).values for l in LABELS], axis=1)
    return np.array([LABELS[i] for i in counts.argmax(axis=1)])


def load_from_db():
    import psycopg2

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(HERE, "..", "backend", ".env"))
    except ImportError:
        pass

    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL not found. Set it in backend/.env or use --csv.")

    conn = psycopg2.connect(url, sslmode="require")
    cur = conn.cursor()
    cur.execute("select answers from students where answers is not null")
    rows = cur.fetchall()
    conn.close()

    data = []
    for (answers,) in rows:
        parsed = answers if isinstance(answers, list) else json.loads(answers)
        if isinstance(parsed, list) and len(parsed) == 16:
            data.append({f"q{i + 1}": parsed[i] for i in range(16)})
    return pd.DataFrame(data)


def features(df):
    """One-hot encoding of q1..q16 (64 columns) plus the four VARK style counts."""
    X = pd.DataFrame(index=df.index)
    for q in QCOLS:
        for l in LABELS:
            X[f"{q}_{l}"] = (df[q] == l).astype(int)
    for l in LABELS:
        X[f"count_{l}"] = (df[QCOLS] == l).sum(axis=1)
    return X


def summarise(y_true, y_pred):
    rep = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "per_class": {k: rep[k] for k in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--export")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "backend", "model_metrics.json"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "output"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv) if args.csv else load_from_db()
    if df.empty:
        sys.exit("No student records with 16 saved answers found.")

    df = df.dropna(subset=QCOLS)
    df["fleming_label"] = fleming_label(df)

    if args.export:
        df.to_csv(args.export, index=False)
        sys.exit(f"Exported {len(df)} rows to {args.export}")

    n = len(df)
    print(f"{n} students with 16 saved answers (Fleming VARK reference labels)")
    print(df["fleming_label"].value_counts().reindex(LABELS, fill_value=0).to_string())

    if n < MIN_LABELED:
        sys.exit(
            f"Need at least {MIN_LABELED} students with saved answers (have {n}). "
            f"Aim for {RECOMMENDED}+ before reporting results."
        )

    X = features(df)
    y = df["fleming_label"].values

    smallest = int(pd.Series(y).value_counts().min())
    folds = int(min(5, smallest))
    if folds < 2:
        sys.exit("At least one VARK learning style has fewer than 2 students; collect more data.")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    results = {"Fleming dominant rule": summarise(y, fleming_label(df))}
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=42),
    }
    for name, model in models.items():
        results[name] = summarise(y, cross_val_predict(model, X, y, cv=cv))

    ml_names = list(models)
    best = max(ml_names, key=lambda k: results[k]["macro_f1"])
    b = results[best]

    print(f"\n{'Model':36s} Accuracy  MacroP  MacroR  MacroF1")
    for k, r in results.items():
        print(
            f"{k:36s} {r['accuracy']:.3f}     {r['macro_precision']:.3f}   "
            f"{r['macro_recall']:.3f}   {r['macro_f1']:.3f}"
        )

    print(f"\nBest model: {best}\n")
    print(
        classification_report(
            y,
            cross_val_predict(models[best], X, y, cv=cv),
            labels=LABELS,
            target_names=[NAMES[l] for l in LABELS],
            zero_division=0,
        )
    )

    note = (
        f"{folds}-fold cross-validation against Fleming VARK labels derived from q1..q16. "
        f"Best model selected by cross-validation Macro F1."
        + (f" Small sample (n={n}); treat results as preliminary." if n < RECOMMENDED else "")
    )
    metrics = {
        "best_model": best,
        "accuracy": b["accuracy"],
        "macro_f1": b["macro_f1"],
        "macro_precision": b["macro_precision"],
        "macro_recall": b["macro_recall"],
        "labels": LABELS,
        "per_class": b["per_class"],
        "confusion_matrix": b["confusion_matrix"],
        "n_total": n,
        "folds": folds,
        "all_models": {
            k: {
                m: v
                for m, v in r.items()
                if m in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
            }
            for k, r in results.items()
        },
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame(metrics["all_models"]).T.round(4).to_csv(
        os.path.join(args.outdir, "model_comparison.csv")
    )
    pd.DataFrame(b["per_class"]).T.round(4).to_csv(
        os.path.join(args.outdir, "per_class_metrics.csv")
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(
        np.array(b["confusion_matrix"]),
        display_labels=[NAMES[l] for l in LABELS],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Predicted style")
    ax.set_ylabel("Fleming VARK style")
    ax.set_title(f"Confusion matrix - {best}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "confusion_matrix.png"), dpi=200)
    print(f"Wrote {os.path.abspath(args.out)} and files in {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()