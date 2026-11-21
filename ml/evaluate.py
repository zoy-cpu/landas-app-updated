"""
Landas model evaluation.

Reads students' 16 saved answers and the learning style their TEACHER confirmed,
trains a Decision Tree and a Neural Network, and measures how well each one
(and the current "most-chosen style" rule) matches the teacher's label.

Results are written to:
  ../backend/model_metrics.json   -> shown on the website's "Model performance" page
  output/                         -> tables and the confusion-matrix image for your paper

Usage (run on your own computer):
  pip install pandas scikit-learn matplotlib psycopg2-binary python-dotenv
  python evaluate.py                    # reads DATABASE_URL from ../backend/.env
  python evaluate.py --csv data.csv     # or from a CSV (columns: q1..q16, teacher_label)
  python evaluate.py --export data.csv  # just export the dataset and stop

Then commit backend/model_metrics.json and redeploy so the website shows it.
"""
import argparse, json, os, sys, datetime
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

LABELS = ["V", "A", "R", "K"]
NAMES = {"V": "Visual", "A": "Auditory", "R": "Read/Write", "K": "Kinesthetic"}
HERE = os.path.dirname(os.path.abspath(__file__))
QCOLS = [f"q{i}" for i in range(1, 17)]
MIN_LABELED = 30      # below this the numbers are not meaningful
RECOMMENDED = 100


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
    cur.execute("select answers, teacher_label, style from students "
                "where answers is not null and teacher_label is not null")
    rows = cur.fetchall()
    conn.close()
    data = []
    for answers, label, style in rows:
        answers = answers if isinstance(answers, list) else json.loads(answers)
        data.append({**{f"q{i+1}": a for i, a in enumerate(answers)}, "teacher_label": label})
    return pd.DataFrame(data)


def features(df):
    """One-hot of each question's chosen style (64 columns) + the four style counts."""
    X = pd.DataFrame(index=df.index)
    for q in QCOLS:
        for l in LABELS:
            X[f"{q}_{l}"] = (df[q] == l).astype(int)
    for l in LABELS:
        X[f"count_{l}"] = (df[QCOLS] == l).sum(axis=1)
    return X


def rule_prediction(df):
    """The scoring rule the website uses today: most-chosen style (ties -> V, A, R, K order)."""
    counts = np.stack([(df[QCOLS] == l).sum(axis=1).values for l in LABELS], axis=1)
    return np.array([LABELS[i] for i in counts.argmax(axis=1)])


def summarise(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": rep["macro avg"]["precision"],
        "macro_recall": rep["macro avg"]["recall"],
        "macro_f1": rep["macro avg"]["f1-score"],
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
    df = df.dropna(subset=QCOLS + ["teacher_label"])
    if args.export:
        df.to_csv(args.export, index=False)
        sys.exit(f"Exported {len(df)} rows to {args.export}")

    n = len(df)
    print(f"{n} students with saved answers and a teacher label")
    print(df["teacher_label"].value_counts().reindex(LABELS, fill_value=0).to_string())
    if n < MIN_LABELED:
        sys.exit(f"Need at least {MIN_LABELED} teacher-confirmed students (have {n}). "
                 f"Aim for {RECOMMENDED}+ before reporting results.")

    X, y = features(df), df["teacher_label"].values
    smallest = pd.Series(y).value_counts().min()
    folds = int(min(5, smallest))
    if folds < 2:
        sys.exit("At least one learning style has fewer than 2 labeled students; collect more data.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    results = {"Current rule (most-chosen style)": summarise(y, rule_prediction(df))}
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
        print(f"{k:36s} {r['accuracy']:.3f}     {r['macro_precision']:.3f}   {r['macro_recall']:.3f}   {r['macro_f1']:.3f}")
    print(f"\nBest model: {best}\n")
    print(classification_report(y, cross_val_predict(models[best], X, y, cv=cv), labels=LABELS,
                                target_names=[NAMES[l] for l in LABELS], zero_division=0))

    note = (f"{folds}-fold cross-validation against teacher-confirmed labels. "
            f"Current scoring rule accuracy: {results['Current rule (most-chosen style)']['accuracy']*100:.1f}%."
            + (f" Small sample (n={n}); treat results as preliminary." if n < RECOMMENDED else ""))
    metrics = {
        "best_model": best,
        "accuracy": b["accuracy"], "macro_f1": b["macro_f1"],
        "macro_precision": b["macro_precision"], "macro_recall": b["macro_recall"],
        "labels": LABELS, "per_class": b["per_class"], "confusion_matrix": b["confusion_matrix"],
        "n_total": n, "n_test": n, "n_train": n - int(np.ceil(n / folds)), "folds": folds,
        "all_models": {k: {m: v for m, v in r.items() if m in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
                       for k, r in results.items()},
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame(metrics["all_models"]).T.round(4).to_csv(os.path.join(args.outdir, "model_comparison.csv"))
    pd.DataFrame(b["per_class"]).T.round(4).to_csv(os.path.join(args.outdir, "per_class_metrics.csv"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(np.array(b["confusion_matrix"]), display_labels=[NAMES[l] for l in LABELS]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Predicted style"); ax.set_ylabel("Teacher-confirmed style"); ax.set_title(f"Confusion matrix - {best}")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "confusion_matrix.png"), dpi=200)
    print(f"Wrote {os.path.abspath(args.out)} and files in {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
