"""Train + evaluate the scorecard and export dashboard data.   Usage: python -m src.train"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from . import config as C
from .analytics import auc_safe, psi_numeric
from .scorecard import Scorecard

SAMPLE_SIZE = 250_000


def metrics(y, p):
    auc = roc_auc_score(y, p)
    ks = ks_2samp(p[y == 1], p[y == 0]).statistic
    return {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ks": round(ks, 4)}


def main():
    C.REPORT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(C.PROCESSED_PATH, parse_dates=["issue_d"])
    train = df[df["issue_d"].dt.year <= C.TRAIN_END_YEAR]
    test = df[df["issue_d"].dt.year > C.TRAIN_END_YEAR]
    print(f"Train: {len(train):,} loans (<= {C.TRAIN_END_YEAR}) | Test: {len(test):,} loans")
    Xtr, ytr = train[C.FEATURES], train[C.TARGET].values
    Xte, yte = test[C.FEATURES], test[C.TARGET].values

    # 1) Scorecard
    sc = Scorecard().fit(Xtr, ytr)
    res = {
        "n_loans": int(len(df)), "n_train": int(len(train)), "n_test": int(len(test)),
        "default_rate": float(df[C.TARGET].mean()),
        "train_default_rate": float(ytr.mean()), "test_default_rate": float(yte.mean()),
        "train_end_year": C.TRAIN_END_YEAR,
        "scorecard_train": metrics(ytr, sc.pd_(Xtr)),
        "scorecard_test": metrics(yte, sc.pd_(Xte)),
        "features_used": sc.features,
        "score_psi_train_vs_test": round(psi_numeric(sc.score(Xtr), sc.score(Xte)), 4),
        "scaling": {"base_score": C.BASE_SCORE, "base_odds": C.BASE_ODDS, "pdo": C.PDO},
    }

    # 2) Benchmark: gradient boosting on the same features
    both = pd.get_dummies(pd.concat([Xtr, Xte]), columns=C.CATEGORICAL_FEATURES, dtype=float)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)
    gb.fit(both.iloc[:len(Xtr)], ytr)
    pgb_all = gb.predict_proba(both)[:, 1]
    p_gb_test = pgb_all[len(Xtr):]
    res["gradient_boosting_test"] = metrics(yte, p_gb_test)
    res["gradient_boosting_train"] = metrics(ytr, pgb_all[:len(Xtr)])

    # 3) Score everything (train first, then test: same order as `both`)
    full = pd.concat([train.assign(set="train"), test.assign(set="test")])
    full["score"] = sc.score(full[C.FEATURES])
    full["pd_hat"] = sc.pd_(full[C.FEATURES])
    full["pd_gb"] = pgb_all

    # 4) Dashboard exports
    rows = []
    for yr, g in full.groupby(full["issue_d"].dt.year):
        rows.append({"year": int(yr), "loans": len(g), "default_rate": g[C.TARGET].mean(),
                     "avg_score": g["score"].mean(), "avg_pd": g["pd_hat"].mean(),
                     "auc": auc_safe(g[C.TARGET], g["pd_hat"]), "auc_gb": auc_safe(g[C.TARGET], g["pd_gb"])})
    yearly = pd.DataFrame(rows)
    yearly.to_csv(C.REPORT_DIR / "yearly_summary.csv", index=False)

    sample = full.sample(n=min(SAMPLE_SIZE, len(full)), random_state=42).sort_values("issue_d")
    for c in ["pd_hat", "pd_gb"]:
        sample[c] = sample[c].round(5)
    sample.round(3).to_csv(C.REPORT_DIR / "scored_sample.csv.gz", index=False)

    # 5) Static reports
    sc.binner.iv_.rename("iv").to_csv(C.REPORT_DIR / "iv_table.csv")
    sc.points_table().to_csv(C.REPORT_DIR / "scorecard_table.csv", index=False)
    t = full[full["set"] == "test"].copy()
    t["band"] = pd.qcut(t["score"], 10, duplicates="drop")
    t.groupby("band", observed=True).agg(loans=(C.TARGET, "size"), actual_bad_rate=(C.TARGET, "mean"),
                                         avg_pd=("pd_hat", "mean")).to_csv(C.REPORT_DIR / "score_bands.csv")
    cuts = []
    for cut in range(int(t["score"].min()), int(t["score"].max()), 10):
        ap = t[t["score"] >= cut]
        if len(ap):
            cuts.append({"cutoff": cut, "approval_rate": len(ap) / len(t),
                         "bad_rate_of_approved": ap[C.TARGET].mean()})
    pd.DataFrame(cuts).to_csv(C.REPORT_DIR / "cutoff_analysis.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, p in [("Scorecard", sc.pd_(Xte)), ("Gradient boosting", p_gb_test)]:
        fpr, tpr, _ = roc_curve(yte, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[0].set(title="ROC (out-of-time test)", xlabel="FPR", ylabel="TPR"); ax[0].legend()
    ax[1].hist(t.loc[t[C.TARGET] == 0, "score"], bins=40, alpha=.6, label="Paid")
    ax[1].hist(t.loc[t[C.TARGET] == 1, "score"], bins=40, alpha=.6, label="Defaulted")
    ax[1].set(title="Score distribution", xlabel="Credit score"); ax[1].legend()
    fig.tight_layout(); fig.savefig(C.REPORT_DIR / "evaluation.png", dpi=130)

    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(res, indent=2))
    joblib.dump(sc, C.MODEL_PATH)
    print(json.dumps(res, indent=2))
    print(f"\nSaved model -> {C.MODEL_PATH}\nSaved dashboard data -> {C.REPORT_DIR}")


if __name__ == "__main__":
    main()
