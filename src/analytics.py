"""Pure-pandas analytics used by the dashboard (no Streamlit/Plotly here)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def auc_safe(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def roc_df(y, p, max_points=600):
    fpr, tpr, _ = roc_curve(y, p)
    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def ks_df(y, p, max_points=500):
    """Cumulative bad/good capture sorted from riskiest to safest. Returns (curve, ks, pct_at_ks)."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    B, G = d["y"].sum(), len(d) - d["y"].sum()
    d["cum_bad"] = d["y"].cumsum() / B
    d["cum_good"] = (1 - d["y"]).cumsum() / G
    d["pct_population"] = (np.arange(len(d)) + 1) / len(d)
    d["ks"] = (d["cum_bad"] - d["cum_good"]).abs()
    step = max(1, len(d) // max_points)
    curve = d.iloc[::step][["pct_population", "cum_bad", "cum_good", "ks"]]
    i = d["ks"].idxmax()
    return curve, float(d.loc[i, "ks"]), float(d.loc[i, "pct_population"])


def lift_table(y, p, bins=10):
    """Decile 1 = riskiest 10% by predicted PD."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), bins, labels=False) + 1
    g = d.groupby("decile").agg(loans=("y", "size"), bads=("y", "sum"),
                                bad_rate=("y", "mean"), avg_pd=("p", "mean")).reset_index()
    g["lift"] = g["bad_rate"] / d["y"].mean()
    g["cum_bads_pct"] = g["bads"].cumsum() / g["bads"].sum()
    g["cum_pop_pct"] = g["loans"].cumsum() / len(d)
    return g


def calibration_table(y, p, bins=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False) + 1
    return d.groupby("bin").agg(loans=("y", "size"), avg_pd=("p", "mean"), actual=("y", "mean")).reset_index()


def confusion(y, p, thr):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "accuracy": (tp + tn) / max(1, len(y)),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def threshold_sweep(y, p, thresholds):
    return pd.DataFrame([{"threshold": t, **{k: v for k, v in confusion(y, p, t).items()
                                              if k in ("precision", "recall", "f1")}} for t in thresholds])


# ---------- population stability ----------
def psi_status(v):
    if pd.isna(v):
        return "n/a"
    return "Stable" if v < 0.1 else ("Moderate shift" if v < 0.25 else "Significant shift")


def psi_numeric(expected, actual, bins=10):
    e = pd.Series(expected).dropna().to_numpy(float)
    a = pd.Series(actual).dropna().to_numpy(float)
    if len(e) == 0 or len(a) == 0:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ep = np.histogram(e, edges)[0] / len(e) + 1e-6
    ap = np.histogram(a, edges)[0] / len(a) + 1e-6
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def psi_categorical(expected, actual):
    e = pd.Series(expected).fillna("Missing").astype(str)
    a = pd.Series(actual).fillna("Missing").astype(str)
    cats = sorted(set(e) | set(a))
    ep = e.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy() + 1e-6
    ap = a.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy() + 1e-6
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def feature_psi_table(train, test, numeric, categorical):
    rows = [{"feature": c, "psi": psi_numeric(train[c], test[c])} for c in numeric if c in train]
    rows += [{"feature": c, "psi": psi_categorical(train[c], test[c])} for c in categorical if c in train]
    t = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    t["status"] = t["psi"].map(psi_status)
    return t


# ---------- BI helpers ----------
def _fmt(v):
    if np.isinf(v):
        return "inf"
    return f"{v / 1000:g}k" if abs(v) >= 1000 and v % 1000 == 0 else f"{v:g}"


def make_labels(bins):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if np.isinf(hi):
            out.append(f"≥{_fmt(lo)}")
        elif np.isinf(lo):
            out.append(f"<{_fmt(hi)}")
        else:
            out.append(f"{_fmt(lo)}–{_fmt(hi)}")
    return out


def _agg(df, target):
    agg = {"loans": (target, "size"), "defaults": (target, "sum"), "default_rate": (target, "mean")}
    if "pd_hat" in df:
        agg["avg_pd"] = ("pd_hat", "mean")
    if "score" in df:
        agg["avg_score"] = ("score", "mean")
    return agg


def band_table(df, col, bins, target="default"):
    """Default rate by fixed bands [lo, hi)."""
    b = pd.cut(df[col], bins=bins, labels=make_labels(bins), right=False)
    g = df.groupby(b, observed=True).agg(**_agg(df, target))
    g.index = g.index.astype(str)
    g.index.name = "band"
    return g.reset_index()


def quantile_band_table(df, col, q=10, target="default"):
    b = pd.qcut(df[col], q, duplicates="drop")
    g = df.groupby(b, observed=True).agg(**_agg(df, target))
    g.index = g.index.astype(str)
    g.index.name = "band"
    return g.reset_index()


def segment_table(df, col, target="default"):
    g = df.groupby(col, observed=True, dropna=False).agg(**_agg(df, target), avg_loan=("loan_amnt", "mean"))
    g.index = g.index.astype(str)
    g.index.name = "segment"
    g = g.reset_index()
    numeric = pd.api.types.is_numeric_dtype(df[col])
    return g.sort_values("segment", key=lambda s: pd.to_numeric(s, errors="coerce")) if numeric \
        else g.sort_values("loans", ascending=False)


def pivot_rate(df, xcol, ycol, q=6, min_count=30):
    d = df[[xcol, ycol, "default"]].dropna()
    xb = pd.qcut(d[xcol], q, duplicates="drop")
    yb = pd.qcut(d[ycol], q, duplicates="drop")
    rate = d.pivot_table(index=yb, columns=xb, values="default", aggfunc="mean", observed=True)
    cnt = d.pivot_table(index=yb, columns=xb, values="default", aggfunc="size", observed=True)
    rate = rate.where(cnt >= min_count)
    rate.index, rate.columns = rate.index.astype(str), rate.columns.astype(str)
    return rate


# ---------- strategy / profit ----------
def profit_vector(df, margin=0.12, lgd=0.60):
    """Per-loan profit: good loan earns margin*amount, defaulted loan loses lgd*amount."""
    L = df["loan_amnt"].to_numpy(float)
    return np.where(df["default"].to_numpy() == 0, margin * L, -lgd * L)


def cutoff_table(df, margin=0.12, lgd=0.60, step=2):
    s, y = df["score"].to_numpy(), df["default"].to_numpy()
    L, pr = df["loan_amnt"].to_numpy(float), profit_vector(df, margin, lgd)
    rows = []
    for c in np.arange(int(s.min()), int(s.max()) + 1, step):
        m = s >= c
        n = int(m.sum())
        if n == 0:
            break
        rows.append({"cutoff": int(c), "approval_rate": n / len(df), "bad_rate": float(y[m].mean()),
                     "bads_avoided_pct": 1 - y[m].sum() / max(1, y.sum()),
                     "exposure": float(L[m].sum()), "net_profit": float(pr[m].sum()),
                     "return_on_exposure": float(pr[m].sum() / L[m].sum())})
    return pd.DataFrame(rows)


def best_cutoff(df, margin=0.12, lgd=0.60):
    t = cutoff_table(df, margin, lgd)
    return int(t.loc[t["net_profit"].idxmax(), "cutoff"])
