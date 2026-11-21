"""Credit Scoring System: single-file Streamlit dashboard.

Run:  python -m streamlit run app.py
Needs: models/scorecard.joblib and reports/*  (created by `python -m src.train`),
       src/config.py and src/scorecard.py
"""
import inspect
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, roc_curve

from src import config as C
from src.scorecard import Scorecard  # noqa: F401  (needed to unpickle the model)

st.set_page_config(page_title="Credit Scoring System", page_icon="💳", layout="wide")


# ============================================================================
# UI HELPERS
# ============================================================================
BLUE, RED, GREEN, AMBER, GREY, LIGHT_BLUE = "#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#64748b", "#bfdbfe"

LABELS = {
    "loan_amnt": "Loan amount", "term_months": "Term (months)", "annual_inc": "Annual income",
    "dti": "Debt-to-income", "delinq_2yrs": "Delinquencies (2y)", "fico_avg": "FICO score",
    "inq_last_6mths": "Inquiries (6m)", "open_acc": "Open accounts", "pub_rec": "Public records",
    "revol_bal": "Revolving balance", "revol_util": "Revolving utilisation", "total_acc": "Total accounts",
    "mort_acc": "Mortgage accounts", "pub_rec_bankruptcies": "Bankruptcies",
    "emp_length_yrs": "Employment length (yrs)", "credit_history_yrs": "Credit history (yrs)",
    "loan_to_income": "Loan-to-income", "home_ownership": "Home ownership",
    "verification_status": "Income verification", "purpose": "Loan purpose",
}


def label(f):
    return LABELS.get(f, f.replace("_", " ").title())


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def money(x):
    return f"${x:,.0f}"


def iv_strength(v):
    return "Not useful" if v < 0.02 else "Weak" if v < 0.1 else "Medium" if v < 0.3 else "Strong"


def inject_css():
    st.markdown("""
    <style>
    [data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;padding:14px 16px;border-radius:12px}
    [data-testid="stMetricLabel"]{color:#475569}
    .block-container{padding-top:2rem}
    .tag{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:600;font-size:0.9rem}
    </style>""", unsafe_allow_html=True)


_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show_chart(fig, key=None, height=None):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=60, b=10), font=dict(size=13))
    if height:
        fig.update_layout(height=height)
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df, height=None):
    kw = {"hide_index": True}
    if height is not None:          # newer Streamlit rejects height=None
        kw["height"] = height
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:               # older Streamlit versions
        st.dataframe(df, use_container_width=True, **kw)


def kpi_row(items):
    """items: (label, value[, delta[, inverse_colour]])"""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        lab, val, *rest = it
        delta = rest[0] if rest else None
        inverse = rest[1] if len(rest) > 1 else False
        col.metric(lab, val, delta, delta_color="inverse" if inverse else "normal")


def volume_rate_chart(tbl, x, title, rate_cols=("default_rate",), count_col="loans", names=None):
    """Bars = loan count, line(s) = rate on secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=tbl[x].astype(str), y=tbl[count_col], name="Loans", marker_color=LIGHT_BLUE, secondary_y=False)
    colors = [RED, BLUE, AMBER]
    for i, rc in enumerate(rate_cols):
        nm = (names or {}).get(rc, "Default rate" if rc == "default_rate" else rc)
        fig.add_scatter(x=tbl[x].astype(str), y=tbl[rc], name=nm, mode="lines+markers",
                        line=dict(color=colors[i % 3], dash="solid" if i == 0 else "dash"), secondary_y=True)
    fig.update_yaxes(title_text="Loans", secondary_y=False)
    fig.update_yaxes(title_text="Rate", tickformat=".0%", secondary_y=True)
    fig.update_xaxes(type="category")
    fig.update_layout(title=title, legend=dict(orientation="h", y=1.12))
    return fig

# ============================================================================
# ANALYTICS
# ============================================================================
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

# ============================================================================
# PAGE: EXECUTIVE OVERVIEW
# ============================================================================
def render_overview(ctx):
    m, yearly, df = ctx["metrics"], ctx["yearly"], ctx["df"]
    te, gb = m["scorecard_test"], m["gradient_boosting_test"]
    st.title("💳 Credit Scoring System")
    st.caption("LendingClub loans issued 2012–2016 · WOE logistic-regression scorecard · out-of-time validation")

    kpi_row([
        ("Loans analysed", f"{m['n_loans']:,}"),
        ("Overall default rate", pct(m["default_rate"])),
        ("Test AUC", f"{te['auc']:.3f}", f"{te['auc'] - gb['auc']:+.3f} vs GBM"),
        ("Test Gini", f"{te['gini']:.3f}"),
        ("Test KS", f"{te['ks']:.3f}"),
        ("Score PSI", f"{m['score_psi_train_vs_test']:.3f}", psi_status(m["score_psi_train_vs_test"])),
    ])
    st.write("")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = volume_rate_chart(yearly, "year", "Loan volume, actual default rate and predicted PD by year",
                                   rate_cols=("default_rate", "avg_pd"),
                                   names={"default_rate": "Actual default rate", "avg_pd": "Avg predicted PD"})
        show_chart(fig, key="ov_year")
    with c2:
        test = df[df["set"] == "test"].copy()
        test["outcome"] = test["default"].map({0: "Paid", 1: "Defaulted"})
        fig = px.histogram(test, x="score", color="outcome", barmode="overlay", nbins=50, opacity=0.65,
                           color_discrete_map={"Paid": GREEN, "Defaulted": RED},
                           title="Score distribution by outcome (test)")
        show_chart(fig, key="ov_dist")

    bands = quantile_band_table(test, "score", 10)
    c3, c4 = st.columns([3, 2])
    with c3:
        fig = px.bar(bands, x="band", y="default_rate", text=bands["default_rate"].map(lambda v: f"{v:.0%}"),
                     title="Default rate by score decile (test) — lower score = higher risk",
                     color="default_rate", color_continuous_scale="RdYlGn_r")
        fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
        show_chart(fig, key="ov_bands")
    with c4:
        worst, best = bands["default_rate"].iloc[0], bands["default_rate"].iloc[-1]
        gap = test["default"].mean() - test["pd_hat"].mean()
        st.subheader("Key takeaways")
        st.markdown(f"""
- **Risk separation:** the lowest-score decile defaults at **{worst:.1%}** vs **{best:.1%}** for the highest
  (**{worst / max(best, 1e-9):.1f}×** difference).
- **Scorecard vs GBM:** AUC {te['auc']:.3f} vs {gb['auc']:.3f}. The scorecard gives up
  {gb['auc'] - te['auc']:.3f} AUC but is fully explainable.
- **Drift:** on the test period actual default is **{gap * 100:+.1f} pp** above predicted PD
  (default rates rose over time) → consider recalibration.
- **Stability:** train/test score PSI = {m['score_psi_train_vs_test']:.3f} ({psi_status(m['score_psi_train_vs_test']).lower()}).
""")

    with st.expander("ℹ️ About this project"):
        sc = m["scaling"]
        st.markdown(f"""
**Pipeline:** raw Kaggle data → resolved loans 2012–2016 → feature engineering → WOE binning & Information Value →
logistic regression → points scaling ({sc['base_score']} pts at {sc['base_odds']}:1 odds, +{sc['pdo']} pts doubles odds) → evaluation.
Trained on loans issued ≤ {m['train_end_year']} ({m['n_train']:,}), tested on later loans ({m['n_test']:,}).
Dashboards use a random sample of {len(df):,} scored loans; headline metrics use the full data.
""")

# ============================================================================
# PAGE: PORTFOLIO BI DASHBOARD
# ============================================================================
BANDS = {
    "FICO score": ("fico_avg", [0, 680, 700, 720, 740, 760, 780, np.inf]),
    "Debt-to-income (%)": ("dti", [-np.inf, 10, 15, 20, 25, 30, 35, np.inf]),
    "Annual income ($)": ("annual_inc", [0, 30000, 50000, 75000, 100000, 150000, np.inf]),
    "Revolving utilisation (%)": ("revol_util", [0, 20, 40, 60, 80, 100, np.inf]),
    "Loan amount ($)": ("loan_amnt", [0, 5000, 10000, 15000, 20000, 30000, np.inf]),
    "Credit history (yrs)": ("credit_history_yrs", [0, 5, 10, 15, 20, 30, np.inf]),
    "Inquiries (6m)": ("inq_last_6mths", [0, 1, 2, 3, 4, np.inf]),
    "Loan-to-income": ("loan_to_income", [0, 0.1, 0.2, 0.3, 0.5, np.inf]),
}
SEGMENTS = ["purpose", "home_ownership", "verification_status", "term_months", "emp_length_yrs"]
HEAT_VARS = ["fico_avg", "dti", "annual_inc", "loan_amnt", "revol_util", "credit_history_yrs", "loan_to_income"]


def _filters(df):
    with st.expander("🔎 Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        y0, y1 = int(df["year"].min()), int(df["year"].max())
        yrs = c1.slider("Issue year", y0, y1, (y0, y1))
        terms = c2.multiselect("Term (months)", sorted(df["term_months"].dropna().unique()))
        purposes = c3.multiselect("Purpose", sorted(df["purpose"].dropna().unique()))
        homes = c4.multiselect("Home ownership", sorted(df["home_ownership"].dropna().unique()))
        c5, c6, c7 = st.columns(3)
        verifs = c5.multiselect("Verification", sorted(df["verification_status"].dropna().unique()))
        f0, f1 = int(df["fico_avg"].min()), int(df["fico_avg"].max())
        fico = c6.slider("FICO range", f0, f1, (f0, f1))
        s0, s1 = int(df["score"].min()), int(df["score"].max())
        sc = c7.slider("Score range", s0, s1, (s0, s1))
    m = df["year"].between(*yrs) & df["fico_avg"].between(*fico) & df["score"].between(*sc)
    for col, sel in [("term_months", terms), ("purpose", purposes), ("home_ownership", homes),
                     ("verification_status", verifs)]:
        if sel:
            m &= df[col].isin(sel)
    return df[m]


def render_bi(ctx):
    df = ctx["df"]
    st.header("📊 Portfolio BI Dashboard")
    st.caption(f"Interactive analysis of a {len(df):,}-loan random sample of the 2012–2016 portfolio")
    f = _filters(df)
    if f.empty:
        st.warning("No loans match these filters.")
        return

    base = df["default"].mean()
    kpi_row([
        ("Loans", f"{len(f):,}", f"{len(f) / len(df):.0%} of portfolio"),
        ("Default rate", pct(f["default"].mean()), f"{(f['default'].mean() - base) * 100:+.1f} pp vs all", True),
        ("Avg loan", money(f["loan_amnt"].mean())),
        ("Exposure", f"${f['loan_amnt'].sum() / 1e6:,.1f}M"),
        ("Avg FICO", f"{f['fico_avg'].mean():.0f}"),
        ("Avg score", f"{f['score'].mean():.0f}"),
        ("Avg DTI", f"{f['dti'].mean():.1f}%"),
    ])
    st.write("")

    tabs = st.tabs(["📈 Trends", "🧩 Segments", "📉 Risk drivers", "🔥 Heatmap", "🎯 Score bands", "🗂 Data"])

    with tabs[0]:
        mth = f.assign(month=f["issue_d"].dt.to_period("M").dt.to_timestamp())
        t = mth.groupby("month").agg(loans=("default", "size"), default_rate=("default", "mean"),
                                     avg_pd=("pd_hat", "mean")).reset_index()
        fig = volume_rate_chart(t, "month", "Monthly vintage: volume, actual default rate vs predicted PD",
                                   rate_cols=("default_rate", "avg_pd"),
                                   names={"default_rate": "Actual default rate", "avg_pd": "Avg predicted PD"})
        fig.update_xaxes(type="date")
        show_chart(fig, key="bi_month")
        ty = f.groupby(["year", "term_months"]).agg(default_rate=("default", "mean")).reset_index()
        ty["term_months"] = ty["term_months"].astype(int).astype(str) + " months"
        fig = px.line(ty, x="year", y="default_rate", color="term_months", markers=True,
                      title="Default rate by issue year and term")
        fig.update_layout(yaxis_tickformat=".0%")
        fig.update_xaxes(dtick=1)
        show_chart(fig, key="bi_term_year")

    with tabs[1]:
        dim = st.selectbox("Segment by", SEGMENTS, format_func=label)
        seg = segment_table(f, dim)
        c1, c2 = st.columns([3, 2])
        with c1:
            show_chart(volume_rate_chart(seg, "segment", f"Default rate by {label(dim).lower()}"),
                          key="bi_seg")
        with c2:
            show = seg.assign(default_rate=seg["default_rate"].map(pct), avg_loan=seg["avg_loan"].round(0),
                              avg_score=seg["avg_score"].round(0)).drop(columns=["avg_pd"], errors="ignore")
            show_df(show)

    with tabs[2]:
        names = list(BANDS)
        picks = st.multiselect("Risk drivers to show", names, default=names[:4])
        cols = st.columns(2)
        for i, nm in enumerate(picks):
            col, bins = BANDS[nm]
            t = band_table(f.dropna(subset=[col]), col, bins)
            with cols[i % 2]:
                show_chart(volume_rate_chart(t, "band", f"Default rate by {nm}"), key=f"bi_drv_{i}")

    with tabs[3]:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis", HEAT_VARS, index=0, format_func=label)
        y = c2.selectbox("Y axis", HEAT_VARS, index=1, format_func=label)
        q = c3.slider("Bins per axis", 3, 10, 6)
        if x == y:
            st.info("Choose two different variables.")
        else:
            rate = pivot_rate(f, x, y, q=q)
            fig = px.imshow(rate, text_auto=".1%", aspect="auto", color_continuous_scale="RdYlGn_r",
                            labels=dict(x=label(x), y=label(y), color="Default rate"),
                            title=f"Default rate: {label(y)} × {label(x)} (cells with <30 loans hidden)")
            show_chart(fig, key="bi_heat", height=520)

    with tabs[4]:
        bands = quantile_band_table(f, "score", 10)
        show_chart(volume_rate_chart(bands, "band", "Score deciles: volume, actual default vs predicted PD",
                                           rate_cols=("default_rate", "avg_pd"),
                                           names={"default_rate": "Actual default rate", "avg_pd": "Avg predicted PD"}),
                      key="bi_bands")
        show_df(bands.assign(default_rate=bands["default_rate"].map(pct),
                                avg_pd=bands["avg_pd"].map(pct), avg_score=bands["avg_score"].round(0)))

    with tabs[5]:
        cols = ["issue_d", "set", "default", "score", "pd_hat"] + [c for c in ctx["features"]]
        st.write(f"Showing the first 500 of {len(f):,} filtered loans")
        show_df(f[cols].head(500), height=420)
        st.download_button("⬇️ Download filtered data (up to 50,000 rows)",
                           f[cols].head(50_000).to_csv(index=False).encode(), "filtered_loans.csv", "text/csv")

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
def _subset(df, choice):
    if choice.startswith("Test"):
        return df[df["set"] == "test"]
    if choice.startswith("Train"):
        return df[df["set"] == "train"]
    return df


def render_performance(ctx):
    df, m = ctx["df"], ctx["metrics"]
    st.header("🎯 Model Performance")
    choice = st.radio("Evaluation dataset", [f"Test (out-of-time, after {m['train_end_year']})",
                                             f"Train (≤ {m['train_end_year']})", "All"], horizontal=True)
    d = _subset(df, choice)
    y, p, pg = d["default"].to_numpy(), d["pd_hat"].to_numpy(), d["pd_gb"].to_numpy()
    st.caption(f"{len(d):,} sampled loans · default rate {d['default'].mean():.1%}")

    tabs = st.tabs(["📋 Summary", "📈 ROC & KS", "🚀 Lift & Gain", "⚖️ Calibration", "🧮 Confusion matrix", "🕒 Stability"])

    with tabs[0]:
        rows = [("Scorecard", "Train", m["scorecard_train"]), ("Scorecard", "Test", m["scorecard_test"]),
                ("Gradient boosting", "Train", m["gradient_boosting_train"]),
                ("Gradient boosting", "Test", m["gradient_boosting_test"])]
        tbl = pd.DataFrame([{"Model": a, "Set": b, **c} for a, b, c in rows]).rename(
            columns={"auc": "AUC", "gini": "Gini", "ks": "KS"})
        te, gb = m["scorecard_test"], m["gradient_boosting_test"]
        kpi_row([("Scorecard AUC (test)", f"{te['auc']:.3f}", f"{te['auc'] - gb['auc']:+.3f} vs GBM"),
                    ("Scorecard Gini (test)", f"{te['gini']:.3f}", f"{te['gini'] - gb['gini']:+.3f} vs GBM"),
                    ("Scorecard KS (test)", f"{te['ks']:.3f}", f"{te['ks'] - gb['ks']:+.3f} vs GBM"),
                    ("Overfit gap (AUC)", f"{m['scorecard_train']['auc'] - te['auc']:+.3f}", "train − test")])
        long = tbl.melt(id_vars=["Model", "Set"], var_name="Metric", value_name="Value")
        long["Series"] = long["Model"] + " – " + long["Set"]
        fig = px.bar(long, x="Metric", y="Value", color="Series", barmode="group", text_auto=".3f",
                     title="Model comparison (full data)")
        show_chart(fig, key="pf_cmp")
        show_df(tbl)
        st.info("Industry guide for application scorecards: AUC 0.65–0.75 · KS 0.25–0.40 is typical. "
                "Scorecard test AUC above train AUC means the model is not overfitting.")

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for nm, pp, col in [("Scorecard", p, BLUE), ("Gradient boosting", pg, AMBER)]:
                r = roc_df(y, pp)
                fig.add_scatter(x=r["fpr"], y=r["tpr"], name=f"{nm} (AUC {auc_safe(y, pp):.3f})",
                                line=dict(color=col))
            fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=GREY, dash="dash"))
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
            show_chart(fig, key="pf_roc")
        with c2:
            curve, ks, at = ks_df(y, p)
            fig = go.Figure()
            fig.add_scatter(x=curve["pct_population"], y=curve["cum_bad"], name="Cumulative bad", line=dict(color=RED))
            fig.add_scatter(x=curve["pct_population"], y=curve["cum_good"], name="Cumulative good", line=dict(color=GREEN))
            fig.add_vline(x=at, line_dash="dash", line_color=GREY)
            fig.update_layout(title=f"KS plot — KS = {ks:.3f} at {at:.0%} of population (sorted risky → safe)",
                              xaxis_title="% of applicants (riskiest first)", yaxis_title="Cumulative share",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            show_chart(fig, key="pf_ks")

    with tabs[2]:
        lt = lift_table(y, p)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(lt, x="decile", y="lift", text=lt["lift"].map(lambda v: f"{v:.2f}x"),
                         title="Lift by risk decile (1 = riskiest 10%)")
            fig.add_hline(y=1, line_dash="dash", line_color=GREY)
            fig.update_xaxes(type="category")
            show_chart(fig, key="pf_lift")
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=[0] + list(lt["cum_pop_pct"]), y=[0] + list(lt["cum_bads_pct"]), mode="lines+markers",
                            name="Scorecard", line=dict(color=BLUE))
            fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=GREY, dash="dash"))
            fig.update_layout(title="Cumulative gains: share of defaults captured",
                              xaxis_title="% of applicants (riskiest first)", yaxis_title="% of defaults captured",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            show_chart(fig, key="pf_gain")
        top30 = lt.loc[lt["decile"] <= 3, "bads"].sum() / lt["bads"].sum()
        st.success(f"Rejecting the riskiest **30%** of applicants would catch **{top30:.0%}** of all defaults.")
        show_df(lt.assign(bad_rate=lt["bad_rate"].map(pct), avg_pd=lt["avg_pd"].map(pct),
                             lift=lt["lift"].round(2), cum_bads_pct=lt["cum_bads_pct"].map(pct),
                             cum_pop_pct=lt["cum_pop_pct"].map(pct)))

    with tabs[3]:
        cal = calibration_table(y, p)
        fig = go.Figure()
        mx = float(max(cal["avg_pd"].max(), cal["actual"].max()) * 1.05)
        fig.add_scatter(x=[0, mx], y=[0, mx], name="Perfect calibration", line=dict(color=GREY, dash="dash"))
        fig.add_scatter(x=cal["avg_pd"], y=cal["actual"], mode="lines+markers", name="Scorecard",
                        line=dict(color=BLUE))
        fig.update_layout(title="Calibration: predicted PD vs actual default rate (deciles)",
                          xaxis_title="Average predicted PD", yaxis_title="Actual default rate",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        show_chart(fig, key="pf_cal")
        gap = (y.mean() - p.mean()) * 100
        if abs(gap) > 2:
            st.warning(f"Average predicted PD is {p.mean():.1%} vs actual {y.mean():.1%} on this dataset "
                       f"({'under' if gap > 0 else 'over'}estimating risk by {abs(gap):.1f} pp). "
                       "Ranking power is unaffected, but PD should be recalibrated before use in pricing/provisioning.")
        else:
            st.success(f"Well calibrated: predicted {p.mean():.1%} vs actual {y.mean():.1%}.")

    with tabs[4]:
        thr = st.slider("Classify as 'bad' when predicted PD ≥", 0.05, 0.95, float(min(0.95, max(0.05, round(y.mean(), 2)))), 0.01)
        c = confusion(y, p, thr)
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = px.imshow(np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]]), text_auto=True,
                            x=["Predicted good", "Predicted bad"], y=["Actual good", "Actual bad"],
                            color_continuous_scale="Blues", title="Confusion matrix")
            fig.update_layout(coloraxis_showscale=False)
            show_chart(fig, key="pf_conf")
        with c2:
            kpi_row([("Precision", pct(c["precision"])), ("Recall", pct(c["recall"])),
                        ("Specificity", pct(c["specificity"])), ("F1", f"{c['f1']:.3f}")])
            sw = threshold_sweep(y, p, np.arange(0.05, 0.96, 0.025))
            fig = px.line(sw.melt(id_vars="threshold", var_name="metric"), x="threshold", y="value", color="metric",
                          title="Precision / recall trade-off by threshold")
            fig.add_vline(x=thr, line_dash="dash", line_color=GREY)
            show_chart(fig, key="pf_sweep")

    with tabs[5]:
        ys = ctx["yearly"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_scatter(x=ys["year"], y=ys["auc"], name="Scorecard", mode="lines+markers", line=dict(color=BLUE))
            fig.add_scatter(x=ys["year"], y=ys["auc_gb"], name="Gradient boosting", mode="lines+markers",
                            line=dict(color=AMBER))
            fig.add_vline(x=m["train_end_year"] + 0.5, line_dash="dash", line_color=GREY)
            fig.update_layout(title="AUC by issue year (dashed line = train/test split)", xaxis_title="Issue year",
                              yaxis_title="AUC")
            fig.update_xaxes(dtick=1)
            show_chart(fig, key="pf_auc_year")
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=ys["year"], y=ys["default_rate"], name="Actual default rate", mode="lines+markers",
                            line=dict(color=RED))
            fig.add_scatter(x=ys["year"], y=ys["avg_pd"], name="Predicted PD", mode="lines+markers",
                            line=dict(color=BLUE, dash="dash"))
            fig.update_layout(title="Actual vs predicted default rate by year", yaxis_tickformat=".0%",
                              xaxis_title="Issue year")
            fig.update_xaxes(dtick=1)
            show_chart(fig, key="pf_pd_year")
        show_df(ys.assign(year=ys["year"].astype(int), loans=ys["loans"].astype(int),
                             default_rate=ys["default_rate"].map(pct), avg_pd=ys["avg_pd"].map(pct),
                             avg_score=ys["avg_score"].round(0), auc=ys["auc"].round(4), auc_gb=ys["auc_gb"].round(4)))

# ============================================================================
# PAGE: SCORECARD & FEATURES
# ============================================================================
def render_scorecard(ctx):
    sc, pts = ctx["model"], ctx["points"]
    st.header("🧮 Scorecard & Feature Analysis")
    st.caption("How each factor contributes to the credit score — fully transparent and auditable")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Base score", C.BASE_SCORE)
    sc2.metric("Base odds (good:bad)", f"{C.BASE_ODDS}:1")
    sc3.metric("Points to double odds", C.PDO)
    sc4.metric("Features in model", len(sc.features))

    tabs = st.tabs(["📊 Information Value", "⚙️ Coefficients", "🔍 Feature explorer", "📄 Full scorecard"])

    with tabs[0]:
        iv = pd.DataFrame({"feature": sc.binner.iv_.index, "iv": sc.binner.iv_.values})
        iv["strength"] = iv["iv"].map(iv_strength)
        iv["name"] = iv["feature"].map(label)
        iv["in_model"] = iv["feature"].isin(sc.features).map({True: "Selected", False: "Dropped"})
        fig = px.bar(iv.sort_values("iv"), x="iv", y="name", orientation="h", color="strength",
                     color_discrete_map={"Strong": GREEN, "Medium": BLUE, "Weak": AMBER, "Not useful": GREY},
                     title="Information Value by feature (predictive power)")
        fig.add_vline(x=C.IV_THRESHOLD, line_dash="dash", line_color=RED)
        fig.update_layout(yaxis_title="", xaxis_title="Information Value", height=560)
        show_chart(fig, key="sc_iv")
        st.caption("Rule of thumb: IV < 0.02 not useful · 0.02–0.1 weak · 0.1–0.3 medium · > 0.3 strong. "
                   "Red line = selection threshold.")
        show_df(iv[["name", "iv", "strength", "in_model"]].assign(iv=iv["iv"].round(4)))

    with tabs[1]:
        co = sc.coef_.rename("coef").reset_index()
        co.columns = ["feature", "coef"]
        co["name"] = co["feature"].map(label)
        fig = px.bar(co.sort_values("coef"), x="coef", y="name", orientation="h",
                     title="Logistic regression coefficients (on WOE values)", color_discrete_sequence=[BLUE])
        fig.update_layout(yaxis_title="", xaxis_title="Coefficient")
        show_chart(fig, key="sc_coef")
        st.caption(f"Intercept: {sc.model.intercept_[0]:.4f}. Positive coefficients are expected because "
                   "WOE = ln(%bad / %good): a riskier bin has higher WOE, which raises the default probability.")
        if (co["coef"] < 0).any():
            st.warning("Some coefficients are negative (" + ", ".join(co.loc[co["coef"] < 0, "name"]) +
                       "), which can indicate multicollinearity. Consider dropping one of the correlated features.")

    with tabs[2]:
        feat = st.selectbox("Feature", list(sc.features), format_func=label)
        t = pts[pts["feature"] == feat].copy()
        if feat in C.NUMERIC_FEATURES:
            t = t.assign(_k=t["bin_id"].astype(float)).sort_values("_k").drop(columns="_k")
        else:
            t = t.sort_values("woe")
        c1, c2 = st.columns(2)
        with c1:
            show_chart(volume_rate_chart(t.assign(default_rate=t["bad_rate"]), "bin",
                                               f"{label(feat)}: volume and default rate", count_col="count"),
                          key="sc_feat_rate")
        with c2:
            fig = go.Figure(go.Bar(x=t["bin"].astype(str), y=t["points"],
                                   marker_color=[GREEN if v >= t["points"].mean() else RED for v in t["points"]],
                                   text=t["points"].round(1), textposition="outside"))
            fig.update_layout(title=f"Points awarded per bin — {label(feat)}", yaxis_title="Points")
            fig.update_xaxes(type="category")
            show_chart(fig, key="sc_feat_pts")
        show_df(t[["bin", "count", "bad_rate", "woe", "points"]].assign(
            bad_rate=t["bad_rate"].map(pct), woe=t["woe"].round(3)))
        st.caption(f"Information Value: {sc.binner.iv_[feat]:.3f} ({iv_strength(sc.binner.iv_[feat])})")

    with tabs[3]:
        full = pts[["feature", "bin", "count", "bad_rate", "woe", "points"]].copy()
        full["feature"] = full["feature"].map(label)
        show_df(full.assign(bad_rate=full["bad_rate"].map(pct), woe=full["woe"].round(3)), height=520)
        st.download_button("⬇️ Download scorecard (CSV)", pts.to_csv(index=False).encode(),
                           "scorecard_table.csv", "text/csv")

# ============================================================================
# PAGE: CUT-OFF & STRATEGY
# ============================================================================
def render_strategy(ctx):
    df = ctx["df"]
    d = df[df["set"] == "test"]
    st.header("💰 Cut-off & Business Strategy")
    st.caption("Choose where to draw the approval line. Uses the out-of-time test sample. "
               "Profit figures are illustrative — set the assumptions to match your product.")

    a1, a2 = st.columns(2)
    margin = a1.slider("Net margin on a repaid loan (% of amount)", 2, 30, 12) / 100
    lgd = a2.slider("Loss given default (% of amount)", 20, 100, 60) / 100
    tbl = cutoff_table(d, margin, lgd, step=2)
    opt = int(tbl.loc[tbl["net_profit"].idxmax(), "cutoff"])
    lo, hi = int(tbl["cutoff"].min()), int(tbl["cutoff"].max())
    cut = st.slider("Approval cut-off score", lo, hi, opt)

    row = tbl.loc[(tbl["cutoff"] - cut).abs().idxmin()]
    base_bad, base_roe = d["default"].mean(), profit_vector(d, margin, lgd).sum() / d["loan_amnt"].sum()
    kpi_row([
        ("Approval rate", pct(row["approval_rate"])),
        ("Bad rate of approved", pct(row["bad_rate"]), f"{(row['bad_rate'] - base_bad) * 100:+.1f} pp vs approve-all", True),
        ("Defaults avoided", pct(row["bads_avoided_pct"])),
        ("Return on exposure", pct(row["return_on_exposure"]), f"{(row['return_on_exposure'] - base_roe) * 100:+.1f} pp vs approve-all"),
        ("Profit-max cut-off", f"{opt}"),
    ])
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        fig = volume_rate_chart(tbl.assign(loans=tbl["approval_rate"] * len(d)), "cutoff",
                                   "Approval rate vs bad rate by cut-off", rate_cols=("bad_rate",),
                                   names={"bad_rate": "Bad rate of approved"}, count_col="loans")
        fig.update_xaxes(type="linear")
        fig.add_vline(x=cut, line_dash="dash", line_color=GREY)
        show_chart(fig, key="st_curve")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=tbl["cutoff"], y=tbl["net_profit"] / 1e6, name="Net profit", line=dict(color=GREEN))
        fig.add_scatter(x=[opt], y=[tbl.loc[tbl["cutoff"] == opt, "net_profit"].iloc[0] / 1e6], mode="markers",
                        marker=dict(size=12, color=RED), name="Optimum")
        fig.add_vline(x=cut, line_dash="dash", line_color=GREY)
        fig.update_layout(title="Net profit by cut-off (sample, $M)", xaxis_title="Cut-off score",
                          yaxis_title="Net profit ($M)")
        show_chart(fig, key="st_profit")

    st.subheader("Three-zone decision policy")
    z1, z2 = st.columns(2)
    approve_at = z1.slider("Auto-approve at or above", lo, hi, min(hi, cut + 15))
    decline_below = z2.slider("Auto-decline below", lo, hi, max(lo, cut - 15))
    if decline_below > approve_at:
        st.warning("The decline threshold must not exceed the approve threshold.")
    else:
        s = d["score"].to_numpy()
        zone = np.where(s >= approve_at, "Auto-approve", np.where(s < decline_below, "Auto-decline", "Manual review"))
        z = d.assign(zone=zone, profit=profit_vector(d, margin, lgd))
        g = z.groupby("zone").agg(loans=("default", "size"), bad_rate=("default", "mean"),
                                  exposure=("loan_amnt", "sum"), profit=("profit", "sum")).reindex(
            ["Auto-approve", "Manual review", "Auto-decline"]).fillna(0).reset_index()
        g["share"] = g["loans"] / len(z)
        g["return_on_exposure"] = np.where(g["exposure"] > 0, g["profit"] / g["exposure"].replace(0, np.nan), 0)
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = px.pie(g, names="zone", values="loans", hole=0.5, title="Applicant share by zone",
                         color="zone", color_discrete_map={"Auto-approve": GREEN, "Manual review": AMBER,
                                                           "Auto-decline": RED})
            show_chart(fig, key="st_pie")
        with c2:
            show_df(g.assign(share=g["share"].map(pct), bad_rate=g["bad_rate"].map(pct),
                                exposure=g["exposure"].map(money), profit=g["profit"].map(money),
                                return_on_exposure=g["return_on_exposure"].map(pct)))

    with st.expander("Full cut-off table"):
        show_df(tbl.assign(approval_rate=tbl["approval_rate"].map(pct), bad_rate=tbl["bad_rate"].map(pct),
                              bads_avoided_pct=tbl["bads_avoided_pct"].map(pct),
                              return_on_exposure=tbl["return_on_exposure"].map(pct),
                              exposure=tbl["exposure"].map(money), net_profit=tbl["net_profit"].map(money)),
                   height=400)
        st.download_button("⬇️ Download cut-off table", tbl.to_csv(index=False).encode(), "cutoff_table.csv", "text/csv")

# ============================================================================
# PAGE: MONITORING & DRIFT
# ============================================================================
COLORS = {"Stable": GREEN, "Moderate shift": AMBER, "Significant shift": RED, "n/a": GREY}


def render_monitoring(ctx):
    df, m, ys = ctx["df"], ctx["metrics"], ctx["yearly"]
    st.header("🛰️ Model Monitoring & Drift")
    st.caption("Population Stability Index (PSI): < 0.10 stable · 0.10–0.25 moderate shift · > 0.25 significant shift")
    train, test = df[df["set"] == "train"], df[df["set"] == "test"]

    score_psi = psi_numeric(train["score"], test["score"])
    pdiff = test["default"].mean() - test["pd_hat"].mean()
    ft = feature_psi_table(train, test, C.NUMERIC_FEATURES, C.CATEGORICAL_FEATURES)
    kpi_row([("Score PSI (train→test)", f"{score_psi:.3f}", psi_status(score_psi)),
                ("Features drifting", f"{(ft['psi'] >= 0.1).sum()} / {len(ft)}"),
                ("Actual − predicted default", f"{pdiff * 100:+.1f} pp"),
                ("Default rate train → test", f"{train['default'].mean():.1%} → {test['default'].mean():.1%}")])
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="score", color="set", barmode="overlay", nbins=60, opacity=0.6,
                           histnorm="probability density", title="Score distribution: train vs test",
                           color_discrete_map={"train": BLUE, "test": AMBER})
        show_chart(fig, key="mo_score")
    with c2:
        fig = px.bar(ft.sort_values("psi"), x="psi", y=ft.sort_values("psi")["feature"].map(label),
                     orientation="h", color="status", color_discrete_map=COLORS, title="Feature PSI (train vs test)")
        fig.add_vline(x=0.1, line_dash="dash", line_color=AMBER)
        fig.add_vline(x=0.25, line_dash="dash", line_color=RED)
        fig.update_layout(yaxis_title="", xaxis_title="PSI", height=520)
        show_chart(fig, key="mo_feat")

    st.subheader("Compare a feature's distribution")
    feat = st.selectbox("Feature", list(ft["feature"]), format_func=label)
    if feat in C.NUMERIC_FEATURES:
        fig = px.histogram(df.dropna(subset=[feat]), x=feat, color="set", barmode="overlay", nbins=50, opacity=0.6,
                           histnorm="probability density", color_discrete_map={"train": BLUE, "test": AMBER},
                           title=f"{label(feat)}: train vs test")
    else:
        prop = df.assign(v=df[feat].fillna("Missing")).groupby(["set", "v"]).size().reset_index(name="n")
        prop["share"] = prop["n"] / prop.groupby("set")["n"].transform("sum")
        fig = px.bar(prop, x="v", y="share", color="set", barmode="group", title=f"{label(feat)}: train vs test",
                     color_discrete_map={"train": BLUE, "test": AMBER})
        fig.update_layout(yaxis_tickformat=".0%")
    show_chart(fig, key="mo_featdist")

    st.subheader("Score drift by issue year (vs training population)")
    rows = [{"year": int(y), "psi": psi_numeric(train["score"], g["score"]), "loans": len(g)}
            for y, g in df.groupby("year")]
    sy = pd.DataFrame(rows)
    sy["status"] = sy["psi"].map(psi_status)
    fig = px.bar(sy, x="year", y="psi", color="status", color_discrete_map=COLORS, title="Score PSI by issue year")
    fig.update_xaxes(dtick=1)
    show_chart(fig, key="mo_year")

    if pdiff > 0.02:
        st.warning(f"**Recommendation:** actual defaults exceed predicted PD by {pdiff * 100:.1f} pp in the test period. "
                   "Recalibrate PD (e.g. Platt/isotonic on recent data) or retrain on more recent vintages. "
                   "Monitor monthly and trigger a review if score PSI > 0.25.")
    else:
        st.success("No material calibration drift detected between train and test periods.")
    show_df(ft.assign(psi=ft["psi"].round(4), feature=ft["feature"].map(label)))

# ============================================================================
# PAGE: APPLICANT SCORING
# ============================================================================
TIERS = [(0.2, "Very high risk", RED), (0.4, "High risk", "#f97316"), (0.6, "Medium risk", AMBER),
         (0.8, "Low risk", "#84cc16"), (1.01, "Very low risk", GREEN)]


def _tier(pctile):
    return next((n, c) for q, n, c in TIERS if pctile < q)


def _gauge(score, df):
    q = df["score"].quantile([0, .2, .4, .6, .8, 1]).to_list()
    steps = [{"range": [q[i], q[i + 1]], "color": c} for i, (_, _, c) in enumerate(TIERS)]
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, title={"text": "Credit score"},
                                 gauge={"axis": {"range": [q[0], q[-1]]}, "bar": {"color": "#0f172a"}, "steps": steps}))
    fig.update_layout(height=300)
    return fig


def _single(ctx):
    sc, df, pts = ctx["model"], ctx["df"], ctx["points"]
    med = df[C.NUMERIC_FEATURES].median()
    default_cut = ctx["policy_cutoff"]
    with st.form("applicant"):
        st.subheader("Applicant & loan details")
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, int(round(med["loan_amnt"], -2)), 500)
            term = st.selectbox("Term (months)", [36, 60])
            annual_inc = st.number_input("Annual income ($)", 10000, 1000000, int(round(med["annual_inc"], -3)), 1000)
            dti = st.number_input("Debt-to-income (%)", 0.0, 60.0, float(round(med["dti"], 1)))
            fico = st.slider("FICO score", 600, 850, int(med["fico_avg"]))
            emp = st.slider("Employment length (years)", 0, 10, int(med["emp_length_yrs"]))
        with c2:
            hist = st.slider("Credit history (years)", 0, 50, int(med["credit_history_yrs"]))
            home = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])
            verif = st.selectbox("Income verification", ["Verified", "Source Verified", "Not Verified"])
            purpose = st.selectbox("Purpose", sorted(df["purpose"].dropna().unique()))
            revol_bal = st.number_input("Revolving balance ($)", 0, 500000, int(round(med["revol_bal"], -2)), 500)
            revol_util = st.slider("Revolving utilisation (%)", 0, 150, int(med["revol_util"]))
        with c3:
            delinq = st.number_input("Delinquencies (last 2 yrs)", 0, 20, 0)
            inq = st.number_input("Credit inquiries (last 6 mo)", 0, 10, int(med["inq_last_6mths"]))
            open_acc = st.number_input("Open accounts", 0, 60, int(med["open_acc"]))
            total_acc = st.number_input("Total accounts", 0, 120, int(med["total_acc"]))
            pub_rec = st.number_input("Public records", 0, 10, 0)
            bankr = st.number_input("Public bankruptcies", 0, 5, 0)
            mort = st.number_input("Mortgage accounts", 0, 10, int(med["mort_acc"]))
            cutoff = st.number_input("Approval cut-off score", 300, 850, int(default_cut))
        go_ = st.form_submit_button("Calculate score", type="primary")

    if not go_:
        st.info("Fill in the form and press **Calculate score**. Default cut-off = profit-maximising cut-off from the Strategy page.")
        return
    row = pd.DataFrame([{
        "loan_amnt": loan_amnt, "term_months": term, "annual_inc": annual_inc, "dti": dti, "delinq_2yrs": delinq,
        "fico_avg": fico, "inq_last_6mths": inq, "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal,
        "revol_util": revol_util, "total_acc": total_acc, "mort_acc": mort, "pub_rec_bankruptcies": bankr,
        "emp_length_yrs": emp, "credit_history_yrs": hist, "loan_to_income": loan_amnt / annual_inc,
        "home_ownership": home, "verification_status": verif, "purpose": purpose}])[C.FEATURES]
    score, p = int(sc.score(row)[0]), float(sc.pd_(row)[0])
    pctile = float((df["score"] <= score).mean())
    tier, colour = _tier(pctile)
    approved = score >= cutoff

    c1, c2 = st.columns([2, 3])
    with c1:
        show_chart(_gauge(score, df), key="sg_gauge")
    with c2:
        kpi_row([("Score", score), ("Probability of default", pct(p)),
                    ("Better than", f"{pctile:.0%} of applicants")])
        st.markdown(f"<span class='tag' style='background:{colour};color:white'>{tier}</span> &nbsp; "
                    f"<span class='tag' style='background:{'#16a34a' if approved else '#dc2626'};color:white'>"
                    f"{'✅ APPROVE' if approved else '❌ DECLINE'}</span>", unsafe_allow_html=True)
        st.caption(f"Cut-off {int(cutoff)} · score is {abs(score - cutoff)} points {'above' if approved else 'below'} the line")

    expl = sc.explain(row)
    best = pts.groupby("feature")["points"].max()
    loss = (best.reindex(expl.index) - expl).sort_values(ascending=False)
    fig = px.bar(loss.sort_values().rename("points_lost").reset_index().rename(columns={"index": "feature"}),
                 x="points_lost", y=loss.sort_values().index.map(label), orientation="h",
                 title="Points lost vs best possible answer, by factor", color_discrete_sequence=[RED])
    fig.update_layout(yaxis_title="", xaxis_title="Points lost", height=420)
    show_chart(fig, key="sg_expl")
    st.subheader("Main reasons for this score (adverse-action reason codes)")
    for f_, v in loss.head(3).items():
        st.markdown(f"- **{label(f_)}** — cost {v:.0f} points compared with the best category")


def _batch(ctx):
    sc, df = ctx["model"], ctx["df"]
    st.subheader("Batch scoring")
    st.write("Upload a CSV with these columns (`loan_to_income` is optional):")
    st.code(", ".join(C.FEATURES), language="text")
    st.download_button("⬇️ Download template (5 example rows)", df[C.FEATURES].head(5).to_csv(index=False).encode(),
                       "scoring_template.csv", "text/csv")
    up = st.file_uploader("Upload applicants CSV", type="csv")
    if up is None:
        return
    data = pd.read_csv(up)
    if "loan_to_income" not in data and {"loan_amnt", "annual_inc"} <= set(data.columns):
        data["loan_to_income"] = data["loan_amnt"] / data["annual_inc"].replace(0, np.nan)
    missing = [c for c in C.FEATURES if c not in data]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return
    out = data.copy()
    out["score"] = sc.score(data[C.FEATURES]).astype(int)
    out["pd"] = sc.pd_(data[C.FEATURES]).round(4)
    cut = st.number_input("Approval cut-off score", 300, 850, int(ctx["policy_cutoff"]), key="batch_cut")
    out["decision"] = np.where(out["score"] >= cut, "APPROVE", "DECLINE")
    kpi_row([("Applicants", f"{len(out):,}"), ("Approved", pct((out['decision'] == 'APPROVE').mean())),
                ("Avg score", f"{out['score'].mean():.0f}"), ("Avg PD", pct(out["pd"].mean()))])
    fig = px.histogram(out, x="score", color="decision", nbins=40, title="Score distribution",
                       color_discrete_map={"APPROVE": GREEN, "DECLINE": RED})
    show_chart(fig, key="sg_batch_hist")
    show_df(out.head(200), height=360)
    st.download_button("⬇️ Download scored file", out.to_csv(index=False).encode(), "scored_applicants.csv", "text/csv")


def render_scoring(ctx):
    st.header("🧑‍💼 Applicant Scoring")
    t1, t2 = st.tabs(["Single applicant", "Batch (CSV upload)"])
    with t1:
        _single(ctx)
    with t2:
        _batch(ctx)

# ============================================================================
# MAIN
# ============================================================================
SAMPLE = C.REPORT_DIR / "scored_sample.csv.gz"
PAGES = {
    "🏠 Executive Overview": render_overview,
    "📊 Portfolio BI Dashboard": render_bi,
    "🎯 Model Performance": render_performance,
    "🧮 Scorecard & Features": render_scorecard,
    "💰 Cut-off & Strategy": render_strategy,
    "🛰️ Monitoring & Drift": render_monitoring,
    "🧑‍💼 Applicant Scoring": render_scoring,
}


@st.cache_resource(show_spinner="Loading model and data…")
def load_context():
    model = joblib.load(C.MODEL_PATH)
    df = pd.read_csv(SAMPLE, parse_dates=["issue_d"])
    df["year"] = df["issue_d"].dt.year
    return {
        "model": model, "df": df, "features": C.FEATURES, "points": model.points_table(),
        "metrics": json.loads((C.REPORT_DIR / "metrics.json").read_text()),
        "yearly": pd.read_csv(C.REPORT_DIR / "yearly_summary.csv"),
        "policy_cutoff": best_cutoff(df[df["set"] == "test"]),
    }


def main():
    inject_css()
    try:
        ctx = load_context()
    except FileNotFoundError:
        st.error("Model or dashboard data not found. Run these commands first, then refresh:")
        st.code("python -m src.data_prep\npython -m src.train", language="bash")
        st.stop()

    with st.sidebar:
        st.title("💳 Credit Scoring")
        page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
        st.divider()
        m = ctx["metrics"]
        st.caption("**Model:** WOE + logistic regression scorecard")
        st.caption(f"**Data:** LendingClub {C.START_YEAR}–{C.END_YEAR} · {m['n_loans']:,} loans")
        st.caption(f"**Split:** train ≤ {m['train_end_year']} · test after")
        st.caption(f"**Test AUC:** {m['scorecard_test']['auc']:.3f} · **KS:** {m['scorecard_test']['ks']:.3f}")
    PAGES[page](ctx)


main()
