import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src import analytics as an
from . import common as cm

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


def render(ctx):
    df = ctx["df"]
    st.header("📊 Portfolio BI Dashboard")
    st.caption(f"Interactive analysis of a {len(df):,}-loan random sample of the 2012–2016 portfolio")
    f = _filters(df)
    if f.empty:
        st.warning("No loans match these filters.")
        return

    base = df["default"].mean()
    cm.kpi_row([
        ("Loans", f"{len(f):,}", f"{len(f) / len(df):.0%} of portfolio"),
        ("Default rate", cm.pct(f["default"].mean()), f"{(f['default'].mean() - base) * 100:+.1f} pp vs all", True),
        ("Avg loan", cm.money(f["loan_amnt"].mean())),
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
        fig = cm.volume_rate_chart(t, "month", "Monthly vintage: volume, actual default rate vs predicted PD",
                                   rate_cols=("default_rate", "avg_pd"),
                                   names={"default_rate": "Actual default rate", "avg_pd": "Avg predicted PD"})
        fig.update_xaxes(type="date")
        cm.show_chart(fig, key="bi_month")
        ty = f.groupby(["year", "term_months"]).agg(default_rate=("default", "mean")).reset_index()
        ty["term_months"] = ty["term_months"].astype(int).astype(str) + " months"
        fig = px.line(ty, x="year", y="default_rate", color="term_months", markers=True,
                      title="Default rate by issue year and term")
        fig.update_layout(yaxis_tickformat=".0%")
        fig.update_xaxes(dtick=1)
        cm.show_chart(fig, key="bi_term_year")

    with tabs[1]:
        dim = st.selectbox("Segment by", SEGMENTS, format_func=cm.label)
        seg = an.segment_table(f, dim)
        c1, c2 = st.columns([3, 2])
        with c1:
            cm.show_chart(cm.volume_rate_chart(seg, "segment", f"Default rate by {cm.label(dim).lower()}"),
                          key="bi_seg")
        with c2:
            show = seg.assign(default_rate=seg["default_rate"].map(cm.pct), avg_loan=seg["avg_loan"].round(0),
                              avg_score=seg["avg_score"].round(0)).drop(columns=["avg_pd"], errors="ignore")
            cm.show_df(show)

    with tabs[2]:
        names = list(BANDS)
        picks = st.multiselect("Risk drivers to show", names, default=names[:4])
        cols = st.columns(2)
        for i, nm in enumerate(picks):
            col, bins = BANDS[nm]
            t = an.band_table(f.dropna(subset=[col]), col, bins)
            with cols[i % 2]:
                cm.show_chart(cm.volume_rate_chart(t, "band", f"Default rate by {nm}"), key=f"bi_drv_{i}")

    with tabs[3]:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis", HEAT_VARS, index=0, format_func=cm.label)
        y = c2.selectbox("Y axis", HEAT_VARS, index=1, format_func=cm.label)
        q = c3.slider("Bins per axis", 3, 10, 6)
        if x == y:
            st.info("Choose two different variables.")
        else:
            rate = an.pivot_rate(f, x, y, q=q)
            fig = px.imshow(rate, text_auto=".1%", aspect="auto", color_continuous_scale="RdYlGn_r",
                            labels=dict(x=cm.label(x), y=cm.label(y), color="Default rate"),
                            title=f"Default rate: {cm.label(y)} × {cm.label(x)} (cells with <30 loans hidden)")
            cm.show_chart(fig, key="bi_heat", height=520)

    with tabs[4]:
        bands = an.quantile_band_table(f, "score", 10)
        cm.show_chart(cm.volume_rate_chart(bands, "band", "Score deciles: volume, actual default vs predicted PD",
                                           rate_cols=("default_rate", "avg_pd"),
                                           names={"default_rate": "Actual default rate", "avg_pd": "Avg predicted PD"}),
                      key="bi_bands")
        cm.show_df(bands.assign(default_rate=bands["default_rate"].map(cm.pct),
                                avg_pd=bands["avg_pd"].map(cm.pct), avg_score=bands["avg_score"].round(0)))

    with tabs[5]:
        cols = ["issue_d", "set", "default", "score", "pd_hat"] + [c for c in ctx["features"]]
        st.write(f"Showing the first 500 of {len(f):,} filtered loans")
        cm.show_df(f[cols].head(500), height=420)
        st.download_button("⬇️ Download filtered data (up to 50,000 rows)",
                           f[cols].head(50_000).to_csv(index=False).encode(), "filtered_loans.csv", "text/csv")
