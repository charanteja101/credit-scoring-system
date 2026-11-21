import plotly.express as px
import streamlit as st
from src import analytics as an
from . import common as cm


def render(ctx):
    m, yearly, df = ctx["metrics"], ctx["yearly"], ctx["df"]
    te, gb = m["scorecard_test"], m["gradient_boosting_test"]
    st.title("💳 Credit Scoring System")
    st.caption("LendingClub loans issued 2012–2016 · WOE logistic-regression scorecard · out-of-time validation")

    cm.kpi_row([
        ("Loans analysed", f"{m['n_loans']:,}"),
        ("Overall default rate", cm.pct(m["default_rate"])),
        ("Test AUC", f"{te['auc']:.3f}", f"{te['auc'] - gb['auc']:+.3f} vs GBM"),
        ("Test Gini", f"{te['gini']:.3f}"),
        ("Test KS", f"{te['ks']:.3f}"),
        ("Score PSI", f"{m['score_psi_train_vs_test']:.3f}", an.psi_status(m["score_psi_train_vs_test"])),
    ])
    st.write("")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = cm.volume_rate_chart(yearly, "year", "Loan volume, actual default rate and predicted PD by year",
                                   rate_cols=("default_rate", "avg_pd"),
                                   names={"default_rate": "Actual default rate", "avg_pd": "Avg predicted PD"})
        cm.show_chart(fig, key="ov_year")
    with c2:
        test = df[df["set"] == "test"].copy()
        test["outcome"] = test["default"].map({0: "Paid", 1: "Defaulted"})
        fig = px.histogram(test, x="score", color="outcome", barmode="overlay", nbins=50, opacity=0.65,
                           color_discrete_map={"Paid": cm.GREEN, "Defaulted": cm.RED},
                           title="Score distribution by outcome (test)")
        cm.show_chart(fig, key="ov_dist")

    bands = an.quantile_band_table(test, "score", 10)
    c3, c4 = st.columns([3, 2])
    with c3:
        fig = px.bar(bands, x="band", y="default_rate", text=bands["default_rate"].map(lambda v: f"{v:.0%}"),
                     title="Default rate by score decile (test) — lower score = higher risk",
                     color="default_rate", color_continuous_scale="RdYlGn_r")
        fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
        cm.show_chart(fig, key="ov_bands")
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
- **Stability:** train/test score PSI = {m['score_psi_train_vs_test']:.3f} ({an.psi_status(m['score_psi_train_vs_test']).lower()}).
""")

    with st.expander("ℹ️ About this project"):
        sc = m["scaling"]
        st.markdown(f"""
**Pipeline:** raw Kaggle data → resolved loans 2012–2016 → feature engineering → WOE binning & Information Value →
logistic regression → points scaling ({sc['base_score']} pts at {sc['base_odds']}:1 odds, +{sc['pdo']} pts doubles odds) → evaluation.
Trained on loans issued ≤ {m['train_end_year']} ({m['n_train']:,}), tested on later loans ({m['n_test']:,}).
Dashboards use a random sample of {len(df):,} scored loans; headline metrics use the full data.
""")
