import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src import analytics as an
from . import common as cm


def _subset(df, choice):
    if choice.startswith("Test"):
        return df[df["set"] == "test"]
    if choice.startswith("Train"):
        return df[df["set"] == "train"]
    return df


def render(ctx):
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
        cm.kpi_row([("Scorecard AUC (test)", f"{te['auc']:.3f}", f"{te['auc'] - gb['auc']:+.3f} vs GBM"),
                    ("Scorecard Gini (test)", f"{te['gini']:.3f}", f"{te['gini'] - gb['gini']:+.3f} vs GBM"),
                    ("Scorecard KS (test)", f"{te['ks']:.3f}", f"{te['ks'] - gb['ks']:+.3f} vs GBM"),
                    ("Overfit gap (AUC)", f"{m['scorecard_train']['auc'] - te['auc']:+.3f}", "train − test")])
        long = tbl.melt(id_vars=["Model", "Set"], var_name="Metric", value_name="Value")
        long["Series"] = long["Model"] + " – " + long["Set"]
        fig = px.bar(long, x="Metric", y="Value", color="Series", barmode="group", text_auto=".3f",
                     title="Model comparison (full data)")
        cm.show_chart(fig, key="pf_cmp")
        cm.show_df(tbl)
        st.info("Industry guide for application scorecards: AUC 0.65–0.75 · KS 0.25–0.40 is typical. "
                "Scorecard test AUC above train AUC means the model is not overfitting.")

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for nm, pp, col in [("Scorecard", p, cm.BLUE), ("Gradient boosting", pg, cm.AMBER)]:
                r = an.roc_df(y, pp)
                fig.add_scatter(x=r["fpr"], y=r["tpr"], name=f"{nm} (AUC {an.auc_safe(y, pp):.3f})",
                                line=dict(color=col))
            fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=cm.GREY, dash="dash"))
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
            cm.show_chart(fig, key="pf_roc")
        with c2:
            curve, ks, at = an.ks_df(y, p)
            fig = go.Figure()
            fig.add_scatter(x=curve["pct_population"], y=curve["cum_bad"], name="Cumulative bad", line=dict(color=cm.RED))
            fig.add_scatter(x=curve["pct_population"], y=curve["cum_good"], name="Cumulative good", line=dict(color=cm.GREEN))
            fig.add_vline(x=at, line_dash="dash", line_color=cm.GREY)
            fig.update_layout(title=f"KS plot — KS = {ks:.3f} at {at:.0%} of population (sorted risky → safe)",
                              xaxis_title="% of applicants (riskiest first)", yaxis_title="Cumulative share",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            cm.show_chart(fig, key="pf_ks")

    with tabs[2]:
        lt = an.lift_table(y, p)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(lt, x="decile", y="lift", text=lt["lift"].map(lambda v: f"{v:.2f}x"),
                         title="Lift by risk decile (1 = riskiest 10%)")
            fig.add_hline(y=1, line_dash="dash", line_color=cm.GREY)
            fig.update_xaxes(type="category")
            cm.show_chart(fig, key="pf_lift")
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=[0] + list(lt["cum_pop_pct"]), y=[0] + list(lt["cum_bads_pct"]), mode="lines+markers",
                            name="Scorecard", line=dict(color=cm.BLUE))
            fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=cm.GREY, dash="dash"))
            fig.update_layout(title="Cumulative gains: share of defaults captured",
                              xaxis_title="% of applicants (riskiest first)", yaxis_title="% of defaults captured",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            cm.show_chart(fig, key="pf_gain")
        top30 = lt.loc[lt["decile"] <= 3, "bads"].sum() / lt["bads"].sum()
        st.success(f"Rejecting the riskiest **30%** of applicants would catch **{top30:.0%}** of all defaults.")
        cm.show_df(lt.assign(bad_rate=lt["bad_rate"].map(cm.pct), avg_pd=lt["avg_pd"].map(cm.pct),
                             lift=lt["lift"].round(2), cum_bads_pct=lt["cum_bads_pct"].map(cm.pct),
                             cum_pop_pct=lt["cum_pop_pct"].map(cm.pct)))

    with tabs[3]:
        cal = an.calibration_table(y, p)
        fig = go.Figure()
        mx = float(max(cal["avg_pd"].max(), cal["actual"].max()) * 1.05)
        fig.add_scatter(x=[0, mx], y=[0, mx], name="Perfect calibration", line=dict(color=cm.GREY, dash="dash"))
        fig.add_scatter(x=cal["avg_pd"], y=cal["actual"], mode="lines+markers", name="Scorecard",
                        line=dict(color=cm.BLUE))
        fig.update_layout(title="Calibration: predicted PD vs actual default rate (deciles)",
                          xaxis_title="Average predicted PD", yaxis_title="Actual default rate",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        cm.show_chart(fig, key="pf_cal")
        gap = (y.mean() - p.mean()) * 100
        if abs(gap) > 2:
            st.warning(f"Average predicted PD is {p.mean():.1%} vs actual {y.mean():.1%} on this dataset "
                       f"({'under' if gap > 0 else 'over'}estimating risk by {abs(gap):.1f} pp). "
                       "Ranking power is unaffected, but PD should be recalibrated before use in pricing/provisioning.")
        else:
            st.success(f"Well calibrated: predicted {p.mean():.1%} vs actual {y.mean():.1%}.")

    with tabs[4]:
        thr = st.slider("Classify as 'bad' when predicted PD ≥", 0.05, 0.95, float(round(y.mean(), 2)), 0.01)
        c = an.confusion(y, p, thr)
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = px.imshow(np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]]), text_auto=True,
                            x=["Predicted good", "Predicted bad"], y=["Actual good", "Actual bad"],
                            color_continuous_scale="Blues", title="Confusion matrix")
            fig.update_layout(coloraxis_showscale=False)
            cm.show_chart(fig, key="pf_conf")
        with c2:
            cm.kpi_row([("Precision", cm.pct(c["precision"])), ("Recall", cm.pct(c["recall"])),
                        ("Specificity", cm.pct(c["specificity"])), ("F1", f"{c['f1']:.3f}")])
            sw = an.threshold_sweep(y, p, np.arange(0.05, 0.96, 0.025))
            fig = px.line(sw.melt(id_vars="threshold", var_name="metric"), x="threshold", y="value", color="metric",
                          title="Precision / recall trade-off by threshold")
            fig.add_vline(x=thr, line_dash="dash", line_color=cm.GREY)
            cm.show_chart(fig, key="pf_sweep")

    with tabs[5]:
        ys = ctx["yearly"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_scatter(x=ys["year"], y=ys["auc"], name="Scorecard", mode="lines+markers", line=dict(color=cm.BLUE))
            fig.add_scatter(x=ys["year"], y=ys["auc_gb"], name="Gradient boosting", mode="lines+markers",
                            line=dict(color=cm.AMBER))
            fig.add_vline(x=m["train_end_year"] + 0.5, line_dash="dash", line_color=cm.GREY)
            fig.update_layout(title="AUC by issue year (dashed line = train/test split)", xaxis_title="Issue year",
                              yaxis_title="AUC")
            fig.update_xaxes(dtick=1)
            cm.show_chart(fig, key="pf_auc_year")
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=ys["year"], y=ys["default_rate"], name="Actual default rate", mode="lines+markers",
                            line=dict(color=cm.RED))
            fig.add_scatter(x=ys["year"], y=ys["avg_pd"], name="Predicted PD", mode="lines+markers",
                            line=dict(color=cm.BLUE, dash="dash"))
            fig.update_layout(title="Actual vs predicted default rate by year", yaxis_tickformat=".0%",
                              xaxis_title="Issue year")
            fig.update_xaxes(dtick=1)
            cm.show_chart(fig, key="pf_pd_year")
        cm.show_df(ys.assign(year=ys["year"].astype(int), loans=ys["loans"].astype(int),
                             default_rate=ys["default_rate"].map(cm.pct), avg_pd=ys["avg_pd"].map(cm.pct),
                             avg_score=ys["avg_score"].round(0), auc=ys["auc"].round(4), auc_gb=ys["auc_gb"].round(4)))
