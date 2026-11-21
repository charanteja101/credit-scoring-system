import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src import config as C
from . import common as cm


def render(ctx):
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
        iv["strength"] = iv["iv"].map(cm.iv_strength)
        iv["name"] = iv["feature"].map(cm.label)
        iv["in_model"] = iv["feature"].isin(sc.features).map({True: "Selected", False: "Dropped"})
        fig = px.bar(iv.sort_values("iv"), x="iv", y="name", orientation="h", color="strength",
                     color_discrete_map={"Strong": cm.GREEN, "Medium": cm.BLUE, "Weak": cm.AMBER, "Not useful": cm.GREY},
                     title="Information Value by feature (predictive power)")
        fig.add_vline(x=C.IV_THRESHOLD, line_dash="dash", line_color=cm.RED)
        fig.update_layout(yaxis_title="", xaxis_title="Information Value", height=560)
        cm.show_chart(fig, key="sc_iv")
        st.caption("Rule of thumb: IV < 0.02 not useful · 0.02–0.1 weak · 0.1–0.3 medium · > 0.3 strong. "
                   "Red line = selection threshold.")
        cm.show_df(iv[["name", "iv", "strength", "in_model"]].assign(iv=iv["iv"].round(4)))

    with tabs[1]:
        co = sc.coef_.rename("coef").reset_index()
        co.columns = ["feature", "coef"]
        co["name"] = co["feature"].map(cm.label)
        fig = px.bar(co.sort_values("coef"), x="coef", y="name", orientation="h",
                     title="Logistic regression coefficients (on WOE values)", color_discrete_sequence=[cm.BLUE])
        fig.update_layout(yaxis_title="", xaxis_title="Coefficient")
        cm.show_chart(fig, key="sc_coef")
        st.caption(f"Intercept: {sc.model.intercept_[0]:.4f}. Positive coefficients are expected because "
                   "WOE = ln(%bad / %good): a riskier bin has higher WOE, which raises the default probability.")
        if (co["coef"] < 0).any():
            st.warning("Some coefficients are negative (" + ", ".join(co.loc[co["coef"] < 0, "name"]) +
                       "), which can indicate multicollinearity. Consider dropping one of the correlated features.")

    with tabs[2]:
        feat = st.selectbox("Feature", list(sc.features), format_func=cm.label)
        t = pts[pts["feature"] == feat].copy()
        if feat in C.NUMERIC_FEATURES:
            t = t.assign(_k=t["bin_id"].astype(float)).sort_values("_k").drop(columns="_k")
        else:
            t = t.sort_values("woe")
        c1, c2 = st.columns(2)
        with c1:
            cm.show_chart(cm.volume_rate_chart(t.assign(default_rate=t["bad_rate"]), "bin",
                                               f"{cm.label(feat)}: volume and default rate", count_col="count"),
                          key="sc_feat_rate")
        with c2:
            fig = go.Figure(go.Bar(x=t["bin"].astype(str), y=t["points"],
                                   marker_color=[cm.GREEN if v >= t["points"].mean() else cm.RED for v in t["points"]],
                                   text=t["points"].round(1), textposition="outside"))
            fig.update_layout(title=f"Points awarded per bin — {cm.label(feat)}", yaxis_title="Points")
            fig.update_xaxes(type="category")
            cm.show_chart(fig, key="sc_feat_pts")
        cm.show_df(t[["bin", "count", "bad_rate", "woe", "points"]].assign(
            bad_rate=t["bad_rate"].map(cm.pct), woe=t["woe"].round(3)))
        st.caption(f"Information Value: {sc.binner.iv_[feat]:.3f} ({cm.iv_strength(sc.binner.iv_[feat])})")

    with tabs[3]:
        full = pts[["feature", "bin", "count", "bad_rate", "woe", "points"]].copy()
        full["feature"] = full["feature"].map(cm.label)
        cm.show_df(full.assign(bad_rate=full["bad_rate"].map(cm.pct), woe=full["woe"].round(3)), height=520)
        st.download_button("⬇️ Download scorecard (CSV)", pts.to_csv(index=False).encode(),
                           "scorecard_table.csv", "text/csv")
