import pandas as pd
import plotly.express as px
import streamlit as st
from src import analytics as an
from src import config as C
from . import common as cm

COLORS = {"Stable": cm.GREEN, "Moderate shift": cm.AMBER, "Significant shift": cm.RED, "n/a": cm.GREY}


def render(ctx):
    df, m, ys = ctx["df"], ctx["metrics"], ctx["yearly"]
    st.header("🛰️ Model Monitoring & Drift")
    st.caption("Population Stability Index (PSI): < 0.10 stable · 0.10–0.25 moderate shift · > 0.25 significant shift")
    train, test = df[df["set"] == "train"], df[df["set"] == "test"]

    score_psi = an.psi_numeric(train["score"], test["score"])
    pdiff = test["default"].mean() - test["pd_hat"].mean()
    ft = an.feature_psi_table(train, test, C.NUMERIC_FEATURES, C.CATEGORICAL_FEATURES)
    cm.kpi_row([("Score PSI (train→test)", f"{score_psi:.3f}", an.psi_status(score_psi)),
                ("Features drifting", f"{(ft['psi'] >= 0.1).sum()} / {len(ft)}"),
                ("Actual − predicted default", f"{pdiff * 100:+.1f} pp"),
                ("Default rate train → test", f"{train['default'].mean():.1%} → {test['default'].mean():.1%}")])
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="score", color="set", barmode="overlay", nbins=60, opacity=0.6,
                           histnorm="probability density", title="Score distribution: train vs test",
                           color_discrete_map={"train": cm.BLUE, "test": cm.AMBER})
        cm.show_chart(fig, key="mo_score")
    with c2:
        fig = px.bar(ft.sort_values("psi"), x="psi", y=ft.sort_values("psi")["feature"].map(cm.label),
                     orientation="h", color="status", color_discrete_map=COLORS, title="Feature PSI (train vs test)")
        fig.add_vline(x=0.1, line_dash="dash", line_color=cm.AMBER)
        fig.add_vline(x=0.25, line_dash="dash", line_color=cm.RED)
        fig.update_layout(yaxis_title="", xaxis_title="PSI", height=520)
        cm.show_chart(fig, key="mo_feat")

    st.subheader("Compare a feature's distribution")
    feat = st.selectbox("Feature", list(ft["feature"]), format_func=cm.label)
    if feat in C.NUMERIC_FEATURES:
        fig = px.histogram(df.dropna(subset=[feat]), x=feat, color="set", barmode="overlay", nbins=50, opacity=0.6,
                           histnorm="probability density", color_discrete_map={"train": cm.BLUE, "test": cm.AMBER},
                           title=f"{cm.label(feat)}: train vs test")
    else:
        prop = df.assign(v=df[feat].fillna("Missing")).groupby(["set", "v"]).size().reset_index(name="n")
        prop["share"] = prop["n"] / prop.groupby("set")["n"].transform("sum")
        fig = px.bar(prop, x="v", y="share", color="set", barmode="group", title=f"{cm.label(feat)}: train vs test",
                     color_discrete_map={"train": cm.BLUE, "test": cm.AMBER})
        fig.update_layout(yaxis_tickformat=".0%")
    cm.show_chart(fig, key="mo_featdist")

    st.subheader("Score drift by issue year (vs training population)")
    rows = [{"year": int(y), "psi": an.psi_numeric(train["score"], g["score"]), "loans": len(g)}
            for y, g in df.groupby("year")]
    sy = pd.DataFrame(rows)
    sy["status"] = sy["psi"].map(an.psi_status)
    fig = px.bar(sy, x="year", y="psi", color="status", color_discrete_map=COLORS, title="Score PSI by issue year")
    fig.update_xaxes(dtick=1)
    cm.show_chart(fig, key="mo_year")

    if pdiff > 0.02:
        st.warning(f"**Recommendation:** actual defaults exceed predicted PD by {pdiff * 100:.1f} pp in the test period. "
                   "Recalibrate PD (e.g. Platt/isotonic on recent data) or retrain on more recent vintages. "
                   "Monitor monthly and trigger a review if score PSI > 0.25.")
    else:
        st.success("No material calibration drift detected between train and test periods.")
    cm.show_df(ft.assign(psi=ft["psi"].round(4), feature=ft["feature"].map(cm.label)))
