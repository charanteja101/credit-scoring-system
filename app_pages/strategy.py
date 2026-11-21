import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src import analytics as an
from . import common as cm


def render(ctx):
    df = ctx["df"]
    d = df[df["set"] == "test"]
    st.header("💰 Cut-off & Business Strategy")
    st.caption("Choose where to draw the approval line. Uses the out-of-time test sample. "
               "Profit figures are illustrative — set the assumptions to match your product.")

    a1, a2 = st.columns(2)
    margin = a1.slider("Net margin on a repaid loan (% of amount)", 2, 30, 12) / 100
    lgd = a2.slider("Loss given default (% of amount)", 20, 100, 60) / 100
    tbl = an.cutoff_table(d, margin, lgd, step=2)
    opt = int(tbl.loc[tbl["net_profit"].idxmax(), "cutoff"])
    lo, hi = int(tbl["cutoff"].min()), int(tbl["cutoff"].max())
    cut = st.slider("Approval cut-off score", lo, hi, opt)

    row = tbl.loc[(tbl["cutoff"] - cut).abs().idxmin()]
    base_bad, base_roe = d["default"].mean(), an.profit_vector(d, margin, lgd).sum() / d["loan_amnt"].sum()
    cm.kpi_row([
        ("Approval rate", cm.pct(row["approval_rate"])),
        ("Bad rate of approved", cm.pct(row["bad_rate"]), f"{(row['bad_rate'] - base_bad) * 100:+.1f} pp vs approve-all", True),
        ("Defaults avoided", cm.pct(row["bads_avoided_pct"])),
        ("Return on exposure", cm.pct(row["return_on_exposure"]), f"{(row['return_on_exposure'] - base_roe) * 100:+.1f} pp vs approve-all"),
        ("Profit-max cut-off", f"{opt}"),
    ])
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        fig = cm.volume_rate_chart(tbl.assign(loans=tbl["approval_rate"] * len(d)), "cutoff",
                                   "Approval rate vs bad rate by cut-off", rate_cols=("bad_rate",),
                                   names={"bad_rate": "Bad rate of approved"}, count_col="loans")
        fig.update_xaxes(type="linear")
        fig.add_vline(x=cut, line_dash="dash", line_color=cm.GREY)
        cm.show_chart(fig, key="st_curve")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=tbl["cutoff"], y=tbl["net_profit"] / 1e6, name="Net profit", line=dict(color=cm.GREEN))
        fig.add_scatter(x=[opt], y=[tbl.loc[tbl["cutoff"] == opt, "net_profit"].iloc[0] / 1e6], mode="markers",
                        marker=dict(size=12, color=cm.RED), name="Optimum")
        fig.add_vline(x=cut, line_dash="dash", line_color=cm.GREY)
        fig.update_layout(title="Net profit by cut-off (sample, $M)", xaxis_title="Cut-off score",
                          yaxis_title="Net profit ($M)")
        cm.show_chart(fig, key="st_profit")

    st.subheader("Three-zone decision policy")
    z1, z2 = st.columns(2)
    approve_at = z1.slider("Auto-approve at or above", lo, hi, min(hi, cut + 15))
    decline_below = z2.slider("Auto-decline below", lo, hi, max(lo, cut - 15))
    if decline_below > approve_at:
        st.warning("The decline threshold must not exceed the approve threshold.")
    else:
        s = d["score"].to_numpy()
        zone = np.where(s >= approve_at, "Auto-approve", np.where(s < decline_below, "Auto-decline", "Manual review"))
        z = d.assign(zone=zone, profit=an.profit_vector(d, margin, lgd))
        g = z.groupby("zone").agg(loans=("default", "size"), bad_rate=("default", "mean"),
                                  exposure=("loan_amnt", "sum"), profit=("profit", "sum")).reindex(
            ["Auto-approve", "Manual review", "Auto-decline"]).fillna(0).reset_index()
        g["share"] = g["loans"] / len(z)
        g["return_on_exposure"] = np.where(g["exposure"] > 0, g["profit"] / g["exposure"].replace(0, np.nan), 0)
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = px.pie(g, names="zone", values="loans", hole=0.5, title="Applicant share by zone",
                         color="zone", color_discrete_map={"Auto-approve": cm.GREEN, "Manual review": cm.AMBER,
                                                           "Auto-decline": cm.RED})
            cm.show_chart(fig, key="st_pie")
        with c2:
            cm.show_df(g.assign(share=g["share"].map(cm.pct), bad_rate=g["bad_rate"].map(cm.pct),
                                exposure=g["exposure"].map(cm.money), profit=g["profit"].map(cm.money),
                                return_on_exposure=g["return_on_exposure"].map(cm.pct)))

    with st.expander("Full cut-off table"):
        cm.show_df(tbl.assign(approval_rate=tbl["approval_rate"].map(cm.pct), bad_rate=tbl["bad_rate"].map(cm.pct),
                              bads_avoided_pct=tbl["bads_avoided_pct"].map(cm.pct),
                              return_on_exposure=tbl["return_on_exposure"].map(cm.pct),
                              exposure=tbl["exposure"].map(cm.money), net_profit=tbl["net_profit"].map(cm.money)),
                   height=400)
        st.download_button("⬇️ Download cut-off table", tbl.to_csv(index=False).encode(), "cutoff_table.csv", "text/csv")
