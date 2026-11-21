import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src import analytics as an
from src import config as C
from . import common as cm

TIERS = [(0.2, "Very high risk", cm.RED), (0.4, "High risk", "#f97316"), (0.6, "Medium risk", cm.AMBER),
         (0.8, "Low risk", "#84cc16"), (1.01, "Very low risk", cm.GREEN)]


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
        cm.show_chart(_gauge(score, df), key="sg_gauge")
    with c2:
        cm.kpi_row([("Score", score), ("Probability of default", cm.pct(p)),
                    ("Better than", f"{pctile:.0%} of applicants")])
        st.markdown(f"<span class='tag' style='background:{colour};color:white'>{tier}</span> &nbsp; "
                    f"<span class='tag' style='background:{'#16a34a' if approved else '#dc2626'};color:white'>"
                    f"{'✅ APPROVE' if approved else '❌ DECLINE'}</span>", unsafe_allow_html=True)
        st.caption(f"Cut-off {int(cutoff)} · score is {abs(score - cutoff)} points {'above' if approved else 'below'} the line")

    expl = sc.explain(row)
    best = pts.groupby("feature")["points"].max()
    loss = (best.reindex(expl.index) - expl).sort_values(ascending=False)
    fig = px.bar(loss.sort_values().rename("points_lost").reset_index().rename(columns={"index": "feature"}),
                 x="points_lost", y=loss.sort_values().index.map(cm.label), orientation="h",
                 title="Points lost vs best possible answer, by factor", color_discrete_sequence=[cm.RED])
    fig.update_layout(yaxis_title="", xaxis_title="Points lost", height=420)
    cm.show_chart(fig, key="sg_expl")
    st.subheader("Main reasons for this score (adverse-action reason codes)")
    for f_, v in loss.head(3).items():
        st.markdown(f"- **{cm.label(f_)}** — cost {v:.0f} points compared with the best category")


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
    cm.kpi_row([("Applicants", f"{len(out):,}"), ("Approved", cm.pct((out['decision'] == 'APPROVE').mean())),
                ("Avg score", f"{out['score'].mean():.0f}"), ("Avg PD", cm.pct(out["pd"].mean()))])
    fig = px.histogram(out, x="score", color="decision", nbins=40, title="Score distribution",
                       color_discrete_map={"APPROVE": cm.GREEN, "DECLINE": cm.RED})
    cm.show_chart(fig, key="sg_batch_hist")
    cm.show_df(out.head(200), height=360)
    st.download_button("⬇️ Download scored file", out.to_csv(index=False).encode(), "scored_applicants.csv", "text/csv")


def render(ctx):
    st.header("🧑‍💼 Applicant Scoring")
    t1, t2 = st.tabs(["Single applicant", "Batch (CSV upload)"])
    with t1:
        _single(ctx)
    with t2:
        _batch(ctx)
