"""Shared UI helpers (charts, KPI cards, formatting)."""
import inspect
import streamlit as st
from plotly.subplots import make_subplots

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
    try:
        st.dataframe(df, hide_index=True, width="stretch", height=height)
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True, height=height)


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
