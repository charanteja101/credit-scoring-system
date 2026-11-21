"""Load the Kaggle LendingClub file, keep 2012-2016, build target + features.

Usage:  python -m src.data_prep [path/to/accepted_2007_to_2018Q4.csv(.gz)]
"""
import sys
import numpy as np
import pandas as pd
from . import config as C

GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}


def find_raw_file(arg=None):
    if arg:
        return arg
    files = sorted(list(C.RAW_DIR.glob("*.csv")) + list(C.RAW_DIR.glob("*.csv.gz")))
    if not files:
        sys.exit(f"No CSV found in {C.RAW_DIR}. Download the Kaggle LendingClub file there.")
    return files[0]


def load_raw(path, chunksize=200_000):
    """Read in chunks and keep only 2012-2016 loans (the full file is ~2.2M rows)."""
    keep, wanted = [], set(C.RAW_COLUMNS)
    for chunk in pd.read_csv(path, usecols=lambda c: c in wanted,
                             chunksize=chunksize, low_memory=False):
        chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        chunk = chunk[chunk["issue_d"].dt.year.between(C.START_YEAR, C.END_YEAR)]
        chunk = chunk[chunk["loan_status"].isin(GOOD | BAD)]  # only resolved loans
        keep.append(chunk)
    return pd.concat(keep, ignore_index=True)


def _num(s):
    if s.dtype == object:
        s = s.astype(str).str.replace("%", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def engineer(df):
    """Turn raw columns into model features."""
    out = pd.DataFrame(index=df.index)
    out["loan_amnt"] = _num(df["loan_amnt"])
    out["term_months"] = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    out["annual_inc"] = _num(df["annual_inc"])
    out["dti"] = _num(df["dti"])
    out["delinq_2yrs"] = _num(df["delinq_2yrs"])
    out["fico_avg"] = (_num(df["fico_range_low"]) + _num(df["fico_range_high"])) / 2
    for col in ["inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
                "total_acc", "mort_acc", "pub_rec_bankruptcies"]:
        out[col] = _num(df[col])
    emp = df["emp_length"].astype(str)
    yrs = emp.str.extract(r"(\d+)")[0].astype(float)
    yrs[emp.str.contains("<", regex=False)] = 0
    out["emp_length_yrs"] = yrs
    out["credit_history_yrs"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].replace(0, np.nan)
    for col in C.CATEGORICAL_FEATURES:
        out[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})
    return out


def build_dataset(path=None):
    raw = load_raw(find_raw_file(path))
    raw["earliest_cr_line"] = pd.to_datetime(raw["earliest_cr_line"], format="%b-%Y", errors="coerce")
    data = engineer(raw)
    data[C.TARGET] = raw["loan_status"].isin(BAD).astype(int)
    data["issue_d"] = raw["issue_d"]
    return data.dropna(subset=["fico_avg", "annual_inc"]).reset_index(drop=True)


if __name__ == "__main__":
    ds = build_dataset(sys.argv[1] if len(sys.argv) > 1 else None)
    ds.to_csv(C.PROCESSED_PATH, index=False)
    print(f"Saved {len(ds):,} loans -> {C.PROCESSED_PATH}")
    print(ds.groupby(ds["issue_d"].dt.year)[C.TARGET].agg(["count", "mean"]).rename(
        columns={"count": "loans", "mean": "default_rate"}))
