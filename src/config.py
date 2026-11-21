"""Central settings for the Credit Scoring System."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "loans_2012_2016.csv.gz"
MODEL_PATH = ROOT / "models" / "scorecard.joblib"
REPORT_DIR = ROOT / "reports"

# Only loans issued in this window are used
START_YEAR, END_YEAR = 2012, 2016
# Out-of-time split: train on <= TRAIN_END_YEAR, test on the later loans
TRAIN_END_YEAR = 2014

# Raw Kaggle columns we need. int_rate / grade / sub_grade / installment are
# deliberately excluded: they are LendingClub's own risk assessment (circular)
# and would not exist before pricing the loan.
RAW_COLUMNS = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "loan_status", "purpose", "dti",
    "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies",
]

NUMERIC_FEATURES = [
    "loan_amnt", "term_months", "annual_inc", "dti", "delinq_2yrs", "fico_avg",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies", "emp_length_yrs",
    "credit_history_yrs", "loan_to_income",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "default"  # 1 = charged off / defaulted, 0 = fully paid

# Scorecard scaling: BASE_SCORE points at BASE_ODDS good:bad, +PDO points doubles the odds
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
MIN_SCORE, MAX_SCORE = 300, 850
IV_THRESHOLD = 0.02   # drop features with weak predictive power
