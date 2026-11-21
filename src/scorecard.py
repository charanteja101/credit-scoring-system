"""WOE binning + logistic regression scorecard (classic bank-style credit score)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from . import config as C


class WoeBinner:
    """Bins each feature and maps bins to Weight of Evidence = ln(%bad / %good)."""

    def __init__(self, numeric, categorical, max_bins=8, min_cat_frac=0.01):
        self.numeric, self.categorical = numeric, categorical
        self.max_bins, self.min_cat_frac = max_bins, min_cat_frac
        self.info = {}

    def _assign(self, col, s):
        info = self.info[col]
        if info["type"] == "num":
            b = np.digitize(s.to_numpy(float), info["edges"])
            return pd.Series(np.where(s.isna().to_numpy(), -1, b), index=s.index)
        v = s.astype(object).where(s.notna(), "Missing")
        return v.where(v.isin(info["cats"]), "Other")

    def _label(self, col, b):
        info = self.info[col]
        if info["type"] == "cat":
            return str(b)
        e = info["edges"]
        if b == -1:
            return "Missing"
        if len(e) == 0:
            return "All"
        if b == 0:
            return f"< {e[0]:.4g}"
        if b == len(e):
            return f">= {e[-1]:.4g}"
        return f"[{e[b-1]:.4g}, {e[b]:.4g})"

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        B, G = y.sum(), (1 - y).sum()
        rows = []
        for col in self.numeric + self.categorical:
            if col in self.numeric:
                qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
                edges = np.unique(np.nanquantile(X[col].to_numpy(float), qs))
                self.info[col] = {"type": "num", "edges": edges}
            else:
                v = X[col].astype(object).where(X[col].notna(), "Missing")
                freq = v.value_counts(normalize=True)
                cats = set(freq[freq >= self.min_cat_frac].index) | {"Missing"}
                self.info[col] = {"type": "cat", "cats": cats}
            bins = self._assign(col, X[col])
            g = pd.DataFrame({"bin": bins, "bad": y}).groupby("bin")["bad"].agg(["count", "sum"])
            g["bad_n"], g["good_n"] = g["sum"], g["count"] - g["sum"]
            g["woe"] = np.log(((g["bad_n"] + .5) / B) / ((g["good_n"] + .5) / G))
            g["iv"] = ((g["bad_n"] + .5) / B - (g["good_n"] + .5) / G) * g["woe"]
            self.info[col]["woe"] = g["woe"].to_dict()
            self.info[col]["iv"] = g["iv"].sum()
            for b, r in g.iterrows():
                rows.append({"feature": col, "bin": self._label(col, b), "bin_id": b,
                             "count": int(r["count"]), "bad_rate": r["bad_n"] / r["count"],
                             "woe": r["woe"], "iv": r["iv"]})
        self.table_ = pd.DataFrame(rows)
        self.iv_ = pd.Series({c: self.info[c]["iv"] for c in self.info}).sort_values(ascending=False)
        return self

    def transform(self, X, cols=None):
        out = {}
        for col in cols or list(self.info):
            bins = self._assign(col, X[col])
            out[col] = bins.map(self.info[col]["woe"]).fillna(0.0)
        return pd.DataFrame(out, index=X.index)


class Scorecard:
    def __init__(self, iv_threshold=C.IV_THRESHOLD):
        self.iv_threshold = iv_threshold
        self.factor = C.PDO / np.log(2)
        self.offset = C.BASE_SCORE - self.factor * np.log(C.BASE_ODDS)

    def fit(self, X, y):
        self.binner = WoeBinner(C.NUMERIC_FEATURES, C.CATEGORICAL_FEATURES).fit(X, y)
        self.features = [f for f, iv in self.binner.iv_.items() if iv >= self.iv_threshold]
        W = self.binner.transform(X, self.features)
        self.model = LogisticRegression(C=1.0, max_iter=1000).fit(W, y)
        self.coef_ = pd.Series(self.model.coef_[0], index=self.features)
        return self

    def pd_(self, X):
        """Probability of default."""
        return self.model.predict_proba(self.binner.transform(X, self.features))[:, 1]

    def score(self, X):
        logit = self.model.decision_function(self.binner.transform(X, self.features))
        return np.clip(np.round(self.offset - self.factor * logit), C.MIN_SCORE, C.MAX_SCORE)

    def points_table(self):
        """Points per bin: the 'why my score is X' table."""
        t = self.binner.table_[self.binner.table_["feature"].isin(self.features)].copy()
        n = len(self.features)
        b0 = self.model.intercept_[0]
        t["points"] = [
            round(-(self.coef_[f] * w + b0 / n) * self.factor + self.offset / n, 1)
            for f, w in zip(t["feature"], t["woe"])
        ]
        return t

    def explain(self, row):
        """Points contributed by each feature for one applicant (1-row DataFrame)."""
        W = self.binner.transform(row, self.features).iloc[0]
        n, b0 = len(self.features), self.model.intercept_[0]
        pts = {f: -(self.coef_[f] * W[f] + b0 / n) * self.factor + self.offset / n
               for f in self.features}
        return pd.Series(pts).sort_values()
