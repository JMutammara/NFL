"""Stage-2 market-relative game models ("edge models").

Stage 1 (``GameEnsemble`` trained without market features) produces a
market-free estimate of the margin and total. Stage 2 predicts the *residual*
of the outcome against a reference line from a compact, hand-curated feature
set, so the learner only has to find systematic deviations from the market
instead of re-deriving the line:

    resid_margin = home_margin - spread_ref
    resid_total  = total_points - total_ref

``spread_ref`` is the opening line where one is known and the closing line
otherwise (``ref_is_open`` flags which). Members: a ridge regression, a bag of
shallow CatBoost regressors, and a bag of CatBoost cover classifiers. The
final cover probability is a logistic calibration over (predicted edge in
points, classifier logit) fitted on out-of-fold predictions, so stated
probabilities and Kelly stakes reflect validated accuracy rather than a
Normal-tail assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..utils import N_JOBS, american_payout, log

COMMON_FEATURES = [
    "pf_margin_dev", "pf_total_dev", "pf_margin", "pf_total",
    "mkt_line_dev_ref", "mkt_total_dev_ref", "mkt_hfa", "mkt_rt_diff",
    "spread_ref", "total_ref", "abs_spread", "home_dog", "big_fav", "pick_em", "ref_is_open",
    "implied_home_ref", "implied_away_ref",
    "div_game", "is_primetime", "is_thu", "is_mon", "is_sat", "is_playoff", "is_neutral", "week",
    "rest_diff", "home_rest", "away_rest", "home_off_bye", "away_off_bye", "home_short_week", "away_short_week",
    "away_travel_mi", "home_tz_shift", "away_tz_shift", "home_body_clock", "away_body_clock", "intl_game",
    "wind_mph", "temp_f", "is_dome", "cold_game", "windy_game", "is_grass", "weather_imputed",
    "home_ats_cover_margin_r3", "away_ats_cover_margin_r3", "home_ats_cover_margin_r6", "away_ats_cover_margin_r6",
    "home_ats_cover_margin_ewm", "away_ats_cover_margin_ewm",
    "home_ats_over_margin_r3", "away_ats_over_margin_r3", "home_ats_over_margin_ewm", "away_ats_over_margin_ewm",
    "home_qb_changed", "away_qb_changed", "home_qb_career_starts", "away_qb_career_starts", "home_qb_season_starts",
    "away_qb_season_starts", "d_qb_epa_db_ewm", "d_qb_epa_db_r3",
    "home_n_prior_games_season", "away_n_prior_games_season",
]
SPREAD_FEATURES = COMMON_FEATURES + [
    "net_epa_pp_adj_ewm", "net_epa_pp_r3", "net_epa_pp_adj_r6", "net_pass_epa_db_adj_ewm", "net_rush_epa_adj_ewm",
    "net_sr_adj_ewm", "net_turnover_rate_ewm", "net_explosive_rate_ewm", "net_ppd_adj_ewm", "net_start_yl100_ewm",
    "home_off_turnover_rate_r3", "away_off_turnover_rate_r3", "home_def_turnover_rate_r3", "away_def_turnover_rate_r3",
    "line_home_pressure_rate_trk_ewm", "line_away_pressure_rate_trk_ewm", "line_home_sack_rate_ewm", "line_away_sack_rate_ewm",
]
TOTAL_FEATURES = COMMON_FEATURES + [
    "mu_home_plays_ewm", "mu_away_plays_ewm", "mu_home_ppd_adj_ewm", "mu_away_ppd_adj_ewm", "mu_home_epa_pp_adj_ewm",
    "mu_away_epa_pp_adj_ewm", "mu_home_pass_epa_db_ewm", "mu_away_pass_epa_db_ewm", "mu_home_sec_per_play_ewm",
    "mu_away_sec_per_play_ewm", "mu_home_pass_rate_neutral_ewm", "mu_away_pass_rate_neutral_ewm",
    "mu_home_explosive_rate_ewm", "mu_away_explosive_rate_ewm", "mu_home_rz_td_rate_ewm", "mu_away_rz_td_rate_ewm",
    "home_off_pass_rate_neutral_ewm", "away_off_pass_rate_neutral_ewm", "home_off_sec_per_play_ewm", "away_off_sec_per_play_ewm",
    "home_off_plays_r3", "away_off_plays_r3", "home_def_plays_r3", "away_def_plays_r3",
    "home_off_adot_ewm", "away_off_adot_ewm", "home_off_explosive_pass_rate_ewm", "away_off_explosive_pass_rate_ewm",
]
FEATURES = {"spread": SPREAD_FEATURES, "total": TOTAL_FEATURES}
ID_COLS = ["game_id", "season", "week", "gameday", "home_team", "away_team", "played"]


def build_edge_frame(gf: pd.DataFrame, pf_margin: np.ndarray, pf_total: np.ndarray, ref: str = "open") -> pd.DataFrame:
    """Derived stage-2 columns for the rows of ``gf`` (any subset, any order).

    ``pf_margin`` / ``pf_total`` are stage-1 predictions aligned with ``gf``:
    out-of-fold during training, final-model predictions when forecasting.
    """
    e = gf.copy()
    e["pf_margin"] = np.asarray(pf_margin, dtype=float)
    e["pf_total"] = np.asarray(pf_total, dtype=float)
    sp_open = e["spread_open"] if "spread_open" in e else pd.Series(np.nan, index=e.index)
    to_open = e["total_open"] if "total_open" in e else pd.Series(np.nan, index=e.index)
    if ref == "open":
        e["spread_ref"] = sp_open.fillna(e["spread_line"])
        e["total_ref"] = to_open.fillna(e["total_line"])
        e["ref_is_open"] = sp_open.notna().astype(float)
    else:
        e["spread_ref"], e["total_ref"], e["ref_is_open"] = e["spread_line"], e["total_line"], 0.0
    e["pf_margin_dev"] = e["pf_margin"] - e["spread_ref"]
    e["pf_total_dev"] = e["pf_total"] - e["total_ref"]
    e["mkt_line_dev_ref"] = e["spread_ref"] - e["mkt_line_hat"] if "mkt_line_hat" in e else np.nan
    e["mkt_total_dev_ref"] = e["total_ref"] - e["mkt_total_hat"] if "mkt_total_hat" in e else np.nan
    e["abs_spread"] = e["spread_ref"].abs()
    e["home_dog"] = (e["spread_ref"] < 0).astype(float)
    e["big_fav"] = (e["abs_spread"] >= 7).astype(float)
    e["pick_em"] = (e["abs_spread"] <= 1).astype(float)
    e["implied_home_ref"] = (e["total_ref"] + e["spread_ref"]) / 2.0
    e["implied_away_ref"] = (e["total_ref"] - e["spread_ref"]) / 2.0
    # targets (NaN for unplayed games)
    e["resid_margin"] = e["home_margin"] - e["spread_ref"]
    e["resid_total"] = e["total_points"] - e["total_ref"]
    e["cover_home"] = np.sign(e["resid_margin"])
    e["over_ref"] = np.sign(e["resid_total"])
    e["clv_spread"] = e["spread_line"] - e["spread_ref"]   # close minus reference, home perspective
    e["clv_total"] = e["total_line"] - e["total_ref"]
    return e


def _cat_params(seed: int, iters: int, kind: str) -> dict[str, Any]:
    p = dict(iterations=iters, learning_rate=0.03, depth=3, l2_leaf_reg=30.0, rsm=0.7, min_data_in_leaf=40,
             random_strength=1.0, bagging_temperature=0.5, random_seed=seed, verbose=0, thread_count=N_JOBS,
             allow_writing_files=False)
    p["loss_function"] = "Logloss" if kind == "cls" else "RMSE"
    return p


@dataclass
class EdgeModel:
    task: str  # "spread" | "total"
    n_bags: int = 3
    iterations: int = 800
    es_rounds: int = 60
    seed: int = 42
    features_: list[str] = field(default_factory=list)
    ridge_: Any = None
    cats_: list = field(default_factory=list)
    clss_: list = field(default_factory=list)
    best_iters_: dict[str, int] = field(default_factory=dict)
    weights_: dict[str, float] = field(default_factory=lambda: {"ridge": 0.5, "cat": 0.5})
    calib_: dict[str, float] | None = None       # P(cover) = expit(a*edge_pts + b*logit(p_cls) + c)
    sigma_: float = np.nan                        # OOF sd of (outcome - (ref + resid_pred))
    win_calib_: dict[str, float] | None = None    # spread only: P(home win) = expit(a*logit(p_norm) + b)

    @property
    def target_col(self) -> str:
        return "resid_margin" if self.task == "spread" else "resid_total"

    @property
    def cover_col(self) -> str:
        return "cover_home" if self.task == "spread" else "over_ref"

    @property
    def ref_col(self) -> str:
        return "spread_ref" if self.task == "spread" else "total_ref"

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: np.ndarray, cover: np.ndarray, X_es: pd.DataFrame | None = None,
            y_es: np.ndarray | None = None, cover_es: np.ndarray | None = None, fixed_iters: dict[str, int] | None = None) -> "EdgeModel":
        from catboost import CatBoostClassifier, CatBoostRegressor

        self.features_ = list(X.columns)
        y = np.asarray(y, dtype=float)
        cover = np.asarray(cover, dtype=float)
        self.ridge_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 25)))
        self.ridge_.fit(X, y)
        use_es = X_es is not None and len(X_es) > 0 and not fixed_iters
        self.cats_, self.clss_ = [], []
        reg_iters, cls_iters = [], []
        live = cover != 0
        live_es = (np.asarray(cover_es, dtype=float) != 0) if cover_es is not None else None
        for b in range(self.n_bags):
            seed = self.seed + 17 * b
            it_reg = fixed_iters.get("cat", self.iterations) if fixed_iters else self.iterations
            it_cls = fixed_iters.get("cls", self.iterations) if fixed_iters else self.iterations
            reg = CatBoostRegressor(**_cat_params(seed, it_reg, "reg"), **({"od_type": "Iter", "od_wait": self.es_rounds} if use_es else {}))
            cls = CatBoostClassifier(**_cat_params(seed, it_cls, "cls"), **({"od_type": "Iter", "od_wait": self.es_rounds} if use_es else {}))
            if use_es:
                reg.fit(X, y, eval_set=(X_es, y_es), use_best_model=True)
                cls.fit(X[live], (cover[live] > 0).astype(int), eval_set=(X_es[live_es], (np.asarray(cover_es)[live_es] > 0).astype(int)), use_best_model=True)
                reg_iters.append(reg.get_best_iteration() + 1)
                cls_iters.append(cls.get_best_iteration() + 1)
            else:
                reg.fit(X, y)
                cls.fit(X[live], (cover[live] > 0).astype(int))
            self.cats_.append(reg)
            self.clss_.append(cls)
        if reg_iters:
            self.best_iters_ = {"cat": int(np.median(reg_iters)), "cls": int(np.median(cls_iters))}
        return self

    # -------------------------------------------------------------- predict
    def predict_members(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features_]
        ridge = self.ridge_.predict(X)
        cat = np.mean([m.predict(X) for m in self.cats_], axis=0)
        p_cls = np.mean([m.predict_proba(X)[:, 1] for m in self.clss_], axis=0)
        return pd.DataFrame({"ridge": ridge, "cat": cat, "p_cls": p_cls}, index=X.index)

    def blend_resid(self, members: pd.DataFrame) -> np.ndarray:
        return members["ridge"].to_numpy() * self.weights_["ridge"] + members["cat"].to_numpy() * self.weights_["cat"]

    def cover_prob(self, edge_pts: np.ndarray, p_cls: np.ndarray) -> np.ndarray:
        """Calibrated P(home covers / over) for a bet at a line that sits ``edge_pts`` below the model mean."""
        edge_pts = np.asarray(edge_pts, dtype=float)
        p_cls = np.clip(np.asarray(p_cls, dtype=float), 1e-6, 1 - 1e-6)
        if self.calib_ is None:
            return norm.sf(-edge_pts / (self.sigma_ if np.isfinite(self.sigma_) else 13.5))
        c = self.calib_
        return expit(c["a"] * edge_pts + c["b"] * logit(p_cls) + c["c"])

    def predict(self, X: pd.DataFrame, ref_line: np.ndarray, bet_line: np.ndarray | None = None) -> pd.DataFrame:
        """Model mean (ref + residual) and calibrated probability at ``bet_line`` (defaults to ``ref_line``)."""
        m = self.predict_members(X)
        resid = self.blend_resid(m)
        ref_line = np.asarray(ref_line, dtype=float)
        mu = ref_line + resid
        line = ref_line if bet_line is None else np.asarray(bet_line, dtype=float)
        edge = mu - line
        out = pd.DataFrame({"resid_pred": resid, "mu": mu, "edge_pts": edge, "p_cls": m["p_cls"].to_numpy(),
                            "p_cover": self.cover_prob(edge, m["p_cls"].to_numpy()),
                            "p_cover_normal": norm.sf(-edge / (self.sigma_ if np.isfinite(self.sigma_) else 13.5))}, index=X.index)
        if self.task == "spread":
            p_norm = np.clip(norm.sf(-mu / (self.sigma_ if np.isfinite(self.sigma_) else 13.5)), 1e-6, 1 - 1e-6)
            if self.win_calib_:
                out["p_home_win"] = expit(self.win_calib_["a"] * logit(p_norm) + self.win_calib_["b"])
            else:
                out["p_home_win"] = p_norm
        return out

    # --------------------------------------------------- fit on OOF outputs
    def fit_blend_and_calibration(self, oof: pd.DataFrame, y_resid: np.ndarray, cover: np.ndarray,
                                  outcome: np.ndarray | None = None, ref_line: np.ndarray | None = None) -> None:
        y_resid = np.asarray(y_resid, dtype=float)
        A = oof[["ridge", "cat"]].to_numpy()
        w, _ = nnls(A, y_resid)
        if w.sum() <= 1e-9:
            w = np.array([0.5, 0.5])
        # do not let the blend inflate: cap total weight at 1 (shrink toward the market)
        if w.sum() > 1.0:
            w = w / w.sum()
        self.weights_ = {"ridge": float(w[0]), "cat": float(w[1])}
        resid_pred = A @ w
        self.sigma_ = float(np.std(y_resid - resid_pred, ddof=1))
        live = np.asarray(cover) != 0
        Z = np.column_stack([resid_pred[live], logit(np.clip(oof["p_cls"].to_numpy()[live], 1e-6, 1 - 1e-6))])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z, (np.asarray(cover)[live] > 0).astype(int))
        self.calib_ = {"a": float(lr.coef_[0][0]), "b": float(lr.coef_[0][1]), "c": float(lr.intercept_[0])}
        if self.task == "spread" and outcome is not None and ref_line is not None:
            mu = np.asarray(ref_line, dtype=float) + resid_pred
            p_norm = np.clip(norm.sf(-mu / self.sigma_), 1e-6, 1 - 1e-6)
            yw = (np.asarray(outcome, dtype=float) > 0).astype(int)
            ok = np.asarray(outcome, dtype=float) != 0
            lw = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_norm[ok]).reshape(-1, 1), yw[ok])
            self.win_calib_ = {"a": float(lw.coef_[0][0]), "b": float(lw.intercept_[0])}
        log.info("%s edge blend: ridge %.2f cat %.2f | sigma %.2f | calib a=%.3f b=%.3f c=%.3f", self.task,
                 self.weights_["ridge"], self.weights_["cat"], self.sigma_, self.calib_["a"], self.calib_["b"], self.calib_["c"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "EdgeModel":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Evaluation: betting simulation against the reference line
# ---------------------------------------------------------------------------
def roi_table(p_cover: np.ndarray, cover: np.ndarray, clv: np.ndarray | None = None, odds: float = -110.0,
              thresholds=(0.50, 0.52, 0.54, 0.56, 0.58, 0.60), n_boot: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Flat-stake ROI when betting the model's side whenever max(p, 1-p) >= threshold.

    ``cover`` is +1 (home covers / over), -1, or 0 (push). ``clv`` is the
    close-minus-reference movement from the home/over perspective; a pick on
    the home/over side gains clv points of closing-line value, the other side
    loses it.
    """
    p_cover = np.asarray(p_cover, dtype=float)
    cover = np.asarray(cover, dtype=float)
    pay = float(american_payout(odds))
    side = np.where(p_cover >= 0.5, 1.0, -1.0)
    conf = np.maximum(p_cover, 1 - p_cover)
    result = np.where(cover == 0, 0.0, np.where(side == cover, pay, -1.0))
    rng = np.random.default_rng(seed)
    out = {}
    for thr in thresholds:
        pick = conf >= thr
        n = int(pick.sum())
        if n == 0:
            out[f"p>={thr:.2f}"] = {"n": 0}
            continue
        r = result[pick]
        live = r != 0
        hit = float(np.mean(r[live] > 0)) if live.any() else np.nan
        roi = float(r.mean())
        boots = np.array([rng.choice(r, size=len(r), replace=True).mean() for _ in range(n_boot)]) if n >= 20 else np.array([np.nan])
        row = {"n": n, "hit_rate": hit, "roi": roi, "roi_ci_low": float(np.nanpercentile(boots, 2.5)),
               "roi_ci_high": float(np.nanpercentile(boots, 97.5)), "p_roi_positive": float(np.nanmean(boots > 0)),
               "units": float(r.sum())}
        if clv is not None:
            c = np.asarray(clv, dtype=float)[pick] * side[pick]
            ok = ~np.isnan(c)
            row["clv_pts"] = float(c[ok].mean()) if ok.any() else np.nan
            row["clv_positive_rate"] = float(np.mean(c[ok] > 0)) if ok.any() else np.nan
            row["clv_nonneg_rate"] = float(np.mean(c[ok] >= 0)) if ok.any() else np.nan
        out[f"p>={thr:.2f}"] = row
    return out


def breakeven(odds: float = -110.0) -> float:
    pay = float(american_payout(odds))
    return 1.0 / (1.0 + pay)
