"""Betting math: cover / over / moneyline probabilities, EV and Kelly staking.

Conventions
-----------
* ``spread_line`` follows nflverse: positive = home favoured by that many points.
  Home covers when ``home_margin > spread_line``.
* Margin and total predictions are Normal(mu, sigma) with sigma estimated
  from out-of-fold residuals; player yardage props use the model's own
  heteroscedastic sigma; touchdown props use a Poisson rate.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, poisson

from ..utils import american_payout, american_to_prob


def prob_over(mu, sigma, line, continuity: float = 0.0):
    """P(X > line) under Normal(mu, sigma). ``continuity`` shifts the line (e.g. 0.5 for integer props)."""
    mu, sigma, line = (np.asarray(v, dtype=float) for v in (mu, sigma, line))
    sigma = np.maximum(sigma, 1e-6)
    return 1.0 - norm.cdf((line + continuity - mu) / sigma)


def prob_cover_home(mu_margin, sigma, spread_line):
    """P(home margin > spread_line); pushes on integer lines get half credit."""
    mu, s, L = (np.asarray(v, dtype=float) for v in (mu_margin, sigma, spread_line))
    s = np.maximum(s, 1e-6)
    p_gt = 1.0 - norm.cdf((L + 0.5 - mu) / s)
    p_push = norm.cdf((L + 0.5 - mu) / s) - norm.cdf((L - 0.5 - mu) / s)
    integer_line = np.isclose(L, np.round(L))
    return np.where(integer_line, p_gt + 0.5 * p_push, 1.0 - norm.cdf((L - mu) / s))


def poisson_prob_at_least(rate, k=1):
    rate = np.maximum(np.asarray(rate, dtype=float), 1e-9)
    return 1.0 - poisson.cdf(k - 1, rate)


def expected_value(p, odds):
    """EV per unit stake at American odds."""
    p = np.asarray(p, dtype=float)
    return p * american_payout(odds) - (1.0 - p)


def kelly_fraction(p, odds, fraction: float = 0.25, cap: float = 0.03):
    """Fractional Kelly stake as share of bankroll, capped and floored at 0."""
    p = np.asarray(p, dtype=float)
    b = american_payout(odds)
    f = (p * b - (1.0 - p)) / b
    return np.clip(f * fraction, 0.0, cap)


def market_edge_pct(p_model, odds):
    """Model probability minus vig-included implied probability, in percentage points."""
    return 100.0 * (np.asarray(p_model, dtype=float) - american_to_prob(odds))


def novig_two_way(odds_a, odds_b):
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    tot = pa + pb
    return pa / tot, pb / tot
