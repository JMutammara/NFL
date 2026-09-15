import numpy as np

from nfl_pipeline.models.betting import expected_value, kelly_fraction, prob_cover_home, prob_over, poisson_prob_at_least
from nfl_pipeline.utils import american_payout, american_to_prob, prob_to_american


def test_odds_conversions_roundtrip():
    for o in (-110, -250, 120, 300):
        p = american_to_prob(o)
        assert np.isclose(prob_to_american(p), o, atol=1e-6)
    assert np.isclose(american_payout(-110), 100 / 110)
    assert np.isclose(american_payout(150), 1.5)


def test_cover_probability_conventions():
    # home favoured by 3, model says home wins by 3 -> ~50% cover
    assert np.isclose(prob_cover_home(3.0, 13.0, 3.0), 0.5, atol=0.01)
    # model likes home more than the market -> home cover > 50%
    assert prob_cover_home(6.0, 13.0, 3.0) > 0.55
    # integer line push handling keeps probabilities inside [0, 1]
    p = prob_cover_home(np.array([0.0, 10.0]), 13.0, np.array([0.0, 10.0]))
    assert np.all((p > 0.45) & (p < 0.55))


def test_ev_and_kelly():
    assert np.isclose(expected_value(0.5, 100), 0.0)
    assert expected_value(0.6, -110) > 0 > expected_value(0.5, -110)
    assert kelly_fraction(0.5, -110) == 0.0
    assert 0 < kelly_fraction(0.6, -110, fraction=0.25, cap=0.03) <= 0.03


def test_prop_probabilities():
    assert np.isclose(prob_over(60, 20, 60), 0.5)
    assert np.isclose(poisson_prob_at_least(0.0, 1), 0.0, atol=1e-6)
    assert np.isclose(poisson_prob_at_least(np.log(2), 1), 0.5)
