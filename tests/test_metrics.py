import numpy as np
import pandas as pd
import pytest

from utils.metrics import annualized_return, annualized_volatility, max_drawdown, portfolio_metrics


def test_annualized_return_simple():
    eq = pd.Series([1.0, 1.1, 1.21])
    expected = (1.21 / 1.0) ** (252 / 2) - 1.0
    assert np.isclose(annualized_return(eq), expected)


def test_annualized_volatility_simple():
    r = [0.1, -0.1]
    expected = pd.Series(r).std() * np.sqrt(252)
    assert np.isclose(annualized_volatility(r), expected)


def test_max_drawdown_simple():
    eq = [1.0, 1.2, 0.9, 1.1]
    assert np.isclose(max_drawdown(eq), -0.25)


def test_portfolio_metrics_fields():
    eq = [1.0, 1.05, 1.0]
    r = [0.05, -0.047619]
    stats = portfolio_metrics(eq, r, risk_free_rate=0.0)
    assert np.isclose(stats["final_value"], 1.0)
    assert np.isclose(stats["cumulative_return"], 0.0)
    assert "annualized_return" in stats
    assert "annualized_volatility" in stats
    assert "max_drawdown" in stats


def test_sharpe_annualized_matches_daily_scaling():
    eq = [1.0, 1.02, 1.01, 1.03]
    r = [0.02, -0.00980392156862745, 0.019801980198019802]
    stats = portfolio_metrics(eq, r, risk_free_rate=0.0, periods_per_year=252)
    assert np.isclose(stats["sharpe_annualized"], stats["sharpe"] * np.sqrt(252.0))


def test_portfolio_metrics_fails_on_constant_returns():
    eq = [1.0, 1.01, 1.02]
    r = [0.01, 0.01, 0.01]
    with pytest.raises(ValueError):
        portfolio_metrics(eq, r, risk_free_rate=0.0)


def test_portfolio_metrics_fails_on_length_misalignment():
    eq = [1.0, 1.01, 1.02]
    r = [0.01]
    with pytest.raises(ValueError):
        portfolio_metrics(eq, r, risk_free_rate=0.0)
