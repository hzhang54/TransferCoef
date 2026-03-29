from __future__ import annotations

"""Optional historical-data helpers for future calibration workflows.

The current Table 2 Monte Carlo implementation runs entirely from synthetic
inputs and does not require any market data.  This model exists to support a 
future historical-calibration mode using local ``cvxportfolio`` data utilities.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import AppConfig, Dataconfig
from .cvxportfolio_adapter import import_cvxportfolio

@dataclass(frozen=True)
class marketDataBundle:
    """Container for market data used in calibration and research workflows."""

    returns: pd.DataFrame
    prices: pd.DataFrame | None = None
    volumes: pd.DataFrame | None = None
    cash_key: str = "USDOLLAR"

    @property
    def asset_columns(self) -> list[str]:
        """Return non-cash asset columns from the returns matrix."""

        if self.returns.empty:
            return []
        return [column for column in self.returns.columns if column != self.cash_key]
        
    @property
    def asset_returns(self) -> pd.DataFrame:
        """Return the returns matrix without the cash column."""
        return self.returns.loc[:, self.asset_columns]

@dataclass(frozen=True)
class HistoricalCalibration:
    """Summary statistics derived from historical market data."""

    returns: pd.DataFrame
    covariance: pd.DataFrame
    mean_returns: pd.Series
    volatilities: pd.Series
    asset_columns: list[str]
    cash_key: str

def load_yahoo_market_data(
    project_root: str | Path,
    tickers: list[str] | tuple[str, ...],
    cash_key: str = "USDOLLAR",
    base_location: str | Path | None = None,
    min_history_days: int = 252,
    trading_frequency: str | None = None,
) -> marketDataBundle:
    """Load market data from the local vendored ``cvxportfolio`` Yahoo Finance source."""

    cvxportfolio = import_cvxportfolio(project_root)
    storage_root = Path(base_location) if base_location is not None else None

    market_data = cvxportfolio.DownloadedMarketData(
        universe=list(tickers),
        datasource="YahooFinance",
        cash_key=cash_key,
        base_location=storage_root if storage_root is not None else cvxportfolio.data.symbol_data.BASE_LOCATION,
        min_history=pd.Timedelta(days=min_history_days),
        trading_frequency=trading_frequency,
    )

    return marketDataBundle(
        returns=market_data.returns.copy(),
        prices=None if market_data.prices is None else market_data.prices.copy(),
        volumes=None if market_data.volumes is None else market_data.volumes.copy(),
        cash_key=cash_key,
    )


def build_user_provided_market_data(
    project_root: str | Path,
    returns: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    cash_key: str = "USDOLLAR",
) -> object:
    """Wrap user-provided data in the local ``cvxportfolio`` market data container."""

    cvxportfolio = import_cvxportfolio(project_root)
    return cvxportfolio.UserProvidedMarketData(
        returns=returns,
        prices=prices,
        volumes=volumes,
        cash_key=cash_key,
    )

def estimate_historical_calibration(
    market_data: marketDataBundle,
    min_periods: int = 20,
) -> HistoricalCalibration:
    """Compute simple historical calibration statistics from a market data bundle."""
    
    asset_returns = market_data.asset_returns.dropna(how="all")
    covariance = asset_returns.cov(min_periods=min_periods)
    mean_returns = asset_returns.mean()
    volatilities = asset_returns.std()
    
    return HistoricalCalibration(
        returns=asset_returns,
        covariance=covariance,
        mean_returns=mean_returns,
        volatilities=volatilities,
        asset_columns=market_data.asset_columns,
        cash_key=market_data.cash_key,
    )

def load_calibration_from_config(
   config: AppConfig,     
) -> HistoricalCalibration | None:
    """Load historical calibration inputs from the application configuration."""

    data_config = config.data
    if not data_config.use_historical_calibration:
        return None

    if not data_config.tickers:
        raise ValueError(
            "Historical calibration is enabled, but no tickers were provided in DataConfig."
        )

    bundle = load_yahoo_market_data(
        project_root=config.paths.project_root,
        tickers=data_config.tickers,
        cash_key=data_config.cash_key,
        base_location=config.paths.local_cvxportfolio_root.parent,
        min_history_days=data_config.min_history_days,
    )
    return estimate_historical_calibration(bundle)

def build_data_config(
    ticker: list[str] | tuple[str, ...],
    cash_key: str = "USDOLLAR",
    use_historical_calibration: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
    min_history_days: int = 252,
) -> DataConfig:
    """Convenience helper for constructing a ``DataConfig`` instance."""

    return DataConfig(
        use_historical_calibration=use_historical_calibration,
        tickers=tuple(ticker),
        cash_key=cash_key,
        start_date=start_date,
        end_date=end_date,
        min_history_days=min_history_days,
    )
    
    