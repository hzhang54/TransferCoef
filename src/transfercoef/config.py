from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the project."""

    project_root: Path
    output_tables_dir: Path
    output_logs_dir: Path
    notebooks_dir: Path
    local_cvxporfolio_root: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "ProjectPaths":
        root = Path(project_root).resolve()
        return cls(
            project_root=root,
            output_tables_dir=root / "output" / "tables",
            output_logs_dir=root / "output" / "logs",
            notebooks_dir=root / "notebooks",
            local_cvxporfolio_root= root / "cvxportfolio-master" / "cvxportfolio-master",
        )

@dataclass(frozen=True)
class Dataconfig:
    """Settings for optional historical-data calibration."""
    
    use_historical_calibration: bool = False
    tickers: tuple[str, ...] = ()
    cash_key: str = "USDOLLAR"
    start_date: str | None = None
    end_date: str | None = None
    min_history_days: int = 252

@dataclass(frozen=True)
class ScenarioConfig:
    """portfolio construction scenario used in Table 2 comparisons."""
    name: str
    description: str
    long_only: bool = False
    leverage_limit: float | None = None
    max_weight: float | None = None
    min_weight: float | None = None
    turnover_limit: float | None = None
    dollar_neutral: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SimulationConfig:
    """Core Monte Carlo simulation parameters."""
    
    random_seed: int = 7
    num_trials: int = 1_000
    num_assets: int = 100
    target_ic: float = 0.05
    breadth: float = 100.0
    risk_aversion: float = 1.0
    alpha_volatility: float = 0.02
    residual_volatility: float = 0.10
    use_risk_weighted_tc: bool = True
    report_plain_tc: bool = True
    include_turnover_path_dynamics: bool = False

@dataclass(frozen=True)
class OutputConfig:
    """Output formatting and export settings."""

    table_name: str = "table2_replication"
    export_csv: bool = True
    export_trial_data: bool = True
    float_precision: int = 6

@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""
    paths: ProjectPaths
    data: Dataconfig = field(default_factory=Dataconfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scenarios: tuple[ScenarioConfig, ...] = field(default_factory=tuple)


DEFAULT_SCENARIOS: tuple[ScenarioConfig, ...] = (
    ScenarioConfig(
        name="unconstrained",
        description="Reference unconstrained portfolio used as the transfer benchmark.",
        leverage_limit=None,
    ),
    ScenarioConfig(
        name="long_only",
        description="Long-only portfolio with no short positions.",
        long_only=True,
        leverage_limit=1.0,
    ),
    ScenarioConfig(
        name="long_only_max_weight",
        description="Long-only portfolio with a maximum position size.",
        long_only=True,
        leverage_limit=1.0,
        max_weight=0.05,
    ),
    ScenarioConfig(
        name="turnover_limited",
        description="Portfolio with a turnover constraint to model implementation frictions.",
        leverage_limit=1.0,
        turnover_limit=0.05,
        ),
    ScenarioConfig(
        name="combined_constraints",
        description="Long-only portfolio with leverage, position, and turnover limits.",
        long_only=True,
        leverage_limit=1.0,
        max_weight=0.05,
        turnover_limit=0.05
    ),  
)

def build_default_config(project_root: str | Path) -> AppConfig:
    """Create the default application configuration for this workspace."""
    
    return AppConfig(
        paths=ProjectPaths.from_project_root(project_root),
        scenarios=DEFAULT_SCENARIOS,
    )
    