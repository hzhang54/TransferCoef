from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import AppConfig, Dataconfig, OutputConfig, SimulationConfig, build_default_config
from .monte_carlo import build_scenarios_overview, run_monte_carlo 
from .table2 import (
    build_table2_report,
    create_table2_summary,
    export_table2_summary,
    summarize_table2_report,
)

def parse_args() -> argparse.Namespace:
    """parse command-line arguments for the Table 2 synthetic workflow."""
    
    