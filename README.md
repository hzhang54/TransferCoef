# TransferCoef

## Problem Statement

This project performs Monte Carlo simulations and optimization for portfolios to evaluate and report on transfer coefficients and other quantitative metrics.

## Running the Application

To run the main application using the default configuration from the project root directory, use the following PowerShell command. This command sets the `PYTHONPATH` so Python can locate the `transfercoef` source code inside the `src` directory:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m transfercoef.main --project-root .
```

## Running Tests

To run specific test files using your virtual environment, execute the following command (for example, to run the diagnostics tests):

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests\test_diagnostics.py
```
