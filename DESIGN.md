# Design Document: Table 2 Replication for Ex Post Correlation Diagnostics

## 1. Purpose

This document defines the design and implementation plan for a Python project that replicates the Monte Carlo simulation-based **Ex Post correlation diagnostics** descirbed in the Transfer Coefficient paper, specifically **Table 2 on page 60**.

The immediate goal is **not** to start coding the full implementation yet, but to provide a reviewable design that:

1. explains the methodology behind Table 2,
2. maps the required Monte Carlo simulation workflow,
3. defines how local `cvxportfolio` utilities will be integrated,
4. breaks the work into a detailed step-by-step implementation plan with a TODO checklist.

This design assumes that:
- the project will be developed in this workspace,
- `cvxportfolio` will be used from the local source tree under `cvxportfolio-master`,
- no dependency installation from remote sites will be required for the design phase,
- executable validation may occur later in a different Python environment where `cvxportfolio` and its dependencies are available.

---

## 2. Problem statement

The paper extends the Fundamental Law of Active Management by introducing the **Transfer Coefficient (TC)** to quantify how portfolio constraints reduce the impementable value of alpha forecasts.

The standard Fundamental Law is:

$$
IR \approx IC \times \sqrt{BR}
$$

where:

- **IR** is the Information Ratio,
- **IC** is the Information Coefficient,
- **BR** is the Breadth.

With relaistic portfolio constrains, the paper proposes:

$$
IR \approx TC \times IC \times \sqrt{BR}
$$

where **TC** measures how effectively forecast information is transferred into portfolio positions.

Table 2 on page 60 reports **Monte Carlo-based ex post diagnostics** that compare theoretical expectations against realized simulation outcomes under different portfolio constraint regimes.

The target deliverable for this project is a reproducible python workflow that can:

- generate simulated alpha and return data,
- solve unconstrained and constrained portfolio optimization problems,
- compute ex post IC / TC / IR diagnostics,
- aggregate results across simulation trials,
- recreate a programmatic equivalent of Table 2.

---

## 3. Scope

### 3.1 In Scope

- Reproduce the Table  2 simulation workflow conceptually and computationally.
- Implement reusable Python modules for simulation, diagnostics, optimization, and reporting.
- Use `cvxportfolio` local utilities where appropriate for:
    - Yahoo Finance market data access,
    - portfolio optimization policies,
    - covariance /forecast infrastructure,
    - backtest and result handling when helpful.
- Support a **synthetic Monthe Carlo path** model for direct replication of the paper.
- Support an optional **historical-data-assisted calibration** mode using Yahoo Finance data via local `cvxportfolio` data utilities.

### Out of scope for the first implementation pass

- Exact reproduction of every number in the published table without calibration review.
- Full live backtesting framework.
- Production packaging and distribution.
- Hyper-parameter search or generalized research UI.
- Remote installation or retrieval of third-party packages.

---

### 4. Paper Methodology Summary for Table 2

### 4.1 Core concepts

The implementation will revolve around the following quantities.

#### Information Coefficient (IC)
A cross-sectional correlation between forecasted returns and realized returns.

Typical simulation definition:

$$
IC_t = Corr(\hat{\alpha}_t, r_t)
$$

where:

- $\hat{\alpha}_t$ is the forecast alpha vector for the cross-section,,
- $r_t$ is the realized return vector for the same cross-section.

#### Transfer Coefficient (TC)

A measure of how closely the constrained portfolio matches the unconstrained alpha-optimal portfolio.

Conceptually:

$$
TC_t = Corr(w_t^{*}, w_t^{c})
$$

where:

- $w_t^{*}$ is the unconstrained target portfolio,
- $w_t^{c}$ is the constrained implemented portfolio.

The paper discusses covariance/risk-aware forms of this relationship.  In impementation we should support both:

1. **plain cross-sectional correlation**, and
2. **risk-weighted transfer coefficient** using inverse residual variance weights.

#### Breadth (BR)

Breadth represents the number of independent bets or effective decision opportunities.

In the Table 2 replication, breadth should be configurable and explicitly documented because multiple practical proxies are possible:

- number of assets,
- effective independent assets after correlation adjustment,
- number of rebalance opportunities,
- composite breadth = cross-sectional bets × time opportunities.

For the first implementation, breadth will be parameterized rather than hard-coded.

#### Information Ratio (IR)
The ex post portfolio-level statistic:

$$
IR = \frac{\text{active return}}{\text{active risk}}
$$

In the simulation pipeline this can be estimated either:

- directly from realized active returns over repeated trials, or
- from repeated simulated periods within each trial.

---
### 4.1 Table 2 replication objective

Table 2 is interpreted as a Monte Carlo diagonostic table that compares:

- theoretical IC/TC/IR relationships, 
- realized ex post values from simulation,
- effects of increasingly restrictive constraints.

The codebase should therefore be designed to evaluate **multiple constraint scenarios** side by side, such as:

1. unconstrained
2. long-only
3. long-only + max weight,
4. turnover-limited,
5. combined constraints.

Each scenario should produce summary diagnostics including at minimum:

- mean ex post IC,
- standard deviation of IC,
- mean ex post TC,
- standard deviation of TC,
- mean ex post IR,
- theoretical IR from $TC \times IC \times \sqrt{BR}$
- realized/theoretical IR ratio.

If the exact row names or scenario structure in the paper differ after later validation against the PDF text, the reporting layer should be easy to adjust without changing the core simulation engine.

---

### 5. Monte Carlo Simulation Design

### 5.1 High-level workflow

Each simulation trial will follow this structure:

1. define cross-sectional universe and model parameters,
2. generate latent true alphas,
3. generate forecast alphas with a target IC structure,
4. generate residual risk / covariance inputs,
5. solve an unconstrained optimization problem,
6. solve one or more constrained optimization problems,
7. realize returns,
8. compute ex post diagnostics,
9. store trial-level outputs,
10. aggregate all trials into a Table 2-style summary.

---

### 5.2 Proposed statistical model

#### Step A: Generate true latent alpha

For each trial and asset $i$:

$$
\alpha_i \sim \mathcal{N}(0, \sigma_{\alpha}^2)
$$

This represents the latent signal component in realized returns.


#### Step B: Generate forecast alphas with target IC

We need a forecast vector that has a controlled correlation with true alpha.  A standard construction is:

$$
\hat{\alpha}_i = \rho \alpha_i + \sqrt{1 - \rho^2} \epsilon_i
$$

where:

- $\rho$ is the target IC
- $\epsilon_i \sim \mathcal{N}(0, \sigma_{\alpha}^2)$ independent noise.

This gives a controllable forecast quality.

#### Step C: Generate realized returns

Realized return for asset $i$:

$$
r_i = \alpha_i + \eta_i
$$

where $\eta_i$ is idiosyncratic noise sampled from a covariance structure.

This covariance may be:

- diagonal only (simpler base case), or
- factor + residual structure (advanced option), or
- historically calibrated from Yahoo Finance data.

---

### 5.3 Optimization formulation

For each trial, we need both unconstrained and constrained portfolios.

#### Unconstrained target portfolio

A simplified Markowitz-style active portfolio:

$$
\max_w \; \hat{\alpha}^T w - \frac{\lambda}{2} w^T \Sigma w
$$

subject to baseline feasibility conditions such as:

- full investment or zero-sum active weights,
- optional benchmark neutrality,
- optional leverage normalization.

This portfolio serves as the reference portfolio for TC.

#### Constrained portfolio

The same objective is re-solved under realistic implementation constraints, such as:

- long-only,
- leverage limits,
- max/min weights,
- turnover limit,
- benchmark neutrality,
- dollar neutrality where applicable.

Each constraint set defines one scenario column in the replicated Table 2 output.

---

### 5.4 Ex post diagnostics to compute per trial

For each simulation trial and scenario, compute:

#### Ex post IC

$$
IC^{expost} = Corr(\hat{\alpha}, r)
$$

#### Ex post TC

At least two supported variants:

1. **Pearson TC**

$$
TC_{pearson} = Corr(w^*, w^c)
$$

2. **Risk-weighted TC**

$$
TC_{risk} = \frac{\sum_i v_i w_i^* w_i^c}{\sqrt{\sum_i v_i (w_i^*)^2} \sqrt{\sum_i v_i (w_i^c)^2}}
$$

where $v_i$ is typically inverse residual variance or another risk weight.

#### Active return

$$
AR = (w^c)^T r
$$

or benchmark-relative active return if a benchmark vector is included.

#### Active risk

$$
\sigma_A = \sqrt{(w^c)^T \Sigma w^c}
$$

#### Ex post IR

$$
IR^{expost} = \frac{AR}{\sigma_A}
$$

#### Theoretical IR

$$
IR^{theory} = TC \times IC \times \sqrt{BR}
$$

This should be computed using configurable conventions:

- plug-in means of ex post IC and ex post TC,
- target IC and average realized TC,
- scenario-specific breadth if applicable.

The design should make the choice explicit in configuration.

---

### 5.5 Aggregation rules

Across all simulation trials and scenarios, aggregate:

- mean,
- standard deviation,
- selected quantiles,
- optional confidence intervals.

The primary output is a tidy summary table with one row per diagnostic and one column per scenario, or a long-format DataFrame pivoted into that view.

---

## 6. cvxportfolio Integration Design

The user requested that optimization and data utilities be based on the local `cvxportfolio` source tree in this workspace rather than a remote installation.

### 6.1 Local source location

The local source tree is expected under:

- `cvxportfolio-master/cvxportfolio-master/`

The implementation should support importing this local package in one of two ways:

1. by adding the source root to `sys.path` or
2. by structuring the project so that the interpreter can import the local package directly.

For the implementation phase, the simplest and safest path is:

- use a small bootstrap helper that appends the local `cvxportfolio-master/cvxportfolio-master` path to `sys.path` if needed.

---

### 6.2 Relevant local cvxportfolio components identified

From the local source review, the following modules/classes are especially relevant.

#### Data utilities

- `cvxportfolio.data.YahooFinance`
- `cvxportfolio.data.DownloadedmarketData`
- `cvxportfolio.data.userProvidedMarketData`
- `cvxportfolio.data.Fred`

These are exposed through - `cvxportfolio.data.__init__` and implemented under: 

- `cvxportfolio.data.market_data.py`
- `cvxportfolio.data.symbol_data.py`

`YahooFinance` provides cached symbold ata with market data abstractions that can be reused for:

- price history,
- return history,
- volume history,
- optional historical calibration.

#### Optimization policies

- `cvxportfolio.SinglePeriodOptimization`
- `cvxportfolio.MultiPeriodOptimization`
- `cvxportfolio.Hold`
- `cvxportfolio.Uniform`
- related policy infrastructure in `cvxportfolio.policy.py`

#### Objective / cost terms

- `cvxportfolio.ReturnsForecast`
- `cvxportfolio.CashReturn`
- `cvxportfolio.FullCovariance`
- `cvxportfolio.DiagonalCovariance`
- `cvxportfolio.FactorModelCovariance`
- transaction/holding costs if later needed.

#### Constraints

Identified locally in `cvxportfolio/constraints/constraints.py`:

- `LongOnly`
- `LeverageLimit`
- `DollarNeutral`
- `TurnoverLimit`
- `MaxWeights`
- `MinWeights`
- plus other reusable constraints.

#### Simulation / results

- `cvxportfolio.MarketSimulator`
- `cvxportfolio.StockMarketSimulator`
- `cvxportfolio.BacktestResult`

These are usful for historical-data calibration and for validating scenario behavior, though the core Table 2 replication may rely more directly on optimization than full backtesting.

---

### 6.3 Proposed integration strategy

The project should use - `cvxportfolio` in two distinct ways.

#### A. historical calibration mode

Use local `YahooFinance` / `DownloadedMarketData` to retrieve and cache historical data for:

- estimating volatilities,
- estimating covariance matrices,
- deriving relaistic asset universes,
- optionally calibrating turnover, leverage, and position-limit assumptions.

This mode helps translate the paper's simulation into a realistic empirical setup.

#### B. optimization engine mode

Use `SinglePeriodOptimization` with explicit objective and scenario constraints to compute:

- uncontrained target weights,
- constrained scenario weights,
- optional benchmark-relative active allocations.

This is the preferred route for implementing the paper's constrained vs. unconstrained portfolio comparison.

---

### 6.4 Design note: separation of responsibilities

Although - `cvxportfolio` can backtest policies, the Table 2 replication is better treated as a **research simulation engine** rather than a full historical backtest.

Therefore the implementation should separate:

- **Monte Carlo synthetic data generation**, and
- **portfolio optimization / diagnostics**.

`cvxportfolio` should be used where it adds value:

- constraint modeling,
- objective specification,
- covariance/risk abstractions,
- historical data access.

Syntehtic alpha/return generation, TC computation, and Table 2 aggregation should remain in project-specific code.

---

## 7. Proposed Project Architecture

The project should be modular from the start so the paper-replication workflow is testable and extensible.

### 7.1 Proposed file layout

```text
TransferCoef/
├── DESIGN.md
├── notebooks/
├── outputs/
│   ├── logs/
│   └── tables/
├── src/
│   └── transfercoef/
│       ├── __init__.py
│       ├── alpha_model.py
│       ├── cli.py
│       ├── config.py
│       ├── covariance.py
│       ├── cvxportfolio_adapter.py
│       ├── data_loader.py
│       ├── diagnostics.py
│       ├── monte_carlo.py
│       ├── optimizer.py
│       └── table2.py
└── tests/
    ├── test_alpha_model.py
    ├── test_diagnostics.py
    ├── test_optimizer.py
    └── test_table2.py

```

This structure keeps domain logic separate from experiments and outputs.

---

### 7.2 Module responsibilities

#### `config.py`
defines all simulation and scenario parameters.

Responsibilities:

- random seeds,
- number of assets,
- number of trials,
- target IC,
- breadth assumption,
- risk aversion,
- scenario definitions,
- covariance model choice,
- output formatting settings.

#### `cvxportfolio_adapter.py`
Handles local import/bootstrap of the local source tree.

Responsibilities:

- resolve local source path,
- safely import `cvxportfolio`,
- expose helper constractors for optimization/data classes,
- keep local path logic out of business modules.

#### `data_loader.py`
Historical calibration utilities.

Responsibilities:

- load Yahoo Finance data via local `cvxportfolio` ,
- produce returns/prices/volumes dataFrames,
- estimate sample covariance inputs,
- define research universes.

#### `alpha_model.py`
Syntehtic alpha generation.

Responsibilities:

- generate latent alpha,
- generate forecast alpha with target IC,
- optionally generate factor-driven alphas,
- validate realized signal quality.

#### `covariance.py`
Covariance and residual risk model utilities.

Responsibilities:

- construct diagnonal covariance,
- build factor + residual covariance,
- calibrate covariance from historical data,
- provide inverse variance weights for TC.

#### `optimizer.py`
Scenario optimization engine.

Responsibilities:

- solve unconstrained portfolio,
- solve constrained portfolios,
- define scenario constraint builders,
- normalize and validate resulting weights.

#### `diagnostics.py`
Statistical metrics and aggregation.

Responsibilities:

- compute IC,
- compute TC (plain and risk-weighted),
- compute active return/risk/IR,
- summarize trials,
- assemble scenario comparison tables.

#### `monte_carlo.py`
Simulation orchestrator.

Responsibilities:

- run repeated simulation trials,
- call alpha/covariance/optimizer/diagnostic components,
- store long-format trial results,
- support reproducibility and logging.

#### `table2.py`
Formatting and export layer.

Responsibilities:

- map aggregated diagnostics into a Table 2-like layout,
- export CSV / Excel / console tables,
- optionally compare against paper target values.

#### `cli.py`
Entry point for reproducible runs.

Responsibilities:

- parse scenario/config arguments,
- run simulation,
- export outputs.

---

## 8. Scenario Desgin for Table 2

The exect scenarios should remain configurable, but the default design should include at least the following.

### Scenario 0: Reference unconstrained

Purpose:

- obtain the unconstrained benchmark portfolio,
- define TC = 1.0 by construction if comparing against itself.

Suggested constraints:

- minimal feasibility only,
- optional leverage normalization.

### Scenario 1: Long-only

Purpose:

- isolate the effect of prohibiting short positions.

Suggested constraint set:

- `LongOnly()`
- `LeverageLimit(1.0)`

### Scenario 2: Long-only + max position

Purpose:

- capture crowding / diversification constraints.

Suggested constraint set:

- `LongOnly()`
- `LeverageLimit(1.0)`
- `MaxWeights(max_weight)`

### Scenario 3: Turnover-limited

Purpose:

- capture implementation friction and slower signal transfer.

Suggested constraint set:

- `TurnoverLimit(turnover_limit)`
- possibly benchmark or leverage controls.

### Scenario 4: Combined realistic constraints

Purpose:

- approximate a practical institutional portfolio.

Suggested constraint set:

- `LongOnly()`
- `LeverageLimit(1.0)`
- `MaxWeights(max_weight)`
- `TurnoverLimit(turnover_limit)`

optional advanced scenarios can add:

- `DollarNeutral()`
- minimum weights / holdings,
- benchmark deviation constraints,
- factor neutrality.

---

## 9. Key Design Decisions

### 9.1 Use synthetic simulation as the primary relication engine

Rationale:

- Table 2 is fundamentally Monte Carlo-based.
- Synthetic data gives control over IC, covariance, and constraints.
- It makes the methodology explicit and testable.

### 9.2 Use cvxportfolio mainly for optimization and optional calibration

Rationale:

- This matches the user's request.
- It avoids overfitting the project design to full historical backtesting.
- It keeps the project modular if another optimizer is later desired.

### 9.3 Support both pure NumPy and cvxportfolio-backed optimization paths

Rationale:

- A pure mathematical fallback is useful for unit tests and debugging.
- A cvxportfolio path ensures alignment with the desired toolchain.

Implementation preference for first build:

- primary path: cvxportfolio-backed optimization,
- fallback path: simple closed form / direct optimization utilities where possible.

### 9.4 keep diagnostics independent of optimizer implementation

Rationale: 

- IC, TC, IR formulas should be reusable regardless of how weights are generated.
- This reduces coupling and improves testability.

---

## 10. Validation Strategy

The implementation must include multiple layers of validation.

### 10.1 mathematical sanity checks

- If constrained portfolio equals unconstrained portfolio, TC should be ~1.
- If forecast alpha is independent of true alpha, ex post IC should average near 0.
- If constraints are tightened, TC should generally weakly decrease.
- Realized IR should broadly track theoretical $TC \times IC \times \sqrt{BR}$.

### 10.2 Unit tests

Test deterministic micro-cases for:

- correlation functions,
- TC formulas,
- risk-weight calculations,
- scenario builders,
- aggregation logic.

### 10.3 Simulation regression tests

Use small fixed-seed runs to validate:

- output schema,
- scenario ordering,
- monotonicity expectations,
- reproducibility.

### 10.4 Historical calibration smoke tests

When the execution environment is ready:

- verify `YahooFinance` downloads data through local `cvxportfolio`,
- verify returns/prices/volumes structures,
- verify covariance estimation and universe selection.

---

## 11. Risks and Open Questions

### 11.1 Exact Table 2 conventions may require later calibration

The paper's exact row definitions, breadth conventions, and weighting choices may need final refinement after a deeper extraction of the PDF content in the target execution environment.

Mitigation:

- make all diagnostics configurable,
- separate computation from presentation,
- preserve trial-level outputs for re-aggregation.

### 11.2 Turnover constraints require previous holdings state

Turnover-limited scenarios are path-dependent.

Mitigation:

- define whether each trial represents a single rebalance or a multi-step path,
- for initial implementation, support both:
  - one-step trial with supplied prior weights,
  - multi-period trial with rolling holdings.

### 11.3 Breadth is not uniquely defined operationally

Mitigation:

- parameterize breadth,
- document chosen convention in outputs,
- allow multiple reported theoretical IR variants if useful.

### 11.4 Local source import issues may vary by environment

Mitigation:

- isolate local import logic in one adapter module,
- avoid hidden path assumptions throughout the codebase.

---

## 12. Detailed Implementation Plan

This plan is intentionally sequenced so you can review and approve each stage before code complexity increases.

### Phase 1: Project scaffolding

1. create source package layout,
2. add local `cvxportfolio` adapter,
3. add configuration objects/dataclasses,
4. define scenario schema and output schema.

Deliverables:

- importable package skeleton,
- scenario registry,
- reproducible config model.

### Phase 2: Diagnostics core

1. implement correlation helpers,
2. implement IC computation,
3. implement plain TC computation,
4. implement risk-weighted TC computation,
5. implement active return, active risk, IR,
6. add unit tests for deterministic examples.

Deliverables:

- `diagnostics.py`,
- iniitial test coverage for formulas.

### Phase 3: Synthetic alpha and covariance engines

1. implement latent alpha generation,
2. implement target-IC forecast generation,
3. implement diagonal covariance generator,
4. implement optional factor covariance generator,
5. add validation helpers for realized IC calibration.

Deliverables:

- `alpha_model.py`,
- `covariance.py`,
- tests for generated moments and correlations.

### Phase 4: Optimization layer

1. implement unconstrained optimizer path,
2. implement cvxportfolio-backed constrained optimizer,
3. define scenario builders using local constraints,
4. validate weight normalization and feasibility,
5. add tests for scenario behavior.

Deliverables:

- `optimizer.py`,
- scenario definitions,
- optimizer tests.

### Phase 5: Monte Carlo runner

1. define trial result record schema,
2. implement single-trial pipeline,
3. implement many-trial ochestration,
4. aggregate by scenario,
5. persist long-format results.

Deliverables:

- `monte_carlo.py`,
- reproducible simulation engine.

### Phase 6: Table 2 reporting

1. define final symmary table layout,
2. compute theoretical vs realized diagnostics,
3. export CSB and pretty console view,
4. optionally add Excel export,
5. compare outputs to paper expectations.

Deliverables:

- `table2.py`,
- summary outputs in `outputs/tables/`.

### Phase 7: Histoical calibration integration

1. load Yahoo Finance data via local `cvxportfolio`,
2. create a calibration utility for volatility/covariance inputs,
3. support universe construction from real tickers,
4. run a calibration-backed simulation example.

Deliverables:

- `data_loader.py`
- optional real-data-assisted configuration model.

### Phase 8: CLI and workflow polish

1. add command-line entry point,
2. add config file support if needed,
3. add logging and run metadata,
4. finalize documentation and examples.

deliverables:

- `cli.py`
- reproducible command-driven workflow.

---

## 13. Implementation TODO Checklist

This checklist is intended for sequential execution after design approval.

### Step 1: Design approval

- [ ] Create comprehensive design document
- [ ] Review methodology assumptions for Table 2
- [ ] Confirm default breadth convention
- [ ] Confirm default scenario set
- [ ] Confirm whether turnover should be single-step or multi-period in v1

### Step 2: Repository Scaffolding

- [ ] Create `src/transfercoef/` package
- [ ] Create `tests/` package
- [ ] Create `outputs/tables/` and `outputs/logs/`
- [ ] Add package `__init__.py`
- [ ] Add base configuration module

### Step 3: Local cvxportfolio integration

- [ ] Create adapter for local `cvxportfolio-master/cvxportfolio-master`
- [ ] Validate import path resolution
- [ ] Expose helper constructors for data and optimization objects
- [ ] Add a smoke-test import check

### Step 4: Diagnostics implementation

- [ ] Implement safe Pearson correlation helper
- [ ] Implement Spearman option if desired
- [ ] Implement ex post IC calculation
- [ ] Implement plain TC calculation
- [ ] Implement risk-weighted TC calculation
- [ ] Implement active return calculation
- [ ] Implement active risk calculation
- [ ] Implement ex post IR calculation
- [ ] Add diagnostics unit tests

### Step 5: Synthetic model implementation

- [ ] Implement latent alpha generator
- [ ] Implement forecast alpha generator with target IC
- [ ] Implement diagonal covariance generator
- [ ] Implement factor covariance option
- [ ] Add reproducibility via random seed control
- [ ] Add distribution/correlation unit tests

### Step 6: Optimization implementation

- [ ] Implement unconstrained weight solver
- [ ] Implement constrained solver using local `cvxportfolio`
- [ ] Implement scenario builder for long-only
- [ ] Implement scenario builder for max-weight constraint
- [ ] Implement scenario builder for turnover limit
- [ ] Implement combined-constraint scenario
- [ ] Add optimizer unit tests

### Step 7: Monte Carlo engine

- [ ] Define trial result dataclass / schema
- [ ] Implement one-trial execution funciton
- [ ] Implement scenario loop per trial
- [ ] Implement many-trial execution function
- [ ] Store long-format trial results in DataFrame
- [ ] Add reproducibility and seed logging

### Step 8: Table 2 reporting

- [ ] Implement aggregated summary statistics
- [ ] Implement theoretical IR calculation variants
- [ ] Format Table 2-style output
- [ ] Export summary to CSV
- [ ] Export trial-level data to CSV
- [ ] Add reporting tests

### Step 9: Historical calibration mode

- [ ] Load Yahoo Finance data through local `cvxportfolio`
- [ ] Build historical return matrix utility
- [ ] Build covariance calibration utility
- [ ] Build universe selection helper
- [ ] Add calibration smoke tests

### Step 10: CLI and examples

- [ ] Implement CLI entry point
- [ ] Add example configuration presets
- [ ] Add example command for synthetic simulation
- [ ] Add example command for calibration-backed run
- [ ] Finalize usage notes

---

## 12. Recommended First Coding Sequence

After this desing is approved, the most efficient coding order is:

1. `cvxportfolio_adapter.py`
2. `config.py`
3. `diagnostics.py`
4. `alpha_model.py`
5. `covariance.py`
6. `optimizer.py`
7. `monte_carlo.py`
8. `table2.py`
9. `data_loader.py`
10. `cli.py`

Reason:

- diagnostics and stochastic model pieces are foundational,
- optimizer depends on scneario/config definitions,
- the Monte Carlo runner depends on all prior components,
- reporting should be built after trial outputs stabilize.

---

## 15. Review Items for Approval

Before implementation starts, the following decisions should be explicitly approved:

1. **Primary TC definition**
    - plain Pearson correlation,
    - risk-weighted correlation,
    - or report both.

2. **Breadth convention**
    - user-specified fixed scalar,
    - asset-count proxy,   
    - effective breadth proxy.

3. **Scenario set for Table 2 v1**
    - minimal four-scenario version,
    - or expanded realistic version.
    

4. **Turnover model**
    - single-step approximation,
    - or fully path-dependent multi-period simulation.

5. **Calibration mode timing**
    - build synthetic-only first,
    - or include Yahoo Finance calibration in the first implementation pass.

---

## 16. Recommendation

Recommended implementation strategy for version 1:

- start with **synthetic Month Carlo + cvxportfolio-backed optimization**,
- report **both plain and risk-weighted TC**,
- use **configurable scalar breadth** first,
- include **four default scenarios** (unconstrained, long-only, long-only+max-weight, combined constraints),
- defer richer turnover path dynamics and historical calibration refinements to the next phase.

This approac gives the cleanest path to a credible and testable Table 2 replication while staying aligned with the paper and the local `cvxportfolio` toolchain.