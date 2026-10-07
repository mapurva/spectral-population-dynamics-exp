# Spectral Population Dynamics: Structural Dependence of Fixation Time Beyond Spectral Gap

This repository contains the reproducible simulation and analysis code for the research study:

**"Spectral Population Dynamics: Structural Dependence of Fixation Time Beyond Spectral Gap"**

The study investigates how graph topology and structural descriptors influence fixation time under a neutral invasion process, with particular emphasis on whether the spectral gap alone is sufficient to characterize fixation dynamics.

---

## Overview

The project evaluates the relationship between fixation time and several structural properties of graphs:

- Spectral gap (λ₂)
- Kirchhoff index K(G)
- Diameter
- Average shortest-path length
- Degree variance

The analysis considers both pooled and family-specific relationships.

A central empirical result is that the spectral gap alone provides weak explanatory power, while the Kirchhoff index provides substantially stronger explanatory power. A combined log-linear model using the Kirchhoff index and spectral gap explains substantially more configuration-level variation.

The analysis also demonstrates that the relationship between structural descriptors and fixation time is topology-dependent, motivating caution against treating a single global structural metric as a universal predictor.

---

## Main Results

The primary pooled models use the logarithmic response:

    log(T)

where T is the configuration-level mean fixation time among successful trials.

| Model | R² |
|---|---:|
| Spectral gap (λ₂) | 0.071 |
| Kirchhoff index K(G) | 0.419 |
| K(G) + λ₂ | 0.660 |

Five-fold cross-validation for the structural model including K(G), λ₂, and log-transformed diameter gives:

    R² = 0.601

A richer model including degree variance gives:

    R² = 0.761

However, leave-one-family-out validation shows substantial degradation for several held-out graph families. This indicates that the pooled relationships should not be interpreted as a topology-independent universal law.

---

## Methodology

The population is represented by a connected graph:

    G = (V, E)

The study simulates a neutral invasion process on the graph.

For each graph configuration:

- A graph is generated using a deterministic configuration seed.
- A single mutant is introduced.
- The neutral invasion process is simulated until fixation or until the maximum simulation time is reached.
- Multiple independent trials are performed.
- Fixation-time statistics and graph-structural descriptors are recorded.

The simulation uses right censoring when fixation has not occurred by the maximum allowed number of steps.

### Graph families

Six graph families are considered:

1. Cycle
2. Complete graph
3. Erdős–Rényi random graph
4. Path
5. Star
6. Barbell

### Graph sizes

    n = 20, 30, 50, 80, 120

This gives:

    6 graph families × 5 sizes = 30 configurations

### Simulation parameters

- Trials per configuration: 500
- Total trials: 15,000
- Maximum simulation steps: 50,000
- Master seed: 20260928

No external datasets are required.

---

## Treatment of Censoring

A simulation is considered censored if fixation has not occurred within 50,000 steps.

The main reported quantity, `mean_fixation`, is the mean fixation time among successful trials only.

The number of censored trials is recorded separately in the `censored` column.

Censoring is not assumed to be negligible. A separate sensitivity analysis evaluates the effect of treating censored observations as occurring at the simulation limit.

---

## Structural Metrics

The following structural descriptors are computed for each graph configuration:

- Spectral gap λ₂
- Kirchhoff index K(G)
- Diameter
- Average shortest-path length
- Degree variance

For a connected graph, the Kirchhoff index is computed from the non-zero Laplacian eigenvalues as:

    K(G) = n Σ(1 / λᵢ),  i = 2,...,n

The project also contains an analysis of the weighted resistance metric associated with the invasion-process backward lineage. This provides a theoretical upper-bound framework distinct from the empirical use of the ordinary Kirchhoff index.

---

## Repository Structure

```text
spectral-population-dynamics-exp/
│
├── src/
│   ├── graphs.py
│   ├── moran.py
│   ├── resistance.py
│   ├── seeding.py
│   └── simulation.py
│
├── experiments/
│   ├── run_experiments.py
│   ├── validate_results.py
│   ├── smoke_test.py
│   ├── analyze_revision.py
│   ├── analyze_revision_diagnostics.py
│   ├── analyze_censoring_sensitivity.py
│   ├── analyze_dual_resistance.py
│   ├── analyze_structural_regimes.py
│   ├── analyze_within_family.py
│   ├── plot_observed_vs_fitted.py
│   └── plot_structural_regimes.py
│
├── data/
│   ├── results_revision.csv
│   ├── revision_diagnostics/
│   ├── revision_dual_metric/
│   ├── revision_structural_analysis/
│   └── revision_within_family/
│
├── tests/
│
├── figures/
│
├── environment.yml
├── requirements.txt
├── requirements-py39.txt
├── pytest.ini
├── .gitignore
└── README.md


---

## Installation

Python 3.9 is recommended.

### Using a virtual environment

Create and activate a virtual environment:

```bash
python -m venv .venv
On Windows:
.venv\Scripts\activate

Install the dependencies:
pip install -r requirements.txt

For the exact tested Python 3.9 dependency versions:
pip install -r requirements-py39.txt

Running the Experiments
The complete experiment consists of 30 graph configurations with 500 trials per configuration.
Run:
python experiments/run_experiments.py

The primary results are written to:
data/results_revision.csv

The experiment uses deterministic graph and simulation seeds derived from the master seed.
Validating the Results
After generating the results, run:
python experiments/validate_results.py

The validation checks the expected configuration count, trial count, censoring accounting, simulation limits, and reproducibility-related fields.
Running the Analysis
Main revision analysis
python experiments/analyze_revision.py

Revision diagnostics
python experiments/analyze_revision_diagnostics.py

Censoring sensitivity analysis
python experiments/analyze_censoring_sensitivity.py

Dual resistance analysis
python experiments/analyze_dual_resistance.py

Structural-regime analysis
python experiments/analyze_structural_regimes.py

Within-family analysis
python experiments/analyze_within_family.py

Generate observed-versus-fitted figure
python experiments/plot_observed_vs_fitted.py

Generate structural-regime figures
python experiments/plot_structural_regimes.py

Testing
Run the complete test suite with:
python -m pytest -q

The current revision passes all tests:
18 passed

Reproducibility
The principal reproducibility parameters are:
Parameter	Value
Graph families	6
Graph sizes	20, 30, 50, 80, 120
Configurations	30
Trials/configuration	500
Total trials	15,000
Maximum steps	50,000
Master seed	20260928


The primary generated dataset is:
data/results_revision.csv

The repository contains the code required to regenerate the simulation results and reproduce the reported analyses.
Interpretation of the Results
The results should not be interpreted as establishing a universal deterministic mapping from spectral gap or Kirchhoff index to fixation time.
The experiments indicate:
1. Spectral gap alone has weak explanatory power for the investigated configurations.
2. The Kirchhoff index provides substantially stronger explanatory power than spectral gap alone.
3. Combining the Kirchhoff index and spectral gap improves pooled explanatory performance.
4. Additional structural descriptors can further improve cross-validated performance.
5. Leave-one-family-out validation reveals substantial topology dependence.
6. Therefore, structural metrics should be interpreted together with graph topology rather than as universally interchangeable predictors.
The theoretical analysis provides a weighted effective-resistance framework for the neutral invasion process. This theoretical metric is used to establish an upper-bound perspective and is distinguished from the ordinary Kirchhoff index used in the empirical analysis.