# Spectral Population Dynamics (SPD)

This repository contains code and experiments for the paper:

**"Spectral Population Dynamics: Structural Determinants of Fixation Time Beyond Spectral Gap"**

---

## 📌 Overview

This project studies how graph structure affects genetic drift under the Moran process.

We investigate:

- Spectral gap (λ₂)
- Kirchhoff index (effective resistance)
- Fixation time dynamics on graphs

Key finding:

> No single structural metric explains fixation time.  
> A combined model using spectral gap and resistance performs significantly better.

---

## 📊 Main Results

| Model | R² |
|------|----|
| Spectral Gap (λ₂) | ~0.06 |
| Kirchhoff Index | ~0.30 |
| Combined Model | ~0.75 |

---

## 🧠 Methodology

- Population modeled as graph \( G = (V, E) \)
- Neutral Moran process simulated
- Fixation time estimated via repeated trials
- Structural metrics computed:
  - Spectral gap (λ₂)
  - Kirchhoff index

---

## 📁 Repository Structure
spectral-population-dynamics-exp/
│
├── src/
│ ├── graphs.py
│ ├── spectral.py
│ ├── simulation.py
│ ├── structure.py
│ └── resistance.py
│
├── experiments/
│ ├── run_experiments.py
│ └── analyze_results.py
│
├── data/
│ └── results_with_resistance.csv
│
├── figures/
│ ├── global_analysis.png
│ ├── per_graph_analysis.png
│ └── combined_model.png
│
├── requirements.txt
└── README.md


---

## ⚙️ Installation

Create environment (recommended: conda):

```bash
conda create -n spd python=3.10
conda activate spd
pip install -r requirements.txt

▶️ Running Experiments
Step 1: Generate data
python -m experiments.run_experiments

This will create:

data/results_with_resistance.csv
Step 2: Analyze results and generate figures
python -m experiments.analyze_results

This will generate:

figures/
├── global_analysis.png
├── per_graph_analysis.png
├── combined_model.png

📊 Graph Types Used
Cycle graph
Path graph
Complete graph
Star graph
Barbell graph
Erdős–Rényi random graph

🔬 Parameters
Graph sizes: n = 20, 30, 50, 80, 120
Trials per configuration: 200
Maximum steps: 50,000

🔁 Reproducibility
All experiments use fixed parameters
Results can be regenerated using provided scripts
No external datasets required
