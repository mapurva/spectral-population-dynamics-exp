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
