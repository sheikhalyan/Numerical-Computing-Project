# 🔢 Application of Nonlinear Algebraic Equations in Mathematical Modeling

### Solving real-world physics, electronics, and biology problems using numerical root-finding methods

[![MATLAB](https://img.shields.io/badge/MATLAB-Numerical_Methods-0076A8?style=flat-square&logo=mathworks&logoColor=white)](https://www.mathworks.com/products/matlab.html)
[![Python](https://img.shields.io/badge/Python-Implementation-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)

> A Numerical Computing project demonstrating how nonlinear algebraic equations model real-world phenomena across physics, electronics, and population biology — solved using classical iterative root-finding methods in both MATLAB and Python.

---

## Overview

Nonlinear algebraic equations — where unknowns appear as variables raised to powers or inside nonlinear functions — are fundamental to modeling complex systems in science and engineering. Unlike linear equations, they rarely have clean analytical solutions, requiring iterative numerical methods to approximate roots.

This project implements and applies three classical root-finding methods to three distinct real-world problems, each drawn from a different domain.

---

## Numerical Methods Implemented

| Method | Description | Convergence |
|--------|-------------|-------------|
| **Newton's Method** | Iterative technique using linear approximation (tangent line) to converge on a root | Quadratic |
| **Bisection Method** | Repeatedly halves an interval, selecting the subinterval where the function changes sign | Linear |
| **Secant Method** | Iterative method approximating the derivative using two prior points, avoiding the need for an explicit derivative | Superlinear |

---

## Problems Solved

### 1. Logistic Growth Model (Biology)

Models how a population grows rapidly at first, then slows as it approaches the environment's carrying capacity.

```
dP/dt = r·P(1 − P/K)
```

- **P** — population size
- **r** — intrinsic growth rate
- **K** — carrying capacity

**Application:** Population dynamics in ecology — understanding how species grow and stabilize within resource-limited ecosystems.

---

### 2. Diode Current Equation (Electronics)

Models the nonlinear current-voltage relationship in a semiconductor diode — a foundational equation in circuit design.

```
I = I_S · (e^(V_D / nV_T) − 1)
```

- **I** — current through the diode
- **I_S** — reverse saturation current
- **V_D** — voltage across the diode
- **n** — ideality factor
- **V_T** — thermal voltage

**Application:** Essential for designing and analyzing electronic circuits involving diodes and semiconductor components.

---

### 3. Rectangular Field Problem (Applied Geometry)

Determines the dimensions of a rectangular field given its perimeter and area — a classic nonlinear system since the two equations together are non-linear in the unknowns.

```
P = 2(l + w)
A = l × w
```

- **P** — perimeter
- **A** — area
- **l, w** — length and width of the field

**Application:** Urban planning and agricultural land division — solving for dimensions when only aggregate measurements (perimeter, area) are known.

---

## Why Nonlinear Equations Matter

| Domain | Example Use Case |
|--------|-------------------|
| **Physics** | Modeling wave propagation in complex mediums — sound waves in air, seismic waves through ground |
| **Engineering** | Describing nonlinear behavior of electrical circuit components and their interactions |
| **Biology** | Modeling population growth and interaction dynamics within ecosystems |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Primary Implementation** | MATLAB |
| **Secondary Implementation** | Python |
| **Methods** | Newton's Method, Bisection Method, Secant Method |

---

## Project Structure

```
nonlinear-equations-modeling/
│
├── Matlab Files/
│   ├── logistic-growth-problem.m        # Population growth model solver
│   ├── Diode-Current-Problem.m          # Diode I-V equation solver
│   └── Rectangular-Field-Problem.m      # Perimeter/area dimension solver
│
├── Python-File/
│   └── app.py                            # Python implementation
│
├── docs/
│   ├── Application of Nonlinear Algebraic Equations in Mathematical Modeling_Report.pdf
│   ├── Application of Nonlinear Algebraic Equations in Mathematical Modeling_PPT.pptx
│   
│
└── README.md
```

---

## Getting Started

### MATLAB

1. Open MATLAB
2. Navigate to the `Matlab Files/` directory
3. Run any of the three `.m` scripts directly:
   ```matlab
   logistic-growth-problem
   Diode-Current-Problem
   Rectangular-Field-Problem
   ```

### Python

1. Navigate to `Python-File/`
2. Install dependencies (NumPy recommended for numerical operations)
   ```bash
   pip install numpy
   ```
3. Run the script
   ```bash
   python app.py
   ```

---

## Documentation

Full methodology, derivations, and results are available in the project report (`docs/Application of Nonlinear Algebraic Equations in Mathematical Modeling_Report.pdf`) and accompanying presentation.

---

## Context

Built as a **Numerical Computing** project at PAF-KIET, applying classical root-finding algorithms to real, physically meaningful nonlinear systems rather than abstract textbook equations.

---

## Author

**Sheikh Alyan** — BS Computer Science, PAF-KIET

[![GitHub](https://img.shields.io/badge/GitHub-@sheikhalyan-181717?style=flat-square&logo=github)](https://github.com/sheikhalyan)
