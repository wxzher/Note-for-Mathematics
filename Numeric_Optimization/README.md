# Numerical Optimization — Course Notes, Homework & Experiments

Course materials from a **Numerical Optimization (数值最优化)** graduate course.  
Covers optimization theory and algorithms: optimality conditions, unconstrained/constrained optimization, gradient-based methods, penalty functions, and quadratic programming.

**Student**: Xiaozhen Wu (巫小珍) — 2024201030025

---

## Repository Structure

```
.
├── README.md
│
├── notes/                                          # Lecture notes (PDF)
│   ├── ch01-prerequisites.pdf                      # Ch.1  Prerequisites
│   ├── ch07-optimality-conditions.pdf              # Ch.7  Optimality Conditions
│   ├── ch08-algorithms.pdf                         # Ch.8  Algorithms Overview
│   ├── ch09-one-dimensional-search.pdf             # Ch.9  One-Dimensional Search
│   ├── ch10-unconstrained-optimization-with-derivatives.pdf  # Ch.10 Unconstrained Optimization (Gradient Methods)
│   ├── ch12-feasible-direction-method.pdf          # Ch.12 Feasible Direction Method
│   ├── ch13-penalty-function-method.pdf            # Ch.13 Penalty Function Method
│   ├── ch14-quadratic-programming.pdf              # Ch.14 Quadratic Programming
│   ├── one-dimensional-search-without-optimization.pdf
│   ├── mean-value-theorem-taylor-cauchy.pdf
│   └── textbook-optimization-theory-and-algorithms-2nd-edition.pdf  # Chen Baolin textbook
│
├── homework/                                       # Homework assignments (PDF)
│   ├── hw01.pdf
│   ├── hw02.pdf
│   ├── hw03.pdf
│   ├── hw04.pdf
│   └── hw05.pdf
│
├── experiment1-gradient-descent/                   # Experiment 1: Gradient Descent Methods
│   ├── assignment.pdf                              # Assignment description
│   ├── report.pdf / report.tex                     # Submitted report
│   ├── experiment.py                               # Main optimization code
│   ├── show.ipynb                                  # Jupyter notebook demo
│   ├── submission/ / submission.zip                # Final submission
│   └── whu-latex-template/                         # WHU LaTeX report template
│
└── experiment2-optimization-methods-comparison/    # Experiment 2: Methods Comparison
    ├── assignment.pdf                              # Assignment description
    ├── report-submitted.pdf                        # Submitted report
    ├── report.tex                                  # Report source
    ├── codes/                                      # Organized implementation
    │   ├── run.py / solver.py / plot_utils.py
    │   ├── optimization_results.csv
    │   └── figures/                                # 50+ convergence figures
    ├── figures/                                    # Summary figures (2D & 4D)
    └── optimization_results.csv
```

## Course Content Overview

The course follows **Chen Baolin's *Optimization Theory and Algorithms* (2nd ed.)**, a standard Chinese graduate-level textbook.

| Chapter | Topic | Key Concepts |
|:-------:|-------|-------------|
| 1 | Prerequisites | Mean value theorem, Taylor expansion, Cauchy-Schwarz inequality, convex sets & functions |
| 7 | Optimality Conditions | KKT conditions, constraint qualifications, first/second-order necessary/sufficient conditions |
| 8 | Algorithms | Convergence rates, termination criteria, line search, trust region |
| 9 | One-Dimensional Search | Golden section, Fibonacci, bisection, quadratic/cubic interpolation, Wolfe conditions |
| 10 | Gradient-Based Unconstrained Methods | Steepest descent, Newton's method, quasi-Newton (BFGS, DFP), conjugate gradient |
| 12 | Feasible Direction Method | Zoutendijk's method, Rosen's gradient projection, reduced gradient |
| 13 | Penalty Function Method | External penalty (SUMT), internal penalty (barrier), augmented Lagrangian |
| 14 | Quadratic Programming | Active set method, Lemke's method, Wolfe's method |

## Homework

Five handwritten/scanned homework assignments covering:
- **HW1**: Convex sets, convex functions, Taylor expansions
- **HW2**: Optimality conditions, KKT systems
- **HW3**: One-dimensional search methods (golden section, interpolation)
- **HW4**: Gradient methods (steepest descent, Newton, quasi-Newton)
- **HW5**: Constrained optimization (penalty methods, QP)

## Experiments

### Experiment 1: Gradient Descent Methods
Implements and compares optimization algorithms on a **Modified Rosenbrock function** (2D) and a **4D extension**:
- Steepest descent with exact line search
- Steepest descent with inexact line search (Armijo / Wolfe conditions)
- Newton's method
- Trust region method
- Outputs: convergence plots, optimization paths, performance tables

**Key files**: `experiment.py`, `show.ipynb`, `report.pdf`

### Experiment 2: Optimization Methods Comparison
Systematic comparison of **10 optimization methods** × **10 test problems** in 2D and 4D:
- Methods: Gradient descent (exact / inexact), Newton, BFGS, DFP, CG (FR / PRP), Trust region, etc.
- Problems: Rosenbrock, Beale, Booth, Himmelblau, Rastrigin, Sphere, etc.
- Performance metrics: iterations, function evaluations, convergence success rate
- Outputs: 50+ convergence plots, summary bar charts, path visualizations

**Key files**: `codes/run.py`, `codes/solver.py`, optimized `report.tex`

## Additional Notes

| File | Content |
|------|---------|
| `notes/one-dimensional-search-without-optimization.pdf` | Supplementary: 1D search methods without optimization constraints |
| `notes/mean-value-theorem-taylor-cauchy.pdf` | Supplementary: Mathematical foundations review |
| `notes/textbook-optimization-theory-and-algorithms-2nd-edition.pdf` | Full textbook: 陈宝林《最优化理论与算法》第2版 |

## References

- 陈宝林. (2005). *最优化理论与算法* (第2版). 清华大学出版社.
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
