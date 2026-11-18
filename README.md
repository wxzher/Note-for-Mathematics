# Mathematics Notes

Graduate-level mathematics course notes, homework, and experiments — **Advanced Mathematics**, **Linear Algebra**, **Probability & Statistics**, **Operations Research**, and **Numerical Optimization**.

**Student**: Xiaozhen Wu (巫小珍) — 2024201030025

---

## Advanced Mathematics（高等数学）

12 chapters of calculus, following 同济大学《高等数学》.

| Chapter | Content |
|:-------:|---------|
| 01 | Functions & Limits — function properties, limit definitions, infinitesimals, continuity |
| 02 | Derivatives & Differentials — derivative rules, implicit differentiation, parametric equations |
| 03 | Mean Value Theorem & Applications — Rolle, Lagrange, Cauchy, L'Hôpital, Taylor expansion |
| 04 | Indefinite Integrals — substitution, integration by parts, rational functions |
| 05 | Definite Integrals — Riemann sums, fundamental theorem, improper integrals |
| 06 | Applications of Definite Integrals — area, volume, arc length, surface of revolution |
| 07 | Differential Equations — separable, linear, homogeneous, higher-order ODEs |
| 08 | Spatial Analytic Geometry & Vector Algebra — vectors, planes, lines, surfaces, space curves |
| 09 | Multivariable Differential Calculus — partial derivatives, total differentials, extrema and applications |
| 10 | Double Integrals — Cartesian and polar coordinates, change of order, applications |
| 11 | Line & Surface Integrals — scalar/vector integrals, Green, Gauss and Stokes formulas |
| 12 | Infinite Series — numerical series, power series, Fourier series |

**Supplementary**: Bernoulli's inequality.  
**Practice**: 100 limits problems, 100 derivative problems, 100 integral problems, formula handbooks (derivatives & integrals), comprehensive review guide.

---

## Linear Algebra（线性代数）

5 chapters covering linear algebra fundamentals.

| Chapter | Content |
|:-------:|---------|
| 01 | Introduction — overview of linear algebra, systems of linear equations |
| 02 | Matrices — matrix operations, determinants, inverse, Cramer's rule |
| 03 | Vectors & Linear Equations — vector spaces, linear dependence/independence, rank, solution structure |
| 04 | Vector Spaces & Linear Transformations — basis, dimension, eigenvalues |
| 05 | Similar Matrices & Quadratic Forms — diagonalization, quadratic forms, positive definiteness |

---

## Probability & Statistics（概率论与数理统计）

Two sets of course notes covering probability theory and mathematical statistics, plus a comprehensive review note.

| Chapter | Content |
|:-------:|---------|
| 01 | Random Events & Probability — sample spaces, axioms, conditional probability, Bayes' theorem |
| 02 | Random Variables & Distributions — discrete/continuous distributions, PDF, CDF, expectation |
| 03 | Multidimensional Random Variables — joint/marginal/conditional distributions, covariance |
| 04 | Numerical Characteristics — expectation, variance, covariance, correlation coefficient |
| 05–06 | Limit Theorems & Mathematical Statistics — laws of large numbers, central limit theorem, sampling distributions |
| 07 | Parameter Estimation — point estimation, maximum likelihood, interval estimation |
| 08 | Hypothesis Testing — tests for means, variances and distributions |

**Note collections**:

- **Sheng Zhou / Zhejiang University edition**: Chapters 01–08, random-events handouts, common discrete and continuous distributions reference, and chi-square test
- **Lian Baosheng edition**: 4 chapters covering random events and conditional probability, random variables and distributions, expectation and variance, statistics and parameter estimation
- **Comprehensive review**: consolidated probability and statistics review notes

---

## Operations Research（运筹学）

8 lectures covering fundamental methods of operations research.

| Lecture | Content |
|:-------:|---------|
| 01 | Graphical Method & Simplex — linear programming formulation, simplex tableau algorithm |
| 02 | Duality — dual problem, weak/strong duality, complementary slackness, shadow prices |
| 03 | Transportation Problem — balanced/unbalanced transportation, northwest corner, Vogel's method |
| 04 | Integer Programming — branch and bound, cutting plane, 0-1 programming |
| 05 | Goal Programming — multi-objective optimization, priority factors, deviation variables |
| 06 | Graph Theory — Euler/Hamilton paths, minimum spanning tree, shortest path, maximum flow |
| 07 | Dynamic Programming — Bellman's principle, stage/state/decision, backward induction |
| 08 | Game Theory — zero-sum games, saddle point, mixed strategies, Nash equilibrium |

**Homework**: simplex method (HW1), graph algorithms (HW10), dynamic programming (HW11), game theory (HW12).

---

## Numerical Optimization（数值最优化）

Following **Chen Baolin's *Optimization Theory and Algorithms*** (2nd ed.), 11 lecture notes across selected chapters, 5 homework assignments, and 2 computational experiments.

| Chapter | Content |
|:-------:|---------|
| 01 | Prerequisites — mean value theorem, Taylor expansion, Cauchy-Schwarz, convex sets & functions |
| 07 | Optimality Conditions — KKT conditions, constraint qualifications, first/second-order conditions |
| 08 | Algorithms — convergence rates, termination criteria, line search, trust region |
| 09 | One-Dimensional Search — golden section, Fibonacci, bisection, interpolation, Wolfe conditions |
| 10 | Unconstrained Optimization with Derivatives — steepest descent, Newton, quasi-Newton (BFGS/DFP), conjugate gradient |
| 12 | Feasible Direction Method — Zoutendijk's method, Rosen's gradient projection, reduced gradient |
| 13 | Penalty Function Method — external penalty (SUMT), internal penalty (barrier), augmented Lagrangian |
| 14 | Quadratic Programming — active set method, Lemke's method, Wolfe's method |

**Supplementary**: one-dimensional search without optimization constraints, mean value theorem & Taylor & Cauchy review.

**Homework** (5 assignments):
- HW1–HW2: Convex sets, optimality conditions, KKT systems
- HW3: One-dimensional search methods
- HW4: Gradient methods (steepest descent, Newton, quasi-Newton)
- HW5: Constrained optimization (penalty methods, QP)

**Experiments**:
- **Experiment 1 — Gradient Descent Methods**: steepest descent (exact/inexact), Newton, trust region on modified Rosenbrock function (2D & 4D)
- **Experiment 2 — Optimization Methods Comparison**: 10 methods × 10 test problems compared on iterations, function evaluations, convergence success rate

---

## References

- 同济大学数学系. *高等数学* (第7版). 高等教育出版社.
- 同济大学数学系. *线性代数* (第6版). 高等教育出版社.
- 盛骤 等. *概率论与数理统计* (第4版). 高等教育出版社.
- 胡运权 等. *运筹学教程*. 清华大学出版社.
- 陈宝林. (2005). *最优化理论与算法* (第2版). 清华大学出版社.
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
