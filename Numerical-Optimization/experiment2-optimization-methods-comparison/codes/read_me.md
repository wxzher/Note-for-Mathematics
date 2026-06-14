# 数值优化算法实验：ALM 与 SQP 对比研究

## 1. 项目简介

本项目旨在通过数值实验，对比研究两种经典的约束优化算法：**增广拉格朗日乘子法 (Augmented Lagrangian Method, ALM)** 与 **序列二次规划法 (Sequential Quadratic Programming, SQP)**。

实验基于 Python 实现，选取了具有代表性的二维和四维非线性约束优化问题（Modified Rosenbrock），考察了算法在不同初始点（包括可行域内、外、边界及鞍点附近）下的收敛性、鲁棒性及计算效率。

## 2. 文件结构

```text
Project_Root/
├── run.py                 # 【入口程序】定义问题、执行实验、保存数据
├── solver.py              # 【核心算法】包含 ALMSolver 和 SQPSolver 类实现
├── plot_utils.py          # 【绘图工具】包含轨迹图、收敛曲线、柱状图绘制函数
├── optimization_results_new.csv  # (运行后生成) 实验详细数据报表
├── README.md              # 项目说明文档
└── figures/               # (运行后生成) 存放所有可视化结果图片
    ├── path_2D_P1.png     # 2D 问题轨迹图
    ├── conv_dual_...png   # 收敛曲线对比图
    └── summary_...png     # 性能统计柱状图
```

## 3. 环境依赖

本项目代码基于 Python 3 编写，需要安装以下科学计算库：

```bash
pip install numpy scipy matplotlib pandas seaborn
```

## 4. 快速开始

1. 确保 `solver.py`, `plot_utils.py` 和 `run.py` 在同一文件夹下。
2. 在终端或 IDE 中运行 `run.py`：

```bash
python run.py
```

3. 程序将自动执行以下操作：
   * **求解**：针对 2D 和 4D 问题，分别使用 ALM 和 SQP 算法从 10 个不同的初始点进行求解。
   * **保存**：将包含迭代步数、最终目标函数值、梯度模长、约束违反度等信息的表格保存为 `optimization_results_new.csv`。
   * **绘图**：在 `figures/` 文件夹下生成轨迹图、收敛曲线及性能对比图。

## 5. 问题定义

### 5.1 二维问题 (2D Modified Rosenbrock)

具有典型的香蕉形山谷，并增加了正弦扰动项，约束包含非线性等式和不等式。

* **目标函数**: $f(x) = (1-x_1)^2 + 100(x_2-x_1^2)^2 + 5\sin(2\pi x_1)\sin(2\pi x_2)$
* **等式约束**: $x_1^2 + x_2 - 2 = 0$
* **不等式约束**: $x_1 + x_2^2 - 3 \le 0$

### 5.2 四维问题 (4D Chain Rosenbrock)

高维链式结构，约束更加复杂。

* **目标函数**: 链式 Rosenbrock 函数
* **约束**: 包含线性超平面约束 ($ \sum x_i = 2 $) 和多个非线性不等式约束。

## 6. 算法实现细节

### ALMSolver (增广拉格朗日法)

* **框架**: PHR (Powell-Hestenes-Rockafellar) 乘子法。
* **子问题求解**: 使用 BFGS 拟牛顿法求解无约束子问题。
* **特色**: 实现了增广拉格朗日函数的解析梯度 (`aug_grad`)，显著提升了子问题求解的稳定性和精度。
* **更新策略**: 动态调整罚参数 $\mu$，当约束违反度下降不足时增大惩罚。

### SQPSolver (序列二次规划)

* **核心**: 在每一步通过求解 QP 子问题确定搜索方向 $d_k$。
* **Hessian 近似**: 使用 BFGS 公式更新拉格朗日函数的 Hessian 矩阵近似 $B_k$。
* **线搜索**: 采用 $\ell_1$ Merit Function 结合 Armijo 准则保证全局收敛性。
* **QP 求解**: 调用 `scipy.optimize.minimize(method='SLSQP')` 求解线性化后的子问题。

## 7. 结果可视化说明

运行结束后，`figures/` 文件夹将包含：

1. **`path_2D_PX.png`**: 2D 问题的优化轨迹。
   * 背景为目标函数等高线。
   * 红色实线为等式约束，蓝色虚线为不等式约束边界。
   * 黑色路径为 ALM，红色路径为 SQP。
2. **`conv_dual_...png`**: 收敛过程详图。
   * **左图**: 综合残差 ($\|\nabla f\| + \|c(x)\|$) 随迭代次数的变化。
   * **右图**: 分项展示“约束违反度”和“梯度范数”的下降趋势。
3. **`perf_bar_...png`**: 性能对比柱状图。
   * 对比两种算法在不同初始点下的总迭代步数。

## 8. 实验观察 (预期)

* **SQP** 通常表现出更快的收敛速度（超线性收敛），迭代步数较少，但在非凸区域或初始点较远时可能受 Hessian 近似准确性的影响。
* **ALM** 更加鲁棒，对初始点敏感度较低，但由于需要反复求解子问题，总计算量通常大于 SQP。

```

```
