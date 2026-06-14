# @coding=utf-8
# @Author: Wuxiaozhen
# @File: run.py
# @Date: 2026-01-05
# @Description: 数值优化实验二主程序代码，包含问题定义、初始点生成、实验执行与结果收集

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, BFGS
import pandas as pd
import warnings
import os
# 基础设置
if not os.path.exists('figures'): os.makedirs('figures')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False

from solver import ALMSolver, SQPSolver
from plot_utils import (plot_convergence_individual, plot_2d_trajectory_individual,
                       plot_convergence_refined,plot_convergence_grid,
                        plot_performance_with_avg, plot_all_problems_comparison)


# 忽略一些计算中的运行时警告
warnings.filterwarnings("ignore")

# ==========================================
# 问题定义模块 (包含目标函数、梯度、约束)
# ==========================================

class OptimizationProblem:
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def fun(self, x):
        pass

    def jac(self, x):
        pass

    def constraints(self, x):
        """返回字典列表，格式符合 scipy.optimize"""
        pass
    
    def eq_cons(self, x):
        """返回等式约束值列表 [c1, c2...]"""
        return []

    def ineq_cons(self, x):
        """返回不等式约束值列表 (<=0 形式) [g1, g2...]"""
        return []
    
    def eq_jac(self, x):
        """返回等式约束的雅可比矩阵"""
        return []

    def ineq_jac(self, x):
        """返回不等式约束的雅可比矩阵"""
        return []

class Problem2D(OptimizationProblem):
    def __init__(self):
        super().__init__("2D Constrained Modified Rosenbrock", 2)
        
    def fun(self, x):
        x1, x2 = x[0], x[1]
        term1 = (1 - x1)**2 + 100 * (x2 - x1**2)**2
        term2 = 5 * np.sin(2 * np.pi * x1) * np.sin(2 * np.pi * x2)
        return term1 + term2

    def jac(self, x):
        x1, x2 = x[0], x[1]
        # df/dx1
        df_dx1 = -2*(1 - x1) - 400*x1*(x2 - x1**2) + \
                 10 * np.pi * np.cos(2 * np.pi * x1) * np.sin(2 * np.pi * x2)
        # df/dx2
        df_dx2 = 200*(x2 - x1**2) + \
                 10 * np.pi * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
        return np.array([df_dx1, df_dx2])

    def eq_cons(self, x):
        # c1(x) = x1^2 + x2 - 2 = 0
        return np.array([x[0]**2 + x[1] - 2])
    
    def eq_jac(self, x):
        # dc1/dx
        return np.array([[2*x[0], 1.0]])

    def ineq_cons(self, x):
        # c2(x) = x1 + x2^2 - 3 <= 0
        return np.array([x[0] + x[1]**2 - 3])
    
    def ineq_jac(self, x):
        # dc2/dx
        return np.array([[1.0, 2*x[1]]])

class Problem4D(OptimizationProblem):
    def __init__(self):
        super().__init__("4D Constrained Rosenbrock", 4)
        
    def fun(self, x):
        # f(x) = sum_{k=1}^3 [(1-xk)^2 + 100(x_{k+1} - xk^2)^2]
        res = 0
        for k in range(3):
            res += (1 - x[k])**2 + 100 * (x[k+1] - x[k]**2)**2
        return res

    def jac(self, x):
        grad = np.zeros_like(x)
        # 链式法则手动推导
        # k=1: (1-x1)^2 + 100(x2-x1^2)^2 -> affects x1, x2
        # k=2: (1-x2)^2 + 100(x3-x2^2)^2 -> affects x2, x3
        # k=3: (1-x3)^2 + 100(x4-x3^2)^2 -> affects x3, x4
        
        # x1: from term k=1
        grad[0] = -2*(1-x[0]) - 400*x[0]*(x[1] - x[0]**2)
        
        # x2: from term k=1 (part 2) and term k=2 (part 1)
        grad[1] = 200*(x[1] - x[0]**2) - 2*(1-x[1]) - 400*x[1]*(x[2] - x[1]**2)
        
        # x3: from term k=2 (part 2) and term k=3 (part 1)
        grad[2] = 200*(x[2] - x[1]**2) - 2*(1-x[2]) - 400*x[2]*(x[3] - x[2]**2)
        
        # x4: from term k=3 (part 2)
        grad[3] = 200*(x[3] - x[2]**2)
        
        return grad

    def eq_cons(self, x):
        # c1: x1 + x2 + x3 + x4 - 2 = 0
        return np.array([np.sum(x) - 2])
    
    def eq_jac(self, x):
        return np.array([[1, 1, 1, 1]])

    def ineq_cons(self, x):
        # c2: x1 - x2 + 1 >= 0 -> -x1 + x2 - 1 <= 0
        c2 = -x[0] + x[1] - 1
        # c3: 2x3 - x4 - 3 <= 0
        c3 = 2*x[2] - x[3] - 3
        # c4: x1^2 + x4^2 - 4 <= 0
        c4 = x[0]**2 + x[3]**2 - 4
        return np.array([c2, c3, c4])

    def ineq_jac(self, x):
        jac = np.zeros((3, 4))
        # c2 grad: [-1, 1, 0, 0]
        jac[0, :] = [-1, 1, 0, 0]
        # c3 grad: [0, 0, 2, -1]
        jac[1, :] = [0, 0, 2, -1]
        # c4 grad: [2x1, 0, 0, 2x4]
        jac[2, :] = [2*x[0], 0, 0, 2*x[3]]
        return jac
    
# ==========================================
#  初始点生成 (每个函数 6 个)
# ==========================================

def get_initial_points():
    """
    包含可行域内、外、边界及鞍点附近的典型位置。
    """
    starts_2d = [
        {"id": "P1", "val": np.array([-1.2, 1.0])},  # 左侧谷底附近
        {"id": "P2", "val": np.array([1.5, 2.5])},   # 右上远离约束区域
        {"id": "P3", "val": np.array([0.0, 0.0])},   # 鞍点/原点
        {"id": "P4", "val": np.array([-1.4, -0.4])}, # 左下边界区域
        {"id": "P5", "val": np.array([1.8, 0.5])},   # 右侧边界附近
        {"id": "P6", "val": np.array([0.5, 2.8])},   # 上方高处
        {"id": "P7", "val": np.array([1.0, 1.0])},   # 理论最优解附近 (测试局部收敛精度)
        {"id": "P8", "val": np.array([-1.5, 3.0])},  # Box约束的左上角极限点
        {"id": "P9", "val": np.array([0.0, 2.0])},   # 等式约束抛物线的顶点 (x1^2 + x2 = 2)
        {"id": "P10", "val": np.array([2.0, -0.5])}, # Box约束的右下角极限点
    ]

    # 4D 问题初始点
    starts_4d = [
        # 原有测试点
        {"id": "P1", "val": np.array([0., 0., 0., 0.])},
        {"id": "P2", "val": np.array([1., 1., 1., 1.])},
        {"id": "P3", "val": np.array([-0.5, 0.5, -0.5, 0.5])},
        {"id": "P4", "val": np.array([0.5, 0.5, 0.5, 0.5])},
        {"id": "P5", "val": np.array([1.2, -0.2, 1.2, -0.2])},
        {"id": "P6", "val": np.array([-1., -1., -1., -1.])},
        {"id": "P7", "val": np.array([2.0, 0.0, 0.0, 0.0])},   # 严格满足线性等式约束 (Sum=2) 的边界点
        {"id": "P8", "val": np.array([0.1, 0.1, 0.1, 0.1])},   # 近原点微扰
        {"id": "P9", "val": np.array([3.0, 3.0, 3.0, 3.0])},   # 远场点 (高度不可行，测试大跨度搜索能力)
        {"id": "P10", "val": np.array([-1.0, 2.0, -1.0, 2.0])},# 符号交替的大数值点
    ]
    
    return starts_2d, starts_4d


# ==========================================
# 实验主程序：执行测试并收集数据
# ==========================================

def execute_full_experiment():
    
    starts_2d, starts_4d = get_initial_points()
    p2d = Problem2D()
    p4d = Problem4D()
    
    
    all_results = []

    # 保存的列：
    # Problem, Start_ID, Start_Coord, Algo, Iter, Final_F, 
    # Grad_Norm (收敛精度), Constr_Viol (约束满足度), 
    # Final_X (收敛点), Success (是否在1000步内满足准则)
        
    # 运行 2D
    for s in starts_2d:
        # ALM
        alm = ALMSolver(p2d,max_iter=1000,patience=1000)
        x_a, h_a = alm.solve(s["val"])
        all_results.append({"Problem": "2D", "Algo": "ALM", "Start_ID": s["id"], 
                            "Start_Coord": str(s["val"]), "Iter": len(h_a['iter']), 
                            "Final_F": h_a['f_val'][-1], "Viol_Eq": h_a["viol_eq"][-1],  
                            "Viol_Ineq": h_a["viol_ineq"][-1], "Total_Viol": h_a["constr_viol"][-1], 
                            "Final_X": np.round(x_a, 4), "Grad_Norm": h_a["grad_norm"][-1],
                            "Status": h_a["status"],"History": h_a})
        # SQP
        sqp = SQPSolver(p2d,max_iter=1000,patience=1000)
        x_s, h_s = sqp.solve(s["val"])
        all_results.append({"Problem": "2D", "Algo": "SQP", "Start_ID": s["id"], 
                            "Start_Coord": str(s["val"]), "Iter": len(h_s['iter']), 
                            "Final_F": h_s['f_val'][-1], "Viol_Eq": h_s["viol_eq"][-1],  
                            "Viol_Ineq": h_s["viol_ineq"][-1], "Total_Viol": h_s["constr_viol"][-1],
                            "Final_X": np.round(x_s, 4), "Grad_Norm": h_s["grad_norm"][-1],
                            "Status": h_s["status"],"History": h_s})

    # 运行 4D
    for s in starts_4d:
        # ALM
        alm = ALMSolver(p4d,max_iter=1000,patience=1000)
        x_a, h_a = alm.solve(s["val"])
        all_results.append({"Problem": "4D", "Algo": "ALM", "Start_ID": s["id"], 
                            "Start_Coord": str(s["val"]), "Iter": len(h_a['iter']), 
                            "Final_F": h_a['f_val'][-1],  "Viol_Eq": h_a["viol_eq"][-1],  
                            "Viol_Ineq": h_a["viol_ineq"][-1], "Total_Viol": h_a["constr_viol"][-1], 
                            "Final_X": np.round(x_a, 4), "Grad_Norm": h_a["grad_norm"][-1],
                            "Status": h_a["status"],"History": h_a})
        # SQP
        sqp = SQPSolver(p4d,max_iter=1000,patience=1000)
        x_s, h_s = sqp.solve(s["val"])
        all_results.append({"Problem": "4D", "Algo": "SQP", "Start_ID": s["id"], 
                            "Start_Coord": str(s["val"]), "Iter": len(h_s['iter']), 
                            "Final_F": h_s['f_val'][-1], "Viol_Eq": h_s["viol_eq"][-1],  
                            "Viol_Ineq": h_s["viol_ineq"][-1], "Total_Viol": h_s["constr_viol"][-1],
                            "Final_X": np.round(x_s, 4), "Grad_Norm": h_s["grad_norm"][-1],
                            "Status": h_s["status"],"History": h_s})

    # 1. 整理表格
    df = pd.DataFrame(all_results).drop(columns=['History'])
    print("\n--- 实验定量结果对比 ---")
    print(df.to_string())
    df.to_csv("optimization_results_new.csv", index=False)
    
    # # 2. 绘图：收敛曲线
    # 创建保存图片的文件夹
    if not os.path.exists('figures'):
        os.makedirs('figures')
    # plot_convergence(all_results, "2D")
    # plot_convergence(all_results, "4D")
    
    # # 3. 绘图：2D 轨迹
    # plot_2d_trajectory(p2d, all_results)
    # # ==========================================
    # # 修改主实验函数中的调用部分
    # # ==========================================

    # # # 2. 调用新版绘图
    # plot_convergence_individual(all_results)
    plot_2d_trajectory_individual(p2d, all_results)
    # plot_performance_summary(all_results)
    plot_convergence_refined(all_results)
    plot_convergence_grid(all_results, problem="2D")
    plot_convergence_grid(all_results, problem="4D")
    plot_performance_with_avg(all_results)
    plot_all_problems_comparison(all_results)


if __name__ == "__main__":
    execute_full_experiment()