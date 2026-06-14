# @coding=utf-8
# @Author: Wuxiaozhen
# @File: run.py
# @Date: 2026-01-05
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os

# ==========================================
# 1. 基础环境设置
# ==========================================
# 获取当前脚本文件的绝对路径目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 强制将工作目录切换到脚本所在目录
os.chdir(current_dir)
# 确保保存图片的文件夹存在
if not os.path.exists('figures'): 
    os.makedirs('figures')

# 设置绘图风格，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 从当前文件夹导入自定义工具包
# solver.py: 包含 ALMSolver 和 SQPSolver 类
# plot_utils.py: 包含各种绘图函数
try:
    from solver import ALMSolver, SQPSolver
    from plot_utils import (plot_2d_trajectory_individual,
                           plot_convergence_refined, plot_convergence_grid,
                           plot_performance_with_avg, plot_all_problems_comparison)
except ImportError as e:
    print(f"错误: 无法导入本地模块。请确保 solver.py 和 plot_utils.py 在当前目录下。\n详细信息: {e}")
    exit(1)

# 忽略数值计算中可能出现的 RuntimeWarning (如 log(0) 或溢出)，以免刷屏
warnings.filterwarnings("ignore")


# ==========================================
# 2. 问题定义模块 
#    包含目标函数(fun)、梯度(jac)、等式/不等式约束及其雅可比
# ==========================================

class OptimizationProblem:
    """
    优化问题基类，定义统一接口
    """
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def fun(self, x): pass
    def jac(self, x): pass
    def eq_cons(self, x): return np.array([])
    def ineq_cons(self, x): return np.array([])
    def eq_jac(self, x): return np.array([])
    def ineq_jac(self, x): return np.array([])

class Problem2D(OptimizationProblem):
    """
    二维约束优化问题 (Modified Rosenbrock)
    目标函数: f(x) = (1-x1)^2 + 100(x2-x1^2)^2 + 干扰项
    约束: 
      c1(x): x1^2 + x2 - 2 = 0  (抛物线等式)
      c2(x): x1 + x2^2 - 3 <= 0 (抛物线区域不等式)
    """
    def __init__(self):
        super().__init__("2D Constrained Modified Rosenbrock", 2)
        
    def fun(self, x):
        x1, x2 = x[0], x[1]
        # 经典 Rosenbrock 香蕉函数项
        term1 = (1 - x1)**2 + 100 * (x2 - x1**2)**2
        # 增加正弦波动的干扰项，制造多模态特征
        term2 = 5 * np.sin(2 * np.pi * x1) * np.sin(2 * np.pi * x2)
        return term1 + term2

    def jac(self, x):
        """目标函数的解析梯度向量 [df/dx1, df/dx2]"""
        x1, x2 = x[0], x[1]
        # 对 Rosenbrock 项求导
        df_dx1 = -2*(1 - x1) - 400*x1*(x2 - x1**2) + \
                 10 * np.pi * np.cos(2 * np.pi * x1) * np.sin(2 * np.pi * x2)
        df_dx2 = 200*(x2 - x1**2) + \
                 10 * np.pi * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
        return np.array([df_dx1, df_dx2])

    def eq_cons(self, x):
        # 等式约束: x1^2 + x2 - 2 = 0
        return np.array([x[0]**2 + x[1] - 2])
    
    def eq_jac(self, x):
        # 等式约束雅可比: [2x1, 1]
        return np.array([[2*x[0], 1.0]])

    def ineq_cons(self, x):
        # 不等式约束: x1 + x2^2 - 3 <= 0
        # 注意：SciPy标准形式通常是 C(x) >= 0，但此处代码逻辑适配的是 C(x) <= 0 或者是根据 solver 内部逻辑处理
        # 根据 solver.py 的逻辑: max(0, ineqs) 计算违约，说明 solver 期望的是 ineqs <= 0 为满足
        return np.array([x[0] + x[1]**2 - 3])
    
    def ineq_jac(self, x):
        # 不等式约束雅可比: [1, 2x2]
        return np.array([[1.0, 2*x[1]]])

class Problem4D(OptimizationProblem):
    """
    四维扩展 Rosenbrock 问题
    目标函数: 链式 Rosenbrock
    约束: 包含线性等式约束和多个非线性不等式约束
    """
    def __init__(self):
        super().__init__("4D Constrained Rosenbrock", 4)
        
    def fun(self, x):
        # f(x) = sum [(1-xk)^2 + 100(x_{k+1} - xk^2)^2]
        res = 0
        for k in range(3):
            res += (1 - x[k])**2 + 100 * (x[k+1] - x[k]**2)**2
        return res

    def jac(self, x):
        # 手动推导的链式法则梯度
        grad = np.zeros_like(x)
        grad[0] = -2*(1-x[0]) - 400*x[0]*(x[1] - x[0]**2)
        grad[1] = 200*(x[1] - x[0]**2) - 2*(1-x[1]) - 400*x[1]*(x[2] - x[1]**2)
        grad[2] = 200*(x[2] - x[1]**2) - 2*(1-x[2]) - 400*x[2]*(x[3] - x[2]**2)
        grad[3] = 200*(x[3] - x[2]**2)
        return grad

    def eq_cons(self, x):
        # 线性超平面约束: Sum(x) = 2
        return np.array([np.sum(x) - 2])
    
    def eq_jac(self, x):
        return np.array([[1, 1, 1, 1]])

    def ineq_cons(self, x):
        # c2: -x1 + x2 - 1 <= 0
        c2 = -x[0] + x[1] - 1
        # c3: 2x3 - x4 - 3 <= 0
        c3 = 2*x[2] - x[3] - 3
        # c4: x1^2 + x4^2 - 4 <= 0 (圆柱面内部)
        c4 = x[0]**2 + x[3]**2 - 4
        return np.array([c2, c3, c4])

    def ineq_jac(self, x):
        jac = np.zeros((3, 4))
        jac[0, :] = [-1, 1, 0, 0]
        jac[1, :] = [0, 0, 2, -1]
        jac[2, :] = [2*x[0], 0, 0, 2*x[3]]
        return jac
    
# ==========================================
# 3. 初始点生成模块
# ==========================================

def get_initial_points():
    """
    生成测试用的初始点集。
    包含：可行域内点、边界点、远场点、鞍点等，用于测试算法的鲁棒性。
    """
    # 2D 问题初始点 (10个)
    starts_2d = [
        {"id": "P1", "val": np.array([-1.2, 1.0])},  # 经典Rosenbrock测试点
        {"id": "P2", "val": np.array([1.5, 2.5])},   # 右上侧
        {"id": "P3", "val": np.array([0.0, 0.0])},   # 原点
        {"id": "P4", "val": np.array([-1.4, -0.4])}, # 左下
        {"id": "P5", "val": np.array([1.8, 0.5])},   # 右侧
        {"id": "P6", "val": np.array([0.5, 2.8])},   # 违背约束较大的点
        {"id": "P7", "val": np.array([1.0, 1.0])},   # 目标函数极小值附近
        {"id": "P8", "val": np.array([-1.5, 3.0])},  
        {"id": "P9", "val": np.array([0.0, 2.0])},   
        {"id": "P10", "val": np.array([2.0, -0.5])}, 
    ]

    # 4D 问题初始点 (10个)
    starts_4d = [
        {"id": "P1", "val": np.array([0., 0., 0., 0.])},
        {"id": "P2", "val": np.array([1., 1., 1., 1.])},
        {"id": "P3", "val": np.array([-0.5, 0.5, -0.5, 0.5])},
        {"id": "P4", "val": np.array([0.5, 0.5, 0.5, 0.5])},
        {"id": "P5", "val": np.array([1.2, -0.2, 1.2, -0.2])},
        {"id": "P6", "val": np.array([-1., -1., -1., -1.])},
        {"id": "P7", "val": np.array([2.0, 0.0, 0.0, 0.0])}, 
        {"id": "P8", "val": np.array([0.1, 0.1, 0.1, 0.1])}, 
        {"id": "P9", "val": np.array([3.0, 3.0, 3.0, 3.0])},   # 远场点
        {"id": "P10", "val": np.array([-1.0, 2.0, -1.0, 2.0])},
    ]
    
    return starts_2d, starts_4d


# ==========================================
# 4. 实验执行与结果收集模块
# ==========================================

def execute_full_experiment():
    print("开始执行数值优化对比实验...")
    print("算法: ALM (Augmented Lagrangian) vs SQP (Sequential Quadratic Programming)")
    
    starts_2d, starts_4d = get_initial_points()
    p2d = Problem2D()
    p4d = Problem4D()
    
    all_results = []

    # 定义通用运行逻辑：输入算法、问题、起始点，返回结果字典
    def run_solver(algo_name, problem, start_point_dict):
        start_val = start_point_dict["val"]
        s_id = start_point_dict["id"]
        
        # 实例化求解器
        if algo_name == "ALM":
            solver = ALMSolver(problem, max_iter=1000, patience=1000) # 注意 patience 设置
        else:
            solver = SQPSolver(problem, max_iter=1000, patience=1000)
            
        # 求解
        x_final, history = solver.solve(start_val)
        
        # --- 获取关键指标 ---
        # 1. 增广拉格朗日梯度模长 (仅 ALM 有)
        final_aug_grad = history['aug_grad_norm'][-1] if 'aug_grad_norm' in history else None
        
        # 2. 梯度向量 (仅 SQP 记录了向量历史，ALM 我们只记录了模长，可视情况统一)
        # 将向量转为字符串以便存入 CSV
        final_grad_vec_str = str(np.round(history['grad_vec'][-1], 4)) if 'grad_vec' in history else "N/A"

        # 构建结果字典
        result_entry = {
            "Problem": problem.name[:2], # "2D" or "4D"
            "Algo": algo_name,
            "Start_ID": s_id,
            "Start_Coord": str(list(start_val)),
            "Iter": len(history['iter']),
            "Final_F": history['f_val'][-1],
            "Viol_Eq": history["viol_eq"][-1],
            "Viol_Ineq": history["viol_ineq"][-1],
            "Total_Viol": history["constr_viol"][-1],
            "Final_X": str(np.round(x_final, 4)), # 转字符串防格式错乱
            "Grad_Norm": history["grad_norm"][-1],
            "Aug_Grad_Norm": final_aug_grad,      # 【新增】ALM 子问题收敛性指标
            "Final_Grad_Vec": final_grad_vec_str, # 【新增】最终梯度向量
            "Status": history["status"],
            "History": history # 保留完整历史用于绘图，保存CSV时会剔除
        }
        return result_entry

    # --- 运行 2D 问题 ---
    print("\n正在运行 2D 问题测试...")
    for s in starts_2d:
        all_results.append(run_solver("ALM", p2d, s))
        all_results.append(run_solver("SQP", p2d, s))

    # --- 运行 4D 问题 ---
    print("正在运行 4D 问题测试...")
    for s in starts_4d:
        all_results.append(run_solver("ALM", p4d, s))
        all_results.append(run_solver("SQP", p4d, s))

    # ==========================================
    # 5. 数据保存与展示
    # ==========================================
    
    # 转换为 DataFrame 并剔除无法保存到 CSV 的复杂对象 (History)
    df = pd.DataFrame(all_results)
    df_csv = df.drop(columns=['History'])
    
    print("\n--- 实验定量结果摘要 (前10行) ---")
    print(df_csv.head(10).to_string())
    
    csv_filename = "optimization_results.csv"
    df_csv.to_csv(csv_filename, index=False)
    print(f"\n完整结果表格已保存至: {csv_filename}")
    
    # ==========================================
    # 6. 调用绘图工具 (plot_utils)
    # ==========================================
    print("\n正在生成可视化图表...")
    
    # 2D 轨迹图 (仅针对 2D 问题)
    plot_2d_trajectory_individual(p2d, all_results)
    
    # 收敛曲线 (精细化对比图)
    plot_convergence_refined(all_results)
    
    # 网格化收敛图 (按问题分类)
    plot_convergence_grid(all_results, problem="2D")
    plot_convergence_grid(all_results, problem="4D")
    
    # 性能概览 (箱线图/平均值)
    plot_performance_with_avg(all_results)
    
    # 综合对比图
    plot_all_problems_comparison(all_results)
    
    print("所有图表已保存至 figures/ 文件夹。实验结束。")

if __name__ == "__main__":
    execute_full_experiment()