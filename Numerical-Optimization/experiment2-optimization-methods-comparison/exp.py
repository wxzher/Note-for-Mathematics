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


# 忽略一些计算中的运行时警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 问题定义模块 (包含目标函数、梯度、约束)
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
    # 2D 问题点：涵盖可行域内外、边界附近
    starts_2d = [
        {"id": "P1", "val": np.array([-1.2, 1.0])},
        {"id": "P2", "val": np.array([1.5, 2.5])},
        {"id": "P3", "val": np.array([0.0, 0.0])},
        {"id": "P4", "val": np.array([-1.4, -0.4])},
        {"id": "P5", "val": np.array([1.8, 0.5])},
        {"id": "P6", "val": np.array([0.5, 2.8])}
    ]
    # 4D 问题点
    starts_4d = [
        {"id": "P1", "val": np.array([0., 0., 0., 0.])},
        {"id": "P2", "val": np.array([1., 1., 1., 1.])},
        {"id": "P3", "val": np.array([-0.5, 0.5, -0.5, 0.5])},
        {"id": "P4", "val": np.array([0.5, 0.5, 0.5, 0.5])},
        {"id": "P5", "val": np.array([1.2, -0.2, 1.2, -0.2])},
        {"id": "P6", "val": np.array([-1., -1., -1., -1.])}
    ]
    return starts_2d, starts_4d


# ==========================================
# 2. 算法实现模块
# ==========================================

class ALMSolver:
    """
    Augmented Lagrangian Method (增广拉格朗日法)
    处理策略：
    - 将不等式约束 c(x)<=0 转化为等式 max(0, c(x) + lambda/mu) 或使用松弛变量。
    - 这里采用常用的 PHR (Powell-Hestenes-Rockafellar) 增广拉格朗日形式。
    """
    def __init__(self, problem, max_iter=1000, tol=1e-6, r=10.0, beta=2.0):
        self.problem = problem
        self.max_iter = max_iter
        self.tol = tol
        self.mu = r # 初始罚参数
        self.beta = beta # 罚参数增长倍率

    def solve(self, x0):
        x = np.array(x0, dtype=float)
        
        # 初始化拉格朗日乘子
        # eq_lambda 对应等式约束, ineq_lambda 对应不等式约束
        num_eq = len(self.problem.eq_cons(x))
        num_ineq = len(self.problem.ineq_cons(x))
        
        lam_eq = np.zeros(num_eq)
        lam_ineq = np.zeros(num_ineq)
        
        history = {'f_val': [], 'constr_viol': [], 'grad_norm': [], 'iter': []}
        
        for k in range(self.max_iter):
            
            # --- 1. 定义子问题：最小化增广拉格朗日函数 L_A(x) ---
            def aug_lagrangian(x_curr):
                f = self.problem.fun(x_curr)
                eqs = self.problem.eq_cons(x_curr)
                ineqs = self.problem.ineq_cons(x_curr)
                
                # 等式部分: lambda*c + (mu/2)*c^2
                term_eq = np.dot(lam_eq, eqs) + (self.mu / 2.0) * np.sum(eqs**2)
                
                # 不等式部分 (PHR形式): (1/2mu) * ( [max(0, mu*g + lambda)]^2 - lambda^2 )
                # 这种形式等价于对违反约束或乘子生效的部分进行惩罚
                term_ineq = 0
                for i, g in enumerate(ineqs):
                    term_ineq += (1.0 / (2.0 * self.mu)) * (
                        max(0, self.mu * g + lam_ineq[i])**2 - lam_ineq[i]**2
                    )
                    
                return f + term_eq + term_ineq

            # --- 2. 求解无约束子问题 (使用 BFGS 或 L-BFGS-B) ---
            # 为了简化，这里直接调用 scipy 的无约束优化器求解子问题
            res = minimize(aug_lagrangian, x, method='BFGS', tol=1e-4) # 子问题精度可以稍低
            x_new = res.x
            
            # --- 3. 计算指标与收敛判断 ---
            eqs_val = self.problem.eq_cons(x_new)
            ineqs_val = self.problem.ineq_cons(x_new)
            
            # 约束违反度 (等式绝对值 + 不等式大于0的部分)
            viol_eq = np.sum(np.abs(eqs_val)) if len(eqs_val) > 0 else 0
            viol_ineq = np.sum(np.maximum(0, ineqs_val)) if len(ineqs_val) > 0 else 0
            total_viol = viol_eq + viol_ineq
            
            grad_norm = np.linalg.norm(res.jac) if hasattr(res, 'jac') else 0 # 子问题的梯度范数
            
            # 记录数据
            history['f_val'].append(self.problem.fun(x_new))
            history['constr_viol'].append(total_viol)
            history['grad_norm'].append(grad_norm)
            history['iter'].append(k)
            
            # 终止条件
            if total_viol < self.tol and np.linalg.norm(self.problem.jac(x_new) - self.problem.jac(x)) < 1e-6:
                # 这是一个简化的收敛判定：可行且x不再显著变化
                # 更严格应当判断 KKT 条件
                if total_viol < self.tol:
                     break
            
            x = x_new
            
            # --- 4. 更新乘子和罚参数 ---
            # 更新等式乘子: lambda <- lambda + mu * c
            lam_eq = lam_eq + self.mu * eqs_val
            
            # 更新不等式乘子: lambda <- max(0, lambda + mu * g)
            for i in range(num_ineq):
                lam_ineq[i] = max(0, lam_ineq[i] + self.mu * ineqs_val[i])
            
            # 增大罚参数 (如果约束违反没有显著改善，这里简单地每次都增加)
            if total_viol > 1e-6:
                self.mu = min(self.mu * self.beta, 1e5) # 设定上限防止数值爆炸

        return x, history

class SQPSolver:
    """
    Sequential Quadratic Programming (序列二次规划)
    核心思想：
    1. 在当前点 x_k 处将目标函数近似为二次函数，约束近似为线性。
    2. 求解 QP 子问题得到方向 d_k。
    3. 线搜索更新 x_{k+1} = x_k + alpha * d_k。
    """
    def __init__(self, problem, max_iter=1000, tol=1e-6):
        self.problem = problem
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, x0):
        x = np.array(x0, dtype=float)
        n = len(x)
        
        # 初始 Hessian 近似 (单位阵)
        B = np.eye(n)
        
        history = {'f_val': [], 'constr_viol': [], 'grad_norm': [], 'iter': []}
        
        for k in range(self.max_iter):
            grad = self.problem.jac(x)
            
            # 获取约束值和雅可比
            eqs = self.problem.eq_cons(x)
            ineqs = self.problem.ineq_cons(x)
            
            J_eq = self.problem.eq_jac(x)
            J_ineq = self.problem.ineq_jac(x)
            
            # 计算当前约束违反度用于记录
            viol_eq = np.sum(np.abs(eqs)) if len(eqs) > 0 else 0
            viol_ineq = np.sum(np.maximum(0, ineqs)) if len(ineqs) > 0 else 0
            total_viol = viol_eq + viol_ineq
            
            history['f_val'].append(self.problem.fun(x))
            history['constr_viol'].append(total_viol)
            history['grad_norm'].append(np.linalg.norm(grad))
            history['iter'].append(k)
            
            # 收敛判定
            # (梯度小且可行)
            # 注意：对于约束问题，梯度不一定为0，这里简化判定可行性达到要求且步长很小
            if total_viol < self.tol and k > 0:
                # 如果这是最后一步的记录，判断上一此步长或Lagrangian梯度
                pass 
                
            # --- 1. 构建 QP 子问题 ---
            # min  0.5 * d^T * B * d + grad^T * d
            # s.t. J_eq * d + eqs = 0
            #      J_ineq * d + ineqs <= 0
            
            def qp_obj(d):
                return 0.5 * d @ B @ d + grad @ d
            
            def qp_jac(d):
                return B @ d + grad
            
            constraints = []
            if len(eqs) > 0:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda d: J_eq @ d + eqs, 
                    'jac': lambda d: J_eq
                })
            if len(ineqs) > 0:
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda d: -(J_ineq @ d + ineqs), # scipy ineq is >= 0
                    'jac': lambda d: -J_ineq
                })
            
            # 求解 QP 寻找方向 d
            # 使用 SLSQP 求解这个局部的 QP 问题
            d0 = np.zeros(n)
            res_qp = minimize(qp_obj, d0, method='SLSQP', jac=qp_jac, 
                              constraints=constraints, tol=1e-8)
            d = res_qp.x
            
            if np.linalg.norm(d) < 1e-8 and total_viol < self.tol:
                # 方向几乎为0且满足约束，收敛
                break
            
            # --- 2. 线搜索 (简单 Armijo 或 衰减) ---
            alpha = 1.0
            rho = 0.5
            c1 = 1e-4
            
            # 定义 L1 罚函数作为 Merit Function
            mu_merit = 100.0 # 简单的固定惩罚系数，实际应动态调整
            def merit(x_val):
                v_eq = np.sum(np.abs(self.problem.eq_cons(x_val))) if len(self.problem.eq_cons(x_val)) else 0
                v_ineq = np.sum(np.maximum(0, self.problem.ineq_cons(x_val))) if len(self.problem.ineq_cons(x_val)) else 0
                return self.problem.fun(x_val) + mu_merit * (v_eq + v_ineq)
            
            phi_0 = merit(x)
            # 简单的回溯
            while alpha > 1e-4:
                x_next = x + alpha * d
                if merit(x_next) < phi_0: # 简化条件
                    break
                alpha *= rho
                
            # --- 3. 更新 x 和 B (BFGS) ---
            x_new = x + alpha * d
            
            # BFGS 更新 Hessian 近似 B
            # 需要计算 y = grad_L(x_new) - grad_L(x)，这里简化使用 grad_f 近似
            # 严格SQP需要拉格朗日梯度的差分
            s = x_new - x
            y = self.problem.jac(x_new) - self.problem.jac(x) # 简化版
            
            if np.dot(s, y) > 1e-8: # 保证正定性
                Bs = B @ s
                term1 = np.outer(Bs, Bs) / (s @ Bs)
                term2 = np.outer(y, y) / (y @ s)
                B = B - term1 + term2
            
            x = x_new
            
        return x, history


# 重新定义 ALM 的 solve 以记录 x_path
def alm_solve_with_path(self, x0):
    x = np.array(x0, dtype=float)
    # ... (初始化乘子逻辑同前) ...
    num_eq = len(self.problem.eq_cons(x))
    num_ineq = len(self.problem.ineq_cons(x))
    lam_eq = np.zeros(num_eq)
    lam_ineq = np.zeros(num_ineq)
    
    # 新增 x_path 记录
    history = {'f_val': [], 'constr_viol': [], 'grad_norm': [], 'iter': [], 'x_path': [x.copy()]}
    
    for k in range(self.max_iter):
        # ... (子问题定义同前) ...
        def aug_lagrangian(x_curr):
            f = self.problem.fun(x_curr)
            eqs = self.problem.eq_cons(x_curr)
            ineqs = self.problem.ineq_cons(x_curr)
            term_eq = np.dot(lam_eq, eqs) + (self.mu / 2.0) * np.sum(eqs**2) if len(eqs)>0 else 0
            term_ineq = 0
            for i, g in enumerate(ineqs):
                term_ineq += (1.0 / (2.0 * self.mu)) * (max(0, self.mu * g + lam_ineq[i])**2 - lam_ineq[i]**2)
            return f + term_eq + term_ineq

        # 求解子问题
        res = minimize(aug_lagrangian, x, method='BFGS', tol=1e-4)
        x_new = res.x
        
        # 计算指标
        eqs_val = self.problem.eq_cons(x_new)
        ineqs_val = self.problem.ineq_cons(x_new)
        viol_eq = np.sum(np.abs(eqs_val)) if len(eqs_val) > 0 else 0
        viol_ineq = np.sum(np.maximum(0, ineqs_val)) if len(ineqs_val) > 0 else 0
        total_viol = viol_eq + viol_ineq
        
        # 记录
        history['f_val'].append(self.problem.fun(x_new))
        history['constr_viol'].append(total_viol)
        history['grad_norm'].append(np.linalg.norm(res.jac) if hasattr(res, 'jac') else 0)
        history['iter'].append(k)
        history['x_path'].append(x_new.copy()) # <--- 关键补充
        
        if total_viol < self.tol and np.linalg.norm(x_new - x) < 1e-6:
             break
        
        x = x_new
        
        # 更新乘子
        if len(eqs_val) > 0: lam_eq += self.mu * eqs_val
        for i in range(num_ineq): lam_ineq[i] = max(0, lam_ineq[i] + self.mu * ineqs_val[i])
        
        if total_viol > 1e-6: self.mu = min(self.mu * self.beta, 1e5)

    return x, history

# 重新定义 SQP 的 solve 以记录 x_path
def sqp_solve_with_path(self, x0):
    x = np.array(x0, dtype=float)
    n = len(x)
    B = np.eye(n)
    
    # 新增 x_path 记录
    history = {'f_val': [], 'constr_viol': [], 'grad_norm': [], 'iter': [], 'x_path': [x.copy()]}
    
    for k in range(self.max_iter):
        grad = self.problem.jac(x)
        eqs = self.problem.eq_cons(x)
        ineqs = self.problem.ineq_cons(x)
        J_eq = self.problem.eq_jac(x)
        J_ineq = self.problem.ineq_jac(x)
        
        viol_eq = np.sum(np.abs(eqs)) if len(eqs) > 0 else 0
        viol_ineq = np.sum(np.maximum(0, ineqs)) if len(ineqs) > 0 else 0
        total_viol = viol_eq + viol_ineq
        
        history['f_val'].append(self.problem.fun(x))
        history['constr_viol'].append(total_viol)
        history['grad_norm'].append(np.linalg.norm(grad))
        history['iter'].append(k)
        
        if total_viol < self.tol and k > 0 and np.linalg.norm(grad)<1.0: # 简单收敛判定
             pass # 继续循环直到步长极小

        # QP 子问题 (同前)
        def qp_obj(d): return 0.5 * d @ B @ d + grad @ d
        def qp_jac(d): return B @ d + grad
        constraints = []
        if len(eqs) > 0: constraints.append({'type': 'eq', 'fun': lambda d: J_eq @ d + eqs, 'jac': lambda d: J_eq})
        if len(ineqs) > 0: constraints.append({'type': 'ineq', 'fun': lambda d: -(J_ineq @ d + ineqs), 'jac': lambda d: -J_ineq})
        
        d0 = np.zeros(n)
        try:
            res_qp = minimize(qp_obj, d0, method='SLSQP', jac=qp_jac, constraints=constraints, tol=1e-8)
            d = res_qp.x
        except:
            break # QP 求解失败
            
        if np.linalg.norm(d) < 1e-6 and total_viol < self.tol: break
        
        # 线搜索 (同前)
        alpha = 1.0; rho = 0.5; mu_merit = 100.0
        def merit(x_val):
            v_eq = np.sum(np.abs(self.problem.eq_cons(x_val))) if len(self.problem.eq_cons(x_val)) else 0
            v_ineq = np.sum(np.maximum(0, self.problem.ineq_cons(x_val))) if len(self.problem.ineq_cons(x_val)) else 0
            return self.problem.fun(x_val) + mu_merit * (v_eq + v_ineq)
        phi_0 = merit(x)
        while alpha > 1e-4:
            if merit(x + alpha * d) < phi_0: break
            alpha *= rho
            
        x_new = x + alpha * d
        
        # BFGS 更新
        s = x_new - x
        y = self.problem.jac(x_new) - self.problem.jac(x)
        if np.dot(s, y) > 1e-8:
            Bs = B @ s
            term1 = np.outer(Bs, Bs) / (s @ Bs)
            term2 = np.outer(y, y) / (y @ s)
            B = B - term1 + term2
        
        x = x_new
        history['x_path'].append(x.copy()) # 记录路径

    return x, history

# 动态替换类方法 (Monkey Patching)，避免你需要重写整个类
ALMSolver.solve = alm_solve_with_path
SQPSolver.solve = sqp_solve_with_path

# ==========================================
# 4. 绘图函数模块
# ==========================================
# 创建保存图片的文件夹
if not os.path.exists('figures'):
    os.makedirs('figures')

def plot_convergence(results_list, problem_name):
    """绘制收敛曲线：log(grad + viol) vs Iteration"""
    plt.figure(figsize=(10, 6))
    
    for res in results_list:
        if res['Problem'] != problem_name: continue
        
        hist = res['History']
        # 构造收敛指标：log10(||grad|| + ||viol|| + 1e-10)
        metric = np.array(hist['grad_norm']) + np.array(hist['constr_viol'])
        metric = np.log10(metric + 1e-10)
        
        label_str = f"{res['Algo']} (Start: {res['Start']})"
        plt.plot(hist['iter'], metric, marker='.', markersize=5, label=label_str)
        
    plt.xlabel('Iterations')
    plt.ylabel('Log10 (||Grad|| + ||Constr Viol||)')
    plt.title(f'Convergence Analysis: {problem_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def plot_2d_trajectory(problem, results_list):

    """
    绘制 2D 优化路径与约束边界
    """
    if problem.dim != 2: return
    
    # 1. 生成网格数据用于等高线
    x1_range = np.linspace(-2.0, 2.5, 200)
    x2_range = np.linspace(-1.0, 3.5, 200)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    
    # 计算目标函数值
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i,j] = problem.fun([X1[i,j], X2[i,j]])
            
    plt.figure(figsize=(10, 8))
    
    # 2. 绘制目标函数等高线
    cp = plt.contour(X1, X2, Z, levels=30, cmap='viridis', alpha=0.6)
    plt.colorbar(cp, label='Objective Function')
    
    # 3. 绘制约束边界
    # c1(x) = x1^2 + x2 - 2 = 0 (红线)
    plt.contour(X1, X2, X1**2 + X2 - 2, levels=[0], colors='red', linewidths=2, linestyles='solid')
    # c2(x) = x1 + x2^2 - 3 <= 0 (蓝虚线边界)
    plt.contour(X1, X2, X1 + X2**2 - 3, levels=[0], colors='blue', linewidths=2, linestyles='dashed')
    
    # 标注可行域提示 (手动添加图例)
    plt.plot([], [], 'r-', label='Equality: $x_1^2+x_2-2=0$')
    plt.plot([], [], 'b--', label='Inequality: $x_1+x_2^2-3 \leq 0$')
    
    # 4. 绘制算法轨迹
    colors = ['k', 'm', 'c', 'orange', 'purple']
    for idx, res in enumerate(results_list):
        if res['Problem'] != "2D": continue
        hist = res['History']
        path = np.array(hist['x_path'])
        
        # 绘制路径
        color = colors[idx % len(colors)]
        plt.plot(path[:, 0], path[:, 1], color=color, marker='o', markersize=4, 
                 linewidth=1.5, label=f"{res['Algo']} (Start {res['Start_ID']})")
        
        # 标注起点和终点
        plt.plot(path[0, 0], path[0, 1], color=color, marker='s', markersize=8) # 起点：方块
        plt.plot(path[-1, 0], path[-1, 1], color=color, marker='*', markersize=10) # 终点：星形

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Optimization Paths on 2D Constrained Modified Rosenbrock')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_convergence_individual(results_list):
    """
    为每个初始点单独绘制收敛曲线对比图
    包含：梯度下降曲线 和 约束违反度下降曲线
    """
    # 按初始点分组
    unique_starts = sorted(list(set([res['Start_ID'] for res in results_list])))
    
    for start_id in unique_starts:
        # 筛选该初始点下的所有算法结果
        group = [res for res in results_list if res['Start_ID'] == start_id]
        problem_name = group[0]['Problem']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Convergence Comparison - Start Point {start_id} ({problem_name})', fontsize=14)

        for res in group:
            hist = res['History']
            label = res['Algo']
            
            # 左图：KKT梯度/目标函数趋势 (Log Scale)
            metric = np.array(hist['grad_norm']) + 1e-12
            ax1.semilogy(hist['iter'], metric, label=f'{label} Grad Norm', linewidth=1.5)
            
            # 右图：约束违反度 (Log Scale)
            viol = np.array(hist['constr_viol']) + 1e-12
            ax2.semilogy(hist['iter'], viol, label=f'{label} Constraint Viol', linestyle='--', linewidth=1.5)

        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Log ||Grad L||')
        ax1.set_title('Gradient Norm Convergence')
        ax1.grid(True, which="both", ls="-", alpha=0.5)
        ax1.legend(loc='best')

        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Log ||Constraint Violation||')
        ax2.set_title('Feasibility Convergence')
        ax2.grid(True, which="both", ls="-", alpha=0.5)
        ax2.legend(loc='best')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"figures/conv_{problem_name}_{start_id}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved convergence plot: {filename}")

def plot_2d_trajectory_individual(problem, results_list):
    """
    为2D问题的每个初始点单独绘制轨迹图
    """
    if problem.dim != 2: return
    
    # 准备网格数据
    x1_range = np.linspace(-1.6, 2.1, 200)
    x2_range = np.linspace(-0.6, 3.1, 200)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i,j] = problem.fun([X1[i,j], X2[i,j]])

    unique_starts = sorted(list(set([res['Start_ID'] for res in results_list if res['Problem'] == "2D"])))
    
    for start_id in unique_starts:
        plt.figure(figsize=(9, 7))
        
        # 1. 绘制等高线
        cp = plt.contour(X1, X2, Z, levels=40, cmap='viridis', alpha=0.4)
        plt.colorbar(cp, label='f(x)')
        
        # 2. 绘制约束边界
        # 等式约束 c1=0: x1^2 + x2 - 2 = 0
        c1_vals = X1**2 + X2 - 2
        plt.contour(X1, X2, c1_vals, levels=[0], colors='red', linewidths=2)
        # 不等式约束 c2=0: x1 + x2^2 - 3 = 0
        c2_vals = X1 + X2**2 - 3
        plt.contour(X1, X2, c2_vals, levels=[0], colors='blue', linewidths=2, linestyles='--')
        
        # 3. 绘制该初始点下的算法路径
        group = [res for res in results_list if res['Start_ID'] == start_id and res['Problem'] == "2D"]
        colors = {'ALM': 'black', 'SQP': 'darkorange'}
        markers = {'ALM': 'o', 'SQP': '^'}

        for res in group:
            path = np.array(res['History']['x_path'])
            algo = res['Algo']
            plt.plot(path[:, 0], path[:, 1], color=colors[algo], label=f'{algo} Path', 
                     marker=markers[algo], markersize=3, linewidth=1.5, alpha=0.8)
            
            # 标注起终点
            plt.scatter(path[0, 0], path[0, 1], c='green', marker='s', s=60, zorder=5) # Start
            plt.scatter(path[-1, 0], path[-1, 1], c='red', marker='*', s=150, zorder=5) # End

        # 图例处理
        plt.plot([], [], 'r-', label='Equality Constraint')
        plt.plot([], [], 'b--', label='Inequality Constraint')
        plt.scatter([], [], c='green', marker='s', label='Start')
        plt.scatter([], [], c='red', marker='*', label='Optimum')
        
        plt.title(f'Optimization Trajectory - Start Point {start_id}')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1))
        plt.grid(True, alpha=0.2)
        
        filename = f"figures/path_2D_{start_id}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved trajectory plot: {filename}")

def plot_performance_summary(results_list):
    """
    新增：性能对比汇总条形图（对比迭代次数）
    """
    problems = sorted(list(set([res['Problem'] for res in results_list])))
    
    for prob in problems:
        group = [res for res in results_list if res['Problem'] == prob]
        starts = sorted(list(set([res['Start_ID'] for res in group])))
        algos = sorted(list(set([res['Algo'] for res in group])))
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(starts))
        width = 0.35
        
        for i, algo in enumerate(algos):
            iters = [next(res['Iter'] for res in group if res['Start_ID'] == s and res['Algo'] == algo) for s in starts]
            plt.bar(x + i*width - width/2, iters, width, label=algo)
            
        plt.ylabel('Total Iterations')
        plt.title(f'Computational Efficiency Comparison ({prob})')
        plt.xticks(x, [f'Start {s}' for s in starts])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        filename = f"figures/summary_{prob}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved summary plot: {filename}")

def plot_convergence_refined(results_list):
    """每个初始点一张图，包含坐标信息，对比 ALM 和 SQP"""
    unique_ids = sorted(list(set([(res['Problem'], res['Start_ID'], res['Start_Coord']) for res in results_list])))
    
    for prob, start_id, coord in unique_ids:
        group = [res for res in results_list if res['Problem'] == prob and res['Start_ID'] == start_id]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # 统一收敛指标：log10(||grad_L|| + ||constr_viol||)
        for res in group:
            hist = res['History']
            metric = np.array(hist['grad_norm']) + np.array(hist['constr_viol']) + 1e-12
            ax.semilogy(hist['iter'], metric, label=f"{res['Algo']} (Final F: {res['Final_F']:.4f})", linewidth=2)
            
        ax.set_title(f"收敛曲线: {prob} at {start_id} {coord}\nMetric: log10(||grad_L|| + ||viol||)")
        ax.set_xlabel("迭代次数 (Iteration)")
        ax.set_ylabel("残差范数 (Residual Norm)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"figures/conv_{prob}_{start_id}.png", dpi=300)
        plt.close()

def plot_performance_with_avg(results_list):
    """性能汇总图：包含每个点对比 + 全局均值对比"""
    problems = sorted(list(set([res['Problem'] for res in results_list])))
    
    for prob in problems:
        group = [res for res in results_list if res['Problem'] == prob]
        starts = sorted(list(set([res['Start_ID'] for res in group])))
        algos = ['ALM', 'SQP']
        
        # 准备数据
        data = {algo: [] for algo in algos}
        for algo in algos:
            for s in starts:
                val = next(res['Iter'] for res in group if res['Start_ID'] == s and res['Algo'] == algo)
                data[algo].append(val)
            # 计算均值
            data[algo].append(np.mean(data[algo]))
        
        labels = starts + ["均值(AVG)"]
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, algo in enumerate(algos):
            bars = ax.bar(x + i*width - width/2, data[algo], width, label=algo)
            # 在条形图上方标注数值
            ax.bar_label(bars, padding=3)
            
        # 突出显示均值列
        ax.axvspan(len(starts)-0.5, len(starts)+0.5, color='gray', alpha=0.1)
        
        ax.set_ylabel('迭代次数 (Iterations)')
        ax.set_title(f'算法效率对比与显著性分析: {prob}')
        ax.set_xticks(x, labels)
        ax.legend()
        plt.savefig(f"figures/summary_{prob}.png", dpi=300)
        plt.close()

# ==========================================
# 5. 实验主程序：执行测试并收集数据
# ==========================================

def execute_full_experiment():
    
    # 设置不同的初始点
    # starts_2d = [
    #     {"id": "A", "val": np.array([-1.2, 1.0])},
    #     {"id": "B", "val": np.array([1.5, 2.5])},
    #     {"id": "C", "val": np.array([0.0, 0.0])}
    # ]
    
    # starts_4d = [
    #     {"id": "D", "val": np.array([0.0, 0.0, 0.0, 0.0])},
    #     {"id": "E", "val": np.array([1.5, 1.5, 1.5, 1.5])}
    # ]

    starts_2d, starts_4d = get_initial_points()
    p2d = Problem2D()
    p4d = Problem4D()
    
    
    all_results = []

    # 建议保存的列：
    # Problem, Start_ID, Start_Coord, Algo, Iter, Final_F, 
    # Grad_Norm (收敛精度), Constr_Viol (约束满足度), 
    # Final_X (收敛点), Success (是否在1000步内满足准则)
        
    # 运行 2D
    for s in starts_2d:
        # ALM
        alm = ALMSolver(p2d)
        x_a, h_a = alm.solve(s["val"])
        all_results.append({"Problem": "2D", "Algo": "ALM", "Start_ID": s["id"], 
                            "Start_Coord": str(s["val"]), "Iter": len(h_a['iter']), 
                            "Final_F": h_a['f_val'][-1], "Viol": h_a['constr_viol'][-1], 
                            "Final_X": np.round(x_a, 4), "History": h_a})
        # SQP
        sqp = SQPSolver(p2d)
        x_s, h_s = sqp.solve(s["val"])
        all_results.append({"Problem": "2D", "Algo": "SQP", "Start_ID": s["id"], 
                            "Start_Coord": str(s["val"]), "Iter": len(h_s['iter']), 
                            "Final_F": h_s['f_val'][-1], "Viol": h_s['constr_viol'][-1], 
                            "Final_X": np.round(x_s, 4), "History": h_s})

    # 运行 4D
    for s in starts_4d:
        # ALM
        alm = ALMSolver(p4d)
        x_a, h_a = alm.solve(s["val"])
        all_results.append({"Problem": "4D", "Algo": "ALM", "Start_ID": s["id"], 
                            "Start_Coord": str(s["val"]), "Iter": len(h_a['iter']), 
                            "Final_F": h_a['f_val'][-1], "Viol": h_a['constr_viol'][-1], 
                            "Final_X": np.round(x_a, 4), "History": h_a})
        # SQP
        sqp = SQPSolver(p4d)
        x_s, h_s = sqp.solve(s["val"])
        all_results.append({"Problem": "4D", "Algo": "SQP", "Start_ID": s["id"], 
                            "Start_Coord": str(s["val"]), "Iter": len(h_s['iter']), 
                            "Final_F": h_s['f_val'][-1], "Viol": h_s['constr_viol'][-1], 
                            "Final_X": np.round(x_s, 4), "History": h_s})

    # 1. 整理表格
    df = pd.DataFrame(all_results).drop(columns=['History'])
    print("\n--- 实验定量结果对比 ---")
    print(df.to_string())
    df.to_csv("optimization_results.csv", index=False)
    
    # # 2. 绘图：收敛曲线
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
    plot_performance_with_avg(all_results)


if __name__ == "__main__":
    execute_full_experiment()