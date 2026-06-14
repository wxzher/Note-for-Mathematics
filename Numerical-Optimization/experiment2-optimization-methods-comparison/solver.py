# @coding=utf-8
# @Author: Wuxiaozhen
# @File: solver.py
# @Date: 2026-01-05
# @Description: 数值优化实验二算法代码，包含ALM和SQP求解器的实现

import numpy as np
from scipy.optimize import minimize

class ConstrainedProblem:
    """
    问题包装类，用于统一管理目标函数、梯度、约束及其雅可比
    """
    def __init__(self, fun, jac, eq_cons=None, eq_jac=None, ineq_cons=None, ineq_jac=None):
        self.fun = fun
        self.jac = jac
        self.eq_cons = eq_cons if eq_cons else lambda x: np.array([])
        self.eq_jac = eq_jac if eq_jac else lambda x: np.array([]).reshape(0, len(x))
        self.ineq_cons = ineq_cons if ineq_cons else lambda x: np.array([])
        self.ineq_jac = ineq_jac if ineq_jac else lambda x: np.array([]).reshape(0, len(x))

class ALMSolver:
    """
    增广拉格朗日乘子法 (Augmented Lagrangian Method)
    采用 PHR (Powell-Hestenes-Rockafellar) 框架
    """
    def __init__(self, problem, max_iter=2000, tol=1e-6, patience=50):
        self.p = problem
        self.max_iter = max_iter
        self.tol = tol
        self.mu = 10.0  # 初始罚参数
        self.beta = 2.0 # 罚参数增长因子
        self.patience = patience  # 早停机制

    def solve(self, x0):
        x = np.array(x0, dtype=float)
        # 初始化乘子
        n_eq = len(self.p.eq_cons(x))
        n_ineq = len(self.p.ineq_cons(x))
        lam_eq = np.zeros(n_eq)
        lam_ineq = np.zeros(n_ineq)
        
        history = {'f_val': [], 'constr_viol': [], 'grad_norm': [], 'iter': [],'viol_eq': [], 'viol_ineq': [],  'x_path': [x.copy()], 'status': 'MaxIter'}

        best_residual = float('inf')
        no_improve_count = 0
        min_improvement = 1e-4  # 定义什么叫“有优化”：至少下降 0.01%

        for k in range(self.max_iter):
            # 1. 定义增广拉格朗日函数 (子问题目标)
            def aug_lagrangian(x_curr):
                f = self.p.fun(x_curr)
                eqs = self.p.eq_cons(x_curr)
                ineqs = self.p.ineq_cons(x_curr)
                # 等式惩罚: lambda*c + mu/2 * c^2
                term_eq = np.dot(lam_eq, eqs) + 0.5 * self.mu * np.sum(eqs**2) if n_eq > 0 else 0
                # 不等式惩罚 (PHR形式): 1/2mu * ([max(0, mu*g + lambda)]^2 - lambda^2)
                term_ineq = 0
                if n_ineq > 0:
                    for i in range(n_ineq):
                        term_ineq += (1.0 / (2.0 * self.mu)) * (
                            max(0, self.mu * ineqs[i] + lam_ineq[i])**2 - lam_ineq[i]**2
                        )
                return f + term_eq + term_ineq

            # 2. 求解无约束子问题
            res = minimize(aug_lagrangian, x, method='BFGS', tol=1e-8)
            x_new = res.x
            
            # 3. 计算收敛指标 
            f_val = self.p.fun(x_new)
            grad_norm = np.linalg.norm(self.p.jac(x_new))
            
            eqs_v = self.p.eq_cons(x_new)
            ineqs_v = self.p.ineq_cons(x_new)
            viol_eq = np.linalg.norm(eqs_v, np.inf) if n_eq > 0 else 0
            viol_ineq = np.max(np.maximum(0, ineqs_v)) if n_ineq > 0 else 0
            total_viol = max(viol_eq, viol_ineq)

            # 计算当前综合残差
            current_residual = grad_norm + total_viol

            # 记录历史
            history['f_val'].append(f_val)
            history['grad_norm'].append(grad_norm)
            history['viol_eq'].append(viol_eq)
            history['viol_ineq'].append(viol_ineq)
            history['constr_viol'].append(total_viol)
            history['iter'].append(k)
            history['x_path'].append(x_new.copy())

            
            # 4. 判断终止条件
            # === “50步无优化停止”逻辑 ===
            # 如果当前残差比历史最佳还要好（且下降幅度超过阈值）
            if current_residual < best_residual * (1 - min_improvement):
                best_residual = current_residual
                no_improve_count = 0  # 重置计数器
            else:
                no_improve_count += 1 # 没优化，计数+1
            
            if no_improve_count >= self.patience:
                history['status'] = 'Stalled' # 状态标记为陷入停滞
                break

            if grad_norm <= self.tol and total_viol <= self.tol:
                history['status'] = 'Success'
                x = x_new
                break
            
            # 5. 更新乘子和罚参数
            if n_eq > 0: lam_eq += self.mu * eqs_v
            if n_ineq > 0:
                for i in range(n_ineq):
                    lam_ineq[i] = max(0, lam_ineq[i] + self.mu * ineqs_v[i])
            
            # 若约束改善不理想，增大罚参数
            if k > 0 and total_viol > 0.1 * history['constr_viol'][-2]:
                self.mu = min(self.mu * self.beta, 1e6)
            
            x = x_new

        return x, history

class SQPSolver:
    """
    序列二次规划 (Sequential Quadratic Programming)
    包含 BFGS Hessian 更新和 L1 Merit Function 线搜索
    """
    def __init__(self, problem, max_iter=2000, tol=1e-6, patience=50):
        self.p = problem
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience  # 早停机制

    def solve(self, x0):
        x = np.array(x0, dtype=float)
        n = len(x)
        B = np.eye(n) # 初始 Hessian 近似
        
        history = {'f_val': [], 'constr_viol': [], 'grad_norm': [], 'iter': [], 'viol_eq': [], 'viol_ineq': [], 'x_path': [x.copy()], 'status': 'MaxIter'}

        best_residual = float('inf')
        no_improve_count = 0
        min_improvement = 1e-4  # 定义什么叫“有优化”：至少下降 0.01%

        for k in range(self.max_iter):
            # 1. 计算当前点的指标
            f_val = self.p.fun(x)
            grad = self.p.jac(x)
            grad_norm = np.linalg.norm(grad)
            
            eqs = self.p.eq_cons(x)
            ineqs = self.p.ineq_cons(x)
            viol_eq = np.linalg.norm(eqs, np.inf) if len(eqs) > 0 else 0
            viol_ineq = np.max(np.maximum(0, ineqs)) if len(ineqs) > 0 else 0
            total_viol = max(viol_eq, viol_ineq)

             # 计算当前综合残差
            current_residual = grad_norm + total_viol

            history['f_val'].append(f_val)
            history['viol_eq'].append(viol_eq)
            history['viol_ineq'].append(viol_ineq)
            history['constr_viol'].append(total_viol)
            history['grad_norm'].append(grad_norm)
            history['iter'].append(k)


            # 2. 判断终止条件
            # === “N步无优化停止”逻辑 ===
            # 如果当前残差比历史最佳还要好（且下降幅度超过阈值）
            if current_residual < best_residual * (1 - min_improvement):
                best_residual = current_residual
                no_improve_count = 0  # 重置计数器
            else:
                no_improve_count += 1 # 没优化，计数+1
            
            if no_improve_count >= self.patience:
                history['status'] = 'Stalled' # 状态标记为陷入停滞
                break

            if grad_norm <= self.tol and total_viol <= self.tol and k > 0:
                history['status'] = 'Success'
                break

            # 3. 构造并求解 QP 子问题: min 0.5*d'Bd + g'd  s.t. Linearized Cons
            def qp_obj(d): return 0.5 * d @ B @ d + grad @ d
            def qp_grad(d): return B @ d + grad
            
            J_eq = self.p.eq_jac(x)
            J_ineq = self.p.ineq_jac(x)
            
            cons = []
            if len(eqs) > 0:
                cons.append({'type': 'eq', 'fun': lambda d: J_eq @ d + eqs, 'jac': lambda d: J_eq})
            if len(ineqs) > 0:
                cons.append({'type': 'ineq', 'fun': lambda d: -(J_ineq @ d + ineqs), 'jac': lambda d: -J_ineq})
            
            res_qp = minimize(qp_obj, np.zeros(n), method='SLSQP', jac=qp_grad, constraints=cons, tol=1e-9)
            d = res_qp.x
            
        
            # 如果步长几乎为0，说明到了极限。
            # 如果此时梯度依然很大，我们要把 status 记为 'ConvergedToBoundary' 而非简单 Success
            if np.linalg.norm(d) < 1e-10:
                if total_viol <= self.tol and grad_norm <= self.tol:
                    history['status'] = 'Success'
                elif total_viol <= self.tol:
                    history['status'] = 'ConvergedOnBoundary' # 停在边界，实验标准下不算全成，但数学上是最优
                break

            # 4. 线搜索 (L1 Merit Function)
            mu_merit = 10.0 # 罚参数
            def merit_fun(xt):
                v = np.sum(np.abs(self.p.eq_cons(xt))) + np.sum(np.maximum(0, self.p.ineq_cons(xt)))
                return self.p.fun(xt) + mu_merit * v
            
            alpha = 1.0
            phi_0 = merit_fun(x)
            while alpha > 1e-4:
                if merit_fun(x + alpha * d) < phi_0: break
                alpha *= 0.5
            
            # 5. 更新 B (BFGS)
            x_new = x + alpha * d
            s = x_new - x
            # 严格SQP应使用拉格朗日梯度差分，此处简化为目标梯度差分以保证稳定性
            y = self.p.jac(x_new) - grad
            if s @ y > 1e-10:
                Bs = B @ s
                B = B - np.outer(Bs, Bs)/(s @ Bs) + np.outer(y, y)/(s @ y)
            
            x = x_new
            history['x_path'].append(x.copy())

        return x, history