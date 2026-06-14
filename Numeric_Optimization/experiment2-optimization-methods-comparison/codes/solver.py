
# @coding=utf-8
# @Author: Wuxiaozhen
# @File: solver.py
# @Date: 2026-01-05


import numpy as np
from scipy.optimize import minimize

class ConstrainedProblem:
    """
    问题包装类 (Problem Wrapper)
    功能：统一管理目标函数、梯度、约束条件及其雅可比矩阵，防止在算法内部处理形状不一致的问题。
    """
    def __init__(self, fun, jac, eq_cons=None, eq_jac=None, ineq_cons=None, ineq_jac=None):
        """
        :param fun: 目标函数 f(x)
        :param jac: 目标函数梯度 df/dx
        :param eq_cons: 等式约束函数列表 c_eq(x) = 0
        :param eq_jac: 等式约束雅可比 dc_eq/dx
        :param ineq_cons: 不等式约束函数列表 c_ineq(x) >= 0 (注意：此处通常约定为 >= 0)
        :param ineq_jac: 不等式约束雅可比 dc_ineq/dx
        """
        self.fun = fun
        self.jac = jac
        # 如果未提供约束，给予默认的空返回函数，防止报错
        self.eq_cons = eq_cons if eq_cons else lambda x: np.array([])
        self.eq_jac = eq_jac if eq_jac else lambda x: np.array([]).reshape(0, len(x))
        self.ineq_cons = ineq_cons if ineq_cons else lambda x: np.array([])
        self.ineq_jac = ineq_jac if ineq_jac else lambda x: np.array([]).reshape(0, len(x))

class ALMSolver:
    """
    增广拉格朗日乘子法 (Augmented Lagrangian Method)
    
    核心思想：
    将约束优化问题转化为一系列无约束优化子问题。
    通过引入乘子(Lambda)和罚参数(Mu)，构造增广拉格朗日函数 L_aug(x)。
    
    框架：
    采用 PHR (Powell-Hestenes-Rockafellar) 形式处理不等式约束。
    """
    def __init__(self, problem, max_iter=2000, tol=1e-6, patience=50):
        self.p = problem
        self.max_iter = max_iter
        self.tol = tol
        self.mu = 10.0  # 初始罚参数 (Penalty Parameter)
        self.beta = 2.0 # 罚参数增长因子，用于当约束违背没有明显改善时增大惩罚力度
        self.patience = patience  # 早停机制：允许目标函数不下降的步数

    def solve(self, x0):
        x = np.array(x0, dtype=float)
        n_features = len(x)
        
        # 初始化乘子 (Lagrange Multipliers)
        # 初始时假设没有惩罚，乘子为0
        n_eq = len(self.p.eq_cons(x))
        n_ineq = len(self.p.ineq_cons(x))
        lam_eq = np.zeros(n_eq)
        lam_ineq = np.zeros(n_ineq)
        
        # 初始化历史记录字典，新增 'aug_grad_norm' 和 'grad_vec'
        history = {
            'f_val': [], 
            'constr_viol': [], 
            'grad_norm': [],        # 原始目标函数的梯度模长
            'aug_grad_norm': [],    # 增广拉格朗日函数的梯度模长（子问题收敛依据）
            'grad_vec': [],         # 记录每一步的目标函数梯度向量
            'iter': [],
            'viol_eq': [], 
            'viol_ineq': [],  
            'x_path': [x.copy()], 
            'status': 'MaxIter'
        }

        best_residual = float('inf')
        no_improve_count = 0
        min_improvement = 1e-4

        for k in range(self.max_iter):
            # ==========================================================
            # 1. 构造增广拉格朗日函数 (Obj) 及其梯度 (Grad)
            #    L(x) = f(x) + <lambda, c(x)> + (mu/2)||c(x)||^2  (针对等式)
            #    PHR形式针对不等式 g(x)>=0: 
            #    项为 (1/2mu) * ( [max(0, lambda + mu*g(x))]^2 - lambda^2 )
            # ==========================================================
            
            def aug_lagrangian(x_curr):
                f = self.p.fun(x_curr)
                eqs = self.p.eq_cons(x_curr)
                ineqs = self.p.ineq_cons(x_curr)
                
                # 等式部分：拉格朗日项 + 二次罚项
                term_eq = 0
                if n_eq > 0:
                    term_eq = np.dot(lam_eq, eqs) + 0.5 * self.mu * np.sum(eqs**2)
                
                # 不等式部分 (PHR)
                term_ineq = 0
                if n_ineq > 0:
                    # 向量化计算: term = 1/(2mu) * (max(0, mu*g + lam)^2 - lam^2)
                    shifted = self.mu * ineqs + lam_ineq
                    term_ineq = (1.0 / (2.0 * self.mu)) * np.sum(np.maximum(0, shifted)**2 - lam_ineq**2)
                
                return f + term_eq + term_ineq

            def aug_lagrangian_grad(x_curr):
                """
                计算增广拉格朗日函数的解析梯度
                用于传给 minimize 求解器，大幅提升求解精度和速度，
                并用于返回给用户查看子问题是否真正收敛。
                """
                g_f = self.p.jac(x_curr) # 目标梯度
                
                # 等式梯度: sum( (lam + mu*c) * grad_c )
                g_eq = np.zeros_like(g_f)
                if n_eq > 0:
                    eqs = self.p.eq_cons(x_curr)
                    J_eq = self.p.eq_jac(x_curr)
                    # 系数向量
                    coeffs = lam_eq + self.mu * eqs
                    # 矩阵乘法: coeffs @ J_eq (注意维度: (m,) @ (m, n) -> (n,))
                    if J_eq.shape[0] == n_eq: # 确保维度正确
                        g_eq = np.dot(coeffs, J_eq)

                # 不等式梯度: sum( max(0, mu*g + lam) * grad_g )
                g_ineq = np.zeros_like(g_f)
                if n_ineq > 0:
                    ineqs = self.p.ineq_cons(x_curr)
                    J_ineq = self.p.ineq_jac(x_curr)
                    # 只有当 (mu*g + lam) > 0 时才有梯度，系数为 (mu*g + lam) * 1/mu * mu = (mu*g + lam)
                    # 实际上导数是 max(0, mu*g + lam)
                    coeffs = np.maximum(0, self.mu * ineqs + lam_ineq)
                    if J_ineq.shape[0] == n_ineq:
                        g_ineq = np.dot(coeffs, J_ineq)
                
                return g_f + g_eq + g_ineq

            # ==========================================================
            # 2. 求解无约束子问题
            #    传入解析梯度 jac=aug_lagrangian_grad
            # ==========================================================
            # 【修改点1+2：传入解析梯度 jac=aug_lagrangian_grad，以及阈值改成1e-6】
            # res = minimize(aug_lagrangian, x, jac=aug_lagrangian_grad, method='BFGS', tol=1e-6)
            res = minimize(aug_lagrangian, x, method='BFGS', tol=1e-6)
            x_new = res.x
            
            # 3. 计算收敛指标 
            f_val = self.p.fun(x_new)
            grad_f_vec = self.p.jac(x_new)
            grad_norm = np.linalg.norm(grad_f_vec) # 原始函数梯度模长
            
            # 计算子问题梯度模长（理论上应趋近于0）
            aug_grad_vec = aug_lagrangian_grad(x_new)
            aug_grad_norm = np.linalg.norm(aug_grad_vec)
            
            # 计算违约度
            eqs_v = self.p.eq_cons(x_new)
            ineqs_v = self.p.ineq_cons(x_new)
            viol_eq = np.linalg.norm(eqs_v, np.inf) if n_eq > 0 else 0
            viol_ineq = np.max(np.maximum(0, -ineqs_v)) if n_ineq > 0 else 0 # 假设 g(x)>=0, 则违约为 -g(x)>0的部分
            # 这里沿用原代码的违约度计算逻辑：viol = max(0, ineqs) 表示约束是 ineqs <= 0。
            viol_ineq_val = np.max(np.maximum(0, ineqs_v)) if n_ineq > 0 else 0
            
            total_viol = max(viol_eq, viol_ineq_val)

            # 综合残差用于早停
            current_residual = grad_norm + total_viol

            # 记录历史
            history['f_val'].append(f_val)
            history['grad_norm'].append(grad_norm)
            history['aug_grad_norm'].append(aug_grad_norm) # 【新增】记录AL函数梯度
            history['grad_vec'].append(grad_f_vec.copy()) # 保存梯度向量
            history['viol_eq'].append(viol_eq)
            history['viol_ineq'].append(viol_ineq_val)
            history['constr_viol'].append(total_viol)
            history['iter'].append(k)
            history['x_path'].append(x_new.copy())

            # 4. 判断终止条件
            # 逻辑：如果连续N步（patience）残差没有显著下降（1e-4），则认为陷入停滞
            if current_residual < best_residual * (1 - min_improvement):
                best_residual = current_residual
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= self.patience:
                history['status'] = 'Stalled'
                break

            # 成功条件：原始梯度小 且 约束满足
            if grad_norm <= self.tol and total_viol <= self.tol:
                history['status'] = 'Success'
                x = x_new
                break
            
            # ==========================================================
            # 5. 更新乘子和罚参数
            #    乘子更新规则：lambda_new = lambda + mu * c(x)
            # ==========================================================
            if n_eq > 0: 
                lam_eq += self.mu * eqs_v
            if n_ineq > 0:
                for i in range(n_ineq):
                    # 对应 PHR 的乘子更新
                    lam_ineq[i] = max(0, lam_ineq[i] + self.mu * ineqs_v[i])
            
            # 若约束违背程度改善不明显（小于上一轮的10%），则增大罚参数 mu
            # 【修改点 3】: 放弃激进的罚参数更新策略 
            # if k > 0 and total_viol > 0.9 * history['constr_viol'][-2]:
            #     self.mu = min(self.mu * self.beta, 1e6)
            
            if k > 0 and total_viol > 0.1 * history['constr_viol'][-2]:
                self.mu = min(self.mu * self.beta, 1e6)
            x = x_new

        return x, history

class SQPSolver:
    """
    序列二次规划 (Sequential Quadratic Programming)
    
    核心思想：
    在每一步迭代点 x_k 处，将原非线性问题近似为一个二次规划(QP)问题：
      min  1/2 d^T B d + \nabla f^T d
      s.t. \nabla c_eq^T d + c_eq = 0
           \nabla c_ineq^T d + c_ineq >= 0
    其中 B 是 Hessian 的近似矩阵（这里使用 BFGS 更新）。
    """
    def __init__(self, problem, max_iter=2000, tol=1e-6, patience=50):
        self.p = problem
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience

    def solve(self, x0):
        x = np.array(x0, dtype=float)
        n = len(x)
        B = np.eye(n) # Hessian 近似矩阵，初始化为单位阵
        
        # 初始化历史记录，新增 'grad_vec'
        history = {
            'f_val': [], 
            'constr_viol': [], 
            'grad_norm': [], 
            'grad_vec': [], # 记录每一步的目标函数梯度向量
            'iter': [], 
            'viol_eq': [], 
            'viol_ineq': [], 
            'x_path': [x.copy()], 
            'status': 'MaxIter'
        }

        best_residual = float('inf')
        no_improve_count = 0
        min_improvement = 1e-4

        for k in range(self.max_iter):
            # 1. 计算当前点的基础信息
            f_val = self.p.fun(x)
            grad = self.p.jac(x) # 目标函数梯度
            grad_norm = np.linalg.norm(grad)
            
            eqs = self.p.eq_cons(x)
            ineqs = self.p.ineq_cons(x)
            
            # 计算约束违背度
            viol_eq = np.linalg.norm(eqs, np.inf) if len(eqs) > 0 else 0
            viol_ineq = np.max(np.maximum(0, ineqs)) if len(ineqs) > 0 else 0 # 假设 g(x) <= 0
            total_viol = max(viol_eq, viol_ineq)

            current_residual = grad_norm + total_viol

            # 记录数据
            history['f_val'].append(f_val)
            history['viol_eq'].append(viol_eq)
            history['viol_ineq'].append(viol_ineq)
            history['constr_viol'].append(total_viol)
            history['grad_norm'].append(grad_norm)
            history['grad_vec'].append(grad.copy()) # 保存梯度向量
            history['iter'].append(k)

            # 2. 判断终止条件
            if current_residual < best_residual * (1 - min_improvement):
                best_residual = current_residual
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= self.patience:
                history['status'] = 'Stalled'
                break

            if grad_norm <= self.tol and total_viol <= self.tol and k > 0:
                history['status'] = 'Success'
                break

            # ==========================================================
            # 3. 构造 QP 子问题
            #    目标：min 0.5*d'Bd + g'd
            #    约束：线性化后的原问题约束
            # ==========================================================
            def qp_obj(d): 
                return 0.5 * d @ B @ d + grad @ d
            
            # QP子问题的梯度 (针对变量 d)
            def qp_grad(d): 
                return B @ d + grad
            
            J_eq = self.p.eq_jac(x)
            J_ineq = self.p.ineq_jac(x)
            
            cons = []
            # 线性化等式: J*d + c = 0
            if len(eqs) > 0:
                cons.append({'type': 'eq', 'fun': lambda d: J_eq @ d + eqs, 'jac': lambda d: J_eq})
            # 线性化不等式: J*d + c <= 0 -> -(J*d + c) >= 0
            if len(ineqs) > 0:
                cons.append({'type': 'ineq', 'fun': lambda d: -(J_ineq @ d + ineqs), 'jac': lambda d: -J_ineq})
            
            # 使用 SLSQP 求解 QP 问题寻找搜索方向 d
            res_qp = minimize(qp_obj, np.zeros(n), method='SLSQP', jac=qp_grad, constraints=cons, tol=1e-9)
            d = res_qp.x
            
            # 边界情况处理：步长极小
            if np.linalg.norm(d) < 1e-10:
                if total_viol <= self.tol:
                    history['status'] = 'ConvergedOnBoundary' # 此时可能在边界最优
                else:
                    history['status'] = 'FeasibilityIssue' # 无法找到可行下降方向
                break

            # ==========================================================
            # 4. 线搜索 (Line Search) - L1 Merit Function
            #    用于确保步长 alpha 不仅能降低目标函数，也能降低约束违背
            # ==========================================================
            mu_merit = 10.0 # 评价函数的罚参数
            def merit_fun(xt):
                v = np.sum(np.abs(self.p.eq_cons(xt))) + np.sum(np.maximum(0, self.p.ineq_cons(xt)))
                return self.p.fun(xt) + mu_merit * v
            
            alpha = 1.0
            phi_0 = merit_fun(x)
            # Armijo 回溯
            while alpha > 1e-4:
                if merit_fun(x + alpha * d) < phi_0: break
                alpha *= 0.5
            
            # ==========================================================
            # 5. 更新 Hessian 近似矩阵 B (BFGS Update)
            #    利用梯度差分信息来修正二阶近似
            # ==========================================================
            x_new = x + alpha * d
            s = x_new - x
            # 注意：标准的 SQP BFGS 更新应该使用拉格朗日函数的梯度差
            # 这里简化为目标函数梯度差，对于某些非凸问题可能不稳定，但在一般工程问题中可用
            y = self.p.jac(x_new) - grad
            
            # 保证正定性校验 (s^T y > 0)
            if s @ y > 1e-10:
                Bs = B @ s
                # BFGS 公式
                B = B - np.outer(Bs, Bs)/(s @ Bs) + np.outer(y, y)/(s @ y)
            
            x = x_new
            history['x_path'].append(x.copy())

        return x, history