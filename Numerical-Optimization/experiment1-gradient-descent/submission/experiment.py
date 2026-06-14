# @coding=utf-8
# @Author: Wuxiaozhen
# @File: experiment.py
# @Date: 2025-12-15
# @Description: 数值优化实验一代码，实现多种优化算法并进行对比实验

# ==========================================
# 导入必要的安装包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar
import time
from matplotlib.colors import LogNorm


#================ 定义问题的父类Problem，存储必要的函数、维度、梯度和海森矩阵信息==============================
class Problem:
    """
    统一问题的表示，避免维度混乱。子类应实现 func, grad, hess。
    Attributes:
        name (str): 问题名称，用于输出与图例。
        dim (int): 自变量的维度。
    """
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def func(self, x):pass
    def grad(self, x):pass
    def hess(self, x):pass


#================ 定义两个测试函数实例==============================
# 直接定义计算好的梯度和海森矩阵。
# 2 维多极值函数
class ModifiedRosenbrock2D(Problem):
    """2D 的修正 Rosenbrock 函数（带额外正弦耦合项）。
    函数: f(x1,x2) = (1-x1)^2 + 100(x2-x1^2)^2 + 5*sin(2π x1)*sin(2π x2)
    """
    def __init__(self):
        super().__init__("Modified Rosenbrock (2D)", 2) # 继承父类Problem，传入当前问题的名称和维度
    # 函数
    def func(self, x):
        x1, x2 = x[0], x[1]
        term1 = (1 - x1)**2
        term2 = 100 * (x2 - x1**2)**2
        term3 = 5 * np.sin(2 * np.pi * x1) * np.sin(2 * np.pi * x2)
        return term1 + term2 + term3
    # 梯度
    def grad(self, x):
        x1, x2 = x[0], x[1]
        df_dx1 = -2*(1 - x1) - 400*x1*(x2 - x1**2) + \
                 10*np.pi * np.cos(2*np.pi*x1) * np.sin(2*np.pi*x2)
        df_dx2 = 200*(x2 - x1**2) + \
                 10*np.pi * np.sin(2*np.pi*x1) * np.cos(2*np.pi*x2)
        return np.array([df_dx1, df_dx2])
    # 海森矩阵
    def hess(self, x):
        x1, x2 = x[0], x[1]
        h11 = 2 - 400*(x2 - x1**2) + 800*x1**2 - \
              20 * (np.pi**2) * np.sin(2*np.pi*x1) * np.sin(2*np.pi*x2)
        h12 = -400*x1 + 20 * (np.pi**2) * np.cos(2*np.pi*x1) * np.cos(2*np.pi*x2)
        h21 = h12
        h22 = 200 - 20 * (np.pi**2) * np.sin(2*np.pi*x1) * np.sin(2*np.pi*x2)
        return np.array([[h11, h12], [h21, h22]])
# 4 维耦合函数
class ExtendedRosenbrock4D(Problem):
    """扩展的 Rosenbrock（4D）函数。
    函数: sum_{k=0}^{n-2} (1-x_k)^2 + 100(x_{k+1}-x_k^2)^2
    """
    def __init__(self):
        super().__init__("Extended Rosenbrock (4D)", 4) # 继承父类Problem，传入当前问题的名称和维度

    # 函数
    def func(self, x):
        val = 0
        for k in range(self.dim - 1):
            val += (1 - x[k])**2 + 100 * (x[k+1] - x[k]**2)**2
        return val
    # 梯度
    def grad(self, x):
        g = np.zeros_like(x)
        # 对每一对相邻变量累加贡献
        n = self.dim
        for k in range(n - 1):
            g[k] += -2*(1 - x[k]) - 400*x[k]*(x[k+1] - x[k]**2)
            g[k+1] += 200*(x[k+1] - x[k]**2)
        return g
    # 海森矩阵
    def hess(self, x):
        n = self.dim
        H = np.zeros((n, n))
        for k in range(n - 1):
            # 对 x_k, x_k
            H[k, k] += 2 - 400*(x[k+1] - x[k]**2) + 800*x[k]**2
            H[k, k+1] += -400*x[k]
            H[k+1, k] += -400*x[k]
            # 对 x_{k+1}, x_{k+1}
            H[k+1, k+1] += 200
        return H


#===========定义通用的线搜索策略 (包含精确线搜索Exact & 非精确线搜索Inexact)=========
class LineSearch:
    """
    线搜索策略类
    包含了精确线搜索和非精确线搜索（Armijo-Wolfe）的实现。
    """
    @staticmethod
    def exact(func, x, d):
        """
        精确线搜索 (Exact Line Search)
        目标：找到步长 alpha，使得 f(x + alpha * d) 达到局部最小。
        
        参数:
            func: 目标函数
            x: 当前迭代点
            d: 搜索方向
        
        返回:
            最优步长 alpha
        """
        # 使用 Brent 方法在射线方向上寻找标量函数的最小值
        # lambda alpha: func(x + alpha * d) 将多元函数转化为关于步长 alpha 的一元函数
        res = minimize_scalar(lambda alpha: func(x + alpha * d), method='brent')
        return res.x

    @staticmethod
    def armijo_wolfe(func, grad, x, d, f_val, g_val, c1=1e-4, c2=0.9):
        """
        非精确线搜索 (Inexact Line Search)
        尝试满足强 Wolfe 条件 (Strong Wolfe Conditions)。
        如果失败，则回退到简单的 Armijo 回溯。

        参数:
            func: 目标函数
            grad: 梯度函数
            x: 当前点
            d: 搜索方向
            f_val: 当前点的函数值 f(x)
            g_val: 当前点的梯度值 g(x)
            c1: Armijo 条件参数 (充分下降)
            c2: Curvature 条件参数 (曲率条件)
        
        返回:
            步长 alpha
        """
        alpha = 1.0  # 初始步长通常设为 1 (特别是对于牛顿类方法)
        rho = 0.5    # 回溯时的缩减比例
        
        # 尝试使用 scipy 优化库中的 line_search (实现了强 Wolfe 条件)
        from scipy.optimize import line_search
        # gfk 是当前点的梯度，old_fval 是当前函数值
        res = line_search(func, grad, x, d, gfk=g_val, old_fval=f_val, c1=c1, c2=c2)
        
        # 如果 scipy 找到了满足条件的 alpha (res[0] 不为 None)
        if res[0] is not None:
            return res[0]
        else:
            # Fallback (回退方案): 简单的 Armijo 回溯搜索
            # 当强 Wolfe 条件难以满足时，仅保证函数值充分下降
            while alpha > 1e-8:
                # Armijo 条件: f(x + alpha*d) <= f(x) + c1 * alpha * g^T d
                if func(x + alpha * d) <= f_val + c1 * alpha * np.dot(g_val, d):
                    return alpha
                alpha *= rho # 步长减半
            # 如果实在找不到，返回一个极小值避免程序崩溃
            return 1e-5


#===========定义通用的五种优化算法===================

def steepest_descent(problem, x0, ls_type='inexact', max_iter=1000, tol=1e-6):
    """
    最速下降法 (Steepest Descent Method)
    
    原理：沿着梯度的反方向（函数下降最快的方向）进行搜索。
    
    参数:
        - problem: 问题对象，包含 .func(), .grad() 方法
        - x0: 初始点
        - ls_type: 线搜索类型 ('exact' 或 'inexact')
        - max_iter: 最大迭代次数
        - tol: 收敛容差 (梯度的范数)

    返回:
        - x : 最终解。
        - k : 最后一次迭代索引（从0起）。
        - history : 记录历史信息，包含 'f', 'gnorm', 'x'。
    """
    x = x0.copy()
    # 用于记录优化过程的历史数据
    history = {'f': [], 'gnorm': [], 'x': []}
    
    for k in range(max_iter):
        g = problem.grad(x)          # 计算当前梯度
        gnorm = np.linalg.norm(g)    # 计算梯度范数（模长）
        f_val = problem.func(x)      # 计算当前函数值
        
        # 记录日志
        history['f'].append(f_val)
        history['gnorm'].append(gnorm)
        history['x'].append(x.copy())
        
        # 终止条件判定：梯度趋近于 0
        if gnorm < tol: break
        
        # 确定搜索方向：负梯度方向
        d = -g
        
        # 确定步长 (线搜索)
        if ls_type == 'exact':
            alpha = LineSearch.exact(problem.func, x, d)
        else:
            alpha = LineSearch.armijo_wolfe(problem.func, problem.grad, x, d, f_val, g)

        # 更新迭代点
        x = x + alpha * d
        
    return x, k, history

def newton_method(problem, x0, ls_type='inexact', max_iter=1000, tol=1e-6):
    """
    经典牛顿法 (Newton's Method)
    
    原理：利用二阶泰勒展开，利用 Hessian 矩阵信息加速收敛。
    需要处理 Hessian 矩阵可能不正定或奇异的情况。

    参数:
        - problem: 问题对象，包含 .func(), .grad() 方法
        - x0: 初始点
        - ls_type: 线搜索类型 ('exact' 或 'inexact')
        - max_iter: 最大迭代次数
        - tol: 收敛容差 (梯度的范数)

    返回:
        - x : 最终解。
        - k : 最后一次迭代索引（从0起）。
        - history : 记录历史信息，包含 'f', 'gnorm', 'x'
    
    """
    x = x0.copy()
    history = {'f': [], 'gnorm': [], 'x': []}
    
    for k in range(max_iter):
        g = problem.grad(x)
        gnorm = np.linalg.norm(g)
        f_val = problem.func(x)
        history['f'].append(f_val)
        history['gnorm'].append(gnorm)
        history['x'].append(x.copy())
        
        if gnorm < tol: break
        
        H = problem.hess(x) # 计算 Hessian 矩阵
        
        # 求解牛顿方程 H * d = -g
        # 为了实验稳定性，增加了阻尼项
        try:
            # np.eye(len(x)) 生成单位矩阵 I
            # H + 1e-6 * I 防止矩阵奇异（不可逆）或不正定
            d = np.linalg.solve(H + 1e-6 * np.eye(len(x)), -g)
        except np.linalg.LinAlgError:
            # 如果 Hessian 实在太差导致无法求解，退化为负梯度方向
            d = -g 
            
        # 线搜索
        if ls_type == 'exact':
            alpha = LineSearch.exact(problem.func, x, d)
        else:
            alpha = LineSearch.armijo_wolfe(problem.func, problem.grad, x, d, f_val, g)
            
        x = x + alpha * d
        
    return x, k, history

def bfgs_method(problem, x0, ls_type='inexact', max_iter=1000, tol=1e-6):
    """
    BFGS 拟牛顿法 (Quasi-Newton Method)

    原理：不直接计算 Hessian，而是通过梯度的变化来迭代更新 Hessian 的逆矩阵近似 (H_k)。
    优点：收敛速度接近牛顿法，但不需要计算二阶导数。

    说明:
      - 初始 H_k 取单位矩阵。
      - 使用标准 BFGS 公式更新 H_k 的逆近似：
        H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T
      - 当 s^T y 很小时跳过更新以避免数值不稳定。

    参数:
        - problem: 问题对象，包含 .func(), .grad() 方法
        - x0: 初始点
        - ls_type: 线搜索类型 ('exact' 或 'inexact')
        - max_iter: 最大迭代次数
        - tol: 收敛容差 (梯度的范数)

    返回:
        - x : 最终解。
        - k : 最后一次迭代索引（从0起）。
        - history : 记录历史信息，包含 'f', 'gnorm', 'x'
    
    """
    x = x0.copy()
    n = len(x)
    H_k = np.eye(n) # 初始化 H_0 为单位矩阵 (近似逆 Hessian)
    history = {'f': [], 'gnorm': [], 'x': []}
    I = np.eye(n)
    
    g = problem.grad(x)
    
    for k in range(max_iter):
        gnorm = np.linalg.norm(g)
        f_val = problem.func(x)
        history['f'].append(f_val)
        history['gnorm'].append(gnorm)
        history['x'].append(x.copy())
        
        if gnorm < tol: break
        
        # 确定搜索方向：d = -H_k * g
        d = -np.dot(H_k, g)
        
        # 线搜索
        if ls_type == 'exact':
            alpha = LineSearch.exact(problem.func, x, d)
        else:
            alpha = LineSearch.armijo_wolfe(problem.func, problem.grad, x, d, f_val, g)
        
        # 记录更新前的点和梯度
        x_new = x + alpha * d
        g_new = problem.grad(x_new)
        
        # 计算位移 s 和 梯度差 y
        s = x_new - x   # 步长向量
        y = g_new - g   # 梯度变化向量
        
        # BFGS 更新公式 (更新 H_k)
        # 公式: H_{k+1} = (I - rho*s*y^T) H_k (I - rho*y*s^T) + rho*s*s^T
        if np.dot(s, y) > 1e-10: # 保证分母不为 0 (曲率条件)
            rho = 1.0 / np.dot(y, s)
            term1 = I - rho * np.outer(s, y)
            term2 = I - rho * np.outer(y, s)
            # 更新矩阵
            H_k = np.dot(term1, np.dot(H_k, term2)) + rho * np.outer(s, s)
        
        # 更新状态
        x = x_new
        g = g_new
        
    return x, k, history

def conjugate_gradient(problem, x0, ls_type='inexact', method='PRP', max_iter=1000, tol=1e-6):
    """非线性共轭梯度法（NLCG），支持 Fletcher-Reeves (FR) 与 Polak-Ribiere-Polyak (PRP) 公式。

    参数:
        - problem: 问题对象，包含 .func(), .grad() 方法
        - x0: 初始点
        - ls_type: 线搜索类型 ('exact' 或 'inexact')
        - method (str): 'FR' 或 'PRP'，决定 beta 的计算方式。
        - max_iter: 最大迭代次数
        - tol: 收敛容差 (梯度的范数)
    
    返回:
        - x : 最终解。
        - k : 最后一次迭代索引（从0起）。
        - history : 记录历史信息，包含 'f', 'gnorm', 'x'。

    关键点:
        - 使用 PRP 时若 beta < 0 则重启（取负梯度方向）。
        - 每步通过线搜索确定步长。
    """
    x = x0.copy()
    history = {'f': [], 'gnorm': [], 'x': []}

    g = problem.grad(x)
    d = -g  # 初始搜索方向为负梯度
    g_old = g.copy()
    d_old = d.copy()

    for k in range(max_iter):
        gnorm = np.linalg.norm(g)
        f_val = problem.func(x)
        history['f'].append(f_val)
        history['gnorm'].append(gnorm)
        history['x'].append(x.copy())

        if gnorm < tol:
            break

        # 计算共轭系数 beta
        if k > 0:
            if method == 'FR':
                # Fletcher-Reeves 公式: beta = |g_k|^2 / |g_{k-1}|^2
                beta = np.dot(g, g) / (np.dot(g_old, g_old) + 1e-10)
            else: 
                # PRP 公式: beta = g_k^T (g_k - g_{k-1}) / |g_{k-1}|^2
                beta = np.dot(g, g - g_old) / (np.dot(g_old, g_old) + 1e-10)
                # PRP 需要 Restart 策略：如果 beta < 0，置为 0（即重置为最速下降）
                beta = max(0, beta) 
            
            # 更新搜索方向: d_k = -g_k + beta * d_{k-1}
            d = -g + beta * d_old
            
            # 检查是否满足下降方向条件 (g^T d < 0)
            # 如果共轭方向导致函数上升，重置为负梯度方向
            if np.dot(g, d) > 0:
                d = -g
        else:
            d = -g

        # 线搜索
        if ls_type == 'exact':
            alpha = LineSearch.exact(problem.func, x, d)
        else:
            alpha = LineSearch.armijo_wolfe(problem.func, problem.grad, x, d, f_val, g)
            
        # 更新
        x = x + alpha * d
        g_old = g.copy() # 保存旧梯度用于下一次计算 beta
        d_old = d.copy() # 保存旧方向
        g = problem.grad(x)
        

    return x, k, history

def trust_region(problem, x0, max_iter=1000, tol=1e-6):
    """
    信赖域算法 (Trust Region Method) - Dogleg 策略

    原理：
    不是先定方向再定步长，而是先在当前点附近划定一个“信赖域”(半径 radius)。
    在信赖域内利用二次模型 m(p) 近似目标函数，求解最优步长 p。
    根据实际下降量和模型预测下降量的比值 (rho) 来调整信赖域半径。

    参数:
        - problem: 问题对象，包含 .func(), .grad() 方法
        - x0: 初始点
        - max_iter: 最大迭代次数
        - tol: 收敛容差 (梯度的范数)
    返回:
        - x : 最终解。
        - k : 最后一次迭代索引（从0起）。
        - history : 记录历史信息，包含 'f', 'gnorm', 'x'。
    """
    x = x0.copy()
    radius = 1.0       # 初始信赖域半径
    max_radius = 10.0  # 最大半径限制
    eta = 0.15         # 接受步长的阈值
    history = {'f': [], 'gnorm': [], 'x': []}
    
    for k in range(max_iter):
        g = problem.grad(x)
        gnorm = np.linalg.norm(g)
        f_val = problem.func(x)
        history['f'].append(f_val)
        history['gnorm'].append(gnorm)
        history['x'].append(x.copy())
        
        if gnorm < tol: break
        
        H = problem.hess(x)
        
        # 求解子问题 (Subproblem) - 使用 Dogleg 方法近似
        # 目标：min m(p) = f + g^T p + 0.5 p^T H p, subject to ||p|| <= radius
        
        # A.计算柯西点 (Cauchy Point, p_u)
        # 这是沿着负梯度方向的最优点（即最速下降方向在信赖域内的最优解）
        gTg = np.dot(g, g)
        gHg = np.dot(g, np.dot(H, g))
        
        # 计算最优步长因子 tau
        if gHg <= 0:
            tau = 1.0 # 曲率为负，尽量走远
        else:
            # 二次函数沿直线的极小值点公式
            tau = min(1.0, gTg**3 / (radius * gHg) if radius>0 else 1) 
            tau = gTg / gHg 
            
        # 柯西点位移
        p_u = -(tau * gTg / (gHg + 1e-10)) * g if gHg > 0 else - (radius/gnorm)*g
        
        # B. 检查是否超出信赖域
        if np.linalg.norm(p_u) > radius:
            p = - (radius / gnorm) * g # 截断在边界上
        else:
            # C. 计算牛顿点 (Newton Point, p_B)
            # 无约束下的二次模型极小值点: H * p = -g
            try:
                p_B = -np.linalg.solve(H, g)
            except:
                p_B = p_u # 求解失败则退回柯西点
                
            # D. Dogleg 路径选择
            # 简单策略：如果牛顿点在信赖域内，选牛顿点；否则简化处理为柯西点
            if np.linalg.norm(p_B) <= radius:
                p = p_B
            else:
                # 理论上应该寻找 p_u 到 p_B 线段与信赖域边界的交点
                # 这里做简化处理：直接使用柯西点（相当于在信赖域受限时退化为最速下降）
                p = p_u 
        
        # 评估模型准确度 (Ratio rho)
        # rho = (实际下降量) / (模型预测下降量)
        actual_reduction = f_val - problem.func(x + p)
        predicted_reduction = -np.dot(g, p) - 0.5 * np.dot(p, np.dot(H, p))
        
        rho = actual_reduction / (predicted_reduction + 1e-10)
        
        # 更新信赖域半径和迭代点
        if rho > eta:
            # 模型预测尚可，接受该步
            x = x + p
            # 如果预测非常准 (rho > 0.75) 且步长已经顶到了边界，说明可以扩大信赖域
            if rho > 0.75 and np.linalg.norm(p) < 0.8 * radius:
                pass 
            elif rho > 0.75:
                radius = min(2 * radius, max_radius) # 半径翻倍
        elif rho < 0.25:
            # 模型预测很差，缩小信赖域
            radius = 0.5 * radius
        # 如果 0.25 <= rho <= eta，半径不变，但不接受该步（x不更新），下一轮用同半径重新计算
            
    return x, k, history


#=======运行实验并收集数据========================

def run_experiment():
    # 初始化问题
    prob2d = ModifiedRosenbrock2D()
    prob4d = ExtendedRosenbrock4D()
    
    # 初始点
    start_points_2d = [
        np.array([-1.2, 1.0]),  # 初始点1 （题目指定）
        np.array([1.2, 0.5]),   # 初始点2
        np.array([-0.5, -0.5])  # 初始点3
    ]
    
    start_points_4d = [
        np.array([-1.2, 1.0, -0.8, 0.6]), # 初始点1 （题目指定）
        np.array([0.0, 0.0, 0.0, 0.0]),   # 初始点2
        np.array([-1.0, -1.0, -1.0, -1.0])# 初始点3
    ]

    algorithms = [
        ("Steepest Descent", steepest_descent),
        ("Newton", newton_method),
        ("BFGS", bfgs_method),
        ("CG-PRP", conjugate_gradient),
        ("CG-FP", conjugate_gradient),
        ("Trust Region", trust_region)
    ]
    
    results = []
    
    # 针对两个函数运行
    for prob, starts in [(prob2d, start_points_2d), (prob4d, start_points_4d)]:
        print(f"对问题{prob.name}进行实验中...")
        
        for x0 in starts:
            for algo_name, algo_func in algorithms:
                # 配置策略
                search_strategies = []
                if algo_name == "Trust Region":
                    search_strategies = ["TR"]
                else:
                    search_strategies = ["Exact", "Inexact"]
                
                for strat in search_strategies:
                    # 运行
                    start_time = time.time()
                    if strat == "TR":
                        x_opt, k, hist = algo_func(prob, x0)
                    else:
                        if algo_name == "CG-PRP":
                            x_opt, k, hist = algo_func(prob, x0, ls_type=strat.lower(), method='PRP')
                        elif algo_name == "CG-FP":
                            x_opt, k, hist = algo_func(prob, x0, ls_type=strat.lower(), method='FR')
                        else:   
                            x_opt, k, hist = algo_func(prob, x0, ls_type=strat.lower())
                    elapsed = time.time() - start_time
                    
                    # 记录
                    final_grad = hist['gnorm'][-1]
                    final_f = hist['f'][-1]
                    converged = final_grad < 1e-6
                    
                    results.append({
                        "Problem": prob.name,
                        "Start Point": str(x0),
                        "Algorithm": algo_name,
                        "Strategy": strat,
                        "Iter": k,
                        "Time(s)": round(elapsed, 4),
                        "Final f": "{:.4f}".format(final_f),
                        "Final |g|": "{:.2e}".format(final_grad),
                        "Converged": converged,
                        "History": hist # 保存用于绘图
                    })

    df = pd.DataFrame(results)
    # df.sort_values(by=["Problem", "Start Point", "Algorithm", "Strategy"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # 保存结果到 CSV 文件
    df.drop(columns=['History']).to_csv("experiment_results_summary.csv", index=False)
 
    # ============可视化与输出 ===========================
    
    # 打印表格 (定量结果)
    print("\n=== Experiment Results Summary ===")
    # 简化显示，去掉History列
    print(df.drop(columns=['History']).to_string())
    
    # 绘制收敛曲线 (定性结果) 
    # 选取题目指定的初始点进行绘图
    target_start_2d = str(start_points_2d[0])
    target_start_4d = str(start_points_4d[0])
    
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.figure(figsize=(12, 6))
    
    # 2D 函数收敛图
    subset_2d = df[(df["Problem"] == prob2d.name) & (df["Start Point"] == target_start_2d)]
    for _, row in subset_2d.iterrows():
        hist = row['History']
        label = f"{row['Algorithm']}-{row['Strategy']}"
        plt.plot(np.log10(np.array(hist['gnorm']) + 1e-20), label=label)
    plt.title(f"Convergence: {prob2d.name}\nStart: {target_start_2d}")
    plt.xlabel("Iterations")
    plt.ylabel("log10(||g||)")
    # plt.legend(fontsize='small')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergence_plot_2d.png", dpi=300)

    plt.figure(figsize=(12, 6))
    subset_4d = df[(df["Problem"] == prob4d.name) & (df["Start Point"] == target_start_4d)]

    for _, row in subset_4d.iterrows():
        hist = row['History']
        label = f"{row['Algorithm']}-{row['Strategy']}"
        # 添加 log10 处理
        plt.plot(np.log10(np.array(hist['gnorm']) + 1e-20), label=label)

    plt.title(f"Convergence: {prob4d.name}\nStart: {target_start_4d}")
    plt.xlabel("Iterations")
    plt.ylabel("log10(||g||)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')

    plt.grid(True)

    # 自动调整布局以容纳图例
    plt.tight_layout()

    # 保存时确保包含所有元素（防止图例被裁剪）
    plt.savefig("convergence_plot_4d.png", dpi=300, bbox_inches='tight')

    # 绘制路径图 (仅针对 2D Modified Rosenbrock) 
    # 绘制等高线
    # x = np.linspace(-1.5, 2.0, 400)
    # y = np.linspace(-0.5, 3.0, 400)
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
   
    # 为了绘图速度，向量化计算Z需要调整func写法或者循环，这里循环比较稳妥
    # 也可以利用numpy broadcasting但func里有索引操作
    # 简单起见，使用循环计算Z
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = prob2d.func(np.array([X[i,j], Y[i,j]]))
            
    plt.figure(figsize=(10, 6))
    # 使用对数色标让等高线更清晰
    # plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet', alpha=0.6)
    # plt.colorbar(label='Objective Value')
    
    # 叠加轨迹 (只选非精确线搜索和TR，避免图太乱)
    subset_path = df[(df["Problem"] == prob2d.name) & 
                     (df["Start Point"] == target_start_2d) & 
                     (df["Strategy"].isin(["Inexact", "TR"]))]
    
    colors = ['r', 'b', 'g', 'm', 'k', 'c']
    markers = ['o', 'x', '^', 's', 'd','o']
    
    for idx, (_, row) in enumerate(subset_path.iterrows()):
        hist = row['History']
        path = np.array(hist['x'])
        label = f"{row['Algorithm']}"
        plt.plot(path[:, 0], path[:, 1], color=colors[idx % len(colors)], 
                 marker=markers[idx % len(markers)], markersize=3, 
                 linestyle='-',  label=label)
        # 标记起点和终点
        plt.plot(path[0, 0], path[0, 1], 'k*')

    plt.plot(-1.2, 1.0, 'k*', markersize=10, label='Start')
    plt.plot(1, 1, 'kX', markersize=10, label='Global Min (1,1)')
    plt.title(f"Optimization Path on {prob2d.name}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    # plt.show()
    plt.savefig("optimization_paths_2d_inexact.png", dpi=300)

    # 叠加轨迹 (只选精确线搜索和TR，避免图太乱)
    plt.figure(figsize=(10, 6))
    # 使用对数色标让等高线更清晰
    # plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet', alpha=0.6)
    # plt.colorbar(label='Objective Value')
    
    # 叠加轨迹 (只选非精确线搜索和TR，避免图太乱)
    subset_path = df[(df["Problem"] == prob2d.name) & 
                     (df["Start Point"] == target_start_2d) & 
                     (df["Strategy"].isin(["Exact"]))]

    colors = ['r', 'b', 'g', 'm', 'k', 'c']
    markers = ['o', 'x', '^', 's', 'd','o']
    
    for idx, (_, row) in enumerate(subset_path.iterrows()):
        hist = row['History']
        path = np.array(hist['x'])
        label = f"{row['Algorithm']}"
        plt.plot(path[:, 0], path[:, 1], color=colors[idx % len(colors)], 
                 marker=markers[idx % len(markers)], markersize=3, 
                 linestyle='-',  label=label)
        # 标记起点和终点
        plt.plot(path[0, 0], path[0, 1], 'k*')

    plt.plot(-1.2, 1.0, 'k*', markersize=10, label='Start')
    plt.plot(1, 1, 'kX', markersize=10, label='Global Min (1,1)')
    plt.title(f"Optimization Path on {prob2d.name}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    # plt.show()
    plt.savefig("optimization_paths_2d_exact.png", dpi=300)

if __name__ == "__main__":
    run_experiment()