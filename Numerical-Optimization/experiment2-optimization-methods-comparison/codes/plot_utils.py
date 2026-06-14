# @coding=utf-8
# @Author: Wuxiaozhen
# @File: plot_utils.py
# @Date: 2026-01-05
# @Description: 数值优化实验二绘图工具包
# 包含：2D轨迹绘制、收敛性分析（单点/网格）、性能对比柱状图等。
# 采用 Science/Professional 风格，适用于学术报告。

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings

# ==========================================
# 0. 基础设置
# ==========================================
if not os.path.exists('figures'): 
    os.makedirs('figures')

# 设置通用绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 优先中文，兼顾英文
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# ==========================================
# 1. 2D 轨迹绘制函数
# ==========================================
def plot_2d_trajectory_individual(problem, results_list):
    """
    绘制2D问题的优化路径图。
    背景为目标函数等高线，叠加等式(红)和不等式(蓝)约束边界。
    """
    if problem.dim != 2: 
        return
    
    # --- 准备网格数据 ---
    x1_range = np.linspace(-2.0, 2.5, 200)
    x2_range = np.linspace(-1.0, 3.5, 200)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    
    # 计算网格点函数值
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i,j] = problem.fun([X1[i,j], X2[i,j]])

    # 获取所有针对2D问题的结果
    results_2d = [res for res in results_list if res['Problem'] == "2D"]
    # 提取唯一的起始点ID
    unique_ids = sorted(list(set([res['Start_ID'] for res in results_2d])))

    print(f"正在生成 {len(unique_ids)} 张 2D 轨迹图...")

    for start_id in unique_ids:
        # 获取该起点的 ALM 和 SQP 结果
        group = [res for res in results_2d if res['Start_ID'] == start_id]
        if not group: continue
        
        start_coord = group[0]['Start_Coord'] # 获取坐标字符串用于标题
        
        plt.figure(figsize=(9, 7), dpi=120)
        
        # 1. 绘制等高线 (背景)
        cp = plt.contour(X1, X2, Z, levels=30, cmap='viridis', alpha=0.5, linewidths=0.8)
        plt.colorbar(cp, label='Objective Function $f(x)$')
        
        # 2. 绘制约束边界
        # 等式约束 c1(x) = 0 (红色实线)
        plt.contour(X1, X2, X1**2 + X2 - 2, levels=[0], colors='red', linewidths=2.5, alpha=0.8)
        # 不等式约束 c2(x) <= 0 (蓝色虚线)
        plt.contour(X1, X2, X1 + X2**2 - 3, levels=[0], colors='blue', linewidths=2.5, linestyles='--')
        
        # 3. 绘制算法轨迹
        colors = {'ALM': 'black', 'SQP': '#D62828'} # 黑 vs 深红
        markers = {'ALM': 'o', 'SQP': '^'}
        
        for res in group:
            path = np.array(res['History']['x_path'])
            algo = res['Algo']
            
            # 绘制线条
            plt.plot(path[:, 0], path[:, 1], color=colors.get(algo, 'k'), 
                     label=f'{algo} Path', linewidth=1.5, alpha=0.8)
            # 绘制节点
            plt.plot(path[:, 0], path[:, 1], color=colors.get(algo, 'k'), 
                     marker=markers.get(algo, 'o'), markersize=4, linestyle='None', alpha=0.6)
            
            # 标注起点 (绿色方块)
            plt.scatter(path[0, 0], path[0, 1], c='green', marker='s', s=80, zorder=5, edgecolors='w')
            # 标注终点 (对应颜色星号)
            plt.scatter(path[-1, 0], path[-1, 1], c=colors.get(algo, 'k'), marker='*', s=180, zorder=5, edgecolors='w')

        # 4. 装饰图表
        # 自定义图例元素
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='red', lw=2),
            Line2D([0], [0], color='blue', lw=2, ls='--'),
            Line2D([0], [0], color='green', marker='s', lw=0),
            Line2D([0], [0], color='black', lw=2),
            Line2D([0], [0], color='#D62828', lw=2)
        ]
        plt.legend(custom_lines, ['Eq Cons ($c_1=0$)', 'Ineq Cons ($c_2 \leq 0$)', 'Start', 'ALM', 'SQP'], 
                   loc='upper left', bbox_to_anchor=(1.2, 1), frameon=True)
        
        plt.title(f'Trajectory - Start {start_id}\n{start_coord}', fontsize=12)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        
        filename = f"figures/path_2D_{start_id}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

# ==========================================
# 2. 精细化收敛分析图 (Combined vs Component)
# ==========================================
def plot_convergence_refined(results_list, figsize=(14, 6)):
    """
    绘制收敛曲线。
    左图：综合收敛指标 (Grad + Viol)
    右图：分项指标对比 (实线:约束违背, 虚线:梯度范数)
    """
    # 颜色配置: ALM(蓝), SQP(红)
    colors = {'ALM': '#2E86AB', 'SQP': '#A23B72'} 
    
    unique_combinations = sorted(list(set(
        (res['Problem'], res['Start_ID'], res['Start_Coord']) for res in results_list
    )))
    
    print(f"正在生成 {len(unique_combinations)} 张收敛分析对比图...")

    for prob, start_id, coord in unique_combinations:
        group = [res for res in results_list 
                if res['Problem'] == prob and res['Start_ID'] == start_id]
        if not group: continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=120)
        
        # 强制 ALM 在前，SQP 在后 (图例统一)
        algos = sorted([res['Algo'] for res in group])
        if 'ALM' in algos: 
            algos.remove('ALM'); algos.insert(0, 'ALM')

        # --- 左图：综合指标 (KKT Residual) ---
        for algo in algos:
            res = next(r for r in group if r['Algo'] == algo)
            hist = res['History']
            c_color = colors.get(algo, 'k')
            
            grad = np.array(hist['grad_norm'])
            viol = np.array(hist['constr_viol'])
            combined = grad + viol + 1e-15 # 避免 log(0)
            
            ax1.semilogy(hist['iter'], combined, color=c_color, linewidth=2, 
                         label=f"{algo} (Final: {combined[-1]:.1e})")

       # === 左图：PDF 要求指标 ===
        ax1.set_title(f"Combined Metric \n{prob} | Start {start_id} | Coordinates: {coord}", fontweight='bold')
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel(r"$\|\nabla f\| + \|c(x)\|$")
        ax1.set_yscale('log')
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.axhline(y=1e-6, color='gray', linestyle=':', label='Tol 1e-6')

        # --- 右图：分项指标 (Viol vs Grad) ---
        for algo in algos:
            res = next(r for r in group if r['Algo'] == algo)
            hist = res['History']
            c_color = colors.get(algo, 'k')
            
            # 约束违背 (实线)
            ax2.semilogy(hist['iter'], np.array(hist['constr_viol']) + 1e-15, 
                         color=c_color, linestyle='-', linewidth=2, label=f"{algo} Viol")
            # 梯度范数 (虚线)
            ax2.semilogy(hist['iter'], np.array(hist['grad_norm']) + 1e-15, 
                         color=c_color, linestyle='--', linewidth=1.5, alpha=0.7, label=f"{algo} Grad")

        ax2.set_title(f"Component Analysis\n(Solid: Viol, Dashed: Grad)", fontweight='bold')
        ax2.set_xlabel("Iterations")
        ax2.grid(True, which='both', linestyle='--', alpha=0.4)
        # 分栏显示图例
        ax2.legend(loc='upper right', ncol=2, framealpha=0.9, fontsize=9)
        
        # 统一设置Y轴范围，防止极小值导致的显示问题
        for ax in [ax1, ax2]:
            ax.set_ylim(bottom=1e-10)

        plt.tight_layout()
        save_path = f"figures/conv_dual_{prob}_{start_id}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# ==========================================
# 3. 网格化收敛图 (Grid View)
# ==========================================
def plot_convergence_grid(results_list, problem, problems_per_row=2, figsize_per_subplot=(6, 4)):
    """
    将多个收敛曲线放在一个网格中对比
    
    参数:
        results_list: 包含实验结果的字典列表
        problem: 当前要绘制的具体问题名称 (如 'Prob1')
        problems_per_row: 每行显示的子图数量
        figsize_per_subplot: 每个子图的尺寸
    """
    # 1. 筛选当前问题的结果
    results_list = [res for res in results_list if res['Problem'] == problem]
    
    # 2. 获取唯一的问题-起点组合 (去重)
    unique_combinations = []
    seen = set()
    
    for res in results_list:
        key = (res['Problem'], res['Start_ID'])
        if key not in seen:
            seen.add(key)
            # 保存 (Problem, Start_ID, Start_Coord) 用于后续绘图
            unique_combinations.append((res['Problem'], res['Start_ID'], res['Start_Coord']))
    
    # ==========================================
    # 关键修改：自定义排序逻辑 (解决 P1, P10, P2 问题)
    # ==========================================
    try:
        # x[1] 是 Start_ID (例如 "P1")
        # x[1][1:] 取出 "P" 后面的字符串 (例如 "1")
        # int() 将其转换为整数，按数字大小排序 -> P1, P2, ..., P9, P10
        unique_combinations.sort(key=lambda x: int(x[1][1:]))
    except ValueError:
        # 如果 ID 格式不是 "P+数字" (例如 "StartA"), 则回退到默认的字典序排序
        unique_combinations.sort()
    
    # 3. 计算网格布局 (行数和列数)
    n_plots = len(unique_combinations)
    n_cols = min(problems_per_row, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols  # 向上取整计算行数
    
    # 4. 创建画布
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * figsize_per_subplot[0], 
                                    n_rows * figsize_per_subplot[1]),
                            constrained_layout=True, dpi=100)
    
    # 确保 axes 始终是数组格式，方便后续遍历
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 绘图样式配置
    colors = ['#2E86AB', '#A23B72']  # 蓝色 (ALM), 紫红色 (SQP)
    markers = ['o', 's']             # 圆圈, 方块
    line_styles = ['-', '--']        # 实线, 虚线
    
    # 5. 遍历每个起点进行绘图
    for idx, (prob, start_id, coord) in enumerate(unique_combinations):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        # 获取当前起点下的所有结果 (包含 ALM 和 SQP)
        group = [res for res in results_list 
                if res['Problem'] == prob and res['Start_ID'] == start_id]
        
        if not group:
            continue
        
        # 分别绘制 ALM 和 SQP 的曲线
        for algo_idx, algo in enumerate(['ALM', 'SQP']):
            algo_results = [r for r in group if r['Algo'] == algo]
            if not algo_results:
                continue
                
            res = algo_results[0]
            hist = res['History']
            
            # 计算收敛指标: log10(梯度范数 + 约束违背量)
            grad_norm = np.array(hist['grad_norm'])
            constr_viol = np.array(hist['constr_viol'])
            metric = np.log10(grad_norm + constr_viol + 1e-12) # 加 1e-12 防止 log(0)
            
            # 绘制半对数曲线
            ax.semilogy(hist['iter'], 10**metric,
                       color=colors[algo_idx],
                       marker=markers[algo_idx],
                       markersize=3,
                       markevery=max(1, len(hist['iter'])//10), # 稀疏显示标记点，避免拥挤
                       linestyle=line_styles[algo_idx],
                       linewidth=1.5,
                       label=f"{algo} (F={res['Final_F']:.2e})",
                       alpha=0.8)
        
        # 设置子图样式
        ax.set_title(f"{prob} - Start {start_id}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        
        # 添加收敛阈值参考线 (10^-6)
        ax.axhline(y=1e-6, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylim(bottom=1e-8)
    
    # 6. 隐藏多余的空白子图 (如果 plot 数量填不满网格)
    for idx in range(len(unique_combinations), len(axes)):
        axes[idx].set_visible(False)
    
    # 添加总标题并保存
    plt.suptitle("Convergence Comparison Across Different Start Points", 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(f"figures/convergence_grid_comparison_{problem}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 4. 迭代次数/性能对比图 (Bar Chart)
# ==========================================
def plot_performance_with_avg(results_list):
    """
    绘制每个问题下，各初始点的迭代次数对比柱状图，并计算平均值。
    """
    problems = sorted(list(set([res['Problem'] for res in results_list])))
    colors = ['#4A90E2', '#E25A5A'] # 蓝 红
    
    print("正在生成性能对比柱状图...")

    for prob in problems:
        group = [res for res in results_list if res['Problem'] == prob]
        # 修改后：同样提取数字进行排序
        raw_starts = list(set([res['Start_ID'] for res in group]))
        starts = sorted(raw_starts, key=lambda s: int(s[1:]) if s.startswith('P') else s)
        algos = ['ALM', 'SQP']
        
        # 准备数据
        data = {algo: [] for algo in algos}
        for algo in algos:
            for s in starts:
                # 找到对应结果，如果没有则设为0
                match = [res for res in group if res['Start_ID'] == s and res['Algo'] == algo]
                val = match[0]['Iter'] if match else 0
                data[algo].append(val)
            # 计算平均值
            data[algo].append(np.mean(data[algo]))
        
        labels = starts + ["AVG"]
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
        
        # 绘制柱状图
        for i, algo in enumerate(algos):
            bars = ax.bar(x + i*width - width/2, data[algo], width, 
                         label=algo, color=colors[i], alpha=0.9, edgecolor='white')
            # 标数值
            ax.bar_label(bars, fmt='%.0f', padding=2, fontsize=8)
            
        # 突出均值区域
        ax.axvspan(len(starts)-0.5, len(starts)+0.5, color='gray', alpha=0.1, zorder=0)
        ax.text(len(starts), ax.get_ylim()[1]*0.95, "Average", ha='center', fontweight='bold', color='gray')

        ax.set_title(f'Iteration Count Comparison: {prob}', fontweight='bold', pad=15)
        ax.set_ylabel('Total Iterations')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"figures/perf_bar_{prob}.png", dpi=300)
        plt.close()

# ==========================================
# 5. 全局对比汇总图
# ==========================================
def plot_all_problems_comparison(results_list):
    """
    将所有问题（2D, 4D等）的性能对比放在一张大图中。
    """
    print("正在生成全局对比汇总图...")
    problems = sorted(list(set([res['Problem'] for res in results_list])))
    
    fig, axes = plt.subplots(1, len(problems), figsize=(6*len(problems), 5), dpi=120)
    if len(problems) == 1: axes = [axes]
    
    colors = ['#2E86AB', '#A23B72']
    
    for idx, prob in enumerate(problems):
        ax = axes[idx]
        group = [res for res in results_list if res['Problem'] == prob]
        starts = sorted(list(set([res['Start_ID'] for res in group])))
        algos = ['ALM', 'SQP']
        
        # 仅对比平均值
        means = []
        for algo in algos:
            vals = [res['Iter'] for res in group if res['Algo'] == algo]
            means.append(np.mean(vals))
            
        # 绘制简单柱状图
        ax.bar(algos, means, color=colors, alpha=0.8, width=0.5)
        
        # 标数值
        for i, v in enumerate(means):
            ax.text(i, v + 0.5, f"{v:.1f}", ha='center', va='bottom', fontweight='bold')
            
        ax.set_title(f"{prob} Average Iterations", fontweight='bold')
        ax.set_ylabel("Avg Iterations")
        ax.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("figures/global_comparison_summary.png", dpi=300)
    plt.close()

print("plot_utils.py 模块加载成功。")