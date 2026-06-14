# @coding=utf-8
# @Author: Wuxiaozhen
# @File: plot_utils.py
# @Date: 2026-01-05
# @Description: 数值优化实验二绘图工具代码，包含各种绘图函数的实现

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, BFGS
import pandas as pd
import warnings
import os
# 基础设置
if not os.path.exists('figures'): os.makedirs('figures')
plt.rcParams['font.sans-serif'] = ['sans-serif'] # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 绘图函数模块
# ==========================================
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

    # unique_starts = sorted(list(set([res['Start_ID'] for res in results_list if res['Problem'] == "2D"])))
    unique_ids = sorted(list(set([(res['Start_ID'], res['Start_Coord']) for res in results_list if res['Problem'] == "2D"])))

    # for start_id in unique_starts:
    for start_id, coord in unique_ids:
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
        
        plt.title(f'Optimization Trajectory - Start Point {start_id} {coord}')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1))
        plt.grid(True, alpha=0.2)
        
        filename = f"figures/path_2D_{start_id}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved trajectory plot: {filename}")

# def plot_convergence_refined(results_list):
#     """每个初始点一张图，包含坐标信息，对比 ALM 和 SQP"""
#     unique_ids = sorted(list(set([(res['Problem'], res['Start_ID'], res['Start_Coord']) for res in results_list])))
    
#     for prob, start_id, coord in unique_ids:
#         group = [res for res in results_list if res['Problem'] == prob and res['Start_ID'] == start_id]
        
#         fig, ax = plt.subplots(figsize=(10, 6))
#         # 统一收敛指标：log10(||grad_L|| + ||constr_viol||)
#         for res in group:
#             hist = res['History']
#             metric = np.array(hist['grad_norm']) + np.array(hist['constr_viol']) + 1e-12
#             ax.semilogy(hist['iter'], metric, label=f"{res['Algo']} (Final F: {res['Final_F']:.4f})", linewidth=2)
            
#         ax.set_title(f"Convergence Comparison: {prob} at {start_id} {coord}\nMetric: log10(||grad_L|| + ||viol||)")
#         ax.set_xlabel("Iteration")
#         ax.set_ylabel("Residual Norm")
#         ax.grid(True, which="both", ls="--", alpha=0.5)
#         ax.legend()
        
#         plt.tight_layout()
#         plt.savefig(f"figures/conv_{prob}_{start_id}.png", dpi=300)
#         plt.close()

# def plot_performance_with_avg(results_list):
#     """性能汇总图：包含每个点对比 + 全局均值对比"""
#     problems = sorted(list(set([res['Problem'] for res in results_list])))
    
#     for prob in problems:
#         group = [res for res in results_list if res['Problem'] == prob]
#         starts = sorted(list(set([res['Start_ID'] for res in group])))
#         algos = ['ALM', 'SQP']
        
#         # 准备数据
#         data = {algo: [] for algo in algos}
#         for algo in algos:
#             for s in starts:
#                 val = next(res['Iter'] for res in group if res['Start_ID'] == s and res['Algo'] == algo)
#                 data[algo].append(val)
#             # 计算均值
#             data[algo].append(np.mean(data[algo]))
        
#         labels = starts + ["AVG"]
#         x = np.arange(len(labels))
#         width = 0.35
        
#         fig, ax = plt.subplots(figsize=(12, 6))
#         for i, algo in enumerate(algos):
#             bars = ax.bar(x + i*width - width/2, data[algo], width, label=algo)
#             # 在条形图上方标注数值
#             ax.bar_label(bars, padding=3)
            
#         # 突出显示均值列
#         ax.axvspan(len(starts)-0.5, len(starts)+0.5, color='gray', alpha=0.1)
        
#         ax.set_ylabel('Iterations')
#         ax.set_title(f'Computational Efficiency Comparison: {prob}')
#         ax.set_xticks(x, labels)
#         ax.legend()
#         plt.savefig(f"figures/summary_{prob}.png", dpi=300)
#         plt.close()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.patches import FancyBboxPatch
import matplotlib as mpl

# def plot_convergence_refined(results_list, style='science', figsize=(12, 7)):
#     """
#     收敛曲线图：对比 ALM 和 SQP
    
#     参数：
#     ----------
#     results_list : list
#         包含结果的字典列表
#     style : str, optional
#         绘图风格，可选 'science', 'elegant', 'dark', 'minimal'
#     figsize : tuple, optional
#         图形大小
#     """
    
#     # 设置不同风格
#     if style == 'science':
#         # 科学论文风格
#         colors = ['#2E86AB', '#A23B72']  # ALM, SQP
#         bg_color = '#FFFFFF'
#         grid_color = "#B9B3B3"
#         text_color = '#2C3E50'
#         line_styles = ['-', '--']
#         line_widths = [2.5, 2.2]
        
#     elif style == 'elegant':
#         # 优雅风格
#         colors = ['#006D77', '#E76F51']  # 深青 + 珊瑚橙
#         bg_color = '#FAF9F6'
#         grid_color = '#E5E5E5'
#         text_color = '#2A2D34'
#         line_styles = ['-', '-.']
#         line_widths = [2.3, 2.0]
        
#     elif style == 'dark':
#         # 深色风格
#         colors = ['#4ECDC4', '#FF6B6B']  # 青绿 + 珊瑚红
#         bg_color = '#2C3E50'
#         grid_color = '#34495E'
#         text_color = '#ECF0F1'
#         line_styles = ['-', '--']
#         line_widths = [2.5, 2.2]
        
#     else:  # minimal
#         # 简约风格
#         colors = ['#3498DB', '#E74C3C']  # 蓝 + 红
#         bg_color = '#FFFFFF'
#         grid_color = '#BDC3C7'
#         text_color = '#2C3E50'
#         line_styles = ['-', '--']
#         line_widths = [2.0, 1.8]
    
#     # 设置字体
#     plt.rcParams['font.size'] = 11
#     plt.rcParams['axes.titlesize'] = 13
#     plt.rcParams['axes.labelsize'] = 12
#     plt.rcParams['xtick.labelsize'] = 10
#     plt.rcParams['ytick.labelsize'] = 10
#     plt.rcParams['legend.fontsize'] = 10
    
#     # 获取唯一的问题-起点组合
#     unique_combinations = []
#     seen = set()
    
#     for res in results_list:
#         key = (res['Problem'], res['Start_ID'], res['Start_Coord'])
#         if key not in seen:
#             seen.add(key)
#             unique_combinations.append(key)
    
#     unique_combinations = sorted(unique_combinations)
    
#     for prob, start_id, coord in unique_combinations:
#         # 筛选该起点的所有结果
#         group = [res for res in results_list 
#                 if res['Problem'] == prob and res['Start_ID'] == start_id]
        
#         if not group:
#             continue
        
#         # 创建图形
#         fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
#         if style == 'dark':
#             fig.patch.set_facecolor(bg_color)
#             ax.set_facecolor(bg_color)
        
#         # 提取算法顺序，确保一致性
#         algo_order = ['ALM', 'SQP']
#         present_algos = sorted(list(set([res['Algo'] for res in group])))
        
#         # 绘制收敛曲线
#         lines = []
#         labels = []
        
#         for idx, algo in enumerate(algo_order):
#             if algo not in present_algos:
#                 continue
                
#             res = next(r for r in group if r['Algo'] == algo)
#             hist = res['History']
            
#             # 计算收敛指标
#             grad_norm = np.array(hist['grad_norm'])
#             constr_viol = np.array(hist['constr_viol'])
#             metric = np.log10(grad_norm + constr_viol + 1e-12)
            
#             # 绘制收敛曲线
#             line, = ax.semilogy(hist['iter'], 10**metric,  # 转回线性坐标绘图
#                              color=colors[idx],
#                              linestyle=line_styles[idx],
#                              linewidth=line_widths[idx],
#                              marker='o' if style == 'science' else None,
#                              markersize=4 if style == 'science' else 0,
#                              markevery=5,  # 每隔5个点显示一个标记
#                              alpha=0.9)
#             lines.append(line)
            
#             # 创建标签，包含更多信息
#             final_iter = hist['iter'][-1] if hist['iter'] else 0
#             final_metric = metric[-1] if len(metric) > 0 else 0
#             label = (f"{algo}: "
#                     f"F={res['Final_F']:.4e}, "
#                     f"N={final_iter}, "
#                     f"Conv={final_metric:.2e}")
#             labels.append(label)
        
#         # 添加收敛阈值线
#         threshold = 1e-6
#         ax.axhline(y=threshold, color='gray', linestyle=':', 
#                   alpha=0.9, linewidth=2.5, label=f'Threshold ({threshold:.0e})')
#         lines.append(ax.lines[-1])
#         labels.append(f'Threshold ({threshold:.0e})')
        
#         # 标题
#         title_text = f"Convergence Analysis: {prob}\n"
#         title_text += f"Start Point: {start_id} | Coordinates: {coord}"
#         ax.set_title(title_text, fontweight='bold', pad=20, color=text_color)
        
#         # 设置坐标轴标签
#         ax.set_xlabel("Iteration Number", fontweight='bold', labelpad=10, color=text_color)
#         ax.set_ylabel(r"Residual: $\|\nabla L\| + \|c\|$", 
#                      fontweight='bold', labelpad=10, color=text_color)
        
#         # 设置对数坐标
#         ax.set_yscale('log')
        
#         # 设置Y轴刻度为10的幂次方格式
#         from matplotlib.ticker import LogFormatterSciNotation
        
#         # 方法1：使用科学计数法格式
#         ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        
#         # 或者方法2：使用简单的10^n格式
#         # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))
        
#         # 或者方法3：使用标准格式
#         # formatter = ticker.LogFormatterMathtext()
#         # formatter.labelOnlyBase = False  # 显示所有刻度标签
#         # ax.yaxis.set_major_formatter(formatter)
        
#         # 设置网格
#         ax.grid(True, which='major', axis='both', 
#                 linestyle='-', linewidth=1, alpha=0.4, 
#                 color=grid_color, zorder=0)
#         ax.grid(True, which='minor', axis='both', 
#                 linestyle='--', linewidth=0.5, alpha=0.5, 
#                 color=grid_color, zorder=0)
        
#         ax.set_axisbelow(True)  # 将网格放在数据下方
        
#         # 设置合适的Y轴范围
#         all_y_values = []
#         for res in group:
#             hist = res['History']
#             grad_norm = np.array(hist['grad_norm'])
#             constr_viol = np.array(hist['constr_viol'])
#             metric = np.log10(grad_norm + constr_viol + 1e-12)
#             all_y_values.extend(10**metric)
        
#         if all_y_values:
#             y_min = min(all_y_values)
#             y_max = max(all_y_values)
            
#             # 找到最近的10的幂次方
#             y_min_log = np.floor(np.log10(y_min))
#             y_max_log = np.ceil(np.log10(y_max))
            
#             # 确保至少显示3个数量级
#             if y_max_log - y_min_log < 2:
#                 y_min_log = y_max_log - 3
#                 y_min_log = max(y_min_log, -12)  # 避免太小
            
#             # ax.set_ylim(10**y_min_log, 10**y_max_log)
#             ax.set_ylim(1e-8, 1e4)
            
#             # 设置主刻度为10的整数次幂
#             ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
            
#             # # 设置次刻度
#             # subs = [2, 3, 4, 5, 6, 7, 8, 9]
#             # ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=subs, numticks=15))

#         # 优化图例
#         legend = ax.legend(lines, labels, 
#                           loc='upper right',
#                           fontsize=9,
#                           frameon=True,
#                           fancybox=True,
#                           shadow=(style != 'minimal'),
#                           edgecolor=grid_color,
#                           borderpad=1,
#                           labelspacing=0.8,
#                           handlelength=3)
        
#         if style == 'dark':
#             legend.get_frame().set_facecolor('#34495E')
#             legend.get_frame().set_alpha(0.9)
#         else:
#             legend.get_frame().set_facecolor('white')
#             legend.get_frame().set_alpha(0.9)
        
#         # 调整布局
#         plt.tight_layout()
        
#         # 保存图片
#         save_path = f"figures/conv_{prob}_{start_id}.png"
#         plt.savefig(save_path, dpi=300, bbox_inches='tight', 
#                    facecolor=fig.get_facecolor(), edgecolor='none')
        
#         if style == 'dark':
#             plt.savefig(save_path.replace('.png', '_dark.png'), dpi=300, 
#                        bbox_inches='tight', facecolor=fig.get_facecolor())
        
#         plt.close()

def plot_convergence_refined(results_list, style='science', figsize=(15, 6)):
    """
    收敛曲线图：对比 ALM 和 SQP
    布局：左图为混合指标(PDF要求)，右图为分项指标(梯度vs约束)
    """
    
    # --- 1. 风格设置 ---
    if style == 'science':
        colors = ['#2E86AB', '#A23B72']  # ALM(蓝), SQP(红紫)
        bg_color = '#FFFFFF'
        grid_color = "#B9B3B3"
        text_color = '#2C3E50'
    elif style == 'dark':
        colors = ['#4ECDC4', '#FF6B6B']
        bg_color = '#2C3E50'
        grid_color = '#34495E'
        text_color = '#ECF0F1'
    else: # minimal/elegant
        colors = ['#0077B6', '#D62828']
        bg_color = '#FFFFFF'
        grid_color = '#E5E5E5'
        text_color = '#2A2D34'

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.facecolor': bg_color,
        'axes.facecolor': bg_color,
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'axes.edgecolor': grid_color
    })

    # --- 2. 获取唯一组合 ---
    unique_combinations = sorted(list(set(
        (res['Problem'], res['Start_ID'], res['Start_Coord']) for res in results_list
    )))
    
    for prob, start_id, coord in unique_combinations:
        group = [res for res in results_list 
                if res['Problem'] == prob and res['Start_ID'] == start_id]
        if not group: continue
        
        # --- 3. 创建图形 (1行2列) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=150)
        
        # 算法列表
        present_algos = sorted(list(set([res['Algo'] for res in group])))
        # 强制排序 ALM 在前
        if 'ALM' in present_algos: 
            present_algos.remove('ALM'); present_algos.insert(0, 'ALM')
            
        threshold = 1e-6
        
        # === 左图：PDF 要求指标 ===
        ax1.set_title(f"Combined Metric \n{prob} | Start {start_id} | Coordinates: {coord}", fontweight='bold')
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel(r"$\|\nabla f\| + \|c(x)\|$")
        ax1.set_yscale('log')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5, color=grid_color)
        
        # === 右图：分项指标分析 ===
        ax2.set_title(f"Component Analysis\n(Solid: Viol, Dashed: Grad)", fontweight='bold')
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Magnitude (Log Scale)")
        ax2.set_yscale('log')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5, color=grid_color)
        
        # 绘制
        lines_1, labels_1 = [], []
        lines_2, labels_2 = [], []
        
        for idx, algo in enumerate(present_algos):
            res = next(r for r in group if r['Algo'] == algo)
            hist = res['History']
            color = colors[idx % len(colors)]
            
            grad = np.array(hist['grad_norm'])
            viol = np.array(hist['constr_viol'])
            combined = grad + viol + 1e-12
            iters = hist['iter']
            
            # --- 左图绘制 ---
            l1, = ax1.plot(iters, combined, color=color, linewidth=2.0, 
                           label=f"{algo} (Final: {combined[-1]:.1e})")
            
            # --- 右图绘制 (分项) ---
            # 约束违反度 (实线)
            l2_v, = ax2.plot(iters, viol + 1e-12, color=color, linestyle='-', linewidth=2.0,
                             label=f"{algo} Viol")
            # 梯度范数 (虚线)
            l2_g, = ax2.plot(iters, grad + 1e-12, color=color, linestyle='--', linewidth=1.5, alpha=0.7,
                             label=f"{algo} Grad")
        
        # 添加阈值线 (只在右图画显眼的阈值线)
        ax2.axhline(y=threshold, color='gray', linestyle=':', linewidth=2, label=f'Tol ({threshold:.0e})')
        
        # 统一 Y 轴范围美化
        for ax in [ax1, ax2]:
            ax.set_ylim(1e-10, 1e5)
            # 使用更科学的刻度显示
            locmaj = ticker.LogLocator(base=10, numticks=12)
            ax.yaxis.set_major_locator(locmaj)
            locmin = ticker.LogLocator(base=10, subs=np.arange(0.1, 1.0, 0.1), numticks=12)
            ax.yaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        
        # 图例
        ax1.legend(loc='upper right', frameon=True, fontsize=9)
        ax2.legend(loc='upper right', frameon=True, fontsize=8, ncol=2) # 分列显示图例
        
        plt.tight_layout()
        
        # 保存
        save_path = f"figures/conv_dual_{prob}_{start_id}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print("Convergence figures generated in 'figures/' folder.")

# 可选：创建多子图对比版本
def plot_convergence_grid(results_list, problem, problems_per_row=2, figsize_per_subplot=(6, 4)):
    """
    将多个收敛曲线放在一个网格中对比
    """
    results_list = [res for res in results_list if res['Problem'] == problem]
    # 获取唯一的问题-起点组合
    unique_combinations = []
    seen = set()
    
    for res in results_list:
        key = (res['Problem'], res['Start_ID'])
        if key not in seen:
            seen.add(key)
            unique_combinations.append((res['Problem'], res['Start_ID'], res['Start_Coord']))
    
    # unique_combinations = sorted(unique_combinations) # 不是按字符串进行排列，而是1-10
    unique_combinations.sort(key=lambda x: int(x[1][1:]) if x[1].startswith('P') else x[1])
    
    n_plots = len(unique_combinations)
    n_cols = min(problems_per_row, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * figsize_per_subplot[0], 
                                    n_rows * figsize_per_subplot[1]),
                            constrained_layout=True, dpi=100)
    
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 设置颜色
    colors = ['#2E86AB', '#A23B72']
    markers = ['o', 's']
    line_styles = ['-', '--']
    
    for idx, (prob, start_id, coord) in enumerate(unique_combinations):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        group = [res for res in results_list 
                if res['Problem'] == prob and res['Start_ID'] == start_id]
        
        if not group:
            continue
        
        # 绘制每个算法的收敛曲线
        for algo_idx, algo in enumerate(['ALM', 'SQP']):
            algo_results = [r for r in group if r['Algo'] == algo]
            if not algo_results:
                continue
                
            res = algo_results[0]
            hist = res['History']
            
            grad_norm = np.array(hist['grad_norm'])
            constr_viol = np.array(hist['constr_viol'])
            metric = np.log10(grad_norm + constr_viol + 1e-12)
            
            ax.semilogy(hist['iter'], 10**metric,
                       color=colors[algo_idx],
                       marker=markers[algo_idx],
                       markersize=3,
                       markevery=max(1, len(hist['iter'])//10),
                       linestyle=line_styles[algo_idx],
                       linewidth=1.5,
                       label=f"{algo} (F={res['Final_F']:.2e})",
                       alpha=0.8)
        
        # 设置子图
        ax.set_title(f"{prob} - Start {start_id}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        
        # 添加收敛阈值
        ax.axhline(y=1e-6, color='gray', linestyle=':', alpha=0.5)
        
        # 设置坐标轴范围
        ax.set_ylim(bottom=1e-8)
    
    # 隐藏多余的子图
    for idx in range(len(unique_combinations), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Convergence Comparison Across Different Start Points", 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(f"figures/convergence_grid_comparison_{problem}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_with_avg(results_list, style='professional'):
    """
    增强版性能汇总图：包含优雅的配色和美观的视觉效果
    
    参数：
    ----------
    results_list : list
        包含结果的字典列表
    style : str, optional
        绘图风格，可选 'professional', 'elegant', 'vibrant', 'minimal'
    """
    
    # 设置不同风格
    if style == 'professional':
        # 专业蓝调风格
        colors = ['#2E86AB', '#A23B72']  # 深蓝 + 深粉紫
        bg_color = '#F8F9FA'
        avg_color = '#6C757D'
        text_color = '#343A40'
        grid_color = '#DEE2E6'
    elif style == 'elegant':
        # 优雅绿色风格
        colors = ['#006D77', '#E29578']  # 深青 + 橙红
        bg_color = '#FAF9F6'
        avg_color = '#8A817C'
        text_color = '#2A2D34'
        grid_color = '#E5E5E5'
    elif style == 'vibrant':
        # 鲜艳风格
        colors = ['#FF6B6B', '#4ECDC4']  # 珊瑚红 + 青绿
        bg_color = '#F7FFF7'
        avg_color = '#FFE66D'
        text_color = '#1A535C'
        grid_color = '#E0F2F1'
    else:  # minimal
        # 简约风格
        colors = ['#3A506B', '#5C5D8D']  # 灰蓝 + 紫灰
        bg_color = '#FFFFFF'
        avg_color = '#B8B8B8'
        text_color = '#1C1C1C'
        grid_color = '#F0F0F0'
    
    # 设置字体
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 获取问题列表
    problems = sorted(list(set([res['Problem'] for res in results_list])))
    
    for prob in problems:
        group = [res for res in results_list if res['Problem'] == prob]
        # 修改后：同样提取数字进行排序
        raw_starts = list(set([res['Start_ID'] for res in group]))
        starts = sorted(raw_starts, key=lambda s: int(s[1:]) if s.startswith('P') else s)
        # starts = sorted(list(set([res['Start_ID'] for res in group])))
        algos = ['ALM', 'SQP']
        
        # 准备数据
        data = {algo: [] for algo in algos}
        for algo in algos:
            for s in starts:
                val = next(res['Iter'] for res in group if res['Start_ID'] == s and res['Algo'] == algo)
                data[algo].append(val)
            # 计算均值
            data[algo].append(np.mean(data[algo]))
        
        labels = starts + ["AVG"]
        x = np.arange(len(labels))
        width = 0.36  # 稍微调整宽度
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(13, 7), dpi=100)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # 绘制条形图
        bars_dict = {}
        for i, algo in enumerate(algos):
            bars = ax.bar(x + i*width - width/2, data[algo], width, 
                         color=colors[i], 
                         edgecolor='white', 
                         linewidth=1.5,
                         alpha=0.9,
                         label=algo,
                         zorder=3)
            bars_dict[algo] = bars
            
            # 添加数值标签，使用白色边框增强可读性
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(data[algo]),
                        f'{height:.0f}', 
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        color=text_color,
                        bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor='white', 
                                 edgecolor='none', 
                                 alpha=0.8),
                        zorder=4)
        
        # 突出显示均值列 - 使用渐变色背景
        ax.axvspan(len(starts)-0.5, len(starts)+0.5, 
                  color=avg_color, alpha=0.15, zorder=1)
        
        # 在均值列上方添加标注
        avg_x = len(starts)
        for i, algo in enumerate(algos):
            avg_val = data[algo][-1]
            ax.annotate(f'Mean: {avg_val:.1f}', 
                       xy=(avg_x + i*width - width/2, avg_val),
                       xytext=(0, 20), 
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold',
                       color=colors[i],
                       arrowprops=dict(arrowstyle='->', 
                                     color=colors[i], 
                                     alpha=0.7, lw=1))
        
        # 添加参考线
        for i, start in enumerate(starts):
            ax.axvline(i + 0.5, color=grid_color, linestyle='-', 
                      alpha=0.3, linewidth=0.5, zorder=1)
        
        # 设置坐标轴
        ax.set_ylabel('Iterations', fontsize=12, fontweight='bold', labelpad=10, color=text_color)
        ax.set_xlabel('Start Points', fontsize=12, fontweight='bold', labelpad=10, color=text_color)
        
        # 设置标题
        ax.set_title(f'Computational Efficiency Comparison: {prob}', 
                    fontsize=14, fontweight='bold', pad=20, color=text_color)
        
        # 设置刻度
        ax.set_xticks(x, labels)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        
        # 添加网格
        ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5, color=grid_color, zorder=0)
        ax.set_axisbelow(True)
        
        # 设置图例在右上角
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
                         fontsize=11, frameon=True, fancybox=True,
                         shadow=True, edgecolor=grid_color, borderpad=1)
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_facecolor('white')
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
            spine.set_linewidth(0.8)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(f"figures/summary_{prob}.png", dpi=300, bbox_inches='tight', facecolor=bg_color)
        plt.savefig(f"figures/summary_{prob}.pdf", bbox_inches='tight', facecolor=bg_color)
        plt.close()

# 附加功能：创建对比图，一次性展示所有问题
def plot_all_problems_comparison(results_list, style='professional'):
    """创建所有问题的对比子图"""
    
    problems = sorted(list(set([res['Problem'] for res in results_list])))
    n_problems = len(problems)
    
    # 根据问题数量确定子图布局
    n_cols = min(3, n_problems)
    n_rows = (n_problems + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), dpi=100)
    axes = axes.flatten() if n_problems > 1 else [axes]
    
    # 设置颜色
    colors = ['#2E86AB', '#A23B72']
    
    for idx, prob in enumerate(problems):
        ax = axes[idx]
        if idx >= n_problems:  # 隐藏多余的子图
            ax.set_visible(False)
            continue
            
        group = [res for res in results_list if res['Problem'] == prob]
        # starts = sorted(list(set([res['Start_ID'] for res in group])))
        raw_starts = list(set([res['Start_ID'] for res in group]))
        starts = sorted(raw_starts, key=lambda s: int(s[1:]) if s.startswith('P') else s)
        algos = ['ALM', 'SQP']
        
        # 准备数据
        data = {algo: [] for algo in algos}
        for algo in algos:
            for s in starts:
                val = next(res['Iter'] for res in group if res['Start_ID'] == s and res['Algo'] == algo)
                data[algo].append(val)
            data[algo].append(np.mean(data[algo]))
        
        labels = starts + ["AVG"]
        x = np.arange(len(labels))
        width = 0.35
        
        # 绘制条形图
        for i, algo in enumerate(algos):
            bars = ax.bar(x + i*width - width/2, data[algo], width, 
                         color=colors[i], alpha=0.8, label=algo)
            
            # 标注数值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(data[algo]),
                        f'{height:.0f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')
        
        # 设置子图
        ax.set_title(f'{prob}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x, labels)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        # 只在第一个子图添加图例
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"figures/all_problems_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()