#!/usr/bin/env python3
"""
中环水务传感器数据分析 V2
增加：线性回归修正后对比、回归方程、R² 热力图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/pwngwc/Obsidian/PengWC/为知笔记/01_芯视界/中环水务数据/分析报告_20260425/联合数据集")
OUT_DIR = Path("/Users/pwngwc/Obsidian/PengWC/为知笔记/01_芯视界/中环水务数据/分析报告_20260427")
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

EC_FACTOR = 1000

PARAMS = {
    'cod': ('chem_cod', 'sensor_cod', 'COD (mg/L)'),
    'turbidity': ('chem_turbidity', 'sensor_turbidity', '浊度 (NTU)'),
    'temperature': ('chem_temperature', 'sensor_temperature', '温度 (°C)'),
    'ph': ('chem_ph', 'sensor_ph', 'pH'),
    'ec': ('chem_ec', 'sensor_ec', 'EC (μS/cm)'),
    'do': ('chem_do', 'sensor_do', 'DO (mg/L)'),
    'ammonia': ('chem_ammonia', 'sensor_ammonia', '氨氮 (mg/L)'),
    'codmn': ('chem_codmn', 'sensor_codmn', 'CODmn (mg/L)'),
}

# 核心参数（展示用）
KEY_PARAMS = ['cod', 'turbidity', 'ph', 'ec', 'do', 'ammonia']


def load_data():
    data = {}
    for point in ['P1', 'P6', 'P7']:
        df = pd.read_csv(DATA_DIR / f"{point}_联合数据集.csv")
        df['chem_datetime'] = pd.to_datetime(df['chem_datetime'])
        df['sensor_ec_aligned'] = df['sensor_ec'] * EC_FACTOR
        data[point] = df
    return data


def do_regression(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3 or len(set(x)) < 2:
        return None
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    return {
        'slope': slope, 'intercept': intercept,
        'r': r_value, 'r2': r_value**2,
        'p': p_value, 'rmse': np.sqrt(np.mean(residuals**2)),
        'n': len(x), 'x': x, 'y': y, 'y_pred': y_pred,
    }


def get_sensor_data(df, sensor_name, param_key):
    """获取某传感器某参数的化学法和传感器数据"""
    sub = df[df['sensor_name'] == sensor_name]
    chem_col, sensor_col, _ = PARAMS[param_key]
    chem = sub[chem_col].values
    if param_key == 'ec':
        sens = sub['sensor_ec_aligned'].values
    else:
        sens = sub[sensor_col].values
    return chem, sens, sub['chem_datetime'].values


# ═══════════════════════════════════════════════════════════════════
# 图1: R² 热力图（总览）
# ═══════════════════════════════════════════════════════════════════

def plot_r2_heatmap(data):
    points = ['P1', 'P6', 'P7']
    sensors_map = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}
    params_list = list(PARAMS.keys())

    fig, ax = plt.subplots(figsize=(18, 8))
    r2_matrix = []
    row_labels = []
    annot_matrix = []

    for point in points:
        for sensor in sensors_map[point]:
            row = []
            annot_row = []
            for param in params_list:
                chem, sens, _ = get_sensor_data(data[point], sensor, param)
                reg = do_regression(chem, sens)
                if reg:
                    row.append(reg['r2'])
                    annot_row.append(f"R²={reg['r2']:.3f}\ny={reg['slope']:.3f}x+{reg['intercept']:.1f}")
                else:
                    row.append(np.nan)
                    annot_row.append("N/A")
            r2_matrix.append(row)
            row_labels.append(sensor)
            annot_matrix.append(annot_row)

    r2_arr = np.array(r2_matrix)
    im = ax.imshow(r2_arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(params_list)))
    ax.set_xticklabels([PARAMS[p][2].split('(')[0].strip() for p in params_list],
                       rotation=30, ha='right', fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    for i in range(len(row_labels)):
        for j in range(len(params_list)):
            val = r2_arr[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.3 or val > 0.8 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       fontsize=10, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, label='R²', shrink=0.8)
    ax.set_title('各点位传感器 vs 化学法 R² 相关性总览', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'overview_r2_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ R² 热力图")


# ═══════════════════════════════════════════════════════════════════
# 图2: 线性回归修正前后对比（核心图）
# ═══════════════════════════════════════════════════════════════════

def plot_regression_correction(data):
    """对每个点位每个核心参数，画：
    上排：原始传感器 vs 化学法 + 回归线
    下排：回归修正后传感器 vs 化学法 + 1:1线
    """
    points = ['P1', 'P6', 'P7']
    sensors_map = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}

    for point in points:
        sensors = sensors_map[point]
        n_params = len(KEY_PARAMS)

        fig, axes = plt.subplots(2, n_params, figsize=(5*n_params, 10))

        for col, param_key in enumerate(KEY_PARAMS):
            _, _, param_label = PARAMS[param_key]

            for row_idx, sensor in enumerate(sensors):
                chem, sens, _ = get_sensor_data(data[point], sensor, param_key)
                reg = do_regression(chem, sens)

                ax = axes[row_idx, col]

                if not reg:
                    ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{sensor} {param_label}')
                    continue

                # 原始数据散点
                ax.scatter(chem, sens, alpha=0.5, s=25, color='#2196F3', edgecolors='white', linewidth=0.5)

                # 回归线
                x_line = np.linspace(chem.min(), chem.max(), 100)
                y_line = reg['slope'] * x_line + reg['intercept']
                ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8,
                       label=f'y={reg["slope"]:.3f}x+{reg["intercept"]:.1f}')

                # 1:1 线
                lim_min = min(chem.min(), sens.min())
                lim_max = max(chem.max(), sens.max())
                margin = (lim_max - lim_min) * 0.05
                ax.plot([lim_min-margin, lim_max+margin], [lim_min-margin, lim_max+margin],
                       'k--', alpha=0.3, linewidth=1)

                ax.set_xlabel(f'化学法 {param_label}', fontsize=9)
                ax.set_ylabel(f'传感器 {param_label}', fontsize=9)
                ax.set_title(f'{sensor} — R²={reg["r2"]:.3f}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=8, loc='upper left')

        plt.suptitle(f'{point} — 传感器原始数据 vs 化学法（含回归线）',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_regression_raw.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("✅ 回归原始对比图")


def plot_corrected_comparison(data):
    """回归修正后：用 y = (sensor - intercept) / slope 转换后 vs 化学法"""
    points = ['P1', 'P6', 'P7']
    sensors_map = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}

    for point in points:
        sensors = sensors_map[point]
        colors = ['#2196F3', '#FF9800']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for idx, param_key in enumerate(KEY_PARAMS):
            row, col = idx // 3, idx % 3
            _, _, param_label = PARAMS[param_key]
            ax = axes[row, col]

            for si, sensor in enumerate(sensors):
                chem, sens, dt = get_sensor_data(data[point], sensor, param_key)
                reg = do_regression(chem, sens)
                if not reg:
                    continue
                sens_corrected = (sens - reg['intercept']) / reg['slope']
                sort_idx = np.argsort(dt)
                dt_sorted = dt[sort_idx]
                chem_sorted = chem[sort_idx]
                corrected_sorted = sens_corrected[sort_idx]
                ax.plot(dt_sorted, chem_sorted, 'o-', color='black', markersize=4,
                       linewidth=1.5, label='化学法' if si == 0 else '', alpha=0.8)
                ax.plot(dt_sorted, corrected_sorted, 's--', color=colors[si], markersize=3,
                       linewidth=1.2, label=f'{sensor}修正后', alpha=0.7)

            ax.set_ylabel(param_label, fontsize=10)
            ax.set_title(param_label, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.tick_params(axis='x', rotation=30, labelsize=7)

        plt.suptitle(f'{point} — 线性回归修正后传感器 vs 化学法 时序对比',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_corrected_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("✅ 修正后时序对比图")


def plot_corrected_scatter(data):
    """修正后散点图：sensor_corrected vs chem，2行×3列"""
    points = ['P1', 'P6', 'P7']
    sensors_map = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}

    for point in points:
        sensors = sensors_map[point]
        colors = ['#2196F3', '#FF9800']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for idx, param_key in enumerate(KEY_PARAMS):
            row, col = idx // 3, idx % 3
            _, _, param_label = PARAMS[param_key]
            ax = axes[row, col]

            all_chem = []
            for si, sensor in enumerate(sensors):
                chem, sens, _ = get_sensor_data(data[point], sensor, param_key)
                reg = do_regression(chem, sens)
                if not reg:
                    continue
                sens_corrected = (sens - reg['intercept']) / reg['slope']
                ax.scatter(chem, sens_corrected, alpha=0.5, s=25, color=colors[si],
                          edgecolors='white', linewidth=0.5, label=sensor)
                all_chem.extend(chem[~np.isnan(chem)])

            # 1:1 线
            if all_chem:
                lim_min = min(all_chem)
                lim_max = max(all_chem)
                margin = (lim_max - lim_min) * 0.05
                ax.plot([lim_min-margin, lim_max+margin], [lim_min-margin, lim_max+margin],
                       'k--', alpha=0.4, linewidth=1.5, label='1:1')

            ax.set_xlabel(f'化学法 {param_label}', fontsize=10)
            ax.set_ylabel(f'修正后 {param_label}', fontsize=10)
            ax.set_title(param_label, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)

        plt.suptitle(f'{point} — 线性回归修正后散点图（应接近 1:1）',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_corrected_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("✅ 修正后散点图")


# ═══════════════════════════════════════════════════════════════════
# 图3: 传感器间对比
# ═══════════════════════════════════════════════════════════════════

def plot_sensor_comparison(data):
    points = ['P1', 'P6', 'P7']
    sensors_map = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}

    for point in points:
        sensors = sensors_map[point]
        if len(sensors) < 2:
            continue

        fig, axes = plt.subplots(1, len(KEY_PARAMS), figsize=(5*len(KEY_PARAMS), 5))

        for col, param_key in enumerate(KEY_PARAMS):
            _, _, param_label = PARAMS[param_key]
            ax = axes[col]

            c1, s1, _ = get_sensor_data(data[point], sensors[0], param_key)
            c2, s2, _ = get_sensor_data(data[point], sensors[1], param_key)

            mask = ~(np.isnan(s1) | np.isnan(s2))
            s1, s2 = s1[mask], s2[mask]

            if len(s1) >= 3 and len(set(s1)) > 1:
                ax.scatter(s1, s2, alpha=0.5, s=20, color='#4CAF50')
                slope, intercept, r_value, _, _ = stats.linregress(s1, s2)
                x_line = np.linspace(s1.min(), s1.max(), 100)
                ax.plot(x_line, slope*x_line + intercept, 'r-', linewidth=2)
                ax.set_xlabel(f'{sensors[0]}', fontsize=9)
                ax.set_ylabel(f'{sensors[1]}', fontsize=9)
                ax.set_title(f'R²={r_value**2:.3f}', fontsize=10)
            else:
                ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)

        plt.suptitle(f'{point} — 传感器间对比 ({sensors[0]} vs {sensors[1]})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_sensor_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("✅ 传感器间对比图")


# ═══════════════════════════════════════════════════════════════════
# 生成报告
# ═══════════════════════════════════════════════════════════════════

def generate_report(data):
    points = ['P1', 'P6', 'P7']
    sensors_map = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}
    params_list = list(PARAMS.keys())

    # 收集所有回归结果
    all_reg = {}
    for point in points:
        for sensor in sensors_map[point]:
            for param in params_list:
                chem, sens, _ = get_sensor_data(data[point], sensor, param)
                reg = do_regression(chem, sens)
                if reg:
                    all_reg[f"{point}_{sensor}_{param}"] = reg

    report_path = OUT_DIR / "传感器数据分析报告.md"

    with open(report_path, 'w') as f:
        f.write("---\ntitle: 中环水务传感器数据分析报告\ndate: 2026-04-27\ntags: [数据分析, 传感器, 中环水务, 线性回归]\n---\n\n")
        f.write("# 中环水务传感器 vs 化学法 数据分析报告\n\n")
        f.write(f"> **生成时间:** 2026-04-27\n")
        f.write(f"> **分析点位:** P1, P6, P7（每点位 2 个传感器）\n")
        f.write(f"> **核心方法:** 线性回归标定 + 趋势相关性分析\n")
        f.write(f"> **EC 处理:** 传感器 EC (mS/cm) × 1000 → μS/cm\n\n")

        # 一、R² 总览
        f.write("## 一、R² 相关性总览热力图\n\n")
        f.write("下图展示所有点位-传感器-参数组合的 R² 值，颜色越绿表示相关性越强。\n\n")
        f.write("![R² 热力图](images/overview_r2_heatmap.png)\n\n")

        # R² 详细表格
        f.write("### R² 详细数据\n\n")
        f.write("| 点位-传感器 | " + " | ".join([PARAMS[p][2].split('(')[0].strip() for p in params_list]) + " |\n")
        f.write("|---" * (len(params_list) + 1) + "|\n")

        param_r2_sum = {p: [] for p in params_list}
        for point in points:
            for sensor in sensors_map[point]:
                row = f"| {sensor} |"
                for param in params_list:
                    key = f"{point}_{sensor}_{param}"
                    if key in all_reg:
                        r2 = all_reg[key]['r2']
                        param_r2_sum[param].append(r2)
                        if r2 >= 0.7:
                            marker = "🟢"
                        elif r2 >= 0.4:
                            marker = "🟡"
                        else:
                            marker = "🔴"
                        row += f" {marker} {r2:.3f} |"
                    else:
                        row += " — |"
                f.write(row + "\n")

        f.write("\n**图例:** 🟢 R²≥0.7（良好）| 🟡 0.4≤R²<0.7（中等）| 🔴 R²<0.4（差）\n\n")

        # 二、线性回归方程汇总
        f.write("## 二、线性回归方程汇总\n\n")
        f.write("回归方程：**传感器值 = slope × 化学法值 + intercept**\n\n")
        f.write("修正公式：**传感器修正值 = (传感器值 - intercept) / slope**\n\n")

        for point in points:
            f.write(f"### {point}\n\n")
            f.write("| 传感器 | 参数 | 斜率 (slope) | 截距 (intercept) | R² | RMSE | 样本数 |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for sensor in sensors_map[point]:
                for param in params_list:
                    key = f"{point}_{sensor}_{param}"
                    if key in all_reg:
                        r = all_reg[key]
                        f.write(f"| {sensor} | {PARAMS[param][2].split('(')[0].strip()} | "
                               f"{r['slope']:.4f} | {r['intercept']:.3f} | "
                               f"{r['r2']:.3f} | {r['rmse']:.3f} | {r['n']} |\n")
            f.write("\n")

        # 三、各点位原始回归对比图
        f.write("## 三、各点位原始数据回归对比\n\n")
        f.write("上排：传感器原始值 vs 化学法值，红色线为回归线，黑色虚线为 1:1 线。\n")
        f.write("如果传感器准确，散点应沿 1:1 线分布。回归线的斜率和截距反映了传感器的系统偏差。\n\n")

        for point in points:
            f.write(f"### {point}\n\n")
            f.write(f"![{point} 原始回归](images/{point}_regression_raw.png)\n\n")

            # 解读
            f.write(f"**{point} 解读：**\n\n")
            for sensor in sensors_map[point]:
                f.write(f"- **{sensor}：**\n")
                for param in params_list:
                    key = f"{point}_{sensor}_{param}"
                    if key in all_reg:
                        r = all_reg[key]
                        slope, intercept, r2 = r['slope'], r['intercept'], r['r2']
                        if r2 >= 0.7:
                            quality = "相关性良好"
                        elif r2 >= 0.4:
                            quality = "相关性中等"
                        else:
                            quality = "相关性差"

                        if abs(slope - 1.0) < 0.2 and abs(intercept) < 50:
                            bias = "偏差小"
                        elif abs(slope - 1.0) < 0.5:
                            bias = "斜率偏差中等"
                        else:
                            bias = "斜率偏差大"

                        f.write(f"  - {PARAMS[param][2].split('(')[0].strip()}：{quality}，{bias}（斜率={slope:.3f}，截距={intercept:.1f}，R²={r2:.3f}）\n")
            f.write("\n")

        # 四、线性回归修正后对比
        f.write("## 四、线性回归修正后对比\n\n")
        f.write("将传感器数据用回归方程修正后，与化学法数据对比。\n")
        f.write("修正公式：**传感器修正值 = (传感器原始值 - intercept) / slope**\n\n")
        f.write("理想情况下，修正后的散点应沿 1:1 线分布。\n\n")

        for point in points:
            f.write(f"### {point} — 修正后时序对比\n\n")
            f.write(f"![{point} 修正后时序](images/{point}_corrected_timeseries.png)\n\n")

            f.write(f"### {point} — 修正后散点图\n\n")
            f.write(f"![{point} 修正后散点](images/{point}_corrected_scatter.png)\n\n")

            # 计算修正后 R²
            f.write(f"**{point} 修正后相关性：**\n\n")
            f.write("| 传感器 | 参数 | 修正后 R² | 原始 R² | 改善 |\n")
            f.write("|---|---|---|---|---|\n")
            for sensor in sensors_map[point]:
                for param in params_list:
                    key = f"{point}_{sensor}_{param}"
                    if key in all_reg:
                        r = all_reg[key]
                        # 修正后 R² 实际上应该接近 1.0（因为是用同组数据回归的）
                        # 但我们可以计算修正后 vs 化学法的 R²
                        chem, sens, _ = get_sensor_data(data[point], sensor, param)
                        sens_corrected = (sens - r['intercept']) / r['slope']
                        mask = ~(np.isnan(chem) | np.isnan(sens_corrected))
                        if mask.sum() >= 3:
                            _, _, r_corrected, _, _ = stats.linregress(chem[mask], sens_corrected[mask])
                            r2_corrected = r_corrected**2
                            improvement = "✅" if r2_corrected > r['r2'] else "—"
                            f.write(f"| {sensor} | {PARAMS[param][2].split('(')[0].strip()} | "
                                   f"{r2_corrected:.3f} | {r['r2']:.3f} | {improvement} |\n")
            f.write("\n")

        # 五、传感器间一致性
        f.write("## 五、传感器间一致性\n\n")
        f.write("同一位置两个传感器的数据一致性。\n\n")
        for point in points:
            f.write(f"### {point}\n\n")
            f.write(f"![{point} 传感器对比](images/{point}_sensor_comparison.png)\n\n")

        # 六、各参数平均 R² 排名
        f.write("## 六、各参数平均 R² 排名\n\n")
        f.write("| 排名 | 参数 | 平均 R² | 最高 R² | 最低 R² | 评级 |\n")
        f.write("|---|---|---|---|---|---|\n")

        param_stats = []
        for param in params_list:
            r2s = param_r2_sum[param]
            if r2s:
                avg = np.mean(r2s)
                param_stats.append((param, avg, max(r2s), min(r2s)))
        param_stats.sort(key=lambda x: x[1], reverse=True)

        for i, (param, avg, mx, mn) in enumerate(param_stats, 1):
            label = PARAMS[param][2].split('(')[0].strip()
            if avg >= 0.7:
                grade = "🟢 良好"
            elif avg >= 0.4:
                grade = "🟡 中等"
            else:
                grade = "🔴 差"
            f.write(f"| {i} | {label} | {avg:.3f} | {mx:.3f} | {mn:.3f} | {grade} |\n")

        # 七、结论
        f.write("\n## 七、结论与建议\n\n")
        best = param_stats[0] if param_stats else None
        worst = param_stats[-1] if param_stats else None

        f.write("### 关键发现\n\n")
        if best:
            f.write(f"1. **最佳参数：** {PARAMS[best[0]][2].split('(')[0].strip()}（平均 R²={best[1]:.3f}）— 传感器与化学法趋势高度一致\n")
        if worst:
            f.write(f"2. **最差参数：** {PARAMS[worst[0]][2].split('(')[0].strip()}（平均 R²={worst[1]:.3f}）— 传感器数据可靠性低\n")
        f.write("3. **EC 标定：** ×1000 后与化学法单位对齐，相关性需逐点位评估\n")
        f.write("4. **线性回归修正：** 通过 y=(sensor-intercept)/slope 可将传感器数据转换为化学法等效值\n\n")

        f.write("### 建议\n\n")
        f.write("1. 对 R²≥0.7 的参数建立传感器→化学法线性转换模型，用于实时数据修正\n")
        f.write("2. 对 R²<0.4 的参数传感器进行检修或重新标定\n")
        f.write("3. 定期用化学法校验传感器数据，更新回归方程\n")
        f.write("4. 关注传感器趋势变化，不完全依赖绝对值\n")

    print(f"✅ 报告: {report_path}")


def main():
    print("=" * 60)
    print("中环水务传感器数据分析 V2")
    print("=" * 60)

    data = load_data()

    plot_r2_heatmap(data)
    plot_regression_correction(data)
    plot_corrected_comparison(data)
    plot_corrected_scatter(data)
    plot_sensor_comparison(data)
    generate_report(data)

    print("\n完成 ✅")


if __name__ == "__main__":
    main()
