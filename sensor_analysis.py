#!/usr/bin/env python3
"""
中环水务传感器 vs 化学法 数据分析报告
分析 P1/P6/P7 三个点位的传感器数据与化学法的相关性
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

# ── 配置 ──
DATA_DIR = Path("/Users/pwngwc/Obsidian/PengWC/为知笔记/01_芯视界/中环水务数据/分析报告_20260425/联合数据集")
OUT_DIR = Path("/Users/pwngwc/Obsidian/PengWC/为知笔记/01_芯视界/中环水务数据/分析报告_20260427")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

# ── 参数映射 ──
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

# ── EC 单位换算因子 ──
EC_FACTOR = 1000  # 传感器 mS/cm → μS/cm

def load_data():
    """加载三个点位的联合数据集"""
    data = {}
    for point in ['P1', 'P6', 'P7']:
        df = pd.read_csv(DATA_DIR / f"{point}_联合数据集.csv")
        df['chem_datetime'] = pd.to_datetime(df['chem_datetime'])
        # EC 单位对齐：传感器 EC × 1000 → μS/cm
        df['sensor_ec_aligned'] = df['sensor_ec'] * EC_FACTOR
        data[point] = df
        print(f"{point}: {len(df)} 条记录, {df['sensor_name'].nunique()} 个传感器")
    return data


def linear_regression_analysis(x, y):
    """线性回归分析"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    return {
        'slope': slope, 'intercept': intercept,
        'r_value': r_value, 'r_squared': r_value**2,
        'p_value': p_value, 'std_err': std_err,
        'rmse': rmse, 'mae': mae,
        'n': len(x),
        'x_mean': np.mean(x), 'y_mean': np.mean(y),
    }


def plot_scatter_with_regression(ax, x, y, title, xlabel, ylabel, color='#2196F3'):
    """散点图 + 回归线"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    ax.scatter(x, y, alpha=0.6, s=30, color=color, edgecolors='white', linewidth=0.5)

    # 回归线
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8)

    # 1:1 线
    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())
    margin = (lim_max - lim_min) * 0.05
    ax.plot([lim_min-margin, lim_max+margin], [lim_min-margin, lim_max+margin],
            'k--', alpha=0.3, linewidth=1, label='1:1')

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{title}\nR²={r_value**2:.3f}, slope={slope:.3f}', fontsize=10)
    ax.legend(fontsize=8)


def plot_time_series(ax, df, chem_col, sensor_col, sensor_name, param_name, point):
    """时序对比图"""
    sub = df[df['sensor_name'] == sensor_name].copy()
    sub = sub.dropna(subset=[chem_col, sensor_col])
    if len(sub) < 2:
        ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
        return

    sub = sub.sort_values('chem_datetime')

    # 双 Y 轴
    color1, color2 = '#2196F3', '#FF5722'
    ax.plot(sub['chem_datetime'], sub[chem_col], 'o-', color=color1, markersize=4,
            linewidth=1.5, label=f'化学法', alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(sub['chem_datetime'], sub[sensor_col], 's-', color=color2, markersize=4,
             linewidth=1.5, label=f'传感器', alpha=0.8)

    ax.set_ylabel(f'化学法 {param_name}', color=color1, fontsize=9)
    ax2.set_ylabel(f'传感器 {param_name}', color=color2, fontsize=9)
    ax.set_title(f'{point} {sensor_name} — {param_name}', fontsize=10)

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax.tick_params(axis='x', rotation=30, labelsize=8)


def analyze_point(point, df):
    """分析单个点位"""
    results = {}
    sensor_names = df['sensor_name'].unique()

    for param_key, (chem_col, sensor_col, param_label) in PARAMS.items():
        for sensor in sensor_names:
            sub = df[df['sensor_name'] == sensor]
            chem_data = sub[chem_col].values

            # EC 对齐
            if param_key == 'ec':
                sensor_data = sub['sensor_ec_aligned'].values
            else:
                sensor_data = sub[sensor_col].values

            reg = linear_regression_analysis(chem_data, sensor_data)
            if reg:
                key = f"{point}_{sensor}_{param_key}"
                results[key] = {
                    'point': point, 'sensor': sensor, 'param': param_key,
                    'param_label': param_label, **reg
                }

    return results


def generate_plots(data):
    """生成所有图表"""
    fig_count = 0

    # ── 图1: 各点位各参数散点图（化学法 vs 传感器）──
    for point, df in data.items():
        sensor_names = sorted(df['sensor_name'].unique())
        for param_key, (chem_col, sensor_col, param_label) in PARAMS.items():
            fig, axes = plt.subplots(1, len(sensor_names), figsize=(6*len(sensor_names), 5))
            if len(sensor_names) == 1:
                axes = [axes]

            for i, sensor in enumerate(sensor_names):
                sub = df[df['sensor_name'] == sensor]
                chem_data = sub[chem_col].values
                if param_key == 'ec':
                    sensor_data = sub['sensor_ec_aligned'].values
                    xlabel = f'化学法 {param_label}'
                    ylabel = f'传感器 {param_label}'
                else:
                    sensor_data = sub[sensor_col].values
                    xlabel = f'化学法 {param_label}'
                    ylabel = f'传感器 {param_label}'

                plot_scatter_with_regression(
                    axes[i], chem_data, sensor_data,
                    f'{point} {sensor}', xlabel, ylabel,
                    color='#2196F3' if i == 0 else '#FF9800'
                )

            plt.suptitle(f'{point} — {param_label} 化学法 vs 传感器', fontsize=13, fontweight='bold')
            plt.tight_layout()
            fig_count += 1
            plt.savefig(IMG_DIR / f'fig{fig_count:02d}_{point}_{param_key}_scatter.png',
                       dpi=150, bbox_inches='tight')
            plt.close()

    # ── 图2: 各点位时序对比 ──
    for point, df in data.items():
        sensor_names = sorted(df['sensor_name'].unique())
        key_params = ['cod', 'turbidity', 'ph', 'ec', 'do']

        fig, axes = plt.subplots(len(key_params), 1, figsize=(14, 4*len(key_params)))
        for idx, param_key in enumerate(key_params):
            chem_col, sensor_col, param_label = PARAMS[param_key]
            # 用第一个传感器
            plot_time_series(axes[idx], df, chem_col, sensor_col,
                           sensor_names[0], param_label, point)

        plt.suptitle(f'{point} — 化学法 vs {sensor_names[0]} 时序对比', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig_count += 1
        plt.savefig(IMG_DIR / f'fig{fig_count:02d}_{point}_timeseries.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    # ── 图3: 传感器间对比（P1-1 vs P1-2 等）──
    for point, df in data.items():
        sensor_names = sorted(df['sensor_name'].unique())
        if len(sensor_names) < 2:
            continue

        key_params = ['cod', 'turbidity', 'ph', 'ec', 'do']
        fig, axes = plt.subplots(1, len(key_params), figsize=(5*len(key_params), 5))

        for idx, param_key in enumerate(key_params):
            chem_col, sensor_col, param_label = PARAMS[param_key]
            s1 = df[df['sensor_name'] == sensor_names[0]][sensor_col].values
            s2 = df[df['sensor_name'] == sensor_names[1]][sensor_col].values

            if param_key == 'ec':
                s1 = s1 * EC_FACTOR
                s2 = s2 * EC_FACTOR

            mask = ~(np.isnan(s1) | np.isnan(s2))
            s1, s2 = s1[mask], s2[mask]

            if len(s1) >= 3 and len(set(s1)) > 1 and len(set(s2)) > 1:
                axes[idx].scatter(s1, s2, alpha=0.5, s=20, color='#4CAF50')
                slope, intercept, r_value, _, _ = stats.linregress(s1, s2)
                x_line = np.linspace(s1.min(), s1.max(), 100)
                axes[idx].plot(x_line, slope*x_line + intercept, 'r-', linewidth=2)
                axes[idx].set_xlabel(f'{sensor_names[0]} {param_label}', fontsize=9)
                axes[idx].set_ylabel(f'{sensor_names[1]} {param_label}', fontsize=9)
                axes[idx].set_title(f'R²={r_value**2:.3f}', fontsize=10)

        plt.suptitle(f'{point} — 传感器间对比 ({sensor_names[0]} vs {sensor_names[1]})',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig_count += 1
        plt.savefig(IMG_DIR / f'fig{fig_count:02d}_{point}_sensor_comparison.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    # ── 图4: 总览热力图（R² 矩阵）──
    all_results = {}
    for point, df in data.items():
        all_results.update(analyze_point(point, df))

    # 构建 R² 矩阵
    points = ['P1', 'P6', 'P7']
    params = ['cod', 'turbidity', 'temperature', 'ph', 'ec', 'do', 'ammonia', 'codmn']
    sensors_per_point = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}

    fig, ax = plt.subplots(figsize=(16, 8))
    r2_matrix = []
    row_labels = []

    for point in points:
        for sensor in sensors_per_point[point]:
            row = []
            for param in params:
                key = f"{point}_{sensor}_{param}"
                if key in all_results:
                    row.append(all_results[key]['r_squared'])
                else:
                    row.append(np.nan)
            r2_matrix.append(row)
            row_labels.append(f"{sensor}")

    r2_arr = np.array(r2_matrix)
    im = ax.imshow(r2_arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([PARAMS[p][2].split('(')[0].strip() for p in params], rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # 标注数值
    for i in range(len(row_labels)):
        for j in range(len(params)):
            val = r2_arr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color='white' if val < 0.3 or val > 0.8 else 'black')

    plt.colorbar(im, ax=ax, label='R²')
    ax.set_title('各点位传感器 vs 化学法 R² 热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_count += 1
    plt.savefig(IMG_DIR / f'fig{fig_count:02d}_r2_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── 图5: EC 标定前后对比 ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (point, df) in enumerate(data.items()):
        sensor_names = sorted(df['sensor_name'].unique())
        sub = df[df['sensor_name'] == sensor_names[0]]

        # 标定前
        axes[0, idx].scatter(sub['chem_ec'], sub['sensor_ec'], alpha=0.5, s=20, color='#F44336')
        axes[0, idx].set_xlabel('化学法 EC (μS/cm)')
        axes[0, idx].set_ylabel(f'传感器 EC (mS/cm)')
        axes[0, idx].set_title(f'{point} {sensor_names[0]} EC 标定前')

        # 标定后
        axes[1, idx].scatter(sub['chem_ec'], sub['sensor_ec']*EC_FACTOR, alpha=0.5, s=20, color='#4CAF50')
        slope, intercept, r_value, _, _ = stats.linregress(sub['chem_ec'].dropna(), (sub['sensor_ec']*EC_FACTOR).dropna())
        x_line = np.linspace(sub['chem_ec'].min(), sub['chem_ec'].max(), 100)
        axes[1, idx].plot(x_line, slope*x_line + intercept, 'r-', linewidth=2)
        axes[1, idx].set_xlabel('化学法 EC (μS/cm)')
        axes[1, idx].set_ylabel(f'传感器 EC (μS/cm)')
        axes[1, idx].set_title(f'{point} {sensor_names[0]} EC 标定后 (×1000)\nR²={r_value**2:.3f}')

    plt.suptitle('EC 单位标定前后对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_count += 1
    plt.savefig(IMG_DIR / f'fig{fig_count:02d}_ec_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"共生成 {fig_count} 张图表")
    return fig_count


def generate_report(data, all_results):
    """生成 Markdown 报告"""
    report_path = OUT_DIR / "传感器数据分析报告.md"

    # 统计汇总
    points = ['P1', 'P6', 'P7']
    params = ['cod', 'turbidity', 'temperature', 'ph', 'ec', 'do', 'ammonia', 'codmn']
    sensors_per_point = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}

    with open(report_path, 'w') as f:
        f.write("---\ntitle: 中环水务传感器数据分析报告\ndate: 2026-04-27\ntags: [数据分析, 传感器, 中环水务]\n---\n\n")
        f.write("# 中环水务传感器 vs 化学法 数据分析报告\n\n")
        f.write(f"> **生成时间:** 2026-04-27 16:20\n")
        f.write(f"> **数据来源:** 分析报告_20260425/联合数据集\n")
        f.write(f"> **分析点位:** P1, P6, P7\n")
        f.write(f"> **传感器:** P1-1, P1-2, P6-1, P6-2, P7-1, P7-2\n\n")

        # 1. 数据概况
        f.write("## 一、数据概况\n\n")
        f.write("| 点位 | 记录数 | 传感器 | 时间范围 |\n")
        f.write("|------|--------|--------|----------|\n")
        for point in points:
            df = data[point]
            sensors = sorted(df['sensor_name'].unique())
            time_range = f"{df['chem_datetime'].min().strftime('%m-%d %H:%M')} ~ {df['chem_datetime'].max().strftime('%m-%d %H:%M')}"
            f.write(f"| {point} | {len(df)} | {', '.join(sensors)} | {time_range} |\n")

        f.write(f"\n**化学法采样次数:** P1={len(data['P1'])//2}, P6={len(data['P6'])//2}, P7={len(data['P7'])//2}\n")
        f.write(f"\n**EC 单位处理:** 传感器 EC (mS/cm) × 1000 = μS/cm（与化学法对齐）\n\n")

        # 2. R² 总览热力图
        f.write("## 二、R² 相关性总览\n\n")
        f.write("![R² 热力图](images/fig09_r2_heatmap.png)\n\n")

        # R² 表格
        f.write("### R² 详细数据\n\n")
        header = "| 点位-传感器 | " + " | ".join([PARAMS[p][2].split('(')[0].strip() for p in params]) + " |\n"
        separator = "|---" * (len(params) + 1) + "|\n"
        f.write(header)
        f.write(separator)

        for point in points:
            for sensor in sensors_per_point[point]:
                row = f"| {sensor} |"
                for param in params:
                    key = f"{point}_{sensor}_{param}"
                    if key in all_results:
                        r2 = all_results[key]['r_squared']
                        # 标记优秀/良好/差
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

        # 3. EC 标定分析
        f.write("## 三、EC 标定分析\n\n")
        f.write("传感器 EC 单位为 mS/cm，化学法 EC 单位为 μS/cm，相差约 1000 倍。\n\n")
        f.write("![EC 标定](images/fig10_ec_calibration.png)\n\n")

        f.write("### EC 标定后回归结果\n\n")
        f.write("| 点位-传感器 | 斜率 | 截距 | R² | RMSE |\n")
        f.write("|---|---|---|---|---|\n")
        for point in points:
            for sensor in sensors_per_point[point]:
                key = f"{point}_{sensor}_ec"
                if key in all_results:
                    r = all_results[key]
                    f.write(f"| {sensor} | {r['slope']:.4f} | {r['intercept']:.1f} | {r['r_squared']:.3f} | {r['rmse']:.1f} |\n")

        f.write("\n**解读:** 斜率接近 1.0 表示传感器与化学法线性关系好，截距接近 0 表示无系统偏差。\n\n")

        # 4. 各参数详细分析
        f.write("## 四、各参数详细分析\n\n")

        for param_key in params:
            chem_col, sensor_col, param_label = PARAMS[param_key]
            f.write(f"### 4.{params.index(param_key)+1} {param_label}\n\n")

            # 散点图
            for point in points:
                img_files = list(IMG_DIR.glob(f'*_{point}_{param_key}_scatter.png'))
                if img_files:
                    f.write(f"![{point} {param_label}](images/{img_files[0].name})\n\n")

            # 回归数据表
            f.write(f"| 点位-传感器 | 斜率 | 截距 | R² | RMSE | MAE | 样本数 |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for point in points:
                for sensor in sensors_per_point[point]:
                    key = f"{point}_{sensor}_{param_key}"
                    if key in all_results:
                        r = all_results[key]
                        f.write(f"| {sensor} | {r['slope']:.4f} | {r['intercept']:.3f} | "
                               f"{r['r_squared']:.3f} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['n']} |\n")

            # 分析结论
            f.write("\n**分析:** ")
            r2_values = []
            for point in points:
                for sensor in sensors_per_point[point]:
                    key = f"{point}_{sensor}_{param_key}"
                    if key in all_results:
                        r2_values.append(all_results[key]['r_squared'])

            if r2_values:
                avg_r2 = np.mean(r2_values)
                if avg_r2 >= 0.7:
                    f.write(f"平均 R²={avg_r2:.3f}，相关性**良好**，传感器能较好反映化学法变化趋势。\n\n")
                elif avg_r2 >= 0.4:
                    f.write(f"平均 R²={avg_r2:.3f}，相关性**中等**，传感器趋势基本一致但存在偏差。\n\n")
                else:
                    f.write(f"平均 R²={avg_r2:.3f}，相关性**较差**，传感器数据需进一步标定或该参数传感器不可靠。\n\n")

        # 5. 传感器间对比
        f.write("## 五、传感器间一致性分析\n\n")
        f.write("同一位置两个传感器的数据一致性，评估传感器可靠性。\n\n")

        for point in points:
            img_files = list(IMG_DIR.glob(f'*_{point}_sensor_comparison.png'))
            if img_files:
                f.write(f"![{point} 传感器对比](images/{img_files[0].name})\n\n")

        # 6. 时序图
        f.write("## 六、时序对比图\n\n")
        for point in points:
            img_files = list(IMG_DIR.glob(f'*_{point}_timeseries.png'))
            if img_files:
                f.write(f"### {point}\n\n")
                f.write(f"![{point} 时序](images/{img_files[0].name})\n\n")

        # 7. 总结与建议
        f.write("## 七、总结与建议\n\n")

        # 统计各参数平均 R²
        f.write("### 各参数平均 R² 排名\n\n")
        param_avg_r2 = {}
        for param_key in params:
            r2_values = []
            for point in points:
                for sensor in sensors_per_point[point]:
                    key = f"{point}_{sensor}_{param_key}"
                    if key in all_results:
                        r2_values.append(all_results[key]['r_squared'])
            if r2_values:
                param_avg_r2[param_key] = np.mean(r2_values)

        sorted_params = sorted(param_avg_r2.items(), key=lambda x: x[1], reverse=True)
        f.write("| 排名 | 参数 | 平均 R² | 评级 |\n")
        f.write("|---|---|---|---|\n")
        for i, (param, r2) in enumerate(sorted_params, 1):
            label = PARAMS[param][2].split('(')[0].strip()
            if r2 >= 0.7:
                grade = "🟢 良好"
            elif r2 >= 0.4:
                grade = "🟡 中等"
            else:
                grade = "🔴 差"
            f.write(f"| {i} | {label} | {r2:.3f} | {grade} |\n")

        f.write("\n### 关键发现\n\n")
        f.write("1. **EC 标定：** 传感器 EC × 1000 后与化学法对齐，相关性需逐点位评估\n")
        best_param = sorted_params[0] if sorted_params else None
        worst_param = sorted_params[-1] if sorted_params else None
        if best_param:
            f.write(f"2. **最佳参数：** {PARAMS[best_param[0]][2].split('(')[0].strip()}（R²={best_param[1]:.3f}）\n")
        if worst_param:
            f.write(f"3. **最差参数：** {PARAMS[worst_param[0]][2].split('(')[0].strip()}（R²={worst_param[1]:.3f}）\n")
        f.write("4. **传感器一致性：** 同一位置两个传感器之间需检查线性关系\n")
        f.write("5. **趋势关注：** 未归零率定的传感器重点关注变化趋势而非绝对值\n\n")

        f.write("### 建议\n\n")
        f.write("1. 对 R²<0.4 的参数传感器进行检修或重新标定\n")
        f.write("2. 建立传感器→化学法的线性转换模型，用于实时数据修正\n")
        f.write("3. EC 传感器持续监控单位一致性\n")
        f.write("4. 定期用化学法校验传感器数据\n")

    print(f"✅ 报告已生成: {report_path}")
    return report_path


def main():
    print("=" * 60)
    print("中环水务传感器数据分析")
    print("=" * 60)

    # 加载数据
    data = load_data()

    # 分析所有点位
    all_results = {}
    for point, df in data.items():
        results = analyze_point(point, df)
        all_results.update(results)
        print(f"{point}: {len(results)} 个回归分析完成")

    # 生成图表
    fig_count = generate_plots(data)

    # 生成报告
    report_path = generate_report(data, all_results)

    print(f"\n总计: {len(all_results)} 个回归分析, {fig_count} 张图表")
    print(f"报告: {report_path}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
