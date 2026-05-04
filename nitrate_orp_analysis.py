#!/usr/bin/env python3
"""
中环水务 硝氮 & ORP 传感器数据分析
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
import re, warnings
warnings.filterwarnings('ignore')

SENSOR_DIR = Path("/Users/pwngwc/中环水务数据/ORP")
CHEM_CSV = Path("/Users/pwngwc/Obsidian/PengWC/为知笔记/01_芯视界/中环水务数据/分析报告_20260425/联合数据集/化学法金标准.csv")
OUT_DIR = Path("/Users/pwngwc/Obsidian/PengWC/为知笔记/01_芯视界/中环水务数据/硝氮和ORP报告")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)


def parse_sensor_md(filepath):
    """解析 ORP 目录下的 md 表格文件"""
    lines = Path(filepath).read_text(encoding='utf-8').splitlines()
    # 找到表头行（包含 collection_date）
    header_idx = None
    for i, line in enumerate(lines):
        if 'collection_date' in line:
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    # 解析表头
    header = [h.strip() for h in lines[header_idx].split('|')[1:-1]]

    # 解析数据行（跳过分隔符行）
    rows = []
    for line in lines[header_idx+2:]:
        if not line.strip() or '|' not in line:
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))

    df = pd.DataFrame(rows)
    # 转换数值列
    for col in ['ph_an_an', 'ph_an_ph', 'rtd_temp_f_val', 'an_f_val']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'collection_date' in df.columns:
        df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce')

    return df


def load_all_sensors():
    """加载所有传感器数据"""
    sensors = {}
    for f in sorted(SENSOR_DIR.glob("*.md")):
        name = f.stem  # P1-1, P1-2, etc.
        df = parse_sensor_md(f)
        if len(df) > 0:
            sensors[name] = df
            print(f"  {name}: {len(df)} 条, 硝氮有效={df['ph_an_an'].notna().sum()}, ORP有效={df['ph_an_ph'].notna().sum()}")
    return sensors


def load_chemical():
    """加载化学法数据"""
    df = pd.read_csv(CHEM_CSV)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def match_sensor_to_chem(sensor_df, chem_df, point_name, time_window_min=30):
    """将传感器数据与化学法数据按时间匹配"""
    chem_sub = chem_df[chem_df['point'] == point_name].copy()
    if len(chem_sub) == 0:
        return pd.DataFrame()

    matches = []
    for _, chem_row in chem_sub.iterrows():
        chem_time = chem_row['datetime']
        # 找时间窗口内的传感器数据
        time_diff = abs((sensor_df['collection_date'] - chem_time).dt.total_seconds() / 60)
        within_window = sensor_df[time_diff <= time_window_min]

        if len(within_window) > 0:
            # 取中位数
            match = {
                'chem_time': chem_time,
                'chem_nitrate': chem_row.get('nitrate', np.nan),
                'sensor_nitrate': within_window['ph_an_an'].median(),
                'sensor_orp': within_window['ph_an_ph'].median(),
                'sensor_temp': within_window['rtd_temp_f_val'].median() if 'rtd_temp_f_val' in within_window.columns else np.nan,
                'sensor_nitrate_std': within_window['ph_an_an'].std(),
                'sensor_orp_std': within_window['ph_an_ph'].std(),
                'n_points': len(within_window),
                'time_diff_min': time_diff[within_window.index].min(),
            }
            matches.append(match)

    return pd.DataFrame(matches)


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
        'n': len(x),
    }


def plot_nitrate_comparison(matched_data, sensors_map):
    """硝氮：化学法 vs 传感器散点图 + 时序对比"""
    points = ['P1', 'P6', 'P7']
    colors = {'P1-1': '#2196F3', 'P1-2': '#FF9800', 'P6-1': '#2196F3', 'P6-2': '#FF9800',
              'P7-1': '#2196F3', 'P7-2': '#FF9800'}

    # 散点图：2行3列
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, point in enumerate(points):
        for row, sensor_name in enumerate(sensors_map[point]):
            ax = axes[row, idx]
            if sensor_name not in matched_data or len(matched_data[sensor_name]) == 0:
                ax.text(0.5, 0.5, '无匹配数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{sensor_name}')
                continue

            md = matched_data[sensor_name]
            chem = md['chem_nitrate'].values
            sens = md['sensor_nitrate'].values

            ax.scatter(chem, sens, alpha=0.6, s=40, color=colors[sensor_name],
                      edgecolors='white', linewidth=0.5)

            reg = do_regression(chem, sens)
            if reg:
                x_line = np.linspace(chem.min(), chem.max(), 100)
                ax.plot(x_line, reg['slope'] * x_line + reg['intercept'], 'r-', linewidth=2,
                       label=f"y={reg['slope']:.3f}x+{reg['intercept']:.2f}\nR²={reg['r2']:.3f}")
                # 1:1 线
                lim_min = min(chem.min(), sens.min())
                lim_max = max(chem.max(), sens.max())
                margin = (lim_max - lim_min) * 0.05
                ax.plot([lim_min-margin, lim_max+margin], [lim_min-margin, lim_max+margin],
                       'k--', alpha=0.3, linewidth=1)
                ax.legend(fontsize=9, loc='upper left')

            ax.set_xlabel('化学法 硝氮 (mg/L)', fontsize=10)
            ax.set_ylabel('传感器 硝氮 (mg/L)', fontsize=10)
            ax.set_title(f'{sensor_name}', fontsize=12, fontweight='bold')

    plt.suptitle('硝氮 — 化学法 vs 传感器 散点图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'nitrate_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 时序对比：每个点位一张图
    for point in points:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        for row, sensor_name in enumerate(sensors_map[point]):
            ax = axes[row]
            if sensor_name not in matched_data or len(matched_data[sensor_name]) == 0:
                ax.text(0.5, 0.5, '无匹配数据', ha='center', va='center', transform=ax.transAxes)
                continue

            md = matched_data[sensor_name].sort_values('chem_time')
            ax.errorbar(md['chem_time'], md['sensor_nitrate'],
                       yerr=md['sensor_nitrate_std'].fillna(0),
                       fmt='o-', color=colors[sensor_name], markersize=5, linewidth=1.5,
                       capsize=3, label=f'{sensor_name} 传感器', alpha=0.8)
            ax.plot(md['chem_time'], md['chem_nitrate'], 's--', color='black',
                   markersize=6, linewidth=2, label='化学法', alpha=0.8)
            ax.set_ylabel('硝氮 (mg/L)', fontsize=10)
            ax.set_title(f'{point} {sensor_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.tick_params(axis='x', rotation=30, labelsize=8)

        plt.suptitle(f'{point} — 硝氮 时序对比（传感器 vs 化学法）', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_nitrate_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("  ✅ 硝氮图表")


def plot_orp_analysis(sensors, sensors_map):
    """ORP 分析：传感器间对比 + 时序趋势"""
    points = ['P1', 'P6', 'P7']

    # 传感器间 ORP 散点图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, point in enumerate(points):
        ax = axes[idx]
        s1_name, s2_name = sensors_map[point]
        if s1_name not in sensors or s2_name not in sensors:
            continue

        s1 = sensors[s1_name]
        s2 = sensors[s2_name]

        # 按时间匹配
        merged = pd.merge_asof(
            s1.sort_values('collection_date'),
            s2.sort_values('collection_date'),
            on='collection_date',
            tolerance=pd.Timedelta('2min'),
            suffixes=('_1', '_2')
        ).dropna(subset=['ph_an_ph_1', 'ph_an_ph_2'])

        if len(merged) < 3:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            continue

        x = merged['ph_an_ph_1'].values
        y = merged['ph_an_ph_2'].values

        ax.scatter(x, y, alpha=0.3, s=10, color='#4CAF50')
        reg = do_regression(x, y)
        if reg:
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, reg['slope'] * x_line + reg['intercept'], 'r-', linewidth=2,
                   label=f"R²={reg['r2']:.3f}")
            ax.legend(fontsize=10)

        ax.set_xlabel(f'{s1_name} ORP (mV)', fontsize=10)
        ax.set_ylabel(f'{s2_name} ORP (mV)', fontsize=10)
        ax.set_title(f'{point} 传感器间 ORP 对比', fontsize=12, fontweight='bold')

    plt.suptitle('ORP — 传感器间一致性', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'orp_sensor_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ORP 时序图
    for point in points:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        for row, sensor_name in enumerate(sensors_map[point]):
            ax = axes[row]
            if sensor_name not in sensors:
                continue
            s = sensors[sensor_name].sort_values('collection_date')
            # 降采样到分钟级
            s_min = s.set_index('collection_date').resample('10min')['ph_an_ph'].median().dropna()
            ax.plot(s_min.index, s_min.values, '-', color='#9C27B0', linewidth=1, alpha=0.8)
            ax.set_ylabel('ORP (mV)', fontsize=10)
            ax.set_title(f'{sensor_name} ORP 时序', fontsize=12, fontweight='bold')
            ax.axhline(y=s_min.mean(), color='red', linestyle='--', alpha=0.5,
                      label=f'均值={s_min.mean():.1f}mV')
            ax.legend(fontsize=9)
            ax.tick_params(axis='x', rotation=30, labelsize=8)

        plt.suptitle(f'{point} — ORP 时序趋势', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_orp_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("  ✅ ORP 图表")


def plot_orp_vs_nitrate(sensors, sensors_map):
    """ORP vs 硝氮 相关性（传感器内部）"""
    points = ['P1', 'P6', 'P7']
    colors = {'P1-1': '#2196F3', 'P1-2': '#FF9800', 'P6-1': '#2196F3', 'P6-2': '#FF9800',
              'P7-1': '#2196F3', 'P7-2': '#FF9800'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, point in enumerate(points):
        for row, sensor_name in enumerate(sensors_map[point]):
            ax = axes[row, idx]
            if sensor_name not in sensors:
                continue

            s = sensors[sensor_name]
            x = s['ph_an_ph'].values  # ORP
            y = s['ph_an_an'].values  # 硝氮

            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]

            if len(x) < 3:
                ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
                continue

            ax.scatter(x, y, alpha=0.3, s=10, color=colors[sensor_name])
            reg = do_regression(x, y)
            if reg:
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, reg['slope'] * x_line + reg['intercept'], 'r-', linewidth=2,
                       label=f"R²={reg['r2']:.3f}")
                ax.legend(fontsize=9)

            ax.set_xlabel('ORP (mV)', fontsize=10)
            ax.set_ylabel('硝氮 (mg/L)', fontsize=10)
            ax.set_title(f'{sensor_name}', fontsize=12, fontweight='bold')

    plt.suptitle('ORP vs 硝氮 — 传感器内部相关性', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'orp_vs_nitrate.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  ✅ ORP vs 硝氮图表")


def generate_report(matched_data, sensors, sensors_map):
    """生成报告"""
    points = ['P1', 'P6', 'P7']

    with open(OUT_DIR / "硝氮和ORP分析报告.md", 'w') as f:
        f.write("---\ntitle: 硝氮和ORP传感器分析报告\ndate: 2026-04-27\ntags: [数据分析, 传感器, 硝氮, ORP, 中环水务]\n---\n\n")
        f.write("# 硝氮和ORP传感器数据分析报告\n\n")
        f.write(f"> **生成时间:** 2026-04-27\n")
        f.write(f"> **传感器数据:** `/Users/pwngwc/中环水务数据/ORP/`\n")
        f.write(f"> **化学法数据:** `化学法金标准.csv`\n")
        f.write(f"> **分析参数:** 硝氮 (ph_an_an)、ORP (ph_an_ph)\n\n")

        # 一、数据概况
        f.write("## 一、数据概况\n\n")
        f.write("| 传感器 | 总记录数 | 硝氮有效 | ORP有效 | 时间范围 |\n")
        f.write("|---|---|---|---|---|\n")
        for point in points:
            for sensor_name in sensors_map[point]:
                if sensor_name in sensors:
                    s = sensors[sensor_name]
                    t_min = s['collection_date'].min()
                    t_max = s['collection_date'].max()
                    f.write(f"| {sensor_name} | {len(s)} | {s['ph_an_an'].notna().sum()} | "
                           f"{s['ph_an_ph'].notna().sum()} | "
                           f"{t_min.strftime('%m-%d %H:%M')} ~ {t_max.strftime('%m-%d %H:%M')} |\n")

        f.write(f"\n**化学法硝氮采样:** {len(matched_data.get('P1-1', pd.DataFrame())) + len(matched_data.get('P6-1', pd.DataFrame())) + len(matched_data.get('P7-1', pd.DataFrame()))} 次匹配\n\n")

        # 二、硝氮分析
        f.write("## 二、硝氮 — 化学法 vs 传感器\n\n")
        f.write("### 2.1 散点图\n\n")
        f.write("![硝氮散点](images/nitrate_scatter.png)\n\n")

        f.write("### 2.2 回归方程\n\n")
        f.write("| 传感器 | 斜率 | 截距 | R² | RMSE | 样本数 |\n")
        f.write("|---|---|---|---|---|---|\n")

        for point in points:
            for sensor_name in sensors_map[point]:
                if sensor_name in matched_data and len(matched_data[sensor_name]) > 0:
                    md = matched_data[sensor_name]
                    reg = do_regression(md['chem_nitrate'].values, md['sensor_nitrate'].values)
                    if reg:
                        f.write(f"| {sensor_name} | {reg['slope']:.4f} | {reg['intercept']:.3f} | "
                               f"{reg['r2']:.3f} | {reg['rmse']:.3f} | {reg['n']} |\n")

        f.write("\n### 2.3 时序对比\n\n")
        for point in points:
            f.write(f"**{point}:**\n\n")
            f.write(f"![{point} 硝氮时序](images/{point}_nitrate_timeseries.png)\n\n")

        # 三、ORP 分析
        f.write("## 三、ORP 分析\n\n")
        f.write("化学法数据中无 ORP 参数，以下仅分析传感器 ORP 的趋势和传感器间一致性。\n\n")

        f.write("### 3.1 传感器间 ORP 对比\n\n")
        f.write("同一位置两个传感器的 ORP 值对比，评估传感器一致性。\n\n")
        f.write("![ORP 传感器对比](images/orp_sensor_comparison.png)\n\n")

        f.write("### 3.2 ORP 时序趋势\n\n")
        for point in points:
            f.write(f"**{point}:**\n\n")
            f.write(f"![{point} ORP时序](images/{point}_orp_timeseries.png)\n\n")

        # 四、ORP vs 硝氮
        f.write("## 四、ORP vs 硝氮（传感器内部相关性）\n\n")
        f.write("分析同一传感器的 ORP 和硝氮读数之间是否存在相关性。\n\n")
        f.write("![ORP vs 硝氮](images/orp_vs_nitrate.png)\n\n")

        f.write("### ORP vs 硝氮 R²\n\n")
        f.write("| 传感器 | R² | 斜率 | 说明 |\n")
        f.write("|---|---|---|---|\n")
        for point in points:
            for sensor_name in sensors_map[point]:
                if sensor_name in sensors:
                    s = sensors[sensor_name]
                    x = s['ph_an_ph'].values
                    y = s['ph_an_an'].values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() >= 3 and len(set(x[mask])) > 1:
                        reg = do_regression(x[mask], y[mask])
                        if reg:
                            direction = "正相关" if reg['slope'] > 0 else "负相关"
                            strength = "强" if abs(reg['r2']) > 0.5 else "弱" if abs(reg['r2']) > 0.2 else "极弱"
                            f.write(f"| {sensor_name} | {reg['r2']:.3f} | {reg['slope']:.4f} | {strength}{direction} |\n")

        # 五、结论
        f.write("\n## 五、结论与建议\n\n")
        f.write("### 硝氮\n\n")

        # 计算平均 R²
        r2_list = []
        for point in points:
            for sensor_name in sensors_map[point]:
                if sensor_name in matched_data and len(matched_data[sensor_name]) > 0:
                    md = matched_data[sensor_name]
                    reg = do_regression(md['chem_nitrate'].values, md['sensor_nitrate'].values)
                    if reg:
                        r2_list.append(reg['r2'])

        if r2_list:
            avg_r2 = np.mean(r2_list)
            f.write(f"- 平均 R² = {avg_r2:.3f}\n")
            if avg_r2 >= 0.7:
                f.write("- 硝氮传感器与化学法**相关性良好**，可建立线性转换模型\n")
            elif avg_r2 >= 0.4:
                f.write("- 硝氮传感器与化学法**相关性中等**，趋势基本一致但存在偏差\n")
            else:
                f.write("- 硝氮传感器与化学法**相关性较差**，需进一步标定\n")

        f.write("\n### ORP\n\n")
        f.write("- 化学法无 ORP 数据，无法直接评估准确性\n")
        f.write("- 传感器间 ORP 一致性需关注\n")
        f.write("- ORP 与硝氮的相关性分析见第四节\n")

        f.write("\n### 建议\n\n")
        f.write("1. 对硝氮传感器建立线性回归修正模型\n")
        f.write("2. ORP 传感器建议补充化学法 ORP 比对数据\n")
        f.write("3. 关注 ORP 与硝氮的耦合关系，可能反映水体氧化还原状态\n")
        f.write("4. 定期校验传感器，更新回归方程\n")

    print(f"✅ 报告: {OUT_DIR / '硝氮和ORP分析报告.md'}")


def main():
    print("=" * 60)
    print("硝氮 & ORP 传感器分析")
    print("=" * 60)

    sensors_map = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}

    print("\n加载传感器数据...")
    sensors = load_all_sensors()

    print("\n加载化学法数据...")
    chem = load_chemical()

    print("\n时间匹配...")
    matched_data = {}
    for point, sensor_names in sensors_map.items():
        for sensor_name in sensor_names:
            if sensor_name in sensors:
                md = match_sensor_to_chem(sensors[sensor_name], chem, point, time_window_min=30)
                matched_data[sensor_name] = md
                print(f"  {sensor_name}: {len(md)} 次匹配")

    print("\n生成图表...")
    plot_nitrate_comparison(matched_data, sensors_map)
    plot_orp_analysis(sensors, sensors_map)
    plot_orp_vs_nitrate(sensors, sensors_map)

    print("\n生成报告...")
    generate_report(matched_data, sensors, sensors_map)

    print("\n完成 ✅")


if __name__ == "__main__":
    main()
