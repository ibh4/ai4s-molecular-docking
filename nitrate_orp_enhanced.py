#!/usr/bin/env python3
"""
硝氮 & ORP 增强分析 — 全量时序 + 差异分析
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

SENSOR_DIR = Path("/Users/pwngwc/中环水务数据/ORP")
CHEM_CSV = Path("/Users/pwngwc/Obsidian/PengWC/为知笔记/01_芯视界/中环水务数据/分析报告_20260425/联合数据集/化学法金标准.csv")
OUT_DIR = Path("/Users/pwngwc/Obsidian/PengWC/为知笔记/01_芯视界/中环水务数据/硝氮和ORP报告")
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

SENSORS_MAP = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}
COLORS = {'P1-1': '#2196F3', 'P1-2': '#FF5722', 'P6-1': '#2196F3', 'P6-2': '#FF5722',
          'P7-1': '#2196F3', 'P7-2': '#FF5722'}


def parse_sensor_md(filepath):
    lines = Path(filepath).read_text(encoding='utf-8').splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if 'collection_date' in line:
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()
    header = [h.strip() for h in lines[header_idx].split('|')[1:-1]]
    rows = []
    for line in lines[header_idx+2:]:
        if not line.strip() or '|' not in line:
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    df = pd.DataFrame(rows)
    for col in ['ph_an_an', 'ph_an_ph', 'rtd_temp_f_val', 'an_f_val']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'collection_date' in df.columns:
        df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce')
    return df


def load_all():
    sensors = {}
    for f in sorted(SENSOR_DIR.glob("*.md")):
        name = f.stem
        df = parse_sensor_md(f)
        if len(df) > 0:
            sensors[name] = df
    chem = pd.read_csv(CHEM_CSV)
    chem['datetime'] = pd.to_datetime(chem['datetime'])
    return sensors, chem


def do_reg(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3 or len(set(x)) < 2:
        return None
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    return {'slope': slope, 'intercept': intercept, 'r2': r_value**2, 'rmse': np.sqrt(np.mean(residuals**2)), 'n': len(x)}


def match_chem(sensors, chem, point, sensor_name, window_min=30):
    """匹配化学法和传感器数据"""
    chem_sub = chem[chem['point'] == point]
    s = sensors[sensor_name]
    matches = []
    for _, row in chem_sub.iterrows():
        ct = row['datetime']
        diff = abs((s['collection_date'] - ct).dt.total_seconds() / 60)
        within = s[diff <= window_min]
        if len(within) > 0:
            matches.append({
                'chem_time': ct,
                'chem_nitrate': row['nitrate'],
                'sensor_nitrate': within['ph_an_an'].median(),
                'sensor_orp': within['ph_an_ph'].median(),
                'sensor_nitrate_std': within['ph_an_an'].std(),
                'sensor_orp_std': within['ph_an_ph'].std(),
                'n_points': len(within),
                'time_diff_min': diff[within.index].min(),
            })
    return pd.DataFrame(matches)


# ═══════════════════════════════════════════════════════════════════
# 图1: 硝氮全量时序 — 传感器全量 + 化学法标记点
# ═══════════════════════════════════════════════════════════════════

def plot_nitrate_full_timeline(sensors, chem):
    for point, sensor_names in SENSORS_MAP.items():
        fig, axes = plt.subplots(len(sensor_names), 1, figsize=(16, 6*len(sensor_names)))
        if len(sensor_names) == 1:
            axes = [axes]

        chem_sub = chem[chem['point'] == point].sort_values('datetime')

        for i, sn in enumerate(sensor_names):
            ax = axes[i]
            s = sensors[sn].sort_values('collection_date')

            # 传感器全量数据（降采样到 5 分钟中位数）
            s_resampled = s.set_index('collection_date').resample('5min')['ph_an_an'].median().dropna()
            ax.plot(s_resampled.index, s_resampled.values, '-', color=COLORS[sn],
                   linewidth=0.8, alpha=0.6, label=f'{sn} 传感器（5min中位数）')

            # 化学法标记
            ax.scatter(chem_sub['datetime'], chem_sub['nitrate'], color='black', s=80,
                      zorder=5, marker='D', edgecolors='white', linewidth=1.5, label='化学法')

            ax.set_ylabel('硝氮 (mg/L)', fontsize=11)
            ax.set_title(f'{point} {sn} — 硝氮全量时序 vs 化学法', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10, loc='upper left')
            ax.tick_params(axis='x', rotation=30, labelsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{point} — 硝氮传感器全量数据 vs 化学法', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_nitrate_full_timeline.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ 硝氮全量时序")


# ═══════════════════════════════════════════════════════════════════
# 图2: 硝氮散点图 — 匹配点 + 回归线 + 1:1线 + 残差
# ═══════════════════════════════════════════════════════════════════

def plot_nitrate_scatter_detail(sensors, chem):
    for point, sensor_names in SENSORS_MAP.items():
        fig, axes = plt.subplots(1, len(sensor_names), figsize=(8*len(sensor_names), 6))
        if len(sensor_names) == 1:
            axes = [axes]

        for i, sn in enumerate(sensor_names):
            ax = axes[i]
            md = match_chem(sensors, chem, point, sn)
            if len(md) == 0:
                ax.text(0.5, 0.5, '无匹配数据', ha='center', va='center', transform=ax.transAxes)
                continue

            chem_vals = md['chem_nitrate'].values
            sens_vals = md['sensor_nitrate'].values

            # 散点
            ax.scatter(chem_vals, sens_vals, alpha=0.7, s=60, color=COLORS[sn],
                      edgecolors='white', linewidth=1, zorder=5)

            # 误差棒
            if md['sensor_nitrate_std'].notna().any():
                ax.errorbar(chem_vals, sens_vals, yerr=md['sensor_nitrate_std'].fillna(0).values,
                           fmt='none', ecolor='gray', alpha=0.5, capsize=3)

            # 回归线
            reg = do_reg(chem_vals, sens_vals)
            if reg:
                x_line = np.linspace(chem_vals.min(), chem_vals.max(), 100)
                y_line = reg['slope'] * x_line + reg['intercept']
                ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8,
                       label=f"回归: y={reg['slope']:.3f}x+{reg['intercept']:.2f}\nR²={reg['r2']:.3f}, RMSE={reg['rmse']:.2f}")

            # 1:1 线
            lim_min = min(chem_vals.min(), sens_vals.min()) - 1
            lim_max = max(chem_vals.max(), sens_vals.max()) + 1
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.4, linewidth=1.5, label='1:1')

            # 标注每个点的时间
            for j, row in md.iterrows():
                ax.annotate(row['chem_time'].strftime('%m-%d %H:%M'),
                           (row['chem_nitrate'], row['sensor_nitrate']),
                           fontsize=6, alpha=0.6, textcoords="offset points", xytext=(5, 5))

            ax.set_xlabel('化学法 硝氮 (mg/L)', fontsize=11)
            ax.set_ylabel(f'{sn} 传感器 硝氮 (mg/L)', fontsize=11)
            ax.set_title(f'{point} {sn}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{point} — 硝氮 化学法 vs 传感器 散点图', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_nitrate_scatter_detail.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ 硝氮散点图")


# ═══════════════════════════════════════════════════════════════════
# 图3: 硝氮差异分析 — (传感器 - 化学法) 随时间变化
# ═══════════════════════════════════════════════════════════════════

def plot_nitrate_difference(sensors, chem):
    for point, sensor_names in SENSORS_MAP.items():
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        for i, sn in enumerate(sensor_names):
            md = match_chem(sensors, chem, point, sn)
            if len(md) == 0:
                continue

            diff = md['sensor_nitrate'].values - md['chem_nitrate'].values
            pct_diff = diff / md['chem_nitrate'].values * 100

            # 差值时序
            axes[0].plot(md['chem_time'], diff, 'o-', color=COLORS[sn], markersize=6,
                        linewidth=1.5, label=f'{sn}', alpha=0.8)

            # 百分比差异时序
            axes[1].plot(md['chem_time'], pct_diff, 's--', color=COLORS[sn], markersize=5,
                        linewidth=1.2, label=f'{sn}', alpha=0.8)

        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0].set_ylabel('传感器 - 化学法 (mg/L)', fontsize=11)
        axes[0].set_title(f'{point} — 硝氮绝对差值', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=30, labelsize=8)

        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_ylabel('差异百分比 (%)', fontsize=11)
        axes[1].set_title(f'{point} — 硝氮相对差异（%）', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=30, labelsize=8)

        plt.suptitle(f'{point} — 硝氮传感器 vs 化学法 差异分析', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_nitrate_difference.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ 硝氮差异图")


# ═══════════════════════════════════════════════════════════════════
# 图4: ORP 全量时序 + 传感器间对比
# ═══════════════════════════════════════════════════════════════════

def plot_orp_full_timeline(sensors):
    for point, sensor_names in SENSORS_MAP.items():
        fig, axes = plt.subplots(len(sensor_names), 1, figsize=(16, 6*len(sensor_names)))
        if len(sensor_names) == 1:
            axes = [axes]

        for i, sn in enumerate(sensor_names):
            ax = axes[i]
            s = sensors[sn].sort_values('collection_date')
            s_resampled = s.set_index('collection_date').resample('5min')['ph_an_ph'].median().dropna()

            ax.plot(s_resampled.index, s_resampled.values, '-', color='#9C27B0',
                   linewidth=0.8, alpha=0.7, label=f'{sn} ORP')

            # 均值线
            mean_val = s_resampled.mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.6,
                      label=f'均值={mean_val:.1f}mV')

            # 标准差区间
            std_val = s_resampled.std()
            ax.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.1, color='red')

            ax.set_ylabel('ORP (mV)', fontsize=11)
            ax.set_title(f'{sn} — ORP 时序', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=30, labelsize=8)

        plt.suptitle(f'{point} — ORP 全量时序', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_orp_full_timeline.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 传感器间 ORP 对比散点
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (point, sensor_names) in enumerate(SENSORS_MAP.items()):
        ax = axes[idx]
        s1 = sensors[sensor_names[0]]
        s2 = sensors[sensor_names[1]]

        merged = pd.merge_asof(
            s1.sort_values('collection_date'),
            s2.sort_values('collection_date'),
            on='collection_date',
            tolerance=pd.Timedelta('2min'),
            suffixes=('_1', '_2')
        ).dropna(subset=['ph_an_ph_1', 'ph_an_ph_2'])

        if len(merged) > 3:
            x = merged['ph_an_ph_1'].values
            y = merged['ph_an_ph_2'].values
            ax.scatter(x, y, alpha=0.2, s=5, color='#4CAF50')
            reg = do_reg(x, y)
            if reg:
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, reg['slope'] * x_line + reg['intercept'], 'r-', linewidth=2,
                       label=f"R²={reg['r2']:.3f}")
                ax.legend(fontsize=10)
            ax.set_xlabel(f'{sensor_names[0]} ORP (mV)', fontsize=10)
            ax.set_ylabel(f'{sensor_names[1]} ORP (mV)', fontsize=10)
        ax.set_title(f'{point}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('ORP — 传感器间一致性', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'orp_sensor_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ ORP 图表")


# ═══════════════════════════════════════════════════════════════════
# 图5: 硝氮 + ORP 双轴时序
# ═══════════════════════════════════════════════════════════════════

def plot_nitrate_orp_dual(sensors):
    for point, sensor_names in SENSORS_MAP.items():
        fig, axes = plt.subplots(len(sensor_names), 1, figsize=(16, 6*len(sensor_names)))
        if len(sensor_names) == 1:
            axes = [axes]

        for i, sn in enumerate(sensor_names):
            ax = axes[i]
            s = sensors[sn].sort_values('collection_date')
            s_resampled = s.set_index('collection_date').resample('10min')[['ph_an_an', 'ph_an_ph']].median().dropna()

            color1, color2 = '#2196F3', '#9C27B0'
            ax.plot(s_resampled.index, s_resampled['ph_an_an'], '-', color=color1,
                   linewidth=1, alpha=0.8, label='硝氮')
            ax.set_ylabel('硝氮 (mg/L)', color=color1, fontsize=11)
            ax.tick_params(axis='y', labelcolor=color1)

            ax2 = ax.twinx()
            ax2.plot(s_resampled.index, s_resampled['ph_an_ph'], '-', color=color2,
                    linewidth=1, alpha=0.8, label='ORP')
            ax2.set_ylabel('ORP (mV)', color=color2, fontsize=11)
            ax2.tick_params(axis='y', labelcolor=color2)

            ax.set_title(f'{sn} — 硝氮 + ORP 双轴时序', fontsize=13, fontweight='bold')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=30, labelsize=8)

        plt.suptitle(f'{point} — 硝氮 & ORP 趋势对比', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_nitrate_orp_dual.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ 硝氮+ORP双轴图")


# ═══════════════════════════════════════════════════════════════════
# 更新报告
# ═══════════════════════════════════════════════════════════════════

def update_report(sensors, chem):
    points = ['P1', 'P6', 'P7']

    # 收集匹配数据
    matched = {}
    for point in points:
        for sn in SENSORS_MAP[point]:
            matched[sn] = match_chem(sensors, chem, point, sn)

    with open(OUT_DIR / "硝氮和ORP分析报告.md", 'w') as f:
        f.write("---\ntitle: 硝氮和ORP传感器分析报告\ndate: 2026-04-27\ntags: [数据分析, 传感器, 硝氮, ORP, 中环水务]\n---\n\n")
        f.write("# 硝氮和ORP传感器数据分析报告\n\n")
        f.write(f"> **生成时间:** 2026-04-27 20:50\n")
        f.write(f"> **传感器数据:** 6个传感器 × ~69000条/个\n")
        f.write(f"> **化学法采样:** 每点位 ~32 次匹配\n")
        f.write(f"> **分析参数:** 硝氮 (ph_an_an)、ORP (ph_an_ph)\n\n")

        # 一、硝氮全量时序
        f.write("## 一、硝氮 — 传感器全量时序 vs 化学法\n\n")
        f.write("蓝色线为传感器 5 分钟中位数降采样，黑色菱形为化学法采样点。\n\n")
        for point in points:
            f.write(f"### {point}\n\n")
            f.write(f"![{point} 硝氮全量时序](images/{point}_nitrate_full_timeline.png)\n\n")

        # 二、硝氮散点图
        f.write("## 二、硝氮 — 化学法 vs 传感器 散点图\n\n")
        f.write("每个标注点为一次化学法采样时刻对应的传感器中位数。\n\n")
        for point in points:
            f.write(f"### {point}\n\n")
            f.write(f"![{point} 硝氮散点](images/{point}_nitrate_scatter_detail.png)\n\n")

            # 回归方程
            f.write("**回归方程：**\n\n")
            f.write("| 传感器 | 斜率 | 截距 | R² | RMSE | 样本数 |\n")
            f.write("|---|---|---|---|---|---|\n")
            for sn in SENSORS_MAP[point]:
                md = matched[sn]
                if len(md) > 0:
                    reg = do_reg(md['chem_nitrate'].values, md['sensor_nitrate'].values)
                    if reg:
                        f.write(f"| {sn} | {reg['slope']:.4f} | {reg['intercept']:.3f} | "
                               f"{reg['r2']:.3f} | {reg['rmse']:.3f} | {reg['n']} |\n")
            f.write("\n")

        # 三、硝氮差异分析
        f.write("## 三、硝氮 — 传感器 vs 化学法 差异分析\n\n")
        f.write("上图：绝对差值（传感器 - 化学法）随时间变化\n")
        f.write("下图：相对差异百分比随时间变化\n\n")
        for point in points:
            f.write(f"### {point}\n\n")
            f.write(f"![{point} 硝氮差异](images/{point}_nitrate_difference.png)\n\n")

            # 差异统计
            f.write("**差异统计：**\n\n")
            f.write("| 传感器 | 平均差值 | 标准差 | 平均绝对差 | 最大差值 | 平均相对差(%) |\n")
            f.write("|---|---|---|---|---|---|\n")
            for sn in SENSORS_MAP[point]:
                md = matched[sn]
                if len(md) > 0:
                    diff = md['sensor_nitrate'].values - md['chem_nitrate'].values
                    pct = diff / md['chem_nitrate'].values * 100
                    f.write(f"| {sn} | {np.mean(diff):.2f} | {np.std(diff):.2f} | "
                           f"{np.mean(np.abs(diff)):.2f} | {np.max(np.abs(diff)):.2f} | "
                           f"{np.mean(np.abs(pct)):.1f}% |\n")
            f.write("\n")

        # 四、ORP 全量时序
        f.write("## 四、ORP — 传感器全量时序\n\n")
        f.write("化学法无 ORP 数据，仅展示传感器 ORP 趋势和传感器间一致性。\n\n")
        for point in points:
            f.write(f"### {point}\n\n")
            f.write(f"![{point} ORP全量时序](images/{point}_orp_full_timeline.png)\n\n")

        # 五、ORP 传感器间对比
        f.write("## 五、ORP — 传感器间一致性\n\n")
        f.write("![ORP 传感器对比](images/orp_sensor_comparison.png)\n\n")

        # 六、硝氮 + ORP 双轴
        f.write("## 六、硝氮 & ORP 趋势对比\n\n")
        f.write("双轴图：蓝线为硝氮，紫线为 ORP，观察两者变化趋势是否相关。\n\n")
        for point in points:
            f.write(f"### {point}\n\n")
            f.write(f"![{point} 硝氮+ORP](images/{point}_nitrate_orp_dual.png)\n\n")

        # 七、结论
        f.write("## 七、结论与建议\n\n")

        # 硝氮 R² 统计
        r2_list = []
        for point in points:
            for sn in SENSORS_MAP[point]:
                md = matched[sn]
                if len(md) > 0:
                    reg = do_reg(md['chem_nitrate'].values, md['sensor_nitrate'].values)
                    if reg:
                        r2_list.append(reg['r2'])

        f.write("### 硝氮\n\n")
        if r2_list:
            avg_r2 = np.mean(r2_list)
            f.write(f"- **平均 R² = {avg_r2:.3f}**\n")
            if avg_r2 >= 0.7:
                f.write("- 相关性良好，传感器趋势与化学法一致\n")
            elif avg_r2 >= 0.4:
                f.write("- 相关性中等，趋势基本一致但存在系统偏差\n")
            else:
                f.write("- 相关性较差，需检修或重新标定\n")

        f.write("\n### ORP\n\n")
        f.write("- 无化学法 ORP 数据，无法评估准确性\n")
        f.write("- 传感器间一致性见第五节\n")
        f.write("- ORP 与硝氮趋势对比见第六节\n")

        f.write("\n### 建议\n\n")
        f.write("1. 对硝氮传感器建立线性回归修正模型：**修正值 = (传感器 - 截距) / 斜率**\n")
        f.write("2. 补充化学法 ORP 比对数据\n")
        f.write("3. 关注 ORP 与硝氮耦合关系（氧化还原状态指示）\n")
        f.write("4. 定期用化学法校验传感器\n")

    print(f"✅ 报告: {OUT_DIR / '硝氮和ORP分析报告.md'}")


def main():
    print("=" * 60)
    print("硝氮 & ORP 增强分析")
    print("=" * 60)

    sensors, chem = load_all()

    print("\n生成图表...")
    plot_nitrate_full_timeline(sensors, chem)
    plot_nitrate_scatter_detail(sensors, chem)
    plot_nitrate_difference(sensors, chem)
    plot_orp_full_timeline(sensors)
    plot_nitrate_orp_dual(sensors)

    print("\n更新报告...")
    update_report(sensors, chem)

    print("\n完成 ✅")


if __name__ == "__main__":
    main()
