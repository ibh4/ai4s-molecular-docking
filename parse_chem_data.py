#!/usr/bin/env python3
"""
解析化学法数据（彭先生提供的完整表格）并重绘全部图表
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
OUT_DIR = Path("/Users/pwngwc/Obsidian/PengWC/为知笔记/01_芯视界/中环水务数据/硝氮和ORP报告")
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE_MAP = {
    'NOA10Q10KBA0001': 'P6-1', 'NOA10Q10KBA0002': 'P6-2',
    'NOA10Q10KBA0003': 'P1-1', 'NOA10Q10KBA0004': 'P1-2',
    'NOA10Q10KBA0005': 'P7-1', 'NOA10Q10KBA0006': 'P7-2',
}
SENSORS_MAP = {'P1': ['P1-1', 'P1-2'], 'P6': ['P6-1', 'P6-2'], 'P7': ['P7-1', 'P7-2']}
COLORS = {'P1-1': '#2196F3', 'P1-2': '#FF5722', 'P6-1': '#2196F3', 'P6-2': '#FF5722',
          'P7-1': '#2196F3', 'P7-2': '#FF5722'}


def build_chem_data():
    """构建化学法数据（手动录入）"""
    # 格式: (采样时间, P1硝氮, P1_ORP, P6硝氮, P6_ORP, P7硝氮, P7_ORP)
    # 多个值取平均
    def avg(s):
        vals = [float(x) for x in s.replace('、', ',').split(',') if x.strip()]
        return np.mean(vals) if vals else np.nan

    rows = [
        # 4/8
        ("2026-04-08 08:35", "1.1,0.8,1.0", 291.2, 3.3, 395.7, 4.6, 435.3),
        ("2026-04-08 11:20", 0.6, 274.1, 3.4, 373.5, 4.9, 428.8),
        ("2026-04-08 12:25", 0.3, 253.1, 3.6, 376.2, 4.9, 451.9),
        ("2026-04-08 13:44", 0.7, 279.7, 3.6, 433.2, 4.9, 467.2),
        # 4/9
        ("2026-04-09 08:25", 0.7, 269.4, "3.5,3.3,3.3", 456.7, 4.0, 482.2),
        ("2026-04-09 10:50", 0.0, 271.9, 2.8, 371.0, 3.6, 494.5),
        ("2026-04-09 11:36", 0.7, 264.6, 2.5, 356.8, 2.9, 484.6),
        ("2026-04-09 14:06", 0.6, 226.9, 2.2, 331.4, 2.6, 478.2),
        # 4/10
        ("2026-04-10 09:06", 0.9, 695.4, 5.6, 717.2, "5.9,5.5,5.5", 727.8),
        ("2026-04-10 10:30", 0.7, 692.7, 5.7, 723.4, 5.6, 722.1),
        ("2026-04-10 12:10", 0.9, 696.7, 5.8, 720.9, 5.6, 724.3),
        ("2026-04-10 13:50", 1.1, 694.3, 5.8, 721.7, 5.7, 724.4),
        # 4/11
        ("2026-04-11 09:12", 1.1, 687.5, 5.9, 714.2, 5.6, 721.9),
        ("2026-04-11 10:23", "1.1,1.1,1.2", 689.6, 6.2, 710.5, 5.9, 714.8),
        ("2026-04-11 11:46", 1.1, 694.4, 6.1, 710.4, 5.7, 714.2),
        ("2026-04-11 13:39", 0.9, 692.1, 5.8, 715.2, 5.4, 717.5),
        # 4/12
        ("2026-04-12 09:00", 0.7, 679.4, 7.4, 694.8, 6.4, 697.1),
        ("2026-04-12 10:24", 0.7, 678.5, 7.0, 693.7, 6.4, 696.2),
        ("2026-04-12 11:57", 1.1, 681.5, 7.0, 692.6, 6.2, 693.3),
        ("2026-04-12 13:26", 0.8, 671.6, 6.6, 688.3, 5.6, 689.0),
        # 4/13
        ("2026-04-13 08:58", 1.0, 633.4, 7.4, 650.8, 6.4, 660.0),
        ("2026-04-13 10:24", 1.0, 348.4, 7.0, 457.8, "6.4,6.2,6.3", 646.9),
        ("2026-04-13 11:52", 1.0, 341.5, 7.3, 438.8, 5.8, 415.1),
        ("2026-04-13 13:50", 0.9, 293.8, 6.6, 424.4, 5.2, 426.5),
        # 4/14
        ("2026-04-14 08:49", 0.9, 310.1, 8.4, 433.3, 6.8, 409.7),
        ("2026-04-14 10:29", 1.1, 319.4, 8.2, 399.5, 6.7, 472.5),
        ("2026-04-14 11:47", "1.0,0.6,1.1", 266.0, 8.3, 423.8, 7.0, 455.7),
        ("2026-04-14 13:27", 1.8, 262.2, 8.3, 410.7, 8.3, 396.6),
        # 4/15
        ("2026-04-15 08:57", 0.9, 306.7, 8.5, 390.4, 7.0, 429.9),
        ("2026-04-15 10:24", 0.7, 288.3, 8.8, 384.4, 6.8, 395.2),
        ("2026-04-15 11:57", 0.8, 279.2, "8.5,8.4,8.1", 397.0, 6.6, 387.9),
        ("2026-04-15 13:29", 0.8, 251.5, 8.2, 380.8, 6.6, 411.5),
        # 4/16
        ("2026-04-16 08:52", 1.1, 306.1, 8.4, 430.4, 6.9, 530.8),
        ("2026-04-16 10:23", 0.8, 254.7, 8.3, 372.2, 6.8, 403.2),
        ("2026-04-16 11:45", 0.8, 271.9, 8.7, 362.3, "6.6,6.7,6.6", 377.6),
        ("2026-04-16 13:25", 0.4, 255.6, 8.0, 369.4, 6.9, 381.8),
        # 4/17
        ("2026-04-17 08:58", 0.8, 269.4, 7.5, 351.8, 6.3, 397.6),
        ("2026-04-17 10:23", 0.9, 234.6, 7.5, 355.4, 6.3, 384.2),
        ("2026-04-17 11:45", 0.7, 260.9, 7.5, 353.1, 6.2, 395.8),
        ("2026-04-17 13:27", "1,1.1,1", 204.5, 7.4, 360.1, 6.1, 389.3),
        # 4/18
        ("2026-04-18 08:59", 0.8, 215.9, 6.3, 344.9, 5.5, 402.5),
        ("2026-04-18 10:21", 0.9, 204.1, 6.4, 350.3, 5.6, 361.7),
        ("2026-04-18 11:43", 0.7, 238.0, 6.8, 343.3, 5.7, 376.9),
        ("2026-04-18 13:24", 0.8, 192.5, "6.6,6.5,6.7", 346.9, 5.8, 397.7),
        # 4/19
        ("2026-04-19 08:59", 0.9, 161.2, 4.3, 340.1, 4.5, 401.1),
        ("2026-04-19 10:23", 0.7, 203.8, 4.0, 366.8, 4.2, 388.3),
        ("2026-04-19 11:39", 0.8, 229.6, 4.5, 380.0, 4.5, 394.5),
        ("2026-04-19 13:22", 0.9, 216.4, 4.1, 360.4, "4.9,4.5,4.6", 393.2),
        # 4/20
        ("2026-04-20 08:56", "0.9,0.5,0.8", 183.4, 5.6, 383.5, 5.7, 426.6),
        ("2026-04-20 10:23", 1.0, 220.3, 5.3, 376.1, 5.4, 393.6),
        ("2026-04-20 11:54", 0.8, 188.5, 4.9, 395.9, 5.0, 392.7),
        ("2026-04-20 13:27", 0.9, 187.3, 5.5, 372.4, 5.4, 390.7),
        # 4/21
        ("2026-04-21 08:58", 1.0, 265.8, "5.7,5.9,5.8", 384.1, 5.4, 424.8),
        ("2026-04-21 10:23", 1.1, 262.8, 5.9, 393.5, 5.5, 413.2),
        ("2026-04-21 11:45", 0.7, 222.1, 5.9, 403.3, 5.3, 430.3),
        ("2026-04-21 13:25", 0.8, 199.2, 5.8, 405.4, 5.1, 404.1),
        # 4/22
        ("2026-04-22 09:00", 0.8, 232.2, 7.3, 374.1, "6.5,6.7,6.6", 383.4),
        ("2026-04-22 10:28", 0.8, 211.9, 7.1, 361.8, 6.6, 395.0),
        ("2026-04-22 11:46", 0.7, 235.8, 8.1, 366.0, 6.7, 382.5),
        ("2026-04-22 13:28", 0.6, 220.1, 6.9, 383.1, 6.5, 408.0),
        # 4/23
        ("2026-04-23 08:46", 1.4, 248.1, 6.7, 407.1, 7.2, 468.5),
        ("2026-04-23 10:22", "0.8,0.7,0.7", 230.6, 6.8, 372.4, 7.1, 428.1),
        ("2026-04-23 11:45", 0.6, 251.2, 6.6, 345.9, 7.0, 431.5),
        ("2026-04-23 13:23", 0.7, 180.5, 6.5, 390.6, 6.9, 425.3),
        # 4/24
        ("2026-04-24 08:52", 0.6, 255.3, 6.6, 371.6, 7.2, 436.9),
        ("2026-04-24 10:22", 0.9, 232.1, "6.3,6.3,6.3", 374.8, 7.0, 417.4),
        ("2026-04-24 11:49", 0.8, 233.3, 6.3, 378.3, 7.0, 396.2),
        ("2026-04-24 13:25", 0.9, 149.5, 5.8, 389.0, 6.7, 400.9),
        # 4/25
        ("2026-04-25 08:25", 0.4, 239.3, 5.9, 380.7, 6.1, 407.8),
        ("2026-04-25 10:22", 0.5, 248.6, 6.0, 393.7, "6.1,6.2,6.2", 423.8),
        ("2026-04-25 11:44", 0.4, 257.2, 6.0, 375.3, 6.1, 397.0),
        ("2026-04-25 13:22", 0.6, 185.1, 6.0, 394.0, 5.8, 419.5),
        # 4/26
        ("2026-04-26 08:49", 1.0, 203.7, 5.1, 327.8, 5.3, 433.0),
        ("2026-04-26 10:20", 0.6, 187.9, 5.0, 373.3, 5.2, 392.3),
        ("2026-04-26 11:42", "0.5,0.6,0.8", 254.3, 5.5, 383.3, 5.3, 411.6),
        ("2026-04-26 13:20", 0.5, 223.8, 5.3, 365.4, 5.5, 404.0),
        # 4/27
        ("2026-04-27 08:52", 0.6, 239.8, 6.7, 387.9, 5.2, 375.2),
        ("2026-04-27 10:26", 0.6, 242.4, 6.7, 363.0, 5.0, 367.2),
        ("2026-04-27 11:47", 0.8, 223.9, "6.2,6.2,6.1", 339.0, 5.0, 387.0),
        ("2026-04-27 13:19", 0.4, 104.4, 5.8, 348.4, 4.6, 363.8),
    ]

    def parse_val(v):
        if isinstance(v, str):
            vals = [float(x) for x in v.replace('、', ',').split(',') if x.strip()]
            return np.mean(vals) if vals else np.nan
        return float(v)

    records = []
    for r in rows:
        t = pd.to_datetime(r[0])
        records.append({
            'datetime': t,
            'P1_nitrate': parse_val(r[1]), 'P1_orp': r[2],
            'P6_nitrate': parse_val(r[3]), 'P6_orp': r[4],
            'P7_nitrate': parse_val(r[5]), 'P7_orp': r[6],
        })

    return pd.DataFrame(records)


def parse_all_sensors():
    """解析传感器数据"""
    all_rows = []
    for f in sorted(SENSOR_DIR.glob("*.md")):
        lines = f.read_text(encoding='utf-8').splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if 'collection_date' in line:
                header_idx = i
                break
        if header_idx is None:
            continue
        header = [h.strip() for h in lines[header_idx].split('|')[1:-1]]
        for line in lines[header_idx+2:]:
            if not line.strip() or '|' not in line:
                continue
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if len(cells) == len(header):
                all_rows.append(dict(zip(header, cells)))

    df = pd.DataFrame(all_rows)
    for col in ['ph_an_an', 'ph_an_ph', 'rtd_temp_f_val']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'collection_date' in df.columns:
        df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce')

    df['sensor_name'] = df['device_name'].map(DEVICE_MAP)
    df = df.dropna(subset=['sensor_name'])

    sensors = {}
    for sn, group in df.groupby('sensor_name'):
        sensors[sn] = group.sort_values('collection_date').reset_index(drop=True)
    return sensors


def do_reg(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3 or len(set(x)) < 2:
        return None
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    return {'slope': slope, 'intercept': intercept, 'r2': r_value**2,
            'rmse': np.sqrt(np.mean(residuals**2)), 'n': len(x)}


def match_sensor_to_chem(sensors, chem_df, point, sensor_name, window_min=30):
    """匹配传感器与化学法数据"""
    s = sensors[sensor_name]
    nitrate_col = f'{point}_nitrate'
    orp_col = f'{point}_orp'

    matches = []
    for _, row in chem_df.iterrows():
        chem_time = row['datetime']
        chem_nitrate = row[nitrate_col]
        chem_orp = row[orp_col]

        diff = abs((s['collection_date'] - chem_time).dt.total_seconds() / 60)
        within = s[diff <= window_min]

        if len(within) > 0:
            matches.append({
                'chem_time': chem_time,
                'chem_nitrate': chem_nitrate,
                'chem_orp': chem_orp,
                'sensor_nitrate': within['ph_an_an'].median(),
                'sensor_orp': within['ph_an_ph'].median(),
                'sensor_nitrate_std': within['ph_an_an'].std(),
                'sensor_orp_std': within['ph_an_ph'].std(),
                'n_points': len(within),
            })

    return pd.DataFrame(matches)


# ═══════════════════════════════════════════════════════════════════
# 图表生成
# ═══════════════════════════════════════════════════════════════════

def plot_nitrate_timeline(sensors, chem):
    """硝氮全量时序"""
    for point, sns in SENSORS_MAP.items():
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        for i, sn in enumerate(sns):
            ax = axes[i]
            s = sensors[sn].sort_values('collection_date')
            s_res = s.set_index('collection_date').resample('5min')['ph_an_an'].median().dropna()
            ax.plot(s_res.index, s_res.values, '-', color=COLORS[sn], linewidth=0.8, alpha=0.6, label=f'{sn} 传感器')
            ax.scatter(chem['datetime'], chem[f'{point}_nitrate'], color='black', s=80,
                      zorder=5, marker='D', edgecolors='white', linewidth=1.5, label='化学法')
            ax.set_ylabel('硝氮 (mg/L)', fontsize=11)
            ax.set_title(f'{sn}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=30, labelsize=8)
        plt.suptitle(f'{point} — 硝氮传感器 vs 化学法', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_nitrate_full_timeline.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ 硝氮全量时序")


def plot_nitrate_scatter(sensors, chem):
    """硝氮散点图"""
    for point, sns in SENSORS_MAP.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for i, sn in enumerate(sns):
            ax = axes[i]
            md = match_sensor_to_chem(sensors, chem, point, sn)
            if len(md) == 0:
                ax.text(0.5, 0.5, '无匹配数据', ha='center', va='center', transform=ax.transAxes)
                continue

            chem_v = md['chem_nitrate'].values
            sens_v = md['sensor_nitrate'].values
            ax.scatter(chem_v, sens_v, alpha=0.7, s=60, color=COLORS[sn], edgecolors='white', linewidth=1)

            if md['sensor_nitrate_std'].notna().any():
                ax.errorbar(chem_v, sens_v, yerr=md['sensor_nitrate_std'].fillna(0).values,
                           fmt='none', ecolor='gray', alpha=0.5, capsize=3)

            reg = do_reg(chem_v, sens_v)
            if reg:
                x_line = np.linspace(chem_v.min(), chem_v.max(), 100)
                ax.plot(x_line, reg['slope'] * x_line + reg['intercept'], 'r-', linewidth=2,
                       label=f"y={reg['slope']:.3f}x+{reg['intercept']:.2f}\nR²={reg['r2']:.3f}, RMSE={reg['rmse']:.2f}")

            lim_min = min(chem_v.min(), sens_v.min()) - 0.5
            lim_max = max(chem_v.max(), sens_v.max()) + 0.5
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.4, linewidth=1.5, label='1:1')

            for j, row in md.iterrows():
                ax.annotate(row['chem_time'].strftime('%m-%d %H:%M'),
                           (row['chem_nitrate'], row['sensor_nitrate']),
                           fontsize=7, alpha=0.6, textcoords="offset points", xytext=(5, 5))

            ax.set_xlabel('化学法 硝氮 (mg/L)', fontsize=11)
            ax.set_ylabel(f'{sn} 传感器 硝氮 (mg/L)', fontsize=11)
            ax.set_title(f'{sn}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{point} — 硝氮 散点图', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_nitrate_scatter_detail.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ 硝氮散点图")


def plot_nitrate_diff(sensors, chem):
    """硝氮差异分析"""
    for point, sns in SENSORS_MAP.items():
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        for i, sn in enumerate(sns):
            md = match_sensor_to_chem(sensors, chem, point, sn)
            if len(md) == 0:
                continue
            diff = md['sensor_nitrate'].values - md['chem_nitrate'].values
            pct = diff / np.where(md['chem_nitrate'].values == 0, 0.01, md['chem_nitrate'].values) * 100
            axes[0].plot(md['chem_time'], diff, 'o-', color=COLORS[sn], markersize=6, linewidth=1.5, label=sn, alpha=0.8)
            axes[1].plot(md['chem_time'], pct, 's--', color=COLORS[sn], markersize=5, linewidth=1.2, label=sn, alpha=0.8)

        axes[0].axhline(y=0, color='black', linewidth=1)
        axes[0].set_ylabel('传感器 - 化学法 (mg/L)', fontsize=11)
        axes[0].set_title(f'{point} — 硝氮绝对差值', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].axhline(y=0, color='black', linewidth=1)
        axes[1].set_ylabel('差异百分比 (%)', fontsize=11)
        axes[1].set_title(f'{point} — 硝氮相对差异', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'{point} — 硝氮差异分析', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_nitrate_difference.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ 硝氮差异图")


def plot_orp_timeline(sensors, chem):
    """ORP 全量时序 + 化学法对比"""
    for point, sns in SENSORS_MAP.items():
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        for i, sn in enumerate(sns):
            ax = axes[i]
            s = sensors[sn].sort_values('collection_date')
            s_res = s.set_index('collection_date').resample('5min')['ph_an_ph'].median().dropna()

            ax.plot(s_res.index, s_res.values, '-', color='#9C27B0', linewidth=0.8, alpha=0.7, label=f'{sn} 传感器')
            ax.scatter(chem['datetime'], chem[f'{point}_orp'], color='black', s=80,
                      zorder=5, marker='D', edgecolors='white', linewidth=1.5, label='化学法')

            mean_val = s_res.mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.6, label=f'传感器均值={mean_val:.0f}mV')

            ax.set_ylabel('ORP (mV)', fontsize=11)
            ax.set_title(f'{sn}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=30, labelsize=8)

        plt.suptitle(f'{point} — ORP 传感器 vs 化学法', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_orp_full_timeline.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ ORP 全量时序")


def plot_orp_scatter(sensors, chem):
    """ORP 散点图"""
    for point, sns in SENSORS_MAP.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for i, sn in enumerate(sns):
            ax = axes[i]
            md = match_sensor_to_chem(sensors, chem, point, sn)
            if len(md) == 0:
                continue

            chem_v = md['chem_orp'].values
            sens_v = md['sensor_orp'].values
            ax.scatter(chem_v, sens_v, alpha=0.7, s=60, color=COLORS[sn], edgecolors='white', linewidth=1)

            if md['sensor_orp_std'].notna().any():
                ax.errorbar(chem_v, sens_v, yerr=md['sensor_orp_std'].fillna(0).values,
                           fmt='none', ecolor='gray', alpha=0.5, capsize=3)

            reg = do_reg(chem_v, sens_v)
            if reg:
                x_line = np.linspace(chem_v.min(), chem_v.max(), 100)
                ax.plot(x_line, reg['slope'] * x_line + reg['intercept'], 'r-', linewidth=2,
                       label=f"y={reg['slope']:.3f}x+{reg['intercept']:.1f}\nR²={reg['r2']:.3f}, RMSE={reg['rmse']:.1f}")

            lim_min = min(chem_v.min(), sens_v.min()) - 20
            lim_max = max(chem_v.max(), sens_v.max()) + 20
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.4, linewidth=1.5, label='1:1')

            for j, row in md.iterrows():
                ax.annotate(row['chem_time'].strftime('%m-%d %H:%M'),
                           (row['chem_orp'], row['sensor_orp']),
                           fontsize=7, alpha=0.6, textcoords="offset points", xytext=(5, 5))

            ax.set_xlabel('化学法 ORP (mV)', fontsize=11)
            ax.set_ylabel(f'{sn} 传感器 ORP (mV)', fontsize=11)
            ax.set_title(f'{sn}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{point} — ORP 散点图', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_orp_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ ORP 散点图")


def plot_orp_diff(sensors, chem):
    """ORP 差异分析"""
    for point, sns in SENSORS_MAP.items():
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        for i, sn in enumerate(sns):
            md = match_sensor_to_chem(sensors, chem, point, sn)
            if len(md) == 0:
                continue
            diff = md['sensor_orp'].values - md['chem_orp'].values
            axes[0].plot(md['chem_time'], diff, 'o-', color=COLORS[sn], markersize=6, linewidth=1.5, label=sn, alpha=0.8)
            axes[1].plot(md['chem_time'], diff / np.where(md['chem_orp'].values == 0, 1, md['chem_orp'].values) * 100,
                        's--', color=COLORS[sn], markersize=5, linewidth=1.2, label=sn, alpha=0.8)

        axes[0].axhline(y=0, color='black', linewidth=1)
        axes[0].set_ylabel('传感器 - 化学法 (mV)', fontsize=11)
        axes[0].set_title(f'{point} — ORP绝对差值', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].axhline(y=0, color='black', linewidth=1)
        axes[1].set_ylabel('差异百分比 (%)', fontsize=11)
        axes[1].set_title(f'{point} — ORP相对差异', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'{point} — ORP差异分析', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_orp_difference.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ ORP 差异图")


def plot_dual_axis(sensors):
    """硝氮+ORP 双轴"""
    for point, sns in SENSORS_MAP.items():
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        for i, sn in enumerate(sns):
            ax = axes[i]
            s = sensors[sn].sort_values('collection_date')
            s_res = s.set_index('collection_date').resample('10min')[['ph_an_an', 'ph_an_ph']].median().dropna()

            ax.plot(s_res.index, s_res['ph_an_an'], '-', color='#2196F3', linewidth=1, alpha=0.8, label='硝氮')
            ax.set_ylabel('硝氮 (mg/L)', color='#2196F3', fontsize=11)
            ax.tick_params(axis='y', labelcolor='#2196F3')

            ax2 = ax.twinx()
            ax2.plot(s_res.index, s_res['ph_an_ph'], '-', color='#9C27B0', linewidth=1, alpha=0.8, label='ORP')
            ax2.set_ylabel('ORP (mV)', color='#9C27B0', fontsize=11)
            ax2.tick_params(axis='y', labelcolor='#9C27B0')

            ax.set_title(sn, fontsize=13, fontweight='bold')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=30, labelsize=8)

        plt.suptitle(f'{point} — 硝氮 & ORP 双轴时序', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{point}_nitrate_orp_dual.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ 双轴时序")


def generate_report(sensors, chem):
    """生成报告"""
    points = ['P1', 'P6', 'P7']

    # 匹配数据
    matched = {}
    for point in points:
        for sn in SENSORS_MAP[point]:
            matched[sn] = match_sensor_to_chem(sensors, chem, point, sn)

    with open(OUT_DIR / "硝氮和ORP分析报告.md", 'w') as f:
        f.write("---\ntitle: 硝氮和ORP传感器分析报告\ndate: 2026-04-27\ntags: [数据分析, 传感器, 硝氮, ORP, 中环水务]\n---\n\n")
        f.write("# 硝氮和ORP传感器数据分析报告\n\n")
        f.write(f"> **生成时间:** 2026-04-27 21:30\n")
        f.write(f"> **设备映射:** NOA10Q10KBA0003=P1-1, 0004=P1-2, 0001=P6-1, 0002=P6-2, 0005=P7-1, 0006=P7-2\n")
        f.write(f"> **化学法数据:** 4/8—4/27，每日 3-4 次采样，含硝酸盐氮 + ORP\n\n")

        # 一、硝氮
        f.write("## 一、硝氮 — 传感器 vs 化学法\n\n")
        for point in points:
            f.write(f"### {point} — 全量时序\n\n")
            f.write(f"![{point} 硝氮时序](images/{point}_nitrate_full_timeline.png)\n\n")

            f.write(f"### {point} — 散点图\n\n")
            f.write(f"![{point} 硝氮散点](images/{point}_nitrate_scatter_detail.png)\n\n")

            f.write("**回归方程：**\n\n")
            f.write("| 传感器 | 斜率 | 截距 | R² | RMSE | 样本数 |\n")
            f.write("|---|---|---|---|---|---|\n")
            for sn in SENSORS_MAP[point]:
                md = matched[sn]
                if len(md) > 0:
                    reg = do_reg(md['chem_nitrate'].values, md['sensor_nitrate'].values)
                    if reg:
                        f.write(f"| {sn} | {reg['slope']:.4f} | {reg['intercept']:.3f} | {reg['r2']:.3f} | {reg['rmse']:.3f} | {reg['n']} |\n")
            f.write("\n")

            f.write(f"### {point} — 差异分析\n\n")
            f.write(f"![{point} 硝氮差异](images/{point}_nitrate_difference.png)\n\n")

            f.write("**差异统计：**\n\n")
            f.write("| 传感器 | 平均差值 | 标准差 | 平均绝对差 | 最大差值 | 平均相对差(%) |\n")
            f.write("|---|---|---|---|---|---|\n")
            for sn in SENSORS_MAP[point]:
                md = matched[sn]
                if len(md) > 0:
                    diff = md['sensor_nitrate'].values - md['chem_nitrate'].values
                    pct = diff / np.where(md['chem_nitrate'].values == 0, 0.01, md['chem_nitrate'].values) * 100
                    f.write(f"| {sn} | {np.mean(diff):.2f} | {np.std(diff):.2f} | {np.mean(np.abs(diff)):.2f} | {np.max(np.abs(diff)):.2f} | {np.mean(np.abs(pct)):.1f}% |\n")
            f.write("\n")

        # 二、ORP
        f.write("## 二、ORP — 传感器 vs 化学法\n\n")
        for point in points:
            f.write(f"### {point} — 全量时序\n\n")
            f.write(f"![{point} ORP时序](images/{point}_orp_full_timeline.png)\n\n")

            f.write(f"### {point} — 散点图\n\n")
            f.write(f"![{point} ORP散点](images/{point}_orp_scatter.png)\n\n")

            f.write("**回归方程：**\n\n")
            f.write("| 传感器 | 斜率 | 截距 | R² | RMSE | 样本数 |\n")
            f.write("|---|---|---|---|---|---|\n")
            for sn in SENSORS_MAP[point]:
                md = matched[sn]
                if len(md) > 0:
                    reg = do_reg(md['chem_orp'].values, md['sensor_orp'].values)
                    if reg:
                        f.write(f"| {sn} | {reg['slope']:.4f} | {reg['intercept']:.1f} | {reg['r2']:.3f} | {reg['rmse']:.1f} | {reg['n']} |\n")
            f.write("\n")

            f.write(f"### {point} — 差异分析\n\n")
            f.write(f"![{point} ORP差异](images/{point}_orp_difference.png)\n\n")

            f.write("**差异统计：**\n\n")
            f.write("| 传感器 | 平均差值(mV) | 标准差 | 平均绝对差 | 最大差值 |\n")
            f.write("|---|---|---|---|---|\n")
            for sn in SENSORS_MAP[point]:
                md = matched[sn]
                if len(md) > 0:
                    diff = md['sensor_orp'].values - md['chem_orp'].values
                    f.write(f"| {sn} | {np.mean(diff):.1f} | {np.std(diff):.1f} | {np.mean(np.abs(diff)):.1f} | {np.max(np.abs(diff)):.1f} |\n")
            f.write("\n")

        # 三、双轴
        f.write("## 三、硝氮 & ORP 双轴趋势\n\n")
        for point in points:
            f.write(f"### {point}\n\n")
            f.write(f"![{point} 双轴](images/{point}_nitrate_orp_dual.png)\n\n")

        # 四、总结
        f.write("## 四、总结\n\n")

        # 硝氮 R²
        nitrate_r2 = []
        orp_r2 = []
        for point in points:
            for sn in SENSORS_MAP[point]:
                md = matched[sn]
                if len(md) > 0:
                    reg_n = do_reg(md['chem_nitrate'].values, md['sensor_nitrate'].values)
                    reg_o = do_reg(md['chem_orp'].values, md['sensor_orp'].values)
                    if reg_n:
                        nitrate_r2.append(reg_n['r2'])
                    if reg_o:
                        orp_r2.append(reg_o['r2'])

        f.write(f"- **硝氮平均 R²:** {np.mean(nitrate_r2):.3f}\n")
        f.write(f"- **ORP平均 R²:** {np.mean(orp_r2):.3f}\n\n")
        f.write("### 建议\n\n")
        f.write("1. 硝氮传感器建立线性回归修正模型\n")
        f.write("2. ORP 传感器建立线性回归修正模型\n")
        f.write("3. 定期用化学法校验传感器\n")

    # 附录：化学法原始数据
    with open(OUT_DIR / "硝氮和ORP分析报告.md", 'a') as f:
        f.write("\n\n---\n\n## 附录：化学法原始数据\n\n")
        f.write("| 采样时间 | P1硝氮(mg/L) | P1 ORP(mV) | P6硝氮(mg/L) | P6 ORP(mV) | P7硝氮(mg/L) | P7 ORP(mV) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, row in chem.iterrows():
            t = row['datetime'].strftime('%m-%d %H:%M')
            f.write(f"| {t} | {row['P1_nitrate']:.1f} | {row['P1_orp']:.1f} | {row['P6_nitrate']:.1f} | {row['P6_orp']:.1f} | {row['P7_nitrate']:.1f} | {row['P7_orp']:.1f} |\n")

    print(f"✅ 报告: {OUT_DIR / '硝氮和ORP分析报告.md'}")


def main():
    print("=" * 60)
    print("硝氮 & ORP 完整分析（含化学法 ORP）")
    print("=" * 60)

    print("\n构建化学法数据...")
    chem = build_chem_data()
    print(f"  {len(chem)} 条化学法记录")

    print("\n加载传感器数据...")
    sensors = parse_all_sensors()
    for sn in sorted(sensors.keys()):
        print(f"  {sn}: {len(sensors[sn])} 条")

    print("\n生成图表...")
    plot_nitrate_timeline(sensors, chem)
    plot_nitrate_scatter(sensors, chem)
    plot_nitrate_diff(sensors, chem)
    plot_orp_timeline(sensors, chem)
    plot_orp_scatter(sensors, chem)
    plot_orp_diff(sensors, chem)
    plot_dual_axis(sensors)

    print("\n生成报告...")
    generate_report(sensors, chem)

    print("\n完成 ✅")


if __name__ == "__main__":
    main()
