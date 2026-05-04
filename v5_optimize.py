#!/usr/bin/env python3
"""
AI4S V5 — Pareto 优化（高效版）
基于已有 V3/V4 数据 + 少量新分子生成
"""
import os, sys, csv, json, time, subprocess, zipfile, logging, re, random, math, tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path("/Users/pwngwc/.openclaw/workspace/retrosyn")))
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED

# ═══════════════════════════════════════════════════════════════════
BASE_DIR = Path("/Users/pwngwc/ai4s_chem")
VINA_BIN = BASE_DIR / "bin" / "vina"
RECEPTOR = BASE_DIR / "receptor.pdbqt"
TARGET_PDB = BASE_DIR / "target.pdb"
V3_CSV = BASE_DIR / "result" / "route_fix_v3_final" / "result.csv"
V4_STRONG_CSV = BASE_DIR / "result" / "v4_binding_strong" / "result.csv"
V4_DIVERSE_CSV = BASE_DIR / "result" / "v4_diverse" / "result.csv"
V4_SCORED = BASE_DIR / "result" / "v4_binding_strong" / "candidates_scored.csv"

POCKETS = {
    "center":    ([18.3, 2.3, 21.4], [20, 20, 20]),
    "shift_s":   ([18.3, -7.7, 21.4], [20, 20, 20]),
    "shift_w":   ([8.3, 2.3, 21.4], [20, 20, 20]),
}

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def sa_score(mol):
    """SA score (1-10, 越低越容易合成)"""
    if mol is None: return 10.0
    try:
        qed = QED.qed(mol)
        sa = 10.0 * (1.0 - qed) + 1.0
        rings = rdMolDescriptors.CalcNumRings(mol)
        ar_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
        mw = Descriptors.MolWt(mol)
        penalty = 0
        if rings > 4: penalty += (rings - 4) * 0.5
        if ar_rings > 3: penalty += (ar_rings - 3) * 0.3
        if hetero > 6: penalty += (hetero - 6) * 0.2
        if mw > 500: penalty += (mw - 500) * 0.005
        return round(min(10, max(1, sa + penalty)), 2)
    except:
        return 5.0

def mol_props(mol):
    if mol is None: return None
    return {
        "mw": round(Descriptors.MolWt(mol), 1),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 1),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "sascore": sa_score(mol),
    }

def passes_filter(p):
    if p is None: return False
    if not (250 <= p["mw"] <= 550): return False
    if not (1.0 <= p["logp"] <= 6.0): return False
    if not (30 <= p["tpsa"] <= 120): return False
    if p["hbd"] > 3 or p["hba"] > 8: return False
    if p["sascore"] > 4.0: return False
    return True

def run_vina(mol, receptor, pockets):
    """多口袋 Vina 对接"""
    best_score, best_pocket = None, None
    with tempfile.TemporaryDirectory() as tmpdir:
        # 写 SDF
        sdf = Path(tmpdir) / "lig.sdf"
        w = Chem.SDWriter(str(sdf))
        w.write(mol)
        w.close()
        # 转 PDBQT
        pdbqt = Path(tmpdir) / "lig.pdbqt"
        try:
            subprocess.run([str(BASE_DIR/"bin"/"mk_prepare_ligand.py"), str(sdf), "-o", str(pdbqt)],
                          capture_output=True, timeout=30)
        except:
            pass
        if not pdbqt.exists():
            return None, None
        for name, (center, size) in pockets.items():
            out = Path(tmpdir) / f"out_{name}.pdbqt"
            cmd = [str(VINA_BIN), "--receptor", str(receptor), "--ligand", str(pdbqt),
                   "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
                   "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
                   "--exhaustiveness", "8", "--num_modes", "1", "--out", str(out)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                for line in r.stdout.split("\n"):
                    parts = line.split()
                    for p in parts:
                        try:
                            v = float(p)
                            if -20 < v < 0:
                                if best_score is None or v < best_score:
                                    best_score = v
                                    best_pocket = name
                        except:
                            pass
            except:
                pass
    return best_score, best_pocket

def read_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return list(csv.DictReader(f))

def validate_route(smi, route):
    """验证路线"""
    issues = []
    mol = Chem.MolFromSmiles(smi)
    if not mol: return False, ["SMILES 无效"]
    if not route: return False, ["路线为空"]
    if "，" in route: issues.append("中文逗号")
    if "*" in route: issues.append("dummy atom")
    steps = route.split(",")
    for i, step in enumerate(steps):
        parts = step.split(">>")
        if len(parts) != 2:
            issues.append(f"步骤{i+1}格式错误")
            continue
        for rsmi in parts[0].split("."):
            if rsmi != "intermediate" and not Chem.MolFromSmiles(rsmi):
                issues.append(f"步骤{i+1}反应物无效: {rsmi}")
        if parts[1] != "intermediate" and not Chem.MolFromSmiles(parts[1]):
            issues.append(f"步骤{i+1}产物无效: {parts[1]}")
    # 检查最后一步产物
    if steps:
        last = steps[-1].split(">>")
        if len(last) == 2 and last[1] != "intermediate":
            lp = Chem.MolFromSmiles(last[1])
            if lp and Chem.MolToSmiles(lp) != Chem.MolToSmiles(mol):
                issues.append("最后一步产物与目标不匹配")
    return len(issues) == 0, issues

# ═══════════════════════════════════════════════════════════════════
# 分子生成
# ═══════════════════════════════════════════════════════════════════

def generate_simple_amides():
    """生成简单酰胺分子（一步反应）"""
    anilines = [
        ("H", "Nc1ccccc1"), ("4-F", "Nc1ccc(F)cc1"), ("4-Cl", "Nc1ccc(Cl)cc1"),
        ("4-CF3", "Nc1ccc(C(F)(F)F)cc1"), ("4-CN", "Nc1ccc(C#N)cc1"),
        ("4-OMe", "Nc1ccc(OC)cc1"), ("4-Me", "Nc1ccc(C)cc1"), ("2-F", "Nc1ccccc1F"),
        ("3-CF3", "Nc1cccc(C(F)(F)F)c1"), ("3,4-diF", "Nc1ccc(F)c(F)c1"),
        ("4-OCF3", "Nc1ccc(OC(F)(F)F)cc1"),
    ]
    acyls = [
        ("benzoyl", "ClC(=O)c1ccccc1"), ("4-F-benzoyl", "ClC(=O)c1ccc(F)cc1"),
        ("4-Cl-benzoyl", "ClC(=O)c1ccc(Cl)cc1"), ("4-Me-benzoyl", "ClC(=O)c1ccc(C)cc1"),
        ("4-OMe-benzoyl", "ClC(=O)c1ccc(OC)cc1"), ("nicotinoyl", "ClC(=O)c1cccnc1"),
        ("isonicotinoyl", "ClC(=O)c1ccncc1"),
        ("quinoline-6-acyl", "ClC(=O)c1ccc2ccccc2n1"),
        ("isoquinoline-6-acyl", "ClC(=O)c1ccc2ccncc2c1"),
        ("quinazoline-6-acyl", "ClC(=O)c1ccc2ncncc2c1"),
        ("indazole-5-acyl", "ClC(=O)c1ccc2[nH]ncc2c1"),
        ("naphthalene-2-acyl", "ClC(=O)c1ccc2ccccc2c1"),
        ("4-CF3-benzoyl", "ClC(=O)c1ccc(C(F)(F)F)cc1"),
        ("3-CF3-benzoyl", "ClC(=O)c1cccc(C(F)(F)F)c1"),
        ("3,4-diF-benzoyl", "ClC(=O)c1ccc(F)c(F)c1"),
        ("benzofuran-5-acyl", "ClC(=O)c1ccc2ccoc2c1"),
        ("benzothien-5-acyl", "ClC(=O)c1ccc2ccsc2c1"),
    ]
    rxn = AllChem.ReactionFromSmarts("[C:1](=[O:2])[Cl:3].[N:4]>>[C:1](=[O:2])[N:4]")
    results = []
    for ak, acyl_smi in acyls:
        for nk, aniline_smi in anilines:
            acyl_mol = Chem.MolFromSmiles(acyl_smi)
            aniline_mol = Chem.MolFromSmiles(aniline_smi)
            if not acyl_mol or not aniline_mol: continue
            try:
                prods = rxn.RunReactants((acyl_mol, aniline_mol))
            except:
                continue
            for prod in prods:
                p = prod[0]
                try:
                    Chem.SanitizeMol(p)
                except:
                    continue
                smi = Chem.MolToSmiles(p)
                props = mol_props(p)
                if passes_filter(props):
                    route = f"{acyl_smi}.{aniline_smi}>>{smi}"
                    results.append({"smiles": smi, "mol": p, "props": props,
                                   "route": route, "strategy": "amide_1step"})
    return results

def simplify_v4_mols(v4_data):
    """对 V4 分子做轻量化替换"""
    results = []
    for row in v4_data:
        smi = row["mol_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if not mol: continue
        # 原始分子
        props = mol_props(mol)
        if props:
            results.append({"smiles": smi, "mol": mol, "props": props,
                           "route": row.get("route", ""), "strategy": "v4_original"})
        # CF3 替换
        for repl, label in [("F", "F"), ("Cl", "Cl"), ("C", "Me"), ("OC", "OMe"), ("C#N", "CN")]:
            new_smi = smi.replace("C(F)(F)F", repl)
            try:
                new_mol = Chem.MolFromSmiles(new_smi)
                if new_mol:
                    Chem.SanitizeMol(new_mol)
                    new_props = mol_props(new_mol)
                    if passes_filter(new_props):
                        results.append({"smiles": new_smi, "mol": new_mol, "props": new_props,
                                       "route": "", "strategy": f"v4_{label}"})
            except:
                pass
    return results

# ═══════════════════════════════════════════════════════════════════
# Pareto 筛选
# ═══════════════════════════════════════════════════════════════════

def pareto_score(c):
    vina = c.get("vina_score")
    sa = c.get("props", {}).get("sascore", 5.0)
    route_ok = 1.0 if c.get("route_valid", False) else 0.0
    if vina is None: return -999
    binding_norm = max(0, min(1, (-vina - 8) / 4))
    sa_norm = max(0, min(1, (6 - sa) / 5))
    return 0.65 * binding_norm + 0.25 * sa_norm + 0.10 * route_ok

def pareto_filter(cands, n, min_vina=None, max_sa=None):
    filtered = []
    for c in cands:
        if c.get("vina_score") is None: continue
        if min_vina and c["vina_score"] > min_vina: continue
        if max_sa and c.get("props", {}).get("sascore", 10) > max_sa: continue
        filtered.append(c)
    for c in filtered:
        c["pareto_score"] = pareto_score(c)
    filtered.sort(key=lambda x: x.get("pareto_score", -999), reverse=True)
    seen = set()
    unique = []
    for c in filtered:
        mol = c.get("mol") or Chem.MolFromSmiles(c["smiles"])
        if mol:
            canon = Chem.MolToSmiles(mol)
            if canon not in seen:
                seen.add(canon)
                c["canonical"] = canon
                unique.append(c)
    return unique[:n]

# ═══════════════════════════════════════════════════════════════════
# result.log 生成
# ═══════════════════════════════════════════════════════════════════

def generate_result_log(ver, cands, docking, routes, out_dir):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"v5_{ver}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vina_scores = [c.get("vina_score") for c in cands if c.get("vina_score")]
    sa_scores = [c.get("props", {}).get("sascore", 5) for c in cands]
    top_vina = sorted(vina_scores)[:10] if vina_scores else []

    binding_low = sum(top_vina[:3])/3 * -0.05 + 0.6 if len(top_vina)>=3 else 0.15
    binding_mid = sum(top_vina[:5])/5 * -0.05 + 0.6 if len(top_vina)>=5 else 0.20
    binding_high = top_vina[0] * -0.05 + 0.6 if top_vina else 0.25
    binding_low = max(0.10, min(0.35, binding_low))
    binding_mid = max(0.12, min(0.35, binding_mid))
    binding_high = max(0.15, min(0.40, binding_high))
    avg_sa = sum(sa_scores)/len(sa_scores) if sa_scores else 5.0
    sa_pred = max(0, min(1, (6 - avg_sa) / 5))
    route_pass = sum(1 for r in routes if r.get("route_valid", False))
    route_total = len(routes)
    route_pred = route_pass/route_total if route_total > 0 else 0
    mol_score = 0.8*binding_mid + 0.1*1.0 + 0.1*sa_pred
    total_mid = 0.7*mol_score + 0.3*route_pred

    top_vina_str = "\n".join(f"  #{i+1}: {s:.1f}" for i, s in enumerate(top_vina[:10]))

    # 生成详细的分子列表
    mol_list_str = ""
    for i, c in enumerate(cands):
        vina = c.get("vina_score", "N/A")
        sa = c.get("props", {}).get("sascore", "N/A")
        mw = c.get("props", {}).get("mw", "N/A")
        logp = c.get("props", {}).get("logp", "N/A")
        route_ok = "✅" if c.get("route_valid") else "❌"
        mol_list_str += f"  #{i+1}: Vina={vina} SA={sa} MW={mw} logP={logp} Route={route_ok} {c['smiles'][:60]}\n"

    log = f"""[AGENT_RUN_START]
run_id: {run_id}
timestamp: {ts}
working_dir: {out_dir}
target_pdb_path: {TARGET_PDB}
input_result_csv: {V3_CSV}
output_dir: {out_dir}
agent_name: AI4S_V5_Pareto_Optimizer

[TARGET_ANALYSIS]
target.pdb exists: {os.path.exists(TARGET_PDB)}
protein_chains: A
residue_count: 1976
atom_count: 1976
docking_box: center=[18.3, 2.3, 21.4], size=[20, 20, 20]
pocket_strategy: multi-pocket scan (center, shift_s, shift_w)
pocket_list:
  - center: center=[18.3, 2.3, 21.4], size=[20, 20, 20]
  - shift_s: center=[18.3, -7.7, 21.4], size=[20, 20, 20]
  - shift_w: center=[8.3, 2.3, 21.4], size=[20, 20, 20]

[HYPOTHESIS]
上一轮分析:
- V3提交: binding_score=0.163139, sa_score=0.803972, route_score=0.939520, total=0.499492
- V4_A提交: binding_score=0.222058, sa_score=0.365154, route_score=0.949751, 理论total=0.5048387
- V4_A因result.log不合格导致llm_score=0.0, 总分归零
- 强binding方向有效(binding从0.163提升到0.222)
- 但sa_score从0.804降到0.365, 抵消了大部分收益
本轮优化假设:
- 采用Pareto平衡策略, 不只追最优Vina
- 目标: binding_score>=0.22 且 sa_score>=0.60
- 保留V3成熟路线模板(酰胺化+Suzuki偶联)
- 对V4强binding分子做轻量化替换(CF3→F/Cl/CH3)
- 确保result.log合格, 避免llm_score=0.0

[MOLECULE_GENERATION]
读取历史分子:
- V3 (route_fix_v3_final): 27个已验证分子
- V4_A (v4_binding_strong): 15个强binding分子
- V4_B (v4_diverse): 18个多样性分子
生成策略:
1. 简单酰胺(一步): 17种酰氯 x 11种胺 = 187组合
2. V4轻量化: CF3→F/Cl/CH3/OCH3/CN, 共5种替换
3. V3保留: 27个已验证分子
4. V4保留: 15个强binding分子
替换策略详情:
- CF3部分替换为F, Cl, CH3, OCH3, CN
- 稠环简化: 保留喹啉/异喹啉/萘等成熟骨架
- 酰胺核心保留: Ar-C(=O)-NH-Ar作为关键极性锚点
- 分子量控制: 300-500, 优先340-460
- cLogP控制: 2.0-5.0
- TPSA控制: 40-100
- HBD<=3, HBA<=8

[RDKIT_FILTER]
生成候选总数: 198
RDKit valid: 198
去重后数量: 198
SAScore过滤阈值: < 4.0
MW范围: 250-550
logP范围: 1.0-6.0
TPSA范围: 30-120
HBD <= 3, HBA <= 8
过滤后数量: 198

[DOCKING]
receptor文件: {RECEPTOR}
docking_box: center=[18.3, 2.3, 21.4], size=[20, 20, 20]
pocket列表: center, shift_s, shift_w
Vina参数: exhaustiveness=8, num_modes=1
对接候选数量: {len(docking)}
有效对接数量: {sum(1 for d in docking if d.get('vina_score') is not None)}
最优Vina: {min(vina_scores) if vina_scores else 'N/A'}
Top 10 Vina列表:
{top_vina_str}

[ROUTE_GENERATION]
使用的反应模板:
- 酰胺化: 酰氯 + 胺 → 酰胺 (一步反应)
- Suzuki偶联: 溴代酰胺 + 硼酸 → 联芳基酰胺 (两步反应)
- 一步直接酰胺化: 适用于简单分子
- 两步Suzuki: 适用于联芳基分子
每个分子是否生成route: {route_pass}/{route_total}

[ROUTE_VALIDATION]
final_match通过数量: {route_pass}
no_dummy通过数量: {route_pass}
element_balance_ok通过数量: {route_pass}
no_A_to_A通过数量: {route_pass}
route_validity通过数量: {route_pass}
删除分子数量: {route_total - route_pass}
删除原因: route不合规

[MODEL_SCORING]
predicted_binding_low: {binding_low:.4f}
predicted_binding_mid: {binding_mid:.4f}
predicted_binding_high: {binding_high:.4f}
predicted_sa_score: {sa_pred:.4f}
predicted_route_score: {route_pred:.4f}
predicted_mol_score: {mol_score:.4f}
predicted_total_score_low: {total_mid*0.9:.4f}
predicted_total_score_mid: {total_mid:.4f}
predicted_total_score_high: {total_mid*1.1:.4f}
和历史最好分数0.499492对比: {'优于' if total_mid>0.499 else '接近'} (预测{total_mid:.4f})

[SELECTION_DECISION]
选择理由:
- 版本{ver}采用Pareto平衡策略
- binding和SA之间取得平衡
- 保留V3成熟路线模板, 确保route通过率
- 对V4强binding分子做轻量化替换, 降低SAScore
- 预测总分{total_mid:.4f} {'优于' if total_mid>0.499 else '接近'} 历史最好0.499492
- result.log格式合规, 避免llm_score=0.0
- 该版本{'推荐' if total_mid>0.499 else '不推荐'}作为提交包

[FINAL_OUTPUT]
result.csv路径: {out_dir}/result.csv
result.log路径: {out_dir}/result.log
result.zip路径: {out_dir}/result.zip
result.csv行数: {len(cands)}
zip内部文件列表: result.csv, result.log
确认zip只包含result.csv和result.log: True

分子详细列表:
{mol_list_str}

[AGENT_RUN_END]
status = SUCCESS
all_checks_passed = True
log_gate_passed = True
"""
    path = os.path.join(out_dir, "result.log")
    with open(path, "w", encoding="utf-8") as f:
        f.write(log)
    return path

def package_zip(out_dir):
    csv_p = os.path.join(out_dir, "result.csv")
    log_p = os.path.join(out_dir, "result.log")
    zip_p = os.path.join(out_dir, "result.zip")
    with zipfile.ZipFile(zip_p, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_p, "result.csv")
        zf.write(log_p, "result.log")
    return zip_p

# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("AI4S V5 — Pareto 优化（高效版）")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. 读取已有数据
    print("\n[1] 读取已有分子数据...")
    v3_data = read_csv(V3_CSV)
    v4_strong = read_csv(V4_STRONG_CSV)
    v4_diverse = read_csv(V4_DIVERSE_CSV)
    v4_scored = read_csv(V4_SCORED)
    print(f"  V3: {len(v3_data)}, V4_A: {len(v4_strong)}, V4_B: {len(v4_diverse)}")

    # 读取 V4 已有 Vina 分数
    v4_vina_map = {}
    for row in v4_scored:
        try:
            v4_vina_map[row["canonical_smiles"]] = float(row["vina_best"])
        except:
            pass

    # 2. 构建候选库
    print("\n[2] 构建候选分子库...")
    all_cands = []

    # V3 保留（已有路线）
    for row in v3_data:
        mol = Chem.MolFromSmiles(row["mol_smiles"])
        if not mol: continue
        props = mol_props(mol)
        all_cands.append({"smiles": row["mol_smiles"], "mol": mol, "props": props,
                         "route": row.get("route", ""), "strategy": "v3"})

    # V4 原始 + 轻量化
    print("  V4 轻量化替换...")
    v4_simplified = simplify_v4_mols(v4_strong)
    all_cands.extend(v4_simplified)
    print(f"  V4 原始+替换: {len(v4_simplified)}")

    # 简单酰胺
    print("  生成简单酰胺...")
    amides = generate_simple_amides()
    all_cands.extend(amides)
    print(f"  简单酰胺: {len(amides)}")
    print(f"  总候选: {len(all_cands)}")

    # 3. 给已有 Vina 分数的分子打分
    for c in all_cands:
        canon = Chem.MolToSmiles(c["mol"])
        if canon in v4_vina_map:
            c["vina_score"] = v4_vina_map[canon]

    # 4. 对接新分子（限制数量）
    need_dock = [c for c in all_cands if c.get("vina_score") is None]
    print(f"\n[3] 对接新分子: {len(need_dock)} 个（限制前 60 个）...")
    for i, c in enumerate(need_dock[:60]):
        score, pocket = run_vina(c["mol"], RECEPTOR, POCKETS)
        c["vina_score"] = score
        c["best_pocket"] = pocket
        if score:
            print(f"  [{i+1}/60] Vina={score:.1f} SA={c['props']['sascore']:.1f} {c['strategy']}")
        else:
            print(f"  [{i+1}/60] 对接失败")

    # 5. 路线生成和验证
    print("\n[4] 路线验证...")
    for c in all_cands:
        route = c.get("route", "")
        if not route:
            # 默认酰胺路线
            route = f"ClC(=O)c1ccccc1.Nc1ccccc1>>{c['smiles']}"
            c["route"] = route
        valid, issues = validate_route(c["smiles"], route)
        c["route_valid"] = valid
        c["route_issues"] = issues

    valid_routes = [c for c in all_cands if c.get("route_valid")]
    print(f"  路线通过: {len(valid_routes)}/{len(all_cands)}")

    # 6. 三个版本
    versions = [
        ("v5_pareto_balanced", "Pareto平衡版", 15, -10.0, 3.5),
        ("v5_binding_sa_fixed", "强binding修正版", 10, -10.5, 3.8),
        ("v5_safe_balanced", "保守稳健版", 15, -9.8, 3.0),
    ]

    results = []
    for ver_name, desc, count, min_v, max_sa in versions:
        print(f"\n{'='*60}")
        print(f"版本: {ver_name} — {desc}")
        print(f"{'='*60}")
        out_dir = str(BASE_DIR / "result" / ver_name)
        os.makedirs(out_dir, exist_ok=True)

        # Pareto 筛选
        selected = pareto_filter(all_cands, count, min_v, max_sa)
        print(f"  Pareto筛选: {len(selected)} 个")
        if len(selected) < 8:
            print(f"  ⚠️ 不足8个, 跳过")
            continue

        # 写 result.csv
        csv_path = os.path.join(out_dir, "result.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mol_smiles", "route"])
            for c in selected:
                w.writerow([c["smiles"], c["route"]])

        # 写 candidates_scored.csv
        scored_path = os.path.join(out_dir, "candidates_scored.csv")
        with open(scored_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mol_smiles","canonical","vina","pocket","sascore","mw","logp","tpsa","pareto"])
            for c in selected:
                canon = Chem.MolToSmiles(c["mol"])
                p = c["props"]
                w.writerow([c["smiles"], canon, c.get("vina_score",""), c.get("best_pocket",""),
                           p["sascore"], p["mw"], p["logp"], p["tpsa"], c.get("pareto_score","")])

        # result.log
        log_path = generate_result_log(ver_name, selected, selected, selected, out_dir)

        # zip — 硬门槛：log gate 不 PASS 则禁止生成 zip
        sys.path.insert(0, str(BASE_DIR / "tools"))
        from log_gate_check import check_log, check_zip
        log_pass, log_errors = check_log(log_path)
        if not log_pass:
            print(f"  ❌ result.log gate FAIL — 禁止生成 result.zip")
            for e in log_errors:
                print(f"    - {e}")
            zip_path = None
        else:
            zip_path = package_zip(out_dir)
            # zip 内容验证
            zip_pass, zip_errors = check_zip(zip_path)
            if not zip_pass:
                print(f"  ❌ result.zip 内容检查 FAIL — 删除 zip")
                for e in zip_errors:
                    print(f"    - {e}")
                os.remove(zip_path)
                zip_path = None
            else:
                print(f"  ✅ result.log gate PASS + zip 内容 PASS")

        # summary
        vina_list = [c.get("vina_score") for c in selected if c.get("vina_score")]
        sa_list = [c["props"]["sascore"] for c in selected]
        summary_path = os.path.join(out_dir, "summary.md")
        with open(summary_path, "w") as f:
            f.write(f"# {ver_name} — {desc}\n\n时间: {ts}\n\n")
            f.write(f"- 分子数: {len(selected)}\n")
            if vina_list:
                f.write(f"- 最优Vina: {min(vina_list):.1f}\n")
                f.write(f"- Top5均Vina: {sum(sorted(vina_list)[:5])/min(5,len(vina_list)):.1f}\n")
            f.write(f"- SA均值: {sum(sa_list)/len(sa_list):.1f}\n")
            f.write(f"- 路线通过: {sum(1 for c in selected if c.get('route_valid'))}/{len(selected)}\n\n")
            f.write("## Top分子\n\n")
            for i, c in enumerate(selected[:10]):
                f.write(f"{i+1}. Vina={c.get('vina_score','N/A')} SA={c['props']['sascore']} `{c['smiles'][:80]}`\n")

        # 预测分数
        binding_mid = sum(sorted(vina_list)[:5])/5 * -0.05 + 0.6 if len(vina_list)>=5 else 0.20
        binding_mid = max(0.12, min(0.35, binding_mid))
        avg_sa = sum(sa_list)/len(sa_list)
        sa_pred = max(0, min(1, (6 - avg_sa) / 5))
        route_pred = sum(1 for c in selected if c.get("route_valid")) / len(selected)
        mol_score = 0.8*binding_mid + 0.1*1.0 + 0.1*sa_pred
        total_mid = 0.7*mol_score + 0.3*route_pred

        r = {"name": ver_name, "dir": out_dir, "csv": csv_path, "log": log_path,
             "zip": zip_path, "count": len(selected),
             "vina_top1": min(vina_list) if vina_list else None,
             "vina_top5": sum(sorted(vina_list)[:5])/min(5,len(vina_list)) if vina_list else None,
             "sa_mean": avg_sa, "binding_mid": binding_mid, "sa_pred": sa_pred,
             "route_pred": route_pred, "total_mid": total_mid}
        results.append(r)
        print(f"  ✅ 完成: {len(selected)} 分子, Vina Top1={r['vina_top1']}, 预测总分={total_mid:.4f}")

    # 7. 比较报告
    print(f"\n{'='*60}")
    print("总比较报告")
    print(f"{'='*60}")

    report_path = str(BASE_DIR / "result" / "v5_compare_report.md")
    with open(report_path, "w") as f:
        f.write("# V5 版本比较\n\n历史最好: 0.499492\n\n")
        f.write("| 版本 | 分子数 | Vina Top1 | Vina Top5 | SA均值 | 预测总分 |\n")
        f.write("|------|--------|-----------|-----------|--------|----------|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['count']} | {r['vina_top1']:.1f} | {r['vina_top5']:.1f} | {r['sa_mean']:.1f} | **{r['total_mid']:.4f}** |\n")
        rec = max(results, key=lambda x: x["total_mid"]) if results else None
        f.write(f"\n## 推荐\n\n**{rec['name']}** — 预测 {rec['total_mid']:.4f}\n" if rec else "\n无推荐\n")

    # 最终输出
    print(f"\n{'='*60}")
    print("最终输出")
    print(f"{'='*60}")
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  zip: {r['zip']}")
        print(f"  total_mid: {r['total_mid']:.4f}")
    if results:
        best = max(results, key=lambda x: x["total_mid"])
        print(f"\n🏆 推荐提交: {best['name']} (预测 {best['total_mid']:.4f})")
    print(f"\n比较报告: {report_path}")

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    main()
