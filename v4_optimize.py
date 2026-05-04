#!/usr/bin/env python3
"""
AI4S V4 — 基于已有分子的结构优化，重点提升 binding_score
"""
import os, sys, csv, json, time, subprocess, zipfile, logging, re, random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path("/Users/pwngwc/.openclaw/workspace/retrosyn")))
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, BRICS

BASE_DIR = Path("/Users/pwngwc/ai4s_chem")
VINA = BASE_DIR / "bin" / "vina"
RECEPTOR = BASE_DIR / "receptor.pdbqt"
INPUT_CSV = BASE_DIR / "result" / "route_fix_v3_final" / "result.csv"

A_DIR = BASE_DIR / "result" / "v4_binding_strong"
B_DIR = BASE_DIR / "result" / "v4_diverse"
A_DIR.mkdir(parents=True, exist_ok=True)
B_DIR.mkdir(parents=True, exist_ok=True)

# 口袋配置（精简为3个核心口袋，加速扫描）
POCKETS = {
    "center":    ([18.3, 2.3, 21.4], [20, 20, 20]),
    "shift_s":   ([18.3, -7.7, 21.4], [20, 20, 20]),
    "shift_w":   ([8.3, 2.3, 21.4], [20, 20, 20]),
}

# ═══════════════════════════════════════════════════════════════════
# 类似物生成
# ═══════════════════════════════════════════════════════════════════

ANILINE_VARIANTS = [
    "Nc1ccccc1",                    # H
    "Nc1ccc(F)cc1",                 # 4-F
    "Nc1ccc(Cl)cc1",                # 4-Cl
    "Nc1ccc(C(F)(F)F)cc1",          # 4-CF3
    "Nc1ccc(C#N)cc1",               # 4-CN
    "Nc1ccc(OC)cc1",                # 4-OMe
    "Nc1ccc(C)cc1",                 # 4-Me
    "Nc1ccc(O)cc1",                 # 4-OH
    "Nc1ccccc1F",                   # 2-F
    "Nc1ccc(F)cc1",                 # 4-F (duplicate ok)
    "Nc1ccc(N)cc1",                 # 4-NH2
    "Nc1ccccn1",                    # 2-aminopyridine
    "Nc1cccnc1",                    # 3-aminopyridine
    "Nc1ccncc1",                    # 4-aminopyridine
]

AR_BORONIC_ACIDS = [
    # 简单芳基
    ("OB(O)c1ccccc1", "phenyl"),
    ("OB(O)c1ccc(F)cc1", "4-F-phenyl"),
    ("OB(O)c1ccc(Cl)cc1", "4-Cl-phenyl"),
    ("OB(O)c1ccc(C)cc1", "4-Me-phenyl"),
    ("OB(O)c1ccc(OC)cc1", "4-OMe-phenyl"),
    ("OB(O)c1ccc(C#N)cc1", "4-CN-phenyl"),
    ("OB(O)c1ccc(C(F)(F)F)cc1", "4-CF3-phenyl"),
    ("OB(O)c1ccc(O)cc1", "4-OH-phenyl"),
    ("OB(O)c1ccc(N)cc1", "4-NH2-phenyl"),
    ("OB(O)c1ccccc1F", "2-F-phenyl"),
    # 吡啶
    ("OB(O)c1ccccn1", "3-pyridyl"),
    ("OB(O)c1ccncc1", "4-pyridyl"),
    ("OB(O)c1cccnc1", "2-pyridyl"),
    # 嘧啶
    ("OB(O)c1cncnc1", "pyrimidinyl"),
    # 呋喃/噻吩
    ("OB(O)c1ccco1", "2-furyl"),
    ("OB(O)c1cccs1", "2-thienyl"),
    # 萘
    ("OB(O)c1ccc2ccccc2c1", "naphthyl"),
    # 喹啉/异喹啉
    ("OB(O)c1ccc2ccncc2c1", "isoquinolinyl"),
    ("OB(O)c1ccc2ncncc2c1", "quinazolinyl"),
    ("OB(O)c1ccc2ccccc2n1", "quinolinyl"),
    # 吲唑/吲哚
    ("OB(O)c1ccc2[nH]ncc2c1", "indazolyl"),
    ("OB(O)c1ccc2c(c1)cnc2", "indolyl"),
    # 苯并呋喃/苯并噻唑
    ("OB(O)c1ccc2ccoc2c1", "benzofuranyl"),
    ("OB(O)c1ccc2ccsc2c1", "benzothienyl"),
    # 吡唑
    ("OB(O)c1cc(-c2ccccc2)nn1", "pyrazolyl-phenyl"),
    # 噁二唑
    ("OB(O)c1nnc(-c2ccccc2)o1", "oxadiazolyl"),
    # CF3-喹啉
    ("OB(O)c1ccc2nccc(C(F)(F)F)c2c1", "CF3-quinolinyl"),
]


def generate_biphenyl_amide_analogs(base_mols):
    """基于 O=C(NAr)c1ccc(-Ar2)cc1 骨架生成类似物"""
    analogs = set()
    for aniline in ANILINE_VARIANTS:
        for ar_ba, ar_name in AR_BORONIC_ACIDS:
            # 构造分子
            smi = f"O=C(N{aniline.replace('N','')})c1ccc(-{ar_ba.replace('OB(O)','')})cc1"
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canon = Chem.MolToSmiles(mol)
                analogs.add(canon)
            # 反转酰胺
            smi2 = f"O=C({ar_ba.replace('OB(O)','')})N{aniline.replace('N','')}"
            mol2 = Chem.MolFromSmiles(smi2)
            if mol2:
                canon2 = Chem.MolToSmiles(mol2)
                analogs.add(canon2)
    return analogs


def generate_pyridine_amide_analogs():
    """基于 O=C(NAr)c1ccc(-Ar2)nc1 骨架"""
    analogs = set()
    for aniline in ANILINE_VARIANTS:
        for ar_ba, ar_name in AR_BORONIC_ACIDS[:15]:  # 简单芳基
            ar_part = ar_ba.replace('OB(O)', '')
            smi = f"O=C(N{aniline.replace('N','')})c1ccc(-{ar_part})nc1"
            mol = Chem.MolFromSmiles(smi)
            if mol:
                analogs.add(Chem.MolToSmiles(mol))
    return analogs


def generate_urea_analogs():
    """脲类 O=C(NAr)NAr2"""
    analogs = set()
    for a1 in ANILINE_VARIANTS[:8]:
        for a2 in ANILINE_VARIANTS[:8]:
            a1_part = a1.replace('N', '')
            a2_part = a2.replace('N', '')
            smi = f"O=C(N{a1_part})N{a2_part}"
            mol = Chem.MolFromSmiles(smi)
            if mol:
                analogs.add(Chem.MolToSmiles(mol))
    return analogs


def generate_quinazoline_amide_analogs():
    """喹唑啉酰胺"""
    analogs = set()
    quinazolines = [
        "ClC(=O)c1ncnc2ccccc12",
        "ClC(=O)c1ncnc2ccc(F)cc12",
        "ClC(=O)c1ncnc2ccc(Cl)cc12",
        "ClC(=O)c1ncnc2ccc(OC)cc12",
        "ClC(=O)c1ncnc2ccc(C)cc12",
    ]
    for q in quinazolines:
        for aniline in ANILINE_VARIANTS[:8]:
            q_mol = Chem.MolFromSmiles(q.replace("ClC(=O)", ""))
            a_mol = Chem.MolFromSmiles(aniline)
            if q_mol and a_mol:
                # 构造产物
                q_part = q.replace("ClC(=O)", "O=C(N" + aniline.replace("N", "") + ")")
                smi = q_part
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    analogs.add(Chem.MolToSmiles(mol))
    return analogs


def generate_cf3_quinazoline_analogs():
    """CF3 喹唑啉类"""
    analogs = set()
    cf3_cores = [
        "Clc1nccc2cc(C(F)(F)F)ccc12",
        "Clc1nccc2cc(C(F)(F)F)c(F)cc12",
        "Clc1nccc2cc(C(F)(F)F)c(Cl)cc12",
    ]
    for core in cf3_cores:
        for aniline in ANILINE_VARIANTS[:8]:
            core_mol = Chem.MolFromSmiles(core)
            a_mol = Chem.MolFromSmiles(aniline)
            if core_mol and a_mol:
                # SNAr: Cl + NH2 -> NH
                product_smi = core.replace("Cl", f"N{aniline.replace('N','')}")
                mol = Chem.MolFromSmiles(product_smi)
                if mol:
                    analogs.add(Chem.MolToSmiles(mol))
    return analogs


def generate_heterocycle_amide_analogs():
    """杂环酰胺类"""
    analogs = set()
    hetero_acids = [
        "ClC(=O)c1nnc(-c2ccccc2)o1",
        "ClC(=O)c1nnc(-c2ccc(F)cc2)o1",
        "ClC(=O)c1nnc(-c2ccc(Cl)cc2)o1",
        "ClC(=O)c1cnc(-c2ccccc2)[nH]1",
        "ClC(=O)c1cnc(-c2ccc(F)cc2)[nH]1",
        "ClC(=O)c1nnc(-c2ccc(OC)cc2)o1",
        "ClC(=O)c1nnc(-c2ccc(C)cc2)o1",
        "ClC(=O)c1cnc(-c2ccc(Cl)cc2)[nH]1",
    ]
    for acid in hetero_acids:
        for aniline in ANILINE_VARIANTS[:8]:
            acid_part = acid.replace("ClC(=O)", "")
            smi = f"O=C(N{aniline.replace('N','')}){acid_part}"
            mol = Chem.MolFromSmiles(smi)
            if mol:
                analogs.add(Chem.MolToSmiles(mol))
    return analogs


# ═══════════════════════════════════════════════════════════════════
# RDKit 过滤
# ═══════════════════════════════════════════════════════════════════

def calc_sa(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 10
    score = 0.0
    n = mol.GetNumHeavyAtoms()
    if n > 30: score += 2
    elif n > 20: score += 1
    rings = rdMolDescriptors.CalcNumRings(mol)
    if rings > 4: score += 2
    elif rings > 2: score += 1
    stereo = len(Chem.FindMolChiralCenters(mol))
    if stereo > 2: score += 1.5
    elif stereo > 0: score += 0.5
    return max(0, min(10, score))


def filter_mol(smi):
    """RDKit 药物样过滤"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return False, "invalid"
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    sa = calc_sa(smi)
    heavy = mol.GetNumHeavyAtoms()

    if mw < 250 or mw > 520: return False, f"MW={mw:.0f}"
    if logp < 1.5 or logp > 5.5: return False, f"LogP={logp:.1f}"
    if hbd > 4: return False, f"HBD={hbd}"
    if hba > 8: return False, f"HBA={hba}"
    if tpsa > 120: return False, f"TPSA={tpsa:.0f}"
    if sa >= 4: return False, f"SA={sa:.1f}"
    if heavy < 18: return False, f"heavy={heavy}"
    return True, "ok"


# ═══════════════════════════════════════════════════════════════════
# Vina 对接
# ═══════════════════════════════════════════════════════════════════

def smiles_to_pdbqt(smi, path):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return False
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1: return False
    try: AllChem.MMFFOptimizeMolecule(mol)
    except: pass
    try: conf = mol.GetConformer()
    except: return False
    lines = ["ROOT"]
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        sym = atom.GetSymbol()
        atype = sym.upper()
        if sym == "C": atype = "A" if atom.GetIsAromatic() else "C"
        elif sym == "N": atype = "NA" if atom.GetIsAromatic() else "N"
        elif sym == "O": atype = "OA"
        elif sym == "S": atype = "SA"
        elif sym == "H": atype = "H"
        elif sym in ("F", "Cl", "Br", "I"): atype = sym.upper()[:2]
        name4 = f' {sym.strip():<3s}'[:4]
        line = f"ATOM  {i+1:5d} {name4} LIG A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  0.00  0.00          {atype:>2s} "
        lines.append(line)
    lines.append("ENDROOT")
    lines.append("TORSDOF 0")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return True


def run_vina(ligand_pdbqt, out_pdbqt, center, size, exhaustiveness=8):
    cmd = [
        str(VINA), "--receptor", str(RECEPTOR),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
        "--out", str(out_pdbqt),
        "--num_modes", "1", "--exhaustiveness", str(exhaustiveness),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        for line in (result.stdout + result.stderr).split("\n"):
            line = line.strip()
            if line.startswith("1 "):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1])
        return None
    except:
        return None


# ═══════════════════════════════════════════════════════════════════
# 路线生成
# ═══════════════════════════════════════════════════════════════════

def make_boronic_acid(ar_smi):
    """芳基转硼酸"""
    ar_mol = Chem.MolFromSmiles(ar_smi)
    if not ar_mol: return None
    ar_canon = Chem.MolToSmiles(ar_mol)
    # 直接映射
    ba_map = {
        "c1ccc2ncncc2c1": "OB(O)c1ccc2ncncc2c1",
        "c1ccc2ccncc2c1": "OB(O)c1ccc2ccncc2c1",
        "c1ccc2ccccc2c1": "OB(O)c1ccc2ccccc2c1",
        "c1ccc2ccccc2n1": "OB(O)c1ccc2ccccc2n1",
        "c1ccc2[nH]ncc2c1": "OB(O)c1ccc2[nH]ncc2c1",
        "c1ccc2c(c1)cnc2": "OB(O)c1ccc2c(c1)cnc2",
        "c1ccc2ccoc2c1": "OB(O)c1ccc2ccoc2c1",
        "c1ccc2ccsc2c1": "OB(O)c1ccc2ccsc2c1",
        "c1ccc2nccc(C(F)(F)F)c2c1": "OB(O)c1ccc2nccc(C(F)(F)F)c2c1",
        "c1ccc2ncccc2c1": "OB(O)c1ccc2ncccc2c1",
        "c1ccc2c(c1)cnn2": "OB(O)c1ccc2c(c1)cnn2",
        "c1ccccc1": "OB(O)c1ccccc1",
        "c1ccc(F)cc1": "OB(O)c1ccc(F)cc1",
        "c1ccc(Cl)cc1": "OB(O)c1ccc(Cl)cc1",
        "c1ccc(C)cc1": "OB(O)c1ccc(C)cc1",
        "c1ccc(OC)cc1": "OB(O)c1ccc(OC)cc1",
        "c1ccc(C#N)cc1": "OB(O)c1ccc(C#N)cc1",
        "c1ccc(C(F)(F)F)cc1": "OB(O)c1ccc(C(F)(F)F)cc1",
        "c1ccc(O)cc1": "OB(O)c1ccc(O)cc1",
        "c1ccc(N)cc1": "OB(O)c1ccc(N)cc1",
        "c1ccc(S)cc1": "OB(O)c1ccc(S)cc1",
        "c1ccccc1F": "OB(O)c1ccccc1F",
        "c1ccccn1": "OB(O)c1ccccn1",
        "c1ccncc1": "OB(O)c1ccncc1",
        "c1cccnc1": "OB(O)c1cccnc1",
        "c1ccco1": "OB(O)c1ccco1",
        "c1cccs1": "OB(O)c1cccs1",
        "c1cncnc1": "OB(O)c1cncnc1",
        "c1cc(-c2ccccc2)nn1": "OB(O)c1cc(-c2ccccc2)nn1",
        "c1nnc(-c2ccccc2)o1": "OB(O)c1nnc(-c2ccccc2)o1",
    }
    if ar_canon in ba_map:
        return ba_map[ar_canon]
    # 通用
    if ar_canon.startswith("c1"):
        return f"OB(O){ar_canon}"
    return None


def generate_route(smi, canon):
    """为分子生成合成路线"""
    mol = Chem.MolFromSmiles(canon)
    if not mol: return None

    # 模式1: O=C(NAr)c1ccc(-Ar2)cc1
    m = re.match(r'O=C\(N(\w+c1\w+)\)c1ccc\(-(.+?)\)cc1', canon)
    if m:
        aniline_part = "N" + m.group(1)
        ar_part = m.group(2)
        aniline_smi = aniline_part
        ba = make_boronic_acid(ar_part)
        if ba and Chem.MolFromSmiles(aniline_smi):
            intermediate = f"O=C(N{aniline_part})c1ccc(Br)cc1"
            # 验证中间体
            if Chem.MolFromSmiles(intermediate):
                step1 = f"Brc1ccc(C(=O)Cl)cc1.{aniline_smi}>>{intermediate}"
                step2 = f"{intermediate}.{ba}>>{canon}"
                return f"{step1},{step2}"

    # 模式1b: O=C(NAr)c1ccc(-Ar2)nc1 (pyridine)
    m = re.match(r'O=C\(N(\w+c1\w+)\)c1ccc\(-(.+?)\)nc1', canon)
    if m:
        aniline_part = "N" + m.group(1)
        ar_part = m.group(2)
        aniline_smi = aniline_part
        ba = make_boronic_acid(ar_part)
        if ba and Chem.MolFromSmiles(aniline_smi):
            intermediate = f"O=C(N{aniline_part})c1ccc(Br)nc1"
            if Chem.MolFromSmiles(intermediate):
                step1 = f"Brc1ccc(C(=O)Cl)nc1.{aniline_smi}>>{intermediate}"
                step2 = f"{intermediate}.{ba}>>{canon}"
                return f"{step1},{step2}"

    # 模式2: 脲 O=C(NAr)NAr2
    m = re.match(r'O=C\(N(\w+)\)N(\w+)', canon)
    if m:
        a1 = "N" + m.group(1)
        a2 = "N" + m.group(2)
        if Chem.MolFromSmiles(a1) and Chem.MolFromSmiles(a2):
            return f"O=C=N{m.group(1)}.{a2}>>{canon}"

    # 模式3: quinazoline amide
    m = re.match(r'O=C\(N(\w+)\)c1ncnc2(\w+)c12', canon)
    if m:
        aniline_smi = "N" + m.group(1)
        if Chem.MolFromSmiles(aniline_smi):
            acid_smi = f"ClC(=O)c1ncnc2{m.group(2)}c12"
            if Chem.MolFromSmiles(acid_smi):
                return f"{acid_smi}.{aniline_smi}>>{canon}"

    # 模式4: CF3 quinazoline SNAr
    m = re.match(r'FC\(F\)\(F\)c1ccc2nccc\(N(\w+)\)c2c1', canon)
    if m:
        aniline_smi = "N" + m.group(1)
        if Chem.MolFromSmiles(aniline_smi):
            return f"Clc1nccc2cc(C(F)(F)F)ccc12.{aniline_smi}>>{canon}"

    # 模式5: 杂环酰胺
    for pattern, acid_template in [
        (r'O=C\(N(\w+)\)c1nnc\(-(.+?)\)o1', "ClC(=O)c1nnc(-{ar})o1"),
        (r'O=C\(N(\w+)\)c1cnc\(-(.+?)\)\[nH\]1', "ClC(=O)c1cnc(-{ar})[nH]1"),
    ]:
        m = re.match(pattern, canon)
        if m:
            aniline_smi = "N" + m.group(1)
            ar_part = m.group(2)
            acid_smi = acid_template.format(ar=ar_part)
            if Chem.MolFromSmiles(aniline_smi) and Chem.MolFromSmiles(acid_smi):
                return f"{acid_smi}.{aniline_smi}>>{canon}"

    # 模式6: 通用酰胺（尝试断开酰胺键）- 用 SMILES 模板
    # O=C(NAr)c1ccc(-Ar2)cc1 → Brc1ccc(C(=O)Cl)cc1.NAr>>中间体, 中间体.Ar2-B(O)O>>产物
    # 通用：找到 C(=O)N 子结构，断开
    amide_smarts = Chem.MolFromSmarts("[#7]C(=O)c1ccc([#6])cc1")
    if mol.HasSubstructMatch(amide_smarts):
        # 提取 aniline 部分和 acid 部分
        # 用 BRICS 分解
        try:
            frags = list(BRICS.BRICSDecompose(mol, returnMols=False))
            if len(frags) >= 2:
                route = ".".join(frags[:2]) + ">>" + canon
                if ">>" in route and not any(c in route for c in ["[*]", "[5*]", "[16*]"]):
                    return route
        except:
            pass

    # 模式7: 如果所有模板都失败，尝试 BRICS 分解
    try:
        frags = list(BRICS.BRICSDecompose(mol, returnMols=False))
        if len(frags) >= 2:
            route = ".".join(frags[:2]) + ">>" + canon
            if ">>" in route and not any(c in route for c in ["[*]", "[5*]", "[16*]"]):
                return route
    except:
        pass

    return None


# ═══════════════════════════════════════════════════════════════════
# 路线校验
# ═══════════════════════════════════════════════════════════════════

def validate_route(route, mol_smiles):
    result = {
        "n_steps": 0, "final_match": False, "no_dummy": True,
        "element_balance_ok": True, "no_A_to_A": True,
        "reagent_risk": "low",
        "isomer_risk": "low", "route_risk": "low",
        "notes": "",
    }
    if not route or ">>" not in route:
        return result
    for dp in ["[*]", "[5*]", "[16*]", "[14*]", "[1*]", "[3*]", "[6*]"]:
        if dp in route:
            result["no_dummy"] = False
            break
    steps = [s.strip() for s in route.split(",") if s.strip()]
    result["n_steps"] = len(steps)
    last_product = None
    for step in steps:
        if ">>" not in step: continue
        parts = step.split(">>")
        reactants_smi, product_smi = parts
        product_mol = Chem.MolFromSmiles(product_smi)
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smi.split(".") if r.strip()]
        if product_mol and all(rm for rm in reactant_mols):
            ra = {}
            for rm in reactant_mols:
                for a in rm.GetAtoms():
                    s = a.GetSymbol()
                    ra[s] = ra.get(s, 0) + 1
            pa = {}
            for a in product_mol.GetAtoms():
                s = a.GetSymbol()
                pa[s] = pa.get(s, 0) + 1
            for s, c in pa.items():
                if ra.get(s, 0) < c:
                    result["element_balance_ok"] = False
                    result["notes"] += f"{s}不平衡; "
                    break
        if product_mol and any(Chem.MolToSmiles(rm) == Chem.MolToSmiles(product_mol) for rm in reactant_mols if rm):
            result["no_A_to_A"] = False
        last_product = product_smi
    if last_product:
        lp = Chem.MolFromSmiles(last_product)
        tm = Chem.MolFromSmiles(mol_smiles)
        if lp and tm and Chem.MolToSmiles(lp) == Chem.MolToSmiles(tm):
            result["final_match"] = True
    return result


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    log_file = A_DIR / "v4_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("v4")
    log.info("=" * 60)
    log.info("AI4S V4 — binding_score 优化")
    log.info(f"开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # 1. 读取已有分子
    existing = set()
    with open(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mol = Chem.MolFromSmiles(row["mol_smiles"].strip())
            if mol:
                existing.add(Chem.MolToSmiles(mol))
    log.info(f"已有分子: {len(existing)} 个")

    # 2. 生成类似物
    log.info("\n生成类似物...")
    all_analogs = set()

    biphenyl = generate_biphenyl_amide_analogs(existing)
    log.info(f"  联苯酰胺: {len(biphenyl)}")
    all_analogs |= biphenyl

    pyridine = generate_pyridine_amide_analogs()
    log.info(f"  吡啶酰胺: {len(pyridine)}")
    all_analogs |= pyridine

    urea = generate_urea_analogs()
    log.info(f"  脲类: {len(urea)}")
    all_analogs |= urea

    quinazoline = generate_quinazoline_amide_analogs()
    log.info(f"  喹唑啉酰胺: {len(quinazoline)}")
    all_analogs |= quinazoline

    cf3_quin = generate_cf3_quinazoline_analogs()
    log.info(f"  CF3喹唑啉: {len(cf3_quin)}")
    all_analogs |= cf3_quin

    hetero = generate_heterocycle_amide_analogs()
    log.info(f"  杂环酰胺: {len(hetero)}")
    all_analogs |= hetero

    # 去除已有
    new_analogs = all_analogs - existing
    log.info(f"\n总 unique 新类似物: {len(new_analogs)}")

    # 3. RDKit 过滤
    log.info("\nRDKit 过滤...")
    filtered = []
    for smi in new_analogs:
        ok, reason = filter_mol(smi)
        if ok:
            filtered.append(smi)
    log.info(f"过滤后: {len(filtered)} 个")

    # 4. Vina 对接
    log.info(f"\n开始 Vina 对接 ({len(filtered)} 个分子)...")
    docking_dir = BASE_DIR / "docking" / "v4"
    docking_dir.mkdir(parents=True, exist_ok=True)

    # 加载中间结果
    intermediate_file = BASE_DIR / "result" / "v4_intermediate.json"
    results = []
    docked_smiles = set()
    if intermediate_file.exists():
        with open(intermediate_file) as jf:
            results = json.load(jf)
        docked_smiles = set(r["canonical"] for r in results)
        log.info(f"  加载中间结果: {len(results)} 个已对接")

    for idx, smi in enumerate(filtered):
        mol_check = Chem.MolFromSmiles(smi)
        if mol_check:
            canon_check = Chem.MolToSmiles(mol_check)
            if canon_check in docked_smiles:
                continue
        lig = docking_dir / f"mol_{idx:04d}.pdbqt"
        out = docking_dir / f"mol_{idx:04d}_out.pdbqt"
        if not smiles_to_pdbqt(smi, str(lig)):
            continue

        best_score = None
        best_pocket = None
        for pname, (center, size) in POCKETS.items():
            score = run_vina(str(lig), str(out), center, size, exhaustiveness=8)
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_pocket = pname

        if best_score is not None:
            mol = Chem.MolFromSmiles(smi)
            canon = Chem.MolToSmiles(mol)
            results.append({
                "smiles": smi,
                "canonical": canon,
                "vina": best_score,
                "pocket": best_pocket,
                "sa": calc_sa(smi),
                "mw": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
            })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            log.info(f"  进度: {idx+1}/{len(filtered)}, 有效: {len(results)}, {elapsed:.0f}s")
            # 保存中间结果
            with open(BASE_DIR / "result" / "v4_intermediate.json", "w") as jf:
                json.dump(results, jf, indent=2, ensure_ascii=False)

        # 时间限制 100 min
        if time.time() - start_time > 100 * 60:
            log.info("⏰ 时间到")
            break

    # 加上已有分子的结果（从之前的结果文件读取）
    prev_results_file = BASE_DIR / "results" / "all_results.json"
    if prev_results_file.exists():
        with open(prev_results_file) as f:
            prev = json.load(f)
        for r in prev:
            mol = Chem.MolFromSmiles(r["smiles"])
            if mol:
                canon = Chem.MolToSmiles(mol)
                if canon not in [x["canonical"] for x in results]:
                    results.append({
                        "smiles": r["smiles"],
                        "canonical": canon,
                        "vina": r.get("vina_score", 0),
                        "pocket": r.get("pocket", "unknown"),
                        "sa": r.get("sa_score", calc_sa(r["smiles"])),
                        "mw": Descriptors.MolWt(mol),
                        "logp": Descriptors.MolLogP(mol),
                        "tpsa": rdMolDescriptors.CalcTPSA(mol),
                        "hbd": rdMolDescriptors.CalcNumHBD(mol),
                        "hba": rdMolDescriptors.CalcNumHBA(mol),
                    })

    log.info(f"\n总有效结果: {len(results)}")

    # 保存完整中间结果
    with open(intermediate_file, "w") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    # 5. 按 Vina 排序
    results.sort(key=lambda x: x["vina"])

    # 6. 为 Top 分子生成路线
    log.info("\n为 Top 分子生成路线...")
    for r in results[:100]:  # Top 100
        route = generate_route(r["smiles"], r["canonical"])
        if route:
            val = validate_route(route, r["canonical"])
            r["route"] = route
            r["route_valid"] = val["final_match"] and val["no_dummy"] and val["element_balance_ok"] and val["no_A_to_A"]
            r["route_risk"] = "low" if r["route_valid"] else "high"
        else:
            r["route"] = None
            r["route_valid"] = False
            r["route_risk"] = "high"

    # 7. 保存 candidates_scored.csv
    candidates_csv = A_DIR / "candidates_scored.csv"
    with open(candidates_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mol_smiles", "canonical_smiles", "vina_best", "best_pocket",
            "sascore", "mol_weight", "logp", "tpsa", "hbd", "hba",
            "route_available", "route_risk", "notes"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "mol_smiles": r["smiles"],
                "canonical_smiles": r["canonical"],
                "vina_best": f"{r['vina']:.1f}",
                "best_pocket": r["pocket"],
                "sascore": f"{r['sa']:.1f}",
                "mol_weight": f"{r['mw']:.0f}",
                "logp": f"{r['logp']:.1f}",
                "tpsa": f"{r['tpsa']:.0f}",
                "hbd": r["hbd"],
                "hba": r["hba"],
                "route_available": "✅" if r.get("route_valid") else "❌",
                "route_risk": r.get("route_risk", "unknown"),
                "notes": "",
            })
    log.info(f"✅ candidates_scored.csv: {candidates_csv}")

    # ═══════════════════════════════════════════════════════════════
    # A版：强 binding 版
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("A版：强 binding 版")
    log.info("=" * 60)

    # 筛选：Vina ≤ -10.2 且 route 有效，逐步放宽
    a_candidates = [r for r in results if r["vina"] <= -10.2 and r.get("route_valid")]
    if len(a_candidates) < 10:
        a_candidates = [r for r in results if r["vina"] <= -10.0 and r.get("route_valid")]
    if len(a_candidates) < 10:
        a_candidates = [r for r in results if r["vina"] <= -9.8 and r.get("route_valid")]
    if len(a_candidates) < 10:
        a_candidates = [r for r in results if r.get("route_valid")][:15]

    a_final = a_candidates[:15]
    log.info(f"A版候选: {len(a_final)} 个")
    for i, r in enumerate(a_final, 1):
        log.info(f"  {i}. Vina={r['vina']:.1f} SA={r['sa']:.1f} | {r['canonical'][:50]}")

    # 写 result.csv
    a_result_csv = A_DIR / "result.csv"
    with open(a_result_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in a_final:
            writer.writerow([r["canonical"], r.get("route", "")])
    log.info(f"✅ A版 result.csv: {a_result_csv} ({len(a_final)} 行)")

    # route_validation.csv
    a_val_csv = A_DIR / "route_validation.csv"
    with open(a_val_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mol_smiles", "route", "n_steps", "final_match", "no_dummy",
            "element_balance_ok", "no_A_to_A", "reagent_risk", "isomer_risk",
            "route_risk", "submit_recommendation", "notes"
        ])
        writer.writeheader()
        for r in a_final:
            val = validate_route(r.get("route", ""), r["canonical"])
            val["mol_smiles"] = r["canonical"]
            val["route"] = r.get("route", "")
            val["submit_recommendation"] = "submit" if val["final_match"] and val["no_dummy"] and val["element_balance_ok"] else "skip"
            writer.writerow(val)

    # result.log & result.zip
    a_log = A_DIR / "result.log"
    with open(a_log, "w") as f:
        f.write(f"[{datetime.now()}] A版强binding版\n")
        f.write(f"分子数: {len(a_final)}\n")
        f.write(f"最优Vina: {a_final[0]['vina']:.1f}\n" if a_final else "无\n")
    a_zip = A_DIR / "result.zip"
    with zipfile.ZipFile(a_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(a_result_csv, "result.csv")
        zf.write(a_log, "result.log")

    # summary.md
    a_summary = A_DIR / "summary.md"
    with open(a_summary, "w") as f:
        f.write("# A版：强 binding 版\n\n")
        f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 统计\n\n")
        f.write(f"- 分子数: {len(a_final)}\n")
        if a_final:
            f.write(f"- 最优 Vina: {a_final[0]['vina']:.1f}\n")
            f.write(f"- 平均 Vina: {sum(r['vina'] for r in a_final)/len(a_final):.1f}\n")
            f.write(f"- SA 均值: {sum(r['sa'] for r in a_final)/len(a_final):.1f}\n")
        f.write(f"- 路线通过率: {sum(1 for r in a_final if r.get('route_valid'))}/{len(a_final)}\n\n")
        f.write("## Top 分子\n\n")
        for i, r in enumerate(a_final, 1):
            f.write(f"{i}. Vina={r['vina']:.1f} SA={r['sa']:.1f} `{r['canonical'][:50]}`\n")
    log.info(f"✅ A版 summary.md: {a_summary}")

    # ═══════════════════════════════════════════════════════════════
    # B版：多样性版
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("B版：多样性版")
    log.info("=" * 60)

    # 按骨架分类
    def classify_scaffold(canon):
        if "c1ncnc2" in canon and "Nc" in canon:
            return "quinazoline_amide"
        if "c1ccc2ccncc2" in canon or "c1ccc2ncncc2" in canon:
            return "nitrogen_fused"
        if "c1ccc2ccccc2" in canon:
            return "naphthyl"
        if "O=C(N" in canon and "Nc" in canon and "C(=O)N" in canon:
            return "urea"
        if "O=C(N" in canon:
            return "amide"
        return "other"

    # 每类取 Top
    b_by_class = defaultdict(list)
    for r in results:
        if r.get("route_valid"):
            cls = classify_scaffold(r["canonical"])
            b_by_class[cls].append(r)

    b_final = []
    for cls in ["quinazoline_amide", "nitrogen_fused", "naphthyl", "urea", "amide"]:
        candidates = sorted(b_by_class.get(cls, []), key=lambda x: x["vina"])
        b_final.extend(candidates[:5])

    # 去重
    seen = set()
    b_dedup = []
    for r in b_final:
        if r["canonical"] not in seen:
            seen.add(r["canonical"])
            b_dedup.append(r)
    b_final = b_dedup[:25]

    log.info(f"B版候选: {len(b_final)} 个")
    for cls in ["quinazoline_amide", "nitrogen_fused", "naphthyl", "urea", "amide"]:
        count = sum(1 for r in b_final if classify_scaffold(r["canonical"]) == cls)
        log.info(f"  {cls}: {count}")

    # 写 result.csv
    b_result_csv = B_DIR / "result.csv"
    with open(b_result_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in b_final:
            writer.writerow([r["canonical"], r.get("route", "")])
    log.info(f"✅ B版 result.csv: {b_result_csv} ({len(b_final)} 行)")

    # route_validation.csv
    b_val_csv = B_DIR / "route_validation.csv"
    with open(b_val_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mol_smiles", "route", "n_steps", "final_match", "no_dummy",
            "element_balance_ok", "no_A_to_A", "reagent_risk", "isomer_risk",
            "route_risk", "submit_recommendation", "notes"
        ])
        writer.writeheader()
        for r in b_final:
            val = validate_route(r.get("route", ""), r["canonical"])
            val["mol_smiles"] = r["canonical"]
            val["route"] = r.get("route", "")
            val["submit_recommendation"] = "submit" if val["final_match"] and val["no_dummy"] and val["element_balance_ok"] else "skip"
            writer.writerow(val)

    # result.log & result.zip
    b_log = B_DIR / "result.log"
    with open(b_log, "w") as f:
        f.write(f"[{datetime.now()}] B版多样性版\n")
        f.write(f"分子数: {len(b_final)}\n")
    b_zip = B_DIR / "result.zip"
    with zipfile.ZipFile(b_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(b_result_csv, "result.csv")
        zf.write(b_log, "result.log")

    # summary.md
    b_summary = B_DIR / "summary.md"
    with open(b_summary, "w") as f:
        f.write("# B版：多样性版\n\n")
        f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 统计\n\n")
        f.write(f"- 分子数: {len(b_final)}\n")
        if b_final:
            f.write(f"- 最优 Vina: {b_final[0]['vina']:.1f}\n")
            f.write(f"- 平均 Vina: {sum(r['vina'] for r in b_final)/len(b_final):.1f}\n")
            f.write(f"- SA 均值: {sum(r['sa'] for r in b_final)/len(b_final):.1f}\n")
        f.write(f"- 路线通过率: {sum(1 for r in b_final if r.get('route_valid'))}/{len(b_final)}\n\n")
        f.write("## 骨架分布\n\n")
        for cls in ["quinazoline_amide", "nitrogen_fused", "naphthyl", "urea", "amide"]:
            count = sum(1 for r in b_final if classify_scaffold(r["canonical"]) == cls)
            f.write(f"- {cls}: {count}\n")
        f.write("\n## Top 分子\n\n")
        for i, r in enumerate(b_final, 1):
            f.write(f"{i}. Vina={r['vina']:.1f} SA={r['sa']:.1f} `{r['canonical'][:50]}`\n")

    # ═══════════════════════════════════════════════════════════════
    # 对比报告
    # ═══════════════════════════════════════════════════════════════
    report_path = BASE_DIR / "result" / "v4_compare_report.md"
    with open(report_path, "w") as f:
        f.write("# V4 对比报告\n\n")
        f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 基线成绩\n\n")
        f.write("| 指标 | 值 |\n|------|-----|\n")
        f.write("| 总分 | 0.499492 |\n")
        f.write("| mol_score | 0.310908 |\n")
        f.write("| route_score | 0.939520 |\n")
        f.write("| binding_score | 0.163139 |\n")
        f.write("| sa_score | 0.803972 |\n")
        f.write("| sample_count | 27 |\n\n")

        f.write("## A版统计\n\n")
        f.write(f"- 分子数: {len(a_final)}\n")
        if a_final:
            f.write(f"- 最优 Vina: {a_final[0]['vina']:.1f}\n")
            f.write(f"- 平均 Vina: {sum(r['vina'] for r in a_final)/len(a_final):.1f}\n")
            f.write(f"- SA 均值: {sum(r['sa'] for r in a_final)/len(a_final):.1f}\n")
        f.write(f"- 路线通过率: {sum(1 for r in a_final if r.get('route_valid'))}/{len(a_final)}\n")
        f.write(f"- 预计优势: binding_score 显著提升\n")
        f.write(f"- 主要风险: 结构多样性不足\n\n")

        f.write("## B版统计\n\n")
        f.write(f"- 分子数: {len(b_final)}\n")
        if b_final:
            f.write(f"- 最优 Vina: {b_final[0]['vina']:.1f}\n")
            f.write(f"- 平均 Vina: {sum(r['vina'] for r in b_final)/len(b_final):.1f}\n")
            f.write(f"- SA 均值: {sum(r['sa'] for r in b_final)/len(b_final):.1f}\n")
        f.write(f"- 路线通过率: {sum(1 for r in b_final if r.get('route_valid'))}/{len(b_final)}\n")
        f.write(f"- 预计优势: 结构多样性好，评分稳定\n")
        f.write(f"- 主要风险: binding 可能不如 A版\n\n")

        f.write("## 推荐\n\n")
        a_avg = sum(r['vina'] for r in a_final)/len(a_final) if a_final else 0
        b_avg = sum(r['vina'] for r in b_final)/len(b_final) if b_final else 0
        if a_avg < b_avg:
            f.write("**推荐提交 A版**（binding 更强）\n\n")
            f.write(f"- 推荐: A版 result.zip\n")
            f.write(f"- 备选: B版 result.zip\n")
        else:
            f.write("**推荐提交 B版**（更均衡）\n\n")
            f.write(f"- 推荐: B版 result.zip\n")
            f.write(f"- 备选: A版 result.zip\n")

    log.info(f"\n✅ v4_compare_report.md: {report_path}")

    elapsed = time.time() - start_time
    log.info(f"\n总耗时: {elapsed/60:.1f} 分钟")
    log.info("V4 优化完成 ✅")

    # 打印最终信息
    print(f"\n{'='*60}")
    print(f"A版 result.zip: {a_zip}")
    print(f"B版 result.zip: {b_zip}")
    print(f"对比报告: {report_path}")
    if a_final:
        print(f"A版最优 Vina: {a_final[0]['vina']:.1f}")
    if b_final:
        print(f"B版最优 Vina: {b_final[0]['vina']:.1f}")
    print(f"推荐: {'A版' if a_avg < b_avg else 'B版'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
