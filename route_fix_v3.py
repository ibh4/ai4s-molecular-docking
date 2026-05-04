#!/usr/bin/env python3
"""
AI4S Route Fix V3 — 修复 Top30 分子的合成路线
用真实商业起始原料替换 dummy atom 片段
"""
import os, sys, csv, json, time, zipfile, logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path("/Users/pwngwc/.openclaw/workspace/retrosyn")))
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors

# ── 配置 ──────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/pwngwc/ai4s_chem")
INPUT_CSV = BASE_DIR / "result" / "2026-04-27_073829" / "result.csv"
OUT_DIR = BASE_DIR / "result" / "route_fix_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "result.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("route_fix")

# ═══════════════════════════════════════════════════════════════════
# 一、读取并规范化输入
# ═══════════════════════════════════════════════════════════════════

def read_input(csv_path):
    """读取 result.csv，canonical SMILES，去重"""
    log.info(f"读取输入: {csv_path}")
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row["mol_smiles"].strip()
            route = row["route"].strip()
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canon = Chem.MolToSmiles(mol)
                rows.append({"mol_smiles": smi, "canonical": canon, "route": route})
            else:
                log.warning(f"  ❌ 无法解析: {smi}")

    # 去重（保留第一个，即 Vina 更优的）
    seen = set()
    unique = []
    for r in rows:
        if r["canonical"] not in seen:
            seen.add(r["canonical"])
            unique.append(r)
        else:
            log.info(f"  🔁 删除重复: {r['mol_smiles'][:50]}")

    log.info(f"输入: {len(rows)} 行, 去重后: {len(unique)} 个")
    return unique


# ═══════════════════════════════════════════════════════════════════
# 二、分子结构分析与路线设计
# ═══════════════════════════════════════════════════════════════════

def analyze_molecule(smi):
    """分析分子结构，确定合成策略"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None

    canon = Chem.MolToSmiles(mol)
    info = {
        "smiles": smi,
        "canonical": canon,
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "pattern": None,
        "aniline_part": None,
        "acid_part": None,
        "ar_part": None,
        "strategy": None,
    }

    # 检测核心骨架
    # 模式1: O=C(Nc1ccccc1)c1ccc(-Ar)cc1 (biphenyl amide)
    if "O=C(Nc1ccccc1)c1ccc(-" in canon and ")cc1" in canon:
        info["pattern"] = "biphenyl_amide"
        info["strategy"] = "amide_coupling_suzuki"

    # 模式1b: O=C(Nc1ccc(X)cc1)c1ccc(-Ar)cc1 (substituted aniline)
    elif "O=C(Nc1ccc(" in canon and ")cc1)c1ccc(-" in canon:
        info["pattern"] = "substituted_aniline_biphenyl"
        info["strategy"] = "amide_coupling_suzuki"

    # 模式2: O=C(Nc1ccccc1)c1ccc(-Ar)nc1 (pyridine amide)
    elif "O=C(Nc1ccccc1)c1ccc(-" in canon and ")nc1" in canon:
        info["pattern"] = "pyridine_amide"
        info["strategy"] = "amide_coupling_suzuki"

    # 模式3: O=C(Nc1ccccc1)Nc1ccc(-Ar)cc1 (urea)
    elif "O=C(Nc1ccccc1)Nc1ccc(-" in canon:
        info["pattern"] = "urea"
        info["strategy"] = "urea_formation"

    # 模式4: O=C(Nc1ccccc1)c1ccc(-c2ccccc2)nc1 (pyridyl biphenyl)
    elif "O=C(Nc1ccccc1)c1ccc(-c2ccccc2)nc1" == canon:
        info["pattern"] = "pyridyl_biphenyl"
        info["strategy"] = "amide_coupling_suzuki"

    # 模式5: quinazoline amide
    elif "O=C(Nc1ccccc1)c1ncnc2ccccc12" == canon:
        info["pattern"] = "quinazoline_amide"
        info["strategy"] = "amide_coupling"

    # 模式6: CF3-quinazoline
    elif "FC(F)(F)c1ccc2nccc(Nc3ccccc3)c2c1" == canon:
        info["pattern"] = "cf3_quinazoline"
        info["strategy"] = "nucleophilic_substitution"

    # 模式7: heterocycle-linked amides
    elif "O=C(Nc1ccccc1)c1" in canon and any(h in canon for h in ["nnc", "cnc", "ccn"]):
        info["pattern"] = "heterocycle_amide"
        info["strategy"] = "amide_coupling_heterocycle"

    # 模式8: COc1ccc(NC(=O)c2ccnc3ccccc23)cc1 (methoxy aniline quinoline)
    elif "COc1ccc(NC(=O)" in canon:
        info["pattern"] = "methoxy_aniline_amide"
        info["strategy"] = "amide_coupling"

    else:
        info["pattern"] = "unknown"
        info["strategy"] = "generic"

    return info


# ═══════════════════════════════════════════════════════════════════
# 三、路线生成器
# ═══════════════════════════════════════════════════════════════════

def make_route_biphenyl_amide(smi, canon):
    """
    O=C(Nc1ccccc1)c1ccc(-Ar)cc1
    Step 1: Brc1ccc(C(=O)Cl)cc1 + Nc1ccccc1 >> O=C(Nc1ccccc1)c1ccc(Br)cc1
    Step 2: O=C(Nc1ccccc1)c1ccc(Br)cc1 + Ar-B(O)O >> final
    """
    # 提取 Ar 部分
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None

    # 找到 biphenyl 连接点，提取 Ar 片段
    # 用 SMILES 模式匹配
    import re

    # 模式: O=C(Nc1ccccc1)c1ccc(-AR)cc1
    m = re.match(r'O=C\(Nc1ccccc1\)c1ccc\(-(.+?)\)cc1', canon)
    if m:
        ar_smi = m.group(1)
        ar_mol = Chem.MolFromSmiles(ar_smi)
        if ar_mol:
            # 构造硼酸
            boronic_acid = make_boronic_acid(ar_smi)
            if boronic_acid:
                intermediate = "O=C(Nc1ccccc1)c1ccc(Br)cc1"
                step1 = f"Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>{intermediate}"
                step2 = f"{intermediate}.{boronic_acid}>>{canon}"
                return f"{step1},{step2}"

    # 模式: O=C(Nc1ccccc1)c1ccc(-AR)nc1 (pyridine)
    m = re.match(r'O=C\(Nc1ccccc1\)c1ccc\(-(.+?)\)nc1', canon)
    if m:
        ar_smi = m.group(1)
        boronic_acid = make_boronic_acid(ar_smi)
        if boronic_acid:
            intermediate = "O=C(Nc1ccccc1)c1ccc(Br)nc1"
            step1 = f"Brc1ccc(C(=O)Cl)nc1.Nc1ccccc1>>{intermediate}"
            step2 = f"{intermediate}.{boronic_acid}>>{canon}"
            return f"{step1},{step2}"

    return None


def make_route_substituted_aniline(smi, canon):
    """
    O=C(Nc1ccc(X)cc1)c1ccc(-Ar)cc1
    Step 1: Brc1ccc(C(=O)Cl)cc1 + Nc1ccc(X)cc1 >> O=C(Nc1ccc(X)cc1)c1ccc(Br)cc1
    Step 2: intermediate + Ar-B(O)O >> final
    """
    import re

    # 模式: O=C(Nc1ccc(X)cc1)c1ccc(-AR)cc1
    m = re.match(r'O=C\(Nc1ccc\((.+?)\)cc1\)c1ccc\(-(.+?)\)cc1', canon)
    if m:
        x_smi = m.group(1)
        ar_smi = m.group(2)

        # 构造取代苯胺
        aniline = f"Nc1ccc({x_smi})cc1"
        aniline_mol = Chem.MolFromSmiles(aniline)
        if not aniline_mol:
            return None

        boronic_acid = make_boronic_acid(ar_smi)
        if boronic_acid:
            intermediate = f"O=C(Nc1ccc({x_smi})cc1)c1ccc(Br)cc1"
            step1 = f"Brc1ccc(C(=O)Cl)cc1.{aniline}>>{intermediate}"
            step2 = f"{intermediate}.{boronic_acid}>>{canon}"
            return f"{step1},{step2}"

    return None


def make_route_urea(smi, canon):
    """
    O=C(Nc1ccccc1)Nc1ccc(-Ar)cc1
    Step 1: c1ccc(N=C=O)cc1 + Nc1ccc(-Ar)cc1 >> urea
    或: PhNCO + Ar-NH2 >> urea
    """
    import re

    m = re.match(r'O=C\(Nc1ccccc1\)Nc1ccc\(-(.+?)\)cc1', canon)
    if m:
        ar_smi = m.group(1)
        aniline_ar = f"Nc1ccc(-{ar_smi})cc1"
        aniline_ar_mol = Chem.MolFromSmiles(aniline_ar)
        if aniline_ar_mol:
            step1 = f"O=C=Nc1ccccc1.{aniline_ar}>>{canon}"
            return step1

    return None


def make_route_quinazoline_amide(smi, canon):
    """
    O=C(Nc1ccccc1)c1ncnc2ccccc12
    Step 1: c1ncnc2ccccc12C(=O)Cl + Nc1ccccc1 >> final
    """
    if canon == "O=C(Nc1ccccc1)c1ncnc2ccccc12":
        step1 = "ClC(=O)c1ncnc2ccccc12.Nc1ccccc1>>O=C(Nc1ccccc1)c1ncnc2ccccc12"
        return step1
    return None


def make_route_cf3_quinazoline(smi, canon):
    """
    FC(F)(F)c1ccc2nccc(Nc3ccccc3)c2c1
    Step 1: FC(F)(F)c1ccc2nccc(Cl)c2c1 + Nc1ccccc1 >> final (SNAr)
    """
    if canon == "FC(F)(F)c1ccc2nccc(Nc3ccccc3)c2c1":
        step1 = "Clc1nccc2cc(C(F)(F)F)ccc12.Nc1ccccc1>>FC(F)(F)c1ccc2nccc(Nc3ccccc3)c2c1"
        return step1
    return None


def make_route_methoxy_aniline_amide(smi, canon):
    """
    COc1ccc(NC(=O)c2ccnc3ccccc23)cc1
    Step 1: ClC(=O)c1ccnc2ccccc12 + COc1ccc(N)cc1 >> final
    """
    import re

    if "COc1ccc(NC(=O)" in canon:
        # 提取酰氯部分
        product_mol = Chem.MolFromSmiles(canon)
        if product_mol:
            # 逆向: 断开酰胺键
            step1 = "ClC(=O)c1ccnc2ccccc12.COc1ccc(N)cc1>>" + canon
            return step1
    return None


def make_route_heterocycle_amide(smi, canon):
    """
    O=C(Nc1ccccc1)c1[het](-c2ccccc2)...
    对于含杂环 linker 的酰胺，尝试酰胺偶联
    """
    mol = Chem.MolFromSmiles(canon)
    if not mol:
        return None

    # 尝试断开酰胺键 C(=O)N
    # 逆向: Ar-COCl + H2N-Ar' >> Ar-C(=O)NH-Ar'
    # 找到 C(=O) 键
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if (a1.GetSymbol() == "C" and a2.GetSymbol() == "N" and
            any(n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(a1.GetIdx(), n.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE
                for n in a1.GetNeighbors() if n.GetSymbol() == "O")):
            # 这是酰胺键 C(=O)-N
            # 分割分子
            amide_bond_idx = bond.GetIdx()
            # 用 BRICS 或手动分割
            try:
                frags = list(Chem.rdmolops.FragmentOnBonds(mol, [amide_bond_idx], addDummies=True))
                # 不太好用，换一个方式
            except:
                pass

    # 直接用模板匹配
    import re

    # O=C(Nc1ccccc1)c1nnc(-c2ccccc2)o1
    if "O=C(Nc1ccccc1)c1nnc(-c2ccccc2)o1" == canon:
        return "ClC(=O)c1nnc(-c2ccccc2)o1.Nc1ccccc1>>" + canon

    # O=C(Nc1ccccc1)c1cnc(-c2ccccc2)[nH]1
    if "O=C(Nc1ccccc1)c1cnc(-c2ccccc2)[nH]1" == canon:
        return "ClC(=O)c1cnc(-c2ccccc2)[nH]1.Nc1ccccc1>>" + canon

    # O=C(Nc1ccccc1)c1ccc(-c2ccncc2)cc1
    if "O=C(Nc1ccccc1)c1ccc(-c2ccncc2)cc1" == canon:
        intermediate = "O=C(Nc1ccccc1)c1ccc(Br)cc1"
        return f"Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>{intermediate},{intermediate}.OB(O)c1ccncc1>>{canon}"

    # O=C(Nc1ccccc1)c1ccc(-c2ccco2)cc1
    if "O=C(Nc1ccccc1)c1ccc(-c2ccco2)cc1" == canon:
        intermediate = "O=C(Nc1ccccc1)c1ccc(Br)cc1"
        return f"Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>{intermediate},{intermediate}.OB(O)c1ccco1>>{canon}"

    # O=C(Nc1ccccc1)c1ccc(-c2ccc(S)cc2)cc1
    if "O=C(Nc1ccccc1)c1ccc(-c2ccc(S)cc2)cc1" == canon:
        intermediate = "O=C(Nc1ccccc1)c1ccc(Br)cc1"
        return f"Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>{intermediate},{intermediate}.OB(O)c1ccc(S)cc1>>{canon}"

    # O=C(Nc1ccccc1)c1ccc(-c2ccc(O)cc2)cc1
    if "O=C(Nc1ccccc1)c1ccc(-c2ccc(O)cc2)cc1" == canon:
        intermediate = "O=C(Nc1ccccc1)c1ccc(Br)cc1"
        return f"Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>{intermediate},{intermediate}.OB(O)c1ccc(O)cc1>>{canon}"

    # O=C(Nc1ccccc1)c1ccc(-c2ccccc2F)cc1 (2-fluorobiphenyl)
    if "O=C(Nc1ccccc1)c1ccc(-c2ccccc2F)cc1" == canon:
        intermediate = "O=C(Nc1ccccc1)c1ccc(Br)cc1"
        return f"Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>{intermediate},{intermediate}.OB(O)c1ccccc1F>>{canon}"

    # N#Cc1ccc(NC(=O)c2ccnc3ccccc23)cc1
    if "N#Cc1ccc(NC(=O)c2ccnc3ccccc23)cc1" == canon:
        return "ClC(=O)c1ccnc2ccccc12.N#Cc1ccc(N)cc1>>" + canon

    return None


def make_route_generic(smi, canon):
    """通用路线：尝试酰胺偶联"""
    mol = Chem.MolFromSmiles(canon)
    if not mol:
        return None

    # 检查是否有酰胺键
    amide_pattern = Chem.MolFromSmarts("[#6]-C(=O)-N-[#6]")
    if mol.HasSubstructMatch(amide_pattern):
        # 尝试断开酰胺键
        # 简化：用 BRICS 分解
        try:
            frags = list(BRICS.BRICSDecompose(mol, returnMols=False))
            if len(frags) >= 2:
                return ".".join(frags[:2]) + ">>" + canon
        except:
            pass

    return None


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def make_boronic_acid(ar_smi):
    """将芳基 SMILES 转为硼酸形式"""
    ar_mol = Chem.MolFromSmiles(ar_smi)
    if not ar_mol:
        return None

    # 如果已经含有 B(O)(O)，直接返回
    if "[B]" in ar_smi or "B(O)" in ar_smi:
        return ar_smi

    # 对于简单芳环，添加 B(O)(O)
    # 先检查是否是合法芳基
    ar_canon = Chem.MolToSmiles(ar_mol)

    # 特殊处理已知片段
    boronic_map = {
        "c1ccc2ncncc2c1": "OB(O)c1ccc2ncncc2c1",           # isoquinoline-3-boronic acid
        "c1ccc2ccncc2c1": "OB(O)c1ccc2ccncc2c1",           # quinoline-6-boronic acid
        "c1ccc2ccccc2c1": "OB(O)c1ccc2ccccc2c1",           # naphthalene-2-boronic acid
        "c1ccc2ncncc2c1": "OB(O)c1ccc2ncncc2c1",           # quinazoline boronic acid
        "c1ccccn1": "OB(O)c1ccccn1",                        # pyridine-3-boronic acid
        "c1ccncc1": "OB(O)c1ccncc1",                        # pyridine-4-boronic acid
        "c1cccnc1": "OB(O)c1cccnc1",                        # pyridine-2-boronic acid
        "c1ccco1": "OB(O)c1ccco1",                          # furan-2-boronic acid
        "c1cccs1": "OB(O)c1cccs1",                          # thiophene-2-boronic acid
        "c1ccccc1": "OB(O)c1ccccc1",                        # phenylboronic acid
        "c1ccccc1F": "OB(O)c1ccccc1F",                      # 2-fluorophenylboronic acid
        "c1ccc(F)cc1": "OB(O)c1ccc(F)cc1",                  # 4-fluorophenylboronic acid
        "c1ccc(Cl)cc1": "OB(O)c1ccc(Cl)cc1",
        "c1ccc(C)cc1": "OB(O)c1ccc(C)cc1",
        "c1ccc(OC)cc1": "OB(O)c1ccc(OC)cc1",
        "c1ccc(C#N)cc1": "OB(O)c1ccc(C#N)cc1",
        "c1ccc(C(F)(F)F)cc1": "OB(O)c1ccc(C(F)(F)F)cc1",
        "c1ccc(O)cc1": "OB(O)c1ccc(O)cc1",
        "c1ccc(S)cc1": "OB(O)c1ccc(S)cc1",
        "c1ccc(N)cc1": "OB(O)c1ccc(N)cc1",
        "c1ccc2[nH]ncc2c1": "OB(O)c1ccc2[nH]ncc2c1",      # indazole-5-boronic acid
        "c1ccc2[nH]ccc2c1": "OB(O)c1ccc2[nH]ccc2c1",      # indole-5-boronic acid
        "c1ccc2ccncc2c1": "OB(O)c1ccc2ccncc2c1",           # isoquinoline boronic acid
        "c1ccc2ncncc2c1": "OB(O)c1ccc2ncncc2c1",           # quinazoline boronic acid
        "c1ccc2ccccc2n1": "OB(O)c1ccc2ccccc2n1",           # quinoline boronic acid
    }

    if ar_canon in boronic_map:
        return boronic_map[ar_canon]

    # 通用：在第一个原子上添加 B(O)(O)
    # 对于苯环类，尝试 OB(O)c1ccccc1 模式
    if ar_canon.startswith("c1"):
        return f"OB(O){ar_canon}"

    return None


# ═══════════════════════════════════════════════════════════════════
# 四、路线校验
# ═══════════════════════════════════════════════════════════════════

def validate_route(route, mol_smiles):
    """校验路线合法性"""
    result = {
        "mol_smiles": mol_smiles,
        "canonical_mol": Chem.MolToSmiles(Chem.MolFromSmiles(mol_smiles)) if Chem.MolFromSmiles(mol_smiles) else "",
        "route": route,
        "n_steps": 0,
        "has_dummy_atom": False,
        "final_product_match": False,
        "rdkit_valid_all": True,
        "element_balance_ok": True,
        "has_A_to_A": False,
        "estimated_starting_material_score": 0.0,
        "estimated_route_risk": "low",
        "keep_or_replace": "keep",
        "notes": "",
    }

    if not route or ">>" not in route:
        result["notes"] = "无有效路线"
        result["keep_or_replace"] = "replace"
        return result

    # 检查 dummy atom
    dummy_patterns = ["[*]", "[5*]", "[16*]", "[14*]", "[1*]", "[3*]", "[6*]", "[2*]", "[4*]"]
    for dp in dummy_patterns:
        if dp in route:
            result["has_dummy_atom"] = True
            result["notes"] += f"含dummy atom: {dp}; "
            result["keep_or_replace"] = "replace"
            break

    # 分割步骤
    steps = [s.strip() for s in route.split(",") if s.strip()]
    result["n_steps"] = len(steps)

    # 检查每一步
    last_product = None
    for i, step in enumerate(steps):
        if ">>" not in step:
            result["rdkit_valid_all"] = False
            result["notes"] += f"Step {i+1} 格式错误; "
            continue

        parts = step.split(">>")
        if len(parts) != 2:
            result["rdkit_valid_all"] = False
            continue

        reactants_smi, product_smi = parts
        product_mol = Chem.MolFromSmiles(product_smi)
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smi.split(".") if r.strip()]

        # RDKit 解析检查
        if not product_mol:
            result["rdkit_valid_all"] = False
            result["notes"] += f"Step {i+1} 产物无法解析; "
        for j, rm in enumerate(reactant_mols):
            if not rm:
                result["rdkit_valid_all"] = False
                result["notes"] += f"Step {i+1} 反应物{j+1}无法解析; "

        # A >> A 检查
        if product_mol and any(Chem.MolToSmiles(rm) == Chem.MolToSmiles(product_mol) for rm in reactant_mols if rm):
            result["has_A_to_A"] = True
            result["notes"] += f"Step {i+1} 有 A>>A; "
            result["keep_or_replace"] = "replace"

        # 元素平衡检查
        if product_mol and all(rm for rm in reactant_mols):
            reactant_atoms = {}
            for rm in reactant_mols:
                for a in rm.GetAtoms():
                    sym = a.GetSymbol()
                    reactant_atoms[sym] = reactant_atoms.get(sym, 0) + 1
            product_atoms = {}
            for a in product_mol.GetAtoms():
                sym = a.GetSymbol()
                product_atoms[sym] = product_atoms.get(sym, 0) + 1

            for sym, count in product_atoms.items():
                if reactant_atoms.get(sym, 0) < count:
                    result["element_balance_ok"] = False
                    result["notes"] += f"Step {i+1} 元素{sym}不平衡; "
                    break

        last_product = product_smi

    # 最终产物匹配
    if last_product:
        last_mol = Chem.MolFromSmiles(last_product)
        target_mol = Chem.MolFromSmiles(mol_smiles)
        if last_mol and target_mol:
            if Chem.MolToSmiles(last_mol) == Chem.MolToSmiles(target_mol):
                result["final_product_match"] = True
            else:
                result["notes"] += "最终产物不匹配; "
                result["keep_or_replace"] = "replace"

    # 起始原料评分（启发式）
    score = 1.0
    for step in steps:
        parts = step.split(">>")
        if len(parts) != 2:
            continue
        for r in parts[0].split("."):
            r = r.strip()
            r_mol = Chem.MolFromSmiles(r)
            if not r_mol:
                continue
            r_canon = Chem.MolToSmiles(r_mol)
            # 常见商业可得起始原料
            if is_common_starting_material(r_canon):
                score = min(score, 1.0)
            elif is_boronic_acid(r_canon):
                score = min(score, 0.9)
            elif is_acyl_chloride(r_canon):
                score = min(score, 0.85)
            elif len(r_canon) < 15:
                score = min(score, 0.8)
            else:
                score = min(score, 0.6)

    result["estimated_starting_material_score"] = round(score, 2)

    # 风险评估
    if not result["element_balance_ok"] or not result["final_product_match"]:
        result["estimated_route_risk"] = "high"
    elif result["has_dummy_atom"] or result["has_A_to_A"]:
        result["estimated_route_risk"] = "high"
    elif result["n_steps"] > 3:
        result["estimated_route_risk"] = "medium"
    else:
        result["estimated_route_risk"] = "low"

    if result["keep_or_replace"] != "replace":
        if result["element_balance_ok"] and result["final_product_match"] and not result["has_dummy_atom"]:
            result["keep_or_replace"] = "keep"
        else:
            result["keep_or_replace"] = "replace"

    return result


def is_common_starting_material(smi):
    """判断是否为常见商业可得起始原料"""
    common = [
        "Nc1ccccc1",           # aniline
        "Nc1ccc(F)cc1",        # 4-fluoroaniline
        "Nc1ccc(Cl)cc1",       # 4-chloroaniline
        "Nc1ccc(C)cc1",        # 4-toluidine
        "Nc1ccc(OC)cc1",       # 4-methoxyaniline
        "Nc1ccc(C#N)cc1",      # 4-cyanoaniline
        "Nc1ccc(C(F)(F)F)cc1", # 4-(trifluoromethyl)aniline
        "Nc1ccc(O)cc1",        # 4-aminophenol
        "Nc1ccc(S)cc1",        # 4-aminothiophenol
        "Nc1ccc(N)cc1",        # 1,4-phenylenediamine
        "Nc1ccccc1F",          # 2-fluoroaniline
        "Nc1ccccn1",           # 2-aminopyridine
        "Nc1cccnc1",           # 3-aminopyridine
        "Nc1ccncc1",           # 4-aminopyridine
        "c1ccccc1",            # benzene
        "c1ccc(F)cc1",         # fluorobenzene
        "Brc1ccc(C(=O)Cl)cc1", # 4-bromobenzoyl chloride
        "Brc1ccc(C(=O)O)cc1",  # 4-bromobenzoic acid
        "ClC(=O)c1ccccc1",     # benzoyl chloride
        "O=C=Nc1ccccc1",       # phenyl isocyanate
        "OB(O)c1ccccc1",       # phenylboronic acid
        "OB(O)c1ccc(F)cc1",    # 4-fluorophenylboronic acid
        "OB(O)c1ccc(C)cc1",    # 4-methylphenylboronic acid
        "OB(O)c1ccc(OC)cc1",   # 4-methoxyphenylboronic acid
        "OB(O)c1ccccn1",       # pyridine-3-boronic acid
        "OB(O)c1ccncc1",       # pyridine-4-boronic acid
        "OB(O)c1ccc(C#N)cc1",  # 4-cyanophenylboronic acid
        "OB(O)c1ccc(C(F)(F)F)cc1",
        "OB(O)c1ccc(O)cc1",
        "OB(O)c1ccc(S)cc1",
        "OB(O)c1ccc2ncncc2c1",
        "OB(O)c1ccc2ccncc2c1",
        "OB(O)c1ccc2ccccc2c1",
    ]
    return smi in common


def is_boronic_acid(smi):
    return "B(O)" in smi or "OB(O)" in smi


def is_acyl_chloride(smi):
    return "C(=O)Cl" in smi


# ═══════════════════════════════════════════════════════════════════
# 五、主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("AI4S Route Fix V3 — 合成路线修复")
    log.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # 1. 读取输入
    molecules = read_input(INPUT_CSV)
    log.info(f"\n共 {len(molecules)} 个唯一分子需要处理")

    # 2. 修复路线
    fixed = []
    validation_rows = []

    for mol_info in molecules:
        smi = mol_info["mol_smiles"]
        canon = mol_info["canonical"]
        old_route = mol_info["route"]

        log.info(f"\n{'─'*50}")
        log.info(f"分子: {canon[:60]}")

        # 分析结构
        info = analyze_molecule(smi)
        if info:
            log.info(f"  模式: {info['pattern']}, 策略: {info['strategy']}")
        else:
            log.info(f"  ⚠️ 无法分析结构")

        # 生成新路线
        new_route = None

        if info and info["pattern"] == "biphenyl_amide":
            new_route = make_route_biphenyl_amide(smi, canon)
        elif info and info["pattern"] == "substituted_aniline_biphenyl":
            new_route = make_route_substituted_aniline(smi, canon)
        elif info and info["pattern"] == "urea":
            new_route = make_route_urea(smi, canon)
        elif info and info["pattern"] == "quinazoline_amide":
            new_route = make_route_quinazoline_amide(smi, canon)
        elif info and info["pattern"] == "cf3_quinazoline":
            new_route = make_route_cf3_quinazoline(smi, canon)
        elif info and info["pattern"] == "methoxy_aniline_amide":
            new_route = make_route_methoxy_aniline_amide(smi, canon)
        elif info and info["pattern"] in ("heterocycle_amide", "pyridine_amide", "pyridyl_biphenyl"):
            new_route = make_route_heterocycle_amide(smi, canon)

        if not new_route:
            new_route = make_route_generic(smi, canon)

        if not new_route:
            log.info(f"  ❌ 无法生成路线，保留原路线")
            new_route = old_route

        # 校验
        val = validate_route(new_route, smi)
        log.info(f"  步数: {val['n_steps']}, dummy: {val['has_dummy_atom']}, "
                 f"产物匹配: {val['final_product_match']}, 元素平衡: {val['element_balance_ok']}")
        log.info(f"  起始原料评分: {val['estimated_starting_material_score']}, "
                 f"风险: {val['estimated_route_risk']}, 决定: {val['keep_or_replace']}")

        if val["notes"]:
            log.info(f"  备注: {val['notes']}")

        fixed.append({"mol_smiles": smi, "route": new_route})
        validation_rows.append(val)

    # 3. 输出 result.csv
    result_csv = OUT_DIR / "result.csv"
    with open(result_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in fixed:
            writer.writerow([r["mol_smiles"], r["route"]])
    log.info(f"\n✅ result.csv: {result_csv} ({len(fixed)} 行)")

    # 4. 输出 route_validation.csv
    val_csv = OUT_DIR / "route_validation.csv"
    with open(val_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mol_smiles", "canonical_mol", "route", "n_steps",
            "has_dummy_atom", "final_product_match", "rdkit_valid_all",
            "element_balance_ok", "has_A_to_A",
            "estimated_starting_material_score", "estimated_route_risk",
            "keep_or_replace", "notes"
        ])
        writer.writeheader()
        for v in validation_rows:
            writer.writerow(v)
    log.info(f"✅ route_validation.csv: {val_csv}")

    # 5. 输出 route_fix_report.md
    report_md = OUT_DIR / "route_fix_report.md"
    with open(report_md, "w") as f:
        f.write("# Route Fix Report V3\n\n")
        f.write(f"**时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**输入:** {INPUT_CSV}\n\n")
        f.write(f"## 概览\n\n")
        f.write(f"- 处理分子数: {len(fixed)}\n")

        # 统计
        n_dummy = sum(1 for v in validation_rows if v["has_dummy_atom"])
        n_match = sum(1 for v in validation_rows if v["final_product_match"])
        n_balance = sum(1 for v in validation_rows if v["element_balance_ok"])
        n_low_risk = sum(1 for v in validation_rows if v["estimated_route_risk"] == "low")
        avg_sm = sum(v["estimated_starting_material_score"] for v in validation_rows) / len(validation_rows) if validation_rows else 0

        f.write(f"- 无 dummy atom: {len(fixed) - n_dummy}/{len(fixed)}\n")
        f.write(f"- 产物匹配: {n_match}/{len(fixed)}\n")
        f.write(f"- 元素平衡: {n_balance}/{len(fixed)}\n")
        f.write(f"- 低风险路线: {n_low_risk}/{len(fixed)}\n")
        f.write(f"- 平均起始原料评分: {avg_sm:.2f}\n\n")

        f.write("## 各分子路线\n\n")
        for i, (mol, val) in enumerate(zip(fixed, validation_rows), 1):
            f.write(f"### {i}. {mol['mol_smiles'][:50]}\n\n")
            f.write(f"- **SMILES:** `{mol['mol_smiles']}`\n")
            f.write(f"- **Route:** `{mol['route']}`\n")
            f.write(f"- **步数:** {val['n_steps']}\n")
            f.write(f"- **产物匹配:** {'✅' if val['final_product_match'] else '❌'}\n")
            f.write(f"- **元素平衡:** {'✅' if val['element_balance_ok'] else '❌'}\n")
            f.write(f"- **Dummy atom:** {'❌' if val['has_dummy_atom'] else '✅ 无'}\n")
            f.write(f"- **起始原料评分:** {val['estimated_starting_material_score']}\n")
            f.write(f"- **风险:** {val['estimated_route_risk']}\n")
            if val["notes"]:
                f.write(f"- **备注:** {val['notes']}\n")
            f.write("\n")

    log.info(f"✅ route_fix_report.md: {report_md}")

    # 6. 打包 result.zip
    zip_path = OUT_DIR / "result.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(result_csv, "result.csv")
        zf.write(LOG_FILE, "result.log")
    log.info(f"✅ result.zip: {zip_path}")

    # 7. 最终统计
    log.info(f"\n{'='*60}")
    log.info(f"修复完成统计:")
    log.info(f"  总分子数: {len(fixed)}")
    log.info(f"  无 dummy atom: {len(fixed) - n_dummy}/{len(fixed)}")
    log.info(f"  产物匹配: {n_match}/{len(fixed)}")
    log.info(f"  元素平衡: {n_balance}/{len(fixed)}")
    log.info(f"  低风险路线: {n_low_risk}/{len(fixed)}")
    log.info(f"  平均起始原料评分: {avg_sm:.2f}")
    log.info(f"{'='*60}")
    log.info("Route Fix V3 完成 ✅")


if __name__ == "__main__":
    main()
