#!/usr/bin/env python3
"""
AI4S 最终提交前严审
检查 route 合法性、位置异构风险、起始原料可获得性
"""
import csv, os, zipfile, logging, re
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path("/Users/pwngwc/.openclaw/workspace/retrosyn")))
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors

BASE_DIR = Path("/Users/pwngwc/ai4s_chem")
INPUT_CSV = BASE_DIR / "result" / "route_fix_v3" / "result.csv"
OUT_DIR = BASE_DIR / "result" / "route_fix_v3_final"
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
log = logging.getLogger("final_check")

# ═══════════════════════════════════════════════════════════════════
# 起始原料风险评估
# ═══════════════════════════════════════════════════════════════════

# 低风险：常见商业可得
LOW_RISK_REAGENTS = {
    # 苯胺类
    "Nc1ccccc1", "Nc1ccc(F)cc1", "Nc1ccc(Cl)cc1", "Nc1ccc(C)cc1",
    "Nc1ccc(OC)cc1", "Nc1ccc(C#N)cc1", "Nc1ccc(C(F)(F)F)cc1",
    "Nc1ccc(O)cc1", "Nc1ccc(S)cc1", "Nc1ccc(N)cc1", "Nc1ccccc1F",
    "Nc1ccccn1", "Nc1cccnc1", "Nc1ccncc1",
    # 酰氯
    "ClC(=O)c1ccccc1", "Brc1ccc(C(=O)Cl)cc1",
    # 硼酸
    "OB(O)c1ccccc1", "OB(O)c1ccc(F)cc1", "OB(O)c1ccc(Cl)cc1",
    "OB(O)c1ccc(C)cc1", "OB(O)c1ccc(OC)cc1", "OB(O)c1ccc(C#N)cc1",
    "OB(O)c1ccc(C(F)(F)F)cc1", "OB(O)c1ccc(O)cc1", "OB(O)c1ccc(S)cc1",
    "OB(O)c1ccc(N)cc1", "OB(O)c1ccccc1F",
    # 吡啶硼酸
    "OB(O)c1ccccn1", "OB(O)c1ccncc1", "OB(O)c1cccnc1",
    # 异氰酸酯
    "O=C=Nc1ccccc1",
    # 吡啶卤代物
    "Brc1ccccn1", "Brc1ccncc1", "Brc1cccnc1",
    # 呋喃/噻吩硼酸
    "OB(O)c1ccco1", "OB(O)c1cccs1",
}

# 中风险：可获得但不常见
MEDIUM_RISK_REAGENTS = {
    # 萘硼酸
    "OB(O)c1ccc2ccccc2c1",
    # 喹啉/异喹啉硼酸
    "OB(O)c1ccc2ccncc2c1", "OB(O)c1ccc2ncncc2c1",
    # 吲唑硼酸
    "OB(O)c1ccc2[nH]ncc2c1",
    # 喹啉酰氯
    "ClC(=O)c1ccnc2ccccc12",
    # 杂环酰氯
    "ClC(=O)c1nnc(-c2ccccc2)o1", "ClC(=O)c1cnc(-c2ccccc2)[nH]1",
    # CF3 喹啉氯
    "Clc1nccc2cc(C(F)(F)F)ccc12",
    # 甲氧基苯胺
    "COc1ccc(N)cc1",
    # 氰基苯胺
    "N#Cc1ccc(N)cc1",
    # 吲唑酰氯
    "ClC(=O)c1ccc2[nH]ncc2c1",
}

# 高风险：不稳定或罕见
HIGH_RISK_REAGENTS = {
    # 含游离巯基硼酸
    "OB(O)c1ccc(S)cc1",
}


def classify_reagent(smi):
    """分类起始原料风险"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return "invalid", "无法解析"

    canon = Chem.MolToSmiles(mol)

    if canon in LOW_RISK_REAGENTS:
        return "low", "常见商业可得"
    if canon in MEDIUM_RISK_REAGENTS:
        return "medium", "可获得但不常见"
    if canon in HIGH_RISK_REAGENTS:
        return "high", "不稳定/罕见"

    # 启发式判断
    n_atoms = mol.GetNumHeavyAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)

    if n_atoms > 20:
        return "high", f"分子过大({n_atoms}原子)"
    if n_rings > 3:
        return "high", f"环数过多({n_rings})"

    return "medium", "未分类"


def check_isomer_risk(route, mol_smiles):
    """检查 Suzuki 偶联中位置异构风险"""
    risk = "low"
    notes = ""

    # 检查含氮杂环硼酸的位置异构
    pyridyl_boronic = re.findall(r'OB\(O\)c1(cc[ncccn]|ccncc|cccnc|ccccn)', route)
    quinoline_boronic = re.findall(r'OB\(O\)c1(c2ccncc2|c2ncncc2|c2ccccc2)', route)

    # 对于吡啶硼酸，不同位置异构体价格差异大
    if "OB(O)c1ccncc1" in route:
        # 4-吡啶硼酸（常见）
        pass
    elif "OB(O)c1ccccn1" in route:
        # 3-吡啶硼酸（常见）
        pass
    elif "OB(O)c1cccnc1" in route:
        # 2-吡啶硼酸（不太稳定）
        risk = "medium"
        notes = "2-吡啶硼酸稳定性较差"

    # 喹啉/异喹啉硼酸
    if "OB(O)c1ccc2ccncc2c1" in route:
        risk = "medium"
        notes = "异喹啉硼酸需确认位置异构"
    if "OB(O)c1ccc2ncncc2c1" in route:
        risk = "medium"
        notes = "喹唑啉硼酸需确认位置异构"

    # 萘硼酸
    if "OB(O)c1ccc2ccccc2c1" in route:
        risk = "medium"
        notes = "萘硼酸需确认是1-萘还是2-萘"

    return risk, notes


# ═══════════════════════════════════════════════════════════════════
# 路线校验
# ═══════════════════════════════════════════════════════════════════

def validate_route(route, mol_smiles):
    """完整校验路线"""
    result = {
        "mol_smiles": mol_smiles,
        "route": route,
        "n_steps": 0,
        "final_match": False,
        "no_dummy": True,
        "element_balance_ok": True,
        "no_A_to_A": True,
        "reagent_risk": "low",
        "isomer_risk": "low",
        "submit_recommendation": "submit",
        "notes": "",
    }

    if not route or ">>" not in route:
        result["submit_recommendation"] = "skip"
        result["notes"] = "无有效路线"
        return result

    # 1. Dummy atom 检查
    dummy_patterns = ["[*]", "[5*]", "[16*]", "[14*]", "[1*]", "[3*]", "[6*]", "[2*]", "[4*]"]
    for dp in dummy_patterns:
        if dp in route:
            result["no_dummy"] = False
            result["notes"] += f"含dummy: {dp}; "
            result["submit_recommendation"] = "skip"
            break

    # 2. 分割步骤
    steps = [s.strip() for s in route.split(",") if s.strip()]
    result["n_steps"] = len(steps)

    # 3. 每步校验
    last_product = None
    all_reagents = []

    for i, step in enumerate(steps):
        if ">>" not in step:
            result["notes"] += f"Step{i+1}格式错误; "
            result["submit_recommendation"] = "skip"
            continue

        parts = step.split(">>")
        if len(parts) != 2:
            continue

        reactants_smi, product_smi = parts
        product_mol = Chem.MolFromSmiles(product_smi)
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smi.split(".") if r.strip()]

        # RDKit 解析
        if not product_mol:
            result["notes"] += f"Step{i+1}产物无法解析; "
            result["submit_recommendation"] = "skip"
        for rm in reactant_mols:
            if not rm:
                result["notes"] += f"Step{i+1}反应物无法解析; "
                result["submit_recommendation"] = "skip"

        # A >> A 检查
        if product_mol:
            for rm in reactant_mols:
                if rm and Chem.MolToSmiles(rm) == Chem.MolToSmiles(product_mol):
                    result["no_A_to_A"] = False
                    result["notes"] += f"Step{i+1} A>>A; "
                    result["submit_recommendation"] = "skip"

        # 元素守恒
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
                    result["notes"] += f"Step{i+1} {sym}不平衡(需{count},有{reactant_atoms.get(sym,0)}); "
                    result["submit_recommendation"] = "skip"

        # 收集起始原料
        for rm_smi in reactants_smi.split("."):
            rm_smi = rm_smi.strip()
            rm_mol = Chem.MolFromSmiles(rm_smi)
            if rm_mol:
                # 判断是否是中间体（前一步产物）
                is_intermediate = False
                for prev_step in steps[:i]:
                    prev_product = prev_step.split(">>")[1].strip()
                    if Chem.MolToSmiles(Chem.MolFromSmiles(prev_product)) == Chem.MolToSmiles(rm_mol):
                        is_intermediate = True
                        break
                if not is_intermediate:
                    all_reagents.append(rm_smi)

        last_product = product_smi

    # 4. 最终产物匹配
    if last_product:
        last_mol = Chem.MolFromSmiles(last_product)
        target_mol = Chem.MolFromSmiles(mol_smiles)
        if last_mol and target_mol:
            if Chem.MolToSmiles(last_mol) == Chem.MolToSmiles(target_mol):
                result["final_match"] = True
            else:
                result["notes"] += f"产物不匹配: {Chem.MolToSmiles(last_mol)} != {Chem.MolToSmiles(target_mol)}; "
                result["submit_recommendation"] = "skip"

    # 5. 起始原料风险评估
    max_risk = "low"
    risk_notes = []
    for r in all_reagents:
        r_mol = Chem.MolFromSmiles(r)
        if r_mol:
            r_canon = Chem.MolToSmiles(r_mol)
            risk, note = classify_reagent(r_canon)
            if risk == "high":
                max_risk = "high"
                risk_notes.append(f"{r_canon[:20]}: {note}")
            elif risk == "medium" and max_risk != "high":
                max_risk = "medium"
                risk_notes.append(f"{r_canon[:20]}: {note}")

    result["reagent_risk"] = max_risk
    if risk_notes:
        result["notes"] += "原料风险: " + "; ".join(risk_notes) + "; "

    # 6. 位置异构风险
    isomer_risk, isomer_note = check_isomer_risk(route, mol_smiles)
    result["isomer_risk"] = isomer_risk
    if isomer_note:
        result["notes"] += "异构风险: " + isomer_note + "; "

    # 7. 最终建议
    if result["submit_recommendation"] == "skip":
        pass
    elif max_risk == "high":
        result["submit_recommendation"] = "review"
        result["notes"] += "高风险原料需人工确认; "
    elif isomer_risk == "medium":
        result["submit_recommendation"] = "review"
        result["notes"] += "位置异构需确认; "

    return result


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("AI4S 最终提交前严审")
    log.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"输入: {INPUT_CSV}")
    log.info("=" * 60)

    # 读取
    rows = []
    with open(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    log.info(f"读取 {len(rows)} 行")

    # 逐条校验
    validations = []
    keep_rows = []

    for row in rows:
        smi = row["mol_smiles"].strip()
        route = row["route"].strip()

        # Canonicalize mol_smiles
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            log.info(f"  ❌ {smi[:40]} — 无法解析，删除")
            continue
        canon = Chem.MolToSmiles(mol)

        val = validate_route(route, canon)
        validations.append(val)

        log.info(f"\n{'─'*50}")
        log.info(f"分子: {canon[:55]}")
        log.info(f"  步数: {val['n_steps']}, 匹配: {val['final_match']}, "
                 f"无dummy: {val['no_dummy']}, 平衡: {val['element_balance_ok']}, "
                 f"无A>>A: {val['no_A_to_A']}")
        log.info(f"  原料风险: {val['reagent_risk']}, 异构风险: {val['isomer_risk']}, "
                 f"建议: {val['submit_recommendation']}")
        if val["notes"]:
            log.info(f"  备注: {val['notes']}")

        # 决定保留/删除
        if val["submit_recommendation"] == "skip":
            log.info(f"  ❌ 删除")
            continue

        if val["submit_recommendation"] == "review":
            log.info(f"  ⚠️ 需人工确认，保留但标记")

        # 更新 mol_smiles 为 canonical
        row["mol_smiles"] = canon
        keep_rows.append(row)

    # 输出 final_check.csv
    check_csv = OUT_DIR / "final_check.csv"
    with open(check_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mol_smiles", "route", "n_steps", "final_match", "no_dummy",
            "element_balance_ok", "no_A_to_A", "reagent_risk", "isomer_risk",
            "submit_recommendation", "notes"
        ])
        writer.writeheader()
        for v in validations:
            writer.writerow(v)
    log.info(f"\n✅ final_check.csv: {check_csv}")

    # 输出 result.csv（仅保留合法分子）
    result_csv = OUT_DIR / "result.csv"
    with open(result_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mol_smiles", "route"])
        writer.writeheader()
        for r in keep_rows:
            writer.writerow({"mol_smiles": r["mol_smiles"], "route": r["route"]})
    log.info(f"✅ result.csv: {result_csv} ({len(keep_rows)} 行)")

    # 统计
    n_submit = sum(1 for v in validations if v["submit_recommendation"] == "submit")
    n_review = sum(1 for v in validations if v["submit_recommendation"] == "review")
    n_skip = sum(1 for v in validations if v["submit_recommendation"] == "skip")

    log.info(f"\n{'='*60}")
    log.info(f"严审完成统计:")
    log.info(f"  总校验: {len(validations)}")
    log.info(f"  直接提交: {n_submit}")
    log.info(f"  需确认: {n_review}")
    log.info(f"  已删除: {n_skip}")
    log.info(f"  最终保留: {len(keep_rows)}")

    # 检查最终保留的分子是否全部通过
    all_clean = all(
        v["final_match"] and v["no_dummy"] and v["element_balance_ok"] and v["no_A_to_A"]
        for v in validations if v["submit_recommendation"] != "skip"
    )
    log.info(f"  全部通过基础检查: {'✅ 是' if all_clean else '❌ 否'}")

    # 打包
    zip_path = OUT_DIR / "result.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(result_csv, "result.csv")
        zf.write(LOG_FILE, "result.log")
    log.info(f"✅ result.zip: {zip_path}")

    log.info(f"{'='*60}")
    log.info("最终严审完成 ✅")


if __name__ == "__main__":
    main()
