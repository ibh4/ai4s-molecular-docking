#!/usr/bin/env python3
"""
Route Fix V3 — 手动修复剩余 12 个分子的路线
"""
import csv, os, zipfile, logging
from datetime import datetime
from pathlib import Path
from rdkit import Chem

OUT_DIR = Path("/Users/pwngwc/ai4s_chem/result/route_fix_v3")
LOG_FILE = OUT_DIR / "result.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("manual_fix")

# ── 手动定义的路线 ────────────────────────────────────────────────

MANUAL_ROUTES = {
    # 1. 3-F-biphenyl amide
    "O=C(Nc1ccccc1)c1ccc(-c2cccc(F)c2)cc1":
        "Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccc(Br)cc1,"
        "O=C(Nc1ccccc1)c1ccc(Br)cc1.OB(O)c1cccc(F)c1>>O=C(Nc1ccccc1)c1ccc(-c2cccc(F)c2)cc1",

    # 2. reverse amide pyridyl (O=C(Nc1ccc(-c2ccccn2)cc1)c1ccccc1)
    # 其实是 O=C(c1ccccc1)Nc1ccc(-c2ccccn2)cc1 的 canonical 形式
    # 用酰胺偶联: BzCl + 4-(3-pyridyl)aniline
    "O=C(Nc1ccc(-c2ccccn2)cc1)c1ccccc1":
        "ClC(=O)c1ccccc1.Nc1ccc(-c2ccccn2)cc1>>O=C(Nc1ccc(-c2ccccn2)cc1)c1ccccc1",

    # 3. 2-F-biphenyl amide
    "O=C(Nc1ccccc1)c1ccc(-c2ccccc2F)cc1":
        "Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccc(Br)cc1,"
        "O=C(Nc1ccccc1)c1ccc(Br)cc1.OB(O)c1ccccc1F>>O=C(Nc1ccccc1)c1ccc(-c2ccccc2F)cc1",

    # 4. CF3-biphenyl amide
    "O=C(Nc1ccccc1)c1ccc(-c2ccc(C(F)(F)F)cc2)cc1":
        "Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccc(Br)cc1,"
        "O=C(Nc1ccccc1)c1ccc(Br)cc1.OB(O)c1ccc(C(F)(F)F)cc1>>O=C(Nc1ccccc1)c1ccc(-c2ccc(C(F)(F)F)cc2)cc1",

    # 5. 4-F-biphenyl amide
    "O=C(Nc1ccccc1)c1ccc(-c2ccc(F)cc2)cc1":
        "Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccc(Br)cc1,"
        "O=C(Nc1ccccc1)c1ccc(Br)cc1.OB(O)c1ccc(F)cc1>>O=C(Nc1ccccc1)c1ccc(-c2ccc(F)cc2)cc1",

    # 6. OH-biphenyl amide
    "O=C(Nc1ccccc1)c1ccc(-c2ccc(O)cc2)cc1":
        "Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccc(Br)cc1,"
        "O=C(Nc1ccccc1)c1ccc(Br)cc1.OB(O)c1ccc(O)cc1>>O=C(Nc1ccccc1)c1ccc(-c2ccc(O)cc2)cc1",

    # 7. Me-biphenyl amide
    "Cc1ccc(-c2ccc(C(=O)Nc3ccccc3)cc2)cc1":
        "Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccc(Br)cc1,"
        "O=C(Nc1ccccc1)c1ccc(Br)cc1.OB(O)c1ccc(C)cc1>>Cc1ccc(-c2ccc(C(=O)Nc3ccccc3)cc2)cc1",

    # 8. Pyridine 4-F-biphenyl amide
    "O=C(Nc1ccccc1)c1ccc(-c2ccc(F)cc2)nc1":
        "Brc1ccc(C(=O)Cl)nc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccc(Br)nc1,"
        "O=C(Nc1ccccc1)c1ccc(Br)nc1.OB(O)c1ccc(F)cc1>>O=C(Nc1ccccc1)c1ccc(-c2ccc(F)cc2)nc1",

    # 9. CN-aniline quinoline amide
    "N#Cc1ccc(NC(=O)c2ccnc3ccccc23)cc1":
        "ClC(=O)c1ccnc2ccccc12.N#Cc1ccc(N)cc1>>N#Cc1ccc(NC(=O)c2ccnc3ccccc23)cc1",

    # 10. Indazole-CF3 amide
    "O=C(Nc1ccc(C(F)(F)F)cc1)c1ccc2[nH]ncc2c1":
        "ClC(=O)c1ccc2[nH]ncc2c1.Nc1ccc(C(F)(F)F)cc1>>O=C(Nc1ccc(C(F)(F)F)cc1)c1ccc2[nH]ncc2c1",

    # 11. Pyridyl biphenyl amide
    "O=C(Nc1ccccc1)c1ccc(-c2ccccc2)nc1":
        "Brc1ccc(C(=O)Cl)nc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccc(Br)nc1,"
        "O=C(Nc1ccccc1)c1ccc(Br)nc1.OB(O)c1ccccc1>>O=C(Nc1ccccc1)c1ccc(-c2ccccc2)nc1",

    # 12. SH-biphenyl amide
    "O=C(Nc1ccccc1)c1ccc(-c2ccc(S)cc2)cc1":
        "Brc1ccc(C(=O)Cl)cc1.Nc1ccccc1>>O=C(Nc1ccccc1)c1ccc(Br)cc1,"
        "O=C(Nc1ccccc1)c1ccc(Br)cc1.OB(O)c1ccc(S)cc1>>O=C(Nc1ccccc1)c1ccc(-c2ccc(S)cc2)cc1",
}


def validate_route(route, mol_smiles):
    """校验路线"""
    result = {
        "n_steps": 0,
        "has_dummy_atom": False,
        "final_product_match": False,
        "element_balance_ok": True,
        "has_A_to_A": False,
        "notes": "",
    }

    if not route or ">>" not in route:
        return result

    # dummy atom
    for dp in ["[*]", "[5*]", "[16*]", "[14*]", "[1*]", "[3*]", "[6*]"]:
        if dp in route:
            result["has_dummy_atom"] = True
            break

    steps = [s.strip() for s in route.split(",") if s.strip()]
    result["n_steps"] = len(steps)

    last_product = None
    for step in steps:
        if ">>" not in step:
            continue
        parts = step.split(">>")
        reactants_smi, product_smi = parts
        product_mol = Chem.MolFromSmiles(product_smi)
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smi.split(".") if r.strip()]

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
                    result["notes"] += f"元素{sym}不平衡; "
                    break

        # A >> A
        if product_mol and any(Chem.MolToSmiles(rm) == Chem.MolToSmiles(product_mol) for rm in reactant_mols if rm):
            result["has_A_to_A"] = True

        last_product = product_smi

    if last_product:
        last_mol = Chem.MolFromSmiles(last_product)
        target_mol = Chem.MolFromSmiles(mol_smiles)
        if last_mol and target_mol:
            if Chem.MolToSmiles(last_mol) == Chem.MolToSmiles(target_mol):
                result["final_product_match"] = True

    return result


def main():
    log.info("\n" + "=" * 60)
    log.info("Route Fix V3 — 手动修复剩余分子")
    log.info("=" * 60)

    # 读取当前 result.csv
    csv_path = OUT_DIR / "result.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # 修复
    fixed_count = 0
    for row in rows:
        smi = row["mol_smiles"].strip()
        old_route = row["route"].strip()

        if smi in MANUAL_ROUTES:
            new_route = MANUAL_ROUTES[smi]
            val = validate_route(new_route, smi)

            if val["has_dummy_atom"]:
                log.info(f"  ⚠️ {smi[:40]} — 新路线仍有dummy atom!")
                continue

            row["route"] = new_route
            fixed_count += 1
            log.info(f"  ✅ {smi[:50]}")
            log.info(f"     产物匹配: {val['final_product_match']}, 元素平衡: {val['element_balance_ok']}, 步数: {val['n_steps']}")

    # 重写 result.csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mol_smiles", "route"])
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"\n修复了 {fixed_count} 个分子的路线")

    # 重新生成 validation
    val_csv = OUT_DIR / "route_validation.csv"
    all_validations = []
    for row in rows:
        val = validate_route(row["route"], row["mol_smiles"])
        val["mol_smiles"] = row["mol_smiles"]
        val["canonical_mol"] = Chem.MolToSmiles(Chem.MolFromSmiles(row["mol_smiles"])) if Chem.MolFromSmiles(row["mol_smiles"]) else ""
        val["route"] = row["route"]
        val["estimated_starting_material_score"] = 1.0 if not val["has_dummy_atom"] else 0.6
        val["estimated_route_risk"] = "low" if (not val["has_dummy_atom"] and val["element_balance_ok"] and val["final_product_match"]) else "high"
        val["keep_or_replace"] = "keep" if val["estimated_route_risk"] == "low" else "replace"
        all_validations.append(val)

    with open(val_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mol_smiles", "canonical_mol", "route", "n_steps",
            "has_dummy_atom", "final_product_match", "rdkit_valid_all",
            "element_balance_ok", "has_A_to_A",
            "estimated_starting_material_score", "estimated_route_risk",
            "keep_or_replace", "notes"
        ])
        writer.writeheader()
        for v in all_validations:
            writer.writerow(v)

    # 统计
    n_no_dummy = sum(1 for v in all_validations if not v["has_dummy_atom"])
    n_match = sum(1 for v in all_validations if v["final_product_match"])
    n_balance = sum(1 for v in all_validations if v["element_balance_ok"])
    n_low = sum(1 for v in all_validations if v["estimated_route_risk"] == "low")

    log.info(f"\n最终统计:")
    log.info(f"  总分子数: {len(all_validations)}")
    log.info(f"  无 dummy atom: {n_no_dummy}/{len(all_validations)}")
    log.info(f"  产物匹配: {n_match}/{len(all_validations)}")
    log.info(f"  元素平衡: {n_balance}/{len(all_validations)}")
    log.info(f"  低风险: {n_low}/{len(all_validations)}")

    # 更新 report
    report_path = OUT_DIR / "route_fix_report.md"
    with open(report_path, "a") as f:
        f.write("\n## 手动修复补充\n\n")
        f.write(f"修复了 {fixed_count} 个分子的路线。\n\n")
        f.write(f"- 无 dummy atom: {n_no_dummy}/{len(all_validations)}\n")
        f.write(f"- 产物匹配: {n_match}/{len(all_validations)}\n")
        f.write(f"- 元素平衡: {n_balance}/{len(all_validations)}\n")
        f.write(f"- 低风险路线: {n_low}/{len(all_validations)}\n\n")

        f.write("### 手动修复的分子\n\n")
        for smi, route in MANUAL_ROUTES.items():
            f.write(f"- `{smi[:50]}`\n")
            f.write(f"  Route: `{route[:80]}...`\n\n")

    # 重新打包 zip
    zip_path = OUT_DIR / "result.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, "result.csv")
        zf.write(LOG_FILE, "result.log")
    log.info(f"\n✅ result.zip 已更新: {zip_path}")
    log.info("手动修复完成 ✅")


if __name__ == "__main__":
    main()
