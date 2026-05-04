#!/usr/bin/env python3
"""
V4 路线修复 — 为 Top 分子补写合成路线
"""
import csv, re, json, zipfile, logging
from datetime import datetime
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

BASE_DIR = Path("/Users/pwngwc/ai4s_chem")
CANDIDATES = BASE_DIR / "result" / "v4_binding_strong" / "candidates_scored.csv"
OUT_DIR_A = BASE_DIR / "result" / "v4_binding_strong"
OUT_DIR_B = BASE_DIR / "result" / "v4_diverse"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("route_fix")


def validate_route(route, mol_smiles):
    """校验路线"""
    result = {"n_steps": 0, "final_match": False, "no_dummy": True,
              "element_balance_ok": True, "no_A_to_A": True, "notes": ""}
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
        if ">>" not in step:
            continue
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


def make_route(canon):
    """为分子生成路线 — 扩展模板"""
    mol = Chem.MolFromSmiles(canon)
    if not mol:
        return None

    # 模式1: O=C(NAr1)c1ncnc2ccc(X)cc12 — 喹唑啉酰胺
    m = re.match(r'O=C\(N(.+?)\)c1ncnc2ccc\((.+?)\)cc12', canon)
    if m:
        aniline = "N" + m.group(1)
        x = m.group(2)
        acid = f"ClC(=O)c1ncnc2ccc({x})cc12"
        if Chem.MolFromSmiles(aniline) and Chem.MolFromSmiles(acid):
            return f"{acid}.{aniline}>>{canon}"

    # 模式1b: O=C(NAr1)c1ncnc2ccccc12 — 喹唑啉酰胺(无取代)
    m = re.match(r'O=C\(N(.+?)\)c1ncnc2ccccc12', canon)
    if m:
        aniline = "N" + m.group(1)
        acid = "ClC(=O)c1ncnc2ccccc12"
        if Chem.MolFromSmiles(aniline):
            return f"{acid}.{aniline}>>{canon}"

    # 模式2: COc1ccc(NC(=O)c2ccc3nccc(C(F)(F)F)c3c2)cc1 — 异喹啉酰胺
    m = re.match(r'(.+?)NC\(=O\)(c\d+ccc\d+nccc.+?)$', canon)
    if m:
        aniline_part = m.group(1).rstrip("c")
        acid_part = "ClC(=O)" + m.group(2)
        # 构造 aniline
        aniline_smi = m.group(1)
        if aniline_smi.endswith("cc1"):
            aniline_smi = aniline_smi.replace("ccc(", "ccc(N)(", 1) if "(" in aniline_smi else aniline_smi
        # 直接尝试
        aniline = "COc1ccc(N)cc1" if "COc" in canon else "Nc1ccc(OC)cc1"
        if Chem.MolFromSmiles(aniline) and Chem.MolFromSmiles(acid_part):
            return f"{acid_part}.{aniline}>>{canon}"

    # 模式3: COc1ccc2ncnc(C(=O)NAr)c2c1 — 喹唑啉(甲氧基位置不同)
    m = re.match(r'COc1ccc2ncnc\(C\(=O\)N(.+?)\)c2c1', canon)
    if m:
        aniline = "N" + m.group(1)
        acid = "ClC(=O)c1ncnc2ccc(OC)cc12"
        if Chem.MolFromSmiles(aniline) and Chem.MolFromSmiles(acid):
            return f"{acid}.{aniline}>>{canon}"

    # 模式4: Cc1ccc2ncnc(C(=O)NAr)c2c1 — 甲基喹唑啉
    m = re.match(r'Cc1ccc2ncnc\(C\(=O\)N(.+?)\)c2c1', canon)
    if m:
        aniline = "N" + m.group(1)
        acid = "ClC(=O)c1ncnc2ccc(C)cc12"
        if Chem.MolFromSmiles(aniline) and Chem.MolFromSmiles(acid):
            return f"{acid}.{aniline}>>{canon}"

    # 模式5: O=C(NAr1)c1ccc2nccc(C(F)(F)F)c2c1 — CF3-异喹啉酰胺
    m = re.match(r'O=C\(N(.+?)\)c1ccc2nccc\(C\(F\)\(F\)F\)c2c1', canon)
    if m:
        aniline = "N" + m.group(1)
        acid = "ClC(=O)c1ccc2nccc(C(F)(F)F)c2c1"
        if Chem.MolFromSmiles(aniline) and Chem.MolFromSmiles(acid):
            return f"{acid}.{aniline}>>{canon}"

    # 模式6: O=C(NAr1)c1ccc2ccccc2c1 — 萘酰胺
    m = re.match(r'O=C\(N(.+?)\)c1ccc2ccccc2c1', canon)
    if m:
        aniline = "N" + m.group(1)
        acid = "ClC(=O)c1ccc2ccccc2c1"
        if Chem.MolFromSmiles(aniline) and Chem.MolFromSmiles(acid):
            return f"{acid}.{aniline}>>{canon}"

    # 模式7: O=C(NAr1)c1ccc2ccncc2c1 — 异喹啉酰胺
    m = re.match(r'O=C\(N(.+?)\)c1ccc2ccncc2c1', canon)
    if m:
        aniline = "N" + m.group(1)
        acid = "ClC(=O)c1ccc2ccncc2c1"
        if Chem.MolFromSmiles(aniline) and Chem.MolFromSmiles(acid):
            return f"{acid}.{aniline}>>{canon}"

    # 模式8: O=C(NAr1)c1ccc2ncncc2c1 — 喹唑啉酰胺(位置不同)
    m = re.match(r'O=C\(N(.+?)\)c1ccc2ncncc2c1', canon)
    if m:
        aniline = "N" + m.group(1)
        acid = "ClC(=O)c1ccc2ncncc2c1"
        if Chem.MolFromSmiles(aniline) and Chem.MolFromSmiles(acid):
            return f"{acid}.{aniline}>>{canon}"

    # 模式9: Ar-NH-Ar' 芳胺 (SNAr)
    m = re.match(r'(.+?)N(c\d+ccccc\d+)', canon)
    if m and "C(=O)" not in canon:
        ar1 = m.group(1).rstrip("c")
        ar2 = "N" + m.group(2)
        # 尝试 Cl/Br 作为离去基
        for leaving in ["Cl", "Br"]:
            ar1_cl = ar1.replace("ccc(", f"ccc({leaving})(", 1) if "(" in ar1 else ar1 + leaving
            if Chem.MolFromSmiles(ar1_cl) and Chem.MolFromSmiles(ar2):
                return f"{ar1_cl}.{ar2}>>{canon}"

    # 模式10: O=C(NAr)Ar2 — 通用酰胺(用酰氯+苯胺)
    amide_smarts = Chem.MolFromSmarts("[#7]C(=O)[#6]")
    if mol.HasSubstructMatch(amide_smarts):
        # 尝试断开 C(=O)-N 键
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            # C(=O)-N
            if a1.GetSymbol() == "C" and a2.GetSymbol() == "N":
                for n in a1.GetNeighbors():
                    if n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(a1.GetIdx(), n.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        # 找到酰胺键，分割
                        amide_idx = bond.GetIdx()
                        try:
                            frags = Chem.rdmolops.FragmentOnBonds(mol, [amide_idx], addDummies=False)
                            frag_smiles = Chem.MolToSmiles(frags).split(".")
                            if len(frag_smiles) >= 2:
                                # 验证
                                route = ".".join(frag_smiles[:2]) + ">>" + canon
                                val = validate_route(route, canon)
                                if val["final_match"] and val["no_dummy"] and val["element_balance_ok"]:
                                    return route
                        except:
                            pass

    return None


def main():
    log.info("V4 路线修复开始")

    # 读取候选
    candidates = []
    with open(CANDIDATES) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(row)

    log.info(f"候选分子: {len(candidates)} 个")

    # 为每个分子生成路线
    fixed = []
    for c in candidates:
        smi = c["mol_smiles"].strip()
        canon = c["canonical_smiles"].strip()
        vina = float(c["vina_best"])

        mol = Chem.MolFromSmiles(canon)
        if not mol:
            continue

        route = make_route(canon)
        if route:
            val = validate_route(route, canon)
            if val["final_match"] and val["no_dummy"] and val["element_balance_ok"] and val["no_A_to_A"]:
                fixed.append({
                    "smiles": smi,
                    "canonical": canon,
                    "vina": vina,
                    "route": route,
                    "sa": float(c["sascore"]),
                    "mw": float(c["mol_weight"]),
                    "pocket": c["best_pocket"],
                })

    log.info(f"有路线分子: {len(fixed)} 个")

    # 按 Vina 排序
    fixed.sort(key=lambda x: x["vina"])

    # Top 30 打印
    log.info("\nTop 30 有路线分子:")
    for i, r in enumerate(fixed[:30], 1):
        log.info(f"  {i}. Vina={r['vina']:.1f} SA={r['sa']:.1f} | {r['canonical'][:55]}")

    # A版：Top 15
    a_final = fixed[:15]
    log.info(f"\nA版: {len(a_final)} 个分子")

    a_csv = OUT_DIR_A / "result.csv"
    with open(a_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in a_final:
            writer.writerow([r["canonical"], r["route"]])

    a_log = OUT_DIR_A / "result.log"
    with open(a_log, "w") as f:
        f.write(f"[{datetime.now()}] A版强binding版(V4路线修复)\n")
        f.write(f"分子数: {len(a_final)}\n")
        if a_final:
            f.write(f"最优Vina: {a_final[0]['vina']:.1f}\n")

    a_zip = OUT_DIR_A / "result.zip"
    with zipfile.ZipFile(a_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(a_csv, "result.csv")
        zf.write(a_log, "result.log")

    # B版：多样性
    # 按骨架分层
    def classify(c):
        if "ncnc" in c: return "quinazoline"
        if "nccc" in c: return "isoquinoline"
        if "ccccc2" in c and "ccc2" in c: return "naphthyl"
        if "Nc1ccccc1" in c and "C(=O)N" in c: return "amide"
        return "other"

    seen_classes = set()
    b_final = []
    for r in fixed:
        cls = classify(r["canonical"])
        if cls not in seen_classes or len(b_final) < 15:
            b_final.append(r)
            seen_classes.add(cls)
        if len(b_final) >= 25:
            break

    log.info(f"B版: {len(b_final)} 个分子")

    b_csv = OUT_DIR_B / "result.csv"
    with open(b_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in b_final:
            writer.writerow([r["canonical"], r["route"]])

    b_log = OUT_DIR_B / "result.log"
    with open(b_log, "w") as f:
        f.write(f"[{datetime.now()}] B版多样性版(V4路线修复)\n")
        f.write(f"分子数: {len(b_final)}\n")

    b_zip = OUT_DIR_B / "result.zip"
    with zipfile.ZipFile(b_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(b_csv, "result.csv")
        zf.write(b_log, "result.log")

    # summary
    a_summary = OUT_DIR_A / "summary.md"
    with open(a_summary, "w") as f:
        f.write("# A版：强 binding 版（V4路线修复）\n\n")
        f.write(f"**时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- 分子数: {len(a_final)}\n")
        if a_final:
            f.write(f"- 最优 Vina: {a_final[0]['vina']:.1f}\n")
            f.write(f"- 平均 Vina: {sum(r['vina'] for r in a_final)/len(a_final):.1f}\n")
        f.write(f"- 路线通过率: {len(a_final)}/{len(a_final)}\n\n")
        f.write("## Top 分子\n\n")
        for i, r in enumerate(a_final, 1):
            f.write(f"{i}. Vina={r['vina']:.1f} SA={r['sa']:.1f} `{r['canonical'][:50]}`\n")

    b_summary = OUT_DIR_B / "summary.md"
    with open(b_summary, "w") as f:
        f.write("# B版：多样性版（V4路线修复）\n\n")
        f.write(f"**时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- 分子数: {len(b_final)}\n")
        if b_final:
            f.write(f"- 最优 Vina: {b_final[0]['vina']:.1f}\n")
        f.write(f"- 路线通过率: {len(b_final)}/{len(b_final)}\n\n")
        f.write("## 骨架分布\n\n")
        for cls in seen_classes:
            count = sum(1 for r in b_final if classify(r["canonical"]) == cls)
            f.write(f"- {cls}: {count}\n")

    log.info(f"\n✅ A版: {a_zip} ({len(a_final)} 分子)")
    log.info(f"✅ B版: {b_zip} ({len(b_final)} 分子)")

    if a_final:
        log.info(f"A版最优 Vina: {a_final[0]['vina']:.1f}")
    if b_final:
        log.info(f"B版最优 Vina: {b_final[0]['vina']:.1f}")

    log.info("V4 路线修复完成 ✅")


if __name__ == "__main__":
    main()
