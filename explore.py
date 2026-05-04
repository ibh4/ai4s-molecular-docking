#!/usr/bin/env python3
"""
AI4S 夜间全自动探索系统
目标：系统遍历所有分子生成策略+对接参数，找到最优提交组合
时间预算：8小时（00:30 - 08:30）
"""
import os, sys, csv, json, time, random, logging, subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path("/Users/pwngwc/.openclaw/workspace/retrosyn")))
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, BRICS
from backend.mol_utils import validate_smiles, get_mol_info

# ── 配置 ──────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/pwngwc/ai4s_chem")
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)
VINA = BASE_DIR / "bin" / "vina"
RECEPTOR = BASE_DIR / "receptor.pdbqt"
LOG_FILE = RESULT_DIR / "exploration.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("explore")

# ── Pocket centers to try ──────────────────────────────────────────
# 基于蛋白质几何中心 (18.3, 2.3, 21.4)
POCKETS = {
    "center":    ([18.3, 2.3, 21.4], [20, 20, 20]),
    "small_box": ([18.3, 2.3, 21.4], [15, 15, 15]),
    "shift_n":   ([18.3, 12.3, 21.4], [20, 20, 20]),
    "shift_s":   ([18.3, -7.7, 21.4], [20, 20, 20]),
    "shift_e":   ([28.3, 2.3, 21.4], [20, 20, 20]),
    "shift_w":   ([8.3, 2.3, 21.4], [20, 20, 20]),
}

# ── 分子库 ─────────────────────────────────────────────────────────

DRUG_SCAFFOLDS = {
    "amides": [
        "O=C(Nc1ccccc1)c1ccncc1",
        "c1ccc(NC(=O)c2cccnc2)cc1",
        "O=C(Nc1ccccc1)c1ccc(F)cc1",
        "c1ccc(NC(=O)C(F)(F)F)cc1",
        "c1ccc(NC(=O)c2ccccc2)cc1",
        "O=C(Nc1ccccc1)c1ccccc1",
        "c1ccc(NC(=O)C2CC2)cc1",
        "c1ccc(NC(=O)C2CCNCC2)cc1",
        "c1ccc(NC(=O)C2CCOCC2)cc1",
    ],
    "biphenyls": [
        "c1ccc(-c2ccccc2)cc1",
        "c1ccc(-c2ccc(N)cc2)cc1",
        "c1ccc(-c2ccc(C(=O)O)cc2)cc1",
        "c1ccc(-c2ccccn2)cc1",
        "c1ccc(-c2ccncc2)cc1",
        "Fc1ccc(-c2ccccc2)cc1",
        "COc1ccc(-c2ccccc2)cc1",
        "CCc1ccc(-c2ccccc2)cc1",
    ],
    "heterocycles": [
        "c1ccc2[nH]ccc2c1",
        "c1ccc2[nH]ncc2c1",
        "c1cnc2ccccc2n1",
        "c1ccc2nc(N)ccc2c1",
        "O=c1[nH]c2ccccc2o1",
        "c1ccc2c(c1)c(=O)[nH]c(=O)n2",
        "c1ccncc1",
        "c1ccsc1",
        "c1ccoc1",
    ],
    "fused_rings": [
        "c1ccc2c(c1)cc1ccccc1c2",
        "c1ccc(-c2cc3ccccc3[nH]2)cc1",
        "c1ccc(-c2nc3ccccc3[nH]2)cc1",
        "O=C(Nc1ccccc1)c1ccc2ccccc2n1",
        "c1ccc(Nc2ccnc3ccccc23)cc1",
    ],
    "sulfonamides": [
        "NS(=O)(=O)c1ccccc1",
        "NS(=O)(=O)c1ccc(-c2ccccc2)cc1",
        "CS(=O)(=O)c1ccc(-c2ccccc2)cc1",
        "O=S(=O)(Nc1ccccc1)c1ccccc1",
    ],
    "trifluoromethyl": [
        "FC(F)(F)c1ccc(-c2ccccc2)cc1",
        "FC(F)(F)c1ccc(NC(=O)c2ccccc2)cc1",
        "O=C(Nc1ccccc1)c1ccc(C(F)(F)F)cc1",
        "FC(F)(F)c1ccc(-c2ccccn2)cc1",
    ],
    "cyano": [
        "N#Cc1ccc(-c2ccccc2)cc1",
        "N#Cc1cccnc1Nc1ccccc1",
        "c1ccc(Nc2ncccc2C#N)cc1",
        "N#CC(=O)Nc1ccccc1",
    ],
    "linker_variants": [
        "c1ccc(NC(=O)c2ccc(-c3ccccc3)cc2)cc1",
        "c1ccc(-c2ccc(NC(=O)c3ccccc3)cc2)cc1",
        "c1ccc(NC(=O)c2cc(-c3ccccc3)no2)cc1",
        "CNC(=O)c1ccc(-c2ccccc2)cc1",
        "O=C(Nc1ccccc1)c1ccc2ncccc2c1",
    ],
}


def get_all_molecules():
    """从所有来源收集分子"""
    all_mols = {}
    for cat, smis in DRUG_SCAFFOLDS.items():
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                all_mols[Chem.MolToSmiles(mol)] = cat
    return all_mols


def _make_pdbqt_line(serial, atom_name, resname, chain, resseq, x, y, z, element):
    """79列PDBQT行"""
    name4 = f' {atom_name.strip():<3s}'[:4]
    return (f'ATOM  {serial:5d}{name4}{resname[:3]:>3s} {chain}{resseq:5d}   '
            f'{x:8.3f}{y:8.3f}{z:8.3f}  0.00  0.00          {element:>2s} ')


def prepare_receptor():
    """准备受体PDBQT"""
    if RECEPTOR.exists():
        return True
    import subprocess
    # 尝试obabel
    try:
        subprocess.run(["obabel", str(BASE_DIR/"target.pdb"), "-O", str(RECEPTOR), "-h"],
                       capture_output=True, timeout=60)
        with open(RECEPTOR) as f:
            lines = f.readlines()
        clean = [l for l in lines if l.startswith("ATOM") or l.startswith("HETATM")]
        with open(RECEPTOR, "w") as f:
            f.writelines(clean)
        log.info(f"受体准备完成: {len(clean)}原子")
        return True
    except:
        return False


def smiles_to_pdbqt(smi, path):
    """SMILES → PDBQT 79列精确格式（匹配工作示例）"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return False
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1:
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass
    try:
        conf = mol.GetConformer()
    except:
        return False

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
        # 精确79列
        line = f"ATOM  {i+1:5d} {name4} LIG A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  0.00  0.00          {atype:>2s} "
        lines.append(line)
    lines.append("ENDROOT")
    lines.append("TORSDOF 0")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return True


def run_vina(ligand_pdbqt, out_pdbqt, center, size, exhaustiveness=16):
    """运行Vina，返回binding score"""
    cmd = [
        str(VINA), "--receptor", str(RECEPTOR),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
        "--out", str(out_pdbqt),
        "--num_modes", "1", "--exhaustiveness", str(exhaustiveness),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in (result.stdout + result.stderr).split("\n"):
            line = line.strip()
            if line.startswith("1 "):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1])
        return None
    except:
        return None


def calc_sa(smi):
    """简化SA score"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return 10
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


def generate_route(smi):
    """简单路线生成"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None
    # BRICS
    try:
        frags = list(BRICS.BRICSDecompose(mol, returnMols=False))
        if len(frags) >= 2:
            return ".".join(frags[:2]) + ">>" + smi
    except:
        pass
    # Template
    templates = [
        ("[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[OH].[N:3]", "酰胺"),
        ("[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[O:3][C:4]", "酯"),
        ("[c:1][c:2]>>[c:1][Br].[c:2]B(O)(O)", "Suzuki"),
    ]
    for smarts, name in templates:
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            products = rxn.RunReactants((mol,))
            if products:
                for pset in products[:1]:
                    reactants = []
                    ok = True
                    for p in pset:
                        try:
                            Chem.SanitizeMol(p)
                            reactants.append(Chem.MolToSmiles(p))
                        except:
                            ok = False
                    if ok and reactants:
                        return ".".join(reactants) + ">>" + smi
        except:
            pass
    return None


def main():
    log.info("=" * 60)
    log.info("AI4S 夜间全自动探索系统")
    log.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # 准备受体
    if not RECEPTOR.exists():
        prepare_receptor()

    # 收集所有分子
    all_mols = get_all_molecules()
    log.info(f"分子库: {len(all_mols)} 个分子，{len(set(all_mols.values()))} 个类别")

    # 遍历口袋 × 分子
    all_results = []
    experiment_id = 0

    for pocket_name, (center, size) in POCKETS.items():
        log.info(f"\n{'='*50}")
        log.info(f"口袋: {pocket_name} center={center} size={size}")
        log.info(f"{'='*50}")

        docking_dir = BASE_DIR / "docking" / pocket_name
        docking_dir.mkdir(parents=True, exist_ok=True)

        for mol_idx, (smi, category) in enumerate(all_mols.items()):
            # 准备配体
            lig_pdbqt = docking_dir / f"mol_{mol_idx:04d}.pdbqt"
            out_pdbqt = docking_dir / f"mol_{mol_idx:04d}_out.pdbqt"

            if not smiles_to_pdbqt(smi, str(lig_pdbqt)):
                continue

            # 对接
            score = run_vina(str(lig_pdbqt), str(out_pdbqt), center, size)

            if score is not None:
                sa = calc_sa(smi)
                route = generate_route(smi)
                route_valid = route and ">>" in route and route.split(">>")[0].strip() != route.split(">>")[1].strip()

                all_results.append({
                    "smiles": smi,
                    "category": category,
                    "pocket": pocket_name,
                    "vina_score": score,
                    "sa_score": sa,
                    "route": route,
                    "route_valid": route_valid,
                    "mol_score": 0.8 * (-score) + 0.1 * 1.0 + 0.1 * max(0, (4 - sa) / 4),
                })

        pocket_results = [r for r in all_results if r["pocket"] == pocket_name]
        best = min(pocket_results, key=lambda x: x["vina_score"]) if pocket_results else None
        if best:
            log.info(f"  最优: Vina={best['vina_score']:.1f} SA={best['sa_score']:.1f} 路线={'✅' if best['route_valid'] else '❌'}")
            log.info(f"  分子: {best['smiles'][:50]}")

    # ── 汇总 ──────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"探索完成: {len(all_results)} 个有效结果")
    log.info(f"{'='*60}")

    # 按Vina排序
    all_results.sort(key=lambda x: x["vina_score"])

    # 最优结果
    log.info("\n🏆 Top 20 最优分子:")
    for i, r in enumerate(all_results[:20], 1):
        log.info(f"  {i}. Vina={r['vina_score']:.1f} | SA={r['sa_score']:.1f} | 路线={'✅' if r['route_valid'] else '❌'} | {r['smiles'][:40]}")
        log.info(f"     口袋: {r['pocket']} | 类别: {r['category']}")

    # 按口袋分组最优
    log.info("\n📊 各口袋最优:")
    for pocket_name in POCKETS:
        pocket_results = [r for r in all_results if r["pocket"] == pocket_name]
        if pocket_results:
            best = min(pocket_results, key=lambda x: x["vina_score"])
            log.info(f"  {pocket_name}: Vina={best['vina_score']:.1f} {best['smiles'][:40]}")

    # 按类别分组最优
    log.info("\n📊 各类别最优:")
    for cat in set(r["category"] for r in all_results):
        cat_results = [r for r in all_results if r["category"] == cat]
        best = min(cat_results, key=lambda x: x["vina_score"])
        log.info(f"  {cat}: Vina={best['vina_score']:.1f} {best['smiles'][:40]}")

    # 保存全量结果
    json_path = RESULT_DIR / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"\n全量结果: {json_path}")

    # 生成多个候选CSV
    # 方案1: 最优Vina分子（有路线）
    with_route = [r for r in all_results if r["route_valid"]]
    with_route.sort(key=lambda x: x["vina_score"])
    best_route = with_route[:30]

    csv1 = RESULT_DIR / "plan_best_vina.csv"
    with open(csv1, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in best_route:
            writer.writerow([r["smiles"], r["route"]])
    log.info(f"方案1 (最优Vina+路线): {csv1} ({len(best_route)}个)")

    # 方案2: 最多分子
    all_with_route = [r for r in all_results if r["route_valid"]]
    all_with_route.sort(key=lambda x: x["vina_score"])
    csv2 = RESULT_DIR / "plan_max_molecules.csv"
    with open(csv2, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in all_with_route[:50]:
            writer.writerow([r["smiles"], r["route"]])
    log.info(f"方案2 (最多分子): {csv2} ({min(len(all_with_route), 50)}个)")

    # 方案3: 多样性（每个类别取最优）
    diverse = []
    seen_cats = set()
    for r in all_results:
        if r["category"] not in seen_cats and r["route_valid"]:
            diverse.append(r)
            seen_cats.add(r["category"])
    csv3 = RESULT_DIR / "plan_diverse.csv"
    with open(csv3, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in diverse:
            writer.writerow([r["smiles"], r["route"]])
    log.info(f"方案3 (多样性): {csv3} ({len(diverse)}个)")

    elapsed = time.time() - time.mktime(datetime.now().timetuple())
    log.info(f"\n总耗时: {elapsed:.0f}秒")
    log.info("夜间探索完成 ✅")


if __name__ == "__main__":
    main()
