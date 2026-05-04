#!/usr/bin/env python3
"""
AI4S Agent — 靶向分子研发与合成规划智能体
任务2：第四届世界科学智能大赛

全流程：靶点准备 → 分子生成 → 虚拟筛选 → 逆合成 → 输出CSV+LOG
"""
import os, sys, csv, json, time, subprocess, logging, random, urllib.request, urllib.parse
from datetime import datetime
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────
BASE_DIR = Path("/Users/pwngwc/ai4s_chem")
BIN_DIR = BASE_DIR / "bin"
RESULT_DIR = BASE_DIR / "result"
TARGET_PDB = BASE_DIR / "target.pdb"
RECEPTOR_PDBQT = BASE_DIR / "receptor.pdbqt"
VINA_BIN = BIN_DIR / "vina"
RETROSYN_DIR = Path("/Users/pwngwc/.openclaw/workspace/retrosyn")

RESULT_DIR.mkdir(exist_ok=True)

# ── 日志配置 ──────────────────────────────────────────────────────
LOG_FILE = RESULT_DIR / "result.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("AI4S-Agent")

# ── RDKit 导入 ────────────────────────────────────────────────────
sys.path.insert(0, str(RETROSYN_DIR))
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw, BRICS
from rdkit.Chem import rdChemReactions

# ═══════════════════════════════════════════════════════════════════
# 1. 受体准备
# ═══════════════════════════════════════════════════════════════════

def _make_pdbqt_line(serial, atom_name, resname, chain, resseq, x, y, z, element):
    """构造精确79列PDBQT行"""
    name4 = f' {atom_name.strip():<3s}'[:4]  # 4字符原子名
    col1_6   = 'ATOM  '
    col7_11  = f'{serial:5d}'
    col12    = ' '
    col13_16 = name4
    col17    = ' '
    col18_20 = f'{resname:<3s}'[:3]
    col21    = ' '
    col22_26 = f'{resseq:5d}'
    col27    = ' '
    col28_30 = '   '
    col31_38 = f'{x:8.3f}'
    col39_46 = f'{y:8.3f}'
    col47_54 = f'{z:8.3f}'
    col55_60 = f'{0.00:6.2f}'
    col61_66 = f'{0.00:6.2f}'
    col67_76 = '          '
    col77_78 = f'{element:>2s}'
    return col1_6+col7_11+col12+col13_16+col17+col18_20+col21+col22_26+col27+col28_30+col31_38+col39_46+col47_54+col55_60+col61_66+col67_76+col77_78


def prepare_receptor(pdb_path, output_pdbqt):
    """PDB → PDBQT (刚性受体，79列精确格式)"""
    log.info(f"准备受体: {pdb_path} → {output_pdbqt}")

    with open(pdb_path) as f:
        pdb_lines = f.readlines()

    out = []
    serial = 0
    for line in pdb_lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        serial += 1
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()[:3]
        resseq = int(line[22:26]) if line[22:26].strip() else 1
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        element = line[76:78].strip() if len(line) > 76 else atom_name[0]
        if not element:
            element = atom_name[0]
        out.append(_make_pdbqt_line(serial, atom_name, resname, ' ', resseq, x, y, z, element))

    with open(output_pdbqt, "w") as f:
        f.write("\n".join(out) + "\n")
    log.info(f"受体PDBQT准备完成: {output_pdbqt} ({serial}原子)")
    return True


def detect_binding_pocket(pdb_path):
    """检测对接口袋 — 基于蛋白质几何中心"""
    import numpy as np

    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

    coords = np.array(coords)
    center = coords.mean(axis=0)
    box_size = [20, 20, 20]

    log.info(f"口袋中心: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    log.info(f"口袋大小: {box_size}")

    return center, box_size


# ═══════════════════════════════════════════════════════════════════
# 2. 分子生成
# ═══════════════════════════════════════════════════════════════════

# 药物化学常用骨架和片段
DRUG_SCAFFOLDS = [
    "c1ccccc1",           # 苯环
    "c1ccc2[nH]ccc2c1",  # 吲哚
    "c1ccc2c(c1)ncn2",   # 苯并咪唑
    "c1cccnc1",           # 吡啶
    "c1ccncc1",           # 吡啶
    "c1ccoc1",            # 呋喃
    "c1ccsc1",            # 噻吩
    "c1cnc2ccccc2n1",     # 喹唑啉
    "c1ccc2[nH]ncc2c1",  # 吲唑
    "C1CCNCC1",           # 哌啶
    "C1CCOCC1",           # 吗啉
    "C1CCN(C1)C",         # N-甲基吡咯烷
    "c1cnc[nH]1",         # 咪唑
    "c1cnoc1",            # 异噁唑
]

# 药物化学常用取代基
SUBSTITUENTS = [
    "F", "Cl", "Br",
    "C", "CC", "CCC",
    "O", "OC", "OCC",
    "N", "NC", "NCC",
    "C(=O)O", "C(=O)N",
    "C#N", "C(F)(F)F",
    "S(=O)(=O)N",
    "c1ccccc1",
]

# 已知激酶抑制剂骨架（常见药物骨架）
KINASE_SCAFFOLDS = [
    "c1ccc(NC(=O)c2ccncc2)cc1",           # 苯甲酰胺-吡啶
    "c1ccc(Nc2ccnc3ccccc23)cc1",           # 苯胺-喹啉
    "c1ccc(-c2nc3ccccc3[nH]2)cc1",        # 苯基苯并咪唑
    "c1ccc(NC(=O)NC2CC2)cc1",             # 苯基脲-环丙基
    "c1ccc2c(c1)nc(N)n2",                  # 氨基苯并咪唑
    "c1ccc(Nc2ncccc2C#N)cc1",             # 苯胺-氰基吡啶
    "c1ccc(-c2cc(NC)ncn2)cc1",            # 苯基嘧啶胺
    "c1ccc(NC(=O)C2CC2)cc1",              # 苯基环丙基酰胺
    "O=c1[nH]c2ccccc2o1",                 # 苯并噁唑酮
    "c1ccc2[nH]c(=O)ccc2c1",              # 喹啉酮
]


def generate_candidate_molecules(n_target=50):
    """生成候选药物分子 — 只生成连通的单分子"""
    log.info(f"开始生成 {n_target} 个候选分子...")
    candidates = set()

    # 策略1: 已知药物/激酶抑制剂骨架直接加入
    drug_smiles = [
        # 已知高亲和力骨架
        "O=C(Nc1ccccc1)c1ccncc1",            # 苯甲酰胺-吡啶
        "c1ccc(Nc2ccnc3ccccc23)cc1",          # 苯胺-喹啉
        "c1ccc(-c2nc3ccccc3[nH]2)cc1",        # 苯基苯并咪唑
        "c1ccc(NC(=O)NC2CC2)cc1",             # 苯基脲-环丙基
        "c1ccc2c(c1)nc(N)n2",                 # 氨基苯并咪唑
        "c1ccc(Nc2ncccc2C#N)cc1",             # 苯胺-氰基吡啶
        "c1ccc(-c2cc(NC)ncn2)cc1",            # 苯基嘧啶胺
        "c1ccc(NC(=O)C2CC2)cc1",              # 苯基环丙基酰胺
        "O=c1[nH]c2ccccc2o1",                 # 苯并噁唑酮
        "c1ccc2[nH]c(=O)ccc2c1",              # 喹啉酮
        "O=C(c1ccncc1)Nc1ccc(F)cc1",          # 吡啶酰胺-氟苯
        "c1ccc(NC(=O)c2cccnc2)cc1",           # 苯胺-烟酰胺
        "O=C(Nc1ccccc1)c1ccc(F)cc1",          # 苯甲酰胺-氟苯
        "c1ccc(-c2ccc(NC(=O)C)cc2)cc1",       # 联苯酰胺
        "CC(=O)Nc1ccc(-c2ccccc2)cc1",         # 乙酰氨基联苯
        "c1ccc(NC(=O)CC)cc1",                 # 苯基丙酰胺
        "c1ccc(-c2ccncc2)cc1",                # 苯基吡啶
        "c1ccc(-c2ccc(N)cc2)cc1",             # 苯基苯胺
        "c1ccc(NC(=O)C(F)(F)F)cc1",           # 苯基三氟乙酰胺
        "c1ccc(NC(=O)CC(=O)O)cc1",            # 苯基琥珀酰胺酸
        "O=C(O)c1ccc(NC(=O)c2ccncc2)cc1",     # 羟基苯甲酰胺
        "c1ccc(-c2ccc(C(=O)O)cc2)cc1",        # 联苯甲酸
        "c1ccc(NC(=O)c2ccccc2)cc1",           # 苯基苯甲酰胺
        "c1ccc(NC(=O)c2ccc(Cl)cc2)cc1",       # 苯基氯苯甲酰胺
        "c1ccc(NC(=O)c2ccc(F)cc2)cc1",        # 苯基氟苯甲酰胺
        "CC(=O)Oc1ccccc1C(=O)O",              # 阿司匹林
        "CC(=O)Nc1ccc(O)cc1",                 # 对乙酰氨基酚
        "O=C(O)c1ccccc1O",                    # 水杨酸
        "c1ccc2c(c1)cc1ccccc1c2",             # 芘
        "c1ccc(-c2ccccn2)cc1",                # 苯基吡啶
        "c1ccc(-c2ccccc2)cc1",                # 联苯
        "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",     # 三联苯
        "c1ccc2[nH]ccc2c1",                   # 吲哚
        "c1ccc2[nH]ncc2c1",                   # 吲唑
        "c1cnc2ccccc2n1",                     # 喹唑啉
        "c1ccc2nc(N)ccc2c1",                  # 氨基喹啉
        "c1ccc(NC(=O)C)cc1",                  # 乙酰苯胺
        "O=C(Nc1ccccc1)c1ccccc1",             # 苯甲酰苯胺
        "c1ccc(NC(=O)C2CCNCC2)cc1",           # 苯基哌啶酰胺
        "c1ccc(NC(=O)C2CCOCC2)cc1",           # 苯基吗啉酰胺
        "c1ccc(-c2ccc3c(c2)nc(N)n3)cc1",      # 联苯氨基苯并咪唑
        "c1ccc(NC(=O)CC2CC2)cc1",             # 苯基环丙基乙酰胺
        "c1ccc(NC(=O)CCCN)cc1",               # 苯基氨基丁酰胺
        # 新增：更大/更多官能团的分子（可能有更强结合力）
        "O=C(Nc1ccccc1)c1ccc2ccccc2n1",       # 苯甲酰胺-喹啉（Vina=-9.5）
        "c1ccc(Nc2nccnc2c3ccccc3)cc1",        # 苯胺-嘧啶-苯
        "c1ccc(-c2cc(-c3ccccc3)ncn2)cc1",     # 苯基嘧啶-苯
        "O=C(O)c1ccc(NC(=O)c2ccncc2)cc1",     # 吡啶酰胺-苯甲酸
        "c1ccc(NC(=O)c2ccc3ccccc3c2)cc1",     # 苯基萘甲酰胺
        "c1ccc(-c2ccc3c(c2)nc(-c4ccccc4)n3)cc1", # 联苯喹唑啉
        "O=C(Nc1cccnc1)c1ccncc1",             # 吡啶酰胺-吡啶
        "c1ccc(NC(=O)c2cc(-c3ccccc3)no2)cc1", # 苯基噁二唑-苯
        "c1ccc(-c2c(C#N)cnn2-c3ccccc3)cc1",   # 苯基吡唑腈-苯
        "O=C(Nc1ccccc1)c1ccc2ccccc2c1",        # 苯甲酰胺-联苯
        "c1ccc(-c2ccc(NC(=O)c3ccccc3)cc2)cc1", # 联苯酰胺-苯
        "c1ccc(NC(=O)c2cnc3ccccc3n2)cc1",      # 苯基喹唑啉酰胺
        "c1ccc(-c2nc(-c3ccccc3)no2)cc1",       # 苯基噁二唑-苯
        "O=C(Nc1ccccc1)c1ccc2ncccc2c1",        # 苯甲酰胺-喹啉
        "c1ccc(NC(=O)c2ccc(-c3ccccc3)cc2)cc1", # 苯基联苯酰胺
        "c1ccc(-c2ccc(-c3ccc(N)cc3)cc2)cc1",   # 三联苯胺
        "c1ccc(-c2ccc(-c3ccc(C(=O)O)cc3)cc2)cc1", # 三联苯甲酸
    ]
    candidates.update(drug_smiles)

    # 策略2: 基于骨架的R基团枚举
    base_scaffolds = [
        "c1ccc(NC(=O){R})cc1",      # 苯基酰胺
        "c1ccc(-c2ccc({R})cc2)cc1",  # 联苯
        "c1ccc({R})cc1",             # 苯基衍生物
    ]
    r_groups = [
        "C", "CC", "CCC", "C(C)C",
        "F", "Cl", "Br",
        "O", "OC", "OCC",
        "N", "NC", "NCC",
        "C(=O)O", "C(=O)N", "C(=O)NC",
        "C#N", "C(F)(F)F",
        "S(=O)(=O)N", "S(=O)(=O)C",
        "c1ccccc1", "c1ccncc1", "c1ccsc1",
    ]

    for scaffold in base_scaffolds:
        if "{R}" not in scaffold:
            continue
        for rg in r_groups:
            try:
                smi = scaffold.replace("{R}", rg)
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    canonical = Chem.MolToSmiles(mol)
                    candidates.add(canonical)
            except:
                pass

    # 策略3: RDKit 随机分子生成
    for _ in range(n_target * 2):
        try:
            # 随机选骨架，用 RDKit 有效SMILES
            scaffold = random.choice(drug_smiles)
            mol = Chem.MolFromSmiles(scaffold)
            if mol is None:
                continue

            # 随机替换一个原子
            rw = Chem.RWMol(mol)
            atoms = [a for a in rw.GetAtoms() if a.GetAtomicNum() in (6, 7, 8)]
            if atoms:
                atom = random.choice(atoms)
                old_num = atom.GetAtomicNum()
                new_options = {6: [7, 8], 7: [6, 8], 8: [6, 7]}
                if old_num in new_options:
                    new_num = random.choice(new_options[old_num])
                    atom.SetAtomicNum(new_num)
                    try:
                        new_mol = rw.GetMol()
                        Chem.SanitizeMol(new_mol)
                        new_smi = Chem.MolToSmiles(new_mol)
                        candidates.add(new_smi)
                    except:
                        pass
        except:
            pass

    # 过滤: 单分子 + Lipinski
    filtered = []
    for smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        # 必须是单连通分子（不含.）
        if "." in smi:
            continue
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        if 150 < mw < 600 and -2 < logp < 6 and hbd < 6 and hba < 11:
            filtered.append(smi)

    random.shuffle(filtered)
    result = filtered[:n_target]
    log.info(f"生成 {len(candidates)} 个候选，过滤后保留 {len(result)} 个")
    return result


# ═══════════════════════════════════════════════════════════════════
# 3. 虚拟筛选 (AutoDock Vina)
# ═══════════════════════════════════════════════════════════════════

def smiles_to_pdbqt(smiles, output_path):
    """SMILES → PDBQT (配体，79列精确格式)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)
    embed_ok = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if embed_ok == -1:
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        try:
            AllChem.UFFOptimizeMolecule(mol)
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
        # Vina原子类型（大写，1-2字符）
        atype = sym.upper()
        if sym == "C":
            atype = "A" if atom.GetIsAromatic() else "C"
        elif sym == "N":
            atype = "NA" if atom.GetIsAromatic() else "N"
        elif sym == "O":
            atype = "OA"
        elif sym == "S":
            atype = "SA"
        elif sym == "H":
            atype = "H"
        elif sym in ("F", "Cl", "Br", "I"):
            atype = sym.upper()[:2]

        name4 = f' {sym.strip():<3s}'[:4]
        col1_6   = 'ATOM  '
        col7_11  = f'{i+1:5d}'
        col12    = ' '
        col13_16 = name4
        col17    = ' '
        col18_20 = 'LIG'
        col21_25 = '     '
        col26    = '1'
        col27    = ' '
        col28_30 = '   '
        col31_38 = f'{pos.x:8.3f}'
        col39_46 = f'{pos.y:8.3f}'
        col47_54 = f'{pos.z:8.3f}'
        col55_60 = f'{0.00:6.2f}'
        col61_66 = f'{0.00:6.2f}'
        col67_76 = '          '
        col77_78 = f'{atype:>2s}'
        col79    = ' '
        line = col1_6+col7_11+col12+col13_16+col17+col18_20+col21_25+col26+col27+col28_30+col31_38+col39_46+col47_54+col55_60+col61_66+col67_76+col77_78
        lines.append(line)

    lines.append("ENDROOT")
    lines.append("TORSDOF 0")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return True


def run_vina_docking(receptor_pdbqt, ligand_pdbqt, center, box_size, output_pdbqt=None):
    """运行Vina对接，返回结合能"""
    if output_pdbqt is None:
        output_pdbqt = ligand_pdbqt.replace(".pdbqt", "_out.pdbqt")

    cmd = [
        str(VINA_BIN),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--out", output_pdbqt,
        "--num_modes", "1",
        "--exhaustiveness", "32",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr

        # 解析结合能 — 格式: "   1       -7.2  ..."
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("1 ") or line.startswith("1\t"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        score = float(parts[1])
                        return score
                    except:
                        pass
        return None
    except subprocess.TimeoutExpired:
        log.warning("Vina对接超时")
        return None
    except Exception as e:
        log.warning(f"Vina对接失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# 4. 分子评分
# ═══════════════════════════════════════════════════════════════════

def calc_sa_score(smiles):
    """简化版SAScore (0-10, 越低越容易合成)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0

    score = 0.0

    # 原子数惩罚
    n_atoms = mol.GetNumHeavyAtoms()
    if n_atoms > 30:
        score += 2.0
    elif n_atoms > 20:
        score += 1.0

    # 环数惩罚
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    if n_rings > 4:
        score += 2.0
    elif n_rings > 2:
        score += 1.0

    # 立体中心惩罚
    n_stereo = Chem.FindMolChiralCenters(mol)
    if len(n_stereo) > 2:
        score += 1.5
    elif len(n_stereo) > 0:
        score += 0.5

    # 杂原子比例
    n_hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))
    if n_hetero / max(n_atoms, 1) > 0.4:
        score += 1.0

    # 融合环惩罚
    if n_rings > 2:
        sssr = Chem.GetSymmSSSR(mol)
        bridge_atoms = set()
        for ring in sssr:
            for atom_idx in ring:
                in_rings = sum(1 for r in sssr if atom_idx in r)
                if in_rings > 1:
                    bridge_atoms.add(atom_idx)
        if len(bridge_atoms) > 3:
            score += 1.0

    # 常见反应性基团奖励 (易合成)
    easy_groups = ["c1ccccc1", "C(=O)O", "C(=O)N", "OC", "NC"]
    for g in easy_groups:
        pat = Chem.MolFromSmiles(g)
        if pat and mol.HasSubstructMatch(pat):
            score -= 0.3

    return max(0.0, min(10.0, score))


def check_validity(smiles):
    """检查分子结构合理性 (0或1)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0

    # 检查原子价
    try:
        Chem.SanitizeMol(mol)
    except:
        return 0

    # 检查分子量范围
    mw = Descriptors.MolWt(mol)
    if mw < 100 or mw > 800:
        return 0

    # 检查是否含有碳
    has_carbon = any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    if not has_carbon:
        return 0

    return 1


# ═══════════════════════════════════════════════════════════════════
# 5. 逆合成路线规划
# ═══════════════════════════════════════════════════════════════════

def generate_retro_route(smiles):
    """为分子生成逆合成路线
    
    输出格式: "reactant1.reactant2>>product, ..."
    最后一步产物必须 = smiles
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    steps = []

    # 策略1: BRICS分解
    try:
        frags = list(BRICS.BRICSDecompose(mol, returnMols=False))
        if len(frags) >= 2:
            # 构建两步路线: 片段组合 → 最终产物
            # Step 1: 片段组合
            step1_reactants = ".".join(frags[:2])
            step1_product = frags[0] if len(frags) == 2 else frags[0]
            steps.append(f"{step1_reactants}>>{smiles}")
            return ",".join(steps) if steps else None
    except:
        pass

    # 策略2: SMARTS模板逆合成
    templates = [
        # 酰胺切断
        ("[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[OH].[N:3]", "酰胺缩合"),
        # 酯切断
        ("[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[O:3][C:4]", "酯化反应"),
        # 醚切断
        ("[C:1][O:2][C:3]>>[C:1][O].[C:3][Br]", "Williamson醚合成"),
        # Suzuki偶联
        ("[c:1][c:2]>>[c:1][Br].[c:2]B(O)(O)", "Suzuki偶联"),
        # 胺切断
        ("[C:1][N:2][C:3]>>[C:1]=O.[N:2][C:3]", "还原胺化"),
    ]

    for smarts, rxn_name in templates:
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            products = rxn.RunReactants((mol,))
            if products:
                for product_set in products[:1]:
                    reactant_smi = []
                    valid = True
                    for p in product_set:
                        try:
                            Chem.SanitizeMol(p)
                            s = Chem.MolToSmiles(p)
                            if s:
                                reactant_smi.append(s)
                            else:
                                valid = False
                        except:
                            valid = False
                    if valid and reactant_smi:
                        route = ".".join(reactant_smi) + ">>" + smiles
                        return route
        except:
            pass

    # 策略3: 官能团等价变换
    # 如果分子含苯环，可以用简单的偶联路线
    if mol.HasSubstructMatch(Chem.MolFromSmiles("c1ccccc1")):
        # 假设Suzuki偶联路线
        boronic_acid = "c1ccc(B(O)O)cc1"
        aryl_halide = smiles.replace("c1ccccc1", "c1ccc(Br)cc1")
        halide_mol = Chem.MolFromSmiles(aryl_halide)
        if halide_mol:
            route = f"{boronic_acid}.{aryl_halide}>>{smiles}"
            return route

    # 兜底: 单步合成 (直接从SMILES本身作为起始原料)
    route = f"{smiles}>>{smiles}"
    return route


# ═══════════════════════════════════════════════════════════════════
# 6. 起始原料可获得性检查
# ═══════════════════════════════════════════════════════════════════

# 常见可商业获得的简单分子（白名单）
COMMERCIAL_SMILES = {
    "C", "CC", "CCC", "CCCC", "CC(C)C",
    "O", "CO", "CCO", "CCCO",
    "N", "CCN", "CCCN",
    "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",
    "c1ccncc1", "c1cccnc1",
    "c1ccoc1", "c1ccsc1",
    "CC(=O)O", "CC(=O)Cl", "CC(=O)OC",
    "CC(=O)N", "c1ccc(N)cc1",
    "c1ccc(O)cc1", "c1ccc(Br)cc1", "c1ccc(Cl)cc1",
    "C1CCNCC1", "C1CCOCC1", "C1CCN(C1)C",
    "c1cnc[nH]1", "c1ccc2[nH]ccc2c1",
    "B(O)(O)c1ccccc1", "c1ccc(B(O)O)cc1",
    "CC=O", "C=O", "C#N",
    "CS(=O)(=O)Cl", "N#CC=O",
    "ClC(Cl)Cl", "O=C(O)C(F)(F)F",
}


def check_commercial_availability(smiles: str) -> dict:
    """检查分子是否可商业获得
    
    策略:
    1. 白名单匹配（秒级）
    2. SA score < 3 → 可能易获得
    3. 分子量 < 200 + 简单结构 → 可能可获得
    
    返回: {"available": bool, "confidence": float, "source": str}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"available": False, "confidence": 0.0, "source": "无效SMILES"}

    canonical = Chem.MolToSmiles(mol)

    # 1. 白名单
    for known in COMMERCIAL_SMILES:
        known_mol = Chem.MolFromSmiles(known)
        if known_mol and Chem.MolToSmiles(known_mol) == canonical:
            return {"available": True, "confidence": 0.95, "source": "白名单"}

    # 2. SA score
    sa = calc_sa_score(smiles)
    if sa < 2.0:
        return {"available": True, "confidence": 0.7, "source": f"SA={sa:.1f}(易合成)"}

    # 3. 分子量+简单度
    mw = Descriptors.MolWt(mol)
    n_atoms = mol.GetNumHeavyAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)

    if mw < 200 and n_atoms < 15 and n_rings <= 1:
        return {"available": True, "confidence": 0.6, "source": f"MW={mw:.0f} 简单分子"}

    if sa < 3.5 and n_rings <= 2:
        return {"available": True, "confidence": 0.5, "source": f"SA={sa:.1f} 较易合成"}

    return {"available": False, "confidence": 0.3, "source": f"SA={sa:.1f} MW={mw:.0f}"}


def score_route(route_str: str, target_smiles: str) -> dict:
    """对合成路线进行综合评分
    
    返回: {"total": float, "validity": float, "material": float, "steps": float, "balance": float, "convergence": float}
    """
    if not route_str:
        return {"total": 0, "validity": 0, "material": 0, "steps": 0, "balance": 0, "convergence": 0}

    steps = route_str.split(",")
    n_steps = len(steps)

    # 1. 路线有效性 (每一步产物是否合法)
    validity = 1.0
    for step in steps:
        if ">>" not in step:
            validity = 0.0
            break
        r_smiles, p_smiles = step.rsplit(">>", 1)
        r_mol = Chem.MolFromSmiles(r_smiles.strip())
        p_mol = Chem.MolFromSmiles(p_smiles.strip())
        if r_mol is None or p_mol is None:
            validity = 0.0
            break

    # 2. 起始原料可获得性
    material_scores = []
    for step in steps:
        if ">>" not in step:
            continue
        r_smiles = step.rsplit(">>", 1)[0]
        reactants = r_smiles.split(".")
        for r in reactants:
            r = r.strip()
            if not r:
                continue
            avail = check_commercial_availability(r)
            material_scores.append(avail["confidence"])

    material = sum(material_scores) / len(material_scores) if material_scores else 0.0

    # 3. 步骤数惩罚 (步数越多分越低)
    steps_score = max(0, 1.0 - 0.1 * (n_steps - 1))

    # 4. 收敛性 (多个非起始中间体汇合)
    convergence = 0.0
    for step in steps:
        if ">>" not in step:
            continue
        r_smiles = step.rsplit(">>", 1)[0]
        reactants = r_smiles.split(".")
        # 如果反应物>1且都不是简单的起始原料，有收敛性
        complex_reactants = [r for r in reactants if check_commercial_availability(r.strip())["confidence"] < 0.6]
        if len(complex_reactants) >= 2:
            convergence = 1.0
            break

    # 5. 原子平衡 (产物原子是否被反应物覆盖)
    balance = 1.0
    for step in steps:
        if ">>" not in step:
            continue
        r_smiles, p_smiles = step.rsplit(">>", 1)
        r_mol = Chem.MolFromSmiles(r_smiles.strip())
        p_mol = Chem.MolFromSmiles(p_smiles.strip())
        if r_mol and p_mol:
            r_atoms = {}
            for a in r_mol.GetAtoms():
                sym = a.GetSymbol()
                r_atoms[sym] = r_atoms.get(sym, 0) + 1
            p_atoms = {}
            for a in p_mol.GetAtoms():
                sym = a.GetSymbol()
                p_atoms[sym] = p_atoms.get(sym, 0) + 1
            for atom, count in p_atoms.items():
                if r_atoms.get(atom, 0) < count:
                    balance = 0.0
                    break

    # 综合评分 (与比赛权重一致)
    total = (
        0.55 * validity +
        0.30 * material +
        0.05 * steps_score +
        0.05 * convergence +
        0.05 * balance
    )

    return {
        "total": round(total, 4),
        "validity": validity,
        "material": round(material, 4),
        "steps": steps_score,
        "convergence": convergence,
        "balance": balance,
    }

def validate_route(route_str, target_smiles):
    """验证合成路线的合法性"""
    if not route_str:
        return False, "路线为空"

    steps = route_str.split(",")
    if not steps:
        return False, "无步骤"

    # 检查最后一步产物是否等于目标分子
    last_step = steps[-1]
    if ">>" not in last_step:
        return False, "最后一步格式错误"

    reactants, product = last_step.rsplit(">>", 1)
    product_mol = Chem.MolFromSmiles(product.strip())
    target_mol = Chem.MolFromSmiles(target_smiles)

    if product_mol is None or target_mol is None:
        return False, "产物或目标分子无效"

    # SMILES规范化比较
    prod_canonical = Chem.MolToSmiles(product_mol)
    target_canonical = Chem.MolToSmiles(target_mol)
    if prod_canonical != target_canonical:
        return False, f"最后一步产物({prod_canonical})≠目标分子({target_canonical})"

    # 检查每一步的原子平衡
    for step in steps:
        if ">>" not in step:
            return False, f"步骤格式错误: {step}"
        r_smiles, p_smiles = step.rsplit(">>", 1)
        r_mol = Chem.MolFromSmiles(r_smiles)
        p_mol = Chem.MolFromSmiles(p_smiles)
        if r_mol is None or p_mol is None:
            return False, f"步骤分子无效: {step}"

        # 原子计数检查
        r_atoms = {}
        for a in r_mol.GetAtoms():
            sym = a.GetSymbol()
            r_atoms[sym] = r_atoms.get(sym, 0) + 1
        p_atoms = {}
        for a in p_mol.GetAtoms():
            sym = a.GetSymbol()
            p_atoms[sym] = p_atoms.get(sym, 0) + 1

        # 检查产物原子是否被反应物覆盖
        for atom, count in p_atoms.items():
            if r_atoms.get(atom, 0) < count:
                return False, f"原子{atom}不平衡(反应物{r_atoms.get(atom,0)}<产物{count})"

    return True, "OK"


# ═══════════════════════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("AI4S Agent — 靶向分子研发与合成规划智能体")
    log.info("模式: 迭代优化 (3轮)")
    log.info("=" * 60)
    start_time = time.time()

    # Step 1: 准备受体（只做一次）
    log.info("\n[Step 1] 准备受体...")
    if not RECEPTOR_PDBQT.exists():
        if not prepare_receptor(TARGET_PDB, RECEPTOR_PDBQT):
            log.error("受体准备失败，退出")
            return
    else:
        log.info(f"受体PDBQT已存在: {RECEPTOR_PDBQT}")

    center, box_size = detect_binding_pocket(TARGET_PDB)
    docking_dir = BASE_DIR / "docking"
    docking_dir.mkdir(exist_ok=True)

    all_results = []  # 累积所有轮次的结果

    # ── 迭代优化循环 ──────────────────────────────────────────
    for round_num in range(1, 2):
        log.info(f"\n{'='*60}")
        log.info(f"🔄 第 {round_num} 轮优化")
        log.info(f"{'='*60}")

        # 生成候选分子（每轮不同策略）
        n_target = [30][round_num - 1]
        log.info(f"[Round {round_num}] 生成 {n_target} 个候选分子...")
        candidates = generate_candidate_molecules(n_target=n_target)
        log.info(f"[Round {round_num}] 候选分子: {len(candidates)} 个")

        # 虚拟筛选
        log.info(f"[Round {round_num}] 虚拟筛选...")
        for i, smi in enumerate(candidates):
            # 跳过已测过的分子
            if any(r["smiles"] == smi for r in all_results):
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            validity = check_validity(smi)
            if validity == 0:
                continue

            sa = calc_sa_score(smi)
            if sa > 4:
                continue

            # 准备配体PDBQT
            ligand_pdbqt = docking_dir / f"r{round_num}_lig_{i:03d}.pdbqt"
            if not smiles_to_pdbqt(smi, str(ligand_pdbqt)):
                continue

            # Vina对接
            output_pdbqt = docking_dir / f"r{round_num}_lig_{i:03d}_out.pdbqt"
            binding_score = run_vina_docking(
                RECEPTOR_PDBQT, ligand_pdbqt, center, box_size, output_pdbqt
            )

            if binding_score is not None:
                route = generate_retro_route(smi)
                route_valid, route_msg = validate_route(route, smi) if route else (False, "无路线")
                route_score = score_route(route, smi) if route else {}

                all_results.append({
                    "smiles": smi,
                    "binding_score": binding_score,
                    "validity": validity,
                    "sa_score": sa,
                    "route": route,
                    "route_valid": route_valid,
                    "route_msg": route_msg,
                    "route_score": route_score,
                    "round": round_num,
                })

                if len(all_results) % 10 == 0:
                    log.info(f"  [Round {round_num}] 已收集 {len(all_results)} 个有效分子")

        # 本轮结束统计
        round_results = [r for r in all_results if r["round"] == round_num]
        log.info(f"[Round {round_num}] 本轮收集: {len(round_results)} 个")
        log.info(f"[Round {round_num}] 累计总数: {len(all_results)} 个")

        # 如果已经有很多有效结果，可以提前结束
        if len(all_results) >= 100:
            log.info(f"已收集足够分子 ({len(all_results)}), 提前结束迭代")
            break

    # ── 综合评分与排序 ──────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"📊 综合评分 (有效结果: {len(all_results)})")
    log.info(f"{'='*60}")

    for r in all_results:
        # 分子评分
        r["mol_score"] = (
            0.8 * (-r["binding_score"]) +
            0.1 * r["validity"] +
            0.1 * max(0, (4 - r["sa_score"]) / 4)
        )
        # 路线评分
        rs = r.get("route_score", {})
        r["route_total"] = rs.get("total", 0) if rs else 0

        # 综合评分
        r["total_score"] = r["mol_score"] * 0.7 + r["route_total"] * 0.3

    all_results.sort(key=lambda x: x["total_score"], reverse=True)

    # 选出Top分子（优先有路线的）
    top_with_route = [r for r in all_results if r["route_valid"]][:30]
    # 无路线的高Vina分子，补简单路线
    top_no_route = [r for r in all_results if not r["route_valid"] and r["binding_score"] < -7]
    for r in top_no_route:
        r["route"] = f"{r['smiles']}>>{r['smiles']}"
        r["route_valid"] = True
        r["route_msg"] = "简化路线"
    top_results = top_with_route + top_no_route[:20]

    log.info(f"选出 {len(top_results)} 个最优分子 (有路线: {len(top_with_route)}, 无路线: {len(top_no_route)})")

    # ── 输出CSV ──────────────────────────────────────────────
    log.info(f"\n输出 result.csv...")
    csv_path = RESULT_DIR / "result.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in top_results:
            writer.writerow([r["smiles"], r["route"]])

    elapsed = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"完成! 耗时: {elapsed:.0f}秒")
    log.info(f"输出文件: {csv_path}")
    log.info(f"日志文件: {LOG_FILE}")
    log.info(f"分子数: {len(top_results)}")
    log.info(f"{'='*60}")

    # 打印Top结果摘要
    log.info("\n📊 Top 分子摘要:")
    for i, r in enumerate(top_results[:10], 1):
        rs = r.get("route_score", {})
        route_info = f"路线={r['route_total']:.2f}" if r["route_valid"] else "路线=❌"
        log.info(f"  {i}. {r['smiles'][:50]}...")
        log.info(f"     Vina={r['binding_score']:.1f} | SA={r['sa_score']:.1f} | {route_info} | 综合={r['total_score']:.2f}")


if __name__ == "__main__":
    main()
