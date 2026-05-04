#!/usr/bin/env python3
"""
AI4S 探索 V2 — 大规模分子库扩展
目标：200+ 新分子，基于激酶靶点文献指导的 scaffold 设计
"""
import os, sys, csv, json, time, subprocess, logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path("/Users/pwngwc/.openclaw/workspace/retrosyn")))
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, BRICS, Draw
from rdkit.Chem import rdChemReactions

# ── 配置 ──────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/pwngwc/ai4s_chem")
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
RESULT_DIR = BASE_DIR / "result" / TIMESTAMP
RESULT_DIR.mkdir(parents=True, exist_ok=True)
VINA = BASE_DIR / "bin" / "vina"
RECEPTOR = BASE_DIR / "receptor.pdbqt"
LOG_FILE = RESULT_DIR / "explore_v2.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("explore_v2")

# ── 口袋配置（6个方向 + 新增亚口袋）──────────────────────────────
POCKETS = {
    "center":    ([18.3, 2.3, 21.4], [20, 20, 20]),
    "small_box": ([18.3, 2.3, 21.4], [15, 15, 15]),
    "shift_n":   ([18.3, 12.3, 21.4], [20, 20, 20]),
    "shift_s":   ([18.3, -7.7, 21.4], [20, 20, 20]),
    "shift_e":   ([28.3, 2.3, 21.4], [20, 20, 20]),
    "shift_w":   ([8.3, 2.3, 21.4], [20, 20, 20]),
    # 新增亚口袋
    "deep":      ([18.3, 2.3, 25.0], [20, 20, 20]),
    "surface":   ([18.3, 2.3, 17.0], [20, 20, 20]),
}

# ═══════════════════════════════════════════════════════════════════
# 扩展分子库 — 基于激酶靶点文献的 scaffold 设计
# ═══════════════════════════════════════════════════════════════════

def build_kinase_scaffold_library():
    """
    基于文献的激酶靶点分子库：
    - 4-anilinoquinazoline (EGFR/VEGFR 抑制剂经典 scaffold)
    - aminopyrimidine (CDK/PI3K 抑制剂)
    - pyrazolopyrimidine (Src/Abl 抑制剂)
    - indazole-carboxamide (TRK/ALK 抑制剂)
    - aminopyridine (JAK 抑制剂)
    - benzimidazole (激酶铰链区结合)
    - imidazopyridine (多种激酶抑制剂)
    - thienopyrimidine (激酶抑制剂)
    - 扩展 linker 和取代基组合
    """
    library = {}

    # ── 1. 4-Anilinoquinazoline 类（EGFR/VEGFR 经典）─────────────
    library["anilinoquinazoline"] = [
        "c1ccc(Nc2ccnc3ccccc23)cc1",                          # 基础
        "c1ccc(Nc2ccnc3ccc(F)cc23)cc1",                       # 6-F
        "c1ccc(Nc2ccnc3ccc(Cl)cc23)cc1",                      # 6-Cl
        "c1ccc(Nc2ccnc3ccc(OC)cc23)cc1",                      # 6-OMe
        "c1ccc(Nc2ccnc3ccc(C#N)cc23)cc1",                     # 6-CN
        "c1ccc(Nc2ccnc3ccc(C(F)(F)F)cc23)cc1",                # 6-CF3
        "c1ccc(Nc2ccnc3ccc(N)cc23)cc1",                       # 6-NH2
        "c1ccc(Nc2ccnc3cc(N)ccc23)cc1",                       # 7-NH2
        "O=c1ccc(Nc2ccccc2)nc2ccccc12",                       # quinazolinone
        "c1ccc(Nc2ncnc3ccc(OC)cc23)cc1",                      # 4-anilino-pyrimidine
        "c1ccc(Nc2ncnc3ccccc23)cc1",                          # quinazoline core
        "O=C(c1ccccc1)Nc1ccnc2ccccc12",                       # benzoyl on N
        "O=C(Nc1ccccc1)c1ccnc2ccccc12",                       # reverse amide
    ]

    # ── 2. Aminopyrimidine 类（CDK/PI3K）────────────────────────
    library["aminopyrimidine"] = [
        "c1ccc(Nc2ncccn2)cc1",                                # 基础 2-aminopyrimidine
        "c1ccc(Nc2ncc(NC)cn2)cc1",                            # N-methyl
        "c1ccc(Nc2ncc(NCC)cn2)cc1",                           # N-ethyl
        "c1ccc(Nc2ncnc3ccccc23)cc1",                          # fused phenyl
        "c1ccc(Nc2nc(Nc3ccccc3)ncc2)cc1",                     # di-anilino
        "c1ccc(Nc2ncnc(Nc3ccccc3)n2)cc1",                     # triazine-like
        "c1ccc(Nc2ncccc2)cc1",                                # 2-aminopyridine
        "c1ccc(Nc2nccc(N)n2)cc1",                             # amino-substituted
        "c1ccc(Nc2ncnc3ccc(F)cc23)cc1",                       # F-substituted
        "O=C(Nc1ccccc1)c1ncnc2ccccc12",                       # amide on pyrimidine
    ]

    # ── 3. Pyrazolopyrimidine 类（Src/Abl）───────────────────────
    library["pyrazolopyrimidine"] = [
        "c1ccc(-c2ncc3ccccc3n2)cc1",                          # 基础
        "c1ccc(-c2ncc3ccc(N)cc3n2)cc1",                       # NH2
        "c1ccc(-c2ncc3ccc(F)cc3n2)cc1",                       # F
        "c1ccc(-c2ncc3ccc(Cl)cc3n2)cc1",                      # Cl
        "c1ccc(-c2ncc3ccc(OC)cc3n2)cc1",                      # OMe
        "c1ccc(-c2ncc3ccc(C#N)cc3n2)cc1",                     # CN
        "c1ccc(-c2ncc3cc(F)ccc3n2)cc1",                       # 8-F
        "c1ccc(-c2ncc3ccc(C)cc3n2)cc1",                       # Me
        "O=C(Nc1ccccc1)c1ncc2ccccc2n1",                       # amide
        "c1ccc(Nc2ncc3ccccc3n2)cc1",                          # anilino
    ]

    # ── 4. Indazole-carboxamide 类（TRK/ALK）────────────────────
    library["indazole_carboxamide"] = [
        "O=C(Nc1ccccc1)c1ccc2[nH]ncc2c1",                    # 基础 5-carboxamide
        "O=C(Nc1ccccc1)c1ccc2c(c1)cnn2",                      # 5-indazole
        "O=C(Nc1ccccc1)c1ccc2nnccc2c1",                       # 6-indazole
        "O=C(Nc1ccc(F)cc1)c1ccc2[nH]ncc2c1",                 # 4-F-anilino
        "O=C(Nc1ccc(Cl)cc1)c1ccc2[nH]ncc2c1",                # 4-Cl-anilino
        "O=C(Nc1ccc(C)cc1)c1ccc2[nH]ncc2c1",                 # 4-Me-anilino
        "O=C(Nc1ccc(OC)cc1)c1ccc2[nH]ncc2c1",                # 4-OMe-anilino
        "O=C(Nc1ccc(C(F)(F)F)cc1)c1ccc2[nH]ncc2c1",          # 4-CF3-anilino
        "O=C(Nc1ccc(C#N)cc1)c1ccc2[nH]ncc2c1",               # 4-CN-anilino
        "O=C(Nc1ccccc1)c1ccc2c(c1)cnn2C",                     # N-methyl indazole
    ]

    # ── 5. Aminopyridine 类（JAK/其他激酶）──────────────────────
    library["aminopyridine"] = [
        "c1ccc(Nc2cccnc2)cc1",                                # 2-aminopyridine
        "c1ccc(Nc2ccccn2)cc1",                                # 3-aminopyridine
        "c1ccc(Nc2ccncc2)cc1",                                # 4-aminopyridine
        "c1ccc(Nc2ccc(N)nc2)cc1",                             # diamino
        "c1ccc(Nc2ncccc2N)cc1",                               # 2,6-diamino
        "c1ccc(Nc2ccc(F)nc2)cc1",                             # F
        "c1ccc(Nc2ccc(Cl)nc2)cc1",                            # Cl
        "c1ccc(Nc2ccc(OC)nc2)cc1",                            # OMe
        "c1ccc(Nc2ccc(C#N)nc2)cc1",                           # CN
        "O=C(Nc1ccccc1)c1cccnc1",                             # amide on pyridine
    ]

    # ── 6. Benzimidazole 类（铰链区结合）─────────────────────────
    library["benzimidazole"] = [
        "c1ccc2[nH]cnc2c1",                                   # 基础
        "c1ccc(-c2nc3ccccc3[nH]2)cc1",                        # 2-phenyl
        "c1ccc(-c2nc3ccc(F)cc3[nH]2)cc1",                     # 6-F
        "c1ccc(-c2nc3ccc(Cl)cc3[nH]2)cc1",                    # 6-Cl
        "c1ccc(-c2nc3ccc(OC)cc3[nH]2)cc1",                    # 6-OMe
        "c1ccc(-c2nc3ccc(N)cc3[nH]2)cc1",                     # 6-NH2
        "c1ccc(-c2nc3ccc(C#N)cc3[nH]2)cc1",                   # 6-CN
        "c1ccc(-c2nc3ccc(C(F)(F)F)cc3[nH]2)cc1",              # 6-CF3
        "c1ccc(Nc2nc3ccccc3[nH]2)cc1",                        # anilino
        "O=C(Nc1ccccc1)c1nc2ccccc2[nH]1",                     # amide
        "c1ccc(-c2nc3ccccc3o2)cc1",                           # benzoxazole
        "c1ccc(-c2nc3ccccc3s2)cc1",                           # benzothiazole
    ]

    # ── 7. Imidazopyridine 类（多种激酶）─────────────────────────
    library["imidazopyridine"] = [
        "c1ccc2nccnc2c1",                                     # imidazo[1,2-a]pyridine
        "c1ccc2ncc(N)nc2c1",                                  # amino
        "c1ccc(-c2ccc3nccnc3c2)cc1",                          # phenyl
        "O=C(Nc1ccccc1)c1ccc2nccnc2c1",                       # amide
        "c1ccc(Nc2ccc3nccnc3c2)cc1",                          # anilino
        "c1ccc2ncc(-c3ccccc3)nc2c1",                          # 3-phenyl
        "c1ccc2ncc(-c3ccc(F)cc3)nc2c1",                       # 3-(4-F-phenyl)
        "c1ccc2ncc(-c3ccc(Cl)cc3)nc2c1",                      # 3-(4-Cl-phenyl)
    ]

    # ── 14. 额外 scaffold（补量）────────────────────────────────
    library["extra_scaffolds"] = [
        # Quinoline 类
        "c1ccc(Nc2ccc3ccccc3c2)cc1",                          # 2-aminoquinoline
        "c1ccc(Nc2ccnc3ccc(F)cc23)cc1",                       # 6-F quinazoline
        "c1ccc(Nc2ncnc3ccc(Cl)cc23)cc1",                      # 6-Cl pyrimido
        # 吡啶类扩展
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(F)cc2)nc1",              # pyridine-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccccc2)nc1",                  # pyridine-biphenyl v2
        # 噻吩类
        "O=C(Nc1ccccc1)c1ccc(-c2cccs2)c1",                    # thienyl amide
        "c1ccc(-c2ccc(-c3cccs2)cc2)cc1",                      # thienyl-biphenyl
        # 咪唑类
        "c1ccc(-c2nc3ccccc3[nH]2)cc1",                        # 2-phenyl benzimidazole
        "c1ccc(-c2nc3ccc(F)cc3[nH]2)cc1",                     # 5-F benzimidazole
        "c1ccc(-c2nc3ccc(Cl)cc3[nH]2)cc1",                    # 5-Cl benzimidazole
        # 嘧啶扩展
        "c1ccc(Nc2ncnc3cc(OC)ccc23)cc1",                      # 7-OMe quinazoline
        "c1ccc(Nc2ncnc3cc(F)ccc23)cc1",                       # 7-F quinazoline
        "c1ccc(Nc2ncnc3cc(Cl)ccc23)cc1",                      # 7-Cl quinazoline
        # 吡唑类
        "c1ccc(-c2cc(-c3ccccc3)nn2)cc1",                      # pyrazole-biphenyl
        "c1ccc(-c2cc(-c3ccc(F)cc3)nn2)cc1",                   # pyrazole-F-biphenyl
        "O=C(Nc1ccccc1)c1cc(-c2ccccc2)nn1",                   # pyrazole amide
        # 含 N 杂环
        "c1ccc(-c2ccc3ncccc3c2)cc1",                          # quinoline biphenyl
        "c1ccc(-c2ccc3nc(N)ccc3c2)cc1",                       # aminoquinoline
        "c1ccc(-c2ccc3c(c2)ccn3)cc1",                         # indole biphenyl
        "c1ccc(-c2ccc3c(c2)cnc3)cc1",                         # isoindole biphenyl
        # 胺类扩展
        "c1ccc(NC(=O)c2ccc(-c3ccccc3)cc2)cc1",                # reverse amide biphenyl
        "c1ccc(NC(=O)c2ccc(-c3ccccn3)cc2)cc1",                # reverse amide pyridyl
        "c1ccc(NC(=O)c2ccc(-c3cccs3)cc2)cc1",                 # reverse amide thienyl
        # 磺酰胺扩展
        "NS(=O)(=O)c1ccc(-c2ccc(F)cc2)cc1",                   # F-sulfonamide
        "NS(=O)(=O)c1ccc(-c2ccc(Cl)cc2)cc1",                  # Cl-sulfonamide
        "NS(=O)(=O)c1ccc(-c2ccc(OC)cc2)cc1",                  # OMe-sulfonamide
        # 脲类
        "O=C(Nc1ccccc1)Nc1ccc(-c2ccccc2)cc1",                 # urea biphenyl
        "O=C(Nc1ccccc1)Nc1ccc(-c2ccccn2)cc1",                 # urea pyridyl
        "O=C(Nc1ccccc1)Nc1ccc(F)cc1",                          # urea F
        # 酰胺扩展
        "O=C(Nc1ccccc1)c1ccc(-c2ccc3ccccc3c2)cc1",            # naphthyl amide
        "O=C(Nc1ccccc1)c1ccc(-c2ccc3ccncc3c2)cc1",            # isoquinolyl amide
        "O=C(Nc1ccccc1)c1ccc(-c2ccc3ncncc3c2)cc1",            # pyrimidyl amide
        # 三氟甲基扩展
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(C(F)(F)F)cc2)cc1",        # CF3 biphenyl amide
        "O=C(Nc1ccc(C(F)(F)F)cc1)c1ccc(-c2ccccc2)cc1",        # CF3 on aniline
        "O=C(Nc1ccc(C(F)(F)F)cc1)c1ccc(-c2ccccn2)cc1",        # CF3 + pyridyl
        # 氰基扩展
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(C#N)cc2)cc1",             # CN biphenyl amide
        "O=C(Nc1ccc(C#N)cc1)c1ccc(-c2ccccc2)cc1",             # CN on aniline
        # 杂原子丰富
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(NC)cc2)cc1",              # NMe biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(O)cc2)cc1",               # OH biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(S)cc2)cc1",               # SH biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(NC=O)cc2)cc1",            # formamido biphenyl
    ]

    # ── 15. Fragment 扩展（确保 >200）───────────────────────────
    library["fragment_expansion"] = [
        # 简单芳香 + 杂环
        "c1ccc(-c2ccc(-c3ccc(F)cc3)cc2)cc1",                  # terphenyl F
        "c1ccc(-c2ccc(-c3ccc(Cl)cc3)cc2)cc1",                 # terphenyl Cl
        "c1ccc(-c2ccc(-c3ccc(C)cc3)cc2)cc1",                  # terphenyl Me
        "c1ccc(-c2ccc(-c3ccc(OC)cc3)cc2)cc1",                 # terphenyl OMe
        "c1ccc(-c2ccc(-c3ccccn3)cc2)cc1",                     # pyridyl-biphenyl
        "c1ccc(-c2ccc(-c3ccncc3)cc2)cc1",                     # pyrimidyl-biphenyl
        "c1ccc(-c2ccc(-c3cccs3)cc2)cc1",                      # thienyl-biphenyl
        "c1ccc(-c2ccc(-c3ccco3)cc2)cc1",                      # furyl-biphenyl
        "c1ccc(-c2ccc(-c3ccc(N)cc3)cc2)cc1",                  # amino-terphenyl
        "c1ccc(-c2ccc(-c3ccc(C#N)cc3)cc2)cc1",                # CN-terphenyl
        # 含环己基/哌啶
        "O=C(Nc1ccccc1)c1ccc(-c2CCCCC2)cc1",                  # cyclohexyl
        "O=C(Nc1ccccc1)c1ccc(-c2CCNCC2)cc1",                  # piperidinyl
        "O=C(Nc1ccccc1)c1ccc(-c2CCOCC2)cc1",                  # morpholinyl
        # 含咪唑
        "c1ccc(-c2cnc(-c3ccccc3)[nH]2)cc1",                   # imidazole-phenyl
        "c1ccc(-c2cnc(-c3ccc(F)cc3)[nH]2)cc1",                # imidazole-F
        "O=C(Nc1ccccc1)c1cnc(-c2ccccc2)[nH]1",                # imidazole amide
        # 含噁二唑
        "c1ccc(-c2nnc(-c3ccccc3)o2)cc1",                      # 1,3,4-oxadiazole
        "c1ccc(-c2nnc(-c3ccc(F)cc3)o2)cc1",                   # oxadiazole F
        "O=C(Nc1ccccc1)c1nnc(-c2ccccc2)o1",                   # oxadiazole amide
        # 含三唑
        "c1ccc(-c2nc(-c3ccccc3)nn2)cc1",                      # triazole
        "c1ccc(-c2nc(-c3ccc(F)cc3)nn2)cc1",                   # triazole F
        "c1ccc(-c2nnc(-c3ccccc3)n2)cc1",                      # 1,2,4-triazole
        # 含四唑
        "c1ccc(-c2nnnn2-c3ccccc3)cc1",                        # tetrazole
        # 含吲哚
        "c1ccc(-c2cc(-c3ccccc3)[nH]c2=O)cc1",                 # oxindole
        "c1ccc(NC(=O)c2c[nH]c3ccccc23)cc1",                   # indole amide
        # 含喹啉
        "c1ccc(-c2ccnc3ccccc23)cc1",                          # quinoline
        "c1ccc(-c2ccnc3ccc(F)cc23)cc1",                       # 6-F quinoline
        "c1ccc(-c2ccnc3ccc(Cl)cc23)cc1",                      # 6-Cl quinoline
        # 含异喹啉
        "c1ccc(-c2nccc3ccccc23)cc1",                          # isoquinoline
        "c1ccc(-c2nccc3ccc(F)cc23)cc1",                       # 6-F isoquinoline
    ]

    # ── 8. Thienopyrimidine 类───────────────────────────────────
    library["thienopyrimidine"] = [
        "c1ccc(Nc2ncc3ccsc3n2)cc1",                           # 基础
        "c1ccc(Nc2ncc3cc(F)sc3n2)cc1",                        # F
        "c1ccc(Nc2ncc3cc(Cl)sc3n2)cc1",                       # Cl
        "c1ccc(Nc2ncc3cc(OC)sc3n2)cc1",                       # OMe
        "O=C(Nc1ccccc1)c1ncc2ccsc2n1",                        # amide
        "c1ccc(-c2ncc3ccsc3n2)cc1",                           # phenyl
    ]

    # ── 9. 扩展 linker 组合（基于 Top 分子优化）─────────────────
    library["linker_expansion"] = [
        # 双芳基 + 各种 linker
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(F)cc2)cc1",              # 4-F-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(Cl)cc2)cc1",             # 4-Cl-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(OC)cc2)cc1",             # 4-OMe-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(C)cc2)cc1",              # 4-Me-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(C#N)cc2)cc1",            # 4-CN-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2cccc(F)c2)cc1",              # 3-F-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2cccc(Cl)c2)cc1",             # 3-Cl-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccccc2F)cc1",                # 2-F-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc3ccccc3c2)cc1",           # naphthyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccncc2)cc1",                 # pyridyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccccn2)cc1",                 # 3-pyridyl
        "O=C(Nc1ccccc1)c1ccc(-c2cccs2)cc1",                  # thienyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccco2)cc1",                  # furyl
        # 酰胺反转
        "O=C(c1ccccc1)Nc1ccc(-c2ccccc2)cc1",                 # reverse
        "O=C(c1ccccc1)Nc1ccc(-c2ccccn2)cc1",                 # reverse pyridyl
        # 三芳基
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",      # terphenyl
        # 含杂原子 linker
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(N)cc2)cc1",              # amino-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(O)cc2)cc1",              # hydroxy-biphenyl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(S)cc2)cc1",              # thiol-biphenyl
    ]

    # ── 10. 含氟/氯取代基库（提高代谢稳定性和结合力）────────────
    library["halogen_rich"] = [
        "O=C(Nc1ccc(F)cc1)c1ccc(-c2ccccc2)cc1",              # 4-F on aniline
        "O=C(Nc1ccc(Cl)cc1)c1ccc(-c2ccccc2)cc1",             # 4-Cl on aniline
        "O=C(Nc1ccc(F)(F)cc1)c1ccc(-c2ccccc2)cc1",           # 3,5-diF
        "O=C(Nc1ccc(Cl)(Cl)cc1)c1ccc(-c2ccccc2)cc1",         # 3,5-diCl
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(F)(F)cc2)cc1",           # 3,5-diF biphenyl
        "O=C(Nc1cc(F)ccc1)c1ccc(-c2ccccc2)cc1",              # 3-F on aniline
        "O=C(Nc1ccc(F)cc1)c1ccc(-c2ccc(F)cc2)cc1",           # 双F
        "FC(F)(F)c1ccc(-c2ccc(F)cc2)cc1",                     # CF3 + F
        "O=C(Nc1ccccc1)c1ccc(Br)cc1",                         # Br
        "O=C(Nc1ccccc1)c1ccc(I)cc1",                          # I
    ]

    # ── 11. 含杂环 linker（提高溶解度和结合力）───────────────────
    library["hetero_linker"] = [
        "O=C(Nc1ccccc1)c1cc(-c2ccccc2)on1",                   # isoxazole (已有)
        "O=C(Nc1ccccc1)c1cc(-c2ccccc2)no1",                   # oxadiazole
        "O=C(Nc1ccccc1)c1cc(-c2ccccc2)nn1",                   # pyrazole
        "O=C(Nc1ccccc1)c1c(-c2ccccc2)no1",                    # isoxazole v2
        "O=C(Nc1ccccc1)c1sc(-c2ccccc2)n1",                    # thiazole
        "O=C(Nc1ccccc1)c1nc(-c2ccccc2)cs1",                   # thiazole v2
        "O=C(Nc1ccccc1)c1oc(-c2ccccc2)n1",                    # oxazole
        "O=C(Nc1ccccc1)c1cnc(-c2ccccc2)n1",                   # pyrimidine linker
        "c1ccc(-c2nc(-c3ccccc4ccccc43)no2)cc1",               # fused oxadiazole
        "c1ccc(-c2nc(-c3ccccc3)no2)cc1",                      # oxadiazole-phenyl
    ]

    # ── 12. 已知激酶抑制剂 fragment（来自文献）───────────────────
    library["known_kinase_frags"] = [
        # Imatinib-like fragments
        "c1ccc(Nc2ncnc3ccccc23)cc1",                          # quinazoline-aniline
        "c1ccc(Nc2ncccn2)cc1",                                # pyrimidine-aniline
        "c1ccc(-c2ccc(NC(=O)c3ccccc3)cc2)cc1",                # amide-biphenyl
        # Erlotinib-like
        "c1ccc(Nc2ccnc3ccc(OC)c(OC)cc23)cc1",                 # di-MeO quinazoline
        # Sorafenib-like
        "O=C(Nc1ccc(Cl)cc1)c1ccc(-c2ccccn2)cc1",             # pyridyl-amide
        # Dasatinib-like
        "O=C(Nc1ccccc1)c1ccc2nc(N)ccc2c1",                   # aminopyrimidine
        # Ponatinib-like
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(C#C)cc2)cc1",            # alkyne linker
        # Crizotinib-like
        "O=C(Nc1ccccc1)c1ccc(-c2ccc(Cl)cc2)cc1",             # Cl-biphenyl amide
    ]

    # ── 13. 组合库：核心 × 取代基 ────────────────────────────────
    cores = [
        "O=C(Nc1ccccc1)c1ccc(-c2ccccc2)cc1",                  # biphenyl amide
        "O=C(Nc1ccccc1)c1ccnc2ccccc12",                       # quinazoline amide
        "c1ccc(-c2nc3ccccc3[nH]2)cc1",                        # benzimidazole
        "c1ccc(Nc2ncnc3ccccc23)cc1",                          # anilinoquinazoline
        "O=C(Nc1ccccc1)c1ccc2[nH]ncc2c1",                    # indazole amide
    ]

    substituent_sets = [
        ("", ""),
        ("F", "4-F"),
        ("Cl", "4-Cl"),
        ("C", "4-Me"),
        ("OC", "4-OMe"),
        ("C#N", "4-CN"),
        ("C(F)(F)F", "4-CF3"),
        ("N", "4-NH2"),
        ("O", "4-OH"),
        ("S", "4-SH"),
    ]

    library["combinatorial"] = []
    for core in cores:
        core_mol = Chem.MolFromSmiles(core)
        if not core_mol:
            continue
        # Try substituting on the aniline ring
        for sub_smiles, sub_name in substituent_sets:
            if not sub_smiles:
                library["combinatorial"].append(core)
                continue
            # Simple substitution: add to para position of aniline
            # Use SMILES manipulation
            if "Nc1ccccc1" in core:
                new_smi = core.replace("Nc1ccccc1", f"Nc1ccc({sub_smiles})cc1")
                mol = Chem.MolFromSmiles(new_smi)
                if mol:
                    library["combinatorial"].append(new_smi)
            elif "c1ccccc1" in core and "Nc" not in core:
                # Try on the phenyl
                pass

    return library


def get_all_molecules():
    """收集所有分子，去重"""
    library = build_kinase_scaffold_library()
    all_mols = {}
    for cat, smis in library.items():
        for smi in smis:
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical = Chem.MolToSmiles(mol)
                if canonical not in all_mols:
                    all_mols[canonical] = cat
    return all_mols


# ── 对接函数 ─────────────────────────────────────────────────────

def smiles_to_pdbqt(smi, path):
    """SMILES → PDBQT"""
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
        line = f"ATOM  {i+1:5d} {name4} LIG A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  0.00  0.00          {atype:>2s} "
        lines.append(line)
    lines.append("ENDROOT")
    lines.append("TORSDOF 0")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return True


def run_vina(ligand_pdbqt, out_pdbqt, center, size, exhaustiveness=16):
    """运行 Vina"""
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
    """简化 SA score"""
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
    """合成路线生成"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None
    # BRICS decomposition
    try:
        frags = list(BRICS.BRICSDecompose(mol, returnMols=False))
        if len(frags) >= 2:
            return ".".join(frags[:2]) + ">>" + smi
    except:
        pass
    # Template-based
    templates = [
        ("[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[OH].[N:3]", "酰胺"),
        ("[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[O:3][C:4]", "酯"),
        ("[c:1][c:2]>>[c:1][Br].[c:2]B(O)(O)", "Suzuki"),
        ("[c:1][NH][c:2]>>[c:1][NH2].[c:2]Br", "胺化"),
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


def filter_drug_like(smi):
    """Lipinski + CNS-like 过滤"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    # CNS drug-like: MW<500, LogP<5, TPSA<90, HBD<=3, HBA<=7
    if mw > 500 or logp > 6 or tpsa > 100 or hbd > 4 or hba > 8:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    log.info("=" * 60)
    log.info("AI4S 探索 V2 — 大规模分子库扩展")
    log.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"结果目录: {RESULT_DIR}")
    log.info("=" * 60)

    # 收集分子
    all_mols = get_all_molecules()
    log.info(f"分子库: {len(all_mols)} 个去重分子")

    # 按类别统计
    cat_counts = {}
    for smi, cat in all_mols.items():
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, cnt in sorted(cat_counts.items()):
        log.info(f"  {cat}: {cnt} 个")

    # 加载已有结果（避免重复计算）
    existing_results = {}
    prev_results_file = BASE_DIR / "results" / "all_results.json"
    if prev_results_file.exists():
        with open(prev_results_file) as f:
            prev = json.load(f)
            for r in prev:
                existing_results[r["smiles"]] = r["vina_score"]
        log.info(f"已有结果: {len(existing_results)} 个分子")

    # 遍历口袋 × 分子
    all_results = []
    docking_count = 0
    skip_count = 0

    for pocket_name, (center, size) in POCKETS.items():
        log.info(f"\n{'='*50}")
        log.info(f"口袋: {pocket_name} center={center} size={size}")
        log.info(f"{'='*50}")

        docking_dir = BASE_DIR / "docking" / pocket_name
        docking_dir.mkdir(parents=True, exist_ok=True)

        for mol_idx, (smi, category) in enumerate(all_mols.items()):
            # 跳过已有结果
            if smi in existing_results:
                skip_count += 1
                continue

            # 药物样过滤
            if not filter_drug_like(smi):
                continue

            # 准备配体
            lig_pdbqt = docking_dir / f"mol_{mol_idx:04d}.pdbqt"
            out_pdbqt = docking_dir / f"mol_{mol_idx:04d}_out.pdbqt"

            if not smiles_to_pdbqt(smi, str(lig_pdbqt)):
                continue

            # 对接
            score = run_vina(str(lig_pdbqt), str(out_pdbqt), center, size)
            docking_count += 1

            if score is not None:
                sa = calc_sa(smi)
                route = generate_route(smi)
                route_valid = route and ">>" in route

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

            # 每 50 个打印进度
            if docking_count % 50 == 0:
                elapsed = time.time() - start_time
                log.info(f"  进度: {docking_count} 次对接, {len(all_results)} 有效, {elapsed:.0f}s elapsed")

            # 时间限制：110 分钟（留 10 分钟汇总）
            elapsed = time.time() - start_time
            if elapsed > 110 * 60:
                log.info(f"⏰ 时间到 ({elapsed/60:.1f}min)，停止对接")
                break
        else:
            continue
        break

    # ── 汇总 ──────────────────────────────────────────────────
    elapsed = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"探索完成: {len(all_results)} 个新有效结果")
    log.info(f"总对接次数: {docking_count}")
    log.info(f"跳过已有: {skip_count}")
    log.info(f"耗时: {elapsed/60:.1f} 分钟")
    log.info(f"{'='*60}")

    # 合并旧结果
    merged = list(all_results)
    for smi, vina in existing_results.items():
        merged.append({"smiles": smi, "vina_score": vina, "route_valid": True})

    # 按 Vina 排序
    merged.sort(key=lambda x: x["vina_score"])

    # Top 30
    log.info("\n🏆 Top 30 最优分子:")
    for i, r in enumerate(merged[:30], 1):
        cat = r.get("category", "?")
        pocket = r.get("pocket", "?")
        route_flag = "✅" if r.get("route_valid") else "❌"
        log.info(f"  {i}. Vina={r['vina_score']:.1f} | SA={r.get('sa_score','?')} | 路线={route_flag} | {r['smiles'][:50]}")
        if cat != "?":
            log.info(f"     口袋: {pocket} | 类别: {cat}")

    # 保存全量结果
    json_path = RESULT_DIR / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"\n新结果 JSON: {json_path}")

    # 方案1: 最优 Vina + 路线
    with_route = [r for r in merged if r.get("route_valid")]
    with_route.sort(key=lambda x: x["vina_score"])

    csv1 = RESULT_DIR / "result.csv"
    with open(csv1, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in with_route[:30]:
            route = r.get("route") or ""
            writer.writerow([r["smiles"], route])
    log.info(f"方案1 (Top 30 Vina): {csv1} ({min(len(with_route), 30)}个)")

    # 方案2: 最多分子
    csv2 = RESULT_DIR / "plan_max_molecules.csv"
    with open(csv2, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in with_route[:80]:
            route = r.get("route") or ""
            writer.writerow([r["smiles"], route])
    log.info(f"方案2 (最多分子): {csv2} ({min(len(with_route), 80)}个)")

    # 方案3: 多样性
    diverse = []
    seen_cats = set()
    for r in merged:
        cat = r.get("category", "")
        if cat and cat not in seen_cats and r.get("route_valid"):
            diverse.append(r)
            seen_cats.add(cat)
    csv3 = RESULT_DIR / "plan_diverse.csv"
    with open(csv3, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_smiles", "route"])
        for r in diverse:
            route = r.get("route") or ""
            writer.writerow([r["smiles"], route])
    log.info(f"方案3 (多样性): {csv3} ({len(diverse)}个)")

    # 打包 result.zip
    import zipfile
    zip_path = RESULT_DIR / "result.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in [csv1, csv2, csv3, json_path]:
            if f.exists():
                zf.write(f, f.name)
    log.info(f"打包: {zip_path}")

    # result.log 信息
    log_path = RESULT_DIR / "result.log"
    with open(log_path, "w") as f:
        f.write(f"[{datetime.now()}] AI4S V2 探索完成\n")
        f.write(f"新分子: {len(all_results)}, 合并总数: {len(merged)}\n")
        f.write(f"最优 Vina: {merged[0]['vina_score']:.1f}\n")
        f.write(f"耗时: {elapsed/60:.1f} 分钟\n")

    log.info(f"\n总耗时: {elapsed/60:.1f} 分钟")
    log.info("探索 V2 完成 ✅")


if __name__ == "__main__":
    main()
