#!/usr/bin/env python3
"""
AI4S MVP: 小分子分析 + 逆合成路线规划
最小可行系统 — 验证技术栈可用性
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, rdMolDescriptors, Fragments
from rdkit.Chem import rdChemReactions
import json, os, sys
from datetime import datetime

OUTPUT = "/Users/pwngwc/ai4s_chem/mvp/output"
os.makedirs(OUTPUT, exist_ok=True)

def analyze_molecule(smiles, name="molecule"):
    """全面分析一个小分子"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}
    
    # Basic info
    info = {
        "name": name,
        "smiles": smiles,
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "iupac_name": "(需外部API)",
    }
    
    # Physical properties
    props = {
        "logP": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "heavy_atoms": Descriptors.HeavyAtomCount(mol),
    }
    
    # Drug-likeness (Lipinski Rule of 5)
    lipinski = {
        "mw_ok": info["molecular_weight"] <= 500,
        "logp_ok": props["logP"] <= 5,
        "hbd_ok": props["hbd"] <= 5,
        "hba_ok": props["hba"] <= 10,
        "passes_rule_of_5": all([
            info["molecular_weight"] <= 500,
            props["logP"] <= 5,
            props["hbd"] <= 5,
            props["hba"] <= 10,
        ])
    }
    
    # Synthetic accessibility (simplified)
    sa_score = estimate_sa_score(mol)
    
    # Functional groups
    fgroups = detect_functional_groups(mol)
    
    return {
        "basic": info,
        "properties": props,
        "lipinski": lipinski,
        "sa_score": sa_score,
        "functional_groups": fgroups,
    }

def estimate_sa_score(mol):
    """合成可及性评分 (1=easy, 10=hard)
    
    校准记录 (2026-04-30):
    - Stage5 attack seed (大平面稠环): online SA=0.081623 (极难)
    - 原模型未惩罚 fused polycyclic 高平面性结构
    - 修正：增加稠环系统惩罚，SA 越高 = 越难合成
    
    注意：此函数返回 1-10 分（越高越难），
    与 online SA score (0-1, 越高越容易) 方向相反。
    """
    score = 1.0
    
    # Ring complexity
    rings = mol.GetRingInfo()
    num_rings = rings.NumRings()
    score += num_rings * 0.5
    
    # Stereocenters
    from rdkit.Chem import FindMolChiralCenters
    chiral = len(FindMolChiralCenters(mol))
    score += chiral * 0.3
    
    # Large rings
    for ring in rings.AtomRings():
        if len(ring) > 8:
            score += 1.0
    
    # Rare elements
    rare_elements = {'B', 'Si', 'P', 'S', 'F', 'Cl', 'Br', 'I'}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in rare_elements:
            score += 0.2
    
    # === 2026-04-30 校准：大平面稠环 SA 惩罚 ===
    # Fused polycyclic 系统惩罚
    if num_rings >= 4:
        score += 2.0  # ≥4环系统合成极难
    elif num_rings >= 3:
        score += 1.0
    
    # 高平面性（通过芳香环比例判断）
    ar_rings = 0
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        if is_aromatic:
            ar_rings += 1
    if ar_rings >= 3:
        score += 1.5  # 多芳环稠合合成困难
    
    # 杂环稠合惩罚
    smi = Chem.MolToSmiles(mol)
    if 'c1ccnc' in smi or 'c1ccncc' in smi or 'c3ccnc' in smi:
        score += 0.5  # 吡啶类杂环
    
    return min(round(score, 1), 10.0)

def detect_functional_groups(mol):
    """检测常见官能团"""
    groups = []
    
    patterns = {
        "羟基(-OH)": "[OX2H]",
        "氨基(-NH2)": "[NX3;H2]",
        "羧基(-COOH)": "C(=O)[OX2H]",
        "酯基(-COO-)": "C(=O)[OX2]",
        "醛基(-CHO)": "[CH1](=O)",
        "酮基(C=O)": "[#6][CX3](=O)[#6]",
        "硝基(-NO2)": "[NX3](=O)=O",
        "磺酸基(-SO3H)": "S(=O)(=O)[OX2H]",
        "卤素(-X)": "[F,Cl,Br,I]",
        "苯环": "c1ccccc1",
        "吡啶环": "c1ccncc1",
        "呋喃环": "c1ccoc1",
        "噻吩环": "c1ccsc1",
        "酰胺键": "C(=O)[NX3]",
        "醚键(-O-)": "[OX2]([CX4])[CX4]",
    }
    
    for name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            groups.append(name)
    
    return groups

def generate_synthesis_routes(smiles):
    """基于 RDKit 逆合成分析（简化版）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    routes = []
    
    # Strategy 1: Ring disconnection
    rings = mol.GetRingInfo()
    if rings.NumRings() > 0:
        routes.append({
            "strategy": "环断裂",
            "description": "通过环化反应构建芳香环或杂环",
            "precursors": "需要邻位双官能团化合物",
            "reactions": "Suzuki偶联/Heck反应/Diels-Alder",
            "confidence": "中"
        })
    
    # Strategy 2: C-C bond formation
    routes.append({
        "strategy": "C-C偶联",
        "description": "通过Suzuki/Heck/Sonogashira偶联构建碳骨架",
        "precursors": "芳基卤化物 + 硼酸/烯烃/炔烃",
        "reactions": "Suzuki-Miyaura偶联",
        "confidence": "高"
    })
    
    # Strategy 3: Functional group interconversion
    fgroups = detect_functional_groups(mol)
    if "羟基(-OH)" in fgroups:
        routes.append({
            "strategy": "官能团转化",
            "description": "通过氧化/还原/酯化等反应引入官能团",
            "precursors": "含相应官能团的前体",
            "reactions": "酯化/酰胺化/还原",
            "confidence": "高"
        })
    
    # Strategy 4: Fragment coupling
    mw = Descriptors.MolWt(mol)
    if mw > 300:
        routes.append({
            "strategy": "片段偶联",
            "description": "将分子拆分为2-3个片段，分别合成后偶联",
            "precursors": "较小的分子片段",
            "reactions": "click chemistry/酰胺偶联/醚化",
            "confidence": "中"
        })
    
    return routes

def analyze_target_pdb(pdb_path):
    """分析靶点蛋白"""
    if not os.path.exists(pdb_path):
        return {"error": "PDB file not found"}
    
    info = {
        "file": pdb_path,
        "size_kb": round(os.path.getsize(pdb_path) / 1024, 1),
    }
    
    # Parse PDB for basic info
    with open(pdb_path, 'r') as f:
        content = f.read()
    
    # Count atoms
    atom_lines = [l for l in content.split('\n') if l.startswith('ATOM') or l.startswith('HETATM')]
    info["atom_count"] = len(atom_lines)
    
    # Get chains
    chains = set()
    for line in atom_lines:
        if len(line) > 21:
            chains.add(line[21])
    info["chains"] = sorted(list(chains))
    
    # Get residues
    residues = set()
    for line in atom_lines:
        if len(line) > 17 and line.startswith('ATOM'):
            residues.add(line[17:20].strip())
    info["residue_count"] = len(residues)
    
    # Ligands
    hetatm = [l for l in atom_lines if l.startswith('HETATM')]
    ligands = set()
    for line in hetatm:
        if len(line) > 17:
            ligands.add(line[17:20].strip())
    info["ligands"] = sorted(list(ligands - {'HOH', 'WAT'}))
    
    return info

def run_mvp():
    """运行最小可行系统"""
    print("🧪 AI4S MVP: 小分子分析 + 逆合成路线规划")
    print("="*60)
    
    # Demo molecules
    demo_molecules = {
        "阿司匹林": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "对乙酰氨基酚": "CC(=O)NC1=CC=C(O)C=C1",
        "布洛芬": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "咖啡因": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "青霉素V": "CC1(C)SC2C(NC(=O)COC3=CC=CC=C3)C(=O)N2C1C(=O)O",
    }
    
    all_results = []
    
    for name, smiles in demo_molecules.items():
        print(f"\n{'─'*40}")
        print(f"🔬 分析: {name}")
        
        result = analyze_molecule(smiles, name)
        all_results.append(result)
        
        b = result["basic"]
        p = result["properties"]
        l = result["lipinski"]
        
        print(f"  SMILES: {b['smiles']}")
        print(f"  分子式: {b['molecular_formula']}")
        print(f"  分子量: {b['molecular_weight']} g/mol")
        print(f"  LogP: {p['logP']} | TPSA: {p['tpsa']} Å²")
        print(f"  HBD: {p['hbd']} | HBA: {p['hba']}")
        print(f"  Lipinski Rule of 5: {'✅ PASS' if l['passes_rule_of_5'] else '❌ FAIL'}")
        print(f"  SA Score: {result['sa_score']}/10")
        print(f"  官能团: {', '.join(result['functional_groups'])}")
        
        # Synthesis routes
        routes = generate_synthesis_routes(smiles)
        print(f"  逆合成路线 ({len(routes)} 条):")
        for r in routes:
            print(f"    → {r['strategy']}: {r['description']}")
    
    # Analyze target PDB
    pdb_path = "/Users/pwngwc/ai4s_chem/target.pdb"
    if os.path.exists(pdb_path):
        print(f"\n{'─'*40}")
        print(f"🎯 靶点蛋白分析")
        target = analyze_target_pdb(pdb_path)
        print(f"  原子数: {target.get('atom_count', 'N/A')}")
        print(f"  链: {target.get('chains', 'N/A')}")
        print(f"  配体: {target.get('ligands', 'N/A')}")
    
    # Save results
    report = generate_report(all_results)
    report_path = os.path.join(OUTPUT, "mvp_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"✅ MVP 分析完成!")
    print(f"   报告: {report_path}")
    print(f"   分析了 {len(all_results)} 个分子")
    print(f"   虚拟环境: /Users/pwngwc/ai4s_chem/")

def generate_report(results):
    """生成分析报告"""
    lines = []
    lines.append("# AI4S MVP: 小分子分析报告\n")
    lines.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("---\n")
    
    for r in results:
        b = r["basic"]
        p = r["properties"]
        l = r["lipinski"]
        
        name = b.get("name", "unknown")
        
        lines.append(f"## {b.get('name', 'unknown')}\n")
        lines.append(f"- SMILES: `{b['smiles']}`")
        lines.append(f"- 分子式: {b['molecular_formula']}")
        lines.append(f"- 分子量: {b['molecular_weight']} g/mol")
        lines.append(f"- LogP: {p['logP']} | TPSA: {p['tpsa']} Å²")
        lines.append(f"- HBD: {p['hbd']} | HBA: {p['hba']}")
        lines.append(f"- Lipinski: {'✅' if l['passes_rule_of_5'] else '❌'}")
        lines.append(f"- SA Score: {r['sa_score']}/10")
        lines.append(f"- 官能团: {', '.join(r['functional_groups'])}")
        lines.append("")
    
    lines.append("---\n")
    lines.append("*AI4S MVP 分析报告*\n")
    
    return '\n'.join(lines)

if __name__ == "__main__":
    run_mvp()
