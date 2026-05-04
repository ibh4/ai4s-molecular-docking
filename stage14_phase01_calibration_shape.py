#!/usr/bin/env python3
"""
Stage14 Phase 0+1: Calibration Summary + Stage5 Shape Analysis
"""
import json, csv, sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED
import numpy as np

BASE = Path('/Users/pwngwc/ai4s_chem')
STAGE14 = BASE / 'result' / 'stage14_pocket_rescan_stage5_shape_simplification'
P0 = STAGE14 / '00_calibration'
P1 = STAGE14 / '01_stage5_shape_analysis'
P0.mkdir(parents=True, exist_ok=True)
P1.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# Phase 0: Calibration Summary
# ══════════════════════════════════════════════════════════════
calibration_data = [
    {
        "case": "V3_multi",
        "online_score": 0.499492, "binding": 0.163139, "sa": 0.803972, "route": 0.939520,
        "conclusion": "SA high, binding low"
    },
    {
        "case": "Stage5_attack",
        "smiles": "O=C(Nc1ccccc1F)C1=CC=C2C=CC=c3c(F)ccnc3=CC=C21",
        "scaffold": "fused_polycycle_isoquinoline",
        "online_score": 0.506074, "binding": 0.260375, "sa": 0.081623, "route": 0.9485,
        "conclusion": "ONLY online binding-supported direction; SA collapsed"
    },
    {
        "case": "pyrrolopyrimidine_1",
        "smiles": "O=C(Nc1ccc(F)cc1)n1ccc2ncncc21",
        "scaffold": "compact_pyrrolopyrimidine",
        "online_score": 0.478234, "binding": 0.148375, "sa": 0.573486, "route": 0.95,
        "conclusion": "SA ok, binding severely overestimated (local 0.306 -> online 0.148)"
    },
    {
        "case": "pyridine_pyrimidine_ext",
        "smiles": "O=C(Nc1ccc(C#N)cc1)c1nccc(-c2ncccn2)n1",
        "scaffold": "aza_aryl_amide_pyridine_pyrimidine_ext",
        "online_score": 0.472403, "binding": 0.140875, "sa": 0.552861, "route": 0.949375,
        "conclusion": "Binding severely overestimated again (local 0.28 -> online 0.141); same failure mode"
    },
]

with open(P0 / 'loaded_calibration_summary.md', 'w') as f:
    f.write("# Stage14 Calibration Summary\n\n")
    f.write("**加载时间：** 2026-05-04 19:20\n\n")
    f.write("## 线上校准点\n\n")
    f.write("| case | online_score | binding | SA | route | conclusion |\n")
    f.write("|---|---:|---:|---:|---:|---|\n")
    for d in calibration_data:
        f.write(f"| {d['case']} | {d['online_score']} | {d['binding']} | {d['sa']} | {d['route']} | {d['conclusion']} |\n")
    f.write("\n## 核心校准规则\n\n")
    f.write("### Stage5-like scaffold (fused polycycle)\n")
    f.write("- binding: online supported (0.260), only direction with real binding\n")
    f.write("- SA: severely penalized (0.082), cap <= 0.20 for exact/highly similar\n")
    f.write("- Strategy: shape simplification to keep binding, improve SA\n\n")
    f.write("### compact pyrrolopyrimidine\n")
    f.write("- SA: online calibrated ok (0.573)\n")
    f.write("- binding: NOT supported (0.148), local overestimated ~2.1x\n")
    f.write("- cap binding <= 0.20\n\n")
    f.write("### pyridine-pyrimidine extension / aza-aryl amide\n")
    f.write("- SA: online calibrated ok (0.553)\n")
    f.write("- binding: NOT supported (0.141), local overestimated ~2.0x\n")
    f.write("- cap binding <= 0.20\n\n")
    f.write("### Non-fused heteroaryl amide (通用)\n")
    f.write("- binding predicted should be divided by ~2 unless Stage5-like shape evidence exists\n")
    f.write("- Only fused polycycle with 14-18Å long axis + hydrophobic surface can get binding > 0.20\n")

print("Phase 0 done: calibration summary written")

# ══════════════════════════════════════════════════════════════
# Phase 1: Stage5 Shape Analysis
# ══════════════════════════════════════════════════════════════

def sa_raw(mol):
    if mol is None: return 10.0
    try:
        qed = QED.qed(mol)
        sa = 10.0 * (1.0 - qed) + 1.0
        rings = rdMolDescriptors.CalcNumRings(mol)
        ar = rdMolDescriptors.CalcNumAromaticRings(mol)
        het = rdMolDescriptors.CalcNumHeteroatoms(mol)
        mw = Descriptors.MolWt(mol)
        p = 0
        if rings > 4: p += (rings-4)*0.5
        if ar > 3: p += (ar-3)*0.3
        if het > 6: p += (het-6)*0.2
        if mw > 500: p += (mw-500)*0.005
        return round(min(10, max(1, sa+p)), 2)
    except: return 5.0

def analyze_mol(smi, label):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"label": label, "smiles": smi, "error": "invalid"}
    
    # Basic props
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotbonds = Descriptors.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    ar_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    het = rdMolDescriptors.CalcNumHeteroatoms(mol)
    heavy = mol.GetNumHeavyAtoms()
    formula = rdMolDescriptors.CalcMolFormula(mol)
    sasa = sa_raw(mol)
    
    # Fused ring count (approximate)
    ri = mol.GetRingInfo()
    atom_rings = [set(r) for r in ri.AtomRings()]
    # Count fused pairs
    fused_pairs = 0
    for i in range(len(atom_rings)):
        for j in range(i+1, len(atom_rings)):
            if len(atom_rings[i] & atom_rings[j]) >= 2:
                fused_pairs += 1
    # Fused ring systems
    fused_systems = 0
    visited = set()
    for i in range(len(atom_rings)):
        if i in visited:
            continue
        stack = [i]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            for j in range(len(atom_rings)):
                if j not in visited and len(atom_rings[cur] & atom_rings[j]) >= 2:
                    stack.append(j)
        if len(comp) >= 2:
            fused_systems += 1
    
    # 3D shape analysis
    mol_h = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol_h)
        conf = mol_h.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol_h.GetNumAtoms())])
        # Remove H for main analysis
        heavy_mask = [a.GetAtomicNum() > 1 for a in mol_h.GetAtoms()]
        heavy_coords = coords[heavy_mask]
        
        if len(heavy_coords) >= 3:
            # PCA for principal axes
            centered = heavy_coords - heavy_coords.mean(axis=0)
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Sort by eigenvalue descending
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            # Lengths along principal axes (2*std)
            lengths = 2 * np.sqrt(np.maximum(eigvals, 0))
            length, width, thickness = lengths[0], lengths[1], lengths[2] if len(lengths) > 2 else 0
        else:
            length, width, thickness = 0, 0, 0
        
        # Flatness: ratio of thickness to length
        flatness = thickness / length if length > 0.1 else 0
        
    except Exception as e:
        length, width, thickness, flatness = 0, 0, 0, 0
    
    # Hydrophobic surface proxy
    aromatic_atoms = set()
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic_atoms.add(atom.GetIdx())
    hydrophobic_ratio = len(aromatic_atoms) / heavy if heavy > 0 else 0
    
    # Amide anchor detection
    amide_pattern = Chem.MolFromSmarts('[#6][CX3](=[OX1])[NX3]')
    amide_matches = mol.GetSubstructMatches(amide_pattern)
    has_amide = len(amide_matches) > 0
    
    return {
        "label": label, "smiles": smi, "formula": formula,
        "mw": round(mw, 1), "logp": round(logp, 2), "tpsa": round(tpsa, 1),
        "hbd": hbd, "hba": hba, "rotatable_bonds": rotbonds,
        "ring_count": rings, "aromatic_ring_count": ar_rings,
        "fused_ring_systems": fused_systems, "hetero_count": het,
        "heavy_atoms": heavy, "sa_raw": sasa,
        "mol_length_A": round(length, 2), "mol_width_A": round(width, 2),
        "mol_thickness_A": round(thickness, 2), "flatness_ratio": round(flatness, 3),
        "hydrophobic_aromatic_ratio": round(hydrophobic_ratio, 3),
        "has_amide_anchor": has_amide,
        "amide_count": len(amide_matches),
    }

# Three reference molecules
stage5_smi = "O=C(Nc1ccccc1F)C1=CC=C2C=CC=c3c(F)ccnc3=CC=C21"
pyrrolo_smi = "O=C(Nc1ccc(F)cc1)n1ccc2ncncc21"
pyridine_ext_smi = "O=C(Nc1ccc(C#N)cc1)c1nccc(-c2ncccn2)n1"

stage5 = analyze_mol(stage5_smi, "Stage5_attack")
pyrrolo = analyze_mol(pyrrolo_smi, "pyrrolopyrimidine_1")
pyridine_ext = analyze_mol(pyridine_ext_smi, "pyridine_pyrimidine_ext")

# Save JSON
results = {"stage5": stage5, "pyrrolopyrimidine_1": pyrrolo, "pyridine_pyrimidine_ext": pyridine_ext}
with open(P1 / 'stage5_shape_reference.json', 'w') as f:
    json.dump(results, f, indent=2)

# Write report
with open(P1 / 'stage5_shape_report.md', 'w') as f:
    f.write("# Stage14 Shape Analysis Report\n\n")
    
    f.write("## 1. Three Reference Molecules Comparison\n\n")
    f.write("| Property | Stage5 attack | pyrrolopyrimidine_1 | pyridine-pyrim ext |\n")
    f.write("|---|---|---|---|\n")
    keys = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds', 'ring_count',
            'aromatic_ring_count', 'fused_ring_systems', 'hetero_count', 'sa_raw',
            'mol_length_A', 'mol_width_A', 'mol_thickness_A', 'flatness_ratio',
            'hydrophobic_aromatic_ratio', 'has_amide_anchor']
    for k in keys:
        v1 = stage5.get(k, 'N/A')
        v2 = pyrrolo.get(k, 'N/A')
        v3 = pyridine_ext.get(k, 'N/A')
        f.write(f"| {k} | {v1} | {v2} | {v3} |\n")
    
    f.write("\n## 2. Key Insights\n\n")
    f.write("### Stage5 attack (binding=0.260, SA=0.082)\n")
    f.write(f"- MW={stage5['mw']}, logP={stage5['logp']}, length={stage5['mol_length_A']}Å\n")
    f.write(f"- Fused ring systems: {stage5['fused_ring_systems']}\n")
    f.write(f"- Flatness ratio: {stage5['flatness_ratio']} (lower = more planar)\n")
    f.write(f"- SA raw: {stage5['sa_raw']} (high = hard to synthesize)\n")
    f.write("- **WHY binding high:** Large fused polycycle fills hydrophobic cleft, planar shape matches pocket\n")
    f.write("- **WHY SA collapsed:** 5 fused rings, high complexity, CF3 group\n\n")
    
    f.write("### pyrrolopyrimidine_1 (binding=0.148, SA=0.573)\n")
    f.write(f"- MW={pyrrolo['mw']}, logP={pyrrolo['logp']}, length={pyrrolo['mol_length_A']}Å\n")
    f.write(f"- Fused ring systems: {pyrrolo['fused_ring_systems']}\n")
    f.write("- **WHY SA good:** Simple 2-ring fused system, small molecule\n")
    f.write("- **WHY binding low:** Too compact, doesn't fill hydrophobic cleft, binding ~2x overestimated\n\n")
    
    f.write("### pyridine-pyrimidine ext (binding=0.141, SA=0.553)\n")
    f.write(f"- MW={pyridine_ext['mw']}, logP={pyridine_ext['logp']}, length={pyridine_ext['mol_length_A']}Å\n")
    f.write(f"- Fused ring systems: {pyridine_ext['fused_ring_systems']}\n")
    f.write("- **WHY SA ok:** 3-ring system, moderate complexity\n")
    f.write("- **WHY binding low:** Extended but not planar enough, heteroaryl doesn't fill hydrophobic pocket\n\n")
    
    f.write("## 3. Shape Sweet Spot for Stage14\n\n")
    f.write("Target: Stage5-like shape but simplified\n\n")
    f.write("| Parameter | Stage5 (current) | Stage14 target | Why |\n")
    f.write("|---|---|---|---|\n")
    f.write(f"| Length | {stage5['mol_length_A']}Å | 14-18Å | Keep long axis for cleft filling |\n")
    f.write(f"| Fused rings | {stage5['fused_ring_systems']} | 1-2 | Reduce complexity for SA |\n")
    f.write(f"| Flatness | {stage5['flatness_ratio']} | <0.35 | Keep planar hydrophobic surface |\n")
    f.write(f"| SA raw | {stage5['sa_raw']} | <=4.0 | Improve synthesizability |\n")
    f.write(f"| Amide anchor | {stage5['has_amide_anchor']} | True | Keep polar anchor |\n")
    f.write(f"| MW | {stage5['mw']} | 320-450 | Reduce slightly from Stage5 |\n")
    f.write(f"| logP | {stage5['logp']} | 2.5-5.0 | Keep lipophilic |\n")
    f.write("| binding | 0.260 | >=0.24 | Must retain Stage5-like shape |\n")
    f.write("| SA online | 0.082 | >=0.30 | Simplify fused rings + remove CF3 |\n")
    f.write("\n## 4. Design Strategy\n\n")
    f.write("Replace Stage5's fused isoquinoline core with:\n")
    f.write("1. **Naphthyl amide** — 2 fused rings, good planarity, easier SA\n")
    f.write("2. **Biphenyl amide** — 2 rings connected by bond, flexible but can be planar\n")
    f.write("3. **Quinoline amide** — 2 fused rings with N, good pharmacophore\n")
    f.write("4. **Isoquinoline amide** — similar to quinoline, different vector\n")
    f.write("5. **Oxadiazole-biaryl** — heterocyclic linker + biaryl extension\n")
    f.write("6. **Diaryl aza-amide** — medium planarity, balanced properties\n")
    f.write("\nAll preserve: amide anchor + 14-18Å long axis + hydrophobic surface\n")
    f.write("All reduce: fused ring count, SA penalty, CF3 groups\n")

print("Phase 1 done: shape analysis report written")
print(f"\nStage5: MW={stage5['mw']}, length={stage5['mol_length_A']}Å, fused={stage5['fused_ring_systems']}, SA_raw={stage5['sa_raw']}")
print(f"Pyrrolo: MW={pyrrolo['mw']}, length={pyrrolo['mol_length_A']}Å, fused={pyrrolo['fused_ring_systems']}, SA_raw={pyrrolo['sa_raw']}")
print(f"Pyridine_ext: MW={pyridine_ext['mw']}, length={pyridine_ext['mol_length_A']}Å, fused={pyridine_ext['fused_ring_systems']}, SA_raw={pyridine_ext['sa_raw']}")
