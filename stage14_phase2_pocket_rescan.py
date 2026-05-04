#!/usr/bin/env python3
"""
Stage14 Phase 2: Pocket Rescan (fixed PDBQT writer)
"""
import csv, json, subprocess, tempfile, os, sys, shutil
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

BASE = Path('/Users/pwngwc/ai4s_chem')
VINA = BASE / 'bin' / 'vina'
RECEPTOR = BASE / 'receptor.pdbqt'
P2 = BASE / 'result' / 'stage14_pocket_rescan_stage5_shape_simplification' / '02_pocket_rescan'
P2.mkdir(parents=True, exist_ok=True)

# Import PDBQT writer
sys.path.insert(0, str(BASE))
from rdkit_pdbqt import mol_to_pdbqt

MOLS = {
    'stage5': 'O=C(Nc1ccccc1F)C1=CC=C2C=CC=c3c(F)ccnc3=CC=C21',
    'pyrrolo': 'O=C(Nc1ccc(F)cc1)n1ccc2ncncc21',
    'pyridine_ext': 'O=C(Nc1ccc(C#N)cc1)c1nccc(-c2ncccn2)n1',
}

OLD_CENTER = [18.3, 2.3, 21.4]

# Generate 5x5x5 = 125 centers
dx_vals = [-10, -5, 0, 5, 10]
dy_vals = [-10, -5, 0, 5, 10]
dz_vals = [-5, 0, 5]
centers = [(round(OLD_CENTER[0]+dx,1), round(OLD_CENTER[1]+dy,1), round(OLD_CENTER[2]+dz,1))
           for dx in dx_vals for dy in dy_vals for dz in dz_vals]

print(f"Scanning {len(centers)} pocket centers...")

# Prepare ligands
lig_files = {}
for name, smi in MOLS.items():
    mol = Chem.MolFromSmiles(smi)
    pdbqt_path = P2 / f'{name}.pdbqt'
    mol_to_pdbqt(mol, str(pdbqt_path))
    lig_files[name] = pdbqt_path
    print(f"  Prepared: {name}")

def run_vina(lig_path, center, box=[20,20,20]):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'out.pdbqt'
        try:
            r = subprocess.run([
                str(VINA), '--receptor', str(RECEPTOR),
                '--ligand', str(lig_path),
                '--center_x', str(center[0]), '--center_y', str(center[1]), '--center_z', str(center[2]),
                '--size_x', str(box[0]), '--size_y', str(box[1]), '--size_z', str(box[2]),
                '--exhaustiveness', '4', '--num_modes', '1', '--out', str(out)
            ], capture_output=True, text=True, timeout=60)
            if out.exists():
                for line in out.read_text().split('\n'):
                    if 'VINA RESULT' in line:
                        parts = line.split()
                        idx = parts.index('RESULT:') + 1
                        return float(parts[idx])
        except:
            pass
    return None

results = []
for i, (cx, cy, cz) in enumerate(centers):
    row = {'center_x': cx, 'center_y': cy, 'center_z': cz}
    for name, lig in lig_files.items():
        v = run_vina(lig, [cx, cy, cz])
        row[f'{name}_vina'] = v
    
    v5, vp, vpe = row.get('stage5_vina'), row.get('pyrrolo_vina'), row.get('pyridine_ext_vina')
    if v5 is not None and vp is not None and vpe is not None:
        ranking_ok = (v5 < vp) and (v5 < vpe)
        row['ranking_matches_online'] = ranking_ok
        row['alignment_score'] = round(min(vp - v5, vpe - v5), 3)
    else:
        row['ranking_matches_online'] = None
        row['alignment_score'] = None
    
    results.append(row)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(centers)} done...")

# Save
fields = ['center_x','center_y','center_z','stage5_vina','pyrrolo_vina','pyridine_ext_vina',
          'ranking_matches_online','alignment_score']
with open(P2 / 'pocket_grid_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(results)

good = [r for r in results if r.get('ranking_matches_online') == True]
good.sort(key=lambda x: x.get('alignment_score', 0) or 0, reverse=True)

with open(P2 / 'best_pocket_centers.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(good[:20])

# Report
with open(P2 / 'pocket_alignment_report.md', 'w') as f:
    f.write("# Stage14 Pocket Rescan Report\n\n")
    f.write(f"**Centers scanned:** {len(results)}\n")
    f.write(f"**Centers where Stage5 ranks #1:** {len(good)}\n\n")
    
    if good:
        f.write("## Top 10 Centers (Stage5 beats both heteroaryl amides)\n\n")
        f.write("| # | Center | Stage5 | pyrrolo | pyridine_ext | margin |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for i, r in enumerate(good[:10]):
            f.write(f"| {i+1} | ({r['center_x']},{r['center_y']},{r['center_z']}) | "
                    f"{r['stage5_vina']:.3f} | {r['pyrrolo_vina']:.3f} | {r['pyridine_ext_vina']:.3f} | "
                    f"{r['alignment_score']:.3f} |\n")
        best = good[0]
        f.write(f"\n**Best center:** ({best['center_x']}, {best['center_y']}, {best['center_z']}), margin={best['alignment_score']}\n")
    else:
        f.write("## ⚠️ No center found where Stage5 ranks #1\n\n")
        by_s5 = sorted([r for r in results if r.get('stage5_vina') is not None], key=lambda x: x['stage5_vina'])
        if by_s5:
            f.write("## Top centers by Stage5 absolute Vina\n\n")
            f.write("| # | Center | Stage5 | pyrrolo | pyridine_ext |\n")
            f.write("|---|---|---:|---:|---:|\n")
            for i, r in enumerate(by_s5[:10]):
                f.write(f"| {i+1} | ({r['center_x']},{r['center_y']},{r['center_z']}) | "
                        f"{r['stage5_vina']:.3f} | {r['pyrrolo_vina']:.3f} | {r['pyridine_ext_vina']:.3f} |\n")

print(f"\nPhase 2 done. Centers scanned: {len(results)}")
print(f"Stage5 ranks #1: {len(good)} centers")
if good:
    b = good[0]
    print(f"Best: ({b['center_x']},{b['center_y']},{b['center_z']}), margin={b['alignment_score']}")
else:
    by_s5 = sorted([r for r in results if r.get('stage5_vina') is not None], key=lambda x: x['stage5_vina'])
    if by_s5:
        b = by_s5[0]
        print(f"Best Stage5 absolute: ({b['center_x']},{b['center_y']},{b['center_z']}), vina={b['stage5_vina']:.3f}")
