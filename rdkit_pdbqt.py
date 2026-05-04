#!/usr/bin/env python3
"""
PDBQT writer - exact byte-for-byte format matching working docking files.
Reference line:
ATOM      1  O   LIG A   1      -0.441  -1.989  -0.171  0.00  0.00          OA
"""
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

def get_ad_type(atom):
    anum = atom.GetAtomicNum()
    if anum == 1:
        return 'H'
    if anum == 6:
        return 'A' if atom.GetIsAromatic() else 'C'
    if anum == 7:
        return 'NA' if atom.GetIsAromatic() else 'N'
    if anum == 8:
        return 'OA'
    if anum == 9:
        return 'F'
    if anum == 16:
        return 'SA' if atom.GetIsAromatic() else 'S'
    if anum == 17:
        return 'Cl'
    if anum == 35:
        return 'Br'
    return 'C'

def mol_to_pdbqt(mol, out_path):
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    conf = mol_h.GetConformer()
    
    lines = ['ROOT']
    for i, atom in enumerate(mol_h.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        ad = get_ad_type(atom)
        sym = atom.GetSymbol()
        # Build line matching exact reference format (78 chars)
        # Cols 1-6:  "ATOM  "
        # Cols 7-11: serial right-justified
        # Col 12:    " "
        # Cols 13-16: " {sym}  " (4 chars, space + symbol + padding)
        # Col 17:    " "
        # Cols 18-20: "LIG"
        # Col 21:    " "
        # Col 22:    "A"
        # Cols 23-26: "   1"
        # Cols 27-30: "    "
        # Cols 31-38: x as %8.3f
        # Cols 39-46: y as %8.3f
        # Cols 47-54: z as %8.3f
        # Cols 55-62: charge as " %5.2f  " (space + 5.2f + 2 spaces = 8)
        # Cols 63-70: occupancy as "%4.2f    " (4.2f + 4 spaces = 8)
        # Cols 71-76: "      " (6 spaces)
        # Cols 77-78: ad type right-justified
        
        name = f" {sym:<2s} "  # 4 chars: " O  " or " C  " or " NA "
        if len(name) != 4:
            name = name[:4]
        
        charge_str = f" {0.0:5.2f}  "  # 8 chars: "  0.00  " ... wait let me check
        occ_str = f"{0.0:4.2f}    "     # 8 chars: "0.00    "
        
        # Verify: reference charge field = "  0.00  " and occ = "0.00    "
        # Let me match exactly
        charge_field = "  0.00  "  # 8 chars, exactly from reference
        occ_field = "0.00    "     # 8 chars, exactly from reference
        
        line = (f"ATOM  {i+1:5d} {name} LIG A   1    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                f"{charge_field}{occ_field}      {ad:>2s}")
        lines.append(line)
    lines.append('ENDROOT')
    lines.append('TORSDOF 0')
    
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return True

if __name__ == '__main__':
    smi, out = sys.argv[1], sys.argv[2]
    mol = Chem.MolFromSmiles(smi)
    ok = mol_to_pdbqt(mol, out) if mol else False
    print(f"{'OK' if ok else 'FAIL'}: {out}")
