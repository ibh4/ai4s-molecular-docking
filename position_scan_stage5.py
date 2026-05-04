#!/usr/bin/env python3
import csv
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

BASE = Path('/Users/pwngwc/ai4s_chem')
OUT = BASE / 'result' / 'stage5_position_scan'
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = {
    'attack': BASE / 'result' / 'stage4_submit' / 'attack_submit' / 'result.csv',
    'safe': BASE / 'result' / 'stage4_submit' / 'safe_submit' / 'result.csv',
    'balance': BASE / 'result' / 'stage4_submit' / 'balance_submit' / 'result.csv',
}

SUBS = [
    ('F', 'C', 'f_to_me'),
    ('F', 'OC', 'f_to_ome'),
    ('F', 'O', 'f_to_oh_like'),
    ('F', 'C#N', 'f_to_cn'),
    ('OC', 'F', 'ome_to_f'),
    ('OC', 'C', 'ome_to_me'),
    ('OC', 'O', 'ome_trim_to_oh_like'),
    ('C#N', 'F', 'cn_to_f'),
    ('C#N', 'C', 'cn_to_me'),
    ('C#N', 'OC', 'cn_to_ome'),
]


def read_one(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows[0]


def canonical(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def features(smi):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    return {
        'mol': m,
        'mw': Descriptors.MolWt(m),
        'logp': Descriptors.MolLogP(m),
        'tpsa': Descriptors.TPSA(m),
        'rings': rdMolDescriptors.CalcNumRings(m),
        'ar_rings': rdMolDescriptors.CalcNumAromaticRings(m),
        'hbd': Descriptors.NumHDonors(m),
        'hba': Descriptors.NumHAcceptors(m),
        'hetero': rdMolDescriptors.CalcNumHeteroatoms(m),
        'canon': Chem.MolToSmiles(m),
    }


def substitute_once(smi, old, new):
    outs = []
    start = 0
    while True:
        idx = smi.find(old, start)
        if idx == -1:
            break
        cand = smi[:idx] + new + smi[idx+len(old):]
        m = Chem.MolFromSmiles(cand)
        if m:
            outs.append(Chem.MolToSmiles(m))
        start = idx + 1
    return sorted(set(outs))


def sa_proxy(feat, smi):
    """SA score proxy: higher = easier to synthesize (0.20-0.80)
    
    校准记录 (2026-04-30):
    - Stage5 attack seed: online SA=0.081623, local pred was 0.64 → 严重高估
    - 原因：大平面稠环/异喹啉稠环结构合成极难，原模型未惩罚
    - 修正：对 fused polycyclic / 高平面性结构增加重惩罚
    """
    base = 0.72
    base -= max(0, feat['rings'] - 3) * 0.05
    base -= max(0, feat['ar_rings'] - 2) * 0.04
    base -= 0.04 if feat['mw'] > 380 else 0.0
    base -= 0.04 if feat['logp'] > 4.2 else 0.0
    if 'C(F)(F)F' in smi:
        base -= 0.10
    if 'ncnc' in smi or 'ccnc' in smi or 'cn[nH]' in smi:
        base -= 0.03
    
    # === 2026-04-30 校准：大平面稠环 SA 惩罚 ===
    # 检测 fused polycyclic 核心（≥3个环稠合，含杂原子）
    fused_ring_penalty = 0.0
    # 异喹啉/喹啉稠环模式：C=CC=C2C=CC=c3... 或类似
    fused_patterns = [
        'C=CC=C2C=CC=c',   # 异喹啉稠环
        'C=CC=CC=CC=C',    # 长共轭稠环
        'c3ccnc3',          # 吡啶稠环
        'c3c(F)ccnc3',      # 氟代吡啶稠环
        '=CC=C21',          # 稠环闭合标记
    ]
    for pat in fused_patterns:
        if pat in smi:
            fused_ring_penalty += 0.20
            break
    
    # 大稠环系统（≥4环，含氮杂环）额外惩罚
    if feat['rings'] >= 4 and ('n' in smi.lower() or 'N' in smi):
        fused_ring_penalty += 0.15
    
    # 高平面性结构惩罚（通过 SMILES 中连续稠合判断）
    if smi.count('=C') >= 8 and feat['ar_rings'] >= 3:
        fused_ring_penalty += 0.15
    
    # 含氟芳环 + 稠环 = 合成更难
    if 'F' in smi and feat['rings'] >= 3:
        fused_ring_penalty += 0.05
    
    base -= fused_ring_penalty
    
    return round(max(0.05, min(0.80, base)), 4)


def binding_proxy(feat, smi, seed_bind):
    """Binding score proxy: capped at 0.33
    
    校准记录 (2026-04-30):
    - Stage5 attack seed: online binding=0.260375, local pred was 0.33 → 高估
    - 修正：降低上限，大稠环不再给高 binding 预测
    """
    b = seed_bind
    if feat['mw'] < 320:
        b -= 0.01
    if 330 <= feat['mw'] <= 385:
        b += 0.005
    if feat['mw'] > 400:
        b -= 0.01
    if feat['logp'] < 2.4:
        b -= 0.005
    if feat['logp'] > 4.5:
        b -= 0.01
    if 'F' in smi:
        b += 0.005
    if 'OC' in smi:
        b += 0.002
    if 'O=C(N' in smi or 'NC(=O)' in smi:
        b += 0.005
    if 'ncnc' in smi or 'ccnc' in smi:
        b += 0.006
    # 2026-04-30 校准：大稠环 binding 上限压低
    if feat['rings'] >= 4:
        b = min(b, 0.28)
    return round(max(0.16, min(0.30, b)), 4)


def risk_proxy(feat, smi):
    """Risk proxy: higher = more risky
    
    校准记录 (2026-04-30):
    - Stage5 attack seed: online risk 实际很高（SA=0.08）
    - 修正：大平面稠环 risk 大幅提高
    """
    r = 0.0
    if 'C(F)(F)F' in smi: r += 0.10
    if feat['rings'] >= 4: r += 0.05
    if feat['ar_rings'] >= 3: r += 0.04
    if feat['mw'] > 390: r += 0.04
    if feat['logp'] > 4.2: r += 0.04
    if 'ncnc' in smi or 'ccnc' in smi or 'cn[nH]' in smi: r += 0.03
    # 2026-04-30 校准：大平面稠环高风险
    if 'C=CC=C2C=CC=c' in smi or '=CC=C21' in smi:
        r += 0.15  # 异喹啉稠环合成极难
    if feat['rings'] >= 4 and feat['ar_rings'] >= 3:
        r += 0.08  # 多环多芳环额外风险
    return round(r, 4)


def total(bind, sa, route=0.95):
    mol = 0.8*bind + 0.1 + 0.1*sa
    return round(0.7*mol + 0.3*route, 6)


def route_variant(seed_route, new_smi):
    parts = seed_route.split(',')
    lhs = parts[-1].split('>>')[0]
    parts[-1] = lhs + '>>' + new_smi
    return ','.join(parts)


def scan_seed(name, row):
    seed_smi = row['mol_smiles']
    seed_route = row['route']
    seed_bind_map = {'attack':0.3095,'safe':0.2930,'balance':0.1972}
    seed_bind = seed_bind_map[name]
    res = {}
    # include seed itself
    for cand_smi, tag in [(seed_smi,'seed')]:
        feat = features(cand_smi)
        if feat:
            sa = sa_proxy(feat, cand_smi)
            bind = binding_proxy(feat, cand_smi, seed_bind)
            risk = risk_proxy(feat, cand_smi)
            pred = total(bind, sa)
            score = round(0.60*bind + 0.15*sa + 0.25*0.95 - 0.45*risk, 6)
            res[feat['canon']] = {
                'seed': name, 'mol_smiles': cand_smi, 'route': route_variant(seed_route,cand_smi),
                'transform':'seed','binding_est':bind,'sa_est':sa,'pred_total':pred,'risk':risk,'score':score,
                'mw':round(feat['mw'],2),'logp':round(feat['logp'],2),'tpsa':round(feat['tpsa'],2)
            }
    # substitutions
    for old, new, tag in SUBS:
        for cand_smi in substitute_once(seed_smi, old, new):
            feat = features(cand_smi)
            if not feat:
                continue
            if not (300 <= feat['mw'] <= 420):
                continue
            if not (2.0 <= feat['logp'] <= 4.8):
                continue
            if feat['hbd'] > 3 or feat['hba'] > 8:
                continue
            sa = sa_proxy(feat, cand_smi)
            bind = binding_proxy(feat, cand_smi, seed_bind)
            risk = risk_proxy(feat, cand_smi)
            pred = total(bind, sa)
            score = round(0.60*bind + 0.15*sa + 0.25*0.95 - 0.45*risk, 6)
            res[feat['canon']] = {
                'seed': name, 'mol_smiles': cand_smi, 'route': route_variant(seed_route,cand_smi),
                'transform':tag,'binding_est':bind,'sa_est':sa,'pred_total':pred,'risk':risk,'score':score,
                'mw':round(feat['mw'],2),'logp':round(feat['logp'],2),'tpsa':round(feat['tpsa'],2)
            }
    return list(res.values())


def write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank','seed','mol_smiles','route','transform','binding_est','sa_est','pred_total','risk','score','mw','logp','tpsa'])
        for i, r in enumerate(rows, 1):
            w.writerow([i,r['seed'],r['mol_smiles'],r['route'],r['transform'],r['binding_est'],r['sa_est'],r['pred_total'],r['risk'],r['score'],r['mw'],r['logp'],r['tpsa']])


def main():
    all_rows = []
    per_seed = {}
    for name, path in SEEDS.items():
        row = read_one(path)
        rows = scan_seed(name, row)
        rows.sort(key=lambda x: x['score'], reverse=True)
        per_seed[name] = rows
        write_rows(OUT / f'{name}_position_scan.csv', rows[:20])
        all_rows.extend(rows)
    all_rows.sort(key=lambda x: x['score'], reverse=True)
    write_rows(OUT / 'all_stage5_candidates.csv', all_rows[:60])

    with open(OUT / 'stage5_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Stage5 位点级微调扫描总结\n\n')
        f.write('## 各 seed Top5\n\n')
        for name, rows in per_seed.items():
            f.write(f'### {name}\n\n')
            f.write('| rank | transform | binding | sa | total | risk | smiles |\n')
            f.write('|---|---|---:|---:|---:|---:|---|\n')
            for i, r in enumerate(rows[:5], 1):
                f.write(f"| {i} | {r['transform']} | {r['binding_est']} | {r['sa_est']} | {r['pred_total']} | {r['risk']} | `{r['mol_smiles']}` |\n")
            f.write('\n')
        f.write('## 全局推荐 Top10\n\n')
        f.write('| rank | seed | transform | binding | sa | total | risk | smiles |\n')
        f.write('|---|---|---|---:|---:|---:|---:|---|\n')
        for i, r in enumerate(all_rows[:10], 1):
            f.write(f"| {i} | {r['seed']} | {r['transform']} | {r['binding_est']} | {r['sa_est']} | {r['pred_total']} | {r['risk']} | `{r['mol_smiles']}` |\n")

    print(f'Generated stage5 outputs in {OUT}')
    for r in all_rows[:10]:
        print(r['seed'], r['transform'], r['pred_total'], r['mol_smiles'])

if __name__ == '__main__':
    main()
