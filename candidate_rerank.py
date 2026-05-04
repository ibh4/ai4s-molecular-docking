#!/usr/bin/env python3
import csv, os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

BASE = Path('/Users/pwngwc/ai4s_chem')
RANK_CSV = BASE / 'result' / 'v7_single_candidate_ranking.csv'
OUT_DIR = BASE / 'result' / 'stage2_rerank'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def to_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def mol_features(smi):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    return {
        'mw': Descriptors.MolWt(m),
        'logp': Descriptors.MolLogP(m),
        'tpsa': Descriptors.TPSA(m),
        'rings': rdMolDescriptors.CalcNumRings(m),
        'ar_rings': rdMolDescriptors.CalcNumAromaticRings(m),
        'hetero': rdMolDescriptors.CalcNumHeteroatoms(m),
        'canonical': Chem.MolToSmiles(m),
    }


def route_lookup():
    mapping = {}
    for path in [
        BASE/'result'/'v4_binding_strong'/'result.csv',
        BASE/'result'/'v5_pareto_balanced'/'result.csv',
        BASE/'result'/'v5_safe_balanced'/'result.csv',
        BASE/'result'/'v6_ultra_safe_submit'/'result.csv',
    ]:
        if not path.exists():
            continue
        for row in load_rows(path):
            smi = row.get('mol_smiles','').strip()
            route = row.get('route','').strip()
            if smi and route:
                mapping[smi] = route
                m = Chem.MolFromSmiles(smi)
                if m:
                    mapping[Chem.MolToSmiles(m)] = route
    return mapping


def classify(row, feat):
    smi = row['mol_smiles']
    tags = []
    if 'C(F)(F)F' in smi:
        tags.append('CF3')
    if feat['rings'] >= 4:
        tags.append('polycyclic')
    if feat['ar_rings'] >= 3:
        tags.append('aromatic_dense')
    if 'ncnc' in smi or 'ccnc' in smi or 'cn[nH]' in smi:
        tags.append('hetero_fused')
    if feat['mw'] > 390:
        tags.append('mw_high')
    if feat['logp'] > 4.2:
        tags.append('logp_high')
    if 'C(F)(F)F' not in smi:
        tags.append('non_cf3')
    if ('C' in smi and 'C(F)(F)F' not in smi) or 'O' in smi or 'F' in smi:
        tags.append('sa_friendly_substituent')
    return tags


def rerank_score(row, feat, version):
    bind = to_float(row['estimated_binding_score_mid'])
    sa = to_float(row['estimated_sa_score'])
    route = to_float(row['estimated_route_score'])
    total = to_float(row['predicted_total_mid'])
    sas = to_float(row['sascore_raw'])
    cf3 = 1 if 'C(F)(F)F' in row['mol_smiles'] else 0
    fused = 1 if feat['rings'] >= 4 else 0
    hetero_fused = 1 if ('ncnc' in row['mol_smiles'] or 'ccnc' in row['mol_smiles'] or 'cn[nH]' in row['mol_smiles']) else 0
    risk = 0.0
    risk += 0.07 * cf3
    risk += 0.05 * fused
    risk += 0.03 * hetero_fused
    risk += 0.02 * max(0, feat['mw'] - 390) / 20.0
    risk += 0.02 if feat['logp'] > 4.2 else 0.0

    if version == 'safe_diverse':
        return 0.52*bind + 0.18*sa + 0.25*route + 0.08*min(1.0, (4.2-sas)/2.5) - 0.70*risk
    if version == 'pareto_balanced':
        return 0.58*bind + 0.14*sa + 0.25*route + 0.05*total - 0.55*risk
    if version == 'attack_binding':
        return 0.72*bind + 0.08*sa + 0.20*route - 0.35*risk
    if version == 'non_cf3_priority':
        non_cf3_bonus = 0.08 if cf3 == 0 else -0.10
        return 0.54*bind + 0.16*sa + 0.25*route + non_cf3_bonus - 0.45*risk
    if version == 'medchem_friendly':
        med_bonus = 0.05 if (cf3 == 0 and feat['mw'] < 380 and feat['rings'] <= 4) else 0.0
        return 0.50*bind + 0.20*sa + 0.25*route + med_bonus - 0.50*risk
    return total - risk


def write_pool(name, rows, route_map, topn=10):
    out_csv = OUT_DIR / f'{name}.csv'
    out_md = OUT_DIR / f'{name}.md'
    use = rows[:topn]
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank','mol_smiles','route','source_version','vina_best','sascore_raw','estimated_sa_score','estimated_binding_score_mid','predicted_total_mid','rerank_score','tags'])
        for i, r in enumerate(use, 1):
            route = route_map.get(r['mol_smiles'], route_map.get(r['canonical'], ''))
            w.writerow([i, r['mol_smiles'], route, r['source_version'], r['vina_best'], r['sascore_raw'], r['estimated_sa_score'], r['estimated_binding_score_mid'], r['predicted_total_mid'], r['rerank_score'], ';'.join(r['tags'])])
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(f'# {name}\n\n')
        f.write('| rank | source | vina | sa_raw | est_sa | bind_mid | total_mid | rerank | tags |\n')
        f.write('|---|---|---:|---:|---:|---:|---:|---:|---|\n')
        for i, r in enumerate(use, 1):
            f.write(f"| {i} | {r['source_version']} | {r['vina_best']} | {r['sascore_raw']} | {r['estimated_sa_score']} | {r['estimated_binding_score_mid']} | {r['predicted_total_mid']} | {r['rerank_score']:.4f} | {'/'.join(r['tags'])} |\n")
    return out_csv, out_md


def main():
    rows = load_rows(RANK_CSV)
    route_map = route_lookup()
    enriched = []
    for row in rows:
        feat = mol_features(row['mol_smiles'])
        if not feat:
            continue
        row = dict(row)
        row['canonical'] = feat['canonical']
        row['tags'] = classify(row, feat)
        row['_feat'] = feat
        enriched.append(row)

    version_defs = {
        'safe_diverse': '稳妥优先：保route/保SA/适度binding，适合先恢复有效高分',
        'pareto_balanced': '平衡优先：binding和SA兼顾，适合作为主提交通道',
        'attack_binding': '冲分优先：偏向高binding，但保留基本风险惩罚',
        'non_cf3_priority': '去CF3优先：降低线上SA崩盘风险，测试非CF3替代路线',
        'medchem_friendly': '药化友好优先：更像真实可合成候选，适合稳健路线',
    }

    summary = []
    for name, desc in version_defs.items():
        ranked = []
        for row in enriched:
            score = rerank_score(row, row['_feat'], name)
            x = dict(row)
            x['rerank_score'] = score
            ranked.append(x)
        ranked.sort(key=lambda r: r['rerank_score'], reverse=True)
        out_csv, out_md = write_pool(name, ranked, route_map, topn=10)
        top = ranked[0]
        summary.append({
            'version': name,
            'description': desc,
            'top_smiles': top['mol_smiles'],
            'top_source': top['source_version'],
            'top_vina': top['vina_best'],
            'top_sa': top['sascore_raw'],
            'top_bind': top['estimated_binding_score_mid'],
            'top_total': top['predicted_total_mid'],
            'top_rerank': round(top['rerank_score'], 4),
            'tags': ';'.join(top['tags']),
            'csv': str(out_csv),
            'md': str(out_md),
        })

    with open(OUT_DIR / 'submission_priority_table.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['priority','version','description','top_source','top_vina','top_sa','top_bind_mid','top_total_mid','top_rerank','tags','csv','md','recommendation'])
        order = sorted(summary, key=lambda x: x['top_rerank'], reverse=True)
        for i, s in enumerate(order, 1):
            rec = '主推' if i == 1 else '次推' if i <= 3 else '备选'
            w.writerow([i, s['version'], s['description'], s['top_source'], s['top_vina'], s['top_sa'], s['top_bind'], s['top_total'], s['top_rerank'], s['tags'], s['csv'], s['md'], rec])

    with open(OUT_DIR / 'submission_priority_table.md', 'w', encoding='utf-8') as f:
        f.write('# 提交优先级建议表\n\n')
        f.write('| priority | version | 说明 | top source | vina | sa_raw | bind_mid | total_mid | rerank | 推荐 |\n')
        f.write('|---|---|---|---|---:|---:|---:|---:|---:|---|\n')
        order = sorted(summary, key=lambda x: x['top_rerank'], reverse=True)
        for i, s in enumerate(order, 1):
            rec = '主推' if i == 1 else '次推' if i <= 3 else '备选'
            f.write(f"| {i} | {s['version']} | {s['description']} | {s['top_source']} | {s['top_vina']} | {s['top_sa']} | {s['top_bind']} | {s['top_total']} | {s['top_rerank']} | {rec} |\n")
        f.write('\n## 结论\n\n')
        if order:
            f.write(f"- 当前第一优先级版本池：**{order[0]['version']}**\n")
            f.write(f"- Top1 分子：`{order[0]['top_smiles']}`\n")
            f.write(f"- 理由：{order[0]['description']}\n")
            f.write('- 建议并行关注主推/次推/备选三档，不要只押单一路线。\n')

    print('Generated stage2 rerank outputs in', OUT_DIR)
    for s in sorted(summary, key=lambda x: x['top_rerank'], reverse=True):
        print(s['version'], s['top_rerank'], s['top_smiles'])

if __name__ == '__main__':
    main()
