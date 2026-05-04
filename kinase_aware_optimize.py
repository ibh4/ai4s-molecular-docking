#!/usr/bin/env python3
"""
基于 target.pdb 的 kinase-like 口袋认知，生成更适合裂缝型激酶口袋的候选：
- 保留 hinge-binding 风格杂环/酰胺核心
- 做小取代扫描（F/CH3/OCH3/OH/CN）
- 强惩罚 CF3-heavy / 高稠环 / 高MW / 高logP
- 输出多个版本池和推荐表
"""
import csv, json, os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

BASE = Path('/Users/pwngwc/ai4s_chem')
OUT = BASE / 'result' / 'kinase_aware_stage3'
OUT.mkdir(parents=True, exist_ok=True)

RANKING = BASE / 'result' / 'v7_single_candidate_ranking.csv'

SEED_CSVS = [
    BASE/'result'/'v5_pareto_balanced'/'result.csv',
    BASE/'result'/'v4_binding_strong'/'result.csv',
    BASE/'result'/'v6_ultra_safe_submit'/'result.csv',
]

SUB_REPLACEMENTS = [
    ('C(F)(F)F', 'F', 'cf3_to_f'),
    ('C(F)(F)F', 'C', 'cf3_to_me'),
    ('C(F)(F)F', 'OC', 'cf3_to_ome'),
    ('C(F)(F)F', 'O', 'cf3_to_oh_like'),
    ('C(F)(F)F', 'C#N', 'cf3_to_cn'),
]


def load_csv(path):
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def canonical(smi):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    return Chem.MolToSmiles(m)


def mol_features(smi):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    return {
        'mol': m,
        'canonical': Chem.MolToSmiles(m),
        'mw': round(Descriptors.MolWt(m), 2),
        'logp': round(Descriptors.MolLogP(m), 2),
        'tpsa': round(Descriptors.TPSA(m), 2),
        'rings': rdMolDescriptors.CalcNumRings(m),
        'ar_rings': rdMolDescriptors.CalcNumAromaticRings(m),
        'hbd': Descriptors.NumHDonors(m),
        'hba': Descriptors.NumHAcceptors(m),
        'hetero': rdMolDescriptors.CalcNumHeteroatoms(m),
        'formula': rdMolDescriptors.CalcMolFormula(m),
    }


def estimate_sa_raw_proxy(feat, smi):
    # 与你项目保持同风格，但更偏保守
    score = 1.0
    score += max(0, feat['rings'] - 1) * 0.35
    score += max(0, feat['ar_rings'] - 2) * 0.30
    score += 0.20 if 'C(F)(F)F' in smi else 0.0
    score += max(0, feat['mw'] - 360) / 80.0 * 0.45
    score += 0.20 if feat['logp'] > 4.2 else 0.0
    return round(min(4.8, max(1.2, score + 1.0)), 2)


def estimate_sa_score_online(feat, smi):
    # 0~1，越高越好
    raw = estimate_sa_raw_proxy(feat, smi)
    base = max(0.0, min(1.0, (4.2 - raw) / 2.4))
    penalty = 0.0
    if 'C(F)(F)F' in smi:
        penalty += 0.12
    if feat['rings'] >= 4:
        penalty += 0.06
    if feat['ar_rings'] >= 3:
        penalty += 0.05
    if feat['mw'] > 390:
        penalty += 0.05
    if feat['logp'] > 4.2:
        penalty += 0.04
    if 'ncnc' in smi or 'ccnc' in smi or 'cn[nH]' in smi:
        penalty += 0.03
    return round(max(0.0, min(1.0, base - penalty)), 4), raw


def estimate_binding_from_shape(feat, smi, prior_bind=None, prior_vina=None):
    # 激酶裂缝口袋偏爱：中等平面、单核心+单延展，不喜欢过大过肥
    if prior_bind is not None:
        bind = prior_bind
    elif prior_vina is not None:
        bind = max(0.12, min(0.32, 0.0843 * abs(prior_vina) - 0.688))
    else:
        bind = 0.18
    # shape adjustments
    if feat['mw'] < 330:
        bind -= 0.01
    if 330 <= feat['mw'] <= 390:
        bind += 0.01
    if feat['mw'] > 410:
        bind -= 0.02
    if feat['ar_rings'] >= 3:
        bind += 0.005
    if feat['rings'] >= 5:
        bind -= 0.015
    if feat['logp'] > 4.5:
        bind -= 0.01
    if 'C(F)(F)F' in smi:
        bind += 0.005
    if 'NC(=O)' in smi or 'O=C(N' in smi:
        bind += 0.01
    if 'ncnc' in smi or 'ccnc' in smi or '[nH]' in smi:
        bind += 0.008
    return round(max(0.12, min(0.32, bind)), 4)


def kinase_risk(feat, smi):
    risk = 0.0
    tags = []
    if 'C(F)(F)F' in smi:
        risk += 0.10; tags.append('CF3')
    if feat['rings'] >= 4:
        risk += 0.06; tags.append('polycyclic')
    if feat['ar_rings'] >= 3:
        risk += 0.05; tags.append('aromatic_dense')
    if feat['mw'] > 390:
        risk += 0.05; tags.append('mw_high')
    if feat['logp'] > 4.2:
        risk += 0.04; tags.append('logp_high')
    if 'ncnc' in smi or 'ccnc' in smi or 'cn[nH]' in smi:
        risk += 0.03; tags.append('hetero_fused')
    if 'C(F)(F)F' not in smi:
        tags.append('non_cf3')
    if any(x in smi for x in ['C', 'F', 'O']) and 'C(F)(F)F' not in smi:
        tags.append('small_substituent_friendly')
    return round(risk,4), tags


def total_score(bind, sa, route=0.95):
    mol = 0.8*bind + 0.1*1.0 + 0.1*sa
    return round(0.7*mol + 0.3*route, 6)


def transform_smiles(smi):
    outs = [(smi, 'seed')]
    for old, new, tag in SUB_REPLACEMENTS:
        if old in smi:
            cand = smi.replace(old, new, 1)
            if Chem.MolFromSmiles(cand):
                outs.append((Chem.MolToSmiles(Chem.MolFromSmiles(cand)), tag))
    return outs


def route_lookup():
    mp = {}
    for p in SEED_CSVS:
        for row in load_csv(p):
            smi = row.get('mol_smiles','').strip()
            route = row.get('route','').strip()
            if smi and route:
                mp[smi] = route
                can = canonical(smi)
                if can:
                    mp[can] = route
    return mp


def simple_route_for_variant(seed_route, seed_smi, new_smi):
    # 保守策略：如果只做轻微取代，先沿用原路线骨架但最终产物强制替换为新分子
    if not seed_route:
        return f"ClC(=O)c1ccccc1.Nc1ccccc1>>{new_smi}"
    parts = seed_route.split(',')
    last = parts[-1]
    if '>>' not in last:
        return f"{seed_route}>>{new_smi}"
    lhs, _ = last.split('>>',1)
    parts[-1] = lhs + '>>' + new_smi
    return ','.join(parts)


def validate_route_simple(smi, route):
    if not route or '>>' not in route:
        return False
    prod = route.split(',')[-1].split('>>')[-1]
    c1 = canonical(smi)
    c2 = canonical(prod)
    return c1 is not None and c1 == c2


def main():
    route_map = route_lookup()
    ranking_rows = load_csv(RANKING)
    rank_map = {canonical(r['mol_smiles']): r for r in ranking_rows if canonical(r['mol_smiles'])}

    seeds = []
    for p in SEED_CSVS:
        for row in load_csv(p):
            smi = row['mol_smiles'].strip()
            route = row['route'].strip()
            can = canonical(smi)
            seeds.append({'seed_smi': smi, 'seed_can': can, 'seed_route': route, 'source': p.parent.name})

    seen = {}
    for seed in seeds:
        for new_smi, transform in transform_smiles(seed['seed_smi']):
            can = canonical(new_smi)
            if not can or can in seen:
                continue
            feat = mol_features(new_smi)
            if not feat:
                continue
            # 基础过滤：更像适合激酶裂缝口袋的分子
            if not (300 <= feat['mw'] <= 420):
                continue
            if not (2.0 <= feat['logp'] <= 4.6):
                continue
            if feat['hbd'] > 3 or feat['hba'] > 8:
                continue
            if feat['rings'] > 5:
                continue

            prior = rank_map.get(seed['seed_can']) or rank_map.get(can)
            prior_bind = float(prior['estimated_binding_score_mid']) if prior else None
            prior_vina = float(prior['vina_best']) if prior else None

            bind = estimate_binding_from_shape(feat, new_smi, prior_bind=prior_bind, prior_vina=prior_vina)
            sa, sa_raw = estimate_sa_score_online(feat, new_smi)
            risk, tags = kinase_risk(feat, new_smi)
            route = simple_route_for_variant(seed['seed_route'], seed['seed_smi'], new_smi)
            route_valid = validate_route_simple(new_smi, route)
            if not route_valid:
                continue
            total = total_score(bind, sa, 0.95)
            score_safe = total - 0.40*risk
            score_balance = 0.60*bind + 0.15*sa + 0.25*0.95 - 0.45*risk
            score_attack = 0.72*bind + 0.08*sa + 0.20*0.95 - 0.25*risk
            seen[can] = {
                'mol_smiles': new_smi,
                'canonical': can,
                'source_seed': seed['seed_smi'],
                'source_version': seed['source'],
                'transform': transform,
                'route': route,
                'mw': feat['mw'],
                'logp': feat['logp'],
                'tpsa': feat['tpsa'],
                'rings': feat['rings'],
                'ar_rings': feat['ar_rings'],
                'sa_raw_proxy': sa_raw,
                'sa_score_est': sa,
                'binding_est': bind,
                'route_est': 0.95,
                'pred_total': total,
                'risk': risk,
                'tags': ';'.join(tags),
                'score_safe': round(score_safe,6),
                'score_balance': round(score_balance,6),
                'score_attack': round(score_attack,6),
            }

    rows = list(seen.values())
    pools = {
        'safe_pool': ('score_safe', '稳妥优先：先确保高概率有效分'),
        'balance_pool': ('score_balance', '平衡优先：binding/SA/route兼顾'),
        'attack_pool': ('score_attack', '冲分优先：适度提高binding容忍风险'),
        'non_cf3_pool': ('score_balance', '非CF3优先：降低线上SA崩盘风险'),
    }

    summary = []
    for name, (score_key, desc) in pools.items():
        subset = rows[:]
        if name == 'non_cf3_pool':
            subset = [r for r in subset if 'CF3' not in r['tags']]
        subset.sort(key=lambda r: r[score_key], reverse=True)
        top = subset[:12]
        out_csv = OUT / f'{name}.csv'
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['rank','mol_smiles','route','source_version','transform','binding_est','sa_score_est','pred_total','risk','score','mw','logp','tpsa','tags'])
            for i, r in enumerate(top, 1):
                w.writerow([i, r['mol_smiles'], r['route'], r['source_version'], r['transform'], r['binding_est'], r['sa_score_est'], r['pred_total'], r['risk'], r[score_key], r['mw'], r['logp'], r['tpsa'], r['tags']])
        summary.append({
            'version': name,
            'desc': desc,
            'top': top[0] if top else None,
            'csv': str(out_csv),
        })

    with open(OUT / 'submission_priority_stage3.md', 'w', encoding='utf-8') as f:
        f.write('# Stage3 提交优先级建议\n\n')
        f.write('| priority | version | top smiles | binding | sa | total | risk | transform | 说明 |\n')
        f.write('|---|---|---|---:|---:|---:|---:|---|---|\n')
        ordered = [s for s in summary if s['top']]
        ordered.sort(key=lambda s: (s['top']['score_balance'] if 'balance' in s['version'] or 'non_cf3' in s['version'] else s['top']['pred_total']), reverse=True)
        for i, s in enumerate(ordered, 1):
            t = s['top']
            rec = '主推' if i == 1 else '次推' if i <= 3 else '备选'
            f.write(f"| {i} | {s['version']} ({rec}) | `{t['mol_smiles']}` | {t['binding_est']} | {t['sa_score_est']} | {t['pred_total']} | {t['risk']} | {t['transform']} | {s['desc']} |\n")

    with open(OUT / 'all_candidates_stage3.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['mol_smiles'])
        w.writeheader()
        for r in sorted(rows, key=lambda x: x['score_balance'], reverse=True):
            w.writerow(r)

    print(f'Generated {len(rows)} kinase-aware candidates -> {OUT}')
    for s in summary:
        if s['top']:
            print(s['version'], s['top']['mol_smiles'], s['top']['pred_total'], s['top']['score_balance'])

if __name__ == '__main__':
    main()
