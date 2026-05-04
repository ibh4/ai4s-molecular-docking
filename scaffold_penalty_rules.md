# Scaffold Penalty Rules (2026-05-04 更新)

## A. fused polycyclic attack-like scaffold (Stage5)
- pattern: large planar fused isoquinoline-like / quinoline-fused polycycle
- SA cap: 0.20
- if high similarity to Stage5 attack: SA max 0.25
- binding may reach ~0.26 — 唯一线上证明 binding 较高的方向
- **策略：做 shape-based simplification，保留 binding 优势，提升 SA**

## B. compact pyrrolopyrimidine / compact JAK-like anchor
- SA can be 0.55–0.60（线上校准可信）
- binding cap: 0.20（线上 0.148，高估 ~2x）
- 不再推荐为 high-binding anchor

## C. pyridine-pyrimidine extension / aza-aryl amide（Stage13 新增）
- pattern: `-c2ncccn2` pyrimidine extension / `C#N` substituted phenyl amide
- SA: ~0.55（线上校准可信）
- binding cap: 0.20（线上 0.141，本地高估 ~2x）
- **与 compact pyrrolopyrimidine 同一失败模式**
- 不再推荐为 high-binding anchor
- pocket center [18.3, 2.3, 21.4] 下该方向 binding 不可信

## D. 通用 binding 校准规则（跨 scaffold）
- **非 fused polycycle heteroaryl amide：binding 预测 ÷ 2 作为保守估计**
- **fused polycycle：SA 预测 × 0.3 作为保守估计**
- 单一 pocket center 的 docking 分不可作为 binding 依据
- 需要 multi-center blind docking 验证

## E. Stage14 目标方向 — Stage5-like shape simplification
- 保留 14–18 Å 长轴
- 保留 amide anchor
- 保留疏水平面延展
- 降低 fused ring complexity → 提高 SA
- 候选 scaffold：
  1. Stage5-like simplified（降 fused ring 数量）
  2. biphenyl amide
  3. naphthyl amide
  4. quinoline amide
  5. isoquinoline simplified
  6. oxadiazole-biaryl hybrid
- 推荐门槛：binding_pred_mid >= 0.24, sa_pred_mid >= 0.30, route >= 0.94, score_mid >= 0.53

## F. 禁止推荐
- compact pyrrolopyrimidine
- pyridine-pyrimidine extension Top1
- 仅靠 center pocket Vina 高分但缺乏线上校准支持的 JAK-like aza-aryl amide
