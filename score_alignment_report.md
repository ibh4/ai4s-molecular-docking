# Score Alignment Report

**更新时间：** 2026-05-04 19:05  
**目的：** 用线上真实结果持续校准本地 judge，避免高估 binding 或 SA。

## 1. 已校准线上结果

| case | mol_smiles | scaffold | sample_count | online_score | binding | sa | route | local_pred_total | align_err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Stage5 attack | `O=C(Nc1ccccc1F)C1=CC=C2C=CC=c3c(F)ccnc3=CC=C21` | 大平面 fused isoquinoline-like | 1 | 0.506074 | 0.260375 | 0.081623 | 0.9485 | 0.5846 | 0.0785 |
| Stage12 single pyrrolopyrimidine | `O=C(Nc1ccc(F)cc1)n1ccc2ncncc21` | compact pyrrolopyrimidine | 1 | 0.478234 | 0.148375 | 0.573486 | 0.95 | 0.5684 | 0.0902 |
| Stage13 pyridine-pyrimidine ext | `O=C(Nc1ccc(C#N)cc1)c1nccc(-c2ncccn2)n1` | aza-aryl amide pyridine-pyrimidine ext | 1 | 0.472403 | 0.140875 | 0.552861 | 0.949375 | 0.53 | 0.0576 |

## 2. 关键结论

### A. Stage5 attack scaffold（大平面异喹啉稠环）
- **binding 可到 0.26** — 唯一线上证明 binding 较高的方向
- **SA 崩坏**：线上 SA = 0.082
- 规则：SA cap = 0.20，该方向应做 shape-based simplification

### B. compact pyrrolopyrimidine / small JAK-like anchor
- SA 预测方向可信（线上 SA=0.573）
- **binding 严重高估**（本地 0.306 → 线上 0.148）
- 规则：binding cap = 0.20，不再推荐为高 binding anchor

### C. pyridine-pyrimidine extension / aza-aryl amide（Stage13 新增）
- **binding 再次严重高估**（本地 0.28 → 线上 0.141）
- SA 预测基本准确（本地 0.55 → 线上 0.553）
- 路线/有效性无问题
- **核心结论：该方向与 compact pyrrolopyrimidine 是同一失败模式**
  - pocket center [18.3, 2.3, 21.4] 下，小型/中型 heteroaryl amide 的 binding 预测系统性偏高
  - 不是分子太小的问题，是 **pocket center / docking scoring / target interpretation 失配**
- 规则：binding cap = 0.20，不再推荐为 high-binding anchor

### D. 跨三条线上数据的 binding 校准总结

| scaffold class | 线上 binding | 本地 binding | 高估幅度 | cap |
|---|---:|---:|---:|---|
| fused polycycle (Stage5) | 0.260 | 0.33 | ~1.3x | 允许，SA 限制 |
| compact pyrrolopyrimidine | 0.148 | 0.306 | ~2.1x | cap 0.20 |
| pyridine-pyrimidine ext | 0.141 | 0.28 | ~2.0x | cap 0.20 |

**规律：** 非 fused polycycle 的 heteroaryl amide，binding 高估约 2x。只有大平面稠环能真正获得较高 binding。

## 3. Judge 修正规则（binding / SA / scaffold penalty）

### 3.1 SA 校准
- fused polycyclic / 大平面异喹啉稠环：**sa_score cap = 0.20**
- 与 Stage5 attack scaffold 高相似的分子：**sa_score 不允许 > 0.25**
- quinazoline / oxadiazole / naphthamide：可保留较高 SA，但必须保守估计

### 3.2 binding 校准
- **compact pyrrolopyrimidine：binding cap = 0.20**
- **pyridine-pyrimidine extension / aza-aryl amide：binding cap = 0.20**
- 除非满足以下全部条件才允许 >0.20：
  1. 明显 fused hydrophobic scaffold（非单环/双环 heteroaryl）
  2. docking/pose 显示深入 cleft 而不是贴表面
  3. 14–18 Å 长轴，疏水平面延展
  4. MW 350–500, logP 3.0–5.5

### 3.3 下一轮推荐空间（Stage14）
- **主攻方向：Stage5-like shape simplification**
  - 保留 14–18 Å 长轴
  - 保留 amide anchor
  - 保留疏水平面延展
  - 降低 fused ring complexity → 提高 SA
- **候选 scaffold 库：**
  1. Stage5-like simplified（降 fused ring 数量）
  2. biphenyl amide
  3. naphthyl amide
  4. quinoline amide
  5. isoquinoline simplified
  6. oxadiazole-biaryl hybrid
- **推荐门槛：**
  - binding_pred_mid >= 0.24
  - sa_pred_mid >= 0.30
  - route >= 0.94
  - score_mid >= 0.53

### 3.4 禁止推荐
- compact pyrrolopyrimidine
- pyridine-pyrimidine extension Top1
- 仅靠 center pocket Vina 高分但缺乏线上校准支持的 JAK-like aza-aryl amide

## 4. 当前最高有效线上分
- **0.506074**（Stage5 attack single）
- 该 scaffold SA 崩坏，应做 shape simplification 而非继续原 scaffold

## 5. 核心策略切换

**已完成的切换（Stage12→13）：**
- 从 compact pyrrolopyrimidine → pyridine-pyrimidine extension
- 结果：失败，binding 仍然严重高估

**Stage14 切换：**
- 从 JAK-like small heteroaryl → **Stage5-like shape simplification**
- 核心逻辑：只有大平面稠环真正获得线上 binding，但 SA 崩坏
- 解法：保留大平面骨架的 binding 优势，通过简化 fused ring 提升 SA
- 同时进行 **pocket center rescan / blind docking grid scan**，不固定 [18.3, 2.3, 21.4]

## 6. 本地 judge 需要的系统性修正
- 所有非 fused polycycle heteroaryl amide：binding 预测 **除以 2** 作为保守估计
- fused polycycle：SA 预测 **乘以 0.3** 作为保守估计
- pocket center 单一坐标的 docking 分不可作为 binding 依据，需要多 center 验证
