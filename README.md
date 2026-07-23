<img width="1448" height="1086" alt="image" src="https://github.com/user-attachments/assets/20010846-9088-408f-84a9-1010e1f59a4a" />
<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/0e78427c-6b52-47f8-925e-b39ed7749272" />

### 项目目的
本仓库记录了 AI4S 分子对接智能分子优化代理的完整训练过程。目标是发现对靶点蛋白（人类 JAK2 JH2 假激酶结构域）具有强结合亲和力的新型小分子，同时保持良好的合成可及性（SA）和有效的逆合成路线。

### 靶点蛋白 (target.pdb)
| 属性 | 值 |
|------|-----|
| 蛋白名称 | 人类 JAK2 JH2 假激酶结构域 |
| 链 | A |
| 总 ATOM 记录数 | 1976 |
| HETATM 记录数 | 0 |
| 水分子 | 0 |
| 独特残基类型 | 20 |
| 残基 | ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL |

## 在目标蛋白口袋中的 Vina 结合姿态
<img width="1800" height="1518" alt="image" src="https://github.com/user-attachments/assets/12dcf33d-089f-4677-a111-a20a99d313d8" />
图 1 展示了 Stage40 最好的非重复候选 S40_00009 在目标蛋白口袋中的 Vina pose。灰色半透明区域是口袋表面，紫色分子是候选 ligand。

这个图说明：

fused tricyclic acid core 能较好埋入口袋。
amide linker 保留了主 scaffold 的结合方向。
OMe/diF aniline 端位于口袋边缘，适合作为下一轮局部优化位置。
该分子的 surface sticking risk 为 low，但 binding_pred 仍只有 0.3177，距离 0.62 目标所需 binding 还有差距。

<img width="1800" height="1518" alt="image" src="https://github.com/user-attachments/assets/782654e4-af68-4e9d-8ae2-c0b151995aa1" />

图 2 将三个关键分子放在同一个结合口袋中比较：

绿色：S35_00001，当前线上最好参考，score = 0.581261。
橙色：S39_00006，高 binding 参考，online score = 0.577825。
紫色：S40_00009，Stage40 最好非重复 probe，pred score = 0.5774。
这个叠合图说明：

三个分子的 fused tricyclic core 占据的是同一块 pocket-filling 区域。
主要差异集中在 aniline 边缘。
Stage40 的更换逻辑是合理的，但 aniline 边缘的 OMe/diF 改动没有带来足够大的 binding 增益。
后续不能只继续堆大环，而要针对这个边缘区域设计更强的极性或疏水锚点。

<img width="1800" height="1420" alt="image" src="https://github.com/user-attachments/assets/b92de0db-b9e0-4c7e-a00c-aacb00342791" />

图 3 展示了本轮局部优化的核心路径：

S35 -> S39 -> S40_00009

三者保留相似的 fused tricyclic acid scaffold 和 amide linker，只替换 aniline 端：

S35：3,4-diF-aniline，线上最高分。
S39：加入 methyl/diF 边缘，binding 保持强，但 SA 有所下降。
S40_00009：换成 OMe/diF probe，希望提高边缘相互作用，但 binding 没有突破 S35。
这个图说明，本轮不是随机换分子，而是在同一结合姿态基础上做局部替换。问题在于替换后的 edge group 没有形成新的强相互作用，因此分数停留在 0.58 附近。
<img width="1800" height="1420" alt="image" src="https://github.com/user-attachments/assets/21155de7-5c43-488d-8f82-1643bf8406ae" />

图 4：Stage40 的大环/环状 scaffold 遍历样例


xanthene-like ring：测试更大、更刚性的 fused ring。
high-SA naphthoyl：测试更容易合成、SA 更高的轻量环系。
log-safe indane：测试结构更简洁、LLM/log gate 更安全的候选。
结果显示：

这些分子能提供结构多样性。
部分分子的 SA 和 route 更稳定。
但是 calibrated binding 明显低于 S35/S39 fused tricyclic 主线。
因此“大量环状遍历”已经被测试过，结论是：环状 scaffold 本身不是问题，真正缺口是缺少新的 binding anchor。

<img width="1800" height="1518" alt="image" src="https://github.com/user-attachments/assets/acd29f74-8817-4c55-a96e-caeabf587556" />

图 5 是 人工智能设计分子的近距离 interaction 视图。紫色是 ligand，灰色 sticks 是口袋附近残基，黄色虚线表示短距离 polar contact。

这个图说明：

amide linker 仍然是关键 anchor。
aniline 端的 F/OMe 取代基位于可以继续微调的位置。
当前 pose 没有明显 surface sticking，因此结构不是失败样本。
但现有 polar contact 不足以把 binding 推高到 0.36+。
下一轮应重点围绕这个区域做定向设计，例如：

保留 amide anchor。
在 aniline 边缘尝试更合适的 HBA/HBD 或弱极性定位基。
控制 logP，避免纯疏水堆积造成 surface sticking。
保持 acid chloride + aniline 单步路线。
### 高分分子

| 排名 | 名称 | SMILES | Binding | SA | 总分 | 结构 |
|------|------|--------|---------|----|------|------|
| 1 | Stage14a_Top1 (线上最佳) | `O=C(Nc1ccc(C(F)(F)F)cc1F)c1cccc2ccccc12` | 0.224875 | 0.729867 | 0.523255 | ![Stage14a_Top1](images/Stage14a_Top1.png) |
| 2 | Stage16_0001 | `O=C(Nc1ccc(C(F)(F)F)cc1)c1cccc2ccccc12` | 0.2200 | 0.7500 | 0.524700 | ![S15_0002](images/S15_0002.png) |
| 3 | Stage16_0002 | `N#Cc1ccc(NC(=O)c2cccc3ccccc23)cc1` | 0.2200 | 0.7500 | 0.524700 | ![S15_0003](images/S15_0003.png) |
| 4 | Stage16_0003 | `O=C(Nc1ccc(F)cc1)c1cccc2ccccc12` | 0.2200 | 0.7500 | 0.524700 | ![S15_0001](images/S15_0001.png) |

**Stage15 Bug 修复说明**:
- Stage15 所有候选都是相同的 0.4310 分，因为评分逻辑有问题
- Stage16 使用了正确公式: `mol_score = 0.8*binding + 0.1*validity + 0.1*SA; score = 0.7*mol_score + 0.3*route`
- Reference (Stage14a Top1) 被锁定为真实线上得分 0.523255
- 没有新候选明确超过 reference

**Stage14a_Top1 合成路线**: `O=C(Cl)c1cccc2ccccc12.Nc1ccc(C(F)(F)F)cc1F>>O=C(Nc1ccc(C(F)(F)F)cc1F)c1cccc2ccccc12`
(萘甲酰氯 + 取代苯胺 → 目标酰胺)

### 核心评分逻辑
```
总分 = 0.7 × 分子评分 + 0.3 × 路线评分

分子评分 = 0.8 × binding_score
         + 0.1 × validity_score
         + 0.1 × sa_score

路线评分 = 路线有效性
         + 起始原料可获得性
         + 步骤数惩罚
         + 收敛性
         + 原子覆盖/平衡
```

### Agent 工作流 - 闭环优化
Agent 不是"凭空想分子"，而是执行一个自动化科研小循环：

1. **读取 target.pdb** - 分析蛋白结构，识别可能的结合口袋
2. **定义候选空间** - 根据靶点类型选择合适的骨架
3. **生成候选分子** - 文献/数据库驱动、骨架跃迁、取代基枚举
4. **Docking** - 基于 AutoDock Vina 的结合预测
5. **SA & 路线检查** - 验证合成可及性和逆合成路线
6. **提交 & 校准** - 提交到平台，用线上反馈校准下一轮

### 策略演化
- **阶段1**：初始多分子提交 - SA较好，binding一般
- **阶段2**：追求强binding - 大平面稠环，binding提升但SA崩坏
- **阶段3**：JAK样小杂芳 - SA还可以但不被平台认可
- **阶段4**：Stage5风格形状简化 - 保留大疏水平面形状，降低稠环复杂度，改善SA（萘甲酰胺、联苯甲酰胺、喹啉甲酰胺等）
- **Stage14a**：线上最佳提交 - 得分 = 0.523255 (作为校准点被锁定)
- **Stage15**：评分有bug（全是0.4310），公式错误
- **Stage16**：500+候选库，保守评分，没有新分子超过reference

### 关键学习点
1. binding 高不等于总分高 - Stage5 binding 较好但 SA 崩
2. SA 高不等于总分高 - 吡咯并嘧啶 SA 好但 binding 弱
3. 本地 docking 会系统性高估 - 尤其是非稠合杂芳酰胺
4. 路线必须严审 - A→A 伪路线有归零风险
5. 线上反馈是最强老师 - 每次提交都变成新的校准点
6. Agent 最强的地方是批量执行和闭环迭代，不是一次性生成完美分子
# AI4S Molecular Docking Agent

## English

### Competition Purpose
This repository documents the complete training process of an intelligent molecular optimization agent for the AI4S Molecular Docking Competition. The goal is to discover novel small molecules with strong binding affinity to a target protein (human JAK2 JH2 pseudokinase domain), while maintaining good synthetic accessibility (SA) and valid retrosynthesis routes.

### Target Protein (target.pdb)
| Property | Value |
|----------|-------|
| Protein Name | Human JAK2 JH2 pseudokinase domain |
| Chains | A |
| Total ATOM records | 1976 |
| HETATM records | 0 |
| Water molecules | 0 |
| Unique residue types | 20 |
| Residues | ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL |

### Top Performing Molecules

| Rank | Name | SMILES | Binding | SA | Total | Structure |
|------|------|--------|---------|----|-------|-----------|
| 1 | Stage14a_Top1 (Online Best) | `O=C(Nc1ccc(C(F)(F)F)cc1F)c1cccc2ccccc12` | 0.224875 | 0.729867 | 0.523255 | ![Stage14a_Top1](images/Stage14a_Top1.png) |
| 2 | Stage16_0001 | `O=C(Nc1ccc(C(F)(F)F)cc1)c1cccc2ccccc12` | 0.2200 | 0.7500 | 0.524700 | ![S15_0002](images/S15_0002.png) |
| 3 | Stage16_0002 | `N#Cc1ccc(NC(=O)c2cccc3ccccc23)cc1` | 0.2200 | 0.7500 | 0.524700 | ![S15_0003](images/S15_0003.png) |
| 4 | Stage16_0003 | `O=C(Nc1ccc(F)cc1)c1cccc2ccccc12` | 0.2200 | 0.7500 | 0.524700 | ![S15_0001](images/S15_0001.png) |

**Important Stage15 Bug Fix Note**:
- In Stage15, all candidates had the same score (0.4310) due to wrong scoring logic
- In Stage16, correct formula applied: `mol_score = 0.8*binding + 0.1*validity + 0.1*SA; score = 0.7*mol_score + 0.3*route`
- Reference (Stage14a Top1) is LOCKED to actual online score (0.523255)
- No new candidate clearly exceeds reference

**Stage14a_Top1 Route**: `O=C(Cl)c1cccc2ccccc12.Nc1ccc(C(F)(F)F)cc1F>>O=C(Nc1ccc(C(F)(F)F)cc1F)c1cccc2ccccc12`
(naphthoyl chloride + substituted aniline → target amide)

### Core Scoring Logic
```
Total Score = 0.7 × Molecule Score + 0.3 × Route Score

Molecule Score = 0.8 × binding_score
               + 0.1 × validity_score
               + 0.1 × sa_score

Route Score = route_validity
            + starting_material_availability
            + step_penalty
            + convergence
            + atom_coverage_balance
```

### Agent Workflow - Closed Loop Optimization
The agent doesn't "invent molecules out of thin air" - it executes an automated research cycle:

1. **Read target.pdb** - Analyze protein structure, identify potential binding pockets
2. **Define candidate space** - Select appropriate scaffolds based on target type
3. **Generate candidates** - Literature/database driven, scaffold hopping, substituent enumeration
4. **Docking** - AutoDock Vina-based binding prediction
5. **SA & Route Check** - Validate synthetic accessibility and retrosynthesis routes
6. **Submit & Calibrate** - Submit to platform, use online feedback to calibrate next round

### Strategy Evolution
- **Stage 1**: Initial multi-molecule submission - Good SA, mediocre binding
- **Stage 2**: Pursue strong binding - Large planar fused rings, improved binding but SA collapsed
- **Stage 3**: JAK-like small heteroaryls - Good SA but not recognized by platform scoring
- **Stage 4**: Stage5-like shape simplification - Keep large hydrophobic planar shape, reduce fused ring complexity, improve SA (naphthyl amide, biphenyl amide, quinoline amide, etc.)
- **Stage 14a**: Best online submission - Score = 0.523255 (LOCKED as calibration)
- **Stage 15**: Buggy scoring (all 0.4310) due to wrong formula
- **Stage 16**: 500+ candidate library, conservative scoring, no new molecule exceeds reference

### Key Learnings
1. High binding ≠ high total score - Stage5 had good binding but SA collapsed
2. High SA ≠ high total score - Pyrrolopyrimidines had good SA but weak binding
3. Local docking can be systematically overestimated - Especially non-fused heteroaryl amides
4. Routes must be strictly validated - A→A pseudo-routes risk zero score
5. Online feedback is the strongest teacher - Each submission becomes a new calibration point
6. Agent's strength is batch execution and closed-loop iteration, not perfect one-time generation

### Key Files
- `agent.py`: Main intelligent agent
- `candidate_rerank.py`: Candidate ranking & optimization
- `stage14_phase01_calibration_shape.py`: Shape calibration
- `stage14_phase2_pocket_rescan.py`: Pocket rescanning
- `route_fix_v3.py`: Route validation & repair

---



### 核心文件
- `agent.py`: 主要智能代理
- `candidate_rerank.py`: 候选分子排序与优化
- `stage14_phase01_calibration_shape.py`: 形状校准
- `stage14_phase2_pocket_rescan.py`: 口袋重新扫描
- `route_fix_v3.py`: 路线验证与修复
