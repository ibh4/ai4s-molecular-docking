# AI4S Agent CNS挑战赛 — 任务2

## 项目结构
```
ai4s_chem/
├── target.pdb          # 靶点蛋白结构
├── result/             # 输出目录
│   ├── result.csv      # 提交文件（分子+路线）
│   └── result.log      # Agent日志
├── receptor.pdbqt      # 受体文件（待生成）
├── pocket.txt          # 对接口袋定义（待生成）
└── candidates/         # 候选分子中间文件
```

## 快速开始
```bash
# 激活环境
cd ~/.openclaw/workspace/retrosyn && source venv/bin/activate

# 运行Agent
python /Users/pwngwc/ai4s_chem/agent.py
```

## 评分权重
- 分子评分 (0.7): binding(0.8) + validity(0.1) + SA(0.1)
- 路线评分 (0.3): route_validity(0.55) + starting_material(0.30) + step_penalty(0.05) + convergence(0.05) + balance(0.05)
