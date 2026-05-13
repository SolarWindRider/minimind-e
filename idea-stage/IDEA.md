# MiniMind-VLA: Temporal Action Modeling for Small VLA Models

## Research Idea

**核心问题**：0.1B参数的小模型能否通过专门的时序动作建模实现真正的具身智能控制？

**核心假设**：通过显式地将历史动作序列注入到模型的决策过程中，小型VLA模型可以学会时序推理能力，其效果接近甚至超越参数量大50倍的模型。

## Key Innovations

### 1. History-Conditioned Action Prediction (HCAP)
- 每个时间步，显式地将前N个历史动作嵌入后与视觉特征融合
- 使用action embedding + adapter机制，避免破坏预训练语言模型
- 类似于"Hidden State Routing"但专门为动作预测设计

### 2. Lightweight Cross-modal Bridge
- 不用复杂的vision-language对其机制
- 仅从Thinker的第K层提取bridge states，直接注入到Action Module
- 通过可学习的gate机制控制信息流

### 3. Temporal Action Autoencoder (TAA)
- 自监督预训练任务：给定部分动作序列，预测下一步
- 类似于BERT的masked LM，但用于动作序列
- 帮助模型学习时序依赖关系

## Experiment Design

### 消融实验矩阵

| 实验 | 变体 | 目标 |
|------|------|------|
| E1 | 历史动作长度：0, 3, 5, 10 | 找到最优历史长度 |
| E2 | Bridge layer：第2/4/6层 | 验证bridge layer选择 |
| E3 | 有/无TAA预训练 | 验证自监督的作用 |
| E4 | Action Module深度：2/4/6层 | 权衡性能和速度 |
| E5 | 有/无cross-modal gate | 验证gate机制 |

### 基线对比

| 模型 | 参数量 | 时序控制 |
|------|--------|----------|
| MiniMind-VLA (Ours) | 0.1B | ✓ 显式历史注入 |
| MiniMind-O (Omni) | 0.1B | ✗ 无时序控制 |
| RT-2-X | 55B | ✓ 隐式历史 |

## Expected Outcomes

1. **主要指标**：FetchSlide任务成功率
2. **时序控制验证**：相同观察不同历史→不同动作
3. **消融分析**：每个组件的贡献

## Timeline

- Week 1: 完成代码，收集数据
- Week 2: 训练基线和主模型
- Week 3: 消融实验和对比
- Week 4: 分析结果，撰写论文