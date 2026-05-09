# MiniMind-VLA: 真正的时序控制 VLA 模型

## 核心特性

**真正的时序控制能力**：每个时间步接收 (图像/状态 + 历史动作) → 输出下一动作

## 架构

```
时间步 t:
输入 → Thinker (理解) → Action Module (决策) → A_t
                              ↑
历史动作 ←─────────────────────┘
```

## 安装

```bash
pip install gymnasium gymnasium-robotics mujoco pillow
```

## 数据收集（训练集 + 测试集）

```bash
# 收集训练集和测试集（使用不同随机种子确保不重叠）
python scripts/collect_robot_data.py \
    --env_id FetchSlide-v3 \
    --num_episodes 1000 \
    --train_ratio 0.8 \
    --train_seed 42 \
    --test_seed 12345 \
    --output_dir dataset/robot_data
```

**重要**：
- 训练集 seed=42，测试集 seed=12345
- 测试集使用与训练集完全不同的初始状态
- 评估时只用测试集，模拟泛化到未见过的任务

## 训练

```bash
cd trainer
bash train_vla.sh
# 训练数据: dataset/robot_data/train_trajectories.json
```

## 评估（测试集）

### 方式 1：离线评估（数据集）

```bash
python eval_vla.py \
    --checkpoint_path out/vla/final \
    --eval_mode dataset \
    --dataset_path dataset/robot_data/test_trajectories.json
```

### 方式 2：在线评估（仿真器）

```bash
python eval_vla.py \
    --checkpoint_path out/vla/final \
    --eval_mode simulator \
    --env_id FetchSlide-v3 \
    --num_episodes 100 \
    --test_seed 12345
```

### 方式 3：同时评估

```bash
python eval_vla.py \
    --checkpoint_path out/vla/final \
    --eval_mode both
```

## 实验设计建议

### 核心研究问题

**"小规模 VLA (0.1B) 能否具备真正的时序控制能力？"**

### 评估指标

| 指标 | 说明 |
|------|------|
| **Success Rate** | 任务完成率（主要指标） |
| **Episode Reward** | 累计奖励 |
| **Step Accuracy** | 每步动作预测准确率 |
| **History Sensitivity** | 历史依赖性 |

### 消融实验

| 消融项 | 变体 |
|--------|------|
| Bridge Layer | 无 / 第3层 / 第5层 |
| Action Module 深度 | 2层 / 4层 / 6层 |
| 历史动作长度 | 0 / 3 / 5 / 10 |
| 数据规模 | 100 / 1K / 10K episodes |

### 对比基线

| 模型 | 参数量 | 说明 |
|------|--------|------|
| **MiniMind-VLA (Ours)** | 0.1B | 本方案 |
| MiniMind-O (Omni) | 0.1B | 无时序控制版本 |
| RT-2-X | 55B | 斯坦福 VLA 基线 |

## 时序控制验证实验

**验证方法**：给定相同初始观察，不同历史动作序列，观察模型输出差异

```
初始观察: O
历史 A: [move_left] → 模型应输出不同动作
历史 B: [move_right] → 模型应输出不同动作

如果输出确实不同 → 证明模型利用了历史信息
```

## 数据格式

```json
{
  "discrete_actions": ["move_left", "move_right", ...],
  "train_seed": 42,
  "test_seed": 12345,
  "trajectories": [
    {
      "instruction": "puck is to the right of target",
      "initial_state": {
        "achieved_goal": [x, y, z],
        "desired_goal": [x, y, z]
      },
      "steps": [
        {"action": "move_left", "observation_text": "..."},
        ...
      ]
    }
  ]
}
```

## 局限性

1. **参数规模**：0.1B 规模限制复杂任务能力
2. **离散动作**：当前使用离散动作，可扩展到连续动作
3. **无 RL**：仅有 SFT，可参考 ERA 加入 Online RL

## 致谢

基于 [MiniMind-O](https://github.com/jingyaogong/minimind-o) 和 [ERA](https://embodied-reasoning-agent.github.io/) 开发。
