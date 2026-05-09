# MiniMind-VLA: 真正的视觉-语言-动作模型

## 项目概述

MiniMind-VLA 是将 MiniMind-O（文本/语音/图像 → 文本/语音）改造而成的**真正的**视觉-语言-动作模型。

### 与 ERA 的关系

本项目借鉴了 [ERA (Embodied Reasoning Agent)](https://embodied-reasoning-agent.github.io/) 的思路。ERA 使用 3B 参数模型，本项目探索 **0.1B 小模型**做 VLA 的潜力。

### 核心区别

| 特性 | Mask Prediction (ERA EPL) | 真正的 VLA (MiniMind-VLA) |
|------|---------------------------|---------------------------|
| 输入 | instruction + 不完整 action seq + [MASK] | 图像 + instruction |
| 输出 | 预测 [MASK] 位置填哪个 action | **自己生成**完整 action 序列 |
| 训练目标 | 分类（从 vocabulary 选一个） | 序列生成（像翻译一样） |

## 架构说明

```
输入: 图像 + "Put the cup on the table"
         ↓
    ┌─────────────┐
    │   SigLIP2   │  视觉编码器 (冻结)
    │  Vision     │
    │  Encoder    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Vision    │  投影层 (训练)
    │  Projector │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │             │
    │   Thinker   │  MiniMind Transformer (训练)
    │   (8层)     │
    │             │
    └──────┬──────┘
           ↓ bridge_layer
    ┌─────────────┐
    │             │
    │   Action    │  Action Module (训练)
    │   (4层)     │
    │             │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Action     │  输出: 动作序列
    │  Head       │  "find cup → pick up → ..."
    └─────────────┘
```

## 数据格式

**真正的 VLA 数据格式：**
```json
{
  "instruction": "Put the cup on the table",
  "actions": ["find a cup", "pick up the cup", "find a table", "put down the cup"],
  "image": "/path/to/image.jpg"
}
```

## 快速开始

### 1. 环境准备

```bash
pip install torch transformers pyarrow pandas pillow
pip install -r requirements.txt
```

### 2. 下载预训练模型

```bash
# 下载 SigLIP2 视觉编码器
modelscope download --model gongjy/siglip2-base-p32-256-ve --local_dir ./model/siglip2-base-p32-256-ve

# 下载 MiniMind 语言模型权重
modelscope download --model gongjy/minimind-3o-pytorch llm_768.pth --local_dir ./out
```

### 3. 下载 ERA 数据集

```bash
git clone https://huggingface.co/datasets/EmbodiedReasoningAgent/EB-ALFRED_trajectory_augmented_prior_dataset
git clone https://huggingface.co/datasets/EmbodiedReasoningAgent/EB-ALFRED_environment_anchored_prior_dataset
```

### 4. 数据格式转换

```bash
python scripts/convert_era_to_vla.py \
    --input_path /path/to/EB-ALFRED_trajectory_augmented_prior_dataset/data.json \
    --output_path dataset/vla_alfred.json \
    --images_folder /path/to/images
```

### 5. 开始训练

```bash
cd trainer
bash train_vla.sh
```

### 6. 推理评估

```bash
cd scripts
bash eval_vla.sh
```

## 局限性

1. **参数规模**：0.1B 规模远小于 ERA 的 3B
2. **动作空间**：仅支持离散动作，不支持连续动作控制
3. **无 RL 阶段**：仅有 SFT 训练

## 致谢

基于 [MiniMind-O](https://github.com/jingyaogong/minimind-o) 和 [ERA](https://embodied-reasoning-agent.github.io/) 开发。
