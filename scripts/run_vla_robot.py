"""
VLA 控制器：与 Gymnasium-Robotics 仿真器交互

实现真正的时序控制：
观察 → 动作 → 执行 → 新观察 → ...

支持 Barge-in：用户可以随时打断当前执行
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import argparse
import os
import sys
from PIL import Image
from typing import List, Tuple, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_vla import MiniMindVLAStep, VLAController, VLAConfig
from trainer.trainer_utils import load_tokenizer


# 离散动作定义
DISCRETE_ACTIONS = [
    "move_left",      # dx < 0
    "move_right",     # dx > 0
    "move_forward",   # dy > 0
    "move_backward",  # dy < 0
    "move_up",        # dz > 0
    "move_down",      # dz < 0
    "grip_close",     # gripper close
    "grip_open",      # gripper open
    "stay",           # no movement
]


def discrete_to_continuous(action_idx, magnitude=0.05):
    """将离散动作转换为连续动作"""
    action = np.zeros(4)

    if action_idx == 0:   # move_left
        action[0] = -magnitude
    elif action_idx == 1:  # move_right
        action[0] = magnitude
    elif action_idx == 2:  # move_forward
        action[1] = magnitude
    elif action_idx == 3:  # move_backward
        action[1] = -magnitude
    elif action_idx == 4:  # move_up
        action[2] = magnitude
    elif action_idx == 5:  # move_down
        action[2] = -magnitude
    elif action_idx == 6:  # grip_close
        action[3] = -magnitude
    elif action_idx == 7:  # grip_open
        action[3] = magnitude
    else:  # stay
        pass

    return action


def encode_state_as_text(obs):
    """将观察编码为文本指令"""
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']

    rel_x = achieved_goal[0] - desired_goal[0]
    rel_y = achieved_goal[1] - desired_goal[1]
    rel_z = achieved_goal[2] - desired_goal[2]

    dist_to_goal = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)

    instructions = []

    if rel_x < -0.05:
        instructions.append("puck is to the right of target")
    elif rel_x > 0.05:
        instructions.append("puck is to the left of target")

    if rel_y < -0.05:
        instructions.append("puck is behind target")
    elif rel_y > 0.05:
        instructions.append("puck is in front of target")

    if rel_z < -0.02:
        instructions.append("puck is below target")
    elif rel_z > 0.02:
        instructions.append("puck is above target")

    if dist_to_goal < 0.05:
        instructions.append("puck is close to goal")
    elif dist_to_goal > 0.2:
        instructions.append("puck is far from goal")

    if not instructions:
        instructions.append("adjust puck position")

    return " ".join(instructions)


class RobotVLAController:
    """
    机械臂 VLA 控制器
    将 VLA 模型与 Gymnasium-Robotics 环境连接
    """

    def __init__(self, model: MiniMindVLAStep, tokenizer, device='cuda', max_history=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_history = max_history
        self.model.eval()

    def construct_input(self, instruction: str, history_actions: List[str]) -> torch.Tensor:
        """构造模型输入"""
        prompt = f"Task: {instruction}"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids

        input_ids = [self.tokenizer.bos_token_id] + prompt_ids + [2]  # 2 = <start>

        for action in history_actions[-self.max_history:]:
            action_id = self.model.encode_action(action)
            input_ids.append(action_id)

        input_ids.append(1)  # 1 = <stop>

        return torch.tensor([input_ids], dtype=torch.long, device=self.device)

    def predict_action(self, instruction: str, history_actions: List[str], pixel_values=None) -> Tuple[int, float]:
        """
        预测下一动作

        Returns:
            action_idx: 动作索引 (0-8)，直接可用于 discrete_to_continuous()
            confidence: 置信度
        """
        input_ids = self.construct_input(instruction, history_actions)

        if pixel_values is not None:
            if isinstance(pixel_values, dict):
                pixel_values = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in pixel_values.items()}
            elif isinstance(pixel_values, np.ndarray):
                pixel_values = torch.from_numpy(pixel_values).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_idx, confidence = self.model.predict_next_action(
                input_ids,
                pixel_values=pixel_values,
                temperature=0.7,
            )

        return action_idx, float(confidence)

    def run_episode(self, env, instruction: str = None, max_steps: int = 50,
                   interrupt_fn: Optional[Callable[[], bool]] = None) -> dict:
        """
        运行一个完整的 episode

        Args:
            env: Gymnasium 环境
            instruction: 任务指令（如果为 None，从初始观察生成）
            max_steps: 最大步数
            interrupt_fn: 中断检测函数，返回 True 表示需要打断

        Returns:
            result: 包含执行动作列表、奖励等信息
        """
        obs, info = env.reset()

        if instruction is None:
            instruction = encode_state_as_text(obs)

        history = []
        executed_actions = []
        rewards = []
        images = []

        print(f"Starting episode with instruction: {instruction}")

        for step in range(max_steps):
            # 检查打断
            if interrupt_fn and interrupt_fn():
                print(f"Interrupted at step {step}")
                break

            # 获取当前图像
            img = env.render()
            if isinstance(img, list):
                img = img[0]
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            images.append(img)

            # 更新指令（基于当前状态）
            current_instruction = encode_state_as_text(obs)

            # 预测动作
            action_idx, confidence = self.predict_action(
                current_instruction,
                history,
                pixel_values=None  # 当前版本用文本指令，不使用图像
            )

            print(f"Step {step}: action={DISCRETE_ACTIONS[action_idx]} (conf={confidence:.3f})")

            # 转换为连续动作并执行
            continuous_action = discrete_to_continuous(action_idx)
            obs, reward, terminated, truncated, info = env.step(continuous_action)

            executed_actions.append(DISCRETE_ACTIONS[action_idx])
            history.append(DISCRETE_ACTIONS[action_idx])
            rewards.append(reward)

            if terminated or truncated:
                print(f"Episode finished at step {step}, reward={sum(rewards):.3f}")
                break

        return {
            "instruction": instruction,
            "executed_actions": executed_actions,
            "total_reward": sum(rewards),
            "num_steps": len(executed_actions),
            "success": terminated or sum(rewards) > -10,
            "images": images,
        }


def demo_with_random_policy(env_id='FetchSlide-v3', num_episodes=3):
    """使用随机策略演示环境"""
    gym.register_envs(gymnasium_robotics)

    env = gym.make(env_id, render_mode='rgb_array')

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode} ===")
        obs, info = env.reset()

        instruction = encode_state_as_text(obs)
        print(f"Instruction: {instruction}")

        total_reward = 0
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward

            if step % 10 == 0:
                img = env.render()
                if img is not None:
                    Image.fromarray(img).save(f"demo_step_{episode}_{step}.png")

            if terminated or truncated:
                break

        print(f"Episode {episode} finished, total reward: {total_reward:.3f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Run VLA with Gymnasium-Robotics")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to VLA checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument("--vision_model_path", type=str, default="./model/siglip2-base-p32-256-ve")
    parser.add_argument("--env_id", type=str, default="FetchSlide-v3", help="Gymnasium environment ID")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--demo_random", action="store_true", help="Demo with random policy")
    args = parser.parse_args()

    if args.demo_random:
        print("Running demo with random policy...")
        demo_with_random_policy(args.env_id, args.num_episodes)
        return

    print(f"Loading VLA from {args.checkpoint_path}...")

    # 加载模型
    tokenizer = load_tokenizer(args.tokenizer_path)

    checkpoint_file = os.path.join(args.checkpoint_path, "pytorch_model.bin")
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=args.device)
        vla_config = checkpoint.get('config', VLAConfig())
        action_vocab = checkpoint.get('action_vocab', DISCRETE_ACTIONS)
    else:
        vla_config = VLAConfig()
        action_vocab = DISCRETE_ACTIONS

    model = MiniMindVLAStep(vla_config, vision_model_path=args.vision_model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
    model.set_action_vocabulary(action_vocab)
    model = model.to(args.device)

    # 创建控制器
    controller = RobotVLAController(model, tokenizer, device=args.device)

    # 创建环境
    gym.register_envs(gymnasium_robotics)
    env = gym.make(args.env_id, render_mode='rgb_array')

    # 运行 episodes
    results = []
    for episode in range(args.num_episodes):
        print(f"\n=== Episode {episode} ===")
        result = controller.run_episode(env, max_steps=args.max_steps)
        results.append(result)
        print(f"Episode {episode}: {result['num_steps']} steps, reward={result['total_reward']:.3f}, success={result['success']}")

    # 统计
    success_rate = sum(r['success'] for r in results) / len(results)
    avg_reward = sum(r['total_reward'] for r in results) / len(results)
    avg_steps = sum(r['num_steps'] for r in results) / len(results)

    print(f"\n=== Summary ===")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Average steps: {avg_steps:.1f}")

    env.close()


if __name__ == "__main__":
    main()
