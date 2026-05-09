"""
VLA 评估脚本

支持两种评估模式：
1. 离线评估：在收集的数据集上评估
2. 在线评估：直接在仿真器上评估（使用与训练不同的随机种子）
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import argparse
import os
import json
import sys
from tqdm import tqdm
from typing import List, Dict, Tuple
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_vla import MiniMindVLAStep, VLAConfig
from trainer.trainer_utils import load_tokenizer


DISCRETE_ACTIONS = [
    "move_left", "move_right", "move_forward", "move_backward",
    "move_up", "move_down", "grip_close", "grip_open", "stay"
]


def discrete_to_continuous(action_idx, magnitude=0.05):
    action = np.zeros(4)
    if action_idx == 0: action[0] = -magnitude
    elif action_idx == 1: action[0] = magnitude
    elif action_idx == 2: action[1] = magnitude
    elif action_idx == 3: action[1] = -magnitude
    elif action_idx == 4: action[2] = magnitude
    elif action_idx == 5: action[2] = -magnitude
    elif action_idx == 6: action[3] = -magnitude
    elif action_idx == 7: action[3] = magnitude
    return action


def encode_state_as_text(obs):
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    rel_x, rel_y, rel_z = achieved_goal - desired_goal
    dist = np.linalg.norm([rel_x, rel_y, rel_z])

    instructions = []
    if rel_x < -0.05: instructions.append("puck is to the right of target")
    elif rel_x > 0.05: instructions.append("puck is to the left of target")
    if rel_y < -0.05: instructions.append("puck is behind target")
    elif rel_y > 0.05: instructions.append("puck is in front of target")
    if rel_z < -0.02: instructions.append("puck is below target")
    elif rel_z > 0.02: instructions.append("puck is above target")
    if dist < 0.05: instructions.append("puck is close to goal")
    elif dist > 0.2: instructions.append("puck is far from goal")
    if not instructions: instructions.append("adjust puck position")
    return " ".join(instructions)


def compute_metrics(pred_actions: List[str], gt_actions: List[str]) -> Dict:
    """计算动作预测准确率"""
    if len(pred_actions) != len(gt_actions):
        min_len = min(len(pred_actions), len(gt_actions))
        pred_actions = pred_actions[:min_len]
        gt_actions = gt_actions[:min_len]

    correct = sum(1 for p, g in zip(pred_actions, gt_actions) if p == g)
    accuracy = correct / len(gt_actions) if gt_actions else 0.0

    return {
        "accuracy": accuracy,
        "num_steps": len(gt_actions),
        "correct_steps": correct,
    }


class VLARobotEvaluator:
    def __init__(self, model, tokenizer, device='cuda', max_history=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_history = max_history
        self.model.eval()

    def construct_input(self, instruction: str, history: List[str]) -> torch.Tensor:
        prompt = f"Task: {instruction}"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        input_ids = [self.tokenizer.bos_token_id] + prompt_ids + [2]
        for action in history[-self.max_history:]:
            input_ids.append(self.model.encode_action(action))
        input_ids.append(1)
        return torch.tensor([input_ids], dtype=torch.long, device=self.device)

    def predict_action(self, instruction: str, history: List[str], pixel_values=None) -> Tuple[int, float]:
        """
        预测下一动作

        Returns:
            action_idx: 动作索引 (0-8)，直接可用于 discrete_to_continuous()
            confidence: 置信度
        """
        input_ids = self.construct_input(instruction, history)

        with torch.no_grad():
            action_idx, confidence = self.model.predict_next_action(
                input_ids, pixel_values=pixel_values, temperature=0.7
            )

        return action_idx, float(confidence)


def evaluate_on_dataset(evaluator: VLARobotEvaluator, dataset_path: str, max_samples=1000) -> Dict:
    """在离线数据集上评估"""
    with open(dataset_path) as f:
        data = json.load(f)

    trajectories = data.get('trajectories', [])
    if isinstance(data, dict) and 'trajectories' not in data:
        trajectories = data

    trajectories = trajectories[:max_samples]

    all_metrics = []
    for traj in tqdm(trajectories, desc="Evaluating on dataset"):
        instruction = traj.get('instruction', '')
        steps = traj.get('steps', [])

        if not steps:
            continue

        history = []
        pred_actions = []
        gt_actions = []

        for step in steps:
            gt_action = step.get('action', '')
            if not gt_action:
                continue

            action_idx, _ = evaluator.predict_action(instruction, history)

            pred_actions.append(DISCRETE_ACTIONS[action_idx])
            gt_actions.append(gt_action)
            history.append(DISCRETE_ACTIONS[action_idx])

        metrics = compute_metrics(pred_actions, gt_actions)
        all_metrics.append(metrics)

    avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
    avg_steps = np.mean([m['num_steps'] for m in all_metrics])

    return {
        "accuracy": avg_accuracy,
        "avg_steps": avg_steps,
        "num_episodes": len(all_metrics),
    }


def evaluate_on_simulator(evaluator: VLARobotEvaluator, env_id: str, num_episodes: int = 100,
                          seed: int = 99999, max_steps: int = 50) -> Dict:
    """
    在仿真器上直接评估

    使用与训练不同的随机种子，确保测试环境是模型未见过的
    """
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id, render_mode='rgb_array')

    all_results = []
    success_count = 0

    for episode in tqdm(range(num_episodes), desc="Evaluating on simulator"):
        obs, info = env.reset(seed=seed + episode)

        instruction = encode_state_as_text(obs)
        history = []
        episode_reward = 0

        for step in range(max_steps):
            action_idx, _ = evaluator.predict_action(instruction, history)

            # 直接用 action_idx，不需要字符串转换
            continuous_action = discrete_to_continuous(action_idx)
            obs, reward, terminated, truncated, info = env.step(continuous_action)

            episode_reward += reward
            history.append(DISCRETE_ACTIONS[action_idx])

            if terminated or truncated:
                if reward > -10:  # 成功阈值
                    success_count += 1
                break

        all_results.append({
            "episode": episode,
            "reward": episode_reward,
            "steps": len(history),
            "success": terminated or episode_reward > -10,
        })

    env.close()

    success_rate = success_count / num_episodes
    avg_reward = np.mean([r['reward'] for r in all_results])
    avg_steps = np.mean([r['steps'] for r in all_results])

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "num_episodes": num_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA robot controller")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to VLA checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument("--vision_model_path", type=str, default="./model/siglip2-base-p32-256-ve")
    parser.add_argument("--eval_mode", choices=['dataset', 'simulator', 'both'], default='simulator', help="Evaluation mode")
    parser.add_argument("--dataset_path", type=str, default="../dataset/robot_data/test_trajectories.json", help="Path to test dataset")
    parser.add_argument("--env_id", type=str, default="FetchSlide-v3", help="Gymnasium environment ID")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes for evaluation")
    parser.add_argument("--test_seed", type=int, default=12345, help="Test set random seed")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples from dataset")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--output_path", type=str, default="./eval_results.json", help="Output path for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    print(f"Loading VLA from {args.checkpoint_path}...")

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
    if os.path.exists(checkpoint_file):
        model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
    model.set_action_vocabulary(action_vocab)
    model = model.to(args.device)

    evaluator = VLARobotEvaluator(model, tokenizer, device=args.device)

    results = {}

    if args.eval_mode in ['dataset', 'both']:
        if os.path.exists(args.dataset_path):
            print(f"\n=== Evaluating on Dataset: {args.dataset_path} ===")
            dataset_metrics = evaluate_on_dataset(evaluator, args.dataset_path, args.max_samples)
            results['dataset'] = dataset_metrics
            print(f"Dataset Accuracy: {dataset_metrics['accuracy']:.4f}")
        else:
            print(f"Dataset not found: {args.dataset_path}")

    if args.eval_mode in ['simulator', 'both']:
        print(f"\n=== Evaluating on Simulator (seed={args.test_seed}) ===")
        sim_metrics = evaluate_on_simulator(
            evaluator, args.env_id, args.num_episodes, args.test_seed, args.max_steps
        )
        results['simulator'] = sim_metrics
        print(f"Success Rate: {sim_metrics['success_rate']:.4f}")
        print(f"Average Reward: {sim_metrics['avg_reward']:.3f}")
        print(f"Average Steps: {sim_metrics['avg_steps']:.1f}")

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
