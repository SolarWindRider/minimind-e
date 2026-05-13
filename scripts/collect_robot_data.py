"""
Gymnasium-Robotics 数据收集器
用于收集 Fetch 机械臂的 (图像, 动作) 轨迹数据
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from PIL import Image
import io


# 动作空间离散化：将连续的 4D 动作映射到离散的文字动作
DISCRETE_ACTIONS = [
    "move_left",      # dx < 0
    "move_right",     # dx > 0
    "move_forward",   # dy > 0 (robot forward)
    "move_backward",  # dy < 0
    "move_up",        # dz > 0
    "move_down",      # dz < 0
    "grip_close",     # gripper close
    "grip_open",      # gripper open
    "stay",           # no movement
]

ACTION_THRESHOLD = 0.1  # 动作阈值

def continuous_to_discrete(action):
    """
    将连续的 4D 动作 (dx, dy, dz, gripper) 映射到离散动作索引

    Args:
        action: numpy array of shape (4,) with values in [-1, 1]

    Returns:
        action_idx: 离散动作索引
    """
    dx, dy, dz, gripper = action

    # 位置移动
    if abs(dx) > ACTION_THRESHOLD:
        if dx < 0:
            return 0  # move_left
        else:
            return 1  # move_right
    elif abs(dy) > ACTION_THRESHOLD:
        if dy > 0:
            return 2  # move_forward
        else:
            return 3  # move_backward
    elif abs(dz) > ACTION_THRESHOLD:
        if dz > 0:
            return 4  # move_up
        else:
            return 5  # move_down
    else:
        # 夹爪控制
        if gripper < -ACTION_THRESHOLD:
            return 6  # grip_close
        elif gripper > ACTION_THRESHOLD:
            return 7  # grip_open
        else:
            return 8  # stay


def discrete_to_continuous(action_idx, magnitude=0.05):
    """
    将离散动作索引转换回连续动作

    Args:
        action_idx: 离散动作索引
        magnitude: 动作幅度

    Returns:
        action: 连续的 4D 动作 numpy array
    """
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
    """
    将低维状态编码为文本描述（作为 VLA 的指令输入）

    Args:
        obs: observation dict from Gymnasium-Robotics

    Returns:
        instruction: 文本形式的指令
    """
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']

    # 计算相对位置
    rel_x = achieved_goal[0] - desired_goal[0]
    rel_y = achieved_goal[1] - desired_goal[1]
    rel_z = achieved_goal[2] - desired_goal[2]

    dist_to_goal = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)

    instructions = []

    # 方向描述
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

    # 距离描述
    if dist_to_goal < 0.05:
        instructions.append("puck is close to goal")
    elif dist_to_goal > 0.2:
        instructions.append("puck is far from goal")

    if not instructions:
        instructions.append("adjust puck position")

    return " ".join(instructions)


def collect_trajectories(env_id='FetchSlide-v4', num_episodes=100, render_mode='rgb_array', max_steps=50, seed=None):
    """
    收集轨迹数据

    Args:
        env_id: Gymnasium 环境 ID
        num_episodes: 收集的 episode 数量
        render_mode: 'rgb_array' 或 'human'
        max_steps: 每个 episode 的最大步数
        seed: 随机种子（用于生成确定性的初始状态）

    Returns:
        trajectories: 轨迹列表
    """
    gym.register_envs(gymnasium_robotics)

    env = gym.make(env_id, render_mode=render_mode)

    trajectories = []

    for episode in tqdm(range(num_episodes), desc="Collecting trajectories"):
        # 设置种子以确保可重复性
        if seed is not None:
            obs, info = env.reset(seed=seed + episode)
        else:
            obs, info = env.reset()

        episode_trajectory = {
            "instruction": encode_state_as_text(obs),
            "steps": [],
            "episode_reward": 0,
            "initial_state": {
                "achieved_goal": obs['achieved_goal'].tolist(),
                "desired_goal": obs['desired_goal'].tolist(),
            },
            "episode_idx": episode,
        }

        for step in range(max_steps):
            # 获取当前观察的文本描述（不渲染图像，因为无头服务器没有OpenGL）
            instruction = encode_state_as_text(obs)

            # 使用随机策略采集数据（也可以用训练好的策略）
            continuous_action = env.action_space.sample()
            discrete_action = continuous_to_discrete(continuous_action)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(continuous_action)

            episode_trajectory["steps"].append({
                "image": [],  # No image on headless server
                "action": DISCRETE_ACTIONS[discrete_action],
                "action_idx": int(discrete_action),
                "observation_text": instruction,
            })

            episode_trajectory["episode_reward"] += reward

            if terminated or truncated:
                break

        trajectories.append(episode_trajectory)

    env.close()

    return trajectories


def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_trajectories(trajectories, output_path):
    """保存轨迹数据"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # 保存为 JSON（适用于小数据集）
    serializable_trajectories = convert_to_serializable(trajectories)
    with open(output_path, 'w') as f:
        json.dump({
            "discrete_actions": DISCRETE_ACTIONS,
            "action_threshold": ACTION_THRESHOLD,
            "num_episodes": len(serializable_trajectories),
            "trajectories": serializable_trajectories,
        }, f, indent=2)

    print(f"Saved {len(serializable_trajectories)} trajectories to {output_path}")


def save_as_images(trajectories, output_dir):
    """
    保存轨迹图像到单独文件夹（适用于大数据集）
    """
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    metadata = {
        "discrete_actions": DISCRETE_ACTIONS,
        "action_threshold": ACTION_THRESHOLD,
        "num_episodes": 0,
        "trajectories": [],
    }

    img_count = 0
    for episode_idx, traj in enumerate(trajectories):
        episode_steps = []
        for step_idx, step in enumerate(traj["steps"]):
            if isinstance(step["image"], list):
                img_array = np.array(step["image"])
            else:
                img_array = step["image"]

            img_path = f"episode_{episode_idx:04d}_step_{step_idx:04d}.png"
            img_full_path = os.path.join(img_dir, img_path)

            Image.fromarray(img_array).save(img_full_path)

            episode_steps.append({
                "image": img_path,
                "action": step["action"],
                "action_idx": step["action_idx"],
                "observation_text": step["observation_text"],
            })

            img_count += 1

        metadata["trajectories"].append({
            "episode_idx": episode_idx,
            "instruction": traj["instruction"],
            "steps": episode_steps,
            "episode_reward": traj["episode_reward"],
        })

    metadata["num_episodes"] = len(trajectories)
    metadata["num_images"] = img_count

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {img_count} images to {img_dir}")
    print(f"Saved metadata to {os.path.join(output_dir, 'metadata.json')}")


def main():
    parser = argparse.ArgumentParser(description="Collect trajectories from Gymnasium-Robotics")
    parser.add_argument("--env_id", type=str, default="FetchSlide-v3", help="Gymnasium environment ID")
    parser.add_argument("--num_episodes", type=int, default=100, help="Total number of episodes to collect")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--train_seed", type=int, default=42, help="Random seed for training set")
    parser.add_argument("--test_seed", type=int, default=12345, help="Random seed for test set")
    parser.add_argument("--output_dir", type=str, default="../dataset/robot_data", help="Output directory")
    parser.add_argument("--save_images", action="store_true", help="Save images separately instead of in JSON")
    parser.add_argument("--split", choices=['train', 'test', 'both'], default='both', help="Which split to collect")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.split in ['train', 'both']:
        print(f"\n=== Collecting Training Data (seed={args.train_seed}) ===")
        num_train = int(args.num_episodes * args.train_ratio)
        train_trajectories = collect_trajectories(
            env_id=args.env_id,
            num_episodes=num_train,
            max_steps=args.max_steps,
            seed=args.train_seed
        )

        train_path = os.path.join(args.output_dir, "train_trajectories.json")
        save_trajectories(train_trajectories, train_path)
        print(f"Training set: {len(train_trajectories)} episodes")

    if args.split in ['test', 'both']:
        print(f"\n=== Collecting Test Data (seed={args.test_seed}) ===")
        num_test = args.num_episodes - int(args.num_episodes * args.train_ratio)
        test_trajectories = collect_trajectories(
            env_id=args.env_id,
            num_episodes=num_test,
            max_steps=args.max_steps,
            seed=args.test_seed
        )

        test_path = os.path.join(args.output_dir, "test_trajectories.json")
        save_trajectories(test_trajectories, test_path)
        print(f"Test set: {len(test_trajectories)} episodes")

    metadata = {
        "env_id": args.env_id,
        "discrete_actions": DISCRETE_ACTIONS,
        "action_threshold": ACTION_THRESHOLD,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
        "train_ratio": args.train_ratio,
        "max_steps": args.max_steps,
        "train_path": "train_trajectories.json" if args.split in ['train', 'both'] else None,
        "test_path": "test_trajectories.json" if args.split in ['test', 'both'] else None,
    }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Dataset saved to: {args.output_dir}")
    print(f"Metadata: {metadata_path}")

    if args.save_images:
        save_as_images(train_trajectories, args.output_dir) if 'train_trajectories' in dir() else save_as_images(test_trajectories, args.output_dir)
    else:
        if 'train_trajectories' in dir():
            save_trajectories(train_trajectories, os.path.join(args.output_dir, "train_trajectories.json"))

    print("Done!")


if __name__ == "__main__":
    main()
