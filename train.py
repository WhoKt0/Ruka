"""Training entrypoint for the pixel-based robot arm environment.

The script supports two vision backbones:
- "custom": Stable-Baselines3 NatureCNN (fast, lightweight).
- "mobilenet": transfer learning with frozen MobileNetV3-Small.
"""
from __future__ import annotations

import argparse
from typing import Callable, Dict

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from envs.robot_arm_env import CameraConfig, RewardWeights, RobotArmEnv
from models.vision import MobileNetFeatureExtractor


def build_env(args: argparse.Namespace) -> gym.Env:
    camera_cfg = CameraConfig(
        width=args.image_size,
        height=args.image_size,
        eye_in_hand=args.eye_in_hand,
    )
    reward_cfg = RewardWeights(
        distance=args.distance_weight,
        contact_bonus=args.contact_bonus,
        time_penalty=args.time_penalty,
    )

    env = RobotArmEnv(
        use_gui=args.gui,
        camera=camera_cfg,
        reward_weights=reward_cfg,
        frame_stack=args.frame_stack,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        image_grayscale=args.grayscale,
    )
    return Monitor(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pixel-based robot arm RL training")
    parser.add_argument("--vision", choices=["custom", "mobilenet"], default="custom")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--eye-in-hand", action="store_true")
    parser.add_argument("--grayscale", action="store_true", default=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--contact-bonus", type=float, default=5.0)
    parser.add_argument("--time-penalty", type=float, default=0.01)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_policy_kwargs(args: argparse.Namespace, observation_space) -> Dict:
    if args.vision == "custom":
        return {}

    if args.vision == "mobilenet":
        return {
            "features_extractor_class": MobileNetFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
        }

    raise ValueError(f"Unknown vision backbone: {args.vision}")


def main() -> None:
    args = parse_args()

    env_fn: Callable[[], gym.Env] = lambda: build_env(args)
    vec_env = make_vec_env(env_fn, n_envs=args.num_envs, seed=args.seed)

    policy_kwargs = make_policy_kwargs(args, vec_env.observation_space)
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        verbose=1,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.total_timesteps)
    model.save("ppo_robot_arm")
    vec_env.close()


if __name__ == "__main__":
    main()
