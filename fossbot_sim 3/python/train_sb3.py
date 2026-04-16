"""
train_sb3.py — updated with BC pre-training (Chapter 13)

New flags:
    --record-demos             Record demos using DWA scripted policy
    --pretrain-demos  demos/   Run BC pre-training before RL
    --pretrain-epochs 50       BC epochs
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from fossbot_env import FossBotEnv
from scenario_parser import ScenarioParser, write_example_scenario

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C}

def make_env(scenario_path, url, seed):
    def _init():
        env = FossBotEnv(simulator_url=url, scenario_path=scenario_path, steps_per_action=4)
        return Monitor(env)
    return _init

def record_demos(scenario_path, url, save_dir="demos/", num_episodes=20):
    """Chapter 13 - Record DWA scripted demos."""
    from fossbot_client import FossBotClient
    from imitation.recorder import DemonstrationRecorder
    from planners.dwa import DWAPlanner, DWAConfig, state_dict_to_robot_state, obstacles_from_state
    import math
    scenario = ScenarioParser.load(scenario_path)
    dwa = DWAPlanner(DWAConfig())
    recorder = DemonstrationRecorder(save_dir=save_dir)
    goal = (scenario.goal_coordinates.x, scenario.goal_coordinates.z)
    print(f"[RecordDemos] Recording {num_episodes} episodes")
    for ep in range(num_episodes):
        scenario.seed = ep
        with FossBotClient(url, timeout=15.0) as client:
            client.load_scenario(scenario)
            client.enable_lab_mode(steps_per_action=4)
            state = client.get_state()
            recorder.start(f"ep{ep:03d}_seed{ep}")
            success = False
            for _ in range(500):
                rs = state_dict_to_robot_state(state)
                obs_list = obstacles_from_state(state, obstacle_range=1.8)
                v, w, _ = dwa.compute(rs, goal, obs_list)
                action = dwa.speed_to_action(v, w)
                obs = np.zeros(16, dtype=np.float32)  # placeholder obs
                state = client.step(action)
                recorder.record(obs, action, state)
                pose = state.get("pose", {})
                dist = math.hypot(goal[0]-pose.get("x",0), goal[1]-pose.get("z",0))
                if dist < 0.4:
                    success = True; break
            recorder.stop(success=success)
            print(f"  Ep {ep+1}: {'SUCCESS' if success else 'timeout'}")

def train(algo_name="ppo", scenario_path="scenarios/example_maze.yaml",
          total_steps=500_000, url="http://localhost:5000", seed=42,
          log_dir="logs/", checkpoint_freq=10_000,
          pretrain_demos="", pretrain_epochs=50):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([make_env(scenario_path, url, seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    AlgoClass = ALGOS.get(algo_name.lower(), PPO)
    policy_kwargs = dict(net_arch=[256, 256])
    algo_kwargs = dict(policy="MlpPolicy", env=env, verbose=1, seed=seed,
                       tensorboard_log=log_dir, policy_kwargs=policy_kwargs)
    if algo_name in ("ppo", "a2c"):
        algo_kwargs.update({"n_steps": 2048, "batch_size": 64, "learning_rate": 3e-4})
    elif algo_name in ("sac", "td3"):
        algo_kwargs.update({"learning_rate": 3e-4, "buffer_size": 100_000})
    model = AlgoClass(**algo_kwargs)
    # Chapter 13: BC pre-training
    if pretrain_demos and Path(pretrain_demos).exists():
        print(f"\n[train_sb3] BC pre-training from {pretrain_demos}")
        from imitation.recorder import BehavioralCloningTrainer
        trainer = BehavioralCloningTrainer(demo_dir=pretrain_demos, obs_dim=16, action_dim=2)
        trainer.train(epochs=pretrain_epochs)
        trainer.load_into_sb3(model)
        print("[train_sb3] BC weights transferred. Starting RL.\n")
    eval_env = DummyVecEnv([make_env(scenario_path, url, seed+1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    callbacks = CallbackList([
        EvalCallback(eval_env, best_model_save_path=os.path.join(log_dir,"best_model"),
                     log_path=os.path.join(log_dir,"eval"), eval_freq=5_000,
                     n_eval_episodes=3, deterministic=True, verbose=1),
        CheckpointCallback(save_freq=checkpoint_freq,
                           save_path=os.path.join(log_dir,"checkpoints"),
                           name_prefix=f"fossbot_{algo_name}")
    ])
    print(f"\n[train_sb3] {algo_name.upper()} for {total_steps:,} steps")
    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)
    model_path = os.path.join(log_dir, f"fossbot_{algo_name}_final")
    model.save(model_path)
    env.save(model_path + "_vecnorm.pkl")
    print(f"[train_sb3] Saved -> {model_path}")

def evaluate(model_path, vecnorm_path, scenario_path, url, episodes=10):
    env = DummyVecEnv([make_env(scenario_path, url, 0)])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False; env.norm_reward = False
    for name, cls in ALGOS.items():
        if name in model_path.lower():
            model = cls.load(model_path, env=env); break
    else:
        model = PPO.load(model_path, env=env)
    rewards = []
    for ep in range(episodes):
        obs = env.reset(); done = False; total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action); total += r[0]
        rewards.append(total)
        print(f"  Episode {ep+1}: {total:.1f}")
    print(f"\nMean: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",            default="ppo",  choices=list(ALGOS.keys()))
    parser.add_argument("--scenario",        default="scenarios/example_maze.yaml")
    parser.add_argument("--steps",           default=500_000, type=int)
    parser.add_argument("--url",             default="http://localhost:5000")
    parser.add_argument("--seed",            default=42, type=int)
    parser.add_argument("--log-dir",         default="logs/")
    parser.add_argument("--eval",            action="store_true")
    parser.add_argument("--model",           default="")
    parser.add_argument("--vecnorm",         default="")
    parser.add_argument("--record-demos",    action="store_true")
    parser.add_argument("--demo-dir",        default="demos/")
    parser.add_argument("--demo-episodes",   default=20, type=int)
    parser.add_argument("--pretrain-demos",  default="")
    parser.add_argument("--pretrain-epochs", default=50, type=int)
    parser.add_argument("--write-example-scenario", action="store_true")
    args = parser.parse_args()
    if args.write_example_scenario:
        write_example_scenario(args.scenario); return
    if args.record_demos:
        record_demos(args.scenario, args.url, args.demo_dir, args.demo_episodes); return
    if args.eval:
        evaluate(args.model, args.vecnorm, args.scenario, args.url); return
    train(args.algo, args.scenario, args.steps, args.url, args.seed,
          args.log_dir, pretrain_demos=args.pretrain_demos,
          pretrain_epochs=args.pretrain_epochs)

if __name__ == "__main__":
    main()
