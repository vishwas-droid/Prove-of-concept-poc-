"""
fossbot_gym_register.py
=======================
Registers FOSSBot-Gym environments in Gymnasium's global registry.
Chapter 11 — Standardized Algorithmic Portability

Call this once at startup (or import it before using gym.make):

    import fossbot_gym_register    # side-effect: registers envs
    import gymnasium as gym
    env = gym.make("FossBot-v0")
    env = gym.make("FossBot-Maze-v0", simulator_url="http://localhost:5000")
    env = gym.make("FossBot-Terrain-v0")
"""

from __future__ import annotations

from gymnasium.envs.registration import register

# ─────────────────────────────────────────────
#  ENVIRONMENT VARIANTS
# ─────────────────────────────────────────────

register(
    id="FossBot-v0",
    entry_point="fossbot_env:FossBotEnv",
    max_episode_steps=1000,
    kwargs={
        "simulator_url": "http://localhost:5000",
        "steps_per_action": 4,
        "render_mode": "none"
    }
)

register(
    id="FossBot-Maze-v0",
    entry_point="fossbot_env:FossBotEnv",
    max_episode_steps=2000,
    kwargs={
        "simulator_url": "http://localhost:5000",
        "scenario_path": "scenarios/example_maze.yaml",
        "steps_per_action": 4,
        "render_mode": "none"
    }
)

register(
    id="FossBot-Terrain-v0",
    entry_point="fossbot_env:FossBotEnv",
    max_episode_steps=1500,
    kwargs={
        "simulator_url": "http://localhost:5000",
        "scenario_path": "scenarios/terrain_challenge.yaml",
        "steps_per_action": 4,
        "render_mode": "none"
    }
)

register(
    id="FossBot-Random-v0",
    entry_point="fossbot_env:FossBotEnv",
    max_episode_steps=1000,
    kwargs={
        "simulator_url": "http://localhost:5000",
        "steps_per_action": 4,
        "render_mode": "none"
    }
)

# ─────────────────────────────────────────────
#  VERIFICATION HELPER
# ─────────────────────────────────────────────

def list_fossbot_envs() -> None:
    import gymnasium as gym
    all_envs = gym.envs.registry.keys()
    fossbot = [e for e in all_envs if "FossBot" in e]
    print("Registered FOSSBot environments:")
    for e in sorted(fossbot):
        print(f"  {e}")


if __name__ == "__main__":
    list_fossbot_envs()
