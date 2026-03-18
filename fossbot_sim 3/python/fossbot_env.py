"""
fossbot_env.py
==============
FOSSBot-Gym: OpenAI Gymnasium environment wrapping the FOSSBot simulator.
Chapter 11 — Standardized Algorithmic Portability & Research Integration

Compatible with Stable-Baselines3, RLlib, and any Gym-based algorithm.

Install deps:
    pip install gymnasium stable-baselines3 python-socketio[client] pyyaml numpy
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from fossbot_client import FossBotClient
from scenario_parser import ScenarioConfig, ScenarioParser


class FossBotEnv(gym.Env):
    """
    MDP interface for the FOSSBot simulator.

    Observation space:
        Flat vector of:
          - sonar front/left/right (3)
          - IR left/center/right (3)
          - odometry pose x, z, yaw_deg (3)
          - IMU linear vel x/z + angular vel y (3)
          - terrain friction, cost (2)
          - goal relative dx, dz (2)
        Total: 16 floats, all normalised to [-1, 1] or [0, 1].

    Action space:
        Continuous Box:
          - throttle ∈ [-1, 1]
          - steer    ∈ [-1, 1]

    Reward:
        +dense progress toward goal
        -step cost (efficiency)
        -collision penalty
        +1000 on goal reached
        -500 on timeout
    """

    metadata = {"render_modes": ["human", "none"]}

    # ─────────────────────────────────────────
    #  CONFIGURATION
    # ─────────────────────────────────────────

    MAX_SONAR: float = 4.0      # metres
    MAX_ODOM:  float = 50.0     # metres (normalisation)
    GOAL_TOLERANCE: float = 0.4  # metres
    MAX_STEPS: int = 1000

    def __init__(
        self,
        simulator_url: str = "http://localhost:5000",
        scenario_path: Optional[str] = None,
        scenario: Optional[ScenarioConfig] = None,
        steps_per_action: int = 4,
        render_mode: str = "none",
        max_steps: int = MAX_STEPS,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._steps_per_action = steps_per_action

        # Load scenario
        if scenario is not None:
            self._scenario = scenario
        elif scenario_path is not None:
            self._scenario = ScenarioParser.load(scenario_path)
        else:
            self._scenario = ScenarioConfig()   # default flat scenario

        self._goal = np.array([
            self._scenario.goal_coordinates.x,
            self._scenario.goal_coordinates.z
        ], dtype=np.float32)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Client (connected lazily in reset)
        self._client: Optional[FossBotClient] = None
        self._sim_url = simulator_url
        self._step_count: int = 0
        self._prev_dist_to_goal: float = float("inf")
        self._last_state: Dict = {}

    # ─────────────────────────────────────────
    #  GYM API
    # ─────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._scenario.seed = seed

        # Connect (or reconnect) to simulator
        if self._client is None or not self._client._connected:
            self._client = FossBotClient(self._sim_url)
            self._client.connect()
            self._client.load_scenario(self._scenario)
            self._client.enable_lab_mode(self._steps_per_action)
        else:
            # Soft reset
            state = self._client.reset_episode({
                "x": self._scenario.spawn_pose.x,
                "z": self._scenario.spawn_pose.z,
                "yaw_deg": self._scenario.spawn_pose.yaw_deg
            })
            self._last_state = state

        self._step_count = 0
        self._prev_dist_to_goal = float("inf")

        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._client is not None, "Call reset() before step()."

        throttle = float(action[0])
        steer = float(action[1])

        # Send action, receive next state (Chapter 5.1.2)
        state = self._client.step({
            "throttle": throttle,
            "steer":    steer,
            "brake":    0.0
        })
        self._last_state = state
        self._step_count += 1

        obs = self._get_observation()
        reward, terminated, truncated = self._compute_reward_and_done(state)

        # Push diagnostic overlays (Chapter 10)
        self._client.set_state_label(self._get_state_label(state, terminated))

        info = {
            "step":       self._step_count,
            "dist_goal":  self._dist_to_goal(state),
            "terrain":    state.get("terrain", {}),
            "state_label": self._get_state_label(state, terminated)
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Rendering is handled inside Godot. No-op here."""
        pass

    def close(self) -> None:
        if self._client:
            self._client.disconnect()
            self._client = None

    # ─────────────────────────────────────────
    #  OBSERVATION BUILDER
    # ─────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        s = self._last_state
        sensors = s.get("sensors", {})

        def sonar(key: str) -> float:
            return sensors.get(key, {}).get("distance_m", self.MAX_SONAR)

        def ir(key: str) -> float:
            return 1.0 if sensors.get(key, {}).get("detected", False) else 0.0

        odom = sensors.get("odometry", {}).get("pose", {})
        imu = sensors.get("imu", {})
        lin_vel = imu.get("linear_velocity", {})
        ang_vel = imu.get("angular_velocity", {})
        terrain = s.get("terrain", {})
        pose = s.get("pose", {})

        # Goal-relative bearing
        robot_x = pose.get("x", 0.0)
        robot_z = pose.get("z", 0.0)
        dx = (self._goal[0] - robot_x) / self.MAX_ODOM
        dz = (self._goal[1] - robot_z) / self.MAX_ODOM

        obs = np.array([
            sonar("sonar_front")  / self.MAX_SONAR * 2 - 1,
            sonar("sonar_left")   / self.MAX_SONAR * 2 - 1,
            sonar("sonar_right")  / self.MAX_SONAR * 2 - 1,
            ir("ir_left"),
            ir("ir_center"),
            ir("ir_right"),
            odom.get("x", 0.0)        / self.MAX_ODOM,
            odom.get("y", 0.0)        / self.MAX_ODOM,
            odom.get("theta_deg", 0.0) / 180.0,
            lin_vel.get("x", 0.0),
            lin_vel.get("z", 0.0),
            ang_vel.get("y", 0.0),
            terrain.get("friction", 1.0) - 1.0,     # centred around 0
            (terrain.get("cost", 1.0) - 1.5) / 1.5, # centred ~0
            np.clip(dx, -1.0, 1.0),
            np.clip(dz, -1.0, 1.0),
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ─────────────────────────────────────────
    #  REWARD  (Chapter 11 MDP interface)
    # ─────────────────────────────────────────

    def _compute_reward_and_done(
        self, state: Dict
    ) -> Tuple[float, bool, bool]:
        dist = self._dist_to_goal(state)
        terrain = state.get("terrain", {})
        terrain_cost = terrain.get("cost", 1.0)

        # Progress reward (terrain-aware, Chapter 9.1.3 Traversal Costmap)
        progress = self._prev_dist_to_goal - dist
        reward = progress * 5.0 / terrain_cost   # penalise costly terrain

        # Step cost
        reward -= 0.02

        # Proximity collision penalty (front sonar)
        sonar_f = state.get("sensors", {}).get("sonar_front", {}).get("distance_m", 4.0)
        if sonar_f < 0.15:
            reward -= 2.0

        self._prev_dist_to_goal = dist

        terminated = False
        truncated = False

        if dist < self.GOAL_TOLERANCE:
            reward += 1000.0
            terminated = True

        if self._step_count >= self.max_steps:
            reward -= 500.0
            truncated = True

        return reward, terminated, truncated

    def _dist_to_goal(self, state: Dict) -> float:
        pose = state.get("pose", {})
        rx = pose.get("x", 0.0)
        rz = pose.get("z", 0.0)
        return math.hypot(self._goal[0] - rx, self._goal[1] - rz)

    def _get_state_label(self, state: Dict, terminated: bool) -> str:
        if terminated:
            return "Goal Reached"
        sonar_f = state.get("sensors", {}).get("sonar_front", {}).get("distance_m", 4.0)
        if sonar_f < 0.30:
            return "Avoiding Obstacle"
        dist = self._dist_to_goal(state)
        if dist < 1.5:
            return "Approaching Goal"
        return "Exploring"
