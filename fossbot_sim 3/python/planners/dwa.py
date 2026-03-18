"""
planners/dwa.py
===============
Dynamic Window Approach (DWA) Local Planner — Chapter 10.1.2
Generates short-term motion intent for obstacle avoidance,
visualised as the "local intent" overlay in Godot.

Reference:
    Fox, Burgard & Thrun (1997) — "The Dynamic Window Approach to Collision Avoidance"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
#  ROBOT STATE
# ─────────────────────────────────────────────

@dataclass
class RobotState:
    x: float        # world X (metres)
    z: float        # world Z (metres)
    yaw: float      # heading (radians, CCW from +Z)
    vx: float = 0.0 # linear speed (m/s)
    wz: float = 0.0 # angular speed (rad/s)


@dataclass
class DWAConfig:
    # Kinematic limits
    max_speed:      float = 1.5     # m/s
    min_speed:      float = -0.3    # m/s (allow small reverse)
    max_yaw_rate:   float = 2.0     # rad/s
    max_accel:      float = 2.0     # m/s²
    max_delta_yaw:  float = 3.0     # rad/s²

    # Sampling resolution
    speed_resolution:    float = 0.05
    yaw_rate_resolution: float = 0.1

    # Simulation
    dt:             float = 0.1     # seconds per prediction step
    predict_time:   float = 2.0     # seconds to look ahead

    # Cost weights
    w_heading:  float = 5.0     # alignment with goal
    w_clearance: float = 1.5    # distance from obstacles
    w_speed:    float = 1.0     # prefer higher speeds

    # Obstacle model
    obstacle_radius: float = 0.3   # metres (robot radius + margin)


# ─────────────────────────────────────────────
#  PLANNER
# ─────────────────────────────────────────────

class DWAPlanner:
    """
    DWA local planner for reactive obstacle avoidance.

    Usage:
        planner = DWAPlanner(config)
        v, w, traj = planner.compute(state, goal=(5.0, 5.0), obstacles=[(1.5, 2.0)])
        # v  = linear speed command (m/s)
        # w  = angular speed command (rad/s)
        # traj = predicted trajectory [[x,y,z], ...]  for Godot overlay
    """

    def __init__(self, config: Optional[DWAConfig] = None):
        self.cfg = config or DWAConfig()

    def compute(
        self,
        state: RobotState,
        goal: Tuple[float, float],      # (x, z) world coords
        obstacles: List[Tuple[float, float]]  # [(x, z), ...] obstacle positions
    ) -> Tuple[float, float, List[List[float]]]:
        """
        Returns (linear_v, angular_w, trajectory_for_visualisation).
        Trajectory is a list of [x, y, z] world points.
        """
        dw = self._dynamic_window(state)
        best_cost  = float("inf")
        best_v, best_w = 0.0, 0.0
        best_traj: List[List[float]] = []

        v = dw[0]
        while v <= dw[1] + 1e-6:
            w = dw[2]
            while w <= dw[3] + 1e-6:
                traj = self._simulate(state, v, w)
                cost = self._cost(traj, state, goal, obstacles, v)
                if cost < best_cost:
                    best_cost = cost
                    best_v, best_w = v, w
                    best_traj = traj
                w += self.cfg.yaw_rate_resolution
            v += self.cfg.speed_resolution

        # Convert to [[x, y, z]] (y=0.1 for Godot overlay)
        world_traj = [[pt[0], 0.1, pt[1]] for pt in best_traj]
        return best_v, best_w, world_traj

    def speed_to_action(self, v: float, w: float) -> dict:
        """
        Convert DWA (v, w) output → FossBotClient step action dict.
        Normalises to [-1, 1] throttle/steer.
        """
        throttle = v / max(self.cfg.max_speed, 1e-6)
        steer    = w / max(self.cfg.max_yaw_rate, 1e-6)
        return {
            "throttle": max(-1.0, min(1.0, throttle)),
            "steer":    max(-1.0, min(1.0, steer)),
            "brake":    0.0
        }

    # ── Dynamic Window ────────────────────────

    def _dynamic_window(self, state: RobotState) -> Tuple[float, float, float, float]:
        """Compute feasible velocity window [v_min, v_max, w_min, w_max]."""
        cfg = self.cfg
        dt  = cfg.dt
        v_min = max(cfg.min_speed,   state.vx - cfg.max_accel   * dt)
        v_max = min(cfg.max_speed,   state.vx + cfg.max_accel   * dt)
        w_min = max(-cfg.max_yaw_rate, state.wz - cfg.max_delta_yaw * dt)
        w_max = min( cfg.max_yaw_rate, state.wz + cfg.max_delta_yaw * dt)
        return v_min, v_max, w_min, w_max

    # ── Trajectory Simulation ─────────────────

    def _simulate(self, state: RobotState, v: float, w: float) -> List[Tuple[float, float]]:
        """Predict trajectory for (v, w) over predict_time seconds."""
        cfg = self.cfg
        traj = [(state.x, state.z)]
        x, z, yaw = state.x, state.z, state.yaw
        t = 0.0
        while t <= cfg.predict_time:
            yaw += w * cfg.dt
            x   += v * math.cos(yaw) * cfg.dt
            z   += v * math.sin(yaw) * cfg.dt
            traj.append((x, z))
            t += cfg.dt
        return traj

    # ── Cost Function ─────────────────────────

    def _cost(
        self,
        traj: List[Tuple[float, float]],
        state: RobotState,
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float]],
        v: float
    ) -> float:
        cfg = self.cfg

        # 1. Heading cost — angle between final heading and goal direction
        final = traj[-1]
        goal_angle = math.atan2(goal[1] - final[1], goal[0] - final[0])
        # infer final yaw from last two points
        if len(traj) >= 2:
            p1, p2 = traj[-2], traj[-1]
            final_yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        else:
            final_yaw = state.yaw
        heading_cost = abs(self._angle_diff(goal_angle, final_yaw))

        # 2. Clearance cost — minimum distance to obstacles along trajectory
        min_dist = float("inf")
        for tx, tz in traj:
            for ox, oz in obstacles:
                d = math.hypot(tx - ox, tz - oz)
                min_dist = min(min_dist, d)

        if min_dist < cfg.obstacle_radius:
            return float("inf")     # infeasible

        clearance_cost = 1.0 / max(min_dist - cfg.obstacle_radius, 1e-3)

        # 3. Speed cost — reward higher speeds
        speed_cost = cfg.max_speed - v

        return (cfg.w_heading   * heading_cost   +
                cfg.w_clearance * clearance_cost  +
                cfg.w_speed     * speed_cost)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = a - b
        while d >  math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d


# ─────────────────────────────────────────────
#  STATE EXTRACTOR (from FossBotClient state dict)
# ─────────────────────────────────────────────

def state_dict_to_robot_state(state: dict) -> RobotState:
    """
    Convert a FossBotClient state packet to a RobotState for DWA.
    """
    pose    = state.get("pose", {})
    sensors = state.get("sensors", {})
    imu     = sensors.get("imu", {})
    lin_vel = imu.get("linear_velocity", {})
    ang_vel = imu.get("angular_velocity", {})

    return RobotState(
        x   = pose.get("x", 0.0),
        z   = pose.get("z", 0.0),
        yaw = math.radians(pose.get("yaw_deg", 0.0)),
        vx  = lin_vel.get("x", 0.0),
        wz  = ang_vel.get("y", 0.0)
    )


def obstacles_from_state(state: dict, obstacle_range: float = 1.5) -> List[Tuple[float, float]]:
    """
    Extract approximate obstacle positions from sonar readings.
    Returns list of (x, z) obstacle positions in world frame.
    """
    sensors = state.get("sensors", {})
    pose    = state.get("pose", {})
    rx = pose.get("x", 0.0)
    rz = pose.get("z", 0.0)
    yaw = math.radians(pose.get("yaw_deg", 0.0))
    obstacles = []

    sonar_map = {
        "sonar_front": 0.0,
        "sonar_left":  math.pi / 2,
        "sonar_right": -math.pi / 2
    }

    for key, angle_offset in sonar_map.items():
        reading = sensors.get(key, {})
        dist = reading.get("distance_m", 4.0)
        if dist < obstacle_range and reading.get("hit", False):
            abs_angle = yaw + angle_offset
            ox = rx + dist * math.cos(abs_angle)
            oz = rz + dist * math.sin(abs_angle)
            obstacles.append((ox, oz))

    return obstacles
