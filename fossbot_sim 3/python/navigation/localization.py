"""
navigation/localization.py
===========================
Chapter 14 — Hierarchical Navigation Stack
Layer 1: Extended Kalman Filter (EKF) Localizer
Layer 2: Probabilistic Occupancy Grid (SLAM-lite)

Usage:
    ekf   = EKFLocalizer()
    grid  = OccupancyGrid(width=40, height=40, resolution=0.5)

    # Each step:
    ekf.predict(delta_x, delta_z, delta_theta)
    ekf.update_from_sonar(sonar_dist, sonar_angle_world)
    pose = ekf.get_pose()           # (x, z, theta)
    cov  = ekf.get_covariance()     # 3x3 uncertainty matrix

    grid.update(pose, sonar_readings)
    costmap = grid.to_costmap()     # drop-in for SimInfo costmap dict
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────
#  EKF LOCALIZER
# ─────────────────────────────────────────────

class EKFLocalizer:
    """
    Chapter 14.1.1 — Stochastic Pose Fusion via Extended Kalman Filter.

    State vector: [x, z, theta]  (position + heading in 2D)

    Fuses:
      - Odometry increments (predict step)
      - IMU angular velocity (predict step)
      - Ultrasonic distance observations (update step)
    """

    def __init__(
        self,
        init_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        process_noise: Tuple[float, float, float] = (0.01, 0.01, 0.005),
        obs_noise_sonar: float = 0.05
    ):
        # State: [x, z, theta]
        self._mu = np.array([init_pose[0], init_pose[1], init_pose[2]], dtype=float)

        # Covariance
        self._sigma = np.diag([0.1, 0.1, 0.05])

        # Process noise Q
        self._Q = np.diag([process_noise[0]**2, process_noise[1]**2, process_noise[2]**2])

        # Observation noise R (for sonar)
        self._R_sonar = np.array([[obs_noise_sonar**2]])

        # Landmark map: list of (lx, lz) known obstacle positions
        # If empty, sonar updates are skipped
        self._landmarks: List[Tuple[float, float]] = []

    # ── Public API ────────────────────────────

    def set_landmarks(self, landmarks: List[Tuple[float, float]]) -> None:
        """Register known landmark positions for sonar update step."""
        self._landmarks = landmarks

    def predict(
        self,
        delta_x: float,
        delta_z: float,
        delta_theta: float,
        imu_omega: Optional[float] = None
    ) -> None:
        """
        Motion model predict step.
        delta_x, delta_z: odometry increments in robot frame
        delta_theta: heading change from odometry
        imu_omega: angular velocity from IMU (used to refine delta_theta if provided)
        """
        # Fuse odometry and IMU heading if available
        if imu_omega is not None:
            # Simple complementary fusion: trust IMU angular rate more
            delta_theta = 0.4 * delta_theta + 0.6 * imu_omega

        theta = self._mu[2]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Nonlinear motion model: rotate odometry increment into world frame
        dx_world = cos_t * delta_x - sin_t * delta_z
        dz_world = sin_t * delta_x + cos_t * delta_z

        # State prediction
        self._mu = self._mu + np.array([dx_world, dz_world, delta_theta])
        self._mu[2] = self._wrap_angle(self._mu[2])

        # Jacobian of motion model wrt state (linearisation)
        G = np.array([
            [1, 0, -sin_t * delta_x - cos_t * delta_z],
            [0, 1,  cos_t * delta_x - sin_t * delta_z],
            [0, 0,  1]
        ])

        # Covariance prediction
        self._sigma = G @ self._sigma @ G.T + self._Q

    def update_from_sonar(
        self,
        measured_dist: float,
        sensor_angle_offset: float = 0.0
    ) -> bool:
        """
        EKF update step using a sonar distance observation.
        Finds the closest expected landmark and uses it for the update.
        Returns True if update was applied.
        """
        if not self._landmarks:
            return False

        robot_angle = self._mu[2] + sensor_angle_offset
        rx, rz = self._mu[0], self._mu[1]

        # Find closest landmark in the sensor's direction
        best_lm = None
        best_diff = float("inf")
        for lx, lz in self._landmarks:
            expected_dist = math.hypot(lx - rx, lz - rz)
            if abs(expected_dist - measured_dist) < best_diff:
                best_diff = abs(expected_dist - measured_dist)
                best_lm = (lx, lz)

        if best_lm is None or best_diff > 1.5:
            return False   # no plausible landmark match

        lx, lz = best_lm
        expected_dist = math.hypot(lx - rx, lz - rz)
        innovation = measured_dist - expected_dist

        # Jacobian of observation model
        H = np.array([[
            -(lx - rx) / max(expected_dist, 1e-6),
            -(lz - rz) / max(expected_dist, 1e-6),
            0.0
        ]])

        # Innovation covariance
        S = H @ self._sigma @ H.T + self._R_sonar

        # Kalman gain
        K = self._sigma @ H.T @ np.linalg.inv(S)

        # State update
        self._mu = self._mu + (K @ np.array([innovation])).flatten()
        self._mu[2] = self._wrap_angle(self._mu[2])

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(3) - K @ H
        self._sigma = I_KH @ self._sigma @ I_KH.T + K @ self._R_sonar @ K.T

        return True

    def get_pose(self) -> Tuple[float, float, float]:
        """Returns (x, z, theta) best estimate."""
        return float(self._mu[0]), float(self._mu[1]), float(self._mu[2])

    def get_covariance(self) -> np.ndarray:
        """Returns 3x3 covariance matrix."""
        return self._sigma.copy()

    def get_uncertainty_radius(self) -> float:
        """Approximate position uncertainty as radius (metres)."""
        return float(math.sqrt(self._sigma[0, 0] + self._sigma[1, 1]))

    def reset(self, pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self._mu = np.array([pose[0], pose[1], pose[2]], dtype=float)
        self._sigma = np.diag([0.1, 0.1, 0.05])

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        while theta >  math.pi: theta -= 2 * math.pi
        while theta < -math.pi: theta += 2 * math.pi
        return theta

    def pose_to_obs_features(self) -> List[float]:
        """
        Returns normalised [x/50, z/50, theta/pi, uncertainty]
        for inclusion in the RL observation vector.
        """
        x, z, theta = self.get_pose()
        unc = min(self.get_uncertainty_radius() / 2.0, 1.0)
        return [x / 50.0, z / 50.0, theta / math.pi, unc]


# ─────────────────────────────────────────────
#  OCCUPANCY GRID
# ─────────────────────────────────────────────

class OccupancyGrid:
    """
    Chapter 14.1.2 — Dynamic Probabilistic Occupancy Grid (SLAM-lite).

    Uses log-odds representation for numerically stable cell updates.
    Compatible with AStarPlanner as a drop-in costmap.
    """

    # Log-odds update constants
    L_OCC   =  0.85    # log-odds increment when cell is "occupied"
    L_FREE  = -0.40    # log-odds decrement when cell is "free"
    L_MAX   =  5.0     # saturation cap
    L_MIN   = -5.0

    # Obstacle cost threshold for A*
    OCC_COST_THRESHOLD = 8.0

    def __init__(
        self,
        width_m: float = 20.0,
        height_m: float = 20.0,
        resolution: float = 0.5,   # metres per cell
        origin: Tuple[float, float] = (-10.0, -10.0)
    ):
        self.resolution = resolution
        self.origin = origin
        self.width_cells  = int(width_m / resolution)
        self.height_cells = int(height_m / resolution)
        # Log-odds grid initialised to 0 (unknown)
        self._log_odds = np.zeros((self.width_cells, self.height_cells), dtype=float)

    # ── Public API ────────────────────────────

    def update(
        self,
        robot_pose: Tuple[float, float, float],
        sonar_readings: Dict[str, Dict]
    ) -> None:
        """
        Update grid from current sonar readings.
        sonar_readings: {"sonar_front": {"distance_m": 1.2, "hit": True}, ...}
        """
        rx, rz, rtheta = robot_pose

        sonar_angles = {
            "sonar_front":  0.0,
            "sonar_left":   math.pi / 2,
            "sonar_right": -math.pi / 2
        }

        for key, angle_offset in sonar_angles.items():
            reading = sonar_readings.get(key, {})
            dist = reading.get("distance_m", 4.0)
            hit  = reading.get("hit", False)

            sensor_angle = rtheta + angle_offset
            self._ray_update(rx, rz, sensor_angle, dist, hit)

    def get_occupancy(self, world_x: float, world_z: float) -> float:
        """Returns probability [0,1] that the cell is occupied."""
        cx, cz = self._world_to_cell(world_x, world_z)
        if not self._in_bounds(cx, cz):
            return 0.5   # unknown
        lo = self._log_odds[cx, cz]
        return 1.0 / (1.0 + math.exp(-lo))

    def to_costmap(self) -> Dict:
        """
        Convert to costmap format compatible with AStarPlanner.
        Returns dict: (cell_x, cell_z) → cost
        High-occupancy cells get very high cost (treated as walls by A*).
        """
        costmap = {}
        for cx in range(self.width_cells):
            for cz in range(self.height_cells):
                lo = self._log_odds[cx, cz]
                prob = 1.0 / (1.0 + math.exp(-lo))
                # Unknown (prob~0.5) → cost 1.0, occupied → very high
                if prob > 0.7:
                    cost = self.OCC_COST_THRESHOLD + (prob - 0.7) * 20
                elif prob < 0.3:
                    cost = 0.8   # confirmed free = slightly cheaper
                else:
                    cost = 1.0   # unknown = neutral
                wx = int(cx * self.resolution + self.origin[0])
                wz = int(cz * self.resolution + self.origin[1])
                costmap[(wx, wz)] = cost
        return costmap

    def reset(self) -> None:
        self._log_odds[:] = 0.0

    # ── Internals ─────────────────────────────

    def _ray_update(
        self,
        rx: float, rz: float,
        angle: float,
        dist: float,
        hit: bool,
        max_range: float = 4.0
    ) -> None:
        """Bresenham ray trace: mark free cells along ray, occupied at end."""
        end_dist = dist if hit else max_range
        steps = int(end_dist / self.resolution)

        for i in range(steps):
            d = i * self.resolution
            wx = rx + d * math.cos(angle)
            wz = rz + d * math.sin(angle)
            cx, cz = self._world_to_cell(wx, wz)
            if self._in_bounds(cx, cz):
                self._log_odds[cx, cz] = max(
                    self.L_MIN,
                    self._log_odds[cx, cz] + self.L_FREE
                )

        if hit:
            ex = rx + dist * math.cos(angle)
            ez = rz + dist * math.sin(angle)
            cx, cz = self._world_to_cell(ex, ez)
            if self._in_bounds(cx, cz):
                self._log_odds[cx, cz] = min(
                    self.L_MAX,
                    self._log_odds[cx, cz] + self.L_OCC
                )

    def _world_to_cell(self, wx: float, wz: float) -> Tuple[int, int]:
        cx = int((wx - self.origin[0]) / self.resolution)
        cz = int((wz - self.origin[1]) / self.resolution)
        return cx, cz

    def _in_bounds(self, cx: int, cz: int) -> bool:
        return 0 <= cx < self.width_cells and 0 <= cz < self.height_cells
