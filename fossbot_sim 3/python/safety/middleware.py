"""
safety/middleware.py
====================
Chapter 15 — FOSSBot Guardian: Python-Side Safety Filter

Wraps FossBotClient.step() with:
  1. Hard action clipping
  2. Acceleration limiting (anti-jerk)
  3. Sensor-based action masking
  4. NaN / Inf guard
  5. Network watchdog heartbeat
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

@dataclass
class SafetyConfig:
    # Hard limits
    max_throttle: float = 1.0
    min_throttle: float = -1.0
    max_steer: float = 1.0

    # Acceleration limiting
    max_throttle_delta: float = 0.3   # per step
    max_steer_delta: float = 0.4      # per step
    enable_accel_limit: bool = True

    # Sensor-based masking thresholds (metres)
    front_collision_threshold: float = 0.15   # block forward if sonar < this
    cliff_threshold: float = 0.5              # block all motion if all IR off

    # Watchdog
    heartbeat_interval: float = 0.5    # seconds between heartbeats
    watchdog_timeout: float = 1.5      # seconds before safe stop

    # Logging
    log_interventions: bool = True


# ─────────────────────────────────────────────
#  SAFETY MIDDLEWARE
# ─────────────────────────────────────────────

class SafetyMiddleware:
    """
    Wraps a FossBotClient and intercepts every action before it reaches
    the simulator. Independent of RL logic — always active.

    Usage:
        client = FossBotClient(url)
        client.connect()
        safety = SafetyMiddleware(client, SafetyConfig())
        safety.start_watchdog()

        # Use safety.safe_step() instead of client.step()
        state = safety.safe_step({"throttle": 0.8, "steer": 0.0, "brake": 0.0})
    """

    def __init__(self, client, config: Optional[SafetyConfig] = None):
        self.client = client
        self.cfg = config or SafetyConfig()

        self._prev_action: Dict[str, float] = {"throttle": 0.0, "steer": 0.0, "brake": 0.0}
        self._last_state: Dict[str, Any] = {}
        self._intervention_log: list = []

        # Watchdog state
        self._watchdog_thread: Optional[threading.Thread] = None
        self._last_heartbeat: float = time.time()
        self._watchdog_active: bool = False
        self._safe_stopped: bool = False

        # Stats
        self.total_steps: int = 0
        self.total_interventions: int = 0

    # ─────────────────────────────────────────
    #  MAIN API
    # ─────────────────────────────────────────

    def safe_step(self, action: Dict[str, float]) -> Dict[str, Any]:
        """
        Filter action through all safety layers, then execute.
        Returns the resulting state packet.
        """
        self._last_heartbeat = time.time()
        self.total_steps += 1

        original = dict(action)

        # Pipeline
        action = self._guard_nan_inf(action)
        action = self._clip_action(action)
        action = self._limit_acceleration(action)
        action = self._apply_sensor_masks(action, self._last_state)

        # Log if changed
        if self.cfg.log_interventions and action != original:
            self.total_interventions += 1
            self._intervention_log.append({
                "step": self.total_steps,
                "original": original,
                "filtered": dict(action),
                "reason": self._last_intervention_reason
            })

        state = self.client.step(action)
        self._last_state = state
        self._prev_action = dict(action)
        return state

    def reset(self) -> None:
        """Call at episode start to clear accumulated state."""
        self._prev_action = {"throttle": 0.0, "steer": 0.0, "brake": 0.0}
        self._last_state = {}
        self._safe_stopped = False

    def get_stats(self) -> Dict:
        return {
            "total_steps": self.total_steps,
            "total_interventions": self.total_interventions,
            "intervention_rate": self.total_interventions / max(self.total_steps, 1),
            "last_10_interventions": self._intervention_log[-10:]
        }

    # ─────────────────────────────────────────
    #  SAFETY LAYERS
    # ─────────────────────────────────────────

    def _guard_nan_inf(self, action: Dict) -> Dict:
        """Replace NaN/Inf with zeros. Catches broken policy outputs."""
        clean = {}
        triggered = False
        for k, v in action.items():
            if not math.isfinite(float(v)):
                clean[k] = 0.0
                triggered = True
            else:
                clean[k] = v
        if triggered:
            self._last_intervention_reason = "nan_inf_guard"
        return clean

    def _clip_action(self, action: Dict) -> Dict:
        """Hard clip all action dimensions to configured limits."""
        cfg = self.cfg
        clipped = {
            "throttle": float(np.clip(action.get("throttle", 0.0), cfg.min_throttle, cfg.max_throttle)),
            "steer":    float(np.clip(action.get("steer",    0.0), -cfg.max_steer,   cfg.max_steer)),
            "brake":    float(np.clip(action.get("brake",    0.0), 0.0,              1.0))
        }
        if clipped != action:
            self._last_intervention_reason = "hard_clip"
        return clipped

    def _limit_acceleration(self, action: Dict) -> Dict:
        """Smooth out step changes to prevent jerk."""
        if not self.cfg.enable_accel_limit:
            return action

        cfg = self.cfg
        prev = self._prev_action
        limited = dict(action)

        for key, max_delta in [("throttle", cfg.max_throttle_delta),
                                ("steer",    cfg.max_steer_delta)]:
            delta = action[key] - prev.get(key, 0.0)
            if abs(delta) > max_delta:
                limited[key] = prev.get(key, 0.0) + math.copysign(max_delta, delta)
                self._last_intervention_reason = "accel_limit"

        return limited

    def _apply_sensor_masks(self, action: Dict, state: Dict) -> Dict:
        """Block dangerous actions based on current sensor readings."""
        if not state:
            return action

        sensors = state.get("sensors", {})
        masked = dict(action)

        # ── Collision mask: block forward motion if too close
        sonar_f = sensors.get("sonar_front", {}).get("distance_m", 4.0)
        if sonar_f < self.cfg.front_collision_threshold and masked["throttle"] > 0:
            masked["throttle"] = 0.0
            masked["brake"] = 0.5
            self._last_intervention_reason = f"collision_mask(sonar_f={sonar_f:.2f}m)"

        # ── Cliff mask: stop all motion if all IR sensors off-surface
        ir_l = sensors.get("ir_left",   {}).get("detected", True)
        ir_c = sensors.get("ir_center", {}).get("detected", True)
        ir_r = sensors.get("ir_right",  {}).get("detected", True)
        if not ir_l and not ir_c and not ir_r:
            masked = {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
            self._last_intervention_reason = "cliff_mask(all_IR_off)"

        return masked

    # ─────────────────────────────────────────
    #  WATCHDOG
    # ─────────────────────────────────────────

    def start_watchdog(self) -> None:
        """Start background watchdog thread."""
        self._watchdog_active = True
        self._last_heartbeat = time.time()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True
        )
        self._watchdog_thread.start()
        print("[SafetyMiddleware] Watchdog started.")

    def stop_watchdog(self) -> None:
        self._watchdog_active = False

    def _watchdog_loop(self) -> None:
        """
        Chapter 15.1.3 — Network Watchdog Timer.
        If no heartbeat for watchdog_timeout seconds, trigger safe stop.
        """
        while self._watchdog_active:
            time.sleep(self.cfg.heartbeat_interval)
            elapsed = time.time() - self._last_heartbeat
            if elapsed > self.cfg.watchdog_timeout and not self._safe_stopped:
                print(f"[SafetyMiddleware] WATCHDOG TRIGGERED — no heartbeat for {elapsed:.1f}s")
                self._trigger_safe_stop()

    def _trigger_safe_stop(self) -> None:
        """Send emergency stop to simulator."""
        self._safe_stopped = True
        try:
            self.client.stop()
            print("[SafetyMiddleware] Safe stop sent.")
        except Exception as e:
            print(f"[SafetyMiddleware] Could not send safe stop: {e}")
