"""
fossbot_client.py
=================
Python SocketIO client for the FOSSBot simulator.
Provides a clean API used by both direct scripts and the Gym wrapper.
Chapters: 5 (step-sync), 6 (scenario), 9 (costmap), 10 (diagnostics)
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import socketio  # pip install python-socketio[client]

from scenario_parser import ScenarioConfig, ScenarioParser


class FossBotClient:
    """
    Thread-safe synchronous wrapper around SocketIO for the FOSSBot simulator.
    """

    def __init__(self, url: str = "http://localhost:5000", timeout: float = 10.0):
        self.url = url
        self.timeout = timeout

        self._sio = socketio.Client(reconnection=True, reconnection_attempts=5)
        self._state: Dict[str, Any] = {}
        self._state_event = threading.Event()
        self._scenario_event = threading.Event()
        self._reset_event = threading.Event()
        self._costmap: List[Dict] = []
        self._costmap_event = threading.Event()
        self._connected = False

        self._register_handlers()

    # ─────────────────────────────────────────
    #  CONNECTION
    # ─────────────────────────────────────────

    def connect(self) -> None:
        self._sio.connect(self.url)
        self._connected = True
        print(f"[FossBotClient] Connected to {self.url}")

    def disconnect(self) -> None:
        self._sio.disconnect()
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ─────────────────────────────────────────
    #  SOCKET EVENT HANDLERS
    # ─────────────────────────────────────────

    def _register_handlers(self) -> None:
        @self._sio.on("state")
        def _on_state(data):
            self._state = data
            self._state_event.set()

        @self._sio.on("scenario_loaded")
        def _on_scenario_loaded(_data):
            self._scenario_event.set()

        @self._sio.on("episode_reset")
        def _on_reset(data):
            self._state = data
            self._reset_event.set()
            self._state_event.set()

        @self._sio.on("costmap")
        def _on_costmap(data):
            self._costmap = data.get("cells", [])
            self._costmap_event.set()

        @self._sio.on("disconnect")
        def _on_disconnect():
            self._connected = False
            print("[FossBotClient] Disconnected.")

    # ─────────────────────────────────────────
    #  LAB MODE / STEP-SYNC  (Chapter 5)
    # ─────────────────────────────────────────

    def enable_lab_mode(self, steps_per_action: int = 4) -> None:
        """Switch simulator to step-locked Laboratory Mode."""
        self._sio.emit("enable_lab_mode", {
            "enabled": True,
            "steps_per_action": steps_per_action
        })
        # Wait for initial state
        self._state_event.wait(self.timeout)
        self._state_event.clear()

    def step(self, action: Dict[str, float]) -> Dict[str, Any]:
        """
        Send one action and block until the simulator returns the next state.
        action = { "throttle": float, "steer": float, "brake": float }
        Returns the full state packet.
        Chapter 5.1.2 — state → action → next state protocol.
        """
        self._state_event.clear()
        self._sio.emit("step_action", action)
        if not self._state_event.wait(self.timeout):
            raise TimeoutError("Simulator did not respond to step action in time.")
        return dict(self._state)

    def get_state(self) -> Dict[str, Any]:
        """Poll current state (non-lab mode fallback)."""
        self._state_event.clear()
        self._sio.emit("get_state", {})
        self._state_event.wait(self.timeout)
        return dict(self._state)

    # ─────────────────────────────────────────
    #  SCENARIO  (Chapter 6)
    # ─────────────────────────────────────────

    def load_scenario(self, scenario: ScenarioConfig) -> None:
        """Send a scenario config to Godot for procedural instantiation."""
        self._scenario_event.clear()
        self._sio.emit("load_scenario", scenario.to_dict())
        if not self._scenario_event.wait(self.timeout):
            raise TimeoutError("Scenario load timed out.")
        print(f"[FossBotClient] Scenario '{scenario.name}' loaded (seed={scenario.seed})")

    def load_scenario_file(self, path: str) -> None:
        """Parse YAML/JSON scenario file and send to simulator."""
        cfg = ScenarioParser.load(path)
        self.load_scenario(cfg)

    def reset_episode(self, spawn_pose: Optional[Dict] = None) -> Dict[str, Any]:
        """Soft-reset: respawn robot, keep scenario."""
        self._reset_event.clear()
        self._sio.emit("reset_episode", {"spawn_pose": spawn_pose or {}})
        self._reset_event.wait(self.timeout)
        return dict(self._state)

    # ─────────────────────────────────────────
    #  ROBOT COMMANDS  (direct / non-lab mode)
    # ─────────────────────────────────────────

    def move_forward(self, speed: float = 1.0) -> None:
        self._sio.emit("move_forward", {"speed": speed})

    def move_backward(self, speed: float = 1.0) -> None:
        self._sio.emit("move_backward", {"speed": speed})

    def turn_left(self, speed: float = 0.5) -> None:
        self._sio.emit("turn_left", {"speed": speed})

    def turn_right(self, speed: float = 0.5) -> None:
        self._sio.emit("turn_right", {"speed": speed})

    def stop(self) -> None:
        self._sio.emit("stop", {})

    # ─────────────────────────────────────────
    #  DIAGNOSTICS / VISUAL OVERLAYS  (Chapter 10)
    # ─────────────────────────────────────────

    def set_planned_path(self, path: List[List[float]]) -> None:
        """Send global planned path for Godot overlay. path = [[x,y,z], ...]"""
        self._sio.emit("set_planned_path", {"path": path})

    def set_intent_path(self, path: List[List[float]]) -> None:
        """Send local DWA/intent trajectory for overlay."""
        self._sio.emit("set_intent_path", {"path": path})

    def set_state_label(self, label: str) -> None:
        """Update HUDD state-machine label (e.g., 'Exploring', 'Avoiding Obstacle')."""
        self._sio.emit("set_state_label", {"label": label})

    # ─────────────────────────────────────────
    #  COSTMAP  (Chapter 9.1.3)
    # ─────────────────────────────────────────

    def get_costmap(self) -> List[Dict]:
        """Fetch terrain traversal costmap from simulator."""
        self._costmap_event.clear()
        self._sio.emit("get_costmap", {})
        if not self._costmap_event.wait(self.timeout):
            raise TimeoutError("Costmap fetch timed out.")
        return list(self._costmap)

    def get_costmap_grid(self, default_cost: float = 1.0) -> Dict:
        """
        Returns costmap as a dict keyed by (cell_x, cell_y) → cost.
        Ready for use in A* or RL reward shaping.
        """
        cells = self.get_costmap()
        grid: Dict = {}
        for c in cells:
            grid[(c["cell_x"], c["cell_y"])] = c.get("cost", default_cost)
        return grid
