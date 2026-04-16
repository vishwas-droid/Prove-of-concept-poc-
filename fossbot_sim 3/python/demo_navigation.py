"""
demo_navigation.py
==================
Full integration demonstration combining all implemented systems:
  - Lab Mode step-sync (Ch 5)
  - Scenario loading from YAML (Ch 6)
  - Modular sensor readings (Ch 7)
  - Terrain-aware A* global planner (Ch 9)
  - DWA local planner with intent overlay (Ch 10)
  - HUDD state labels + path overlays (Ch 10)

Run the server and Godot first, then:
    python demo_navigation.py [--scenario scenarios/example_maze.yaml] [--url http://localhost:5000]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Optional, Tuple

from fossbot_client import FossBotClient
from scenario_parser import ScenarioParser, ScenarioConfig, write_example_scenario
from planners.astar import AStarPlanner, plan_and_send
from planners.dwa import DWAPlanner, DWAConfig, state_dict_to_robot_state, obstacles_from_state


# ─────────────────────────────────────────────
#  NAVIGATOR
# ─────────────────────────────────────────────

class FossBotNavigator:
    """
    Hybrid navigator: A* global plan + DWA local avoidance.
    Demonstrates full integration of all thesis chapters.
    """

    GOAL_TOLERANCE  = 0.4   # metres — episode success
    REPLAN_INTERVAL = 20    # steps between global replans
    MAX_STEPS       = 2000

    def __init__(
        self,
        client: FossBotClient,
        scenario: ScenarioConfig,
        dwa_config: Optional[DWAConfig] = None
    ):
        self.client   = client
        self.scenario = scenario
        self.goal     = (scenario.goal_coordinates.x, scenario.goal_coordinates.z)
        self.dwa      = DWAPlanner(dwa_config or DWAConfig())

        # Statistics
        self.steps       = 0
        self.total_reward = 0.0
        self.path_history: List[Tuple[float, float]] = []

    # ─────────────────────────────────────────
    #  EPISODE RUN
    # ─────────────────────────────────────────

    def run_episode(self) -> bool:
        """
        Execute one navigation episode.
        Returns True if goal reached, False otherwise.
        """
        print(f"\n[Navigator] Goal: {self.goal}")
        print(f"[Navigator] Max steps: {self.MAX_STEPS}")

        state = self._get_current_state()
        global_path = self._global_plan(state)

        for step in range(self.MAX_STEPS):
            self.steps = step

            # ── Periodic global replan
            if step % self.REPLAN_INTERVAL == 0 and step > 0:
                state = self._get_current_state()
                global_path = self._global_plan(state)

            # ── DWA local control
            state = self._get_current_state()
            robot_state  = state_dict_to_robot_state(state)
            obstacles    = obstacles_from_state(state, obstacle_range=1.8)
            v, w, intent = self.dwa.compute(robot_state, self.goal, obstacles)
            action       = self.dwa.speed_to_action(v, w)

            # ── Send intent path overlay (Chapter 10.1.2)
            self.client.set_intent_path(intent)

            # ── Determine and send HUDD state label (Chapter 10.1.3)
            label = self._classify_state(state, obstacles, v)
            self.client.set_state_label(label)

            # ── Execute action
            state = self.client.step(action)
            pose  = state.get("pose", {})
            rx, rz = pose.get("x", 0.0), pose.get("z", 0.0)
            self.path_history.append((rx, rz))

            # ── Check goal
            dist = math.hypot(self.goal[0] - rx, self.goal[1] - rz)
            if dist < self.GOAL_TOLERANCE:
                self.client.set_state_label("Goal Reached ✓")
                print(f"\n[Navigator] GOAL REACHED in {step+1} steps!")
                return True

            # ── Progress log
            if step % 50 == 0:
                terrain = state.get("terrain", {})
                sonar_f = state.get("sensors", {}).get("sonar_front", {}).get("distance_m", 4.0)
                print(
                    f"  step={step:4d}  dist={dist:.2f}m  "
                    f"v={v:.2f}  w={w:.2f}  "
                    f"sonar_f={sonar_f:.2f}m  "
                    f"terrain={terrain.get('material','?')}(cost={terrain.get('cost',1):.1f})  "
                    f"{label}"
                )

        print(f"\n[Navigator] Timeout after {self.MAX_STEPS} steps.")
        self.client.set_state_label("Timeout")
        return False

    # ─────────────────────────────────────────
    #  GLOBAL PLANNER
    # ─────────────────────────────────────────

    def _global_plan(self, state: dict) -> List[List[float]]:
        pose = state.get("pose", {})
        rx   = pose.get("x", 0.0)
        rz   = pose.get("z", 0.0)

        self.client.set_state_label("Planning...")
        path = plan_and_send(
            client      = self.client,
            start_world = (rx, rz),
            goal_world  = self.goal,
            cell_size   = 1.0,
            smooth      = True
        )
        if path is None:
            print("[Navigator] A* found no path — proceeding with DWA only.")
            return []
        print(f"[Navigator] Global path: {len(path)} waypoints")
        return path

    # ─────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────

    def _get_current_state(self) -> dict:
        """In lab mode the last step() already contains fresh state.
           This helper exists for the initial call and replans."""
        return self.client.get_state()

    def _classify_state(self, state: dict, obstacles: list, speed: float) -> str:
        """Chapter 10.1.3 — State-Machine Monitor label logic."""
        sonar_f = state.get("sensors", {}).get("sonar_front", {}).get("distance_m", 4.0)
        pose    = state.get("pose", {})
        dist    = math.hypot(
            self.goal[0] - pose.get("x", 0.0),
            self.goal[1] - pose.get("z", 0.0)
        )

        if dist < 1.5:
            return "Approaching Goal"
        if len(obstacles) > 0 or sonar_f < 0.4:
            return "Avoiding Obstacle"
        if abs(speed) < 0.05:
            return "Stopped"
        return "Exploring"

    # ─────────────────────────────────────────
    #  REPORT
    # ─────────────────────────────────────────

    def print_summary(self, success: bool) -> None:
        print("\n" + "═" * 50)
        print("  NAVIGATION SUMMARY")
        print("═" * 50)
        print(f"  Result  : {'SUCCESS ✓' if success else 'FAILED ✗'}")
        print(f"  Steps   : {self.steps}")
        print(f"  Waypoints: {len(self.path_history)}")
        if self.path_history:
            total_dist = sum(
                math.hypot(
                    self.path_history[i][0] - self.path_history[i-1][0],
                    self.path_history[i][1] - self.path_history[i-1][1]
                )
                for i in range(1, len(self.path_history))
            )
            print(f"  Distance: {total_dist:.2f} m")
        print("═" * 50 + "\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FOSSBot full-system demo")
    parser.add_argument("--scenario", default="scenarios/example_maze.yaml")
    parser.add_argument("--url",      default="http://localhost:5000")
    parser.add_argument("--seed",     default=None, type=int)
    parser.add_argument("--write-scenario", action="store_true",
                        help="Write example scenario YAML and exit")
    args = parser.parse_args()

    if args.write_scenario:
        write_example_scenario(args.scenario)
        print(f"Written: {args.scenario}")
        return

    # ── Load scenario
    try:
        scenario = ScenarioParser.load(args.scenario)
    except FileNotFoundError:
        print(f"Scenario not found: {args.scenario}")
        print("Run with --write-scenario to create an example, then re-run.")
        sys.exit(1)

    if args.seed is not None:
        scenario.seed = args.seed
    print(f"[Demo] Scenario: {scenario.name}  seed={scenario.seed}")

    # ── Connect and configure
    with FossBotClient(args.url, timeout=15.0) as client:
        print("[Demo] Loading scenario...")
        client.load_scenario(scenario)

        print("[Demo] Enabling lab mode (step-sync)...")
        client.enable_lab_mode(steps_per_action=4)

        # ── Run navigation
        nav = FossBotNavigator(client, scenario)
        success = nav.run_episode()
        nav.print_summary(success)


if __name__ == "__main__":
    main()
