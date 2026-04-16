"""
benchmarking/runner.py
======================
Chapter 16 — Quantitative Benchmarking & Automated Evaluation

Components:
  - KPICollector   : computes 4 standardised metrics per episode
  - BatchRunner    : runs N trials per config from a YAML experiment file
  - ReportGenerator: produces Markdown + CSV reports with statistics

Usage:
    python -m benchmarking.runner --config experiments/terrain_comparison.yaml
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


# ─────────────────────────────────────────────
#  KPI DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class EpisodeKPIs:
    """Chapter 16.1.3 — Standardised KPI Suite for one episode."""
    run_id:           str   = ""
    scenario:         str   = ""
    algorithm:        str   = ""
    seed:             int   = 0
    noise_level:      str   = "medium"

    # KPI 1 — Success
    success:          bool  = False

    # KPI 2 — Path Efficiency Index
    actual_path_m:    float = 0.0
    optimal_path_m:   float = 0.0
    path_efficiency:  float = 0.0   # optimal / actual  (1.0 = perfect)

    # KPI 3 — Smoothness Score
    angular_vel_std:  float = 0.0   # lower = smoother
    throttle_std:     float = 0.0

    # KPI 4 — Steps taken
    steps:            int   = 0
    max_steps:        int   = 1000

    # Metadata
    duration_s:       float = 0.0
    final_dist_m:     float = 0.0
    timestamp:        str   = ""


@dataclass
class ExperimentConfig:
    """Parsed experiment YAML config."""
    name:             str          = "experiment"
    trials_per_config: int         = 10
    scenarios:        List[str]    = field(default_factory=list)
    algorithms:       List[Dict]   = field(default_factory=list)
    noise_levels:     List[str]    = field(default_factory=lambda: ["medium"])
    seeds:            List[int]    = field(default_factory=lambda: list(range(10)))
    simulator_url:    str          = "http://localhost:5000"
    output_dir:       str          = "results/"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        if not _YAML_OK:
            raise ImportError("pip install pyyaml")
        with open(path) as f:
            d = yaml.safe_load(f)
        cfg = cls()
        exp = d.get("experiment", d)
        cfg.name              = exp.get("name", "experiment")
        cfg.trials_per_config = exp.get("trials_per_config", 10)
        cfg.scenarios         = exp.get("scenarios", [])
        cfg.algorithms        = exp.get("algorithms", [])
        cfg.noise_levels      = exp.get("noise_levels", ["medium"])
        cfg.seeds             = exp.get("seeds", list(range(10)))
        cfg.simulator_url     = exp.get("simulator_url", "http://localhost:5000")
        cfg.output_dir        = exp.get("output_dir", "results/")
        return cfg


# ─────────────────────────────────────────────
#  KPI COLLECTOR
# ─────────────────────────────────────────────

class KPICollector:
    """
    Collects data during an episode and computes KPIs at the end.
    Chapter 16.1.3
    """

    def __init__(self, goal: Tuple[float, float], max_steps: int = 1000):
        self.goal = goal
        self.max_steps = max_steps
        self._poses: List[Tuple[float, float]] = []
        self._angular_vels: List[float] = []
        self._throttles: List[float] = []
        self._start_time = time.time()

    def record_step(self, state: Dict, action: Dict) -> None:
        """Call after every env.step()."""
        pose = state.get("pose", {})
        x = pose.get("x", 0.0)
        z = pose.get("z", 0.0)
        self._poses.append((x, z))

        imu = state.get("sensors", {}).get("imu", {})
        ang = imu.get("angular_velocity", {}).get("y", 0.0)
        self._angular_vels.append(ang)
        self._throttles.append(float(action.get("throttle", 0.0)))

    def compute(
        self,
        success: bool,
        steps: int,
        run_id: str = "",
        scenario: str = "",
        algorithm: str = "",
        seed: int = 0,
        noise_level: str = "medium"
    ) -> EpisodeKPIs:
        kpi = EpisodeKPIs(
            run_id=run_id,
            scenario=scenario,
            algorithm=algorithm,
            seed=seed,
            noise_level=noise_level,
            success=success,
            steps=steps,
            max_steps=self.max_steps,
            duration_s=time.time() - self._start_time,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
        )

        # Path length
        kpi.actual_path_m = self._compute_path_length()

        # Optimal path = straight line from start to goal
        if self._poses:
            sx, sz = self._poses[0]
            kpi.optimal_path_m = math.hypot(self.goal[0] - sx, self.goal[1] - sz)
            kpi.final_dist_m   = math.hypot(self.goal[0] - self._poses[-1][0],
                                             self.goal[1] - self._poses[-1][1])

        # Path efficiency (capped at 1.0, lower = worse)
        if kpi.actual_path_m > 0 and kpi.optimal_path_m > 0:
            kpi.path_efficiency = min(1.0, kpi.optimal_path_m / kpi.actual_path_m)

        # Smoothness
        if len(self._angular_vels) > 1:
            kpi.angular_vel_std = float(statistics.stdev(self._angular_vels))
        if len(self._throttles) > 1:
            kpi.throttle_std = float(statistics.stdev(self._throttles))

        return kpi

    def _compute_path_length(self) -> float:
        total = 0.0
        for i in range(1, len(self._poses)):
            dx = self._poses[i][0] - self._poses[i-1][0]
            dz = self._poses[i][1] - self._poses[i-1][1]
            total += math.hypot(dx, dz)
        return total


# ─────────────────────────────────────────────
#  BATCH RUNNER
# ─────────────────────────────────────────────

class BatchRunner:
    """
    Chapter 16.1.2 — Headless Batch Experiment Runner.
    Iterates over all (scenario, algorithm, noise_level, seed) combinations
    and runs trials_per_config episodes for each.
    """

    NOISE_CONFIGS = {
        "low":    {"ultrasonic": {"stddev": 0.01}, "ir": {"prob": 0.01}, "imu": {"drift_rate": 0.0005}},
        "medium": {"ultrasonic": {"stddev": 0.03}, "ir": {"prob": 0.05}, "imu": {"drift_rate": 0.001}},
        "high":   {"ultrasonic": {"stddev": 0.06}, "ir": {"prob": 0.10}, "imu": {"drift_rate": 0.003}},
    }

    def __init__(self, exp_config: ExperimentConfig):
        self.cfg = exp_config
        self.results: List[EpisodeKPIs] = []
        Path(exp_config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self) -> List[EpisodeKPIs]:
        """Run all experiment combinations. Returns list of EpisodeKPIs."""
        total = (len(self.cfg.scenarios) * len(self.cfg.algorithms) *
                 len(self.cfg.noise_levels) * self.cfg.trials_per_config)
        done = 0

        print(f"\n[BatchRunner] Starting: {self.cfg.name}")
        print(f"[BatchRunner] Total trials: {total}\n")

        for scenario_path in self.cfg.scenarios:
            for algo_cfg in self.cfg.algorithms:
                for noise_level in self.cfg.noise_levels:
                    for trial in range(self.cfg.trials_per_config):
                        seed = self.cfg.seeds[trial % len(self.cfg.seeds)]
                        run_id = f"{algo_cfg['name']}_{Path(scenario_path).stem}_{noise_level}_s{seed}_t{trial}"

                        print(f"  [{done+1}/{total}] {run_id}")
                        kpi = self._run_single_trial(
                            scenario_path=scenario_path,
                            algo_cfg=algo_cfg,
                            noise_level=noise_level,
                            seed=seed,
                            run_id=run_id
                        )
                        self.results.append(kpi)
                        done += 1

        # Save raw results
        self._save_results()
        print(f"\n[BatchRunner] Done. Results saved to {self.cfg.output_dir}")
        return self.results

    def _run_single_trial(
        self,
        scenario_path: str,
        algo_cfg: Dict,
        noise_level: str,
        seed: int,
        run_id: str
    ) -> EpisodeKPIs:
        """Run one episode and return its KPIs."""
        # Import here to avoid circular imports at module level
        from fossbot_client import FossBotClient
        from scenario_parser import ScenarioParser
        from safety.middleware import SafetyMiddleware, SafetyConfig

        scenario = ScenarioParser.load(scenario_path)
        scenario.seed = seed
        # Apply noise level override
        scenario.noise_config = self.NOISE_CONFIGS.get(noise_level, {})

        goal = (scenario.goal_coordinates.x, scenario.goal_coordinates.z)

        # Load model
        model = self._load_model(algo_cfg)

        collector = KPICollector(goal=goal, max_steps=1000)
        success = False
        steps = 0

        try:
            with FossBotClient(self.cfg.simulator_url, timeout=15.0) as client:
                safety = SafetyMiddleware(client, SafetyConfig(log_interventions=False))
                client.load_scenario(scenario)
                client.enable_lab_mode(steps_per_action=4)

                obs = self._get_obs(client.get_state())

                for step in range(1000):
                    steps = step + 1
                    action_arr, _ = model.predict(obs, deterministic=True)
                    action = {
                        "throttle": float(action_arr[0]),
                        "steer":    float(action_arr[1]),
                        "brake":    0.0
                    }
                    state = safety.safe_step(action)
                    collector.record_step(state, action)
                    obs = self._get_obs(state)

                    dist = self._dist_to_goal(state, goal)
                    if dist < 0.4:
                        success = True
                        break

                    if state.get("done", False):
                        break

        except Exception as e:
            print(f"    [ERROR] {run_id}: {e}")

        return collector.compute(
            success=success,
            steps=steps,
            run_id=run_id,
            scenario=Path(scenario_path).stem,
            algorithm=algo_cfg["name"],
            seed=seed,
            noise_level=noise_level
        )

    def _load_model(self, algo_cfg: Dict):
        """Load a Stable-Baselines3 model from checkpoint path."""
        from stable_baselines3 import PPO, SAC, TD3, A2C
        ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C}
        cls = ALGOS.get(algo_cfg["name"].lower(), PPO)
        return cls.load(algo_cfg["checkpoint"])

    def _get_obs(self, state: Dict) -> np.ndarray:
        """Convert state dict to observation array (matches fossbot_env.py)."""
        sensors = state.get("sensors", {})
        pose    = state.get("pose", {})
        terrain = state.get("terrain", {})
        imu     = sensors.get("imu", {})

        def sonar(k): return sensors.get(k, {}).get("distance_m", 4.0) / 4.0 * 2 - 1
        def ir(k):    return 1.0 if sensors.get(k, {}).get("detected", False) else 0.0

        obs = np.array([
            sonar("sonar_front"), sonar("sonar_left"), sonar("sonar_right"),
            ir("ir_left"), ir("ir_center"), ir("ir_right"),
            pose.get("x", 0.0) / 50.0,
            pose.get("z", 0.0) / 50.0,
            pose.get("yaw_deg", 0.0) / 180.0,
            imu.get("linear_velocity", {}).get("x", 0.0),
            imu.get("linear_velocity", {}).get("z", 0.0),
            imu.get("angular_velocity", {}).get("y", 0.0),
            terrain.get("friction", 1.0) - 1.0,
            (terrain.get("cost", 1.0) - 1.5) / 1.5,
            0.0, 0.0   # goal relative (not available without scenario ref here)
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _dist_to_goal(self, state: Dict, goal: Tuple) -> float:
        pose = state.get("pose", {})
        return math.hypot(goal[0] - pose.get("x", 0.0), goal[1] - pose.get("z", 0.0))

    def _save_results(self) -> None:
        path = Path(self.cfg.output_dir) / f"{self.cfg.name}_results.json"
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"[BatchRunner] Raw results → {path}")


# ─────────────────────────────────────────────
#  REPORT GENERATOR
# ─────────────────────────────────────────────

class ReportGenerator:
    """
    Chapter 16.1.4 — Automated Report Generator.
    Takes a list of EpisodeKPIs and produces:
      - Markdown summary report
      - CSV for further analysis
    """

    def __init__(self, results: List[EpisodeKPIs], experiment_name: str = "experiment"):
        self.results = results
        self.name = experiment_name

    def generate(self, output_dir: str = "results/") -> Dict[str, str]:
        """Generate all report formats. Returns dict of {format: filepath}."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        paths = {}
        paths["markdown"] = self._write_markdown(output_dir)
        paths["csv"]      = self._write_csv(output_dir)
        return paths

    # ── Markdown Report ───────────────────────

    def _write_markdown(self, output_dir: str) -> str:
        path = os.path.join(output_dir, f"{self.name}_report.md")
        lines = []
        lines.append(f"# FOSSBot Benchmark Report: {self.name}")
        lines.append(f"\n*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
        lines.append("---\n")

        # ── Summary stats per algorithm
        lines.append("## Results by Algorithm\n")
        by_algo = self._group_by("algorithm")
        lines.append("| Algorithm | Trials | Success Rate | Path Efficiency | Smoothness (σω) | Avg Steps |")
        lines.append("|-----------|--------|-------------|-----------------|-----------------|-----------|")
        for algo, kpis in sorted(by_algo.items()):
            sr    = self._mean([1.0 if k.success else 0.0 for k in kpis])
            pe    = self._mean([k.path_efficiency for k in kpis if k.success])
            sm    = self._mean([k.angular_vel_std for k in kpis])
            steps = self._mean([k.steps for k in kpis])
            lines.append(f"| {algo} | {len(kpis)} | {sr:.1%} | {pe:.3f} | {sm:.4f} | {steps:.0f} |")

        # ── By noise level
        lines.append("\n## Results by Noise Level\n")
        by_noise = self._group_by("noise_level")
        lines.append("| Noise Level | Trials | Success Rate | Path Efficiency |")
        lines.append("|-------------|--------|-------------|-----------------|")
        for noise in ["low", "medium", "high"]:
            kpis = by_noise.get(noise, [])
            if not kpis:
                continue
            sr = self._mean([1.0 if k.success else 0.0 for k in kpis])
            pe = self._mean([k.path_efficiency for k in kpis if k.success])
            lines.append(f"| {noise} | {len(kpis)} | {sr:.1%} | {pe:.3f} |")

        # ── Robustness Score (success rate slope across noise levels)
        lines.append("\n## Robustness Analysis\n")
        lines.append("Robustness score = success rate at low noise minus success rate at high noise.")
        lines.append("Lower is more robust (less degradation).\n")
        by_algo_noise = {}
        for kpi in self.results:
            key = kpi.algorithm
            by_algo_noise.setdefault(key, {})
            by_algo_noise[key].setdefault(kpi.noise_level, []).append(kpi)
        lines.append("| Algorithm | Low SR | High SR | Robustness Score |")
        lines.append("|-----------|--------|---------|-----------------|")
        for algo, noise_dict in sorted(by_algo_noise.items()):
            lo = self._mean([1.0 if k.success else 0.0 for k in noise_dict.get("low", [])])
            hi = self._mean([1.0 if k.success else 0.0 for k in noise_dict.get("high", [])])
            rob = lo - hi   # lower = more robust
            lines.append(f"| {algo} | {lo:.1%} | {hi:.1%} | {rob:+.3f} |")

        # ── Per-scenario breakdown
        lines.append("\n## Per-Scenario Breakdown\n")
        by_scenario = self._group_by("scenario")
        for scenario, kpis in sorted(by_scenario.items()):
            sr = self._mean([1.0 if k.success else 0.0 for k in kpis])
            lines.append(f"### {scenario}")
            lines.append(f"- Trials: {len(kpis)}")
            lines.append(f"- Success Rate: {sr:.1%}")
            lines.append(f"- Avg Steps: {self._mean([k.steps for k in kpis]):.0f}")
            lines.append(f"- Avg Path Efficiency: {self._mean([k.path_efficiency for k in kpis if k.success]):.3f}")
            lines.append("")

        # ── Reproducibility footer
        lines.append("---\n")
        lines.append("## Reproducibility\n")
        lines.append("All results are seeded and deterministic. To reproduce:")
        lines.append("```bash")
        lines.append(f"python -m benchmarking.runner --config experiments/{self.name}.yaml")
        lines.append("```")
        seeds_used = sorted(set(k.seed for k in self.results))
        lines.append(f"\nSeeds used: {seeds_used}")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"[ReportGenerator] Markdown → {path}")
        return path

    # ── CSV Export ────────────────────────────

    def _write_csv(self, output_dir: str) -> str:
        path = os.path.join(output_dir, f"{self.name}_results.csv")
        if not self.results:
            return path
        fields = list(asdict(self.results[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for kpi in self.results:
                writer.writerow(asdict(kpi))
        print(f"[ReportGenerator] CSV → {path}")
        return path

    # ── Helpers ───────────────────────────────

    def _group_by(self, field_name: str) -> Dict[str, List[EpisodeKPIs]]:
        groups: Dict[str, List] = {}
        for kpi in self.results:
            key = getattr(kpi, field_name, "unknown")
            groups.setdefault(key, []).append(kpi)
        return groups

    @staticmethod
    def _mean(values: list) -> float:
        return statistics.mean(values) if values else 0.0

    @staticmethod
    def _stdev(values: list) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FOSSBot Benchmark Runner")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--report-only", default="", help="Generate report from existing JSON results file")
    args = parser.parse_args()

    if args.report_only:
        with open(args.report_only) as f:
            raw = json.load(f)
        results = [EpisodeKPIs(**r) for r in raw]
        name = Path(args.report_only).stem.replace("_results", "")
        gen = ReportGenerator(results, experiment_name=name)
        paths = gen.generate()
        print("Reports generated:", paths)
        return

    exp_cfg = ExperimentConfig.from_yaml(args.config)
    runner = BatchRunner(exp_cfg)
    results = runner.run()
    gen = ReportGenerator(results, experiment_name=exp_cfg.name)
    paths = gen.generate(exp_cfg.output_dir)
    print("\nAll done. Reports:", paths)


if __name__ == "__main__":
    main()
