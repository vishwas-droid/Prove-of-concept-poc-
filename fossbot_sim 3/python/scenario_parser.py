"""
scenario_parser.py
==================
YAML/JSON Scenario Configuration Engine — Chapter 6.1
Parses scenario files and sends them to the Godot simulator via FossBot client.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # pip install pyyaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class SpawnPose:
    x: float = 0.0
    y: float = 0.1
    z: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class GoalCoordinates:
    x: float = 5.0
    z: float = 5.0


@dataclass
class ObstacleConfig:
    type: str = "box"           # "box" | "cylinder" | "sphere"
    x: float = 0.0
    y: float = 0.5
    z: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    rotation_deg: float = 0.0


@dataclass
class TerrainScale:
    x: float = 100.0
    y: float = 10.0
    z: float = 100.0


@dataclass
class NoiseOverride:
    """Per-sensor noise override (merged into SimInfo.noise_config)."""
    sensor: str = ""
    enabled: bool = True
    stddev: Optional[float] = None
    prob: Optional[float] = None


@dataclass
class ScenarioConfig:
    name: str = "unnamed"
    seed: int = 42
    spawn_pose: SpawnPose = field(default_factory=SpawnPose)
    goal_coordinates: GoalCoordinates = field(default_factory=GoalCoordinates)
    heightmap_path: str = ""
    material_map_path: str = ""
    floor_texture_map: str = ""
    terrain_scale: TerrainScale = field(default_factory=TerrainScale)
    obstacles: List[ObstacleConfig] = field(default_factory=list)
    noise_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScenarioConfig":
        cfg = cls()
        cfg.name = d.get("name", "unnamed")
        cfg.seed = d.get("seed", 42)

        sp = d.get("spawn_pose", {})
        cfg.spawn_pose = SpawnPose(**{k: sp[k] for k in sp if k in SpawnPose.__dataclass_fields__})

        gc = d.get("goal_coordinates", {})
        cfg.goal_coordinates = GoalCoordinates(**{k: gc[k] for k in gc if k in GoalCoordinates.__dataclass_fields__})

        cfg.heightmap_path    = d.get("heightmap_path", "")
        cfg.material_map_path = d.get("material_map_path", "")
        cfg.floor_texture_map = d.get("floor_texture_map", "")

        ts = d.get("terrain_scale", {})
        cfg.terrain_scale = TerrainScale(**{k: ts[k] for k in ts if k in TerrainScale.__dataclass_fields__})

        cfg.obstacles = [ObstacleConfig(**{k: o[k] for k in o if k in ObstacleConfig.__dataclass_fields__})
                         for o in d.get("obstacles", [])]

        cfg.noise_config = d.get("noise_config", {})
        return cfg


# ─────────────────────────────────────────────
#  PARSER
# ─────────────────────────────────────────────

class ScenarioParser:
    """
    Loads a YAML or JSON scenario file and returns a ScenarioConfig.
    Chapter 6.1.1 — Environment as Data
    """

    @staticmethod
    def load(path: str | Path) -> ScenarioConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        raw: Dict[str, Any]
        if path.suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML scenarios. pip install pyyaml")
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raise ValueError(f"Unsupported scenario format: {path.suffix}")

        return ScenarioConfig.from_dict(raw)

    @staticmethod
    def save(cfg: ScenarioConfig, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = cfg.to_dict()
        if path.suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ImportError("PyYAML required.")
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)

    @staticmethod
    def generate_random(
        name: str = "random_scenario",
        seed: Optional[int] = None,
        num_obstacles: int = 5,
        map_size: float = 20.0
    ) -> ScenarioConfig:
        """
        Procedurally generate a random scenario.
        Chapter 6.1.3 — Deterministic Seeded Randomization
        """
        if seed is None:
            seed = random.randint(0, 999999)
        rng = random.Random(seed)

        obstacles = []
        for _ in range(num_obstacles):
            obs = ObstacleConfig(
                type=rng.choice(["box", "cylinder", "sphere"]),
                x=rng.uniform(-map_size / 2, map_size / 2),
                y=0.5,
                z=rng.uniform(-map_size / 2, map_size / 2),
                scale_x=rng.uniform(0.3, 1.5),
                scale_y=rng.uniform(0.5, 2.0),
                scale_z=rng.uniform(0.3, 1.5),
                rotation_deg=rng.uniform(0, 360)
            )
            obstacles.append(obs)

        return ScenarioConfig(
            name=name,
            seed=seed,
            spawn_pose=SpawnPose(x=0.0, z=0.0, yaw_deg=0.0),
            goal_coordinates=GoalCoordinates(
                x=rng.uniform(map_size * 0.3, map_size * 0.5),
                z=rng.uniform(map_size * 0.3, map_size * 0.5)
            ),
            obstacles=obstacles
        )


# ─────────────────────────────────────────────
#  EXAMPLE SCENARIO (written to file on import)
# ─────────────────────────────────────────────

EXAMPLE_SCENARIO_YAML = """\
name: "example_maze"
seed: 1337

spawn_pose:
  x: 0.0
  y: 0.1
  z: 0.0
  yaw_deg: 90.0

goal_coordinates:
  x: 8.0
  z: 8.0

heightmap_path: "assets/terrain/flat.png"
material_map_path: "assets/terrain/materials.png"
floor_texture_map: "assets/terrain/diffuse.png"

terrain_scale:
  x: 20.0
  y: 2.0
  z: 20.0

obstacles:
  - type: box
    x: 2.0
    y: 0.5
    z: 2.0
    scale_x: 1.0
    scale_y: 1.0
    scale_z: 1.0

  - type: cylinder
    x: -3.0
    y: 0.75
    z: 4.0
    scale_x: 0.6
    scale_y: 1.5
    scale_z: 0.6

  - type: sphere
    x: 5.0
    y: 0.5
    z: -2.0
    scale_x: 0.8
    scale_y: 0.8
    scale_z: 0.8

noise_config:
  ultrasonic:
    enabled: true
    stddev: 0.03
  ir:
    enabled: true
    prob: 0.04
  imu:
    enabled: true
    drift_rate: 0.002
"""


def write_example_scenario(path: str = "scenarios/example_maze.yaml") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(EXAMPLE_SCENARIO_YAML, encoding="utf-8")
    print(f"[ScenarioParser] Example scenario written to {p}")
