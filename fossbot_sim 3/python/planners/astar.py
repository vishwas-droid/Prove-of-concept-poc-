"""
planners/astar.py
=================
Costmap-Aware A* Planner — Chapter 9.1.3
Implements Cost-Aware A* where terrain cost modulates g-score,
enabling the robot to prefer longer-but-cheaper paths over
short-but-costly ones (e.g., avoiding mud/ice).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Type aliases
Cell = Tuple[int, int]          # (cell_x, cell_z) integer grid coords
Path = List[Cell]
Costmap = Dict[Cell, float]     # cell → terrain cost (1.0 = normal, >1 = expensive)


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────

@dataclass(order=True)
class _Node:
    f: float
    cell: Cell = field(compare=False)
    g: float   = field(compare=False)
    parent: Optional["_Node"] = field(compare=False, default=None)


# ─────────────────────────────────────────────
#  PLANNER
# ─────────────────────────────────────────────

class AStarPlanner:
    """
    Terrain-aware A* planner that consults the FOSSBot costmap.

    Usage:
        planner = AStarPlanner(costmap, allow_diagonal=True)
        path = planner.plan(start=(0, 0), goal=(10, 8))
        world_path = planner.to_world_coords(path, cell_size=1.0, y=0.1)
    """

    # 4-connected or 8-connected neighbour offsets
    _DIRS_4 = [(1,0),(-1,0),(0,1),(0,-1)]
    _DIRS_8 = _DIRS_4 + [(1,1),(1,-1),(-1,1),(-1,-1)]

    def __init__(
        self,
        costmap: Costmap,
        allow_diagonal: bool = True,
        heuristic: str = "euclidean",   # "manhattan" | "euclidean" | "octile"
        obstacle_cost_threshold: float = 9.0,   # cells with cost >= this are walls
        default_cost: float = 1.0
    ):
        self.costmap = costmap
        self.dirs = self._DIRS_8 if allow_diagonal else self._DIRS_4
        self.heuristic_name = heuristic
        self.obstacle_threshold = obstacle_cost_threshold
        self.default_cost = default_cost

    # ── Public API ────────────────────────────

    def plan(self, start: Cell, goal: Cell) -> Optional[Path]:
        """
        Find the cost-optimal path from start to goal.
        Returns list of (cell_x, cell_z) tuples, or None if no path found.
        """
        open_heap: List[_Node] = []
        open_set: Dict[Cell, float] = {}    # cell → best g seen
        closed: Set[Cell] = set()

        start_node = _Node(f=self._h(start, goal), cell=start, g=0.0)
        heapq.heappush(open_heap, start_node)
        open_set[start] = 0.0

        while open_heap:
            current = heapq.heappop(open_heap)

            if current.cell in closed:
                continue
            closed.add(current.cell)

            if current.cell == goal:
                return self._reconstruct(current)

            for dx, dz in self.dirs:
                nb = (current.cell[0] + dx, current.cell[1] + dz)
                if nb in closed:
                    continue

                terrain_cost = self.costmap.get(nb, self.default_cost)
                if terrain_cost >= self.obstacle_threshold:
                    continue    # treat as wall

                # Diagonal moves cost √2 * terrain_cost
                move_cost = math.sqrt(2) * terrain_cost if abs(dx) + abs(dz) == 2 else terrain_cost
                tentative_g = current.g + move_cost

                if nb not in open_set or tentative_g < open_set[nb]:
                    open_set[nb] = tentative_g
                    nb_node = _Node(
                        f=tentative_g + self._h(nb, goal),
                        cell=nb,
                        g=tentative_g,
                        parent=current
                    )
                    heapq.heappush(open_heap, nb_node)

        return None     # no path found

    def plan_multi_goal(self, start: Cell, goals: List[Cell]) -> Optional[Path]:
        """
        Find cheapest path that visits the closest goal.
        Useful for multi-target navigation.
        """
        best_path: Optional[Path] = None
        best_cost = float("inf")

        for goal in goals:
            path = self.plan(start, goal)
            if path is None:
                continue
            cost = self._path_cost(path)
            if cost < best_cost:
                best_cost = cost
                best_path = path

        return best_path

    def to_world_coords(
        self,
        path: Path,
        cell_size: float = 1.0,
        y: float = 0.1
    ) -> List[List[float]]:
        """Convert integer cell path → list of [x, y, z] world coordinates."""
        return [[cx * cell_size, y, cz * cell_size] for cx, cz in path]

    def smooth_path(self, path: Path, iterations: int = 3) -> Path:
        """
        Simple path smoothing using line-of-sight pruning.
        Removes waypoints that can be skipped without crossing obstacles.
        """
        if len(path) <= 2:
            return path

        for _ in range(iterations):
            smoothed = [path[0]]
            i = 0
            while i < len(path) - 1:
                j = len(path) - 1
                while j > i + 1:
                    if self._line_of_sight(path[i], path[j]):
                        break
                    j -= 1
                smoothed.append(path[j])
                i = j
            path = smoothed

        return path

    # ── Heuristics ────────────────────────────

    def _h(self, a: Cell, b: Cell) -> float:
        dx = abs(a[0] - b[0])
        dz = abs(a[1] - b[1])
        match self.heuristic_name:
            case "manhattan":
                return dx + dz
            case "octile":
                return max(dx, dz) + (math.sqrt(2) - 1) * min(dx, dz)
            case _:  # euclidean
                return math.hypot(dx, dz)

    # ── Utilities ─────────────────────────────

    @staticmethod
    def _reconstruct(node: _Node) -> Path:
        path: Path = []
        cur: Optional[_Node] = node
        while cur is not None:
            path.append(cur.cell)
            cur = cur.parent
        return list(reversed(path))

    def _path_cost(self, path: Path) -> float:
        total = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dz = path[i][1] - path[i-1][1]
            move = math.sqrt(2) if abs(dx) + abs(dz) == 2 else 1.0
            total += move * self.costmap.get(path[i], self.default_cost)
        return total

    def _line_of_sight(self, a: Cell, b: Cell) -> bool:
        """Bresenham line-of-sight check — True if no obstacle between a and b."""
        x0, z0 = a
        x1, z1 = b
        dx = abs(x1 - x0); dz = abs(z1 - z0)
        sx = 1 if x1 > x0 else -1
        sz = 1 if z1 > z0 else -1
        err = dx - dz

        while True:
            if self.costmap.get((x0, z0), self.default_cost) >= self.obstacle_threshold:
                return False
            if x0 == x1 and z0 == z1:
                return True
            e2 = 2 * err
            if e2 > -dz:
                err -= dz
                x0 += sx
            if e2 < dx:
                err += dx
                z0 += sz


# ─────────────────────────────────────────────
#  CONVENIENCE FUNCTION
# ─────────────────────────────────────────────

def plan_and_send(
    client,     # FossBotClient
    start_world: Tuple[float, float],
    goal_world:  Tuple[float, float],
    cell_size: float = 1.0,
    smooth: bool = True,
    y_height: float = 0.1
) -> Optional[List[List[float]]]:
    """
    Fetch costmap, run A*, smooth, send overlay to Godot, return world path.

    Args:
        client:       Connected FossBotClient instance.
        start_world:  (x, z) robot position in world metres.
        goal_world:   (x, z) goal position in world metres.
        cell_size:    Metres per costmap cell (default 1.0).
        smooth:       Apply line-of-sight smoothing.
        y_height:     Path visualisation Y offset.
    """
    costmap = client.get_costmap_grid()

    start_cell: Cell = (int(start_world[0] / cell_size), int(start_world[1] / cell_size))
    goal_cell:  Cell = (int(goal_world[0]  / cell_size), int(goal_world[1]  / cell_size))

    planner = AStarPlanner(costmap, allow_diagonal=True, heuristic="octile")
    path = planner.plan(start_cell, goal_cell)

    if path is None:
        print("[AStarPlanner] No path found.")
        return None

    if smooth:
        path = planner.smooth_path(path)

    world_path = planner.to_world_coords(path, cell_size=cell_size, y=y_height)
    client.set_planned_path(world_path)
    client.set_state_label("Planning")
    return world_path
