"""
Pheromone Map — spatial scent grids that decay over time.

Three grids are maintained:
  - threat_scent:   where enemy units have recently been seen
  - ally_trail:     where our army has recently moved
  - enemy_building: where we've spotted enemy structures (slow decay)

All grids use the same coordinate space as python-sc2 Point2
(game-space, not pixel-space). Grid resolution is configurable.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from sc2.position import Point2

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot


@dataclass
class PheromoneConfig:
    resolution: float = 2.0         # game-units per cell
    threat_decay: float = 0.92      # per-step multiplier (higher = slower decay)
    ally_decay: float = 0.85
    building_decay: float = 0.97    # structures persist longer
    deposit_strength: float = 1.0   # scent deposited per unit per step
    max_scent: float = 10.0         # clamp ceiling


class PheromoneMap:
    """
    Maintains named scent grids over the SC2 map.
    Call update() once per step from on_step().
    """

    def __init__(self, bot: 'ManifestorBot', config: PheromoneConfig | None = None):
        self.bot = bot
        self.cfg = config or PheromoneConfig()

        # Derived dimensions from map size
        map_data = bot.game_info.pathing_grid
        self._map_w = map_data.width
        self._map_h = map_data.height
        self._cols = max(1, int(self._map_w / self.cfg.resolution))
        self._rows = max(1, int(self._map_h / self.cfg.resolution))

        self.threat_scent   = np.zeros((self._rows, self._cols), dtype=np.float32)
        self.ally_trail     = np.zeros((self._rows, self._cols), dtype=np.float32)
        self.enemy_building = np.zeros((self._rows, self._cols), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, iteration: int) -> None:
        """Call once per on_step. Decays all grids then deposits new scent."""
        self._decay()
        self._deposit_threat()
        self._deposit_ally_trail()
        self._deposit_enemy_buildings()

    def sample_threat(self, pos: Point2, radius: float = 4.0) -> float:
        """Average threat scent within radius of pos."""
        return self._sample_grid(self.threat_scent, pos, radius)

    def sample_ally_trail(self, pos: Point2, radius: float = 4.0) -> float:
        return self._sample_grid(self.ally_trail, pos, radius)

    def hottest_threat_point(self) -> Point2 | None:
        """Return game-space Point2 of the highest threat scent cell."""
        idx = np.unravel_index(np.argmax(self.threat_scent), self.threat_scent.shape)
        if self.threat_scent[idx] < 0.1:
            return None
        return self._cell_to_point(idx[0], idx[1])

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _decay(self) -> None:
        self.threat_scent   *= self.cfg.threat_decay
        self.ally_trail     *= self.cfg.ally_decay
        self.enemy_building *= self.cfg.building_decay

    def _deposit_threat(self) -> None:
        for unit in self.bot.enemy_units:
            r, c = self._point_to_cell(unit.position)
            if self._in_bounds(r, c):
                self.threat_scent[r, c] = min(
                    self.cfg.max_scent,
                    self.threat_scent[r, c] + self.cfg.deposit_strength
                )

    def _deposit_ally_trail(self) -> None:
        army = self.bot.units.exclude_type({self.bot.worker_type, self.bot.supply_type})
        for unit in army:
            r, c = self._point_to_cell(unit.position)
            if self._in_bounds(r, c):
                self.ally_trail[r, c] = min(
                    self.cfg.max_scent,
                    self.ally_trail[r, c] + self.cfg.deposit_strength * 0.5
                )

    def _deposit_enemy_buildings(self) -> None:
        for struct in self.bot.enemy_structures:
            r, c = self._point_to_cell(struct.position)
            if self._in_bounds(r, c):
                self.enemy_building[r, c] = min(
                    self.cfg.max_scent,
                    self.enemy_building[r, c] + self.cfg.deposit_strength * 2.0
                )

    def _point_to_cell(self, pos: Point2) -> tuple[int, int]:
        c = int(pos.x / self.cfg.resolution)
        r = int(pos.y / self.cfg.resolution)
        return r, c

    def _cell_to_point(self, r: int, c: int) -> Point2:
        return Point2((c * self.cfg.resolution + self.cfg.resolution / 2,
                       r * self.cfg.resolution + self.cfg.resolution / 2))

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._rows and 0 <= c < self._cols

    def _sample_grid(self, grid: np.ndarray, pos: Point2, radius: float) -> float:
        cr, cc = self._point_to_cell(pos)
        cell_radius = max(1, int(radius / self.cfg.resolution))
        r0 = max(0, cr - cell_radius)
        r1 = min(self._rows, cr + cell_radius + 1)
        c0 = max(0, cc - cell_radius)
        c1 = min(self._cols, cc + cell_radius + 1)
        patch = grid[r0:r1, c0:c1]
        if patch.size == 0:
            return 0.0
        return float(patch.mean())