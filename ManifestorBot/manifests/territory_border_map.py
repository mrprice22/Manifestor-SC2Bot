"""
Territory Border Map — vision coverage grid and watch-ring computation.

Maintains a 2D boolean grid of tiles the bot currently has vision of,
then computes a "watch ring" — the set of candidate positions just
*outside* that coverage. These are the slots an overlord needs to occupy
to give early warning of incoming attacks.

Design notes
------------
- Grid resolution matches SC2's vision grid (1 game-unit per cell).
  This avoids any floating-point approximation when querying the
  python-sc2 visibility map.
- Watch ring positions are recomputed every N frames (not every step)
  because the topology changes slowly — only when creep spreads, a new
  base completes, or an overlord moves.
- Pheromone integration (optional): if a PheromoneMap is available on
  the bot, candidate slots are sorted by local threat scent so the most
  dangerous approach corridors get covered first.
- Slot assignment is persistent: once an overlord is assigned a slot it
  keeps the assignment until it drifts more than SLOT_TOLERANCE tiles
  away or the slot is no longer on the watch ring.

Public API
----------
    border_map.update(iteration)
    slots = border_map.get_uncovered_slots()   -> list[Point2]
    border_map.assign(overlord_tag, slot)
    border_map.release(overlord_tag)
    covered  = border_map.is_covered(slot)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from ManifestorBot.logger import get_logger
log = get_logger()

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BorderConfig:
    """Tunable constants for TerritoryBorderMap."""

    # How many game-units outside the vision edge we want overlords to sit.
    # Too close = they die to the attack they're spotting.
    # Too far   = gap between creep/vision and the sentinel.
    watch_ring_inner: float = 5.0   # min distance outside vision edge
    watch_ring_outer: float = 12.0  # max distance (don't go too deep into fog)

    # Minimum spacing between candidate slots (avoids clustering).
    slot_spacing: float = 8.0

    # How often to recompute the full watch ring (frames).
    # Recomputing every frame is wasteful — the border moves slowly.
    recompute_interval: int = 112   # ~5 seconds at normal speed

    # An assigned overlord is considered "on slot" if within this distance.
    slot_tolerance: float = 4.0

    # Maximum overlords we'll try to place (keeps list manageable).
    max_slots: int = 12

    # Minimum threat scent to prioritise a slot (pheromone integration).
    # Slots below this are still used but sorted to the back.
    threat_scent_threshold: float = 0.5

    # How many overlords can be assigned to vision-edge scouting
    # (the rest go to creep-edge sentinel positions).
    max_scout_slots: int = 2


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TerritoryBorderMap:
    """
    Tracks the edge of our vision/creep coverage and emits candidate
    watch-ring positions for overlord placement.

    Lifecycle
    ---------
    Instantiate in ManifestorBot.__init__ (as None) and create in on_start
    once game_info is available. Call update() once per on_step.
    """

    def __init__(self, bot: 'ManifestorBot', config: Optional[BorderConfig] = None):
        self.bot = bot
        self.cfg = config or BorderConfig()

        # Map dimensions (game-units, integer)
        self._map_w: int = bot.game_info.pathing_grid.width
        self._map_h: int = bot.game_info.pathing_grid.height

        # Cached creep-edge watch-ring slots (Point2 list, game-space)
        self._watch_slots: list[Point2] = []

        # Vision-edge scout slots (capped at max_scout_slots)
        self._scout_slots: list[Point2] = []

        # Persistent overlord → slot assignments {unit_tag: Point2}
        self._assignments: dict[int, Point2] = {}

        # Scout overlord → slot assignments
        self._scout_assignments: dict[int, Point2] = {}

        # Frame of last full recompute
        self._last_recompute: int = -9999

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, iteration: int) -> None:
        """
        Call once per on_step. Recomputes the watch ring on the configured
        interval and validates existing overlord assignments.
        """
        frame = self.bot.state.game_loop

        if frame - self._last_recompute >= self.cfg.recompute_interval:
            self._recompute_watch_ring()
            self._last_recompute = frame

        self._validate_assignments()

    def get_uncovered_slots(self) -> list[Point2]:
        """
        Return watch-ring slots that have no overlord currently assigned
        and on-station.

        Slots are ordered: pheromone-hot slots first (if pheromone map is
        available), then by distance from map centre (perimeter-first).
        """
        assigned_slots = set(id(s) for s in self._assignments.values())

        # Build set of slot positions already covered
        covered_positions = {
            slot for slot in self._watch_slots
            if self._slot_is_covered(slot)
        }

        uncovered = [s for s in self._watch_slots if s not in covered_positions]
        return self._prioritise_slots(uncovered)

    def assign(self, overlord_tag: int, slot: Point2) -> None:
        """Record that this overlord has been sent to this slot."""
        self._assignments[overlord_tag] = slot

    def release(self, overlord_tag: int) -> None:
        """Remove the assignment for this overlord (called when it dies)."""
        self._assignments.pop(overlord_tag, None)
        self._scout_assignments.pop(overlord_tag, None)

    # --- Scout slot API ---

    def get_uncovered_scout_slots(self) -> list[Point2]:
        """Return scout slots that have no overlord currently assigned."""
        covered_positions = {
            slot for slot in self._scout_slots
            if self._slot_is_covered_in(slot, self._scout_assignments)
        }
        uncovered = [s for s in self._scout_slots if s not in covered_positions]
        return self._prioritise_slots(uncovered)

    def assign_scout(self, overlord_tag: int, slot: Point2) -> None:
        """Record that this overlord has been sent to a scout slot."""
        self._scout_assignments[overlord_tag] = slot

    def release_scout(self, overlord_tag: int) -> None:
        """Remove the scout assignment for this overlord."""
        self._scout_assignments.pop(overlord_tag, None)

    @property
    def scout_slots(self) -> list[Point2]:
        """All current vision-edge scout positions."""
        return list(self._scout_slots)

    @property
    def scout_assignment_count(self) -> int:
        """How many overlords currently have a scout assignment."""
        return len(self._scout_assignments)

    def is_covered(self, slot: Point2) -> bool:
        """Return True if an overlord is on-station at this slot."""
        return self._slot_is_covered(slot)

    @property
    def watch_slots(self) -> list[Point2]:
        """All current watch-ring candidate positions."""
        return list(self._watch_slots)

    @property
    def assignment_count(self) -> int:
        """How many overlords currently have a slot assignment."""
        return len(self._assignments)

    # ------------------------------------------------------------------ #
    # Watch ring computation
    # ------------------------------------------------------------------ #

    def _recompute_watch_ring(self) -> None:
        """
        Full recompute — produces two slot pools:
        1. _watch_slots (creep-edge sentinels): ring around creep boundary.
        2. _scout_slots (vision-edge scouts): ring at vision/unexplored
           boundary, capped at max_scout_slots.

        Fallback: if creep-edge yields nothing (game start), all slots
        come from the unexplored-edge fallback.
        """
        try:
            # --- Creep-edge sentinel slots ---
            creep_mask = self._build_creep_only_mask()
            creep_slots: list[Point2] = []
            if creep_mask is not None and creep_mask.any():
                ring_cells = self._find_ring_cells(creep_mask)
                creep_slots = self._cells_to_points(ring_cells)
                creep_slots = self._thin_slots(creep_slots)

            # Fallback when no creep ring (early game)
            if not creep_slots:
                creep_slots = self._fallback_unexplored_edge_slots()

            self._watch_slots = creep_slots

            # --- Vision-edge scout slots ---
            vision_mask = self._build_coverage_mask()
            scout_slots: list[Point2] = []
            if vision_mask is not None:
                ring_cells = self._find_ring_cells(vision_mask)
                scout_pts = self._cells_to_points(ring_cells)
                scout_pts = self._thin_slots(scout_pts)
                # Remove any that overlap with creep-edge slots
                tol = self.cfg.slot_tolerance
                scout_slots = [
                    s for s in scout_pts
                    if all(s.distance_to(ws) > tol for ws in self._watch_slots)
                ]

            # Cap to max_scout_slots
            self._scout_slots = scout_slots[: self.cfg.max_scout_slots]

            log.info(
                "Watch ring recomputed: %d creep-edge slots, %d scout slots",
                len(self._watch_slots), len(self._scout_slots),
            )

        except Exception as e:
            log.error("_recompute_watch_ring: failed: %s", e)
            

    def _fallback_unexplored_edge_slots(self) -> list[Point2]:
        """
        When creep-edge computation yields nothing (e.g. game start, no creep
        spread yet), place overlords at the boundary between explored and
        unexplored areas of the map.
        
        Uses visibility: cells with value 0 = never seen.
        The boundary = explored cells (value >= 1) adjacent to unexplored (0).
        """
        try:
            vis = self.bot.state.visibility
            h, w = self._map_h, self._map_w
            raw = np.frombuffer(vis.data, dtype=np.uint8).reshape(h, w)
        except Exception:
            return []

        explored = raw >= 1   # seen at any point (value 1 or 2)
        unexplored = raw == 0

        # Boundary = unexplored cells adjacent to explored cells
        # Dilate explored by 1 step, intersect with unexplored
        from numpy import roll, zeros_like
        dilated = (
            roll(explored, 1, axis=0) | roll(explored, -1, axis=0) |
            roll(explored, 1, axis=1) | roll(explored, -1, axis=1) |
            explored
        )
        boundary_mask = dilated & unexplored

        ys, xs = np.where(boundary_mask)
        if len(ys) == 0:
            # Total fallback: just distribute across map edges
            return self._map_edge_slots()

        cells = list(zip(ys.tolist(), xs.tolist()))
        slots = self._cells_to_points(cells)
        return self._thin_slots(slots)

    def _map_edge_slots(self) -> list[Point2]:
        """Last-resort: evenly spaced positions around the map perimeter."""
        w, h = float(self._map_w), float(self._map_h)
        margin = 8.0
        candidates = []
        spacing = self.cfg.slot_spacing * 2
        x = margin
        while x < w - margin:
            candidates.append(Point2((x, margin)))
            candidates.append(Point2((x, h - margin)))
            x += spacing
        y = margin
        while y < h - margin:
            candidates.append(Point2((margin, y)))
            candidates.append(Point2((w - margin, y)))
            y += spacing
        return self._thin_slots(candidates)

    def _build_creep_only_mask(self) -> Optional[np.ndarray]:
        """
        Build a boolean 2D array (map_h x map_w) where True = creep tile.
        Does NOT include vision — produces a ring that hugs the creep boundary.
        """
        h, w = self._map_h, self._map_w
        try:
            creep = self.bot.state.creep
            try:
                creep_raw = np.frombuffer(creep.data, dtype=np.uint8).reshape(h, w)
                return creep_raw.astype(bool)
            except Exception:
                mask = np.zeros((h, w), dtype=bool)
                for y in range(h):
                    for x in range(w):
                        if creep[x, y]:
                            mask[y, x] = True
                return mask
        except AttributeError:
            return None

    def _build_coverage_mask(self) -> Optional[np.ndarray]:
        """
        Build a boolean 2D array (map_h × map_w) where True = covered.
        Uses numpy vectorisation instead of Python loops.
        """
        try:
            vis = self.bot.state.visibility
        except AttributeError:
            print("_build_coverage_mask: visibility is None, returning early", flush=True)
            print(f"bot.state type: {type(self.bot.state)}", flush=True)
            print(f"bot.state: {self.bot.state}", flush=True)
            return None

        h, w = self._map_h, self._map_w

        # python-sc2 PixelMap: access the raw data as a numpy array
        # .data_numpy gives a (h, w) uint8 array where 2 = currently visible
        try:
            raw = np.frombuffer(vis.data, dtype=np.uint8).reshape(h, w)
            mask = raw == 2
        except Exception:
            # Fallback: slower but correct
            mask = np.zeros((h, w), dtype=bool)
            for y in range(h):
                for x in range(w):
                    if vis[x, y] == 2:
                        mask[y, x] = True

        # Also mark creep tiles as covered
        try:
            creep = self.bot.state.creep
            try:
                creep_raw = np.frombuffer(creep.data, dtype=np.uint8).reshape(h, w)
                mask |= creep_raw.astype(bool)
            except Exception:
                for y in range(h):
                    for x in range(w):
                        if creep[x, y]:
                            mask[y, x] = True
        except AttributeError:
            pass

        return mask

    def _find_ring_cells(self, coverage: np.ndarray) -> list[tuple[int, int]]:
        """
        Find cells that are:
          - NOT currently covered (outside our vision/creep)
          - Adjacent (within watch_ring_outer cells) to a covered cell
          - At least watch_ring_inner cells from any covered cell

        Uses binary dilation/erosion from scipy if available, falls back
        to a pure-numpy approach.
        """
        inner_r = max(1, int(self.cfg.watch_ring_inner))
        outer_r = max(inner_r + 1, int(self.cfg.watch_ring_outer))

        try:
            from scipy.ndimage import binary_dilation
            struct_inner = _disk_kernel(inner_r)
            struct_outer = _disk_kernel(outer_r)
            inner_zone = binary_dilation(coverage, structure=struct_inner)
            outer_zone = binary_dilation(coverage, structure=struct_outer)
            # Ring = tiles in outer dilation but NOT in inner dilation and NOT covered
            ring_mask = outer_zone & ~inner_zone & ~coverage
        except ImportError:
            # Fallback: simple box dilation
            from numpy.lib.stride_tricks import as_strided
            inner_zone = _box_dilate(coverage, inner_r)
            outer_zone = _box_dilate(coverage, outer_r)
            ring_mask = outer_zone & ~inner_zone & ~coverage

        ys, xs = np.where(ring_mask)
        return list(zip(ys.tolist(), xs.tolist()))

    def _cells_to_points(self, cells: list[tuple[int, int]]) -> list[Point2]:
        """Convert (row, col) cell indices to game-space Point2."""
        # In SC2, cell (col, row) maps to game point (col + 0.5, row + 0.5)
        return [Point2((float(c) + 0.5, float(r) + 0.5)) for r, c in cells]

    def _thin_slots(self, slots: list[Point2]) -> list[Point2]:
        """
        Reduce the candidate list to at most max_slots well-spaced points.

        Uses a simple greedy farthest-point selection to ensure coverage
        is spread around the perimeter rather than clustered.
        Filters out behind-the-base slots and prioritises forward-facing ones.
        """
        if not slots:
            return []

        spacing = self.cfg.slot_spacing
        start = self.bot.start_location
        enemy_start = self.bot.enemy_start_locations[0] if self.bot.enemy_start_locations else start
        base_to_enemy_dist = start.distance_to(enemy_start)

        # Filter out slots behind our base (further from enemy than we are + margin)
        margin = 10.0
        forward_slots = [
            p for p in slots
            if p.distance_to(enemy_start) <= base_to_enemy_dist + margin
        ]
        # Fall back to all slots if filtering removes everything
        if not forward_slots:
            forward_slots = slots

        # Sort by distance to enemy (closest = most forward = highest priority)
        slots_sorted = sorted(forward_slots, key=lambda p: p.distance_to(enemy_start))

        selected: list[Point2] = []
        for candidate in slots_sorted:
            if len(selected) >= self.cfg.max_slots:
                break
            if all(candidate.distance_to(s) >= spacing for s in selected):
                selected.append(candidate)
        return selected

    # ------------------------------------------------------------------ #
    # Assignment management
    # ------------------------------------------------------------------ #

    def _validate_assignments(self) -> None:
        """
        Remove assignments where:
        - The overlord no longer exists (died or morphed)
        - The slot it was assigned is no longer near any watch ring slot

        Handles both watch (creep-edge) and scout (vision-edge) pools.
        """
        live_tags = {u.tag for u in self.bot.units(
            {UnitID.OVERLORD, UnitID.OVERLORDTRANSPORT, UnitID.OVERSEER}
        )}
        self._validate_pool(self._assignments, self._watch_slots, live_tags)
        self._validate_pool(self._scout_assignments, self._scout_slots, live_tags)

    def _validate_pool(
        self,
        assignments: dict[int, Point2],
        slot_list: list[Point2],
        live_tags: set[int],
    ) -> None:
        """Validate one assignment pool against its slot list."""
        tol = self.cfg.slot_tolerance
        stale = []
        for tag, old_slot in assignments.items():
            if tag not in live_tags:
                stale.append(tag)
                continue
            nearest_new = None
            nearest_dist = float('inf')
            for new_slot in slot_list:
                d = old_slot.distance_to(new_slot)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_new = new_slot
            if nearest_new is not None and nearest_dist <= tol:
                assignments[tag] = nearest_new
            else:
                stale.append(tag)
        for tag in stale:
            assignments.pop(tag, None)

    def _slot_is_covered(self, slot: Point2) -> bool:
        """Check coverage across both watch and scout assignment pools."""
        return (
            self._slot_is_covered_in(slot, self._assignments) or
            self._slot_is_covered_in(slot, self._scout_assignments)
        )

    def _slot_is_covered_in(self, slot: Point2, assignments: dict[int, Point2]) -> bool:
        """
        Return True if there is an overlord assigned to this slot
        (within slot_tolerance distance) in the given assignment dict,
        or physically on-station.
        """
        tol = self.cfg.slot_tolerance
        for tag, assigned_slot in assignments.items():
            if assigned_slot.distance_to(slot) <= tol:
                return True
        overlords = self.bot.units({UnitID.OVERLORD, UnitID.OVERLORDTRANSPORT, UnitID.OVERSEER})
        return any(ol.distance_to(slot) <= tol for ol in overlords)

    # ------------------------------------------------------------------ #
    # Slot prioritisation
    # ------------------------------------------------------------------ #

    def _prioritise_slots(self, slots: list[Point2]) -> list[Point2]:
        """
        Sort slots so the most dangerous approaches come first.

        Priority order:
          1. Slots near a known attack path (pheromone threat scent, if available)
          2. Slots nearest to our main base (protect the most critical approach)
          3. Remainder in arbitrary order
        """
        pm = getattr(self.bot, 'pheromone_map', None)

        def slot_key(slot: Point2) -> tuple[float, float]:
            threat = 0.0
            if pm is not None:
                threat = pm.sample_threat(slot, radius=6.0)
            dist_to_base = slot.distance_to(self.bot.start_location)
            # Negate threat so high threat = low sort key = first
            return (-threat, dist_to_base)

        return sorted(slots, key=slot_key)

    # ------------------------------------------------------------------ #
    # Debug / summary
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        watch_covered = sum(1 for s in self._watch_slots if self._slot_is_covered(s))
        scout_covered = sum(1 for s in self._scout_slots if self._slot_is_covered(s))
        return (
            f"TerritoryBorderMap: {len(self._watch_slots)} creep-edge ({watch_covered} covered, "
            f"{len(self._assignments)} assigned) | "
            f"{len(self._scout_slots)} scout ({scout_covered} covered, "
            f"{len(self._scout_assignments)} assigned)"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _disk_kernel(radius: int) -> np.ndarray:
    """Create a boolean disk structuring element of the given radius."""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2) <= radius ** 2


def _box_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Simple numpy box dilation (no scipy dependency)."""
    from numpy import pad, zeros_like
    result = zeros_like(mask)
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            result |= shifted
    return result
