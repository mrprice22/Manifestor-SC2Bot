"""
ZerglingScouter -- manages two zergling scouting roles.

==========================================================================
ROLE 1: EXPANSION SCOUTS  (NUM_EXPANSION_SCOUTS zerglings, default 2)
==========================================================================
These scouts tour all expansion locations on the map (mineral lines /
possible bases), cycling through them in a priority order:

  Priority order per scout visit:
    1. Expansions that have NEVER been visited (highest priority)
    2. Expansions that haven't been visited recently (stalest first)
    3. Expansions near known enemy threat / building pheromone signal
       (re-check frequently -- something may have changed there)

Each scout is assigned its own current target expansion so the two scouts
naturally split up and cover different parts of the map.

Enemy avoidance:
  Before issuing a move command, the scout checks for enemy military units
  within FLEE_RADIUS tiles.  If any are found:
    - The scout is given a FLEE move away from the enemy centroid.
    - The current target expansion is re-queued (not skipped) so it will
      be revisited once the danger has passed.
    - A FLEE_COOLDOWN_FRAMES timer prevents the scout from immediately
      charging back toward the threat.

Stuck detection:
  If a scout has not moved STUCK_MIN_MOVE tiles in STUCK_CHECK_INTERVAL
  frames it is assumed stuck and given a new target.

==========================================================================
ROLE 2: SUICIDE SCOUT  (1 zergling at a time, timed)
==========================================================================
Every SCOUT_INTERVAL_FRAMES (~60 s) one zergling is sent directly into
the enemy main base then the enemy natural expansion.
  - If the scout dies before visually confirming the main base, a
    replacement is dispatched after RETRY_DELAY_FRAMES (~20 s).
  - Confirmation: scout within CONFIRM_RADIUS tiles of enemy start, OR
    at least one enemy structure visible near the start location.
  - After confirmation or death the timer resets.

==========================================================================
Call update() once per on_step from ManifestorBot.
Significant events are logged at INFO; per-frame detail at DEBUG.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2

from ManifestorBot.logger import get_logger

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot

log = get_logger()

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

NUM_EXPANSION_SCOUTS: int = 2
"""Number of zerglings dedicated to touring expansion locations."""

# --- expansion scout behaviour ---

ARRIVE_RADIUS: float = 5.0
"""Tiles from an expansion centre that counts as 'arrived / visited'."""

VISIT_COOLDOWN_FRAMES: int = 1_792
"""Frames before revisiting an already-seen expansion (~80 s). Keeps scouts
cycling rather than camping at one spot."""

FLEE_RADIUS: float = 12.0
"""Tiles within which an enemy military unit triggers a flee response."""

FLEE_DISTANCE: float = 20.0
"""How far (tiles) to move away from the threat centroid when fleeing."""

FLEE_COOLDOWN_FRAMES: int = 224
"""Frames a scout waits (stays away) after a flee before retargeting (~10 s)."""

RETARGET_INTERVAL: int = 44
"""Frames between routine retargeting checks (~2 s at 22.4 fps).
Lower = more responsive; higher = fewer redundant move commands."""

STUCK_CHECK_INTERVAL: int = 268
"""Frames between stuck-detection checks (~12 s)."""

STUCK_MIN_MOVE: float = 2.0
"""Tiles a scout must move between stuck checks or it is considered stuck."""

# --- suicide scout timing ---

SCOUT_INTERVAL_FRAMES: int = 1_344
"""Send one suicide scout every ~60 seconds (22.4 fps x 60)."""

RETRY_DELAY_FRAMES: int = 448
"""Retry delay when scout dies without confirming base (~20 s)."""

CONFIRM_RADIUS: float = 12.0
"""Distance to enemy start that counts as 'base confirmed'."""

# Non-combat unit types we ignore when checking for threats.
# (Workers are not threats; we don't want to flee from a drone.)
_NONCOMBAT_TYPES: frozenset = frozenset({
    UnitID.DRONE, UnitID.PROBE, UnitID.SCV,
    UnitID.OVERLORD, UnitID.OVERSEER,
    UnitID.LARVA, UnitID.EGG,
})


class _ExpansionRecord:
    """Tracks visit state for one expansion location."""

    def __init__(self, position: Point2) -> None:
        self.position: Point2 = position
        self.last_visited_frame: int = -VISIT_COOLDOWN_FRAMES  # treat as stale at start
        self.visit_count: int = 0

    def mark_visited(self, frame: int) -> None:
        self.last_visited_frame = frame
        self.visit_count += 1

    def staleness(self, frame: int) -> int:
        """Frames since last visit. Higher = more urgently needs re-scouting."""
        return frame - self.last_visited_frame

    def priority_score(self, frame: int, pheromone_signal: float) -> float:
        """
        Higher score = higher priority for the next scout visit.

        Unvisited expansions get a massive bonus so they are always done first.
        After that, staleness drives priority, boosted by any pheromone signal
        near the expansion (threat or building scent = something happening there).
        """
        base = float(self.staleness(frame))
        if self.visit_count == 0:
            base += 1_000_000.0     # ensure unvisited always beats revisit
        base += pheromone_signal * 500.0
        return base


class ZerglingScouter:
    """
    Manages expansion-touring pheromone scouts and timed suicide scouts.
    """

    def __init__(self, bot: 'ManifestorBot') -> None:
        self.bot = bot

        # --- expansion scout state ---
        self._pheromone_scout_tags: set[int] = set()
        # tag -> current target expansion record (or None if between targets / fleeing)
        self._scout_target: dict[int, Optional[_ExpansionRecord]] = {}
        # tag -> frame at which flee cooldown expires (0 = not fleeing)
        self._scout_flee_until: dict[int, int] = {}
        # tag -> (last_position, last_check_frame) for stuck detection
        self._scout_positions: dict[int, tuple[Point2, int]] = {}

        # All expansion locations on the map, including our own and enemy bases
        self._expansions: List[_ExpansionRecord] = []

        # --- suicide scout state ---
        self._suicide_scout_tag: Optional[int] = None
        self._next_scout_frame: int = SCOUT_INTERVAL_FRAMES
        self._base_confirmed: bool = False
        self._scout_dispatched_frame: int = 0

        self._enemy_main: Optional[Point2] = None
        self._enemy_natural: Optional[Point2] = None

        log.info("ZerglingScouter initialised")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def initialise_targets(self) -> None:
        """
        Resolve enemy locations and build the expansion list.
        Call from on_start after super().on_start() so Ares data is ready.
        """
        # --- enemy main ---
        try:
            enemy_locs = self.bot.enemy_start_locations
            if enemy_locs:
                self._enemy_main = enemy_locs[0]
                log.info("ZerglingScouter: enemy main -> %s", self._enemy_main)
            else:
                log.warning("ZerglingScouter: no enemy_start_locations -- suicide scout disabled")
        except Exception as exc:
            log.exception("ZerglingScouter.initialise_targets (enemy main): %s", exc)

        # --- enemy natural ---
        try:
            enemy_nat = self.bot.mediator.get_enemy_nat
            if enemy_nat is not None:
                self._enemy_natural = enemy_nat.position
                log.info("ZerglingScouter: enemy natural -> %s", self._enemy_natural)
            else:
                log.warning("ZerglingScouter: enemy natural unknown -- will skip natural waypoint")
        except Exception as exc:
            log.info("ZerglingScouter: could not resolve enemy natural (%s)", exc)

        # --- expansion list ---
        try:
            self._build_expansion_list()
        except Exception as exc:
            log.exception("ZerglingScouter.initialise_targets (expansions): %s", exc)

    def update(self, iteration: int) -> None:
        """Main tick -- call once per on_step."""
        frame = self.bot.state.game_loop

        try:
            self._prune_dead_tags(frame)
        except Exception as exc:
            log.exception("ZerglingScouter._prune_dead_tags: %s", exc)

        try:
            self._manage_expansion_scouts(frame)
        except Exception as exc:
            log.exception("ZerglingScouter._manage_expansion_scouts: %s", exc)

        try:
            self._manage_suicide_scout(frame)
        except Exception as exc:
            log.exception("ZerglingScouter._manage_suicide_scout: %s", exc)

    # -----------------------------------------------------------------------
    # Internal: expansion list
    # -----------------------------------------------------------------------

    def _build_expansion_list(self) -> None:
        """
        Collect all expansion positions from Ares / python-sc2 and sort them
        so that expansions closest to the enemy start are listed last (we'll
        visit our side first, then push toward the enemy).

        We exclude our own start location from the list -- no need to scout
        somewhere we already occupy.
        """
        try:
            # Ares exposes all expansion positions via the map data mediator
            raw_positions: List[Point2] = self.bot.mediator.get_own_expansions
            if not raw_positions:
                raise ValueError("get_own_expansions returned empty list")
            log.info(
                "ZerglingScouter: got %d expansion positions from mediator",
                len(raw_positions),
            )
        except Exception as exc:
            log.warning(
                "ZerglingScouter._build_expansion_list: mediator failed (%s) -- "
                "falling back to bot.expansion_locations",
                exc,
            )
            try:
                raw_positions = list(self.bot.expansion_locations.keys())
                log.info(
                    "ZerglingScouter: fallback gave %d expansion positions",
                    len(raw_positions),
                )
            except Exception as exc2:
                log.exception(
                    "ZerglingScouter._build_expansion_list: fallback also failed: %s", exc2
                )
                raw_positions = []

        if not raw_positions:
            log.error(
                "ZerglingScouter: no expansion positions found -- "
                "pheromone scouts will have nowhere to go"
            )
            return

        # Exclude our own start location (we already know it)
        own_start = self.bot.start_location
        filtered = [
            p for p in raw_positions
            if own_start.distance_to(p) > ARRIVE_RADIUS
        ]
        log.info(
            "ZerglingScouter: %d expansions after excluding own start",
            len(filtered),
        )

        # Sort: nearest to our base first so scouts visit friendly/neutral
        # expansions before venturing into enemy territory.
        filtered.sort(key=lambda p: own_start.distance_to(p))

        self._expansions = [_ExpansionRecord(pos) for pos in filtered]

        for i, exp in enumerate(self._expansions):
            log.info(
                "ZerglingScouter: expansion[%d] = %s (dist_from_home=%.1f)",
                i, exp.position, own_start.distance_to(exp.position),
            )

    # -----------------------------------------------------------------------
    # Internal: tag lifecycle
    # -----------------------------------------------------------------------

    def _prune_dead_tags(self, frame: int) -> None:
        """Remove any scout tags that no longer exist in the unit list."""
        all_tags = {u.tag for u in self.bot.units}

        dead_scouts = self._pheromone_scout_tags - all_tags
        if dead_scouts:
            log.info(
                "ZerglingScouter: expansion scout(s) died -- tags=%s (frame %d)",
                dead_scouts, frame,
            )
            for t in dead_scouts:
                self._pheromone_scout_tags.discard(t)
                self._scout_target.pop(t, None)
                self._scout_flee_until.pop(t, None)
                self._scout_positions.pop(t, None)

        if (
            self._suicide_scout_tag is not None
            and self._suicide_scout_tag not in all_tags
        ):
            log.info(
                "ZerglingScouter: suicide scout died -- tag=%d (frame %d, confirmed=%s)",
                self._suicide_scout_tag, frame, self._base_confirmed,
            )
            self._handle_suicide_scout_death(frame)

    def _handle_suicide_scout_death(self, frame: int) -> None:
        self._suicide_scout_tag = None
        if not self._base_confirmed:
            retry = frame + RETRY_DELAY_FRAMES
            log.info(
                "ZerglingScouter: unconfirmed death -- retry at frame %d (~%ds)",
                retry, int(RETRY_DELAY_FRAMES / 22.4),
            )
            self._next_scout_frame = retry
        else:
            nxt = frame + SCOUT_INTERVAL_FRAMES
            log.info("ZerglingScouter: confirmed run done -- next at frame %d", nxt)
            self._next_scout_frame = nxt
        self._base_confirmed = False

    # -----------------------------------------------------------------------
    # Internal: expansion scout management
    # -----------------------------------------------------------------------

    def _manage_expansion_scouts(self, frame: int) -> None:
        """
        Fill vacant scout slots, then for each scout:
          1. Check if fleeing (skip retarget if still in cooldown).
          2. Detect nearby threats and issue flee if needed.
          3. Check if arrived at current target.
          4. Check if stuck.
          5. Issue move to current (or new) target.
        """
        if frame % RETARGET_INTERVAL != 0:
            return

        if not self._expansions:
            log.debug(
                "ZerglingScouter._manage_expansion_scouts: no expansion list yet -- skipping"
            )
            return

        self._fill_expansion_scout_slots(frame)

        for tag in list(self._pheromone_scout_tags):
            unit = self._unit_by_tag(tag)
            if unit is None:
                log.warning(
                    "ZerglingScouter: expansion scout tag=%d not in unit list (not yet pruned)",
                    tag,
                )
                continue

            self._tick_expansion_scout(tag, unit, frame)

    def _tick_expansion_scout(self, tag: int, unit, frame: int) -> None:
        """Full per-scout logic for one expansion scout."""

        # --- threat check: are there enemy combatants nearby? ---
        threat_units = [
            e for e in self.bot.enemy_units
            if e.type_id not in _NONCOMBAT_TYPES
            and unit.distance_to(e.position) <= FLEE_RADIUS
        ]

        if threat_units:
            self._issue_flee(tag, unit, threat_units, frame)
            return  # don't retarget while actively fleeing

        # --- flee cooldown: still waiting after a recent flee? ---
        flee_until = self._scout_flee_until.get(tag, 0)
        if frame < flee_until:
            log.debug(
                "ZerglingScouter: scout tag=%d in flee cooldown until frame %d",
                tag, flee_until,
            )
            return

        # --- stuck detection (runs on STUCK_CHECK_INTERVAL cadence) ---
        if frame % STUCK_CHECK_INTERVAL == 0:
            if self._is_stuck(tag, unit, frame):
                # Force a new target by clearing the current one
                old_target = self._scout_target.get(tag)
                self._scout_target[tag] = None
                log.info(
                    "ZerglingScouter: scout tag=%d STUCK -- clearing target (was %s)",
                    tag,
                    old_target.position if old_target else "None",
                )

        # --- arrival check ---
        current_target = self._scout_target.get(tag)
        if current_target is not None:
            dist = unit.distance_to(current_target.position)
            if dist <= ARRIVE_RADIUS:
                current_target.mark_visited(frame)
                log.info(
                    "ZerglingScouter: scout tag=%d ARRIVED at %s "
                    "(visit #%d, frame %d)",
                    tag, current_target.position,
                    current_target.visit_count, frame,
                )
                self._scout_target[tag] = None
                current_target = None
            else:
                log.debug(
                    "ZerglingScouter: scout tag=%d en route to %s, dist=%.1f",
                    tag, current_target.position, dist,
                )

        # --- assign new target if needed ---
        if current_target is None:
            new_target = self._pick_next_expansion(tag, frame)
            if new_target is None:
                log.debug(
                    "ZerglingScouter: scout tag=%d has no suitable expansion target",
                    tag,
                )
                return
            self._scout_target[tag] = new_target
            log.info(
                "ZerglingScouter: scout tag=%d -> new target %s "
                "(staleness=%d, visits=%d)",
                tag, new_target.position,
                new_target.staleness(frame), new_target.visit_count,
            )
            unit.move(new_target.position)
            self._scout_positions[tag] = (unit.position, frame)
        else:
            # Reissue move in case the unit got bumped off its order
            unit.move(current_target.position)

    def _issue_flee(self, tag: int, unit, threat_units: list, frame: int) -> None:
        """Move away from the centroid of nearby threats."""
        cx = sum(e.position.x for e in threat_units) / len(threat_units)
        cy = sum(e.position.y for e in threat_units) / len(threat_units)
        threat_centre = Point2((cx, cy))

        dx = unit.position.x - threat_centre.x
        dy = unit.position.y - threat_centre.y
        dist = max(0.001, (dx ** 2 + dy ** 2) ** 0.5)
        flee_point = Point2((
            unit.position.x + (dx / dist) * FLEE_DISTANCE,
            unit.position.y + (dy / dist) * FLEE_DISTANCE,
        ))

        unit.move(flee_point)
        self._scout_flee_until[tag] = frame + FLEE_COOLDOWN_FRAMES

        log.info(
            "ZerglingScouter: scout tag=%d FLEEING from %d enemies "
            "(nearest=%.1f tiles) -> %s, cooldown until frame %d",
            tag, len(threat_units),
            min(unit.distance_to(e.position) for e in threat_units),
            flee_point,
            frame + FLEE_COOLDOWN_FRAMES,
        )

    def _pick_next_expansion(
        self, tag: int, frame: int
    ) -> Optional[_ExpansionRecord]:
        """
        Choose the highest-priority expansion for this scout to visit next.

        Priorities:
          1. Unvisited expansions (visit_count == 0) always win.
          2. Among visited, stalest wins.
          3. Pheromone signal near the expansion boosts priority.

        The other active scout's current target is excluded so the two scouts
        naturally split up and cover different locations.
        """
        pm = getattr(self.bot, 'pheromone_map', None)

        # Find what the other scout(s) are currently targeting to avoid overlap
        other_targets: set = set()
        for other_tag, target in self._scout_target.items():
            if other_tag != tag and target is not None:
                other_targets.add(id(target))

        best: Optional[_ExpansionRecord] = None
        best_score: float = -1.0

        for exp in self._expansions:
            if id(exp) in other_targets:
                continue  # another scout is already heading here

            # Pheromone signal near this expansion
            pheromone_signal = 0.0
            if pm is not None:
                try:
                    pheromone_signal = (
                        pm.sample_threat(exp.position, radius=8.0)
                        + pm.sample_ally_trail(exp.position, radius=8.0) * 0.0
                        # Only threat matters for reprioritising -- we don't
                        # want ally trail to send scouts where we already go.
                    )
                except Exception as exc:
                    log.debug(
                        "ZerglingScouter._pick_next_expansion: pheromone sample failed: %s", exc
                    )

            score = exp.priority_score(frame, pheromone_signal)
            log.debug(
                "ZerglingScouter._pick_next_expansion: tag=%d exp=%s "
                "score=%.1f (staleness=%d, visits=%d, pheromone=%.3f)",
                tag, exp.position, score,
                exp.staleness(frame), exp.visit_count, pheromone_signal,
            )

            if score > best_score:
                best_score = score
                best = exp

        if best is not None:
            log.debug(
                "ZerglingScouter._pick_next_expansion: tag=%d selected %s (score=%.1f)",
                tag, best.position, best_score,
            )
        else:
            log.warning(
                "ZerglingScouter._pick_next_expansion: tag=%d -- no expansion available",
                tag,
            )

        return best

    def _fill_expansion_scout_slots(self, frame: int) -> None:
        """Assign idle zerglings to fill vacant expansion scout slots."""
        vacancies = NUM_EXPANSION_SCOUTS - len(self._pheromone_scout_tags)
        if vacancies <= 0:
            return

        log.info(
            "ZerglingScouter: filling %d expansion scout slot(s) (frame %d)",
            vacancies, frame,
        )

        reserved = set(self._pheromone_scout_tags)
        if self._suicide_scout_tag is not None:
            reserved.add(self._suicide_scout_tag)

        candidates = [
            z for z in self.bot.units(UnitID.ZERGLING)
            if z.tag not in reserved
        ]

        if not candidates:
            log.warning(
                "ZerglingScouter: no free zerglings for expansion scouting "
                "(reserved=%d, total zerglings=%d)",
                len(reserved),
                len(self.bot.units(UnitID.ZERGLING)),
            )
            return

        candidates.sort(key=lambda u: (len(u.orders) > 0, random.random()))

        for unit in candidates[:vacancies]:
            self._pheromone_scout_tags.add(unit.tag)
            self._scout_target[unit.tag] = None
            self._scout_flee_until[unit.tag] = 0
            self._scout_positions[unit.tag] = (unit.position, frame)
            log.info(
                "ZerglingScouter: assigned zergling tag=%d as expansion scout at %s",
                unit.tag, unit.position,
            )

    def _is_stuck(self, tag: int, unit, frame: int) -> bool:
        """Return True if the scout hasn't moved enough since the last check."""
        prev = self._scout_positions.get(tag)
        if prev is None:
            self._scout_positions[tag] = (unit.position, frame)
            return False

        prev_pos, prev_frame = prev
        dist_moved = unit.position.distance_to(prev_pos)
        frames_elapsed = frame - prev_frame

        log.debug(
            "ZerglingScouter: stuck-check tag=%d moved=%.2f tiles over %d frames",
            tag, dist_moved, frames_elapsed,
        )

        self._scout_positions[tag] = (unit.position, frame)

        if dist_moved < STUCK_MIN_MOVE:
            log.warning(
                "ZerglingScouter: scout tag=%d STUCK -- "
                "only %.2f tiles in %d frames (min=%.1f)",
                tag, dist_moved, frames_elapsed, STUCK_MIN_MOVE,
            )
            return True

        return False

    # -----------------------------------------------------------------------
    # Internal: suicide scout management
    # -----------------------------------------------------------------------

    def _manage_suicide_scout(self, frame: int) -> None:
        if self._enemy_main is None:
            log.debug("ZerglingScouter._manage_suicide_scout: no enemy main -- skipping")
            return

        if self._suicide_scout_tag is not None:
            self._check_base_confirmation(frame)

        if frame < self._next_scout_frame:
            return

        if self._suicide_scout_tag is not None:
            log.debug(
                "ZerglingScouter: suicide scout already active (tag=%d)",
                self._suicide_scout_tag,
            )
            return

        self._dispatch_suicide_scout(frame)

    def _dispatch_suicide_scout(self, frame: int) -> None:
        reserved = set(self._pheromone_scout_tags)

        candidates = [
            z for z in self.bot.units(UnitID.ZERGLING)
            if z.tag not in reserved
        ]

        if not candidates:
            log.warning(
                "ZerglingScouter: no free zergling for suicide scout "
                "(pheromone scouts=%d, total zerglings=%d) -- delaying 112 frames",
                len(self._pheromone_scout_tags),
                len(self.bot.units(UnitID.ZERGLING)),
            )
            self._next_scout_frame = frame + 112
            return

        scout = min(candidates, key=lambda u: u.distance_to(self._enemy_main))
        self._suicide_scout_tag = scout.tag
        self._base_confirmed = False
        self._scout_dispatched_frame = frame

        log.info(
            "ZerglingScouter: dispatching suicide scout tag=%d -> %s "
            "(frame %d, ~%ds into game)",
            scout.tag, self._enemy_main, frame, int(frame / 22.4),
        )

        scout.move(self._enemy_main)

        if self._enemy_natural is not None:
            log.info(
                "ZerglingScouter: suicide scout tag=%d queuing natural=%s",
                scout.tag, self._enemy_natural,
            )
            scout.move(self._enemy_natural, queue=True)

    def _check_base_confirmation(self, frame: int) -> None:
        if self._base_confirmed:
            return

        scout = self._unit_by_tag(self._suicide_scout_tag)
        if scout is None:
            return

        dist_to_main = scout.distance_to(self._enemy_main)
        nearby_structures = self.bot.enemy_structures.closer_than(
            CONFIRM_RADIUS * 1.5, self._enemy_main
        )

        if dist_to_main <= CONFIRM_RADIUS or nearby_structures:
            self._base_confirmed = True
            log.info(
                "ZerglingScouter: base CONFIRMED -- tag=%d dist=%.1f structures=%d (frame %d)",
                self._suicide_scout_tag, dist_to_main, len(nearby_structures), frame,
            )
        else:
            log.debug(
                "ZerglingScouter: suicide scout tag=%d en route -- dist=%.1f structures=%d",
                self._suicide_scout_tag, dist_to_main, len(nearby_structures),
            )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _unit_by_tag(self, tag: int) -> Optional[object]:
        try:
            return self.bot.units.find_by_tag(tag)
        except Exception as exc:
            log.exception("ZerglingScouter._unit_by_tag(%d): %s", tag, exc)
            return None

    # -----------------------------------------------------------------------
    # Public query
    # -----------------------------------------------------------------------

    def is_scouting_tag(self, tag: int) -> bool:
        """True if this tag is a pheromone or suicide scout (exclude from tactic loop)."""
        return tag in self._pheromone_scout_tags or tag == self._suicide_scout_tag

    def summary(self) -> str:
        targets = {
            tag: (rec.position if rec else "none")
            for tag, rec in self._scout_target.items()
        }
        visited = sum(1 for e in self._expansions if e.visit_count > 0)
        return (
            f"ZerglingScouter("
            f"expansion_scouts={self._pheromone_scout_tags}, "
            f"targets={targets}, "
            f"expansions={visited}/{len(self._expansions)} visited, "
            f"suicide={self._suicide_scout_tag}, "
            f"confirmed={self._base_confirmed}, "
            f"next_suicide_frame={self._next_scout_frame})"
        )