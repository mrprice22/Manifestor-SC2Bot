"""
Overlord Border Tactic — sentinel placement for early warning.

Assigns idle or misplaced overlords to watch-ring slots computed by
TerritoryBorderMap. Each slot sits just outside our creep/vision edge,
giving us a ring of sentinels that provides early warning of incoming
attacks from any direction.

Design decisions
----------------
- Only OVERLORD units are managed here (not OVERSEER — those have
  combat roles and are assigned by other tactics).
- An overlord is "misplaced" if it is not assigned to any slot OR if it
  has drifted more than slot_tolerance tiles from its assigned slot.
- We emit one movement idea per overlord per call — the suppression
  system in ManifestorBot handles cooldown naturally.
- Confidence is always high (≥ 0.70) because overlord positioning is
  pure infrastructure — we always want it done unless the overlord is
  doing something explicitly more important (which will outbid us).
- The tactic is never blocked by any strategy: early warning is always
  valuable regardless of whether we're turtling or all-in.

Suppression note
----------------
Overlord movement is cheap and idempotent, but we don't want to spam
move commands every frame. The base suppression system will handle this:
after a successful execute(), the unit gets a ~50-frame cooldown before
its next idea is evaluated. That's fine — overlords move slowly so a
re-check every 2 seconds is more than enough.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, FrozenSet

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit

from ares.behaviors.combat.individual import PathUnitToTarget
from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)

from ManifestorBot.manifests.tactics.base import TacticModule, TacticIdea

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy


class OverlordBorderTactic(TacticModule):
    """
    Move overlords to watch-ring sentinel positions around the territory edge.

    One overlord gets one slot assignment per evaluation cycle.
    Overlords already on-station (within slot_tolerance) generate no idea —
    they're doing their job.

    Confidence
    ----------
    Base confidence: 0.70 (high — this is always useful infrastructure).
    Boosted by:
      +0.10  if there are uncovered slots near known threat approach paths
             (pheromone scent > threshold)
      +0.05  per missing overlord on watch ring (up to +0.15), so we
             prioritise placement when coverage is thin
    Reduced by:
      -0.20  if the overlord is in danger (enemy units nearby) — let
             KeepUnitSafe handle it first
    """

    # Overlords closer than this to their assigned slot don't need to move
    ON_STATION_DISTANCE: float = 4.0

    # Don't send an overlord into a slot if an enemy is this close to it
    DANGER_RADIUS: float = 10.0

    # Base confidence — always high so this fires unless something more urgent wins
    BASE_CONFIDENCE: float = 0.70

    # Only fire for OVERLORD (not OVERSEER — different role)
    _OVERLORD_TYPES = {UnitID.OVERLORD, UnitID.OVERLORDTRANSPORT}

    @property
    def blocked_strategies(self) -> 'FrozenSet[Strategy]':
        # Never blocked — early warning is always valuable
        return frozenset()

    # ------------------------------------------------------------------ #
    # TacticModule interface
    # ------------------------------------------------------------------ #

    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        # Only overlords
        if unit.type_id not in self._OVERLORD_TYPES:
            return False

        # Need the border map to be initialised
        border_map = getattr(bot, 'territory_border_map', None)
        if border_map is None:
            return False

        # If already assigned to a slot and on-station, nothing to do
        assigned_slot = border_map._assignments.get(unit.tag)
        if assigned_slot is not None:
            if unit.distance_to(assigned_slot) <= self.ON_STATION_DISTANCE:
                return False  # Already there — no idea needed this cycle

        return True

    def generate_idea(
        self,
        unit: Unit,
        bot: 'ManifestorBot',
        heuristics: 'HeuristicState',
        current_strategy: 'Strategy',
    ) -> Optional[TacticIdea]:
        border_map = bot.territory_border_map
        confidence = self.BASE_CONFIDENCE
        evidence: dict = {}

        # --- sub-signal: danger near this overlord ---
        nearby_enemies = bot.enemy_units.closer_than(self.DANGER_RADIUS, unit.position)
        if nearby_enemies:
            # Let KeepUnitSafe handle it — lower our confidence so it wins
            sig = min(0.30, len(nearby_enemies) * 0.10)
            confidence -= sig
            evidence['overlord_in_danger'] = round(-sig, 3)

        if confidence < 0.40:
            # Too dangerous to reposition right now
            return None

        # --- find the best available slot for this overlord ---
        slot = self._pick_slot(unit, bot, border_map, heuristics)
        if slot is None:
            return None

        # --- sub-signal: slot is on a hot threat corridor ---
        pm = getattr(bot, 'pheromone_map', None)
        if pm is not None:
            threat_scent = pm.sample_threat(slot, radius=6.0)
            if threat_scent >= border_map.cfg.threat_scent_threshold:
                sig = min(0.10, threat_scent * 0.04)
                confidence += sig
                evidence['threat_corridor_slot'] = round(sig, 3)

        # --- sub-signal: overall coverage thinness ---
        uncovered_count = len(border_map.get_uncovered_slots())
        total_slots = max(1, len(border_map.watch_slots))
        coverage_ratio = 1.0 - (uncovered_count / total_slots)
        if coverage_ratio < 0.5:
            sig = min(0.15, (0.5 - coverage_ratio) * 0.30)
            confidence += sig
            evidence['thin_coverage'] = round(sig, 3)

        evidence['base'] = self.BASE_CONFIDENCE

        # Register the assignment so other overlords don't race for the same slot
        border_map.assign(unit.tag, slot)

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=slot,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: 'ManifestorBot',
    ) -> Optional[CombatIndividualBehavior]:
        slot = idea.target
        if slot is None:
            return None

        # Sanity check: if slot is no longer valid (watch ring changed),
        # release the assignment and return None gracefully
        border_map = getattr(bot, 'territory_border_map', None)
        if border_map is not None and slot not in border_map.watch_slots:
            border_map.release(unit.tag)
            return None

        # Use air grid since overlords fly
        try:
            air_grid = bot.mediator.get_air_grid
        except AttributeError:
            air_grid = bot.mediator.get_ground_grid  # fallback

        return PathUnitToTarget(
            unit=unit,
            grid=air_grid,
            target=slot,
            success_at_distance=self.ON_STATION_DISTANCE,
            sense_danger=True,
            danger_distance=self.DANGER_RADIUS,
        )

    # ------------------------------------------------------------------ #
    # Slot selection
    # ------------------------------------------------------------------ #

    def _pick_slot(
        self,
        unit: Unit,
        bot: 'ManifestorBot',
        border_map,
        heuristics: 'HeuristicState',
    ) -> Optional[Point2]:
        """
        Choose the best uncovered slot for this overlord.

        If the overlord already has an assignment and that slot still
        exists, keep it (stability > thrashing). Otherwise pick the
        nearest uncovered slot from the prioritised list.
        """
        # Honour existing assignment if it's still valid
        current = border_map._assignments.get(unit.tag)
        if current is not None and current in border_map.watch_slots:
            return current

        # Pick from the threat-prioritised uncovered list
        uncovered = border_map.get_uncovered_slots()
        if not uncovered:
            return None

        # Skip slots that have an enemy too close (would just die there)
        safe_slots = [
            s for s in uncovered
            if not bot.enemy_units.closer_than(self.DANGER_RADIUS, s)
        ]

        if not safe_slots:
            return None

        # Among safe slots, pick the closest to this overlord to minimise travel
        return min(safe_slots, key=lambda s: unit.distance_to(s))
