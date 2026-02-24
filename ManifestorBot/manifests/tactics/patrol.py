"""
OpportunisticPatrolTactic — give idle army units low-priority patrol orders
toward safe, unexplored areas for map control and vision.

Fires when:
  - Ground combat unit (no workers, supply, overlords, overseers, queens)
  - No enemies within 15 tiles
  - Not under DRONE_ONLY_FORTRESS strategy

Confidence: 0.42–0.52 (just above suppression threshold of 0.40,
always loses to real combat/positioning tactics that have a signal).

Note: the original design targeted 0.18–0.28, but the dispatch pipeline
suppresses any idea below 0.40 (_should_suppress_idea), so the base was
raised to 0.42 to ensure patrol orders actually execute.

Blocked strategies: DRONE_ONLY_FORTRESS
"""

from typing import Optional, FrozenSet, TYPE_CHECKING

from ares.behaviors.combat.individual import AMove
from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit

from ManifestorBot.manifests.tactics.base import TacticModule, TacticIdea

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy


# Unit types excluded from patrol duty (beyond the worker/supply gate)
_EXCLUDED_TYPES = frozenset({
    UnitID.DRONE,
    UnitID.OVERLORD,
    UnitID.OVERLORDTRANSPORT,
    UnitID.OVERSEER,
    UnitID.OVERSEERSIEGEMODE,
    UnitID.QUEEN,
})


class OpportunisticPatrolTactic(TacticModule):
    """Low-priority patrol orders for idle army units."""

    ENEMY_PROXIMITY_GATE: float = 15.0
    COHESION_LEASH: float = 25.0

    @property
    def blocked_strategies(self) -> "FrozenSet[Strategy]":
        from ManifestorBot.manifests.strategy import Strategy
        return frozenset({Strategy.DRONE_ONLY_FORTRESS})

    # ------------------------------------------------------------------ #
    # Structural gate
    # ------------------------------------------------------------------ #
    def is_applicable(self, unit: Unit, bot: "ManifestorBot") -> bool:
        if self._is_strategy_blocked(bot.current_strategy):
            return False
        if self._is_worker_or_supply(unit, bot):
            return False
        if unit.type_id in _EXCLUDED_TYPES:
            return False
        # Let real combat tactics handle units near enemies
        if bot.enemy_units.closer_than(self.ENEMY_PROXIMITY_GATE, unit.position):
            return False
        return True

    # ------------------------------------------------------------------ #
    # Confidence scoring
    # ------------------------------------------------------------------ #
    def generate_idea(
        self,
        unit: Unit,
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        profile = current_strategy.profile()
        confidence = 0.42
        evidence = {}

        # --- sub-signal: idle at home ---
        if unit.distance_to(bot.start_location) < 30.0:
            sig = 0.05
            confidence += sig
            evidence["idle_at_home"] = sig

        # --- sub-signal: positive momentum (we're winning, safe to push) ---
        if heuristics.momentum > 0:
            sig = 0.03
            confidence += sig
            evidence["positive_momentum"] = sig

        # --- sub-signal: harass strategy bias ---
        harass_sig = profile.harass_bias * 0.10
        if harass_sig != 0:
            confidence += harass_sig
            evidence["harass_bias"] = round(harass_sig, 3)

        # --- sub-signal: threat penalty (stay home when danger rises) ---
        if heuristics.threat_level > 0.3:
            sig = -0.05
            confidence += sig
            evidence["threat_penalty"] = sig

        confidence = max(0.0, min(0.55, confidence))
        if confidence < 0.40:
            return None

        target = self._pick_patrol_point(unit, bot)
        if target is None:
            return None

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=target,
        )

    # ------------------------------------------------------------------ #
    # Patrol-point selection
    # ------------------------------------------------------------------ #
    def _pick_patrol_point(
        self, unit: Unit, bot: "ManifestorBot"
    ) -> Optional[Point2]:
        army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
        centroid = army.center if army else bot.start_location
        enemy_base = bot.enemy_start_locations[0]
        map_center = bot.game_info.map_center

        best: Optional[Point2] = None

        # 1. Pheromone-guided: low threat + low ally trail, biased forward
        pm = getattr(bot, "pheromone_map", None)
        if pm is not None:
            best = self._pheromone_patrol_point(pm, centroid, enemy_base, map_center)

        # 2. Unclaimed Xel'Naga watchtowers
        if best is None:
            best = self._nearest_unclaimed_watchtower(unit, bot)

        # 3. Forward fallback: drift toward enemy half of the map
        if best is None:
            best = centroid.towards(
                enemy_base,
                centroid.distance_to(map_center) * 0.5,
            )

        # Cohesion clamp — keep units within leash distance of the army
        if best is not None and best.distance_to(centroid) > self.COHESION_LEASH:
            best = centroid.towards(best, self.COHESION_LEASH)

        return best

    def _pheromone_patrol_point(
        self,
        pm,
        centroid: Point2,
        enemy_base: Point2,
        map_center: Point2,
    ) -> Optional[Point2]:
        """Pick the candidate point with lowest combined threat + ally scent."""
        candidates = []
        dist_to_enemy = centroid.distance_to(enemy_base)
        for frac in (0.3, 0.5, 0.7):
            candidates.append(centroid.towards(enemy_base, dist_to_enemy * frac))
        # Also try toward map center
        candidates.append(
            centroid.towards(map_center, centroid.distance_to(map_center) * 0.6)
        )

        best_pt: Optional[Point2] = None
        best_score = float("inf")
        for pt in candidates:
            try:
                threat = pm.sample_threat(pt, radius=8.0)
                ally = pm.sample_ally_trail(pt, radius=8.0)
                score = threat * 2.0 + ally
            except Exception:
                continue
            if score < best_score:
                best_score = score
                best_pt = pt

        if best_pt is not None and best_score < 3.0:
            return best_pt
        return None

    def _nearest_unclaimed_watchtower(
        self, unit: Unit, bot: "ManifestorBot"
    ) -> Optional[Point2]:
        watchtowers = getattr(bot, "watchtowers", None)
        if not watchtowers:
            return None
        unclaimed = [
            wt for wt in watchtowers
            if not bot.units.closer_than(4.0, wt.position)
        ]
        if not unclaimed:
            return None
        nearest = min(unclaimed, key=lambda wt: unit.distance_to(wt.position))
        return nearest.position

    # ------------------------------------------------------------------ #
    # Behavior
    # ------------------------------------------------------------------ #
    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: "ManifestorBot",
    ) -> Optional[CombatIndividualBehavior]:
        target = idea.target
        if target is None:
            return None
        return AMove(unit=unit, target=target, success_at_distance=6.0)
