"""
CancelDyingBuildingTactic — cancel buildings that are about to die mid-construction.

When a building is under construction (build_progress < 1.0) and is critically
wounded with enemy units nearby, cancelling it returns the FULL construction cost
plus restores the drone for Zerg.  This is almost always better than letting the
enemy destroy it.

SC2 refund rules:
  - Zerg:   cancelling a building in progress returns 100% of the mineral/gas cost
            AND restores the drone that was morphed into the structure.
  - Terran: 100% refund on all buildings during construction.
  - Protoss: 100% refund during construction (Nexus/gateway/etc.).

Cancellation fires when:
  (a) health_ratio < CRITICAL_HEALTH (0.30)  — unconditional, enemy or not
  (b) health_ratio < DANGER_HEALTH  (0.55)  AND enemy combat units are within
      THREAT_RADIUS (15 tiles) — likely being attacked

Confidence: 0.70–0.95 (high — saving full build cost is nearly always correct).

Registered in _load_building_modules() BEFORE all production modules so it
pre-empts any training or rally decision on a building that is about to die.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, FrozenSet

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID

from ManifestorBot.manifests.tactics.building_base import (
    BuildingTacticModule,
    BuildingAction,
    BuildingIdea,
)
from ManifestorBot.logger import get_logger

log = get_logger()

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy
    from sc2.unit import Unit


# ── constants ──────────────────────────────────────────────────────────────

# Health ratio below which we cancel with no further conditions
CRITICAL_HEALTH: float = 0.30

# Health ratio below which we cancel if enemy combatants are within THREAT_RADIUS
DANGER_HEALTH: float = 0.55

# Radius to search for enemy units threatening the building
THREAT_RADIUS: float = 15.0

# Unit types that count as non-threatening scouts (workers, overlords, etc.)
_NON_COMBATANT_TYPES: frozenset = frozenset({
    UnitID.OVERLORD, UnitID.OVERLORDTRANSPORT, UnitID.OVERSEER,
    UnitID.OVERSEERSIEGEMODE,
    UnitID.SCV, UnitID.PROBE, UnitID.DRONE,
})


# ── tactic ─────────────────────────────────────────────────────────────────

class CancelDyingBuildingTactic(BuildingTacticModule):
    """
    Cancel buildings under construction that are critically wounded.

    This module applies to every building type (BUILDING_TYPES is empty,
    meaning all structures are eligible).  The fast gate in is_applicable()
    filters to only buildings that are:
      - still under construction (build_progress < 1.0)
      - already critically wounded, OR in the danger zone with enemies nearby

    A winning CANCEL idea in the pipeline causes execute() to issue
    AbilityId.CANCEL_BUILDINPROGRESS, which gives a full resource refund.
    """

    # Empty frozenset → applies to every structure type
    BUILDING_TYPES: FrozenSet[UnitID] = frozenset()

    @property
    def blocked_strategies(self) -> "FrozenSet[Strategy]":
        return frozenset()  # cancelling dying buildings is never blocked

    # ------------------------------------------------------------------ #
    # Structural gate
    # ------------------------------------------------------------------ #

    def is_applicable(self, building: "Unit", bot: "ManifestorBot") -> bool:
        # Only target buildings still under construction
        if building.build_progress >= 1.0:
            return False
        if building.health_max <= 0:
            return False

        health_ratio = building.health / building.health_max

        # Unconditional cancel when critically low — building is almost dead
        if health_ratio < CRITICAL_HEALTH:
            return True

        # Cancel in danger zone only when enemy combatants are confirmed nearby
        if health_ratio < DANGER_HEALTH:
            nearby = bot.enemy_units.closer_than(THREAT_RADIUS, building.position)
            return any(
                not e.is_structure and e.type_id not in _NON_COMBATANT_TYPES
                for e in nearby
            )

        return False

    # ------------------------------------------------------------------ #
    # Confidence scoring
    # ------------------------------------------------------------------ #

    def generate_idea(
        self,
        building: "Unit",
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
        counter_ctx=None,
    ) -> Optional[BuildingIdea]:
        if building.health_max <= 0:
            return None

        health_ratio = building.health / building.health_max

        confidence = 0.70
        evidence: dict = {
            "health_ratio": round(health_ratio, 3),
            "build_progress": round(building.build_progress, 3),
        }

        # --- sub-signal: how critically low is health ---
        if health_ratio < CRITICAL_HEALTH:
            # Scale from 0 (at threshold) to 0.20 (at 0 hp)
            sig = (CRITICAL_HEALTH - health_ratio) / CRITICAL_HEALTH * 0.20
            confidence += sig
            evidence["critical_health_sig"] = round(sig, 3)

        # --- sub-signal: nearby enemy combatants ---
        nearby = bot.enemy_units.closer_than(THREAT_RADIUS, building.position)
        attacker_count = sum(
            1 for e in nearby
            if not e.is_structure and e.type_id not in _NON_COMBATANT_TYPES
        )
        if attacker_count > 0:
            sig = min(0.15, attacker_count * 0.05)
            confidence += sig
            evidence["nearby_attackers"] = attacker_count
            evidence["attacker_sig"] = round(sig, 3)

        # --- sub-signal: investment value (costly buildings are more urgent to save) ---
        try:
            cost = bot.calculate_unit_value(building.type_id)
            if cost is not None:
                total_cost = cost.minerals + cost.vespene
                sig = min(0.05, total_cost / 3000.0 * 0.05)
                confidence += sig
                evidence["investment_cost"] = total_cost
                evidence["investment_sig"] = round(sig, 3)
        except Exception:
            pass

        confidence = min(0.95, confidence)

        if confidence < 0.40:
            return None

        log.debug(
            "CancelDyingBuildingTactic: %s tag=%d hp=%.2f progress=%.2f "
            "attackers=%d confidence=%.3f",
            building.type_id.name,
            building.tag,
            health_ratio,
            building.build_progress,
            attacker_count,
            confidence,
        )

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.CANCEL,
            confidence=confidence,
            evidence=evidence,
        )

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def execute(
        self,
        building: "Unit",
        idea: BuildingIdea,
        bot: "ManifestorBot",
    ) -> bool:
        if idea.action != BuildingAction.CANCEL:
            return False

        log.game_event(
            "BUILDING_CANCEL",
            f"{building.type_id.name} tag={building.tag} "
            f"hp={round(building.health / building.health_max, 2)} "
            f"progress={round(building.build_progress, 2)} — cancelling for full refund",
            frame=getattr(getattr(bot, "state", None), "game_loop", None),
        )

        building(AbilityId.CANCEL_BUILDINPROGRESS)
        return True
