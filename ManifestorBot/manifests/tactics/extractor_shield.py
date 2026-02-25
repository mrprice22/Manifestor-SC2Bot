"""
ExtractorShieldTactic — emergency drone survival via extractor morphing.

When a drone is in danger (under attack or critically wounded) and a free
vespene geyser is nearby, the drone can morph into an Extractor.

Why this works:
  - A drone has ~40 HP.  An Extractor has 500 HP with 1 armor — 12x more durable.
  - Cancelling an in-progress Extractor returns 75% of the 25-mineral cost (~18 minerals)
    AND gives the drone back.  Net cost: ~6 minerals.  Net benefit: the drone survives.

Two tactics work together:
  1. ExtractorShieldTactic (unit tactic) — triggers on imperilled drones,
     morphs the drone into an Extractor on a nearby free geyser, and records
     the geyser position in bot._shield_extractor_positions.

  2. CancelSafeExtractorTactic (building tactic) — watches in-progress
     Extractors that were placed as shields.  When the threat has passed
     (no enemy combatants within SAFE_RADIUS), issues CANCEL_BUILDINPROGRESS
     to reclaim the drone and the full 25-mineral cost.

Emergency mineral reserve:
  bot.emergency_mineral_reserve (default 25) is always held back from
  normal production spending so there are always funds available to place
  a shield.  Production helpers in building_base.py use bot.available_minerals
  (= bot.minerals - bot.emergency_mineral_reserve) to respect this floor.

Confidence:
  - ExtractorShieldTactic:   0.75–0.92  (high — saving the drone is critical)
  - CancelSafeExtractorTactic: 0.78     (high — reclaiming resources when safe)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, FrozenSet, Set

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID

from ManifestorBot.manifests.tactics.building_base import (
    BuildingTacticModule,
    BuildingAction,
    BuildingIdea,
)
from ManifestorBot.manifests.tactics.base import TacticIdea
from ManifestorBot.abilities.ability import Ability, AbilityContext
from ManifestorBot.abilities.ability_registry import ability_registry
from ManifestorBot.logger import get_logger

log = get_logger()

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy
    from sc2.unit import Unit
    from sc2.position import Point2


# ── constants ──────────────────────────────────────────────────────────────

# Health ratio below which a drone is considered in immediate danger
DRONE_DANGER_HEALTH: float = 0.60

# Radius to check for enemy combatants threatening the drone
DRONE_THREAT_RADIUS: float = 9.0

# Maximum distance (tiles) from drone to a usable free geyser
GEYSER_SEARCH_RADIUS: float = 12.0

# Radius within which any friendly structure on the geyser blocks it
GEYSER_OCCUPIED_RADIUS: float = 2.0

# Radius within which no enemy combatants means "threat has passed" for cancel
SAFE_RADIUS: float = 15.0

# Radius within which our own army units count as "reinforcements arrived"
REINFORCE_RADIUS: float = 12.0

# Minimum army units nearby to count as "reinforcements arrived"
REINFORCE_THRESHOLD: int = 3

# Extractor mineral cost — must be affordable from bot.available_minerals
EXTRACTOR_COST: int = 25

# Unit types that are non-threatening scouts / non-combatants
_NON_COMBATANT_TYPES: frozenset = frozenset({
    UnitID.OVERLORD, UnitID.OVERLORDTRANSPORT, UnitID.OVERSEER,
    UnitID.OVERSEERSIEGEMODE,
    UnitID.SCV, UnitID.PROBE, UnitID.DRONE,
})


# ── helper ──────────────────────────────────────────────────────────────────

def _nearest_free_geyser(drone: "Unit", bot: "ManifestorBot") -> Optional["Unit"]:
    """
    Find the closest neutral vespene geyser within GEYSER_SEARCH_RADIUS of
    the drone that does NOT already have a gas building or in-progress
    structure on it.

    Returns the geyser Unit, or None if none is available.
    """
    candidates = bot.vespene_geyser.closer_than(GEYSER_SEARCH_RADIUS, drone.position)
    if not candidates:
        return None

    for geyser in candidates.sorted_by_distance_to(drone.position):
        # Skip geysers that already have a friendly gas building or any structure
        if bot.gas_buildings.closer_than(GEYSER_OCCUPIED_RADIUS, geyser.position):
            continue
        if bot.structures.closer_than(GEYSER_OCCUPIED_RADIUS, geyser.position):
            continue
        # Skip geysers already targeted by another pending shield
        already_shielded = any(
            geyser.position.distance_to(pos) < GEYSER_OCCUPIED_RADIUS
            for pos in bot._shield_extractor_positions
        )
        if already_shielded:
            continue
        return geyser

    return None


def _drone_is_in_danger(drone: "Unit", bot: "ManifestorBot") -> bool:
    """True if the drone is under active attack or critically wounded."""
    if drone.health_max > 0 and (drone.health / drone.health_max) < DRONE_DANGER_HEALTH:
        return True
    nearby = bot.enemy_units.closer_than(DRONE_THREAT_RADIUS, drone.position)
    return any(
        not e.is_structure and e.type_id not in _NON_COMBATANT_TYPES
        for e in nearby
    )


def _threat_has_passed(position: "Point2", bot: "ManifestorBot") -> bool:
    """
    True when it is safe to cancel the shield extractor.

    Safe means either:
      (a) No enemy combatants within SAFE_RADIUS, OR
      (b) Enough friendly army units are within REINFORCE_RADIUS (reinforcements
          have arrived and can deal with the threat).
    """
    nearby_enemies = bot.enemy_units.closer_than(SAFE_RADIUS, position)
    has_threat = any(
        not e.is_structure and e.type_id not in _NON_COMBATANT_TYPES
        for e in nearby_enemies
    )
    if not has_threat:
        return True  # (a) threat is gone

    # (b) check reinforcements
    nearby_army = bot.units.closer_than(REINFORCE_RADIUS, position)
    friendly_fighters = sum(
        1 for u in nearby_army
        if not u.is_structure
        and u.type_id not in {bot.worker_type, bot.supply_type,
                               UnitID.OVERLORD, UnitID.OVERSEER}
    )
    return friendly_fighters >= REINFORCE_THRESHOLD


# ── Ability ──────────────────────────────────────────────────────────────────

class ExtractorShieldAbility(Ability):
    """
    Issue the ZERGBUILD_EXTRACTOR command on a drone targeting a free geyser.

    Triggered by ExtractorShieldTactic via the ability registry path.
    Records the geyser position in bot._shield_extractor_positions so that
    CancelSafeExtractorTactic can identify and cancel it when safe.
    """

    UNIT_TYPES: Set[UnitID] = {UnitID.DRONE}
    GOAL: str = "extractor_shield"
    priority: int = 950   # above MineAbility (100) — this is an emergency

    def can_use(self, unit: "Unit", context: AbilityContext, bot: "ManifestorBot") -> bool:
        geyser = context.target_unit
        if geyser is None:
            return False
        if not hasattr(bot, "available_minerals"):
            return bot.minerals >= EXTRACTOR_COST
        return bot.available_minerals >= EXTRACTOR_COST

    def execute(self, unit: "Unit", context: AbilityContext, bot: "ManifestorBot") -> bool:
        geyser = context.target_unit
        if geyser is None:
            return False

        unit(AbilityId.ZERGBUILD_EXTRACTOR, geyser)
        bot._shield_extractor_positions.add(geyser.position)

        log.game_event(
            "EXTRACTOR_SHIELD",
            f"Drone tag={unit.tag} morphing into shield extractor "
            f"at geyser pos={geyser.position} "
            f"hp={round(unit.health / unit.health_max, 2) if unit.health_max > 0 else '?'}",
            frame=getattr(getattr(bot, "state", None), "game_loop", None),
        )

        context.ability_used = self.name
        context.command_issued = True
        return True


# ── Unit tactic ──────────────────────────────────────────────────────────────

class ExtractorShieldTactic:
    """
    Drone self-preservation: morph into a temporary Extractor when imperilled.

    Follows the MiningTactic pattern — it is NOT a full TacticModule subclass.
    Instead it produces a TacticIdea with a pre-built AbilityContext so the
    ability selector routes execution through ExtractorShieldAbility.

    is_applicable conditions:
      - Unit is a Drone
      - Drone is in danger (low health OR enemy combatants very close)
      - A free vespene geyser exists within GEYSER_SEARCH_RADIUS
      - We can afford the extractor from bot.available_minerals
    """

    name = "ExtractorShieldTactic"
    is_group_tactic = False
    blocked_strategies = frozenset()

    def is_applicable(self, unit: "Unit", bot: "ManifestorBot") -> bool:
        if unit.type_id != UnitID.DRONE:
            return False
        if not _drone_is_in_danger(unit, bot):
            return False
        affordable = (
            getattr(bot, "available_minerals", bot.minerals) >= EXTRACTOR_COST
        )
        if not affordable:
            return False
        return _nearest_free_geyser(unit, bot) is not None

    def generate_idea(
        self,
        unit: "Unit",
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        geyser = _nearest_free_geyser(unit, bot)
        if geyser is None:
            return None

        confidence = 0.75
        evidence: dict = {"baseline": 0.75}

        # --- sub-signal: health criticality ---
        if unit.health_max > 0:
            health_ratio = unit.health / unit.health_max
            if health_ratio < DRONE_DANGER_HEALTH:
                sig = (DRONE_DANGER_HEALTH - health_ratio) / DRONE_DANGER_HEALTH * 0.15
                confidence += sig
                evidence["health_danger_sig"] = round(sig, 3)
            evidence["health_ratio"] = round(health_ratio, 3)

        # --- sub-signal: enemy proximity ---
        nearby = bot.enemy_units.closer_than(DRONE_THREAT_RADIUS, unit.position)
        attacker_count = sum(
            1 for e in nearby
            if not e.is_structure and e.type_id not in _NON_COMBATANT_TYPES
        )
        if attacker_count > 0:
            sig = min(0.10, attacker_count * 0.04)
            confidence += sig
            evidence["nearby_attackers"] = attacker_count
            evidence["attacker_sig"] = round(sig, 3)

        # --- sub-signal: geyser reachable fast (closer = better) ---
        dist = unit.distance_to(geyser)
        if dist < 5.0:
            confidence += 0.02
            evidence["geyser_close"] = True

        confidence = min(0.92, confidence)

        if confidence < 0.40:
            return None

        # Build a pre-wired AbilityContext (ability selector uses this directly)
        context = AbilityContext(
            goal="extractor_shield",
            aggression=0.0,          # purely defensive
            target_unit=geyser,
            confidence=confidence,
            evidence=evidence,
        )

        idea = TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=geyser,
        )
        idea.context = context
        return idea

    def create_behavior(self, unit: "Unit", idea: "TacticIdea", bot: "ManifestorBot"):
        # Ability selector handles execution via ExtractorShieldAbility.
        # This path is never reached for drones with the ability registered.
        return None


# ── Building tactic ───────────────────────────────────────────────────────────

class CancelSafeExtractorTactic(BuildingTacticModule):
    """
    Cancel an in-progress shield Extractor once the threat has passed.

    Only applies to Extractors that were recorded in bot._shield_extractor_positions
    (placed by ExtractorShieldTactic).  When no enemy combatants remain within
    SAFE_RADIUS — or enough friendly fighters have arrived — issues
    CANCEL_BUILDINPROGRESS for a full 100% refund + drone restored.
    """

    BUILDING_TYPES: FrozenSet[UnitID] = frozenset({UnitID.EXTRACTOR})

    @property
    def blocked_strategies(self) -> "FrozenSet[Strategy]":
        return frozenset()  # cancelling shield extractors is never blocked

    # ------------------------------------------------------------------ #
    # Structural gate
    # ------------------------------------------------------------------ #

    def is_applicable(self, building: "Unit", bot: "ManifestorBot") -> bool:
        if building.type_id != UnitID.EXTRACTOR:
            return False
        if building.build_progress >= 1.0:
            return False  # only cancel in-progress (full refund only)
        # Only act on extractors explicitly placed as emergency shields
        return any(
            building.position.distance_to(pos) < GEYSER_OCCUPIED_RADIUS
            for pos in getattr(bot, "_shield_extractor_positions", set())
        )

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
        # Only cancel when it is safe to do so
        if not _threat_has_passed(building.position, bot):
            return None  # threat still present — keep the shield up

        confidence = 0.78
        evidence: dict = {
            "build_progress": round(building.build_progress, 3),
            "safe": True,
        }

        # Bonus for being very early in construction (more drone-time saved)
        if building.build_progress < 0.3:
            confidence += 0.08
            evidence["early_cancel_bonus"] = 0.08

        confidence = min(0.90, confidence)

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

        # Remove this position from the shield tracking set
        shield_positions = getattr(bot, "_shield_extractor_positions", set())
        for pos in list(shield_positions):
            if building.position.distance_to(pos) < GEYSER_OCCUPIED_RADIUS:
                shield_positions.discard(pos)
                break

        refund = int(EXTRACTOR_COST * 0.75)
        log.game_event(
            "EXTRACTOR_SHIELD_CANCEL",
            f"Cancelling shield extractor tag={building.tag} "
            f"progress={round(building.build_progress, 2)} — threat passed, "
            f"reclaiming drone + ~{refund} minerals (75% of {EXTRACTOR_COST})",
            frame=getattr(getattr(bot, "state", None), "game_loop", None),
        )

        building(AbilityId.CANCEL_BUILDINPROGRESS)
        return True


# ── Registration helper ───────────────────────────────────────────────────────

def register_extractor_shield_ability() -> None:
    """
    Register ExtractorShieldAbility with the ability registry.

    Call once from ManifestorBot.on_start() after register_worker_abilities().
    """
    ability_registry.register(UnitID.DRONE, ExtractorShieldAbility())
