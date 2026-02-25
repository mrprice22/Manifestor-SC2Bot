"""
Worker abilities — Drone (and future SCV / Probe) ability implementations.

MineAbility
-----------
Refactors mining from implicit Ares macro behavior into an explicit, inspectable
ability. The ability:
  1. Finds the worker's assigned mineral field / gas via Ares mediator.
  2. Issues the gather command if the worker is idle or returning cargo.
  3. Returns False (so the selector falls through) if the worker already has
     a valid gather order — Ares' own mining loop handles the fine-grained
     saturation logic and we don't want to fight it.

Registration
------------
Call register_worker_abilities() once from on_start(). This populates the
ability_registry with the drone's ability list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.unit import Unit

from ManifestorBot.abilities.ability import Ability, AbilityContext
from ManifestorBot.abilities.ability_registry import ability_registry

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot


# ---------------------------------------------------------------------------
# MineAbility
# ---------------------------------------------------------------------------

class MineAbility(Ability):
    """
    Send a drone to mine minerals or gas.

    Triggers when context.goal == "mine".
    Priority 100 — mining is the default drone behavior and should win unless
    the tactic layer has explicitly set a higher-confidence override.
    """

    UNIT_TYPES: Set[UnitID] = {UnitID.DRONE}
    GOAL: str = "mine"
    priority: int = 100

    def can_use(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        """
        A drone can mine if:
        - It is idle, OR its current gather target has been depleted / is at a
          fully-saturated base (needs reassignment).
        - There is at least one mineral field or gas building accessible.
        """
        if unit.orders:
            order = unit.orders[0]
            if order.ability.id in _GATHER_ABILITIES:
                # Check if the drone's nearest townhall is mined-out or over-
                # saturated — if so, intervene and reassign.
                #
                # IMPORTANT: use surplus_harvesters (assigned - ideal) rather
                # than comparing assigned against a mineral-only ideal count.
                # assigned_harvesters includes gas workers; ideal_harvesters
                # includes both mineral and gas slots.  Using the raw patch
                # count would make gas workers appear as "over-saturation" and
                # cause every drone to be perpetually redirected to gas.
                if bot.townhalls.ready:
                    closest_th = bot.townhalls.ready.closest_to(unit.position)
                    nearby_minerals = bot.mineral_field.closer_than(10, closest_th.position)
                    if not nearby_minerals:
                        return True  # base mined out — reassign
                    if closest_th.surplus_harvesters > 0:
                        # Only reassign if there's actually an under-saturated
                        # base to move to.  Without this check the drone
                        # oscillates every frame: can_use fires → execute
                        # returns the nearest mineral (at the same full base)
                        # → new gather command → repeat.
                        has_destination = any(
                            th.surplus_harvesters < 0
                            and bool(bot.mineral_field.closer_than(10, th.position))
                            for th in bot.townhalls.ready
                        )
                        return has_destination
                return False  # gather order is fine — let Ares handle it

        # Need somewhere to mine
        return bool(bot.mineral_field or bot.gas_buildings.ready)

    def execute(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        """
        Issue a gather command to the nearest accessible resource.

        Prefers the mineral line at the nearest townhall to keep drones
        local. Falls back to the global mineral list if no townhall found.
        """
        target = self._find_gather_target(unit, bot)
        if target is None:
            return False

        unit.gather(target)
        context.ability_used = self.name
        context.command_issued = True
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_gather_target(self, unit: Unit, bot: "ManifestorBot"):
        """
        Find the best resource for this drone to gather.

        Priority order:
        1. Gas buildings that are ready and under-saturated (≤ 2 workers).
        2. The nearest *under-saturated* townhall's mineral line — this is the
           key change that makes drones transfer away from mined-out bases.
        3. Any mineral field as a global fallback.
        """
        # 1. Unsaturated gas
        for geyser in bot.gas_buildings.ready:
            if geyser.assigned_harvesters < geyser.ideal_harvesters:
                return geyser

        # 2. Minerals at the nearest under-saturated base.
        #    We rank townhalls by saturation headroom (most under-saturated
        #    first) and break ties by distance so drones prefer nearby bases.
        if bot.townhalls.ready:
            best_target: Optional[Unit] = None
            best_score: float = float("inf")

            for th in bot.townhalls.ready:
                nearby_minerals = bot.mineral_field.closer_than(10, th.position)
                if not nearby_minerals:
                    continue  # mined-out base — skip entirely

                # surplus_harvesters = assigned - ideal (accounts for gas too).
                # Negative means under-saturated — this base wants more workers.
                if th.surplus_harvesters >= 0:
                    continue  # at or above ideal — skip

                headroom = abs(th.surplus_harvesters)
                distance = unit.position.distance_to(th.position)
                score = distance / max(headroom, 1)

                if score < best_score:
                    best_score = score
                    best_target = nearby_minerals.closest_to(unit.position)

            if best_target is not None:
                return best_target

            # All ready townhalls are at/above ideal saturation.
            # Don't reassign — returning None keeps the drone's existing
            # gather order and prevents the oversaturation oscillation bug
            # (drone ping-pongs between a mineral patch at the same full base
            # each frame because step 3 used to send it back to the nearest
            # mineral, which was at that same oversaturated base).
            return None

        # 3. No ready townhalls yet (early game — hatchery still morphing).
        #    Mine the nearest mineral so drones don't sit idle.
        if bot.mineral_field:
            return bot.mineral_field.closest_to(unit.position)

        return None


# ---------------------------------------------------------------------------
# WorkerBuildAbility
# ---------------------------------------------------------------------------

class WorkerBuildAbility(Ability):
    """
    Placeholder ability for when a drone is ordered to construct a building.

    Not yet fully implemented — exists to claim the "build" goal slot so that
    future building-placement tactics can route through the ability layer.
    """

    UNIT_TYPES: Set[UnitID] = {UnitID.DRONE, UnitID.SCV, UnitID.PROBE}
    GOAL: str = "build"
    priority: int = 90

    def can_use(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        # Needs a target position to build at
        return context.target_position is not None

    def execute(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        # Delegated to the building tactic — return False so it falls through.
        return False


# ---------------------------------------------------------------------------
# MiningTactic — a thin TacticModule that wraps MineAbility
# ---------------------------------------------------------------------------
# This makes mining visible in the tactic layer so it participates in the
# confidence comparison. Workers can be "out-bid" by higher-confidence tactics
# (e.g. CitizensArrestTactic) and switch off mining for that tick.

class MiningTactic:
    """
    Tactic wrapper that makes mining participate in the idea auction.

    This is NOT a full TacticModule subclass — it produces a TacticIdea with
    a pre-built AbilityContext so the ability selector uses the registry path
    directly, bypassing create_behavior().
    """

    name = "MiningTactic"
    is_group_tactic = False
    blocked_strategies = frozenset()

    def is_applicable(self, unit: Unit, bot: "ManifestorBot") -> bool:
        return unit.type_id in {UnitID.DRONE, UnitID.SCV, UnitID.PROBE}

    def generate_idea(self, unit: Unit, bot, heuristics, current_strategy):
        from ManifestorBot.manifests.tactics.base import TacticIdea

        # Mining confidence: moderate baseline, boosted when economy heuristic
        # is high (strategy wants more income) and depressed when threat is low
        # and other tactics should drive workers to fight.
        profile = current_strategy.profile()
        confidence = 0.45  # baseline — mining is the default worker job
        evidence = {"baseline": 0.45}

        econ_bias = getattr(profile, "econ_bias", 0.0)
        confidence += econ_bias
        evidence["econ_bias"] = econ_bias

        # Suppress mining if threat is high (enables CitizensArrest to out-bid)
        threat = heuristics.threat_level
        if threat > 0.6:
            penalty = (threat - 0.6) * 0.3
            confidence -= penalty
            evidence["threat_penalty"] = -round(penalty, 3)

        if confidence < 0.20:
            return None

        # Attach a pre-built AbilityContext so the selector uses the registry
        context = AbilityContext(
            goal="mine",
            aggression=1.0 - confidence,
            confidence=confidence,
            evidence=evidence,
        )
        idea = TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=None,
        )
        idea.context = context  # selector checks for this attribute
        return idea

    def create_behavior(self, unit, idea, bot):
        # Should not be called — ability selector intercepts first.
        # Returning None is safe (no-op).
        return None


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_worker_abilities() -> None:
    """
    Populate the ability registry with all worker abilities.

    Call once from ManifestorBot.on_start().
    """
    ability_registry.register(UnitID.DRONE, MineAbility())
    ability_registry.register(UnitID.DRONE, WorkerBuildAbility())
    # Future: SCV, Probe, and race-specific abilities registered here.


# ---------------------------------------------------------------------------
# SC2 gather ability IDs (used in can_use to detect active mining)
# ---------------------------------------------------------------------------
_GATHER_ABILITIES: frozenset[AbilityId] = frozenset({
    AbilityId.HARVEST_GATHER,
    AbilityId.HARVEST_GATHER_DRONE,
    AbilityId.HARVEST_GATHER_SCV,
    AbilityId.HARVEST_GATHER_PROBE,
    AbilityId.HARVEST_RETURN,
    AbilityId.HARVEST_RETURN_DRONE,
    AbilityId.HARVEST_RETURN_SCV,
    AbilityId.HARVEST_RETURN_PROBE,
})
