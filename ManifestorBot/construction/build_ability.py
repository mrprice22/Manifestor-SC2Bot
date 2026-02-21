"""
BuildAbility and BuildingTactic — the unit-level construction layer.

Stack position
--------------
    BuildingTacticModule   (macro: decides "build Spawning Pool")
        ↓  enqueues ConstructionOrder
    ConstructionQueue
        ↓  BuildingTactic reads pending orders
    BuildingTactic         (unit tactic: "I should go build something")
        ↓  generates TacticIdea with context.goal="build"
    BuildAbility           (unit ability: claims order, calls Ares, registers claim)
        ↓
    MorphTracker.register_claim()
        ↓
    bot.request_zerg_placement()   (Ares handles worker selection internally)
        ↓
    SC2 wire command

Why a unit-level tactic for construction?
-----------------------------------------
The current building loop (BuildingTacticModule) decides what to build but uses
Ares' own worker-selection machinery. That's fine for basic macro, but it means:
  - No suppression logic (a pool could be re-requested every 20 frames).
  - No confidence auction (a drone in danger can't resist a build command).
  - No visibility in the idea log.

By routing construction through the unit tactic auction:
  - A drone under attack can have KeepUnitSafeTactic out-bid BuildingTactic.
  - The idea log shows why a drone was chosen for construction.
  - MorphTracker gets clean lifecycle notifications.

One important distinction: BuildingTactic does NOT compete with MiningTactic
on worker selection. Only one ConstructionOrder is claimed at a time, and
BuildingTactic's is_applicable() returns False for drones already claimed by
an order, drones already building, and drones in danger. MiningTactic stays
at conf=0.45; BuildingTactic fires at conf=0.75 when an order is pending,
so construction wins the auction comfortably.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.unit import Unit

from ManifestorBot.abilities.ability import Ability, AbilityContext
from ManifestorBot.abilities.ability_registry import ability_registry
from ManifestorBot.construction.construction_queue import ConstructionOrder, OrderStatus
from ManifestorBot.logger import get_logger

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy
    from ManifestorBot.manifests.tactics.base import TacticIdea

log = get_logger()


# ---------------------------------------------------------------------------
# BuildAbility — issues the actual construction command
# ---------------------------------------------------------------------------

class BuildAbility(Ability):
    """
    Claims the highest-priority pending ConstructionOrder and dispatches
    the drone via Ares' async Zerg placement system.

    Registration: this ability is registered for DRONE in register_construction_abilities().
    It has priority 95 — higher than MineAbility (100)? No: priority 95 < 100 means
    MineAbility wins ties. But BuildingTactic generates ideas at conf=0.75, beating
    MiningTactic's 0.45, so the tactic layer decides which goal wins, not priority.
    Priority within the ability registry is only used when two abilities share the
    same goal. Here BuildAbility is the only ability responding to goal="build",
    so priority is irrelevant in practice.
    """

    UNIT_TYPES: Set[UnitID] = {UnitID.DRONE}
    GOAL: str = "build"
    priority: int = 95
    

    def can_use(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        """
        Eligible if:
        1. There is a pending ConstructionOrder.
        2. That order is not already claimed by another drone.
        3. The drone is not already committed to another order.
        4. A valid placement exists (quick sanity check).
        5. Bot can afford the structure.
        """
        queue = bot.construction_queue
        order = queue.next_pending()
        if order is None:
            return False

        # Is this specific drone already claimed for something?
        if queue.claimed_by_drone(unit.tag) is not None:
            return False

        # Can we afford it?
        if not self._can_afford(order, bot):
            return False

        return True

    def execute(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        """
        1. Claim the top pending order.
        2. Submit the async Ares placement request (Ares selects the worker).
        3. Register the claim with MorphTracker.
        4. Mark the order CLAIMED in the queue.

        Note: we do NOT issue the build command directly. We tell Ares what we want
        (request_zerg_placement) and let it handle worker selection. The specific
        drone executing this ability is the *requesting* drone, but Ares may choose
        a different (closer) drone. MorphTracker will detect whichever drone
        disappears and attribute the morph to this order.

        This is the correct Ares integration pattern — fighting Ares' worker
        selection would cause race conditions.
        """
        queue = bot.construction_queue
        order = queue.next_pending()
        if order is None:
            return False

        frame = bot.state.game_loop

        # Mark claimed in the queue
        queue.mark_claimed(order, unit.tag, frame)

        # Request Ares placement (async — Ares dispatches worker next tick)
        bot.placement_resolver.request_async(
            bot,
            order.structure_type,
            base_location=order.base_location,
            frame=frame,
        )

        # Register with MorphTracker so disappearance is tracked
        bot.morph_tracker.register_claim(order, unit.tag)

        context.ability_used = self.name
        context.command_issued = True

        log.game_event(
            "BUILD_DISPATCHED",
            f"{order.structure_type.name} near {order.base_location} | drone={unit.tag}",
            frame=frame,
        )
        return True

    @staticmethod
    def _can_afford(order: ConstructionOrder, bot: "ManifestorBot") -> bool:
        """Check minerals + gas for the order's structure type."""
        try:
            cost = bot.calculate_unit_value(order.structure_type)
            if cost is None:
                return False
            return (
                bot.minerals >= cost.minerals
                and bot.vespene >= cost.vespene
            )
        except Exception:
            return False


# ---------------------------------------------------------------------------
# BuildingTactic — unit-level tactic that feeds the ability
# ---------------------------------------------------------------------------

class BuildingTactic:
    """
    Tactic wrapper that surfaces pending ConstructionOrders to the unit auction.

    When there is a pending build order, one drone should win the auction and
    go build. This tactic generates a high-confidence idea (0.75) so it beats
    MiningTactic (0.45) but can still be overridden by KeepUnitSafeTactic in
    genuine emergencies.

    Workers currently committed to another order are excluded by is_applicable().
    """

    name = "BuildingTactic"
    is_group_tactic = False
    blocked_strategies = frozenset()

    def is_applicable(self, unit: Unit, bot: "ManifestorBot") -> bool:
        """
        Only applies to:
        - Drone / SCV / Probe (workers)
        - When there's a pending order in the ConstructionQueue
        - When this drone is not already claimed for another order
        """
        if unit.type_id not in {UnitID.DRONE, UnitID.SCV, UnitID.PROBE}:
            return False

        queue = getattr(bot, "construction_queue", None)
        if queue is None or not queue.has_pending():
            return False

        # This drone is already doing a build job
        if queue.claimed_by_drone(unit.tag) is not None:
            return False

        return True

    def generate_idea(
        self,
        unit: Unit,
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional["TacticIdea"]:
        from ManifestorBot.manifests.tactics.base import TacticIdea
        from ManifestorBot.abilities.ability import AbilityContext

        queue = bot.construction_queue
        order = queue.next_pending()
        if order is None:
            return None

        confidence = 0.0
        evidence: dict = {}

        # Base: construction is always high-priority
        confidence += 0.75
        evidence["pending_order"] = 0.75

        # Boost for urgent structures (tech gates)
        _URGENT_STRUCTURES = {
            UnitID.SPAWNINGPOOL,
            UnitID.LAIR,
            UnitID.HIVE,
            UnitID.ROACHWARREN,
            UnitID.HYDRALISKDEN,
            UnitID.SPIRE,
        }
        if order.structure_type in _URGENT_STRUCTURES:
            confidence += 0.10
            evidence["urgent_tech"] = 0.10

        # Strategy aggression: aggressive strategies want tech faster
        profile = current_strategy.profile()
        agg_boost = profile.engage_bias * 0.05
        confidence += agg_boost
        evidence["strategy_agg_boost"] = round(agg_boost, 3)

        # Distance proximity: prefer the closest drone to the build site
        dist = unit.distance_to(order.base_location)
        proximity = max(0.0, 1.0 - (dist / 60.0)) * 0.10
        confidence += proximity
        evidence["proximity"] = round(proximity, 3)

        # Cap at 0.95 — KeepUnitSafeTactic caps at 0.60, so construction
        # always wins except in extreme emergencies where threat is overwhelming
        confidence = min(0.95, confidence)

        # Attach pre-built AbilityContext so AbilitySelector uses registry path
        context = AbilityContext(
            goal="build",
            aggression=0.0,
            target_position=order.base_location,
            confidence=confidence,
            evidence=evidence,
        )
        idea = TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=order.base_location,
        )
        idea.context = context
        return idea

    def create_behavior(self, unit: Unit, idea, bot: "ManifestorBot"):
        # Should not be called — ability selector intercepts first.
        return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_construction_abilities() -> None:
    """
    Register BuildAbility in the ability registry.

    Call once from ManifestorBot.on_start(), after register_worker_abilities().
    """
    ability_registry.register(UnitID.DRONE, BuildAbility())
    # Future: register for SCV, Probe here
