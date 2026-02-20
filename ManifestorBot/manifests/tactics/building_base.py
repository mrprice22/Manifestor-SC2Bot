"""
Building Tactic Module — Base Interface.

The unit tactic system generates ideas for *mobile units*. Buildings are a
separate concern: they cannot move, and their actions fall into exactly three
categories:

    1. Train / morph a unit  (e.g. Hatchery → Drone, Barracks → Marine)
    2. Research an upgrade   (e.g. Lair → Metabolic Boost)
    3. Set / correct a rally point  (e.g. send new units to the army rally)

This module mirrors the unit TacticModule / TacticIdea pattern so that
building logic plugs naturally into the same confidence→suppression→execute
pipeline, with identical logging and strategy-profile integration.

Design principles
-----------------
- ``is_applicable(building, bot)`` is the fast gate — wrong building type,
  already busy, not yet idle — all return False before any math runs.
- ``generate_idea`` scores the situation and returns a ``BuildingIdea`` with
  an additive evidence trail, exactly like unit ideas.
- ``execute`` issues the actual python-sc2 command (train, research, rally).
  It returns True on success so the caller can stamp a suppression cooldown.
- Buildings share the same suppression clock as units (keyed by unit.tag)
  so a Hatchery that just queued a Drone won't get spammed again for 50 frames.
- ``BuildingAction`` is an enum so the execute logic is explicit and easy to
  extend without subclass proliferation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, FrozenSet

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.unit import Unit
from sc2.position import Point2

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------

class BuildingAction(Enum):
    """The three things a building can do on its own initiative."""
    TRAIN         = auto()   # Queue a unit
    RESEARCH      = auto()   # Start an upgrade
    SET_RALLY     = auto()   # Move the rally point


# ---------------------------------------------------------------------------
# Idea dataclass
# ---------------------------------------------------------------------------

@dataclass
class BuildingIdea:
    """
    A building's generated idea, parallel to unit TacticIdea.

    Fields
    ------
    building_module : BuildingTacticModule
        The module that generated this idea.
    action : BuildingAction
        What the building wants to do.
    confidence : float
        0.0–1.0 score. Ideas below 0.40 are suppressed before execution.
    evidence : dict
        Named sub-signal contributions. Used for logging and future LLM
        commentary ("I queued a Drone because saturation_delta was +3").
    train_type : UnitTypeId | None
        Unit to train (only used when action == TRAIN).
    upgrade : UpgradeId | None
        Upgrade to research (only used when action == RESEARCH).
    rally_point : Point2 | None
        Target position (only used when action == SET_RALLY).
    """
    building_module: "BuildingTacticModule"
    action: BuildingAction
    confidence: float
    evidence: dict = field(default_factory=dict)
    train_type: Optional[UnitID] = None
    upgrade: Optional[UpgradeId] = None
    rally_point: Optional[Point2] = None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BuildingTacticModule(ABC):
    """
    Base class for all building tactic modules.

    Each module knows:
      1. Which buildings it applies to     (BUILDING_TYPES)
      2. Whether a specific building is eligible right now  (is_applicable)
      3. How to score and decide what to do  (generate_idea)
      4. How to execute the decided action   (execute)

    Strategy integration
    --------------------
    Building modules receive the current Strategy in ``generate_idea`` but
    should only interact with it through ``current_strategy.profile()``, just
    like unit tactics. The TacticalProfile does not currently have
    production-specific bias fields — add ``produce_bias`` and
    ``research_bias`` to TacticalProfile when you want strategy to influence
    production decisions.

    Suppression
    -----------
    The caller (ManifestorBot._generate_building_ideas) applies the same
    50-frame cooldown used for unit ideas. Modules don't need to track this.
    """

    def __init__(self) -> None:
        self.name: str = self.__class__.__name__

    # ---------------------------------------------------------------- #
    # Class-level configuration (override in subclasses)
    # ---------------------------------------------------------------- #

    #: Set of UnitTypeId that this module handles.
    #: is_applicable() checks this automatically.
    BUILDING_TYPES: FrozenSet[UnitID] = frozenset()

    #: Strategies that suppress this module entirely (same pattern as unit tactics).
    @property
    def blocked_strategies(self) -> "FrozenSet[Strategy]":
        return frozenset()

    # ---------------------------------------------------------------- #
    # Required interface
    # ---------------------------------------------------------------- #

    @abstractmethod
    def is_applicable(self, building: Unit, bot: "ManifestorBot") -> bool:
        """
        Fast structural gate.

        Return False immediately if:
          - building.type_id not in self.BUILDING_TYPES
          - Building is currently producing / researching
          - Building is not ready (build_progress < 1.0)
          - Current strategy is blocked
          - Any other cheap structural check

        Do NOT score confidence here. Save that for generate_idea().

        Typical implementation::

            def is_applicable(self, building, bot):
                if building.type_id not in self.BUILDING_TYPES:
                    return False
                if not building.is_ready:
                    return False
                if building.orders:     # already busy
                    return False
                if bot.current_strategy in self.blocked_strategies:
                    return False
                return True
        """

    @abstractmethod
    def generate_idea(
        self,
        building: Unit,
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[BuildingIdea]:
        """
        Score the situation and return a BuildingIdea if action is warranted.

        Build confidence additively from named sub-signals and record each
        in the evidence dict. Return None if confidence is too low (< 0.15
        is a sensible floor before suppression kicks in at 0.40).

        Example for a training module::

            def generate_idea(self, building, bot, heuristics, current_strategy):
                confidence = 0.0
                evidence = {}

                # Sub-signal: we're below supply cap
                supply_free = bot.supply_cap - bot.supply_used
                if supply_free < 4:
                    return None    # structurally blocked — can't train anyway

                # Sub-signal: we have enough minerals
                if bot.minerals < 50:
                    return None

                # Sub-signal: saturation needs more workers
                delta = heuristics.saturation_delta
                if delta > 0:
                    sig = min(0.5, delta * 0.15)
                    confidence += sig
                    evidence['saturation_delta'] = sig

                # Sub-signal: strategy bias
                profile = current_strategy.profile()
                confidence += profile.engage_bias * -0.1  # eco focus = train drones
                evidence['strategy_engage_bias'] = profile.engage_bias * -0.1

                if confidence < 0.15:
                    return None

                return BuildingIdea(
                    building_module=self,
                    action=BuildingAction.TRAIN,
                    confidence=confidence,
                    evidence=evidence,
                    train_type=UnitID.DRONE,
                )
        """

    @abstractmethod
    def execute(
        self,
        building: Unit,
        idea: BuildingIdea,
        bot: "ManifestorBot",
    ) -> bool:
        """
        Issue the python-sc2 command for this idea.

        Return True on success (the caller stamps a suppression cooldown).
        Return False if the action could not be issued (e.g. resource check
        failed at execution time — prices can change between idea generation
        and execution in the same frame tick).

        Standard implementations are provided by the three mixin helpers
        below (_execute_train, _execute_research, _execute_rally).
        Most concrete modules can simply call one of these and return its result.

        Example::

            def execute(self, building, idea, bot):
                if idea.action == BuildingAction.TRAIN:
                    return self._execute_train(building, idea, bot)
                if idea.action == BuildingAction.SET_RALLY:
                    return self._execute_rally(building, idea, bot)
                return False
        """

    # ---------------------------------------------------------------- #
    # Execution helpers (call from execute() in subclasses)
    # ---------------------------------------------------------------- #

    def _execute_train(
        self,
        building: "Unit",
        idea: BuildingIdea,
        bot: "ManifestorBot",
    ) -> bool:
        if idea.train_type is None:
            return False

        # Zerg units are trained from larva, not from the hatchery itself.
        # If the target unit is a larva-based unit, select a larva near this building.
        from ares.dicts.does_not_use_larva import DOES_NOT_USE_LARVA
        from sc2.ids.unit_typeid import UnitTypeId as UnitID

        if bot.race == "Zerg" and idea.train_type not in DOES_NOT_USE_LARVA:
            nearby_larva = bot.larva.closer_than(15, building.position)
            if not nearby_larva:
                return False  # no larva available yet
            larva = nearby_larva.random
            return larva.train(idea.train_type)

        return building.train(idea.train_type)

    def _execute_research(
        self,
        building: Unit,
        idea: BuildingIdea,
        bot: "ManifestorBot",
    ) -> bool:
        """
        Start researching ``idea.upgrade`` at ``building``.

        Uses python-sc2's ``research()`` helper. Returns True on success.
        """
        if idea.upgrade is None:
            return False
        return building.research(idea.upgrade)

    def _execute_rally(
        self,
        building: Unit,
        idea: BuildingIdea,
        bot: "ManifestorBot",
    ) -> bool:
        """
        Set the rally point of ``building`` to ``idea.rally_point``.

        If rally_point is None the army centroid is used as a fallback.
        Returns True once the command is issued (always succeeds structurally).
        """
        target = idea.rally_point
        if target is None:
            # Fallback: rally to the main army centroid
            army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
            if army:
                target = army.center
            else:
                target = bot.start_location

        building(AbilityId.RALLY_UNITS, target)
        return True

    # ---------------------------------------------------------------- #
    # Convenience helpers (shared by all building modules)
    # ---------------------------------------------------------------- #

    def _building_is_idle(self, building: Unit) -> bool:
        """True if the building has no current orders (not training/researching)."""
        return not building.orders

    def _building_is_ready(self, building: Unit) -> bool:
        """True if the building has finished construction."""
        return building.is_ready and building.build_progress >= 1.0

    def _can_afford_train(self, unit_type: UnitID, bot: "ManifestorBot") -> bool:
        """Check minerals + vespene + supply for a training order."""
        cost = bot.calculate_unit_value(unit_type)
        if cost is None:
            return False
        return (
            bot.minerals >= cost.minerals
            and bot.vespene >= cost.vespene
            and (bot.supply_cap - bot.supply_used) >= bot.calculate_supply_cost(unit_type)
        )

    def _can_afford_research(self, upgrade: UpgradeId, bot: "ManifestorBot") -> bool:
        """Check minerals + vespene for an upgrade order."""
        cost = bot.calculate_cost(upgrade)
        if cost is None:
            return False
        return bot.minerals >= cost.minerals and bot.vespene >= cost.vespene

    def _already_researched(self, upgrade: UpgradeId, bot: "ManifestorBot") -> bool:
        """True if the upgrade is already complete."""
        return upgrade in bot.state.upgrades

    def _is_being_researched(self, upgrade: UpgradeId, bot: "ManifestorBot") -> bool:
        """True if any friendly structure is currently researching this upgrade."""
        from sc2.dicts.upgrade_researched_from import UPGRADE_RESEARCHED_FROM
        researcher_type = UPGRADE_RESEARCHED_FROM.get(upgrade)
        if researcher_type is None:
            return False
        for struct in bot.structures(researcher_type):
            for order in struct.orders:
                # AbilityId names contain the upgrade name — good enough for a gate
                if upgrade.name.lower() in order.ability.name.lower():
                    return True
        return False
