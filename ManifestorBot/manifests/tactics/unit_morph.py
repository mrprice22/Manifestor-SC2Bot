"""
ZergUnitMorphTactic — morphs zergling → baneling and roach → ravager.

Why a separate tactic?
----------------------
Banelings and ravagers are unit *morphs*, not larva-trained units.  They do
not appear in _ARMY_PRIORITY (which covers larva production only) and cannot
be produced from a hatchery.  This tactic sits in the normal unit-idea loop
so morphing competes fairly against other decisions — a zergling that is
actively fighting will have high-confidence combat ideas that outbid the
morph, while an idle/rallying zergling will get morphed when the composition
needs it.

Ability path
------------
Morph commands go through the ability registry (not create_behavior) using
dedicated goals "morph_baneling" / "morph_ravager".  The tactic sets
idea.context directly so _build_context in the selector uses it without
needing a _TACTIC_TO_GOAL entry.

Guards (applied in generate_idea)
----------------------------------
  - Composition target for the morph unit must be > 0 at current phase.
  - Current ratio of morph unit must be below target ratio.
  - Must have enough minerals and gas to pay the morph cost.
  - Must keep a minimum number of the source unit alive (don't turn every
    zergling into a baneling; the production tactic will replenish them).
  - Baneling only: Baneling Nest must be complete.
  - In-flight morphs (cocoons) count toward the current ratio so we don't
    over-commit before the first batch finishes.

Confidence
----------
Base confidence is 0.58, raised by the composition deficit.  This sits
comfortably above the 0.40 suppression floor and below the ~0.70+ range
of active combat tactics, so a zergling in a fight will not stop to morph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID

from ManifestorBot.abilities.ability import Ability, AbilityContext
from ManifestorBot.abilities.ability_registry import ability_registry
from ManifestorBot.manifests.tactics.base import TacticIdea, TacticModule
from ManifestorBot.logger import get_logger

if TYPE_CHECKING:
    from sc2.unit import Unit
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy

log = get_logger()


# ---------------------------------------------------------------------------
# Morph costs (minerals, gas) paid on top of the source unit
# ---------------------------------------------------------------------------
_BANELING_COST = (25, 25)
_RAVAGER_COST  = (25, 75)

# Minimum source units to keep alive after morphing.
# Prevents turning every zergling into a baneling and leaving us with none
# for map control, scouting, or the next wave.
_MIN_ZERGLINGS_AFTER_MORPH = 4
_MIN_ROACHES_AFTER_MORPH   = 2

# Non-army unit types excluded when measuring army composition ratios.
_WORKER_AND_SUPPLY: frozenset = frozenset({
    UnitID.DRONE, UnitID.OVERLORD, UnitID.OVERSEER,
    UnitID.OVERLORDTRANSPORT, UnitID.OVERSEERSIEGEMODE,
    UnitID.QUEEN,
})

# Base confidence for a morph idea.  Raised by the deficit, capped at 0.78
# so active-combat tactics (0.70-0.90+) still usually outbid us.
_BASE_CONFIDENCE = 0.58
_MAX_CONFIDENCE  = 0.78


# ---------------------------------------------------------------------------
# Ability implementations
# ---------------------------------------------------------------------------

class BanelingMorphAbility(Ability):
    """Issues the zergling → baneling morph command."""

    UNIT_TYPES: Set[UnitID] = {UnitID.ZERGLING}
    GOAL: str = "morph_baneling"
    priority: int = 60

    def can_use(self, unit: "Unit", context: AbilityContext, bot: "ManifestorBot") -> bool:
        # All gating is done in the tactic; just confirm the unit is still a zergling.
        return unit.type_id == UnitID.ZERGLING

    def execute(self, unit: "Unit", context: AbilityContext, bot: "ManifestorBot") -> bool:
        unit(AbilityId.MORPHZERGLINGTOBANELING_BANELING)
        context.ability_used = self.name
        context.command_issued = True
        log.debug(
            "BanelingMorphAbility: zergling %d → baneling morph issued",
            unit.tag,
            frame=bot.state.game_loop,
        )
        return True


class RavagerMorphAbility(Ability):
    """Issues the roach → ravager morph command."""

    UNIT_TYPES: Set[UnitID] = {UnitID.ROACH}
    GOAL: str = "morph_ravager"
    priority: int = 60

    def can_use(self, unit: "Unit", context: AbilityContext, bot: "ManifestorBot") -> bool:
        return unit.type_id == UnitID.ROACH

    def execute(self, unit: "Unit", context: AbilityContext, bot: "ManifestorBot") -> bool:
        unit(AbilityId.MORPHTORAVAGER_RAVAGER)
        context.ability_used = self.name
        context.command_issued = True
        log.debug(
            "RavagerMorphAbility: roach %d → ravager morph issued",
            unit.tag,
            frame=bot.state.game_loop,
        )
        return True


def register_morph_abilities() -> None:
    """Register morph abilities into the global ability registry.  Call once from on_start."""
    ability_registry.register(UnitID.ZERGLING, BanelingMorphAbility())
    ability_registry.register(UnitID.ROACH,    RavagerMorphAbility())
    log.info("Unit morph abilities registered (baneling, ravager)")


# ---------------------------------------------------------------------------
# Tactic implementations
# ---------------------------------------------------------------------------

def _army_ratio(unit_type: UnitID, bot: "ManifestorBot") -> float:
    """Current ratio of unit_type relative to total army (excluding workers/supply)."""
    army = [u for u in bot.units if u.type_id not in _WORKER_AND_SUPPLY]
    total = max(len(army), 1)
    count = sum(1 for u in army if u.type_id == unit_type)
    return count / total


class BanelingMorphTactic(TacticModule):
    """
    Generates morph ideas for zergling units when the composition calls for
    more banelings than we currently have.

    The idea competes in the normal unit-idea auction.  A zergling that is
    actively fighting will have higher-confidence combat ideas and will not
    stop to morph.  An idle or rallying zergling will morph when needed.
    """

    name = "BanelingMorphTactic"
    is_group_tactic = False

    def is_applicable(self, unit: "Unit", bot: "ManifestorBot") -> bool:
        if unit.type_id != UnitID.ZERGLING:
            return False
        if unit.is_burrowed:
            return False
        # Quick pre-check: baneling nest must exist before doing any math.
        if not bot.structures(UnitID.BANELINGNEST).ready:
            return False
        return True

    def generate_idea(
        self,
        unit: "Unit",
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        # Affordability
        min_cost, gas_cost = _BANELING_COST
        if bot.available_minerals < min_cost or bot.vespene < gas_cost:
            return None

        # Composition target
        profile = current_strategy.profile()
        comp = profile.active_composition(heuristics.game_phase)
        if comp is None:
            return None
        target_ratio = comp.ratios.get(UnitID.BANELING, 0.0)
        if target_ratio <= 0.0:
            return None

        # Current ratio — count cocoons as in-flight banelings so we don't
        # over-commit before the first batch finishes.
        army = [u for u in bot.units if u.type_id not in _WORKER_AND_SUPPLY]
        total = max(len(army), 1)
        cocoon_count = len(bot.units(UnitID.BANELINGCOCOON))
        baneling_count = len(bot.units(UnitID.BANELING)) + cocoon_count
        current_ratio = baneling_count / total

        if current_ratio >= target_ratio:
            return None  # already at or above target

        # Source-unit guard: keep at least _MIN_ZERGLINGS_AFTER_MORPH alive.
        # (The production tactic will replenish them from larva.)
        zergling_count = len(bot.units(UnitID.ZERGLING))
        if zergling_count <= _MIN_ZERGLINGS_AFTER_MORPH:
            return None

        deficit = target_ratio - current_ratio
        confidence = min(_MAX_CONFIDENCE, _BASE_CONFIDENCE + deficit * 1.0)
        evidence = {
            "baneling_deficit":  round(deficit, 3),
            "current_ratio":     round(current_ratio, 3),
            "target_ratio":      round(target_ratio, 3),
            "cocoons_in_flight": cocoon_count,
        }

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=None,
            context=AbilityContext(goal="morph_baneling", confidence=confidence),
        )

    def create_behavior(self, unit: "Unit", idea: TacticIdea, bot: "ManifestorBot"):
        # Execution goes through the ability registry path; this is never called.
        return None


class RavagerMorphTactic(TacticModule):
    """
    Generates morph ideas for roach units when the composition calls for
    more ravagers than we currently have.

    No extra structure prerequisite beyond Roach Warren (implied since roaches exist).
    """

    name = "RavagerMorphTactic"
    is_group_tactic = False

    def is_applicable(self, unit: "Unit", bot: "ManifestorBot") -> bool:
        if unit.type_id != UnitID.ROACH:
            return False
        if unit.is_burrowed:
            return False
        return True

    def generate_idea(
        self,
        unit: "Unit",
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        # Affordability
        min_cost, gas_cost = _RAVAGER_COST
        if bot.available_minerals < min_cost or bot.vespene < gas_cost:
            return None

        # Composition target
        profile = current_strategy.profile()
        comp = profile.active_composition(heuristics.game_phase)
        if comp is None:
            return None
        target_ratio = comp.ratios.get(UnitID.RAVAGER, 0.0)
        if target_ratio <= 0.0:
            return None

        # Current ratio — count cocoons as in-flight ravagers.
        army = [u for u in bot.units if u.type_id not in _WORKER_AND_SUPPLY]
        total = max(len(army), 1)
        cocoon_count = len(bot.units(UnitID.RAVAGERCOCOON))
        ravager_count = len(bot.units(UnitID.RAVAGER)) + cocoon_count
        current_ratio = ravager_count / total

        if current_ratio >= target_ratio:
            return None

        # Source-unit guard: keep at least _MIN_ROACHES_AFTER_MORPH alive.
        roach_count = len(bot.units(UnitID.ROACH))
        if roach_count <= _MIN_ROACHES_AFTER_MORPH:
            return None

        deficit = target_ratio - current_ratio
        confidence = min(_MAX_CONFIDENCE, _BASE_CONFIDENCE + deficit * 1.0)
        evidence = {
            "ravager_deficit":   round(deficit, 3),
            "current_ratio":     round(current_ratio, 3),
            "target_ratio":      round(target_ratio, 3),
            "cocoons_in_flight": cocoon_count,
        }

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=None,
            context=AbilityContext(goal="morph_ravager", confidence=confidence),
        )

    def create_behavior(self, unit: "Unit", idea: TacticIdea, bot: "ManifestorBot"):
        # Execution goes through the ability registry path; this is never called.
        return None
