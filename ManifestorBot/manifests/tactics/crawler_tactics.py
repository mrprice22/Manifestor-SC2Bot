"""
CrawlerRepositionTactic — Spine/Spore Crawler repositioning system.

Spine and Spore Crawlers are Zerg's only mobile structures. They can uproot
(become a slow ground unit) and re-root anywhere on creep. This module handles
two complementary scenarios:

  1. CrawlerUprootBuildingTactic (BuildingTacticModule)
     Fires on rooted SPINECRAWLER / SPORECRAWLER structures.
     Detects orphaned crawlers — crawlers whose nearest ready hatchery was
     destroyed and is now > ORPHAN_RADIUS tiles away.  Issues the uproot
     ability so the crawler can migrate to the new nearest base.

     Safety: never uproots when enemies are within DANGER_RADIUS (the crawler
     provides more value defending in place than walking away under fire).

  2. CrawlerMoveTactic (TacticModule)
     Fires on SPINECRAWLERUPROOTED / SPORECRAWLERUPROOTED units.
     Computes the ideal target position (nearest ready hatchery + offset toward
     enemy) and either:
       - moves toward the target (if > ROOT_RADIUS away), or
       - issues the root ability (if ≤ ROOT_RADIUS from target).

     Uprooted crawlers are added to the unit candidate pool explicitly in
     ManifestorBot._generate_unit_ideas(), same pattern as creep tumors.

Confidence levels:
  - CrawlerUprootBuildingTactic: 0.65  (below normal building production 0.80;
    repositioning is useful but not urgent)
  - CrawlerMoveTactic: 0.85  (high — an uprooted crawler is useless until
    re-rooted; get it to its spot ASAP)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, FrozenSet

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit

from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)

from ManifestorBot.manifests.tactics.base import TacticIdea, TacticModule
from ManifestorBot.manifests.tactics.building_base import (
    BuildingTacticModule,
    BuildingAction,
    BuildingIdea,
)
from ManifestorBot.logger import get_logger

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy

log = get_logger()


# ── Tuning constants ───────────────────────────────────────────────────────────

# Crawler is "orphaned" when its nearest ready hatchery is farther than this.
ORPHAN_RADIUS: float = 12.0

# Don't uproot when enemies are this close — rooted is safer under fire.
DANGER_RADIUS: float = 9.0

# Uprooted crawler targets a position this many tiles in front of its hatch.
FORWARD_OFFSET: float = 4.0

# Crawler re-roots when it gets within this distance of the target.
ROOT_RADIUS: float = 2.5

# Ability map: uprooted type → root ability
_ROOT_ABILITY: dict[UnitID, AbilityId] = {
    UnitID.SPINECRAWLERUPROOTED: AbilityId.SPINECRAWLERROOT_SPINECRAWLERROOT,
    UnitID.SPORECRAWLERUPROOTED: AbilityId.SPORECRAWLERROOT_SPORECRAWLERROOT,
}

# Ability map: rooted type → uproot ability
_UPROOT_ABILITY: dict[UnitID, AbilityId] = {
    UnitID.SPINECRAWLER: AbilityId.SPINECRAWLERUPROOT_SPINECRAWLERUPROOT,
    UnitID.SPORECRAWLER: AbilityId.SPORECRAWLERUPROOT_SPORECRAWLERUPROOT,
}


# ── Inline behaviors ──────────────────────────────────────────────────────────

class _CrawlerMoveBehavior(CombatIndividualBehavior):
    """Move an uprooted crawler toward a target position."""

    def __init__(self, unit: Unit, target: Point2) -> None:
        self.unit = unit
        self.target = target

    def execute(self, ai, config, mediator) -> bool:
        self.unit.move(self.target)
        return True


class _CrawlerRootBehavior(CombatIndividualBehavior):
    """Issue the root ability for an uprooted crawler at the target position."""

    def __init__(self, unit: Unit, target: Point2) -> None:
        self.unit = unit
        self.target = target

    def execute(self, ai, config, mediator) -> bool:
        ability = _ROOT_ABILITY.get(self.unit.type_id)
        if ability is None:
            log.warning(
                "_CrawlerRootBehavior: unknown type %s",
                self.unit.type_id.name,
            )
            return False
        self.unit(ability, self.target)
        log.info(
            "_CrawlerRootBehavior: %s tag=%d rooting at %s",
            self.unit.type_id.name,
            self.unit.tag,
            self.target,
        )
        return True


# ── Building tactic — detect orphaned crawlers and uproot them ────────────────

class CrawlerUprootBuildingTactic(BuildingTacticModule):
    """
    Uproot an orphaned Spine or Spore Crawler so it can migrate to the
    nearest active base.

    An orphaned crawler is one whose nearest ready hatchery was destroyed
    and is now more than ORPHAN_RADIUS tiles away.  Left in place it provides
    no value (no creep healing, too far from the defensive line).
    """

    BUILDING_TYPES = frozenset({
        UnitID.SPINECRAWLER,
        UnitID.SPORECRAWLER,
    })

    def is_applicable(self, building: Unit, bot: "ManifestorBot") -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            return False
        # Don't uproot if already uprooting or has any active order
        if building.orders:
            return False
        # Don't uproot while enemies are close — we'd lose defensive coverage
        if bot.enemy_units.closer_than(DANGER_RADIUS, building.position):
            return False
        return True

    def generate_idea(
        self,
        building: Unit,
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
        counter_ctx,
    ) -> Optional[BuildingIdea]:
        if not bot.townhalls.ready:
            return None

        nearest_th = bot.townhalls.ready.closest_to(building.position)
        dist = building.distance_to(nearest_th)

        if dist <= ORPHAN_RADIUS:
            return None  # Still near a hatchery — stay rooted

        confidence = 0.65
        evidence = {
            "orphaned": True,
            "dist_to_nearest_hatch": round(dist, 1),
            "orphan_radius": ORPHAN_RADIUS,
        }

        log.info(
            "CrawlerUprootBuildingTactic: %s tag=%d orphaned (dist=%.1f > %.1f) — uprooting",
            building.type_id.name,
            building.tag,
            dist,
            ORPHAN_RADIUS,
        )

        # BuildingAction.TRAIN with train_type=None is a placeholder;
        # execute() issues the actual uproot ability.
        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=None,
        )

    def execute(self, building: Unit, idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        ability = _UPROOT_ABILITY.get(building.type_id)
        if ability is None:
            log.error(
                "CrawlerUprootBuildingTactic.execute: unknown type %s",
                building.type_id.name,
                frame=bot.state.game_loop,
            )
            return False

        building(ability)
        log.game_event(
            "CRAWLER_UPROOT",
            f"{building.type_id.name} tag={building.tag} uprooting",
            frame=bot.state.game_loop,
        )
        return True


# ── Unit tactic — move uprooted crawlers and re-root them ─────────────────────

class CrawlerMoveTactic(TacticModule):
    """
    Guide an uprooted Spine or Spore Crawler to its ideal position and root it.

    Uprooted crawlers are completely defenceless — they must re-root as quickly
    as possible.  This tactic runs at high confidence (0.85) so it wins over
    any patrol or idle behaviour.

    Target position: nearest ready hatchery, offset FORWARD_OFFSET tiles
    toward the enemy start location (so the crawler sits in front of the
    defensive line, not behind it).
    """

    _CRAWLER_TYPES = frozenset({
        UnitID.SPINECRAWLERUPROOTED,
        UnitID.SPORECRAWLERUPROOTED,
    })

    def is_applicable(self, unit: Unit, bot: "ManifestorBot") -> bool:
        return unit.type_id in self._CRAWLER_TYPES

    def generate_idea(
        self,
        unit: Unit,
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        target = self._ideal_position(unit, bot)
        if target is None:
            return None

        dist = unit.distance_to(target)
        action = "root" if dist <= ROOT_RADIUS else "move"

        confidence = 0.85
        evidence = {
            "action": action,
            "dist_to_target": round(dist, 1),
            "root_radius": ROOT_RADIUS,
        }

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=target,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: "ManifestorBot",
    ) -> Optional[CombatIndividualBehavior]:
        target = idea.target
        if target is None:
            return None

        if unit.distance_to(target) <= ROOT_RADIUS:
            return _CrawlerRootBehavior(unit, target)

        return _CrawlerMoveBehavior(unit, target)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ideal_position(self, unit: Unit, bot: "ManifestorBot") -> Optional[Point2]:
        """
        Target position: nearest hatch + FORWARD_OFFSET tiles toward enemy.

        Keeps the crawler on creep (hatchery is the source of creep) and
        positions it slightly in front of the defensive line so attacks are
        intercepted before reaching the workers.
        """
        if not bot.townhalls.ready:
            return None

        nearest_th = bot.townhalls.ready.closest_to(unit.position)

        if not bot.enemy_start_locations:
            return nearest_th.position

        enemy = bot.enemy_start_locations[0]
        return nearest_th.position.towards(enemy, FORWARD_OFFSET)
