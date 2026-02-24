"""
CommitAttackTactic — commit the army to an attack when we have critical mass.

Fires when:
  - No base is currently under attack (BaseDefenseTactic handles that)
  - Army supply >= MIN_ARMY_SUPPLY (we have enough to make a real push)
  - army_value_ratio >= 1.05 (not at a disadvantage)
  - Not DRONE_ONLY_FORTRESS or BLEED_OUT (pure defensive/econ strats)

Target selection:
  1. Visible enemy expansion (not the main) — most exposed
  2. Any visible enemy base
  3. Enemy start location (always known from game info)

Confidence: 0.62–0.82

This outbids OpportunisticPatrolTactic (0.42–0.55) and RallyToArmyTactic,
but loses to BaseDefenseTactic (0.85+) when a base is under attack, and
loses to StutterForwardTactic when enemies are nearby (they score higher
from target_proximity).

The goal is to give the army meaningful direction when it has accumulated
enough force — push a target instead of patrolling aimlessly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, FrozenSet

from ares.behaviors.combat.individual import AMove
from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit

from ManifestorBot.manifests.tactics.base import TacticModule, TacticIdea
from ManifestorBot.manifests.tactics.base_defense import (
    _BASE_TYPES,
    THREAT_RADIUS,
    get_threatened_townhall,
)

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy


# Minimum army supply before we dare commit a push
MIN_ARMY_SUPPLY: int = 30


class CommitAttackTactic(TacticModule):
    """
    Commit the full army to attack an enemy base when we have critical mass.

    This tactic gives the army purpose when no immediate engagement is
    happening. Instead of drifting or patrolling, we push a real target.
    """

    @property
    def blocked_strategies(self) -> 'FrozenSet[Strategy]':
        from ManifestorBot.manifests.strategy import Strategy
        return frozenset({
            Strategy.DRONE_ONLY_FORTRESS,
            Strategy.BLEED_OUT,
        })

    # ------------------------------------------------------------------ #
    # Structural gate
    # ------------------------------------------------------------------ #

    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        if self._is_strategy_blocked(bot.current_strategy):
            return False
        if self._is_worker_or_supply(unit, bot):
            return False
        if unit.type_id == UnitID.QUEEN:
            return False  # Queens stay near hatcheries
        # Yield to BaseDefenseTactic when a base is under attack
        if get_threatened_townhall(bot) is not None:
            return False
        # Need a minimum army before committing
        if bot.supply_army < MIN_ARMY_SUPPLY:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Confidence scoring
    # ------------------------------------------------------------------ #

    def generate_idea(
        self,
        unit: Unit,
        bot: 'ManifestorBot',
        heuristics: 'HeuristicState',
        current_strategy: 'Strategy',
    ) -> Optional[TacticIdea]:
        profile = current_strategy.profile()
        confidence = 0.62
        evidence: dict = {}

        # --- sub-signal: army value ratio — don't commit when behind ---
        avr = heuristics.army_value_ratio
        if avr < 1.05:
            return None
        sig = min(0.15, (avr - 1.0) * 0.15)
        confidence += sig
        evidence['army_advantage'] = round(sig, 3)
        evidence['army_value_ratio'] = round(avr, 3)

        # --- sub-signal: positive momentum ---
        if heuristics.momentum > 0:
            sig = min(0.05, heuristics.momentum * 0.05)
            confidence += sig
            evidence['momentum'] = round(sig, 3)

        # --- sub-signal: army supply depth (more army = more confidence) ---
        supply_depth = min(0.05, (bot.supply_army - MIN_ARMY_SUPPLY) * 0.001)
        if supply_depth > 0:
            confidence += supply_depth
            evidence['supply_depth'] = round(supply_depth, 3)

        # --- sub-signal: strategy engage bias ---
        confidence += profile.engage_bias
        evidence['strategy_engage_bias'] = profile.engage_bias

        confidence = max(0.0, min(0.82, confidence))
        if confidence < 0.60:
            return None

        target = self._pick_attack_target(bot)
        if target is None:
            return None

        evidence['target'] = str(target)
        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=target,
        )

    # ------------------------------------------------------------------ #
    # Behavior
    # ------------------------------------------------------------------ #

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: 'ManifestorBot',
    ) -> Optional[CombatIndividualBehavior]:
        target = idea.target
        if target is None:
            return None
        pos: Point2 = target.position if isinstance(target, Unit) else target
        return AMove(unit=unit, target=pos, success_at_distance=5.0)

    # ------------------------------------------------------------------ #
    # Target selection
    # ------------------------------------------------------------------ #

    def _pick_attack_target(self, bot: 'ManifestorBot') -> Optional[Point2]:
        """
        Return the best enemy target to push toward.

        Priority:
          1. Visible enemy expansion (not the main) — most vulnerable
          2. Any visible enemy base
          3. Enemy start location (always known)
        """
        army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
        centroid = army.center if army else bot.start_location
        main_start = bot.enemy_start_locations[0] if bot.enemy_start_locations else None

        # Visible enemy bases
        enemy_bases = [s for s in bot.enemy_structures if s.type_id in _BASE_TYPES]
        if enemy_bases:
            expansions = [
                b for b in enemy_bases
                if main_start is None or b.distance_to(main_start) > 15.0
            ]
            pool = expansions if expansions else enemy_bases
            return min(pool, key=lambda b: b.distance_to(centroid)).position

        # Fall back to enemy main start
        if main_start is not None:
            return main_start
        return None
