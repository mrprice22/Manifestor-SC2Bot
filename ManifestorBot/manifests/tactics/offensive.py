"""
Offensive Tactics — engaging and harassing.

StutterForwardTactic:
    When we have a favorable army matchup and enemies are nearby, press
    the advantage. Kite forward: attack on cooldown, move toward target
    in between. Uses StutterUnitForward (attack if ready, else advance).

    Blocked under DRONE_ONLY_FORTRESS — we never attack when turtling.

HarassWorkersTactic:
    When we have initiative and a unit is operating near the enemy base,
    preferentially target enemy workers to bleed their economy.
    Uses AttackTarget on the nearest enemy worker.

    Blocked under DRONE_ONLY_FORTRESS and ALL_IN — fortress never attacks;
    all-in doesn't waste time on workers when structures are the target.
"""

from typing import Optional, TYPE_CHECKING, FrozenSet

from ares.behaviors.combat.individual import StutterUnitForward
from ares.behaviors.combat.individual import AttackTarget
from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.unit import Unit

from ManifestorBot.manifests.tactics.base import TacticModule, TacticIdea

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy


# ---------------------------------------------------------------------------
# StutterForwardTactic
# ---------------------------------------------------------------------------

class StutterForwardTactic(TacticModule):
    """
    Press a favorable engagement. Attack on cooldown; advance when not.

    Confidence scales with how favorable the local army matchup is,
    how much positive momentum we have, and how close the highest-
    threat enemy is. The strategy's engage_bias amplifies or dampens.

    Target selection: highest-threat enemy in range, defined as the
    unit maximising (dps * health_ratio / distance). This prefers
    healthy, high-DPS units nearby — killing them is most impactful.
    """

    @property
    def blocked_strategies(self) -> 'FrozenSet[Strategy]':
        from ManifestorBot.manifests.strategy import Strategy
        return frozenset({Strategy.DRONE_ONLY_FORTRESS})

    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        if self._is_strategy_blocked(bot.current_strategy):
            return False
        if self._is_worker_or_supply(unit, bot):
            return False
        # Need at least one visible enemy to engage
        if not bot.enemy_units:
            return False
        return True

    def generate_idea(
        self,
        unit: Unit,
        bot: 'ManifestorBot',
        heuristics: 'HeuristicState',
        current_strategy: 'Strategy',
    ) -> Optional[TacticIdea]:
        profile = current_strategy.profile()
        confidence = 0.0
        evidence = {}

        # --- sub-signal: army value advantage ---
        # We want to press when we're clearly ahead
        ratio = heuristics.army_value_ratio
        if ratio > 1.0:
            sig = min(0.30, (ratio - 1.0) * 0.30)
            confidence += sig
            evidence['army_advantage'] = round(sig, 3)
        else:
            # Penalise pressing into a disadvantage
            penalty = max(-0.20, (ratio - 1.0) * 0.20)
            confidence += penalty
            evidence['army_disadvantage_penalty'] = round(penalty, 3)

        # --- sub-signal: positive momentum ---
        if heuristics.momentum > 0:
            sig = min(0.15, heuristics.momentum * 0.08)
            confidence += sig
            evidence['momentum'] = round(sig, 3)

        # --- sub-signal: target proximity ---
        # The closer the best target, the more urgent it is to engage
        target = self._find_highest_threat_enemy(unit, bot, search_radius=18.0)
        if target is None:
            return None  # No visible target — tactic doesn't apply
        dist = unit.distance_to(target)
        if dist < 12.0:
            sig = (12.0 - dist) / 12.0 * 0.20   # +0.20 at point-blank
            confidence += sig
            evidence['target_proximity'] = round(sig, 3)

        # --- sub-signal: unit health — don't press if nearly dead ---
        hp = self._health_ratio(unit)
        if hp < 0.35:
            penalty = (0.35 - hp) * 0.40
            confidence -= penalty
            evidence['low_health_penalty'] = round(-penalty, 3)

        # --- strategy bias (additive) ---
        confidence += profile.engage_bias
        evidence['strategy_engage_bias'] = profile.engage_bias

        # sacrifice_ok: even low-health units should press
        if profile.sacrifice_ok and hp < 0.35:
            confidence += 0.15
            evidence['sacrifice_ok_bonus'] = 0.15

        if confidence < 0.15:
            return None

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
        bot: 'ManifestorBot',
    ) -> Optional[CombatIndividualBehavior]:
        target = idea.target
        if target is None or target not in bot.enemy_units:
            # Target died — find a new one opportunistically
            target = self._find_highest_threat_enemy(unit, bot)
        if target is None:
            return None
        return StutterUnitForward(unit=unit, target=target)


# ---------------------------------------------------------------------------
# HarassWorkersTactic
# ---------------------------------------------------------------------------

class HarassWorkersTactic(TacticModule):
    """
    Target enemy workers to bleed their economy.

    Only fires when:
      - We have initiative (our army is closer to them than theirs to us)
      - A visible enemy worker exists
      - The unit is reasonably close to the enemy base

    Worker harassment is a multiplier on the whole game — killing workers
    early compounds over many minutes. The harass_bias amplifies this when
    the strategy calls for economic pressure.

    Target selection: nearest visible enemy worker. Workers are soft
    targets; nearest = fastest to kill = least exposure time.
    """

    # Worker unit types across all races
    _WORKER_TYPES: frozenset = frozenset({
        UnitID.SCV, UnitID.MULE,
        UnitID.PROBE,
        UnitID.DRONE,
    })

    @property
    def blocked_strategies(self) -> 'FrozenSet[Strategy]':
        from ManifestorBot.manifests.strategy import Strategy
        return frozenset({
            Strategy.DRONE_ONLY_FORTRESS,  # never attack when turtling
            Strategy.ALL_IN,              # don't waste time on workers — kill structures
        })

    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        if self._is_strategy_blocked(bot.current_strategy):
            return False
        if self._is_worker_or_supply(unit, bot):
            return False
        # Need visible enemy workers
        workers = bot.enemy_units.of_type(self._WORKER_TYPES)
        if not workers:
            return False
        return True

    def generate_idea(
        self,
        unit: Unit,
        bot: 'ManifestorBot',
        heuristics: 'HeuristicState',
        current_strategy: 'Strategy',
    ) -> Optional[TacticIdea]:
        profile = current_strategy.profile()
        confidence = 0.0
        evidence = {}

        workers = bot.enemy_units.of_type(self._WORKER_TYPES)
        if not workers:
            return None

        nearest_worker = workers.closest_to(unit.position)
        worker_dist = unit.distance_to(nearest_worker)

        # --- sub-signal: initiative — are we in their territory? ---
        if heuristics.initiative > 0.1:
            sig = min(0.25, heuristics.initiative * 0.30)
            confidence += sig
            evidence['initiative'] = round(sig, 3)
        else:
            # Harassing without initiative is risky — suppress unless bias high
            penalty = min(0.15, abs(heuristics.initiative) * 0.20)
            confidence -= penalty
            evidence['no_initiative_penalty'] = round(-penalty, 3)

        # --- sub-signal: proximity to nearest worker ---
        if worker_dist < 20.0:
            sig = (20.0 - worker_dist) / 20.0 * 0.25
            confidence += sig
            evidence['worker_proximity'] = round(sig, 3)
        else:
            return None  # Too far — don't even generate the idea

        # --- sub-signal: economic lead gives us room to harass ---
        if heuristics.economic_health > 1.1:
            sig = min(0.10, (heuristics.economic_health - 1.0) * 0.10)
            confidence += sig
            evidence['economic_lead'] = round(sig, 3)

        # --- sub-signal: unit health — don't harass on empty tanks ---
        hp = self._health_ratio(unit)
        if hp < 0.50:
            penalty = (0.50 - hp) * 0.25
            confidence -= penalty
            evidence['low_health_penalty'] = round(-penalty, 3)

        # --- strategy bias (additive) ---
        confidence += profile.harass_bias
        evidence['strategy_harass_bias'] = profile.harass_bias

        if confidence < 0.15:
            return None

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=nearest_worker,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: 'ManifestorBot',
    ) -> Optional[CombatIndividualBehavior]:
        target = idea.target
        # Re-validate: worker may have died or fled since idea was generated
        workers = bot.enemy_units.of_type(self._WORKER_TYPES)
        if not workers:
            return None
        if target not in bot.enemy_units:
            target = workers.closest_to(unit.position)
        return AttackTarget(unit=unit, target=target)
