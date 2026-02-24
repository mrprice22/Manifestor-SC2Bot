"""
BaseDefenseTactic — respond decisively when the enemy attacks our bases.

When enemy combat units are within THREAT_RADIUS of any friendly
townhall, the army must make a critical decision:

  DEFEND:    Pull home and repel the attackers directly.
             Used when army_value_ratio < COUNTER_THRESHOLD — we need
             every unit on defense.

  COUNTER:   Push forward and take the enemy's most exposed base instead.
             Used when army_value_ratio >= COUNTER_THRESHOLD and we have
             non-negative momentum — "trade a base for a base", and we
             win the exchange because we're already ahead.

Confidence: 0.85–0.95 (highest non-queen tactic).
Always outbids patrol and commit tactics when a base is under attack.

Blocked strategies: none — defending our bases is never optional.
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

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy


# ── module-level constants (shared with CommitAttackTactic) ────────────────

# Enemy combat unit within this radius of a townhall triggers base defense
THREAT_RADIUS: float = 25.0

# army_value_ratio threshold above which we counter-attack instead of retreating
COUNTER_THRESHOLD: float = 1.30

# Unit types that are enemy command centres / hatcheries — used for targeting
_BASE_TYPES: frozenset = frozenset({
    UnitID.COMMANDCENTER, UnitID.ORBITALCOMMAND, UnitID.PLANETARYFORTRESS,
    UnitID.NEXUS,
    UnitID.HATCHERY, UnitID.LAIR, UnitID.HIVE,
})

# These are non-combatants — filter them out when checking for attackers
_NON_COMBATANT_TYPES: frozenset = frozenset({
    UnitID.OVERLORD, UnitID.OVERLORDTRANSPORT, UnitID.OVERSEER,
    UnitID.OVERSEERSIEGEMODE,
    UnitID.SCV, UnitID.PROBE, UnitID.DRONE,
})


# ── module-level helper shared with CommitAttackTactic ────────────────────

def get_threatened_townhall(bot: 'ManifestorBot') -> Optional[Unit]:
    """
    Return the first friendly ready townhall that has enemy combat units
    within THREAT_RADIUS, or None if no base is under attack.

    Called from both BaseDefenseTactic and CommitAttackTactic so both
    agree on whether the bot is in base-defense mode.
    """
    for th in bot.townhalls.ready:
        nearby = bot.enemy_units.closer_than(THREAT_RADIUS, th.position)
        if any(
            not e.is_structure and e.type_id not in _NON_COMBATANT_TYPES
            for e in nearby
        ):
            return th
    return None


# ── tactic ────────────────────────────────────────────────────────────────

class BaseDefenseTactic(TacticModule):
    """
    Respond to enemy attacks on our bases — defend or counter-attack.

    Every eligible army unit receives the same decision each frame:
    either "rush home" (DEFEND) or "push to their base" (COUNTER).
    The choice is determined by army_value_ratio vs COUNTER_THRESHOLD.
    """

    @property
    def blocked_strategies(self) -> 'FrozenSet[Strategy]':
        return frozenset()  # Never blocked — defending is always mandatory

    # ------------------------------------------------------------------ #
    # Structural gate
    # ------------------------------------------------------------------ #

    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        if self._is_worker_or_supply(unit, bot):
            return False
        if unit.type_id == UnitID.QUEEN:
            return False  # Queens stay near hatcheries — queen tactics handle them
        return get_threatened_townhall(bot) is not None

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
        threatened_th = get_threatened_townhall(bot)
        if threatened_th is None:
            return None

        confidence = 0.85
        evidence: dict = {}

        # --- sub-signal: scale with attacker count ---
        nearby_enemies = bot.enemy_units.closer_than(THREAT_RADIUS, threatened_th.position)
        attacker_count = sum(
            1 for e in nearby_enemies
            if not e.is_structure and e.type_id not in _NON_COMBATANT_TYPES
        )
        if attacker_count > 0:
            sig = min(0.08, attacker_count * 0.02)
            confidence += sig
            evidence['attackers'] = attacker_count
            evidence['attacker_sig'] = round(sig, 3)

        # --- sub-signal: strategy engage bias (damped — urgency dominates) ---
        sig = profile.engage_bias * 0.3
        confidence += sig
        evidence['strategy_bias'] = round(sig, 3)

        avr = heuristics.army_value_ratio
        evidence['army_value_ratio'] = round(avr, 3)

        # --- decide: counter or defend ---
        if avr >= COUNTER_THRESHOLD and heuristics.momentum >= 0.0:
            counter_target = self._pick_counter_target(bot)
            if counter_target is not None:
                confidence = min(0.95, confidence + 0.05)
                evidence['decision'] = 'counter_attack'
                return TacticIdea(
                    tactic_module=self,
                    confidence=confidence,
                    evidence=evidence,
                    target=counter_target,
                )

        # Defend — move toward the threatened townhall
        evidence['decision'] = 'defend'
        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=threatened_th.position,
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
        # AMove to the target — attack through anything in the way
        return AMove(unit=unit, target=pos, success_at_distance=4.0)

    # ------------------------------------------------------------------ #
    # Target selection helpers
    # ------------------------------------------------------------------ #

    def _pick_counter_target(self, bot: 'ManifestorBot') -> Optional[Point2]:
        """
        Return the best counter-attack target — enemy's most exposed base.

        Priority:
          1. Visible enemy expansion (not the main) — most vulnerable
          2. Any visible enemy base
          3. Enemy main start location
        """
        army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
        centroid = army.center if army else bot.start_location
        main_start = bot.enemy_start_locations[0] if bot.enemy_start_locations else None

        enemy_bases = [s for s in bot.enemy_structures if s.type_id in _BASE_TYPES]
        if enemy_bases:
            expansions = [
                b for b in enemy_bases
                if main_start is None or b.distance_to(main_start) > 15.0
            ]
            pool = expansions if expansions else enemy_bases
            return min(pool, key=lambda b: b.distance_to(centroid)).position

        if main_start is not None:
            return main_start
        return None
