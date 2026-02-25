"""
Citizen's Arrest — Workers mob a lone intruder in the mineral line.

A single worker never wins a fight it wasn't designed for. But 3–4 workers
surrounding a single combat unit and attacking simultaneously is a different
proposition entirely: the combined DPS ends the threat in seconds, and the
geometric surround denies kite-back.

This tactic is fundamentally different from every other tactic in the system:
it is INHERENTLY a group behavior. A lone worker generating this idea should
be ignored — the idea only has meaning if enough allies can join the mob.

Because of this, CitizensArrestTactic declares itself as a GROUP tactic via
the `is_group_tactic = True` flag. The main loop treats it differently:
  - Ideas are not executed immediately unit-by-unit.
  - Instead, all passing CitizensArrest ideas are collected in Phase 1.
  - In Phase 2, if enough workers want to mob the same target, the loop
    executes a single `give_same_action` for all of them together.
  - If the posse is too small, ALL ideas are dropped — the lone worker
    does not suicide-charge.

When to fire
------------
  - An enemy combat unit is inside or very close to one of our active bases
  - We have MIN_POSSE_SIZE or more workers within RECRUIT_RADIUS of the intruder
  - The intruder is not a structure or worker (we're defending against a raider)
  - We are not massively behind on army value (no point burning 6 drones on
    a tank backed by a full bio army)

Target selection
----------------
Priority: the most dangerous intruder near our mineral line, scored as
DPS × health_ratio. If multiple enemies are raiding, we mob the worst one.
Workers then pile on together.

Confidence
----------
Confidence scales with the ratio of committed workers to the intruder's
effective health — more workers relative to the target's staying power = higher
confidence. Strategy biases don't modulate this much: worker defense is
a survival necessity regardless of macro strategy.

The tactic is NEVER blocked — even a full-turtle Fortress strategy will
defend its drones if the math is favourable. The confidence floor is high
(0.65) to ensure the idea only fires when the posse is genuinely viable.
"""

from typing import Optional, TYPE_CHECKING, FrozenSet, List

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
# Tuning constants
# ---------------------------------------------------------------------------

# Enemy must be this close to one of our townhalls to trigger a defense response
_BASE_THREAT_RADIUS: float = 18.0

# Workers within this distance of the intruder are eligible to join the posse
_RECRUIT_RADIUS: float = 14.0

# Minimum workers needed to execute the mob (below this, idea is dropped entirely)
MIN_POSSE_SIZE: int = 3

# Worker DPS proxy (Drone/SCV/Probe are similar; close enough for math purposes)
_WORKER_DPS: float = 5.0


class CitizensArrestTactic(TacticModule):
    """
    Workers mob a lone intruder who has entered the mineral line.

    This is a GROUP tactic — the loop handles it specially. See module
    docstring for the full execution model.
    """

    # Signal to the main loop: don't execute me individually
    is_group_tactic: bool = True

    # No blocked strategies — worker self-defense is always on the table
    @property
    def blocked_strategies(self) -> 'FrozenSet[Strategy]':
        return frozenset()

    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        """Only workers can join a Citizen's Arrest posse."""
        if unit.type_id != bot.worker_type:
            return False
        # Workers currently building cannot abandon the construction
        if unit.is_constructing_scv:
            return False
        # No active bases = nothing to defend
        if not bot.townhalls.ready:
            return False
        # Must be an intruder worth responding to
        return bool(_find_intruder(bot))

    def generate_idea(
        self,
        unit: Unit,
        bot: 'ManifestorBot',
        heuristics: 'HeuristicState',
        current_strategy: 'Strategy',
    ) -> Optional[TacticIdea]:
        intruder = _find_intruder(bot)
        if intruder is None:
            return None

        # Worker must be close enough to realistically join the mob
        dist_to_intruder = unit.distance_to(intruder)
        if dist_to_intruder > _RECRUIT_RADIUS:
            return None

        # Count how many workers are within recruit range (including this one)
        workers_nearby = [
            w for w in bot.workers
            if w.distance_to(intruder) <= _RECRUIT_RADIUS
            and not w.is_constructing_scv
        ]
        posse_size = len(workers_nearby)

        if posse_size < MIN_POSSE_SIZE:
            # Not enough workers — idea exists but confidence is below threshold.
            # The group consolidation pass will drop it anyway, but returning None
            # is cleaner: this unit truly has nothing useful to contribute alone.
            return None

        # --- confidence: combined DPS vs intruder staying power ---
        combined_dps = posse_size * _WORKER_DPS
        intruder_health = intruder.health + intruder.shield
        # Kills-per-second * some time horizon; >1.0 means we win fast
        kill_speed = combined_dps / max(1.0, intruder_health)
        confidence = min(0.95, 0.50 + kill_speed * 0.25)

        evidence = {
            'posse_size':       posse_size,
            'intruder_health':  round(intruder_health, 1),
            'combined_dps':     combined_dps,
            'kill_speed':       round(kill_speed, 3),
            'base_confidence':  round(confidence, 3),
        }

        # Small bonus: the more lopsided the mob, the safer to commit
        if posse_size >= 5:
            confidence = min(0.95, confidence + 0.05)
            evidence['overwhelming_numbers'] = 0.05

        # Small penalty: don't burn the economy if we're already critically behind
        if heuristics.army_value_ratio < 0.4:
            confidence -= 0.10
            evidence['army_disadvantage_penalty'] = -0.10

        # The tactic only fires if confidence clears the bar
        if confidence < 0.55:
            return None

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=intruder,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: 'ManifestorBot',
    ) -> Optional[CombatIndividualBehavior]:
        """
        Individual fallback behavior — used only if the loop somehow bypasses
        group consolidation (should not happen in normal operation).
        Returns an AttackTarget so the worker at least attacks the intruder.
        """
        target = idea.target
        if target is None or target not in bot.enemy_units:
            return None
        return AttackTarget(unit=unit, target=target)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _find_intruder(bot: 'ManifestorBot') -> Optional[Unit]:
    """
    Find the most dangerous enemy combat unit inside one of our active bases.

    'Inside a base' = within BASE_THREAT_RADIUS of any ready townhall.
    We exclude workers (drone vs drone is a different tactic) and structures.
    We take the unit with highest DPS × health_ratio — the biggest immediate
    threat, not just the nearest.

    Returns None if no intruder is found.
    """
    if not bot.townhalls.ready or not bot.enemy_units:
        return None

    best_unit: Optional[Unit] = None
    best_score: float = -1.0

    for th in bot.townhalls.ready:
        nearby = bot.enemy_units.closer_than(_BASE_THREAT_RADIUS, th.position)
        for enemy in nearby:
            if enemy.is_structure or enemy.type_id in {UnitID.DRONE, UnitID.SCV, UnitID.PROBE}:
                continue
            # Burrowed units we can't see are memory units — ignore them
            if enemy.is_memory:
                continue

            dps = enemy.ground_dps if hasattr(enemy, 'ground_dps') else 1.0
            total_hp = enemy.health + enemy.shield
            hp_ratio = total_hp / max(1.0, enemy.health_max + enemy.shield_max)
            score = dps * hp_ratio

            if score > best_score:
                best_score = score
                best_unit = enemy

    return best_unit
