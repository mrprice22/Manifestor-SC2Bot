"""
Flank Tactic — attack from a perpendicular angle to the main engagement.

A flank is most valuable when:
  - Our main army is already engaging (or approaching) the enemy
  - We have a unit that can reach a 90° offset position
  - The enemy is unlikely to intercept before we arrive

When successful, a flank forces the enemy army to split attention,
reduces their effective DPS on the main force, and can cause a rout
if it hits from an undefended side.

Flanking is inherently risky for the flanking unit — it travels alone.
Confidence is capped at 0.75 so the unit won't sacrifice itself on a
poor flank attempt, and low-health units never flank.

Design notes:
  - Flank position is calculated as a 90° offset from the
    our_centroid → enemy_centroid vector, at a distance that keeps
    the unit outside the main enemy engagement radius while still
    being able to engage after arrival.
  - The tactic generates TWO ideas per applicable unit: one for each
    flank direction (left / right 90°). The suppression layer picks
    the higher-confidence one. In practice they score identically from
    global signals; the local geometry signal breaks the tie.
  - create_behavior() uses PathUnitToTarget to reach the flank position
    and then issues an AMove toward the enemy centroid. Once the unit
    is in the fight the next idea cycle will assign it StutterForward
    or KeepUnitSafe depending on how the engagement goes.

Blocked under DRONE_ONLY_FORTRESS (never attack) and KEEP_EM_BUSY
(harass/poke is preferred over committing a unit to a flank).
"""

import math
from typing import Optional, TYPE_CHECKING, FrozenSet

from ares.behaviors.combat.individual import PathUnitToTarget, AMove
from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)
from sc2.position import Point2
from sc2.unit import Unit

from ManifestorBot.manifests.tactics.base import TacticModule, TacticIdea

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy


# How far off-axis to place the flank position (tiles)
_FLANK_OFFSET_DISTANCE: float = 12.0

# The unit must be at least this far from the enemy centroid to bother flanking
# (closer = already in the fight, should use StutterForward instead)
_MIN_DIST_TO_ENGAGE: float = 8.0

# Maximum travel distance for a flank — beyond this the unit will arrive too late
_MAX_FLANK_TRAVEL: float = 40.0


class FlankTactic(TacticModule):
    """
    Move to a 90° offset position then AMove into the enemy.

    The tactic is only worth generating when:
      - Our main army is engaged or advancing (initiative > 0 or momentum > 0)
      - The flanking unit is not already in the main fight
      - The flank position is reachable (within MAX_FLANK_TRAVEL)
      - The unit has enough health to survive the approach

    The engage_bias amplifies flanking under aggressive strategies.
    The sacrifice_ok flag allows low-health units to commit to a flank
    even when their survival is unlikely.
    """

    @property
    def blocked_strategies(self) -> 'FrozenSet[Strategy]':
        from ManifestorBot.manifests.strategy import Strategy
        return frozenset({
            Strategy.DRONE_ONLY_FORTRESS,
            Strategy.KEEP_EM_BUSY,   # harass is better use of isolated units
        })

    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        if self._is_strategy_blocked(bot.current_strategy):
            return False
        if self._is_worker_or_supply(unit, bot):
            return False
        # Need a visible enemy army to flank into
        if not bot.enemy_units:
            return False
        # Need at least a small friendly army for the "main force" to exist
        army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
        if len(army) < 4:
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

        army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
        our_centroid = army.center
        enemy_centroid = bot.enemy_units.center

        dist_to_enemy = unit.distance_to(enemy_centroid)

        # Already in the main fight — StutterForward will handle this
        if dist_to_enemy < _MIN_DIST_TO_ENGAGE:
            return None

        # Calculate both flank positions and pick the better one
        left_pos  = _flank_position(our_centroid, enemy_centroid, side='left')
        right_pos = _flank_position(our_centroid, enemy_centroid, side='right')

        # Choose the flank position that is closer to this unit
        # (minimises travel time and exposure during the approach)
        if unit.distance_to(left_pos) <= unit.distance_to(right_pos):
            flank_pos = left_pos
        else:
            flank_pos = right_pos

        travel_dist = unit.distance_to(flank_pos)
        if travel_dist > _MAX_FLANK_TRAVEL:
            return None   # Too far — would arrive after the battle is decided

        # --- sub-signal: main army is already engaged (makes flank valuable) ---
        if heuristics.initiative > 0.1 or heuristics.momentum > 0.2:
            sig = min(0.25, (max(heuristics.initiative, heuristics.momentum)) * 0.20)
            confidence += sig
            evidence['army_engaged'] = round(sig, 3)

        # --- sub-signal: army value advantage — flank when we can afford the risk ---
        if heuristics.army_value_ratio > 1.1:
            sig = min(0.20, (heuristics.army_value_ratio - 1.0) * 0.25)
            confidence += sig
            evidence['army_advantage'] = round(sig, 3)

        # --- sub-signal: unit is already off-axis (natural flanker) ---
        # Measure how far off the our_centroid→enemy_centroid line this unit is
        off_axis = _off_axis_distance(unit.position, our_centroid, enemy_centroid)
        if off_axis > 5.0:
            sig = min(0.15, off_axis * 0.012)
            confidence += sig
            evidence['natural_flank_position'] = round(sig, 3)

        # --- sub-signal: travel is short — high chance of arriving in time ---
        if travel_dist < 20.0:
            sig = (20.0 - travel_dist) / 20.0 * 0.15
            confidence += sig
            evidence['short_travel'] = round(sig, 3)

        # --- sub-signal: unit health — don't flank with a dying unit ---
        hp = self._health_ratio(unit)
        if hp < 0.50 and not profile.sacrifice_ok:
            penalty = (0.50 - hp) * 0.35
            confidence -= penalty
            evidence['low_health_penalty'] = round(-penalty, 3)

        # --- strategy bias (additive) ---
        confidence += profile.engage_bias
        evidence['strategy_engage_bias'] = profile.engage_bias

        # Cap — a flank is never more confident than a direct engagement
        # (StutterForward can score higher when we're already winning)
        confidence = min(0.75, confidence)

        if confidence < 0.15:
            return None

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=flank_pos,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: 'ManifestorBot',
    ) -> Optional[CombatIndividualBehavior]:
        flank_pos: Optional[Point2] = idea.target
        if flank_pos is None:
            return None

        # If we've reached the flank position, switch to AMove into the enemy
        if unit.distance_to(flank_pos) < 4.0:
            if bot.enemy_units:
                return AMove(
                    unit=unit,
                    target=bot.enemy_units.center,
                    success_at_distance=0.0,
                )
            return None

        # Still travelling — path to the flank position
        # sense_danger=False so the unit doesn't abort mid-approach to a safe
        # flank route; the route itself is designed to be off-axis from
        # the enemy army, so danger is usually low.
        return PathUnitToTarget(
            unit=unit,
            grid=bot.mediator.get_ground_grid,
            target=flank_pos,
            success_at_distance=4.0,
            sense_danger=False,
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _flank_position(
    our_centroid: Point2,
    enemy_centroid: Point2,
    side: str,             # 'left' or 'right'
) -> Point2:
    """
    Calculate a 90° offset position from the our_centroid → enemy_centroid axis.

    The offset is placed at the midpoint of that axis, perpendicular to it,
    at FLANK_OFFSET_DISTANCE tiles to the left or right.
    """
    dx = enemy_centroid.x - our_centroid.x
    dy = enemy_centroid.y - our_centroid.y
    length = max(0.001, math.sqrt(dx * dx + dy * dy))

    # Unit perpendicular vectors (90° rotations of the forward vector)
    if side == 'left':
        perp_x, perp_y = -dy / length, dx / length
    else:
        perp_x, perp_y = dy / length, -dx / length

    # Place the flank position at the midpoint of the axis, offset sideways
    mid_x = (our_centroid.x + enemy_centroid.x) / 2.0
    mid_y = (our_centroid.y + enemy_centroid.y) / 2.0

    return Point2((
        mid_x + perp_x * _FLANK_OFFSET_DISTANCE,
        mid_y + perp_y * _FLANK_OFFSET_DISTANCE,
    ))


def _off_axis_distance(
    unit_pos: Point2,
    our_centroid: Point2,
    enemy_centroid: Point2,
) -> float:
    """
    Perpendicular distance from unit_pos to the line our_centroid → enemy_centroid.

    High value = unit is naturally positioned off to the side,
    making it a good candidate to convert that position into a flank.
    """
    dx = enemy_centroid.x - our_centroid.x
    dy = enemy_centroid.y - our_centroid.y
    line_len = math.sqrt(dx * dx + dy * dy)
    if line_len < 0.001:
        return 0.0

    # Cross product magnitude / line length = perpendicular distance
    cx = unit_pos.x - our_centroid.x
    cy = unit_pos.y - our_centroid.y
    cross = abs(cx * dy - cy * dx)
    return cross / line_len
