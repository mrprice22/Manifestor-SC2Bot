"""
Positioning Tactics — army cohesion and map control.

RallyToArmyTactic:
    When a unit is isolated from the main army body, path it back to
    the centroid. Isolated units die to focus fire; keeping the army
    together increases effective DPS and survivability.

    The confidence scales with how far the unit has strayed. The
    cohesion_bias amplifies grouping under defensive/attrition
    strategies and suppresses it under WAR_ON_SANITY (which wants
    units spread out intentionally).

HoldChokePointTactic:
    When the threat level is rising and we have no strong offensive
    option, move army units to hold a choke point near our natural
    or third. A unit at the choke point gets a massive defensive
    multiplier from the terrain — worth more than a unit in the open.

    Note: Choke point data comes from Ares' map analyzer
    (mediator.get_closest_choke). This tactic degrades gracefully
    if no choke is found — it returns None from create_behavior and
    the bot records the cooldown without issuing a command.
"""

from typing import Optional, TYPE_CHECKING, FrozenSet

from ares.behaviors.combat.individual import PathUnitToTarget
from ares.behaviors.combat.individual import AMove
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


# ---------------------------------------------------------------------------
# RallyToArmyTactic
# ---------------------------------------------------------------------------

class RallyToArmyTactic(TacticModule):
    """
    Path a stray unit back to the army centroid.

    A unit is considered "stray" when its distance to the army centroid
    exceeds STRAY_THRESHOLD tiles. The further it is, the higher the
    confidence. The cohesion_bias modulates how strongly the strategy
    cares about keeping units together.

    This tactic uses safe pathing (sense_danger=True) so the unit
    doesn't run through enemy lines trying to regroup.

    Blocked under WAR_ON_SANITY — that strategy wants units spread.
    """

    STRAY_THRESHOLD: float = 15.0   # tiles from centroid before we care
    MAX_STRAY_SIG:   float = 0.35   # confidence cap from the distance signal

    @property
    def blocked_strategies(self) -> 'FrozenSet[Strategy]':
        from ManifestorBot.manifests.strategy import Strategy
        return frozenset({Strategy.WAR_ON_SANITY})

    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        if self._is_strategy_blocked(bot.current_strategy):
            return False
        if self._is_worker_or_supply(unit, bot):
            return False
        # Need enough army units to have a meaningful centroid
        army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
        if len(army) < 3:
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

        dist = self._unit_distance_to_army_centroid(unit, bot)

        # Not stray enough to worry about
        if dist < self.STRAY_THRESHOLD:
            return None

        # --- sub-signal: distance from centroid ---
        excess = dist - self.STRAY_THRESHOLD
        sig = min(self.MAX_STRAY_SIG, excess * 0.025)
        confidence += sig
        evidence['stray_distance'] = round(sig, 3)

        # --- sub-signal: global army cohesion (reinforce when already scattered) ---
        if heuristics.army_cohesion < 0.4:
            sig = (0.4 - heuristics.army_cohesion) * 0.25
            confidence += sig
            evidence['low_cohesion'] = round(sig, 3)

        # --- sub-signal: threat level — more urgent to group under pressure ---
        if heuristics.threat_level > 0.3:
            sig = (heuristics.threat_level - 0.3) * 0.20
            confidence += sig
            evidence['threat_urgency'] = round(sig, 3)

        # --- sub-signal: unit health — injured units should retreat to group ---
        hp = self._health_ratio(unit)
        if hp < 0.6:
            sig = (0.6 - hp) * 0.20
            confidence += sig
            evidence['injured_rally'] = round(sig, 3)

        # --- strategy bias (additive) ---
        confidence += profile.cohesion_bias
        evidence['strategy_cohesion_bias'] = profile.cohesion_bias

        if confidence < 0.15:
            return None

        army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
        centroid = army.center

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=centroid,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: 'ManifestorBot',
    ) -> Optional[CombatIndividualBehavior]:
        target = idea.target
        if target is None:
            return None
        # Recalculate centroid in case the army moved since idea generation
        army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
        if len(army) >= 3:
            target = army.center

        return PathUnitToTarget(
            unit=unit,
            grid=bot.mediator.get_ground_grid,
            target=target,
            success_at_distance=self.STRAY_THRESHOLD * 0.5,  # stop when close enough
            sense_danger=True,
            danger_distance=15.0,
        )


# ---------------------------------------------------------------------------
# HoldChokePointTactic
# ---------------------------------------------------------------------------

class HoldChokePointTactic(TacticModule):
    """
    Move a unit to hold a defensive choke point.

    Choke control is highest-value defensive positioning. A unit holding
    the natural ramp is worth far more than the same unit standing in
    an open field.

    Confidence scales with incoming threat level, army disadvantage,
    and how far the unit currently is from the ideal choke position.
    The hold_bias amplifies under attrition/fortress strategies and
    suppresses under aggressive ones (which want to push, not hold).

    Choke point source: mediator.get_closest_choke_area — returns the
    nearest Ares ChokeArea object. We use its center as the A-move target.
    If Ares can't find a choke (open maps), we fall back to the natural
    townhall position as a defensible anchor.

    Blocked under ALL_IN and JUST_GO_PUNCH_EM — those strategies never
    hold; they push.
    """

    # Only fire if threat is at least this high (saves compute on quiet maps)
    MIN_THREAT_TO_HOLD: float = 0.25

    @property
    def blocked_strategies(self) -> 'FrozenSet[Strategy]':
        from ManifestorBot.manifests.strategy import Strategy
        return frozenset({
            Strategy.ALL_IN,
            Strategy.JUST_GO_PUNCH_EM,
        })

    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        if self._is_strategy_blocked(bot.current_strategy):
            return False
        if self._is_worker_or_supply(unit, bot):
            return False
        # Only worth holding if there is an actual threat incoming
        heuristics = bot.heuristic_manager.get_state()
        if heuristics.threat_level < self.MIN_THREAT_TO_HOLD:
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

        # --- sub-signal: threat level (primary driver) ---
        if heuristics.threat_level > self.MIN_THREAT_TO_HOLD:
            sig = min(0.35, (heuristics.threat_level - self.MIN_THREAT_TO_HOLD) * 0.55)
            confidence += sig
            evidence['threat_level'] = round(sig, 3)

        # --- sub-signal: army disadvantage — hold when we can't win in the open ---
        if heuristics.army_value_ratio < 0.9:
            sig = min(0.20, (0.9 - heuristics.army_value_ratio) * 0.30)
            confidence += sig
            evidence['army_disadvantage'] = round(sig, 3)

        # --- sub-signal: negative momentum — we've been losing, dig in ---
        if heuristics.momentum < -0.5:
            sig = min(0.15, abs(heuristics.momentum) * 0.08)
            confidence += sig
            evidence['negative_momentum'] = round(sig, 3)

        # --- sub-signal: choke control low — we don't own it yet ---
        if heuristics.choke_control < 0.5:
            sig = (0.5 - heuristics.choke_control) * 0.15
            confidence += sig
            evidence['choke_uncontrolled'] = round(sig, 3)

        # --- strategy bias (additive) ---
        confidence += profile.hold_bias
        evidence['strategy_hold_bias'] = profile.hold_bias

        if confidence < 0.15:
            return None

        choke_pos = self._find_choke_position(bot)
        if choke_pos is None:
            return None

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=choke_pos,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: 'ManifestorBot',
    ) -> Optional[CombatIndividualBehavior]:
        target = idea.target
        if target is None:
            return None

        # If the unit is already at the choke, issue an AMove in place
        # so it attacks anything that walks in rather than standing idle.
        if unit.distance_to(target) < 3.0:
            return AMove(unit=unit, target=target, success_at_distance=2.0)

        # Otherwise path there using the influence grid
        return PathUnitToTarget(
            unit=unit,
            grid=bot.mediator.get_ground_grid,
            target=target,
            success_at_distance=3.0,
            sense_danger=True,
            danger_distance=20.0,
        )

    # ---------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------- #

    @staticmethod
    def _find_choke_position(bot: 'ManifestorBot') -> Optional[Point2]:
        """
        Find the best defensive choke position.

        Tries Ares' choke analyzer first; falls back to the natural
        expansion position as a defensible anchor point.
        """
        # Ares exposes choke areas via the map analyzer mediator.
        # get_choke_areas returns a list of ChokeArea objects; each has a
        # .center Point2 property. We want the one closest to our natural.
        try:
            choke_areas = bot.mediator.get_choke_areas
            if choke_areas:
                natural = bot.mediator.get_own_nat
                if natural:
                    closest_choke = min(
                        choke_areas,
                        key=lambda c: c.center.distance_to(natural.position),
                    )
                    return closest_choke.center
        except (AttributeError, TypeError):
            # Ares API not available or map has no recognized chokes
            pass

        # Fallback: use our natural expansion position
        try:
            natural = bot.mediator.get_own_nat
            if natural:
                return natural.position
        except AttributeError:
            pass

        # Last resort: midpoint between our start and map center
        if bot.townhalls:
            return bot.townhalls.first.position.towards(
                bot.game_info.map_center, 8.0
            )

        return None
