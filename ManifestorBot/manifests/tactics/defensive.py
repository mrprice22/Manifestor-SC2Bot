"""
KeepUnitSafe Tactic — The defensive fallback.

Always applicable as a last resort. If no other tactic fires with
high confidence, this one will suggest retreating to safety.

Strategy integration: consumes profile.retreat_bias and profile.sacrifice_ok.
When sacrifice_ok is True, the retreat tactic's base confidence is further
reduced — the strategy has declared that units are currency.
"""

from typing import Optional, TYPE_CHECKING

from ares.behaviors.combat.individual import KeepUnitSafe as AresKeepUnitSafe
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


class KeepUnitSafeTactic(TacticModule):
    """
    Defensive fallback tactic — retreat when things look bad.

    Confidence is intentionally capped at 0.6 so that any offensive
    tactic scoring above that threshold will always override a retreat.
    The confidence floor is also low: we only generate an idea at all
    when there's genuine danger, not just as a constant low-level noise.

    Blocked under no strategy — this is the universal safety net.
    """

    # No blocked_strategies override — always considered

    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        """Always applicable except for workers and supply units."""
        return not self._is_worker_or_supply(unit, bot)

    def generate_idea(
        self,
        unit: Unit,
        bot: 'ManifestorBot',
        heuristics: 'HeuristicState',
        current_strategy: 'Strategy',
    ) -> Optional[TacticIdea]:
        """
        Retreat confidence is built from unit health, local enemy density,
        global threat level, and army disadvantage — then modulated by
        the strategy's retreat_bias and sacrifice_ok flag.
        """
        profile = current_strategy.profile()
        confidence = 0.0
        evidence = {}

        # --- sub-signal: unit health ---
        hp = self._health_ratio(unit)
        if hp < 0.5:
            sig = (0.5 - hp) * 0.5   # max +0.25 at 0% health
            confidence += sig
            evidence['low_health'] = round(sig, 3)

        # --- sub-signal: nearby enemy density ---
        nearby_enemies = bot.enemy_units.closer_than(10, unit.position)
        if nearby_enemies:
            sig = min(0.25, len(nearby_enemies) * 0.04)
            confidence += sig
            evidence['nearby_enemies'] = round(sig, 3)

        # --- sub-signal: global threat level ---
        if heuristics.threat_level > 0.5:
            sig = (heuristics.threat_level - 0.5) * 0.30
            confidence += sig
            evidence['threat_level'] = round(sig, 3)

        # --- sub-signal: army value disadvantage ---
        if heuristics.army_value_ratio < 0.7:
            sig = (0.7 - heuristics.army_value_ratio) * 0.25
            confidence += sig
            evidence['army_disadvantage'] = round(sig, 3)

        # --- strategy bias (additive) ---
        confidence += profile.retreat_bias
        evidence['strategy_retreat_bias'] = profile.retreat_bias

        # --- sacrifice_ok penalty: strategy declared units are currency ---
        if profile.sacrifice_ok:
            confidence -= 0.20
            evidence['sacrifice_ok_penalty'] = -0.20

        # Cap so offensive tactics can always override a retreat
        confidence = min(0.60, confidence)

        # Don't generate noise — only an idea if there's real danger
        if confidence < 0.20:
            return None

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=None,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: 'ManifestorBot',
    ) -> Optional[CombatIndividualBehavior]:
        """Use Ares' KeepUnitSafe with the ground influence grid."""
        return AresKeepUnitSafe(
            unit=unit,
            grid=bot.mediator.get_ground_grid,
        )
