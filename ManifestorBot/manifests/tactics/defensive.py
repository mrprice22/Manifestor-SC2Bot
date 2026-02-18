"""
KeepUnitSafe Tactic - The defensive fallback.

This is always applicable as a last resort. If no other tactic
fires with high confidence, this one will suggest retreating to safety.
"""

from typing import Optional, TYPE_CHECKING

from ares.behaviors.combat.individual import KeepUnitSafe as AresKeepUnitSafe
from ares.behaviors.combat.individual import CombatIndividualBehavior
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId as UnitID

from ManifestorBot.manifests.tactics.base import TacticModule, TacticIdea

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy


class KeepUnitSafeTactic(TacticModule):
    """
    Defensive fallback tactic - retreat when things look bad.
    
    This tactic has relatively low confidence by default, so it only
    wins when nothing better is available.
    """
    
    def is_applicable(self, unit: Unit, bot: 'ManifestorBot') -> bool:
        """Always applicable - this is the fallback"""
        # Don't apply to workers or overlords
        if unit.type_id in {bot.worker_type, UnitID.OVERLORD, UnitID.OVERSEER}:
            return False
        return True
        
    def generate_idea(
        self,
        unit: Unit,
        bot: 'ManifestorBot',
        heuristics: 'HeuristicState',
        current_strategy: 'Strategy'
    ) -> Optional[TacticIdea]:
        """
        Generate a retreat idea if the situation looks dangerous.
        """
        confidence = 0.0
        evidence = {}
        
        # Sub-signal 1: Unit health
        health_ratio = unit.health / unit.health_max if unit.health_max > 0 else 1.0
        if health_ratio < 0.5:
            confidence += 0.2
            evidence['low_health'] = health_ratio
            
        # Sub-signal 2: Nearby enemy units
        nearby_enemies = bot.enemy_units.closer_than(10, unit.position)
        if nearby_enemies:
            enemy_threat = len(nearby_enemies) / 10.0  # Normalize
            confidence += enemy_threat * 0.3
            evidence['nearby_enemies'] = len(nearby_enemies)
            
        # Sub-signal 3: Threat level heuristic
        if heuristics.threat_level > 0.6:
            confidence += 0.2
            evidence['threat_level'] = heuristics.threat_level
            
        # Sub-signal 4: Army value ratio - if we're outnumbered badly
        if heuristics.army_value_ratio < 0.6:
            confidence += 0.15
            evidence['army_disadvantage'] = heuristics.army_value_ratio
            
        # Sub-signal 5: Strategy modifier - some strategies never retreat
        if current_strategy.is_aggressive():
            confidence *= 0.6  # Aggressive strategies resist retreating
            evidence['strategy_resistance'] = True
            
        # Base confidence is intentionally modest - we only retreat if nothing better
        confidence = min(0.6, confidence)  # Cap at 0.6 so other tactics can win
        
        if confidence < 0.2:
            return None  # Not worth considering
            
        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=None  # No specific target for retreat
        )
        
    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: 'ManifestorBot'
    ) -> Optional[CombatIndividualBehavior]:
        """
        Use Ares' built-in KeepUnitSafe behavior.
        
        This handles pathfinding to safety using the influence grid.
        """
        # Get the ground grid from Ares
        grid = bot.mediator.get_ground_grid
        
        # Create the Ares behavior
        return AresKeepUnitSafe(
            unit=unit,
            grid=grid
        )
