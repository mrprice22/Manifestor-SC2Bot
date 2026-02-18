"""
Manifestor Bot - Main Bot Class

The core loop implementation:
- Named strategy state machine
- Unit idea generation and action suppression
- Tactic module structure
- Chat commentary for debugging and validation
"""

from typing import Optional, Dict, List, Set
from dataclasses import dataclass
from enum import Enum
from sc2.data import Result
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import CombatIndividualBehavior
from ares.consts import UnitRole
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.ability_id import AbilityId
from sc2.unit import Unit
from sc2.position import Point2

# Local imports
from ManifestorBot.manifests.strategy import Strategy
from ManifestorBot.manifests.heuristics import HeuristicManager
from ManifestorBot.manifests.tactics.base import TacticModule, TacticIdea


class ManifestorBot(AresBot):
    """
    The main bot class. Inherits from AresBot to get all the infrastructure,
    then adds the Manifestor-specific cognitive loop on top.
    """

    def __init__(self, game_step_override: Optional[int] = None):
        print("=" * 50)
        print("MANIFESTOR BOT INITIALIZING")
        print("=" * 50)
        super().__init__(game_step_override)
    
        # Initialize tracking attributes that python-sc2 expects
        self._all_units_previous_map = {}
        
        # Initialize AresBot attributes that may not be properly set
        self._used_tumors = set()
        
        # Core state
        self.current_strategy: Strategy = Strategy.STOCK_STANDARD
        self.strategy_started_frame: int = 0
        
        # Managers (created in on_start)
        self.heuristic_manager: Optional[HeuristicManager] = None
        
        # Tactic modules registry
        self.tactic_modules: List[TacticModule] = []
        
        # Suppressed ideas tracker (prevents spam)
        self.suppressed_ideas: Dict[int, int] = {}  # unit_tag -> frame_last_suppressed
        
        # Chat commentary settings
        self.commentary_enabled: bool = True
        self.last_commentary_frame: int = 0
        self.commentary_throttle: int = 112  # ~5 seconds at normal speed
        
    def _prepare_step(self, state, proto_game_info) -> None:
        """Override to handle the async parent method properly"""
        import asyncio
        # Set the state immediately since parent method won't complete synchronously
        self.state = state
        # Schedule the async parent method to run later
        asyncio.create_task(super()._prepare_step(state, proto_game_info))
        
    async def on_start(self) -> None:
        """Initialize managers and load tactic modules"""
        await super().on_start()
        
        # Initialize heuristic manager
        self.heuristic_manager = HeuristicManager(self)
        
        # Load all tactic modules
        self._load_tactic_modules()
        
        # Opening commentary
        await self._chat(f"Manifestor Bot online. Strategy: {self.current_strategy.value}")
        await self._chat(f"Loaded {len(self.tactic_modules)} tactic modules")
        
    async def on_step(self, iteration: int) -> None:
        """Main game loop - this is where the magic happens"""
        await super().on_step(iteration)
        
        # STAGE 1: Calculate all heuristics
        self.heuristic_manager.update(iteration)
        
        # STAGE 2-6: (TODO - tactic detection, strategy labeling, Dream, weights)
        # For now we'll just use raw heuristics
        
        # STAGE 7: Unit idea generation with suppression
        await self._generate_unit_ideas()
        
        # Periodic strategy commentary
        if self.state.game_loop - self.last_commentary_frame > self.commentary_throttle:
            await self._strategy_commentary()
            self.last_commentary_frame = self.state.game_loop
            
    def _get_unit_role(self, unit_tag: int) -> Optional[UnitRole]:
        """Helper function to get the role of a specific unit."""
        if not hasattr(self, 'manager_hub') or not self.manager_hub:
            return None
        
        role_dict = self.mediator.get_unit_role_dict
        for role, unit_tags in role_dict.items():
            if unit_tag in unit_tags:
                return role
        return None
            
    async def _generate_unit_ideas(self) -> None:
        """
        Each unit generates tactic ideas and decides whether to suppress or act.
        This is the core of the two-speed cognitive loop.
        """
        # Only generate ideas every ~10 frames, not every frame
        if self.state.game_loop % 10 != 0:
            return
            
        army_units = self.mediator.get_own_army_dict
        
        for unit_type, units in army_units.items():
            for unit in units:
                # Skip units with explicit roles that shouldn't be interrupted
                unit_role = self._get_unit_role(unit.tag)
                if unit_role in {UnitRole.BUILDING, UnitRole.GATHERING}:
                    continue
                    
                # Generate ideas from all applicable tactics
                ideas = self._generate_ideas_for_unit(unit)
                
                if not ideas:
                    continue
                    
                # Sort by confidence
                ideas.sort(key=lambda x: x.confidence, reverse=True)
                best_idea = ideas[0]
                
                # Action suppression logic
                if self._should_suppress_idea(unit, best_idea):
                    self.suppressed_ideas[unit.tag] = self.state.game_loop
                    continue
                    
                # Idea passed suppression - execute it
                await self._execute_idea(unit, best_idea)
                
    def _generate_ideas_for_unit(self, unit: Unit) -> List[TacticIdea]:
        """
        Cycle through all applicable tactics and let each one generate an idea
        if it thinks it's relevant for this unit right now.
        """
        ideas: List[TacticIdea] = []
        
        for tactic in self.tactic_modules:
            if not tactic.is_applicable(unit, self):
                continue
                
            idea = tactic.generate_idea(
                unit=unit,
                bot=self,
                heuristics=self.heuristic_manager.get_state(),
                current_strategy=self.current_strategy
            )
            
            if idea:
                ideas.append(idea)
                
        return ideas
        
    def _should_suppress_idea(self, unit: Unit, idea: TacticIdea) -> bool:
        """
        Action suppression: only act if the expected value of acting
        clearly exceeds the expected value of doing nothing.
        """
        # Confidence threshold - ideas below this are always suppressed
        if idea.confidence < 0.4:
            return True
            
        # Don't spam the same unit with ideas too frequently
        if unit.tag in self.suppressed_ideas:
            frames_since_last = self.state.game_loop - self.suppressed_ideas[unit.tag]
            if frames_since_last < 50:  # ~2 seconds
                return True
                
        # If unit is executing something important, don't interrupt
        if unit.orders and len(unit.orders) > 1:
            # Has queued commands - probably doing something deliberate
            return True
            
        # Strategic override: some strategies demand action, others demand patience
        strategy_aggression_modifier = self._get_strategy_action_bias()
        adjusted_threshold = 0.6 - (strategy_aggression_modifier * 0.2)
        
        if idea.confidence < adjusted_threshold:
            return True
            
        # Passed all suppression checks
        return False
        
    def _get_strategy_action_bias(self) -> float:
        """
        Different strategies have different biases toward action vs. patience.
        Returns a value from -1.0 (very patient) to 1.0 (very aggressive).
        """
        strategy_biases = {
            Strategy.JUST_GO_PUNCH_EM: 1.0,
            Strategy.ALL_IN: 0.9,
            Strategy.KEEP_EM_BUSY: 0.7,
            Strategy.WAR_ON_SANITY: 0.6,
            Strategy.BLEED_OUT: 0.4,
            Strategy.STOCK_STANDARD: 0.0,
            Strategy.WAR_OF_ATTRITION: -0.2,
            Strategy.DRONE_ONLY_FORTRESS: -0.8,
        }
        return strategy_biases.get(self.current_strategy, 0.0)
        
    async def _execute_idea(self, unit: Unit, idea: TacticIdea) -> None:
        """
        Convert a tactic idea into actual game commands using Ares behaviors.
        """
        # Create a combat maneuver with this tactic's behavior
        maneuver = CombatManeuver()
        
        # The tactic module provides the Ares behavior to execute
        behavior = idea.tactic_module.create_behavior(unit, idea, self)
        if behavior:
            maneuver.add(behavior)
            self.register_behavior(maneuver)
            
        # Commentary on significant ideas
        if idea.confidence > 0.7 and self.commentary_enabled:
            await self._chat(
                f"{unit.type_id.name[:4]}-{unit.tag % 1000}: "
                f"{idea.tactic_module.name} ({idea.confidence:.2f})"
            )
            
    async def _strategy_commentary(self) -> None:
        """Periodic status updates to help understand what the bot is thinking"""
        if not self.commentary_enabled:
            return
            
        h = self.heuristic_manager.get_state()
        
        # Build a concise status string
        status_parts = [
            f"[{self.current_strategy.value}]",
            f"Mom:{h.momentum:.1f}",
            f"ArmyVal:{h.army_value_ratio:.2f}",
            f"Agg:{h.aggression_dial:.0f}",
        ]
        
        await self._chat(" | ".join(status_parts))
        
    def change_strategy(self, new_strategy: Strategy, reason: str = "") -> None:
        """
        Change the current named strategy.
        This is a significant event that should be logged and announced.
        """
        if new_strategy == self.current_strategy:
            return
            
        old_strategy = self.current_strategy
        self.current_strategy = new_strategy
        self.strategy_started_frame = self.state.game_loop
        
        # Commentary on strategy change
        if self.commentary_enabled:
            msg = f"PIVOT: {old_strategy.value} → {new_strategy.value}"
            if reason:
                msg += f" ({reason})"
            # Use asyncio to schedule the chat for next frame
            import asyncio
            asyncio.create_task(self._chat(msg))
            
    def _load_tactic_modules(self) -> None:
        """Load all tactic modules."""
        from ManifestorBot.manifests.tactics.defensive import KeepUnitSafeTactic
        
        # Register all tactics
        self.tactic_modules = [
            KeepUnitSafeTactic(),
            # TODO: Add more tactics as they're implemented
        ]
        
    async def _chat(self, message: str) -> None:
        """Send a chat message (with throttling to avoid spam)"""
        if not self.commentary_enabled:
            return
        await self.chat_send(message, team_only=False)

    async def on_end(self, game_result: Result) -> None:
        """Clean shutdown"""
        # Only call super().on_end() if state was properly initialized
        if hasattr(self, 'state'):
            await super().on_end(game_result)
        
        # Log final commentary
        if self.commentary_enabled:
            print(f"Game ended: {game_result}")
            print(f"Final strategy: {self.current_strategy.value}")