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
        # Create a task for the async parent method but don't wait for it
        # This prevents the "coroutine was never awaited" warning
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

        The loop runs in two phases:

        Phase 1 — Idea collection
            Every eligible unit runs its tactic library and selects its best
            passing idea. Ideas are collected into a pending list rather than
            executed immediately. Group tactics (is_group_tactic=True) are
            separated into their own bucket for special handling.

        Phase 2 — Consolidation and execution
            Group ideas are consolidated: if enough units share the same group
            tactic targeting the same enemy, one coordinated group command is
            issued via give_same_action and all participants get a cooldown stamp.
            If the group is too small, all ideas in that group are dropped.
            Individual ideas are executed one-by-one as before.
        """
        # Only generate ideas every ~10 frames, not every frame
        if self.state.game_loop % 10 != 0:
            return

        # Build the full candidate pool: army units + workers.
        # Workers are excluded from get_own_army_dict but CitizensArrest
        # needs to evaluate them — so we include them explicitly here.
        all_candidate_units: List[Unit] = []

        army_units = self.mediator.get_own_army_dict
        for unit_type, units in army_units.items():
            for unit in units:
                unit_role = self._get_unit_role(unit.tag)
                if unit_role in {UnitRole.BUILDING, UnitRole.GATHERING}:
                    continue
                all_candidate_units.append(unit)

        for worker in self.workers:
            unit_role = self._get_unit_role(worker.tag)
            if unit_role in {UnitRole.BUILDING}:
                continue
            all_candidate_units.append(worker)

        # ------------------------------------------------------------------ #
        # Phase 1: collect all passing ideas
        # ------------------------------------------------------------------ #
        # pending_individual: ready for direct one-by-one execution
        # pending_group: tactic_name -> [(unit, idea)] for group consolidation
        pending_individual: List[tuple] = []
        pending_group: Dict[str, List[tuple]] = {}

        for unit in all_candidate_units:
            ideas = self._generate_ideas_for_unit(unit)
            if not ideas:
                continue

            ideas.sort(key=lambda x: x.confidence, reverse=True)
            best_idea = ideas[0]

            if self._should_suppress_idea(unit, best_idea):
                continue

            tactic = best_idea.tactic_module
            if getattr(tactic, 'is_group_tactic', False):
                bucket = pending_group.setdefault(tactic.name, [])
                bucket.append((unit, best_idea))
            else:
                pending_individual.append((unit, best_idea))

        # ------------------------------------------------------------------ #
        # Phase 2a: group consolidation
        # ------------------------------------------------------------------ #
        await self._execute_group_ideas(pending_group)

        # ------------------------------------------------------------------ #
        # Phase 2b: individual execution (unchanged from original behaviour)
        # ------------------------------------------------------------------ #
        for unit, idea in pending_individual:
            await self._execute_idea(unit, idea)
            self.suppressed_ideas[unit.tag] = self.state.game_loop

    async def _execute_group_ideas(
        self, pending_group: Dict[str, List[tuple]]
    ) -> None:
        """
        Execute group tactics — or drop them if the group is too small.

        For each group tactic bucket:
          - Check the minimum posse size declared by the tactic module.
          - If met: issue one give_same_action for all participants and stamp
            every participant's cooldown.
          - If not met: drop all ideas silently. No lone worker suicide-charges.

        Currently handles CitizensArrest. New group tactics slot in here
        automatically as long as they set is_group_tactic = True and optionally
        declare a MIN_POSSE_SIZE class attribute.
        """
        from sc2.ids.ability_id import AbilityId as _AbilityId

        for tactic_name, unit_idea_pairs in pending_group.items():
            if not unit_idea_pairs:
                continue

            tactic_module = unit_idea_pairs[0][1].tactic_module
            min_size = getattr(tactic_module, 'MIN_POSSE_SIZE', 2)

            if len(unit_idea_pairs) < min_size:
                # Posse too small — don't execute, don't stamp cooldown.
                # Workers go back to mining as if nothing happened.
                continue

            # All ideas in this bucket share the same tactic. Grab the target
            # from the highest-confidence idea.
            unit_idea_pairs.sort(key=lambda x: x[1].confidence, reverse=True)
            best_target = unit_idea_pairs[0][1].target

            if best_target is None or best_target not in self.enemy_units:
                continue  # Target evaporated between idea generation and now

            # Issue one coordinated attack command for the whole posse
            posse_units = [u for u, _ in unit_idea_pairs]
            posse_tags = {u.tag for u in posse_units}
            self.give_same_action(_AbilityId.ATTACK, posse_tags, best_target)

            # Stamp cooldown on every participant
            for unit, _ in unit_idea_pairs:
                self.suppressed_ideas[unit.tag] = self.state.game_loop

            if self.commentary_enabled:
                msg = tactic_module.group_commentary(posse_units, best_target, self)
                if msg:
                    await self._chat(msg)
      
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
        Action suppression: only act if confidence clearly exceeds the threshold.

        The strategy-specific confidence adjustment is now handled inside each
        tactic's generate_idea() via TacticalProfile additive biases. This
        function applies a single universal threshold so the suppression gate
        is simple and predictable.

        New units (not in suppressed_ideas) are never suppressed by the
        cooldown check — they act on their first qualifying idea immediately.
        """
        # Universal confidence floor
        if idea.confidence < 0.40:
            return True

        # Cooldown: has this unit acted recently?
        if unit.tag in self.suppressed_ideas:
            frames_since_last = self.state.game_loop - self.suppressed_ideas[unit.tag]
            if frames_since_last < 50:  # ~2 seconds at 22.4 fps
                return True

        # Don't interrupt units with deliberately queued commands
        if unit.orders and len(unit.orders) > 1:
            return True

        return False
        
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
            msg = f"PIVOT: {old_strategy.value} â†’ {new_strategy.value}"
            if reason:
                msg += f" ({reason})"
            # Use asyncio to schedule the chat for next frame
            import asyncio
            asyncio.create_task(self._chat(msg))
            
    def _load_tactic_modules(self) -> None:
        """
        Load and register all tactic modules in priority order.

        Order matters: when two tactics tie on confidence, the one registered
        earlier wins (because ideas are sorted and the first element taken).
        General ordering principle:
          1. Offensive/positioning tactics (high upside, context-specific)
          2. Defensive fallback (universal, intentionally capped at 0.60)
        """
        from ManifestorBot.manifests.tactics.offensive import (
            StutterForwardTactic,
            HarassWorkersTactic,
        )
        from ManifestorBot.manifests.tactics.positioning import (
            RallyToArmyTactic,
            HoldChokePointTactic,
        )
        from ManifestorBot.manifests.tactics.flank import FlankTactic
        from ManifestorBot.manifests.tactics.defensive import KeepUnitSafeTactic

        self.tactic_modules = [
            StutterForwardTactic(),   # Press favorable engagements
            HarassWorkersTactic(),    # Bleed enemy economy
            FlankTactic(),            # Perpendicular attack vector
            HoldChokePointTactic(),   # Defensive positioning
            RallyToArmyTactic(),      # Cohesion — rejoin the army
            KeepUnitSafeTactic(),     # Fallback: retreat to safety
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