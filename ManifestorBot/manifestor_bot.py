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
from ManifestorBot.logger import get_logger
from ManifestorBot.manifests.tactics.building_base import (
    BuildingTacticModule,
    BuildingIdea,
)
from ManifestorBot.manifests.tactics.building_tactics import (
    ZergWorkerProductionTactic,
    ZergArmyProductionTactic,
    ZergUpgradeResearchTactic,
    ZergRallyTactic,
)
from ManifestorBot.manifests.scout_ledger import ScoutLedger


log = get_logger()


class ManifestorBot(AresBot):
    """
    The main bot class. Inherits from AresBot to get all the infrastructure,
    then adds the Manifestor-specific cognitive loop on top.
    """

    def __init__(self, game_step_override: Optional[int] = None):
        log.info("=" * 50)
        log.info("MANIFESTOR BOT INITIALIZING")
        log.info("=" * 50)
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
        
        # Scout ledger for counter-play intelligence
        self.scout_ledger = ScoutLedger(self)
        
        # Tactic modules registry
        self.tactic_modules: List[TacticModule] = []

        # Building tactic modules registry (parallel to self.tactic_modules)
        self.building_modules: List[BuildingTacticModule] = []

        # Rally cache — tracks the last rally point set per building tag.
        # ZergRallyTactic uses this to avoid redundant rally updates.
        self._building_rally_cache: dict = {}
        
        # Suppressed ideas tracker (prevents spam)
        self.suppressed_ideas: Dict[int, int] = {}  # unit_tag -> frame_last_suppressed
        
        # Chat commentary settings
        self.commentary_enabled: bool = True
        self.last_commentary_frame: int = 0
        self.commentary_throttle: int = 112  # ~5 seconds at normal speed
               
    async def on_start(self) -> None:
        """Initialize managers and load tactic modules"""
        await super().on_start()
        
        # Initialize heuristic manager
        self.heuristic_manager = HeuristicManager(self)
        
        # Load all tactic modules
        self._load_tactic_modules()
        self._load_building_modules()

        log.game_event("GAME_START", f"Strategy: {self.current_strategy.value}", frame=0)
        log.info("Loaded %d tactic modules", len(self.tactic_modules), frame=0)
        
        # Opening commentary
        await self._chat(f"Manifestor Bot online. Strategy: {self.current_strategy.value}")
        await self._chat(f"Loaded {len(self.tactic_modules)} tactic modules")
        
    async def on_step(self, iteration: int) -> None:
        """Main game loop - this is where the magic happens"""
        await super().on_step(iteration)
        
        self.scout_ledger.update(iteration)

        # STAGE 1: Calculate all heuristics
        self.heuristic_manager.update(iteration)

        # Log heuristics periodically (every ~5 seconds) to avoid log spam
        if iteration % 112 == 0:
            log.heuristics(self.heuristic_manager.get_state(), frame=self.state.game_loop)
        
        # STAGE 2-6: (TODO - tactic detection, strategy labeling, Dream, weights)
        # For now we'll just use raw heuristics
        
        # STAGE 7: Unit idea generation with suppression
        await self._generate_unit_ideas()
        await self._generate_building_ideas()
        
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
        pending_individual: List[tuple] = []
        pending_group: Dict[str, List[tuple]] = {}

        for unit in all_candidate_units:
            ideas = self._generate_ideas_for_unit(unit)
            if not ideas:
                continue

            ideas.sort(key=lambda x: x.confidence, reverse=True)
            best_idea = ideas[0]

            suppressed = self._should_suppress_idea(unit, best_idea)
            log.tactic(
                best_idea.tactic_module.name,
                unit_tag=unit.tag,
                confidence=best_idea.confidence,
                frame=self.state.game_loop,
                suppressed=suppressed,
            )

            if suppressed:
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
        # Phase 2b: individual execution
        # ------------------------------------------------------------------ #
        for unit, idea in pending_individual:
            await self._execute_idea(unit, idea)

    async def _execute_group_ideas(self, pending_group: Dict) -> None:
        """
        Consolidate group tactic ideas.

        - If enough units share the same tactic (≥ MIN_POSSE_SIZE), issue one
          coordinated attack command and stamp cooldowns.
        - If not met: drop all ideas silently. No lone worker suicide-charges.
        """
        from sc2.ids.ability_id import AbilityId as _AbilityId

        for tactic_name, unit_idea_pairs in pending_group.items():
            if not unit_idea_pairs:
                continue

            tactic_module = unit_idea_pairs[0][1].tactic_module
            min_size = getattr(tactic_module, 'MIN_POSSE_SIZE', 2)

            if len(unit_idea_pairs) < min_size:
                log.debug(
                    "Group tactic %s dropped — posse too small (%d < %d)",
                    tactic_name, len(unit_idea_pairs), min_size,
                    frame=self.state.game_loop,
                )
                continue

            unit_idea_pairs.sort(key=lambda x: x[1].confidence, reverse=True)
            best_target = unit_idea_pairs[0][1].target

            if best_target is None or best_target not in self.enemy_units:
                log.debug(
                    "Group tactic %s dropped — target evaporated",
                    tactic_name,
                    frame=self.state.game_loop,
                )
                continue

            posse_units = [u for u, _ in unit_idea_pairs]
            posse_tags = {u.tag for u in posse_units}
            self.give_same_action(_AbilityId.ATTACK, posse_tags, best_target)

            log.game_event(
                "GROUP_ATTACK",
                f"{tactic_name} | {len(posse_units)} units → target={best_target.tag}",
                frame=self.state.game_loop,
            )

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
        """
        if idea.confidence < 0.40:
            return True

        if unit.tag in self.suppressed_ideas:
            frames_since_last = self.state.game_loop - self.suppressed_ideas[unit.tag]
            if frames_since_last < 50:
                return True

        if unit.orders and len(unit.orders) > 1:
            return True

        return False
        
    async def _execute_idea(self, unit: Unit, idea: TacticIdea) -> None:
        """
        Convert a tactic idea into actual game commands using Ares behaviors.
        """
        maneuver = CombatManeuver()
        behavior = idea.tactic_module.create_behavior(unit, idea, self)
        if behavior:
            maneuver.add(behavior)
            self.register_behavior(maneuver)
            
        if idea.confidence > 0.7 and self.commentary_enabled:
            await self._chat(
                f"{unit.type_id.name[:4]}-{unit.tag % 1000}: "
                f"{idea.tactic_module.name} ({idea.confidence:.2f})"
            )
        
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

        detail = f"{old_strategy.value} → {new_strategy.value}"
        if reason:
            detail += f" ({reason})"

        log.game_event("PIVOT", detail, frame=self.state.game_loop)
        
        if self.commentary_enabled:
            msg = f"PIVOT: {old_strategy.value} → {new_strategy.value}"
            if reason:
                msg += f" ({reason})"
            import asyncio
            asyncio.create_task(self._chat(msg))
            
    def _load_tactic_modules(self) -> None:
        """
        Load and register all tactic modules in priority order.
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
            StutterForwardTactic(),
            HarassWorkersTactic(),
            FlankTactic(),
            HoldChokePointTactic(),
            RallyToArmyTactic(),
            KeepUnitSafeTactic(),
        ]

        log.info(
            "Tactic modules loaded: %s",
            ", ".join(t.name for t in self.tactic_modules),
        )
        
    async def _chat(self, message: str) -> None:
        """Send a chat message (with throttling to avoid spam)"""
        if not self.commentary_enabled:
            return
        await self.chat_send(message, team_only=False)

    async def on_end(self, game_result: Result) -> None:
        """Clean shutdown"""
        if hasattr(self, 'state'):
            await super().on_end(game_result)

        log.game_event(
            "GAME_END",
            f"result={game_result} | final_strategy={self.current_strategy.value}",
            frame=getattr(getattr(self, "state", None), "game_loop", None),
        )

    def _load_building_modules(self) -> None:
        """
        Register all building tactic modules in priority order.

        Called once from on_start, immediately after _load_tactic_modules().
        Add new modules to this list to activate them.

        Priority is highest-index-first in idea scoring: modules at the
        front of this list will have their ideas considered first. However,
        because ideas are sorted by confidence before execution, priority
        here only matters when two ideas have equal confidence — which is
        rare in practice.
        """
        self.building_modules = [
            ZergRallyTactic(),             # Rally first — cheap and non-disruptive
            ZergWorkerProductionTactic(),  # Workers before army by default
            ZergArmyProductionTactic(),    # Army when strategy pushes for it
            ZergUpgradeResearchTactic(),   # Upgrades when affordable
        ]

        log.info(
            "Building modules loaded: %s",
            ", ".join(m.name for m in self.building_modules),
        )


    # ---- E2: Main building idea loop ----

    async def _generate_building_ideas(self) -> None:
        """
        Generate and execute ideas for all owned structures.

        Runs every 20 frames (building decisions don't need frame-perfect
        timing — training queues and research decisions are coarse-grained
        by nature). This keeps it cheap even with many structures.

        Pipeline
        --------
        For each idle / ready structure:
        1. Each applicable BuildingTacticModule scores the situation.
        2. The highest-confidence idea wins (if ≥ 0.40 threshold).
        3. Suppression is checked against the shared suppressed_ideas dict.
        4. The winning idea is executed via the module's execute() method.
        5. On success, a suppression cooldown is stamped.
        """
        if self.state.game_loop % 20 != 0:
            return

        for structure in self.structures:
            # Collect all passing ideas for this structure
            ideas: list[tuple[BuildingTacticModule, BuildingIdea]] = []
            counter_ctx = self.scout_ledger.get_counter_context(self.state.game_loop)

            for module in self.building_modules:
                if not module.is_applicable(structure, self):
                    continue

                idea = module.generate_idea(
                    building=structure,
                    bot=self,
                    heuristics=self.heuristic_manager.get_state(),
                    current_strategy=self.current_strategy,
                    counter_ctx=counter_ctx,
                )

                if idea is None:
                    continue

                ideas.append((module, idea))

            if not ideas:
                continue

            # Sort by confidence — highest first
            ideas.sort(key=lambda x: x[1].confidence, reverse=True)
            best_module, best_idea = ideas[0]

            # Suppression check (shared clock with unit ideas)
            if self._should_suppress_building_idea(structure, best_idea):
                log.debug(
                    "Building idea suppressed: %s for %s (conf=%.2f)",
                    best_module.name,
                    structure.type_id.name,
                    best_idea.confidence,
                    frame=self.state.game_loop,
                )
                continue

            # Execute
            success = best_module.execute(structure, best_idea, self)

            if success:
                # Stamp cooldown so this building isn't spammed next tick
                self.suppressed_ideas[structure.tag] = self.state.game_loop

                log.tactic(
                    best_module.name,
                    unit_tag=structure.tag,
                    confidence=best_idea.confidence,
                    frame=self.state.game_loop,
                )

                # Commentary for notable decisions
                if best_idea.confidence > 0.7 and self.commentary_enabled:
                    action_str = _building_idea_summary(best_idea, structure)
                    await self._chat(action_str)


    # ---- E3: Building suppression check ----

    def _should_suppress_building_idea(self, structure, idea: BuildingIdea) -> bool:
        """
        Mirror of _should_suppress_idea() for buildings.

        Key differences from unit suppression:
        - Confidence threshold is lower (0.35 vs 0.40) — buildings have fewer
        competing ideas so a lower bar is appropriate.
        - Rally ideas are never suppressed by the cooldown timer because rally
        updates are idempotent and we want them to stay fresh.
        - Research ideas are never suppressed if the building just became idle
        (detected by having no orders), since the cooldown from a just-
        completed research should not block the next upgrade.
        """
        from ManifestorBot.manifests.tactics.building_base import BuildingAction

        # Hard confidence floor
        if idea.confidence < 0.35:
            return True

        # Rally corrections bypass the cooldown — they're cheap and correct drift
        if idea.action == BuildingAction.SET_RALLY:
            return False

        # Check cooldown timer
        if structure.tag in self.suppressed_ideas:
            frames_since = self.state.game_loop - self.suppressed_ideas[structure.tag]
            if frames_since < 20:   # shorter cooldown than units (20 vs 50 frames)
                return True

        return False


    # ---- E4: Commentary helper ----

    def _building_idea_summary(idea: BuildingIdea, structure) -> str:
        """Format a brief chat string for a building's executed idea."""
        from ManifestorBot.manifests.tactics.building_base import BuildingAction

        bname = structure.type_id.name[:6]
        if idea.action == BuildingAction.TRAIN and idea.train_type:
            return f"{bname}: training {idea.train_type.name} ({idea.confidence:.2f})"
        if idea.action == BuildingAction.RESEARCH and idea.upgrade:
            return f"{bname}: researching {idea.upgrade.name} ({idea.confidence:.2f})"
        if idea.action == BuildingAction.SET_RALLY and idea.rally_point:
            return f"{bname}: rally → ({idea.rally_point.x:.0f},{idea.rally_point.y:.0f})"
        return f"{bname}: building action ({idea.confidence:.2f})"
