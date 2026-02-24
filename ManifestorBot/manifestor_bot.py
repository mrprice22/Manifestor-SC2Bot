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
from ManifestorBot.manifests.pheromone_map import PheromoneMap, PheromoneConfig
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
    ZergStructureBuildTactic,
    ZergOverlordProductionTactic,
    ZergQueenProductionTactic,
    ZergGasWorkerTactic,
    ZergTechMorphTactic,
)
from ManifestorBot.manifests.tactics.queen_tactics import (
    QueenInjectTactic,
    QueenCreepSpreadTactic,
    TumorSpreadTactic,
)

from ManifestorBot.abilities.ability_registry import ability_registry
from ManifestorBot.abilities.ability_selector import ability_selector
from ManifestorBot.abilities.worker_abilities import (
    register_worker_abilities,
    MiningTactic,
)
from ManifestorBot.manifests.scout_ledger import ScoutLedger
from ManifestorBot.manifests.zergling_scout import ZerglingScouter
from ManifestorBot.construction import (
    ConstructionQueue,
    ConstructionOrder,
    MorphTracker,
    PlacementResolver,
)
from ManifestorBot.construction.build_ability import (
    BuildingTactic,
    register_construction_abilities,
)
from ManifestorBot.manifests.territory_border_map import (
    TerritoryBorderMap,
    BorderConfig,
)



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

        # Zergling scouting manager (pheromone coverage + timed suicide scouts)
        self.zergling_scouter: ZerglingScouter = ZerglingScouter(self)

        # Overlord early-warning system
        self.territory_border_map: Optional[TerritoryBorderMap] = None

        # Construction system
        self.construction_queue: ConstructionQueue = ConstructionQueue()
        self.morph_tracker: MorphTracker = MorphTracker()
        self.placement_resolver: PlacementResolver = PlacementResolver()
        
        # Tactic modules registry
        self.tactic_modules: List[TacticModule] = []

        # Building tactic modules registry (parallel to self.tactic_modules)
        self.building_modules: List[BuildingTacticModule] = []

        # Rally cache — tracks the last rally point set per building tag.
        # ZergRallyTactic uses this to avoid redundant rally updates.
        self._building_rally_cache: dict = {}
        
        # Suppressed ideas tracker (prevents spam)
        self.suppressed_ideas: Dict[int, int] = {}  # unit_tag -> frame_last_suppressed
    
        # Pheromone map
        self.pheromone_map: Optional[PheromoneMap] = None
        
        # Chat commentary settings
        self.commentary_enabled: bool = True
        self.unit_commentary_enabled: bool = False   # too spammy for normal games
        self.last_commentary_frame: int = 0
        self.commentary_throttle: int = 672 # ~30 seconds at normal speed (22.4 fps)
               
    async def on_start(self) -> None:
        """Initialize managers and load tactic modules"""
        await super().on_start()
        
        # Initialize heuristic manager
        self.heuristic_manager = HeuristicManager(self)
        
        # Initialize pheromone map
        self.pheromone_map = PheromoneMap(self, PheromoneConfig())

        # Resolve enemy locations for the zergling scouter
        self.zergling_scouter.initialise_targets()
        log.info("ZerglingScouter targets initialised")

        # Initialise territory border map for overlord placement
        self.territory_border_map = TerritoryBorderMap(self, BorderConfig())
        log.info("TerritoryBorderMap initialised (%dx%d map)",
                 self.game_info.pathing_grid.width,
                 self.game_info.pathing_grid.height)
        
        # Load all tactic modules
        self._load_tactic_modules()
        self._load_building_modules()
        
        # Register unit abilities
        register_worker_abilities()
        register_construction_abilities()

        # Set workers per gas saturation
        self.mediator.set_workers_per_gas(amount=3)

        # Set opening
        opening = self.current_strategy.profile().opening
        self.build_order_runner.switch_opening(opening)
        log.info("Opening selected: %s (strategy: %s)", opening, self.current_strategy.value)

        log.info("Ability registry:\n%s", ability_registry.summary())
    
        log.game_event("GAME_START", f"Strategy: {self.current_strategy.value}", frame=0)
        log.info("Loaded %d tactic modules", len(self.tactic_modules), frame=0)
        
        # Opening commentary
        await self._chat(f"Manifestor Bot online. Strategy: {self.current_strategy.value}")
        await self._chat(f"Loaded {len(self.tactic_modules)} tactic modules")
        
    async def on_step(self, iteration: int) -> None:
        """Main game loop - this is where the magic happens"""
        await super().on_step(iteration)
        
        self.morph_tracker.update(self)
        self.scout_ledger.update(iteration)
        self.pheromone_map.update(iteration)
        self.zergling_scouter.update(iteration)

        # Log a brief scouter summary every ~30 seconds (672 frames)
        if iteration % 672 == 0:
            log.info(self.zergling_scouter.summary())
        
        if self.territory_border_map is not None:
            self.territory_border_map.update(iteration)
        
        if iteration % 224 == 0 and self.territory_border_map is not None:
            log.info(self.territory_border_map.summary())
            uncovered = self.territory_border_map.get_uncovered_slots()
            if uncovered:
                log.info("Uncovered border slots: %d", len(uncovered))

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
                # Skip units currently managed by the zergling scouter so
                # the normal tactic loop doesn't countermand their orders.
                if self.zergling_scouter.is_scouting_tag(unit.tag):
                    log.debug(
                        "_generate_unit_ideas: skipping scouting zergling tag=%d",
                        unit.tag,
                    )
                    continue
                all_candidate_units.append(unit)

        for tumor in self.structures(UnitID.CREEPTUMORBURROWED):
            all_candidate_units.append(tumor)

        for worker in self.workers:
            unit_role = self._get_unit_role(worker.tag)
            if unit_role in {UnitRole.BUILDING}:
                continue
            # Workers with GATHERING role still join the pool —
            # MiningTactic will generate a mining idea which can be
            # out-bid by combat tactics (CitizensArrest, etc.)
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
            f"Phase:{h.game_phase:.2f}",
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
        Route idea execution through the AbilitySelector.

        The selector:
          1. Checks if the unit has registered abilities → uses registry path.
          2. Falls back to tactic.create_behavior() for unported tactics.

        """
        success = ability_selector.select_and_execute(unit, idea, self)
        if success and self.unit_commentary_enabled and idea.confidence > 0.85:
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
        from ManifestorBot.manifests.tactics.overlord_border import OverlordBorderTactic
        from ManifestorBot.manifests.tactics.patrol import OpportunisticPatrolTactic

        self.tactic_modules = [
            BuildingTactic(),
            MiningTactic(),
            StutterForwardTactic(),
            HarassWorkersTactic(),
            FlankTactic(),
            HoldChokePointTactic(),
            RallyToArmyTactic(),
            OpportunisticPatrolTactic(),
            KeepUnitSafeTactic(),
            OverlordBorderTactic(),
            QueenInjectTactic(),
            QueenCreepSpreadTactic(),
            TumorSpreadTactic(),
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

        if self.territory_border_map is not None:
            log.info("Final border map state: %s", self.territory_border_map.summary())

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
            ZergQueenProductionTactic(),   # Queens before workers — they're infrastructure
            ZergOverlordProductionTactic(),# Train Overlords when supply is running short
            ZergArmyProductionTactic(),    # Army BEFORE workers — fix for eco-only behavior
            ZergWorkerProductionTactic(),  # Workers after army is seeded
            ZergUpgradeResearchTactic(),   # Upgrades when affordable
            ZergTechMorphTactic(),         # Hatchery → Lair → Hive tech progression
            ZergStructureBuildTactic(),    # Build structures when needed
            ZergGasWorkerTactic(),         # Ensure gas workers are always assigned
        ]

        log.info(
            "Building modules loaded: %s",
            ", ".join(m.name for m in self.building_modules),
        )


    # ---- E2: Main building idea loop ----
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
            try:
                self._process_building_ideas_for(structure)
                # NOTE: async commentary call moved out — see below.
                # (If you need the chat call, wrap only that part below.)
            except Exception as exc:
                log.exception(
                    "_generate_building_ideas: unhandled exception for structure "
                    "tag=%d type=%s — skipping this structure this tick. Error: %s",
                    structure.tag,
                    structure.type_id.name,
                    exc,
                    frame=self.state.game_loop,
                )

        # Re-run for commentary (async calls cannot live in a non-async helper)
        if self.commentary_enabled:
            await self._emit_building_commentary()

    def _process_building_ideas_for(self, structure) -> None:
        """
        Synchronous core of the building ideas loop for a single structure.

        Separated so it can be wrapped in a try/except without losing async
        context on the happy path.
        """
        ideas: list[tuple["BuildingTacticModule", "BuildingIdea"]] = []
        counter_ctx = self.scout_ledger.get_counter_context(self.state.game_loop)

        for module in self.building_modules:
            applicable = False
            try:
                applicable = module.is_applicable(structure, self)
            except Exception as exc:
                log.exception(
                    "_generate_building_ideas: %s.is_applicable raised for %s tag=%d: %s",
                    module.name,
                    structure.type_id.name,
                    structure.tag,
                    exc,
                    frame=self.state.game_loop,
                )
                continue  # treat as not applicable

            if not applicable:
                continue

            idea = None
            try:
                idea = module.generate_idea(
                    building=structure,
                    bot=self,
                    heuristics=self.heuristic_manager.get_state(),
                    current_strategy=self.current_strategy,
                    counter_ctx=counter_ctx,
                )
            except Exception as exc:
                log.exception(
                    "_generate_building_ideas: %s.generate_idea raised for %s tag=%d: %s",
                    module.name,
                    structure.type_id.name,
                    structure.tag,
                    exc,
                    frame=self.state.game_loop,
                )
                continue  # skip this module

            if idea is None:
                continue

            ideas.append((module, idea))

        if not ideas:
            return

        # Sort by confidence — highest first
        ideas.sort(key=lambda x: x[1].confidence, reverse=True)
        best_module, best_idea = ideas[0]

        # Log the full candidate list for Hatcheries so the confidence race is visible
        if structure.type_id in {
            UnitID.HATCHERY, UnitID.LAIR, UnitID.HIVE
        } and len(ideas) > 1:
            shortlist = ", ".join(
                f"{m.name}({i.confidence:.2f})"
                for m, i in ideas
            )
            log.debug(
                "_generate_building_ideas: %s tag=%d candidates=[%s] → winner=%s(%.2f)",
                structure.type_id.name,
                structure.tag,
                shortlist,
                best_module.name,
                best_idea.confidence,
                frame=self.state.game_loop,
            )

        # Suppression check (shared clock with unit ideas)
        if self._should_suppress_building_idea(structure, best_idea):
            log.debug(
                "Building idea suppressed: %s for %s tag=%d (conf=%.2f)",
                best_module.name,
                structure.type_id.name,
                structure.tag,
                best_idea.confidence,
                frame=self.state.game_loop,
            )
            return

        # Execute
        success = False
        try:
            success = best_module.execute(structure, best_idea, self)
        except Exception as exc:
            log.exception(
                "_generate_building_ideas: %s.execute raised for %s tag=%d: %s",
                best_module.name,
                structure.type_id.name,
                structure.tag,
                exc,
                frame=self.state.game_loop,
            )
            return

        if success:
            # Stamp cooldown so this building isn't spammed next tick
            self.suppressed_ideas[structure.tag] = self.state.game_loop

            log.tactic(
                best_module.name,
                unit_tag=structure.tag,
                confidence=best_idea.confidence,
                frame=self.state.game_loop,
            )

            # Store for async commentary — emitted in _emit_building_commentary()
            if not hasattr(self, "_pending_building_commentary"):
                self._pending_building_commentary = []
            self._pending_building_commentary.append((structure, best_idea))
        else:
            log.debug(
                "_generate_building_ideas: %s.execute returned False for %s tag=%d (conf=%.2f)",
                best_module.name,
                structure.type_id.name,
                structure.tag,
                best_idea.confidence,
                frame=self.state.game_loop,
            )

    async def _emit_building_commentary(self) -> None:
        """Send chat commentary for notable building decisions this tick."""
        pending = getattr(self, "_pending_building_commentary", [])
        self._pending_building_commentary = []
        for structure, idea in pending:
            if idea.confidence > 0.7:
                action_str = _building_idea_summary(idea, structure)
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

# ---- Outside the ManifestorBot class, at module level ----

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
