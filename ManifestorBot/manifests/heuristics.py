"""
Heuristic Manager - Stage 1 of the data flow.

Calculates all game-state signals every frame. Pure deterministic math.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId as UnitID

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot


@dataclass
class HeuristicState:
    """
    The complete heuristic snapshot for a single frame.
    
    This is the compressed representation of the game state that
    everything else operates on. ~200 dimensions when fully implemented.
    """
    
    # Combat heuristics
    momentum: float = 0.0           # Rolling score of kills, retreats, map control
    initiative: float = 0.0         # Who is choosing where fights happen
    threat_level: float = 0.0       # How dangerous is the enemy army to our bases
    tempo: float = 0.0              # Are we ahead or behind on timings
    army_cohesion: float = 0.0      # Are our units together or scattered
    
    # Economic heuristics
    economic_health: float = 0.0    # Our income vs opponent's estimated income
    spend_efficiency: float = 0.0   # Are we banking resources or spending them
    worker_safety_index: float = 0.0  # How exposed are our mineral lines
    saturation_delta: float = 0.0   # How many more workers could we use productively
    mined_out_bases: int = 0        # Owned townhalls with no minerals remaining
    
    # Map heuristics
    creep_coverage_pct: float = 0.0     # Percentage of map with active creep
    vision_dominance: float = 0.0       # Map tiles we can see vs opponent
    expansion_race_index: float = 0.0   # Our bases vs theirs
    choke_control: float = 0.0          # Do we own key map chokepoints

    # Spatial / pheromone signals
    threat_hotspot_proximity: float = 0.0   # how close is our army to the hottest threat cell?
    
    # Army heuristics
    army_value_ratio: float = 1.0       # Our army value / their army value
    upgrade_advantage: float = 0.0      # Our upgrade level - their upgrade level
    reinforcement_proximity: float = 0.0  # How long until next wave arrives
    
    # Meta heuristics
    opponent_reaction_speed: float = 0.5   # How quickly they responded to last action
    opponent_risk_profile: float = 0.5     # Are they playing greedy or safe
    strategy_confidence: float = 1.0       # How certain are we about their strategy
    
    # Composite signals
    aggression_dial: float = 50.0       # 0-100 composite aggression score
    game_phase: float = 0.0             # 0.0 = early, 1.0 = late
    

class HeuristicManager:
    """
    Calculates all heuristics from raw game state every frame.
    
    This is Stage 1 of the data flow pipeline.
    """
    
    def __init__(self, bot: 'ManifestorBot'):
        self.bot = bot
        self.current_state = HeuristicState()
        
        # Momentum tracking
        self.momentum_events: list = []  # (frame, value) tuples
        self.momentum_decay_frames: int = 336  # ~15 seconds
        
        # Last frame values for delta calculations
        self.last_our_army_value: float = 0.0
        self.last_enemy_army_value: float = 0.0
        
    def update(self, iteration: int) -> None:
        """Calculate all heuristics for this frame"""
        self._update_combat_heuristics()
        self._update_economic_heuristics()
        self._update_map_heuristics()
        self._update_army_heuristics()
        self._update_meta_heuristics()
        self._update_aggression_dial()
        self._update_game_phase()
        self._update_pheromone_signals()
        # After heuristics are calculated, layer in counter-play modifiers
        ctx = self.bot.scout_ledger.get_counter_context(self.bot.state.game_loop)
        # Modify the aggression dial based on counter prescriptions
        self.current_state.aggression_dial += ctx.engage_bias_mod * 10  # dial is 0-100
        self.current_state.aggression_dial = max(0, min(100, self.current_state.aggression_dial))
        
    def get_state(self) -> HeuristicState:
        """Get the current heuristic snapshot"""
        return self.current_state
        
    # ========== Combat Heuristics ==========    
    def _update_combat_heuristics(self) -> None:
        """Calculate momentum, initiative, threat, tempo, cohesion"""
        self._update_momentum()
        self._update_initiative()
        self._update_threat_level()
        self._update_tempo()
        self._update_army_cohesion()
        
    def _update_momentum(self) -> None:
        """
        Momentum is a rolling score of recent events.
        Positive momentum = we're winning engagements, gaining map control.
        Negative momentum = we're losing, retreating, taking damage.
        """
        current_frame = self.bot.state.game_loop
        
        # Decay old momentum events
        self.momentum_events = [
            (frame, value) for frame, value in self.momentum_events
            if current_frame - frame < self.momentum_decay_frames
        ]
        
        # Calculate momentum from recent events
        if self.momentum_events:
            # Apply time-weighted decay
            total_momentum = 0.0
            for frame, value in self.momentum_events:
                age = current_frame - frame
                decay_factor = 1.0 - (age / self.momentum_decay_frames)
                total_momentum += value * decay_factor
            self.current_state.momentum = total_momentum
        else:
            self.current_state.momentum = 0.0
            
        # Detect new momentum events this frame
        # (This is simplified - full implementation would track specific events)
        current_our_value = self._calculate_our_army_value()
        current_enemy_value = self._calculate_enemy_army_value()
        
        if self.last_our_army_value > 0:
            value_gained = current_our_value - self.last_our_army_value
            enemy_value_lost = self.last_enemy_army_value - current_enemy_value
            
            # If we gained value or enemy lost value, that's positive momentum
            if value_gained > 500 or enemy_value_lost > 500:
                momentum_delta = (value_gained + enemy_value_lost) / 1000.0
                self.momentum_events.append((current_frame, momentum_delta))
                
        self.last_our_army_value = current_our_value
        self.last_enemy_army_value = current_enemy_value
        
    def _update_initiative(self) -> None:
        """Who is choosing where fights happen"""
        # Simplified: if our army is closer to their bases than theirs is to ours
        if not self.bot.enemy_structures:
            self.current_state.initiative = 0.0
            return
            
        our_army = self.bot.units.exclude_type(
            {self.bot.worker_type, self.bot.supply_type}
        )
        enemy_army = self.bot.enemy_units
        
        if not our_army or not enemy_army:
            self.current_state.initiative = 0.0
            return
            
        our_centroid = our_army.center
        enemy_centroid = enemy_army.center
        
        their_main = self.bot.enemy_start_locations[0]
        our_main = self.bot.start_location
        
        our_dist_to_them = our_centroid.distance_to(their_main)
        their_dist_to_us = enemy_centroid.distance_to(our_main)
        
        # Normalize to -1.0 to 1.0
        if their_dist_to_us > 0:
            self.current_state.initiative = (their_dist_to_us - our_dist_to_them) / their_dist_to_us
        else:
            self.current_state.initiative = 0.0
            
    def _update_threat_level(self) -> None:
        """How dangerous is the enemy army to our economy"""
        if not self.bot.townhalls:
            self.current_state.threat_level = 1.0
            return
            
        enemy_army = self.bot.enemy_units
        if not enemy_army:
            self.current_state.threat_level = 0.0
            return
            
        closest_townhall = self.bot.townhalls.closest_to(enemy_army.center)
        distance = enemy_army.center.distance_to(closest_townhall.position)
        
        # Threat increases as they get closer
        # At 0 distance = 1.0, at 100 distance = ~0.0
        self.current_state.threat_level = max(0.0, 1.0 - (distance / 100.0))
        
    def _update_tempo(self) -> None:
        """Are we ahead or behind on expected timings"""
        # Simplified: compare our tech progress to game time
        # Positive = ahead, negative = behind
        # TODO: Implement proper timing benchmarks per strategy
        self.current_state.tempo = 0.0
        
    def _update_army_cohesion(self) -> None:
        """Are our units together or scattered"""
        our_army = self.bot.units.exclude_type(
            {self.bot.worker_type, self.bot.supply_type}
        )
        
        if len(our_army) < 3:
            self.current_state.army_cohesion = 1.0
            return
            
        centroid = our_army.center
        avg_distance = sum(u.distance_to(centroid) for u in our_army) / len(our_army)
        
        # Lower distance = higher cohesion
        # Normalize so 0 distance = 1.0, 50 distance = 0.0
        self.current_state.army_cohesion = max(0.0, 1.0 - (avg_distance / 50.0))
        
    # ========== Economic Heuristics ==========
    
    def _update_economic_heuristics(self) -> None:
        """Calculate economic health, spend efficiency, worker safety, saturation"""
        self._update_economic_health()
        self._update_spend_efficiency()
        self._update_worker_safety()
        self._update_saturation_delta()
        self._update_mined_out_bases()
        
    def _update_economic_health(self) -> None:
        """Our income vs opponent's estimated income"""
        our_workers = len(self.bot.workers)
        # Estimate enemy workers (TODO: use actual scouting data)
        enemy_workers = len(self.bot.enemy_workers) if self.bot.enemy_workers else our_workers * 0.8
        
        if enemy_workers > 0:
            self.current_state.economic_health = our_workers / enemy_workers
        else:
            self.current_state.economic_health = 1.0
            
    def _update_spend_efficiency(self) -> None:
        """Are we banking resources or spending them"""
        # High minerals/gas with low production = bad
        if self.bot.supply_left < 2:
            self.current_state.spend_efficiency = 0.0
            return
            
        total_resources = self.bot.minerals + self.bot.vespene
        
        # If we have a lot of resources, we should be spending them
        if total_resources > 1000:
            self.current_state.spend_efficiency = 0.3
        elif total_resources > 500:
            self.current_state.spend_efficiency = 0.7
        else:
            self.current_state.spend_efficiency = 1.0
            
    def _update_worker_safety(self) -> None:
        """How exposed are our mineral lines"""
        # Check if enemy units are near our townhalls
        if not self.bot.townhalls:
            self.current_state.worker_safety_index = 0.0
            return
            
        min_safety = 1.0
        for th in self.bot.townhalls:
            nearby_enemies = self.bot.enemy_units.closer_than(15, th.position)
            if nearby_enemies:
                # More enemies = less safe
                safety = max(0.0, 1.0 - (len(nearby_enemies) / 10.0))
                min_safety = min(min_safety, safety)
                
        self.current_state.worker_safety_index = min_safety
        
    def _update_saturation_delta(self) -> None:
        """How many more workers could we use productively.
        Uses actual mineral patch counts per base rather than a flat 16-per-
        townhall estimate.  Bases whose minerals have been mined out contribute
        nothing to the ideal count, which prevents the bot from over-producing
        drones it cannot employ.

        Also caps ideal_workers at the strategy's max_hatcheries and
        army_supply_target so drone production doesn't overshoot the intended
        supply split.
        """
        # Resolve the active composition target for capping
        profile = self.bot.current_strategy.profile()
        comp = profile.active_composition(self.current_state.game_phase)
        max_hatch = comp.max_hatcheries if comp is not None else 999

        ideal_workers = 0
        counted_bases = 0
        for th in self.bot.townhalls.ready:
            if counted_bases >= max_hatch:
                break  # don't count workers for bases beyond the strategy cap
            # Count mineral patches still alive within mining range of this TH.
            patches = self.bot.mineral_field.closer_than(10, th.position)
            # Standard saturation: 2 workers per mineral patch.
            ideal_workers += len(patches) * 2
            # Add gas slots (3 per ready extractor at this base).
            local_gas = self.bot.gas_buildings.ready.closer_than(10, th.position)
            for g in local_gas:
                ideal_workers += g.ideal_harvesters  # typically 3
            counted_bases += 1
        
    def _update_mined_out_bases(self) -> None:
        """Count ready townhalls that have no mineral patches remaining nearby."""
        count = 0
        for th in self.bot.townhalls.ready:
            if not self.bot.mineral_field.closer_than(10, th.position):
                count += 1
        self.current_state.mined_out_bases = count

    # ========== Map Heuristics ==========

    def _update_map_heuristics(self) -> None:
        """Calculate creep coverage, vision, expansions, choke control"""
        self._update_creep_coverage()
        self._update_vision_dominance()
        self._update_expansion_race()
        self._update_choke_control()
        
    def _update_creep_coverage(self) -> None:
        """Percentage of map with active creep"""
        # TODO: Use Ares grid manager for accurate creep calculation
        # For now, rough estimate based on tumors and time
        num_tumors = len(self.bot.structures.of_type(
            {UnitID.CREEPTUMOR, UnitID.CREEPTUMORBURROWED, UnitID.CREEPTUMORQUEEN}
        ))
        # Very rough heuristic
        self.current_state.creep_coverage_pct = min(1.0, num_tumors / 20.0)
        
    def _update_vision_dominance(self) -> None:
        """Map tiles we can see vs opponent"""
        # Simplified: just count our mobile units vs theirs
        our_mobile = len(self.bot.units)
        their_mobile = len(self.bot.enemy_units)
        total = our_mobile + their_mobile
        
        if total > 0:
            self.current_state.vision_dominance = our_mobile / total
        else:
            self.current_state.vision_dominance = 0.5
            
    def _update_expansion_race(self) -> None:
        """Our bases vs theirs"""
        our_bases = len(self.bot.townhalls)
        enemy_bases = len(self.bot.enemy_structures.of_type(
            {UnitID.HATCHERY, UnitID.LAIR, UnitID.HIVE,
             UnitID.COMMANDCENTER, UnitID.ORBITALCOMMAND, UnitID.PLANETARYFORTRESS,
             UnitID.NEXUS}
        ))
        
        total = our_bases + max(1, enemy_bases)
        self.current_state.expansion_race_index = our_bases / total
        
    def _update_choke_control(self) -> None:
        """Do we own key map chokepoints"""
        # TODO: Use Ares map analyzer to identify key chokes
        self.current_state.choke_control = 0.5

    def _update_pheromone_signals(self) -> None:
        pm = self.bot.pheromone_map
        if pm is None:
            return

        hotspot = pm.hottest_threat_point()
        if hotspot is None:
            self.current_state.threat_hotspot_proximity = 0.0
            return

        army = self.bot.units.exclude_type({self.bot.worker_type, self.bot.supply_type})
        if not army:
            self.current_state.threat_hotspot_proximity = 0.0
            return

        dist = army.center.distance_to(hotspot)
        # Normalize: 0 = very close (high signal), 1 = far away
        self.current_state.threat_hotspot_proximity = max(0.0, 1.0 - dist / 60.0)
        
    # ========== Army Heuristics ==========
    
    def _update_army_heuristics(self) -> None:
        """Calculate army value, upgrades, reinforcements"""
        self._update_army_value_ratio()
        self._update_upgrade_advantage()
        self._update_reinforcement_proximity()
        
    def _update_army_value_ratio(self) -> None:
        """Our army value / their army value"""
        our_value = self._calculate_our_army_value()
        enemy_value = self._calculate_enemy_army_value()
        
        if enemy_value > 0:
            self.current_state.army_value_ratio = our_value / enemy_value
        else:
            self.current_state.army_value_ratio = 2.0  # No enemy army = we dominate
            
    def _update_upgrade_advantage(self) -> None:
        """Our upgrade level - their upgrade level"""
        # Simplified: just count upgrades in progress + completed
        # TODO: Track specific upgrade types
        our_upgrades = len(self.bot.state.upgrades)
        # Estimate enemy upgrades (TODO: use actual scouting)
        enemy_upgrades = 0
        
        self.current_state.upgrade_advantage = our_upgrades - enemy_upgrades
        
    def _update_reinforcement_proximity(self) -> None:
        """How long until next wave arrives"""
        # Check eggs, warp prisms in transit, units rallying
        # Simplified: just check larva count
        if self.bot.larva:
            self.current_state.reinforcement_proximity = len(self.bot.larva) / 3.0
        else:
            self.current_state.reinforcement_proximity = 0.0
            
    # ========== Meta Heuristics ==========
    
    def _update_meta_heuristics(self) -> None:
        """Calculate opponent behavior estimates"""
        # These are placeholders - they'll be updated by the opponent modeling system
        pass
        
    # ========== Composite Signals ==========
    
    def _update_aggression_dial(self) -> None:
        """
        Composite 0-100 score that indicates how aggressive the bot should be.
        
        High aggression = press advantages, take fights, harass.
        Low aggression = defend, expand, tech up.
        """
        score = 50.0  # Start at neutral
        
        # Momentum strongly affects aggression
        score += self.current_state.momentum * 10.0
        
        # Army advantage encourages aggression
        if self.current_state.army_value_ratio > 1.2:
            score += 15.0
        elif self.current_state.army_value_ratio < 0.8:
            score -= 15.0
            
        # Economic lead allows more aggression
        if self.current_state.economic_health > 1.2:
            score += 10.0
        elif self.current_state.economic_health < 0.8:
            score -= 10.0
            
        # Threat to bases demands defensive posture
        score -= self.current_state.threat_level * 20.0
        
        # Clamp to 0-100
        self.current_state.aggression_dial = max(0.0, min(100.0, score))

    def _update_game_phase(self) -> None:
        """
        0.0 = early game  (pool/gate/rax, small armies, 1-2 bases)
        0.5 = mid game    (first tech units, 3-4 bases, upgrades rolling)
        1.0 = late game   (hive/templar/BC, 4+ bases, maxing supply)
        
        Intentionally a continuous float, not a discrete enum.
        The game doesn't snap between phases — it slides.
        """
        score = 0.0
        
        # Base count is the strongest phase signal
        bases = len(self.bot.townhalls)
        score += min(0.3, bases * 0.075)  # 4 bases = 0.3
        
        # Tech tier: Lair = 0.2, Hive = 0.4
        if self.bot.structures(UnitID.HIVE).ready:
            score += 0.4
        elif self.bot.structures(UnitID.LAIR).ready:
            score += 0.2
        
        # Upgrade count (each completed upgrade = small phase bump)
        upgrades = len(self.bot.state.upgrades)
        score += min(0.2, upgrades * 0.03)
        
        # Army supply as fraction of 200
        army_supply = self.bot.supply_used - len(self.bot.workers) - len(self.bot.structures)
        score += min(0.1, army_supply / 200.0)
        
        self.current_state.game_phase = min(1.0, score)
        
    # ========== Helper Methods ==========
    
    def _calculate_our_army_value(self) -> float:
        """Calculate total mineral+gas value of our army"""
        army = self.bot.units.exclude_type({self.bot.worker_type, self.bot.supply_type})
        total = 0.0
        for unit in army:
            cost = self.bot.cost_dict.get(unit.type_id)
            if cost:
                total += cost.minerals + cost.vespene
        return total
        
    def _calculate_enemy_army_value(self) -> float:
        """Calculate total mineral+gas value of enemy army"""
        army = self.bot.enemy_units
        total = 0.0
        for unit in army:
            cost = self.bot.cost_dict.get(unit.type_id)
            if cost:
                total += cost.minerals + cost.vespene
        return total
