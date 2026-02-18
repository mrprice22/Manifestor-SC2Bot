class FlankTactic(TacticModule):
    def generate_idea(self, unit, bot, heuristics, strategy):
        # Decision logic: should we flank?
        enemy_army = bot.enemy_units
        our_army = bot.units.exclude_type(...)
        
        # Calculate flank angle and position
        flank_position = self._calculate_flank_position(
            our_army.center, 
            enemy_army.center
        )
        
        # Build confidence from sub-signals
        confidence = self._score_flank_viability(...)
        
        if confidence > 0.5:
            return TacticIdea(
                self, confidence, 
                evidence={...}, 
                target=flank_position
            )
        return None
        
    def create_behavior(self, unit, idea, bot):
        # Execution: use Ares behaviors to execute the flank
        maneuver = CombatManeuver()
        
        grid = bot.mediator.get_ground_grid
        
        # Move to flank position using safe pathing
        maneuver.add(PathUnitToTarget(
            unit=unit,
            target=idea.target,  # the flank position
            grid=grid
        ))
        
        # Once in position, stutter forward toward enemy
        maneuver.add(StutterGroupForward(
            units=[unit],
            target=self._find_flank_target(unit, bot)
        ))
        
        # Safety: if things go wrong, retreat
        maneuver.add(KeepUnitSafe(unit, grid))
        
        return maneuver