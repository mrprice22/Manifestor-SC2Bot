class BanelingTargetAbility(Ability):
    """
    Issues attack commands for banelings targeting high-splash / low-HP groups.

    Priority logic (in order):
      1. Enemy worker clusters (low HP, economically devastating)
      2. Clumped bio / light units (high splash value — score by nearby enemies)
      3. Lowest HP enemy ground unit (finish off stragglers for chain detonation)
    """
    UNIT_TYPES: Set[UnitID] = {UnitID.BANELING}
    GOAL: str = "attack"
    priority: int = 70  # higher than generic attack abilities

    _WORKER_TYPES: frozenset = frozenset({
        UnitID.SCV, UnitID.PROBE, UnitID.DRONE,
    })
    _SPLASH_RADIUS = 2.2   # baneling explosion radius
    _CLUSTER_BONUS_WEIGHT = 0.4  # per additional enemy in splash range

    def can_use(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        return unit.type_id == UnitID.BANELING and not unit.is_burrowed

    def execute(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        enemies = bot.enemy_units.filter(
            lambda e: e.is_visible and not e.is_flying and e.can_be_attacked
        )
        if not enemies:
            return False

        best_target = self._pick_target(unit, enemies)
        if best_target is None:
            return False

        unit.attack(best_target)
        context.ability_used = self.name
        context.command_issued = True
        return True

    def _pick_target(self, baneling: Unit, enemies) -> Optional[Unit]:
        best_score = -1.0
        best = None
        for enemy in enemies:
            score = self._score_target(baneling, enemy, enemies)
            if score > best_score:
                best_score = score
                best = enemy
        return best

    def _score_target(self, baneling: Unit, target: Unit, all_enemies) -> float:
        score = 0.0

        # 1. Worker bonus — highest priority
        if target.type_id in self._WORKER_TYPES:
            score += 2.5

        # 2. Low HP bonus — prefer near-dead units (chain splash)
        hp_fraction = target.health / max(target.health_max, 1)
        score += (1.0 - hp_fraction) * 1.5   # 0→1.5 bonus as HP drops

        # 3. Splash cluster bonus — count nearby enemies within splash radius
        nearby = sum(
            1 for e in all_enemies
            if e.tag != target.tag
            and cy_distance_to_squared(target.position, e.position) <= self._SPLASH_RADIUS ** 2
        )
        score += nearby * self._CLUSTER_BONUS_WEIGHT

        # 4. Light armor bonus — banelings deal +15 to light
        if target.is_light:
            score += 0.5

        # 5. Proximity — slight bias toward already-close targets
        dist_sq = cy_distance_to_squared(baneling.position, target.position)
        score -= dist_sq * 0.002  # small penalty for distance

        return score