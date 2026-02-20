"""
Concrete Building Tactic Modules — Zerg edition.

Three modules cover the three building action categories:

    ZergWorkerProductionTactic   — Hatcheries queue Drones when under-saturated.
    ZergArmyProductionTactic     — Larva-producing structures train army units
                                   when the strategy is aggressive enough.
    ZergUpgradeResearchTactic    — Lair/Spire/etc. research priority upgrades.
    ZergRallyTactic              — Hatcheries / Lairs set rally to army centroid
                                   when the current rally is stale or wrong.

These are self-contained and registered in ManifestorBot._load_building_modules().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2

from ManifestorBot.manifests.tactics.building_base import (
    BuildingTacticModule,
    BuildingAction,
    BuildingIdea,
)

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy
    from sc2.unit import Unit


# ---------------------------------------------------------------------------
# 1. Worker Production
# ---------------------------------------------------------------------------

class ZergWorkerProductionTactic(BuildingTacticModule):
    """
    Queue a Drone whenever the hatchery is idle and we're under-saturated.

    Confidence is driven primarily by ``saturation_delta`` — how many
    additional workers could be usefully employed right now. Strategy
    engage bias pulls it down (aggressive strategies would rather train army).
    """

    BUILDING_TYPES = frozenset({
        UnitID.HATCHERY,
        UnitID.LAIR,
        UnitID.HIVE,
    })

    def is_applicable(self, building: "Unit", bot: "ManifestorBot") -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            return False
        if not self._building_is_idle(building):
            return False
        if bot.current_strategy in self.blocked_strategies:
            return False
        # Hard supply / mineral gate before even scoring
        if bot.supply_left < 1:
            return False
        if bot.minerals < 50:
            return False
        if not bot.larva:          # ← add this
            return False
        return True

    def generate_idea(
        self,
        building: "Unit",
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
        counter_ctx: "CounterContext",
    ) -> Optional[BuildingIdea]:
        confidence = 0.0
        evidence: dict = {}

        # Sub-signal: saturation delta — how many workers are still needed
        delta = heuristics.saturation_delta
        if delta <= 0:
            return None  # fully saturated, don't queue more drones
        sat_sig = min(0.6, delta * 0.12)
        confidence += sat_sig
        evidence["saturation_delta"] = sat_sig

        # Sub-signal: economic health — lagging economy should drone harder
        econ = heuristics.economic_health
        if econ < 0.9:
            econ_sig = (0.9 - econ) * 0.3
            confidence += econ_sig
            evidence["economic_health_lag"] = econ_sig

        # Sub-signal: strategy profile — aggressive strategies deprioritise drones
        profile = current_strategy.profile()
        strategy_drag = profile.engage_bias * -0.15  # positive engage_bias → less drone pressure
        confidence += strategy_drag
        evidence["strategy_engage_drag"] = strategy_drag

        if confidence < 0.15:
            return None

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=UnitID.DRONE,
        )

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        return self._execute_train(building, idea, bot)


# ---------------------------------------------------------------------------
# 2. Army Production
# ---------------------------------------------------------------------------

# Priority-ordered list of Zerg army units and the structures that make them.
# The first affordable type from this list is queued.
_ARMY_PRIORITY: List[tuple[UnitID, UnitID]] = [
    # (unit_to_train,  producing_structure_type)
    (UnitID.ZERGLING,    UnitID.HATCHERY),
    (UnitID.ZERGLING,    UnitID.LAIR),
    (UnitID.ZERGLING,    UnitID.HIVE),
    (UnitID.ROACH,       UnitID.HATCHERY),
    (UnitID.ROACH,       UnitID.LAIR),
    (UnitID.ROACH,       UnitID.HIVE),
    (UnitID.HYDRALISK,   UnitID.LAIR),
    (UnitID.HYDRALISK,   UnitID.HIVE),
    (UnitID.MUTALISK,    UnitID.LAIR),
    (UnitID.ULTRALISK,   UnitID.HIVE),
]


class ZergArmyProductionTactic(BuildingTacticModule):
    """
    Queue army units from idle hatcheries / lairs / hives when the strategy
    calls for aggression or when we simply have resources to spend.

    The unit type chosen follows the priority list above, filtered by what
    technology is actually available right now. Only fires when confidence
    clearly beats the 0.40 suppression threshold to avoid spamming units
    that aren't actually needed yet.
    """

    BUILDING_TYPES = frozenset({
        UnitID.HATCHERY,
        UnitID.LAIR,
        UnitID.HIVE,
    })

    def is_applicable(self, building: "Unit", bot: "ManifestorBot") -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            return False
        if not self._building_is_idle(building):
            return False
        if bot.current_strategy in self.blocked_strategies:
            return False
        if bot.supply_left < 2:
            return False
        return True

    def generate_idea(
        self,
        building: "Unit",
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
        counter_ctx: "CounterContext",
    ) -> Optional[BuildingIdea]:
        confidence = 0.0
        evidence: dict = {}

        # Determine which unit we'd want to train
        train_type = self._pick_unit(building, bot)
        if train_type is None:
            return None  # can't afford or don't have tech for anything

        # Sub-signal: strategy aggression
        profile = current_strategy.profile()
        agg_sig = profile.engage_bias * 0.4  # aggressive = more army
        confidence += agg_sig
        evidence["strategy_engage_bias"] = agg_sig

        # Sub-signal: army value ratio — train more if we're behind
        avr = heuristics.army_value_ratio
        if avr < 0.8:
            behind_sig = (0.8 - avr) * 0.5
            confidence += behind_sig
            evidence["army_value_behind"] = behind_sig

        # Sub-signal: aggression dial (composite 0–100)
        dial_sig = (heuristics.aggression_dial - 50.0) / 200.0  # -0.25 to +0.25
        confidence += dial_sig
        evidence["aggression_dial"] = dial_sig

        # Sub-signal: spend efficiency — we shouldn't bank resources
        if heuristics.spend_efficiency < 0.5 and bot.minerals > 300:
            spend_sig = 0.15
            confidence += spend_sig
            evidence["resource_float"] = spend_sig

        if confidence < 0.15:
            return None

        # Counter-play bonus: big bump if the unit we'd train is a prescribed counter
        if train_type in counter_ctx.priority_train_types:
            counter_sig = counter_ctx.production_bonus
            confidence += counter_sig
            evidence["counter_play_bonus"] = counter_sig
            
        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=train_type,
        )

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        return self._execute_train(building, idea, bot)

    def _pick_unit(self, building: "Unit", bot: "ManifestorBot") -> Optional[UnitID]:
        
        """
        Walk the priority list and return the first unit type that:
          - Can be produced from this building type
          - We can afford right now
          - We have the tech prerequisite for (tech_requirement check)
        """
        for unit_type, struct_type in _ARMY_PRIORITY:
            if building.type_id != struct_type:
                continue
            if not self._can_afford_train(unit_type, bot):
                continue
            # Quick tech check via bot.tech_requirement_progress
            req = bot.tech_requirement_progress(unit_type)
            if req < 1.0:
                continue
            return unit_type
        return None


# ---------------------------------------------------------------------------
# 3. Upgrade Research
# ---------------------------------------------------------------------------

# Priority-ordered upgrades with the structure type that researches them.
_UPGRADE_PRIORITY: List[tuple[UpgradeId, UnitID]] = [
    # Economy / speed first
    (UpgradeId.ZERGLINGMOVEMENTSPEED, UnitID.SPAWNINGPOOL),
    (UpgradeId.OVERLORDSPEED,         UnitID.HATCHERY),      # any hatch/lair/hive
    (UpgradeId.GLIALRECONSTITUTION,   UnitID.ROACHWARREN),
    (UpgradeId.TUNNELINGCLAWS,        UnitID.ROACHWARREN),
    (UpgradeId.EVOLVEGROOVEDSPINES,    UnitID.ROACHWARREN),   # Ravager
    (UpgradeId.ZERGMELEEWEAPONSLEVEL1, UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.ZERGGROUNDARMORSLEVEL1, UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.ZERGMISSILEWEAPONSLEVEL1, UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.ZERGMELEEWEAPONSLEVEL2, UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.ZERGGROUNDARMORSLEVEL2, UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.ZERGMELEEWEAPONSLEVEL3, UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.ZERGGROUNDARMORSLEVEL3, UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.HYDRALISKSPEED,        UnitID.HYDRALISKDEN),
    (UpgradeId.EVOLVEGROOVEDSPINES,   UnitID.HYDRALISKDEN),
    (UpgradeId.CHITINOUSPLATING,      UnitID.ULTRALISKCAVERN),
    (UpgradeId.ANABOLICSYNTHESIS,     UnitID.ULTRALISKCAVERN),
]

_UPGRADE_STRUCT_TYPES = frozenset(s for _, s in _UPGRADE_PRIORITY)


class ZergUpgradeResearchTactic(BuildingTacticModule):
    """
    Trigger the highest-priority unresearched upgrade from an idle research building.

    Confidence is always high (0.8) once the affordability gate passes, because
    upgrades are unconditionally good. Strategy can nudge this: defensive
    strategies nudge toward armor, aggressive toward weapons (future extension).
    """

    BUILDING_TYPES = _UPGRADE_STRUCT_TYPES

    def is_applicable(self, building: "Unit", bot: "ManifestorBot") -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            return False
        if not self._building_is_idle(building):
            return False
        if bot.current_strategy in self.blocked_strategies:
            return False
        return True

    def generate_idea(
        self,
        building: "Unit",
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
        counter_ctx: "CounterContext",
    ) -> Optional[BuildingIdea]:
        upgrade = self._pick_upgrade(building, bot, counter_ctx)
        if upgrade is None:
            return None

        confidence = 0.8  # upgrades are almost always worth doing
        evidence = {"base_upgrade_priority": 0.8}

        # Strategy modifier: aggressive strategies value attack upgrades more
        profile = current_strategy.profile()
        strategy_mod = profile.engage_bias * 0.1
        confidence += strategy_mod
        evidence["strategy_bias"] = strategy_mod

        # Cap to 1.0
        confidence = min(1.0, confidence)

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.RESEARCH,
            confidence=confidence,
            evidence=evidence,
            upgrade=upgrade,
        )

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        return self._execute_research(building, idea, bot)

    def _pick_upgrade(self, building: "Unit", bot: "ManifestorBot", counter_ctx: "CounterContext") -> Optional[UpgradeId]:
        # Walk counter-priority upgrades first if they're available from this building
        if counter_ctx:
            for upgrade in counter_ctx.priority_upgrades:
                if self._upgrade_available_here(upgrade, building, bot):
                    return upgrade
        # Fall back to the static priority list        
        """Return the first upgrade in priority order that's unresearched and affordable."""
        for upgrade, struct_type in _UPGRADE_PRIORITY:
            if building.type_id != struct_type:
                continue
            if self._already_researched(upgrade, bot):
                continue
            if self._is_being_researched(upgrade, bot):
                continue
            if not self._can_afford_research(upgrade, bot):
                continue
            return upgrade
        return None


# ---------------------------------------------------------------------------
# 4. Rally Point Correction
# ---------------------------------------------------------------------------

# How far (in game units) a rally point is allowed to be from the army centroid
# before we consider it stale and worth correcting.
_RALLY_STALE_DISTANCE = 20.0


class ZergRallyTactic(BuildingTacticModule):
    """
    Update the rally point of a Hatchery / Lair / Hive when it's pointing
    somewhere that no longer makes sense — specifically, when the army
    centroid has moved significantly away from the current rally target.

    This is a low-confidence idea so it doesn't fire constantly — only when
    the gap is large enough to be meaningful (>20 units). Aggressive strategies
    rally to a more forward position; defensive strategies rally back toward
    the main base.

    Implementation note: python-sc2 doesn't expose the current rally point
    through the Unit API directly. We work around this by tracking the last
    rally we set (via ``bot._building_rally_cache``) and comparing that to
    the current army centroid. On the first step (no cache) we always set it.
    """

    BUILDING_TYPES = frozenset({
        UnitID.HATCHERY,
        UnitID.LAIR,
        UnitID.HIVE,
    })

    def is_applicable(self, building: "Unit", bot: "ManifestorBot") -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            return False
        # Rally can be set even while the building is busy — it's non-disruptive.
        if bot.current_strategy in self.blocked_strategies:
            return False
        return True

    def generate_idea(
        self,
        building: "Unit",
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
        counter_ctx: "CounterContext",
    ) -> Optional[BuildingIdea]:
        target = self._compute_rally_target(bot, heuristics, current_strategy)

        # Check how stale the existing rally is
        cache: dict = getattr(bot, "_building_rally_cache", {})
        last_rally = cache.get(building.tag)

        if last_rally is not None:
            dist = target.distance_to(last_rally)
            if dist < _RALLY_STALE_DISTANCE:
                return None  # Close enough — don't bother updating

        confidence = 0.55  # medium confidence — rally correction is useful but not urgent
        evidence = {"rally_drift": "stale" if last_rally else "initial"}

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.SET_RALLY,
            confidence=confidence,
            evidence=evidence,
            rally_point=target,
        )

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        success = self._execute_rally(building, idea, bot)
        if success and idea.rally_point is not None:
            # Update cache so we don't spam this
            if not hasattr(bot, "_building_rally_cache"):
                bot._building_rally_cache = {}
            bot._building_rally_cache[building.tag] = idea.rally_point
        return success

    def _compute_rally_target(
        self,
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> "Point2":
        """
        Pick the rally destination based on strategy and army position.

        Aggressive strategies: rally toward a point between our army and the
            enemy start — units pour into the fight immediately.
        Defensive strategies: rally to the nearest townhall — keep new units
            home until enough gather for a counterattack.
        Balanced: rally to the army centroid.
        """
        army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
        army_center = army.center if army else bot.start_location

        if current_strategy.is_aggressive():
            # Blend army centroid toward enemy base
            enemy_base = bot.enemy_start_locations[0]
            target = army_center.towards(enemy_base, 10)
        elif current_strategy.is_defensive():
            # Rally to the nearest townhall
            if bot.townhalls:
                target = bot.townhalls.closest_to(army_center).position
            else:
                target = bot.start_location
        else:
            target = army_center

        return target
