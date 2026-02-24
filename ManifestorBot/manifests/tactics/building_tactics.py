"""
Concrete Building Tactic Modules — Zerg edition.

Three modules cover the three building action categories:

    ZergWorkerProductionTactic   — Hatcheries queue Drones when under-saturated.
    ZergArmyProductionTactic     — Larva-producing structures train army units
                                   when the strategy is aggressive enough.
    ZergUpgradeResearchTactic    — Lair/Spire/etc. research priority upgrades.
    ZergRallyTactic              — Hatcheries / Lairs set rally to army centroid
                                   when the current rally is stale or wrong.
    ZergGasWorkerTactic          — Ensures drones are assigned to gas buildings.

These are self-contained and registered in ManifestorBot._load_building_modules().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from ManifestorBot.construction import ConstructionQueue, ConstructionOrder

from ManifestorBot.manifests.tactics.building_base import (
    BuildingTacticModule,
    BuildingAction,
    BuildingIdea,
)

from ManifestorBot.logger import get_logger
log = get_logger()

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy
    from sc2.unit import Unit

# Supply cost for each trainable unit type.
# Zerg units cost 1 supply except where noted.
SUPPLY_COST: dict[UnitID, int] = {
    UnitID.QUEEN:        2,
    UnitID.ZERGLING:     1,  # but spawns 2 per egg — handled naturally by amount
    UnitID.BANELING:     0,  # morphed from zergling, uses zergling supply
    UnitID.ROACH:        2,
    UnitID.RAVAGER:      3,
    UnitID.HYDRALISK:    2,
    UnitID.LURKERMP:     3,
    UnitID.MUTALISK:     2,
    UnitID.CORRUPTOR:    2,
    UnitID.BROODLORD:    4,
    UnitID.ULTRALISK:    6,
    UnitID.INFESTOR:     2,
    UnitID.VIPER:        3,
    UnitID.SWARMHOSTMP:  3,
}
    


# Priority-ordered list of structures to build and their prerequisites.
# Each entry: (structure_type, prerequisite_structure_or_None, min_minerals)
_STRUCTURE_PRIORITY = [
    # (what_to_build,              requires_existing,           minerals)
    (UnitID.SPAWNINGPOOL,          None,                        200),
    (UnitID.EXTRACTOR,             UnitID.SPAWNINGPOOL,          75),
    (UnitID.ROACHWARREN,           UnitID.SPAWNINGPOOL,         150),
    (UnitID.EVOLUTIONCHAMBER,      UnitID.SPAWNINGPOOL,         75),
    (UnitID.HYDRALISKDEN,          UnitID.LAIR,                 100),
    (UnitID.SPIRE,                 UnitID.LAIR,                 200),
    (UnitID.BANELINGNEST,          UnitID.SPAWNINGPOOL,         100),
    (UnitID.INFESTATIONPIT,        UnitID.LAIR,                 100),
    (UnitID.ULTRALISKCAVERN,       UnitID.HIVE,                 150),
]

# Maximum number of each structure type we'll ever build. Any type not
# listed here defaults to 1. Hatcheries are handled by _maybe_expand()
# and are NOT in this map. Spore/Spine Crawlers are treated as units
# and bypass this system entirely.
# Extractors are also absent — their cap is dynamic (2 per base), see
# _max_for_structure().
_MAX_STRUCTURE_COUNT: dict[UnitID, int] = {
    UnitID.EVOLUTIONCHAMBER:  2,  # dual upgrade lanes (melee + ranged / armor)
    UnitID.NYDUSNETWORK:      3,
}
_DEFAULT_MAX_STRUCTURES: int = 1


def _max_for_structure(structure_type: UnitID, bot) -> int:
    """Return the max allowed count for a structure, with dynamic caps."""
    if structure_type == UnitID.EXTRACTOR:
        return len(bot.townhalls.ready) * 2
    return _MAX_STRUCTURE_COUNT.get(structure_type, _DEFAULT_MAX_STRUCTURES)

# ---------------------------------------------------------------------------
# 1. Worker Production
# ---------------------------------------------------------------------------

class ZergWorkerProductionTactic(BuildingTacticModule):
    """
    Queue a Drone whenever the hatchery is idle and we're under-saturated.

    Confidence is driven primarily by ``saturation_delta`` — how many
    additional workers could be usefully employed right now. Strategy
    engage bias pulls it down (aggressive strategies would rather train army).

    IMPORTANT: Drones are trained from LARVA, not from the Hatchery itself.
    This means the Hatchery stays "idle" (no .orders) even while drones are
    in production. Other modules (especially ZergQueenProductionTactic) must
    therefore win via confidence, not via the idle check.

    Max theoretical confidence: sat_sig(0.60) + econ_lag(0.27) + drag(0.0) = 0.87.
    ZergQueenProductionTactic is set to 0.95 base to reliably beat this ceiling.
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
            log.debug(
                "ZergWorkerProductionTactic: %s not ready (build_progress=%.2f)",
                building.type_id.name,
                building.build_progress,
                frame=bot.state.game_loop,
            )
            return False
        if not self._building_is_idle(building):
            log.debug(
                "ZergWorkerProductionTactic: %s not idle (orders=%s)",
                building.type_id.name,
                [o.ability.id.name for o in building.orders],
                frame=bot.state.game_loop,
            )
            return False
        if bot.current_strategy in self.blocked_strategies:
            return False
        # Hard supply / mineral gate before even scoring
        if bot.supply_left < 1:
            return False
        if bot.minerals < 50:
            return False
        if not bot.larva:
            log.debug(
                "ZergWorkerProductionTactic: no larva available near %s",
                building.type_id.name,
                frame=bot.state.game_loop,
            )
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
            log.debug(
                "ZergWorkerProductionTactic: fully saturated (delta=%.1f) — returning None",
                delta,
                frame=bot.state.game_loop,
            )
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

        log.debug(
            "ZergWorkerProductionTactic: confidence=%.3f (sat=%.3f econ_lag=%.3f drag=%.3f delta=%.1f)",
            confidence,
            sat_sig,
            evidence.get("economic_health_lag", 0.0),
            strategy_drag,
            delta,
            frame=bot.state.game_loop,
        )

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
# NOTE: BANELING is morphed from ZERGLING at the BANELINGNEST, not trained directly.
# However, in python-sc2 / Ares, BANELING training is issued to the Hatchery larva
# and requires a BanelingNest — it IS in the tech_requirement_progress check.
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

# Minimum army supply we always want, regardless of strategy.
# Below this threshold we ALWAYS push army production (confidence 0.85).
# This prevents the bot from sitting at 0 military indefinitely.
_MIN_ARMY_SUPPLY: int = 6

# Once we have _MIN_ARMY_SUPPLY, switch to composition-target-driven production.
# Base confidence added even when strategy is neutral (engage_bias=0).
# Prevents pure eco builds by ensuring some army gets made.
_ARMY_BASE_CONFIDENCE: float = 0.45

class ZergArmyProductionTactic(BuildingTacticModule):
    """
    Queue army units from idle hatcheries / lairs / hives when the strategy
    calls for aggression or when we simply have resources to spend.

    FIX: Added _MIN_ARMY_SUPPLY floor so we always build SOME military even
    on neutral strategies like STOCK_STANDARD. Previously engage_bias=0.0
    meant army confidence was nearly 0 unless we were losing badly.

    Also added composition-target awareness: if the strategy profile specifies
    unit ratios, we bias toward the under-represented types.
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
        train_type = self._pick_unit(building, bot, current_strategy, heuristics)
        if train_type is None:
            log.debug(
                "ZergArmyProductionTactic: no affordable/available unit for %s",
                building.type_id.name,
                frame=bot.state.game_loop,
            )
            return None  # can't afford or don't have tech for anything

        # --- EMERGENCY FLOOR: always build a minimum army ---
        # Count combat supply (exclude workers, queens, overlords)
        WORKER_AND_SUPPORT = {
            UnitID.DRONE, UnitID.QUEEN, UnitID.OVERLORD,
            UnitID.OVERSEER, UnitID.OVERLORDCOCOON,
        }
        combat_units = bot.units.exclude_type(WORKER_AND_SUPPORT)
        combat_supply = sum(SUPPLY_COST.get(u.type_id, 2) for u in combat_units)

        log.debug(
            "ZergArmyProductionTactic: combat_supply=%d min_floor=%d train_type=%s",
            combat_supply,
            _MIN_ARMY_SUPPLY,
            train_type.name,
            frame=bot.state.game_loop,
        )

        if combat_supply < _MIN_ARMY_SUPPLY:
            # Emergency mode: build army NOW regardless of strategy
            confidence = 0.85
            evidence["emergency_army_floor"] = 0.85
            evidence["combat_supply"] = combat_supply
            evidence["train_type"] = train_type.name
            log.info(
                "ZergArmyProductionTactic: EMERGENCY FLOOR — combat_supply=%d < %d, "
                "forcing %s (conf=%.2f)",
                combat_supply,
                _MIN_ARMY_SUPPLY,
                train_type.name,
                confidence,
                frame=bot.state.game_loop,
            )
        else:
            # --- Normal production scoring ---

            # Sub-signal: base confidence — always produce SOME army on neutral strategies
            confidence += _ARMY_BASE_CONFIDENCE
            evidence["army_base"] = _ARMY_BASE_CONFIDENCE

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

            # Sub-signal: spending efficiency — if we're mineral-floating, spend on army
            spend_eff = heuristics.spend_efficiency
            if spend_eff < 0.7:
                float_sig = (0.7 - spend_eff) * 0.3
                confidence += float_sig
                evidence["mineral_float_pressure"] = float_sig

            # Sub-signal: resource pressure manager (overbanking override)
            rp = getattr(bot, 'resource_pressure', None)
            if rp is not None:
                boost = rp.army_production_boost(bot)
                if boost > 0:
                    confidence += boost
                    evidence['mineral_pressure'] = round(boost, 3)
                if rp.is_panic_mode(bot):
                    confidence = max(confidence, 0.92)
                    evidence['panic_mode'] = 0.92

            log.debug(
                "ZergArmyProductionTactic: confidence=%.3f (base=%.2f agg=%.2f "
                "behind=%.2f float=%.2f mineral_pressure=%.3f) train=%s avr=%.2f",
                confidence,
                _ARMY_BASE_CONFIDENCE,
                agg_sig,
                evidence.get("army_value_behind", 0.0),
                evidence.get("mineral_float_pressure", 0.0),
                evidence.get("mineral_pressure", 0.0),
                train_type.name,
                avr,
                frame=bot.state.game_loop,
            )

        if confidence < 0.15:
            return None

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=train_type,
        )

    def _pick_unit(
        self,
        building: "Unit",
        bot: "ManifestorBot",
        current_strategy: "Strategy",
        heuristics: "HeuristicState",
    ) -> Optional[UnitID]:
        """
        Return the best unit type to train, preferring composition targets if available.

        FIX: Previously just returned the first affordable unit in priority order,
        which meant zerglings were always chosen over roaches even when we needed roaches.
        Now checks composition targets and picks the most-underrepresented affordable type.
        """
        profile = current_strategy.profile()
        target = profile.active_composition(heuristics.game_phase)

        # If we have a composition target, try to satisfy it
        if target and target.ratios:
            best_type = self._pick_by_composition(building, bot, target)
            if best_type is not None:
                log.debug(
                    "ZergArmyProductionTactic: composition-driven pick=%s",
                    best_type.name,
                    frame=bot.state.game_loop,
                )
                return best_type

        # Fallback: priority list
        for unit_type, structure_type in _ARMY_PRIORITY:
            if building.type_id != structure_type:
                continue
            if not bot.can_afford(unit_type):
                continue
            if bot.tech_requirement_progress(unit_type) < 1.0:
                log.debug(
                    "ZergArmyProductionTactic: %s tech not ready (%.2f)",
                    unit_type.name,
                    bot.tech_requirement_progress(unit_type),
                    frame=bot.state.game_loop,
                )
                continue
            return unit_type
        return None

    def _pick_by_composition(
        self,
        building: "Unit",
        bot: "ManifestorBot",
        target,
    ) -> Optional[UnitID]:
        """
        Pick the unit type that is most underrepresented vs the composition target.
        Only considers types that are affordable and have tech available.
        """
        WORKER_AND_SUPPORT = {
            UnitID.DRONE, UnitID.QUEEN, UnitID.OVERLORD,
            UnitID.OVERSEER, UnitID.OVERLORDCOCOON,
        }
        # Calculate current army supply per type
        total_combat_supply = 0
        supply_by_type: dict[UnitID, int] = {}
        for unit in bot.units.exclude_type(WORKER_AND_SUPPORT):
            cost = SUPPLY_COST.get(unit.type_id, 2)
            supply_by_type[unit.type_id] = supply_by_type.get(unit.type_id, 0) + cost
            total_combat_supply += cost

        if total_combat_supply == 0:
            total_combat_supply = 1  # avoid div-by-zero

        # Identify which hatchery types map to valid army units
        valid_structure_types = {
            UnitID.HATCHERY, UnitID.LAIR, UnitID.HIVE
        }
        if building.type_id not in valid_structure_types:
            return None

        # Find the most under-represented affordable type
        best_type = None
        best_deficit = -999.0

        for unit_type, ratio in target.ratios.items():
            if unit_type in WORKER_AND_SUPPORT:
                continue
            # Check this unit can be trained from a hatchery-class structure
            can_train = any(
                structure == building.type_id
                for u, structure in _ARMY_PRIORITY
                if u == unit_type
            )
            if not can_train:
                continue
            if not bot.can_afford(unit_type):
                continue
            if bot.tech_requirement_progress(unit_type) < 1.0:
                continue

            current_fraction = supply_by_type.get(unit_type, 0) / total_combat_supply
            deficit = ratio - current_fraction
            if deficit > best_deficit:
                best_deficit = deficit
                best_type = unit_type

        return best_type

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        result = self._execute_train(building, idea, bot)
        if result:
            log.info(
                "ZergArmyProductionTactic: trained %s from %s tag=%d "
                "(conf=%.2f minerals=%d vespene=%d supply_left=%d)",
                idea.train_type.name if idea.train_type else "?",
                building.type_id.name,
                building.tag,
                idea.confidence,
                bot.minerals,
                bot.vespene,
                bot.supply_left,
                frame=bot.state.game_loop,
            )
        else:
            log.warning(
                "ZergArmyProductionTactic: _execute_train returned False for %s from %s tag=%d "
                "(conf=%.2f minerals=%d vespene=%d supply_left=%d larva=%d)",
                idea.train_type.name if idea.train_type else "?",
                building.type_id.name,
                building.tag,
                idea.confidence,
                bot.minerals,
                bot.vespene,
                bot.supply_left,
                bot.larva.amount,
                frame=bot.state.game_loop,
            )
        return result


# ---------------------------------------------------------------------------
# 3. Upgrade Research
# ---------------------------------------------------------------------------

# Priority-ordered list of upgrades with the structures that research them.
_UPGRADE_PRIORITY: List[tuple[UpgradeId, UnitID]] = [
    (UpgradeId.ZERGLINGMOVEMENTSPEED,   UnitID.SPAWNINGPOOL),
    (UpgradeId.ZERGMELEEWEAPONSLEVEL1,  UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.ZERGGROUNDARMORSLEVEL1,  UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.GLIALRECONSTITUTION,     UnitID.ROACHWARREN),
    (UpgradeId.ZERGMELEEWEAPONSLEVEL2,  UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.ZERGGROUNDARMORSLEVEL2,  UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.CENTRIFICALHOOKS,        UnitID.BANELINGNEST),
    (UpgradeId.ZERGMELEEWEAPONSLEVEL3,  UnitID.EVOLUTIONCHAMBER),
    (UpgradeId.ZERGGROUNDARMORSLEVEL3,  UnitID.EVOLUTIONCHAMBER),
]


class ZergUpgradeResearchTactic(BuildingTacticModule):
    """
    Research priority upgrades from idle tech structures.
    """

    BUILDING_TYPES = frozenset({
        UnitID.SPAWNINGPOOL,
        UnitID.EVOLUTIONCHAMBER,
        UnitID.ROACHWARREN,
        UnitID.BANELINGNEST,
        UnitID.HYDRALISKDEN,
        UnitID.LURKERDENMP,
        UnitID.SPIRE,
        UnitID.ULTRALISKCAVERN,
    })

    def is_applicable(self, building: "Unit", bot: "ManifestorBot") -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            return False
        if not self._building_is_idle(building):
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
            log.debug(
                "ZergUpgradeResearchTactic: no applicable upgrade for %s",
                building.type_id.name,
                frame=bot.state.game_loop,
            )
            return None

        confidence = 0.75
        evidence = {"upgrade": upgrade.name}

        log.debug(
            "ZergUpgradeResearchTactic: %s → researching %s (conf=%.2f)",
            building.type_id.name,
            upgrade.name,
            confidence,
            frame=bot.state.game_loop,
        )

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.RESEARCH,
            confidence=confidence,
            evidence=evidence,
            upgrade=upgrade,
        )

    def _pick_upgrade(self, building: "Unit", bot: "ManifestorBot", counter_ctx: "CounterContext") -> Optional[UpgradeId]:
        for upgrade, structure_type in _UPGRADE_PRIORITY:
            if building.type_id != structure_type:
                continue
            if self._already_researched(upgrade, bot):
                continue
            if self._is_being_researched(upgrade, bot):
                continue
            if not self._can_afford_research(upgrade, bot):
                log.debug(
                    "ZergUpgradeResearchTactic: cannot afford %s (minerals=%d vespene=%d)",
                    upgrade.name,
                    bot.minerals,
                    bot.vespene,
                    frame=bot.state.game_loop,
                )
                continue
            return upgrade
        return None

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        result = self._execute_research(building, idea, bot)
        if result:
            log.info(
                "ZergUpgradeResearchTactic: started research %s from %s tag=%d",
                idea.upgrade.name if idea.upgrade else "?",
                building.type_id.name,
                building.tag,
                frame=bot.state.game_loop,
            )
        else:
            log.warning(
                "ZergUpgradeResearchTactic: _execute_research returned False for %s from %s tag=%d "
                "(minerals=%d vespene=%d)",
                idea.upgrade.name if idea.upgrade else "?",
                building.type_id.name,
                building.tag,
                bot.minerals,
                bot.vespene,
                frame=bot.state.game_loop,
            )
        return result


# ---------------------------------------------------------------------------
# 4. Rally Correction
# ---------------------------------------------------------------------------

_RALLY_STALE_DISTANCE = 5.0


class ZergRallyTactic(BuildingTacticModule):
    """
    Set or correct the rally point of Hatcheries / Lairs / Hives.
    """

    BUILDING_TYPES = frozenset({
        UnitID.HATCHERY,
        UnitID.LAIR,
        UnitID.HIVE,
    })

    @property
    def blocked_strategies(self):
        from ManifestorBot.manifests.strategy import Strategy
        return frozenset({Strategy.DRONE_ONLY_FORTRESS})

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

        log.debug(
            "ZergRallyTactic: %s tag=%d setting rally to (%.0f, %.0f) (%s)",
            building.type_id.name,
            building.tag,
            target.x,
            target.y,
            evidence["rally_drift"],
            frame=bot.state.game_loop,
        )

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
        army = bot.units.exclude_type({bot.worker_type, bot.supply_type})
        army_center = army.center if army else bot.start_location

        if current_strategy.is_aggressive():
            enemy_base = bot.enemy_start_locations[0]
            target = army_center.towards(enemy_base, 10)
        elif current_strategy.is_defensive():
            if bot.townhalls:
                target = bot.townhalls.closest_to(army_center).position
            else:
                target = bot.start_location
        else:
            target = army_center

        return target


# ---------------------------------------------------------------------------
# 4b. Hatchery Rebuild — replace destroyed townhalls
# ---------------------------------------------------------------------------

class ZergHatcheryRebuildTactic(BuildingTacticModule):
    """
    Detects destroyed hatcheries tracked in bot._lost_hatchery_positions and
    enqueues a replacement ConstructionOrder at the original location.

    Priority 90 > normal expand priority 80, so rebuilds beat new expansions
    when both are affordable.
    """

    BUILDING_TYPES = frozenset({UnitID.HATCHERY, UnitID.LAIR, UnitID.HIVE})

    def is_applicable(self, building, bot) -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            return False
        if not self._building_is_idle(building):
            return False
        return bool(getattr(bot, '_lost_hatchery_positions', []))

    def generate_idea(self, building, bot, heuristics, current_strategy, counter_ctx):
        lost = getattr(bot, '_lost_hatchery_positions', [])

        # Remove positions where a townhall already exists (rebuild complete)
        cleaned = [
            pos for pos in lost
            if not bot.townhalls.closer_than(5.0, pos)
        ]
        bot._lost_hatchery_positions = cleaned

        if not cleaned:
            return None

        if bot.minerals < 500:
            log.debug(
                "ZergHatcheryRebuildTactic: minerals=%d < 500 — waiting for funds",
                bot.minerals,
                frame=bot.state.game_loop,
            )
            return None

        if bot.construction_queue.count_active_of_type(UnitID.HATCHERY) > 0:
            log.debug(
                "ZergHatcheryRebuildTactic: hatchery already in queue — skipping",
                frame=bot.state.game_loop,
            )
            return None

        confidence = 0.85
        evidence = {'rebuild_lost_hatch': len(cleaned)}

        log.info(
            "ZergHatcheryRebuildTactic: %d lost position(s) — proposing rebuild (conf=%.2f)",
            len(cleaned),
            confidence,
            frame=bot.state.game_loop,
        )

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=UnitID.HATCHERY,
        )

    def execute(self, building, idea, bot) -> bool:
        lost = getattr(bot, '_lost_hatchery_positions', [])
        if not lost:
            return False

        lost_pos = lost.pop(0)
        bot._lost_hatchery_positions = lost

        order = ConstructionOrder(
            structure_type=UnitID.HATCHERY,
            base_location=lost_pos,
            priority=90,
            created_frame=bot.state.game_loop,
        )

        accepted = bot.construction_queue.enqueue(order)
        if accepted:
            log.game_event(
                "REBUILD_ENQUEUED",
                f"HATCHERY rebuild at {lost_pos}",
                frame=bot.state.game_loop,
            )
        else:
            log.warning(
                "ZergHatcheryRebuildTactic: queue rejected rebuild at %s — re-inserting",
                lost_pos,
                frame=bot.state.game_loop,
            )
            bot._lost_hatchery_positions.insert(0, lost_pos)

        return accepted


class ZergStructureBuildTactic(BuildingTacticModule):
    '''
    Decides when to build a new Zerg structure and enqueues a ConstructionOrder.

    FIX: Relaxed the prerequisite check for EXTRACTOR to also accept a *pending*
    Spawning Pool (not just a ready one). This means the Extractor build order is
    queued immediately after the pool starts, so gas workers can be assigned as
    soon as the extractor completes — matching normal Zerg macro timing.
    '''

    BUILDING_TYPES = frozenset({
        UnitID.HATCHERY,
        UnitID.LAIR,
        UnitID.HIVE,
    })

    def is_applicable(self, building, bot) -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            return False
        return True  # generate_idea handles all the dedup logic

    def generate_idea(self, building, bot, heuristics, current_strategy, counter_ctx):
        # ── Dynamic expansion gate ────────────────────────────────────
        # Check the strategy's composition curve for max_hatcheries at
        # the current game phase and queue a new expansion if we're below
        # the cap. This is the ONLY path that creates hatcheries after
        # the opening build order finishes.
        expansion_idea = self._maybe_expand(building, bot, heuristics, current_strategy)
        if expansion_idea is not None:
            return expansion_idea

        # ── Opening gate ──────────────────────────────────────────────
        # The Ares build order runner and our tactic system share a
        # single-frame state snapshot (_abilities_count_and_build_progress
        # is cached once per frame).  If both fire in the same frame they
        # both see already_pending == 0 and both dispatch a drone → duplicates.
        #
        # Guard: don't touch the _STRUCTURE_PRIORITY list until the Ares
        # opener is complete OR we're safely past its spawning pool window.
        # Supply 20 is comfortably past the pool step in all openings
        # (StandardOpener: 17, TurtleEco: 18, EarlyAggression: 12).
        # By supply 20 the opener's drone is already walking/morphing, so
        # bot.already_pending(SPAWNINGPOOL) >= 1 and our cap check blocks
        # any duplicate from our side.
        opening_done = (
            bot.build_order_runner.build_completed
            or bot.supply_used > 20
        )
        if not opening_done:
            return None

        for structure_type, prerequisite, min_minerals in _STRUCTURE_PRIORITY:
            # ── Hard cap on structure count ────────────────────────────
            max_allowed = _max_for_structure(structure_type, bot)
            existing_count = bot.structures(structure_type).amount
            # NOTE: Do NOT add bot.already_pending() here — it double-counts
            # buildings under construction that bot.structures() already
            # includes.  count_active_of_type covers PENDING + CLAIMED +
            # BUILDING orders in the construction queue, which is the only
            # additional source we need.
            pending_count = bot.construction_queue.count_active_of_type(structure_type)
            total = existing_count + pending_count

            if total >= max_allowed:
                log.debug(
                    "ZergStructureBuildTactic: %s at cap (%d/%d) — skipping",
                    structure_type.name, total, max_allowed,
                    frame=bot.state.game_loop,
                )
                continue

            # Prerequisite check — with special handling for EXTRACTOR:
            # We queue the extractor as soon as the pool is *pending* (not just ready),
            # so gas workers are assigned the moment both buildings complete.
            if prerequisite:
                if structure_type == UnitID.EXTRACTOR:
                    # Accept pending OR ready pool
                    pool_exists = bool(bot.structures(prerequisite))
                    pool_pending = bot.already_pending(prerequisite) > 0
                    if not pool_exists and not pool_pending:
                        log.debug(
                            "ZergStructureBuildTactic: EXTRACTOR skipped — pool not started yet",
                            frame=bot.state.game_loop,
                        )
                        continue
                else:
                    if not bot.structures(prerequisite).ready:
                        log.debug(
                            "ZergStructureBuildTactic: %s skipped — %s not ready",
                            structure_type.name,
                            prerequisite.name,
                            frame=bot.state.game_loop,
                        )
                        continue

            if bot.minerals < min_minerals:
                log.debug(
                    "ZergStructureBuildTactic: %s skipped — minerals=%d < %d",
                    structure_type.name,
                    bot.minerals,
                    min_minerals,
                    frame=bot.state.game_loop,
                )
                continue

            if bot.tech_requirement_progress(structure_type) < 0.85:
                log.debug(
                    "ZergStructureBuildTactic: %s tech not ready (%.2f)",
                    structure_type.name,
                    bot.tech_requirement_progress(structure_type),
                    frame=bot.state.game_loop,
                )
                continue

            confidence = 0.80
            evidence = {"structure_priority": 0.80, "type": structure_type.name}

            profile = current_strategy.profile()
            agg = profile.engage_bias * 0.10
            confidence += agg
            evidence["strategy_agg"] = round(agg, 3)

            log.info(
                "ZergStructureBuildTactic: queuing %s (conf=%.2f minerals=%d)",
                structure_type.name,
                confidence,
                bot.minerals,
                frame=bot.state.game_loop,
            )

            return BuildingIdea(
                building_module=self,
                action=BuildingAction.TRAIN,
                confidence=confidence,
                evidence=evidence,
                train_type=structure_type,
            )

        return None

    def execute(self, building, idea, bot) -> bool:
        if idea.train_type is None:
            log.error(
                "ZergStructureBuildTactic: execute called with None train_type",
                frame=bot.state.game_loop,
            )
            return False

        base_location = building.position

        try:
            base_location = bot.placement_resolver.best_base_for(bot, idea.train_type)
        except Exception as exc:
            log.error(
                "ZergStructureBuildTactic: failed to find best base for %s: %s — using building position",
                idea.train_type.name,
                exc,
                frame=bot.state.game_loop,
            )

        order = ConstructionOrder(
            structure_type=idea.train_type,
            base_location=base_location,
            priority=80,
            created_frame=bot.state.game_loop,
        )

        accepted = bot.construction_queue.enqueue(order)
        if accepted:
            log.game_event(
                "BUILD_ENQUEUED",
                f"{idea.train_type.name} near {base_location}",
                frame=bot.state.game_loop,
            )
        else:
            log.warning(
                "ZergStructureBuildTactic: construction_queue rejected %s (already present?)",
                idea.train_type.name,
                frame=bot.state.game_loop,
            )
        return accepted

    # ------------------------------------------------------------------
    # Dynamic expansion helper
    # ------------------------------------------------------------------

    # Minimum minerals before we consider expanding.
    _EXPAND_MIN_MINERALS: int = 300

    # Cooldown: don't queue another hatchery if one is already building or
    # was queued fewer than this many frames ago.
    _EXPAND_COOLDOWN_FRAMES: int = 224  # ~10 seconds at Faster speed

    def _maybe_expand(
        self, building, bot, heuristics, current_strategy
    ) -> "BuildingIdea | None":
        """
        Queue a new Hatchery expansion when we're below the strategy's
        ``max_hatcheries`` cap for the current game phase.

        Guards:
        - There is a free expansion location on the map.
        - We have enough minerals.
        - No hatchery is already pending / under construction / recently queued.
        - Current base count (including in-progress) < max_hatcheries.
        - At least one existing base is approaching mineral saturation, so the
          expansion actually serves a purpose.
        """
        profile = current_strategy.profile()
        comp = profile.active_composition(heuristics.game_phase)
        if comp is None:
            return None

        max_hatch = comp.max_hatcheries

        # Count all townhalls — ready + morphing — so we don't double-queue.
        # bot.townhalls includes under-construction hatcheries.
        # bot.already_pending(HATCHERY) catches any drone that has a build
        # order but hasn't started morphing yet (worker en route).
        hatch_morphing = bot.already_pending(UnitID.HATCHERY)
        current_bases = bot.townhalls.amount + hatch_morphing
        if current_bases >= max_hatch:
            return None

        # Hard gate: don't queue another expansion while ANY hatchery order
        # is in flight — our queue OR a drone actively en route to build one.
        # This prevents the MorphTracker DONE→prune gap from causing double queues.
        if bot.construction_queue.count_active_of_type(UnitID.HATCHERY) > 0:
            return None
        if hatch_morphing > 0:
            return None

        # Mineral gate.
        if bot.minerals < self._EXPAND_MIN_MINERALS:
            return None

        # Drone count gate: require at least 10 workers per existing base
        # before taking a new one.  No point expanding into empty hatches.
        min_drones_to_expand = bot.townhalls.ready.amount * 10
        if len(bot.workers) < min_drones_to_expand:
            log.debug(
                "ZergStructureBuildTactic: expansion skipped — workers %d < %d needed",
                len(bot.workers), min_drones_to_expand,
                frame=bot.state.game_loop,
            )
            return None

        # Saturation trigger: use surplus_harvesters (accounts for gas workers)
        # so gas workers don't inflate the "assigned" count against a
        # mineral-only ideal.  Expand when bases are ≥75% saturated on average,
        # or when minerals are banking hard (≥600).
        total_surplus = sum(th.surplus_harvesters for th in bot.townhalls.ready)
        total_ideal   = sum(th.ideal_harvesters   for th in bot.townhalls.ready)
        # surplus_harvesters = assigned - ideal.  Negative = under-saturated.
        # avg_saturation: 1.0 = perfect, >1.0 = over, <1.0 = under.
        avg_saturation = (total_ideal + total_surplus) / max(total_ideal, 1)
        banking_hard = bot.minerals > 600

        if avg_saturation < 0.75 and not banking_hard:
            log.debug(
                "ZergStructureBuildTactic: expansion skipped — saturation %.0f%% < 75%%",
                avg_saturation * 100,
                frame=bot.state.game_loop,
            )
            return None

        # Verify a free expansion slot exists.
        taken = {th.position for th in bot.townhalls}
        free_expansion = None
        for exp in bot.expansion_locations_list:
            if exp not in taken:
                free_expansion = exp
                break
        if free_expansion is None:
            return None

        confidence = 0.75
        evidence = {
            "expansion": True,
            "current_bases": current_bases,
            "max_hatcheries": max_hatch,
            "avg_saturation": round(avg_saturation, 2),
        }

        log.info(
            "ZergStructureBuildTactic: queuing HATCHERY expansion #%d/%d "
            "(sat=%.0f%% minerals=%d)",
            current_bases + 1, max_hatch,
            avg_saturation * 100, bot.minerals,
            frame=bot.state.game_loop,
        )

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=UnitID.HATCHERY,
        )


# ---------------------------------------------------------------------------
# 5. Queen Production
# ---------------------------------------------------------------------------

# Maximum queens we'll train per hatchery count.
# One queen per hatchery is the standard macro target.
_MAX_QUEENS_PER_HATCHERY: float = 1.0

# Minimum queens we always want regardless of hatchery count.
_MIN_QUEENS: int = 1

# ─── FIX: Confidence must be set above the worker drone ceiling ──────────────
#
# ZergWorkerProductionTactic can reach up to 0.87 confidence in early game:
#   sat_sig  (max 0.60) + econ_lag (max 0.27) + strategy_drag (~0.0) = 0.87
#
# The old base of 0.85 let drones win the confidence race every tick because
# the Hatchery appears idle even while a drone trains from larva (.orders
# stays empty — the command lands on the larva unit, not the Hatchery).
#
# Set base to 0.97 so queens always win when the quota is unmet, regardless
# of economic stress. Queens are infrastructure (inject, AA, transfuse) — they
# must never be indefinitely deferred for drones.
# ─────────────────────────────────────────────────────────────────────────────
_QUEEN_BASE_CONFIDENCE: float = 0.97


class ZergQueenProductionTactic(BuildingTacticModule):
    """
    Train a Queen from an idle Hatchery/Lair/Hive when we're under the
    per-hatchery queen quota.

    Queens are NOT produced from larva — they are trained directly from the
    Hatchery via TRAIN_QUEEN. This module bypasses the generic larva-based
    _execute_train path and calls building.train() directly.

    Priority
    --------
    Confidence is set to 0.97 base so queen production always beats the drone
    tactic (max ~0.87) when queens are genuinely needed. A bot with zero queens
    cannot inject, cannot transfuse, and has no AA — queens are infrastructure,
    not army.

    The old value of 0.85 was a latent bug: ZergWorkerProductionTactic trains
    drones from LARVA, which means the Hatchery's .orders list stays empty even
    while drones are in production. Both worker and queen tactics see an "idle"
    Hatchery and compete purely on confidence every 20-frame tick — and with
    high saturation demand + poor economic health the drones won 0.87 > 0.85.

    Quota
    -----
    We want at least one queen per hatchery up to a configurable cap.
    Once the quota is met, this module returns None and defers to drones/army.
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
            log.debug(
                "ZergQueenProductionTactic: %s not ready (build_progress=%.2f) — skipping",
                building.type_id.name,
                building.build_progress,
                frame=bot.state.game_loop,
            )
            return False
        if not self._building_is_idle(building):
            log.debug(
                "ZergQueenProductionTactic: %s not idle (orders=%s) — skipping",
                building.type_id.name,
                [o.ability.id.name for o in building.orders],
                frame=bot.state.game_loop,
            )
            return False
        # Queens require a Spawning Pool
        pool_ready = bool(bot.structures(UnitID.SPAWNINGPOOL).ready)
        if not pool_ready:
            log.debug(
                "ZergQueenProductionTactic: SpawningPool not ready — skipping",
                frame=bot.state.game_loop,
            )
            return False
        if bot.supply_left < 2:
            log.debug(
                "ZergQueenProductionTactic: supply_left=%d < 2 — skipping",
                bot.supply_left,
                frame=bot.state.game_loop,
            )
            return False
        if bot.minerals < 150:
            log.debug(
                "ZergQueenProductionTactic: minerals=%d < 150 — skipping",
                bot.minerals,
                frame=bot.state.game_loop,
            )
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
        hatchery_count = bot.structures.filter(
            lambda s: s.type_id in self.BUILDING_TYPES and s.is_ready
        ).amount
        if hatchery_count == 0:
            log.debug(
                "ZergQueenProductionTactic: no ready hatcheries — returning None",
                frame=bot.state.game_loop,
            )
            return None

        # Count existing queens + pending queen eggs
        queen_count = bot.units(UnitID.QUEEN).amount
        pending_queens = bot.already_pending(UnitID.QUEEN)
        effective_queens = queen_count + pending_queens

        quota = max(_MIN_QUEENS, int(hatchery_count * _MAX_QUEENS_PER_HATCHERY))

        log.debug(
            "ZergQueenProductionTactic: queens=%d pending=%.1f effective=%.1f quota=%d hatcheries=%d minerals=%d supply_left=%d",
            queen_count,
            pending_queens,
            effective_queens,
            quota,
            hatchery_count,
            bot.minerals,
            bot.supply_left,
            frame=bot.state.game_loop,
        )

        if effective_queens >= quota:
            log.debug(
                "ZergQueenProductionTactic: quota met (effective=%.1f >= quota=%d) — returning None",
                effective_queens,
                quota,
                frame=bot.state.game_loop,
            )
            return None  # quota met — don't train more

        deficit = quota - effective_queens
        # High base confidence so queens ALWAYS beat the drone tactic.
        # See _QUEEN_BASE_CONFIDENCE comment above for the full explanation.
        confidence = min(1.0, _QUEEN_BASE_CONFIDENCE + deficit * 0.03)
        evidence = {
            "queen_deficit": deficit,
            "queen_count": queen_count,
            "pending_queens": pending_queens,
            "quota": quota,
            "base_confidence": _QUEEN_BASE_CONFIDENCE,
        }

        log.debug(
            "ZergQueenProductionTactic: generating TRAIN_QUEEN idea (conf=%.3f deficit=%.1f)",
            confidence,
            deficit,
            frame=bot.state.game_loop,
        )

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=UnitID.QUEEN,
        )

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        """
        Queens are trained directly from the Hatchery, NOT from larva.
        We must call building.train() here instead of the generic _execute_train,
        which would incorrectly look for larva since QUEEN is not in DOES_NOT_USE_LARVA.
        """
        if idea.train_type != UnitID.QUEEN:
            log.error(
                "ZergQueenProductionTactic.execute called with non-queen type: %s",
                idea.train_type,
                frame=bot.state.game_loop,
            )
            return False

        if not bot.can_afford(UnitID.QUEEN):
            log.warning(
                "ZergQueenProductionTactic: cannot afford Queen at execute time "
                "(minerals=%d vespene=%d supply_left=%d) — idea should not have been generated",
                bot.minerals,
                bot.vespene,
                bot.supply_left,
                frame=bot.state.game_loop,
            )
            return False

        result = building.train(UnitID.QUEEN)
        log.info(
            "ZergQueenProductionTactic: issued TRAIN_QUEEN from tag=%d type=%s (result=%s) "
            "[queens=%d pending=%.1f minerals=%d supply_left=%d]",
            building.tag,
            building.type_id.name,
            result,
            bot.units(UnitID.QUEEN).amount,
            bot.already_pending(UnitID.QUEEN),
            bot.minerals,
            bot.supply_left,
            frame=bot.state.game_loop,
        )
        return bool(result)


# ---------------------------------------------------------------------------
# 6. Overlord Production
# ---------------------------------------------------------------------------

# Train an Overlord when supply headroom drops below this threshold.
_OVERLORD_SUPPLY_THRESHOLD: int = 4  #TODO: build this into the strategy profiles

# Maximum overlords pending at once — don't over-train.
_MAX_PENDING_OVERLORDS: int = 1


class ZergOverlordProductionTactic(BuildingTacticModule):
    """
    Train Overlords from larva (via an idle Hatchery/Lair/Hive as the anchor)
    when supply is running short.

    Without this module the bot will supply-block as soon as the opening
    build order's overlords are consumed.

    Design
    ------
    The Hatchery is used as the anchor structure for the idle check, but
    actual training uses a nearby larva (Overlords ARE larva-produced).
    Confidence is very high (0.95) when supply_left <= threshold, ensuring
    this wins over virtually everything else when we're close to supply-blocked.
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
        if bot.minerals < 100:
            return False
        # Need nearby larva to train an Overlord
        if not bot.larva.closer_than(15, building.position):
            log.debug(
                "ZergOverlordProductionTactic: no larva near %s — skipping",
                building.type_id.name,
                frame=bot.state.game_loop,
            )
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
        supply_left = bot.supply_left
        pending_overlords = bot.already_pending(UnitID.OVERLORD)

        log.debug(
            "ZergOverlordProductionTactic: supply_left=%d pending_overlords=%.1f threshold=%d",
            supply_left,
            pending_overlords,
            _OVERLORD_SUPPLY_THRESHOLD,
            frame=bot.state.game_loop,
        )

        # Don't queue if we already have enough overlords en route
        if pending_overlords >= _MAX_PENDING_OVERLORDS:
            return None

        if supply_left > _OVERLORD_SUPPLY_THRESHOLD:
            return None  # not supply-pressured yet

        deficit = _OVERLORD_SUPPLY_THRESHOLD - supply_left
        # Scale confidence: critical at 0 supply left, moderate near threshold
        confidence = min(1.0, 0.70 + deficit * 0.08)
        evidence = {
            "supply_left": supply_left,
            "pending_overlords": pending_overlords,
            "supply_deficit_vs_threshold": deficit,
        }

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=UnitID.OVERLORD,
        )

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        """Overlords are produced from larva — use the generic _execute_train path."""
        if idea.train_type != UnitID.OVERLORD:
            log.error(
                "ZergOverlordProductionTactic.execute called with non-overlord type: %s",
                idea.train_type,
                frame=bot.state.game_loop,
            )
            return False

        result = self._execute_train(building, idea, bot)
        if result:
            log.info(
                "ZergOverlordProductionTactic: trained Overlord (supply_left=%d)",
                bot.supply_left,
                frame=bot.state.game_loop,
            )
        else:
            log.warning(
                "ZergOverlordProductionTactic: failed to train Overlord "
                "(supply_left=%d minerals=%d larva_near=%d)",
                bot.supply_left,
                bot.minerals,
                bot.larva.closer_than(15, building.position).amount,
                frame=bot.state.game_loop,
            )
        return result


# ---------------------------------------------------------------------------
# 7. Tech Morph (Hatchery → Lair → Hive)
# ---------------------------------------------------------------------------

class ZergTechMorphTactic(BuildingTacticModule):
    """
    Morph a Hatchery into a Lair, or a Lair into a Hive.

    Only one Lair and one Hive are needed — the rest of the townhalls
    stay as Hatcheries.  The morph ties up the building for ~71 seconds
    but larva production and usage continue, so the real cost is just
    blocking Queen production on that one townhall during the morph.

    Timing gates:
      Lair — Spawning Pool ready, 2+ townhalls, 150 min / 100 gas
      Hive — Infestation Pit ready, 3+ townhalls, 200 min / 150 gas
    """

    BUILDING_TYPES = frozenset({
        UnitID.HATCHERY,
        UnitID.LAIR,
    })

    def is_applicable(self, building, bot) -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            return False
        if not self._building_is_idle(building):
            return False
        return True

    def generate_idea(self, building, bot, heuristics, current_strategy, counter_ctx):
        # ── Hatchery → Lair ───────────────────────────────────────────
        if building.type_id == UnitID.HATCHERY:
            return self._consider_lair(building, bot)

        # ── Lair → Hive ──────────────────────────────────────────────
        if building.type_id == UnitID.LAIR:
            return self._consider_hive(building, bot)

        return None

    def _consider_lair(self, building, bot):
        # Already have a Lair or Hive (or one morphing)?
        lair_count = (
            bot.structures(UnitID.LAIR).amount
            + bot.structures(UnitID.HIVE).amount
            + bot.already_pending(UnitID.LAIR)
        )
        if lair_count > 0:
            return None

        # Prerequisite: Spawning Pool ready
        if not bot.structures(UnitID.SPAWNINGPOOL).ready:
            return None

        # Don't morph your only hatchery
        if len(bot.townhalls.ready) < 2:
            return None

        # Resource gate
        if bot.minerals < 150 or bot.vespene < 100:
            return None

        confidence = 0.80
        evidence = {
            "tech_morph": "LAIR",
            "townhalls": len(bot.townhalls.ready),
        }

        log.info(
            "ZergTechMorphTactic: Lair morph ready (minerals=%d gas=%d townhalls=%d)",
            bot.minerals, bot.vespene, len(bot.townhalls.ready),
            frame=bot.state.game_loop,
        )

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=UnitID.LAIR,
        )

    def _consider_hive(self, building, bot):
        # Already have a Hive (or one morphing)?
        hive_count = (
            bot.structures(UnitID.HIVE).amount
            + bot.already_pending(UnitID.HIVE)
        )
        if hive_count > 0:
            return None

        # Prerequisite: Infestation Pit ready
        if not bot.structures(UnitID.INFESTATIONPIT).ready:
            return None

        # Want a solid economy before committing to Hive tech
        if len(bot.townhalls.ready) < 3:
            return None

        # Resource gate
        if bot.minerals < 200 or bot.vespene < 150:
            return None

        confidence = 0.80
        evidence = {
            "tech_morph": "HIVE",
            "townhalls": len(bot.townhalls.ready),
        }

        log.info(
            "ZergTechMorphTactic: Hive morph ready (minerals=%d gas=%d townhalls=%d)",
            bot.minerals, bot.vespene, len(bot.townhalls.ready),
            frame=bot.state.game_loop,
        )

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=UnitID.HIVE,
        )

    def execute(self, building, idea, bot) -> bool:
        """
        Issue the morph command directly on the building.

        Cannot use _execute_train() because that routes through larva for
        Zerg units.  Lair/Hive are structure morphs issued on the building.
        """
        if idea.train_type not in (UnitID.LAIR, UnitID.HIVE):
            log.error(
                "ZergTechMorphTactic: unexpected train_type %s",
                idea.train_type,
                frame=bot.state.game_loop,
            )
            return False

        result = building.train(idea.train_type)
        if result:
            log.game_event(
                "TECH_MORPH",
                f"{building.type_id.name} → {idea.train_type.name} tag={building.tag}",
                frame=bot.state.game_loop,
            )
        else:
            log.warning(
                "ZergTechMorphTactic: morph command failed for %s → %s",
                building.type_id.name,
                idea.train_type.name,
                frame=bot.state.game_loop,
            )
        return result


# ---------------------------------------------------------------------------
# 8. Gas Worker Assignment
# ---------------------------------------------------------------------------

class ZergGasWorkerTactic(BuildingTacticModule):
    """
    Explicitly assign idle or mineral-mining drones to gas buildings that are
    under-saturated.

    FIX: Ares mediator handles long-term gas saturation, but in practice the bot
    was never collecting gas because:
    1. The Extractor was queued too late (fixed in ZergStructureBuildTactic above).
    2. No module was actively redirecting drones to gas when Ares wasn't doing it.

    This module fires every ~40 frames and redirects drones to under-saturated
    gas buildings directly — a belt-AND-suspenders approach that guarantees gas
    collection even if the mediator doesn't handle it automatically.

    Uses the EXTRACTOR / ASSIMILATOR / REFINERY as the anchor building.
    Confidence is high (0.90) — getting gas is critical for any tech units.
    """

    BUILDING_TYPES = frozenset({
        UnitID.EXTRACTOR,
        UnitID.EXTRACTORRICH,
    })

    def is_applicable(self, building: "Unit", bot: "ManifestorBot") -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            log.debug(
                "ZergGasWorkerTactic: extractor tag=%d not ready (progress=%.2f)",
                building.tag,
                building.build_progress,
                frame=bot.state.game_loop,
            )
            return False
        # Only act if under-saturated
        if building.assigned_harvesters >= building.ideal_harvesters:
            log.debug(
                "ZergGasWorkerTactic: extractor tag=%d saturated (%d/%d) — skipping",
                building.tag,
                building.assigned_harvesters,
                building.ideal_harvesters,
                frame=bot.state.game_loop,
            )
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
        deficit = building.ideal_harvesters - building.assigned_harvesters
        log.info(
            "ZergGasWorkerTactic: extractor tag=%d needs %d more workers (%d/%d)",
            building.tag,
            deficit,
            building.assigned_harvesters,
            building.ideal_harvesters,
            frame=bot.state.game_loop,
        )

        confidence = 0.90
        evidence = {
            "gas_deficit": deficit,
            "assigned": building.assigned_harvesters,
            "ideal": building.ideal_harvesters,
        }
        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,   # reusing TRAIN slot; execute() handles the actual redirect
            confidence=confidence,
            evidence=evidence,
            train_type=None,  # not training a unit — redirecting a worker
        )

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        """
        Find a nearby mineral-mining drone and redirect it to this gas building.
        """
        deficit = building.ideal_harvesters - building.assigned_harvesters
        if deficit <= 0:
            return False

        redirected = 0
        for _ in range(deficit):
            drone = self._find_mineral_drone(building, bot)
            if drone is None:
                log.warning(
                    "ZergGasWorkerTactic: no available mineral drone to redirect to extractor tag=%d",
                    building.tag,
                    frame=bot.state.game_loop,
                )
                break
            drone.gather(building)
            log.info(
                "ZergGasWorkerTactic: redirected drone tag=%d → extractor tag=%d",
                drone.tag,
                building.tag,
                frame=bot.state.game_loop,
            )
            redirected += 1

        return redirected > 0

    def _find_mineral_drone(self, gas_building: "Unit", bot: "ManifestorBot"):
        """
        Find the closest drone that is currently mining minerals (not gas, not building).
        """
        from sc2.ids.ability_id import AbilityId
        MINERAL_GATHER = {
            AbilityId.HARVEST_GATHER,
            AbilityId.HARVEST_GATHER_DRONE,
        }
        candidates = []
        for drone in bot.workers:
            if not drone.orders:
                continue
            order = drone.orders[0]
            if order.ability.id not in MINERAL_GATHER:
                continue
            # Make sure it's gathering a mineral (not a gas building)
            # The target is a mineral field or a gas geyser — check unit type
            target_tag = getattr(order, 'target', None)
            if target_tag is None:
                continue
            # Accept any mineral-gathering drone near this extractor's base
            candidates.append(drone)

        if not candidates:
            return None

        # Pick the closest one to the gas building to minimize travel time
        return min(candidates, key=lambda d: d.distance_to(gas_building.position))
