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

        if confidence < 0.15:
            return None

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=confidence,
            evidence=evidence,
            train_type=train_type,
        )

    def _pick_unit(self, building: "Unit", bot: "ManifestorBot") -> Optional[UnitID]:
        """Return the first affordable unit type from the priority list."""
        for unit_type, structure_type in _ARMY_PRIORITY:
            if building.type_id != structure_type:
                continue
            if not bot.can_afford(unit_type):
                continue
            if bot.tech_requirement_progress(unit_type) < 1.0:
                continue
            return unit_type
        return None

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        return self._execute_train(building, idea, bot)


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
            return None

        confidence = 0.75
        evidence = {"upgrade": upgrade.name}

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
                continue
            return upgrade
        return None

    def execute(self, building: "Unit", idea: BuildingIdea, bot: "ManifestorBot") -> bool:
        return self._execute_research(building, idea, bot)


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


class ZergStructureBuildTactic(BuildingTacticModule):
    '''
    Decides when to build a new Zerg structure and enqueues a ConstructionOrder.
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
        for structure_type, prerequisite, min_minerals in _STRUCTURE_PRIORITY:
            if bot.structures(structure_type).amount > 0:
                continue
            if bot.already_pending(structure_type) > 0:
                continue
            if bot.construction_queue.count_active_of_type(structure_type) > 0:
                continue

            if prerequisite and not bot.structures(prerequisite).ready:
                continue

            if bot.minerals < min_minerals:
                continue

            if bot.tech_requirement_progress(structure_type) < 0.85:
                continue

            confidence = 0.80
            evidence = {"structure_priority": 0.80, "type": structure_type.name}

            profile = current_strategy.profile()
            agg = profile.engage_bias * 0.10
            confidence += agg
            evidence["strategy_agg"] = round(agg, 3)

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
            return False

        base_location = building.position

        try:
            base_location = bot.placement_resolver.best_base_for(bot, idea.train_type)
        except Exception as exc:
            log.error(
                "ZergStructureBuildTactic: failed to find best base for %s: %s",
                idea.train_type,
                exc,
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
        return accepted

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
