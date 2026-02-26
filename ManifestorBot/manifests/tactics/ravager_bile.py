"""
ravager_bile.py — Corrosive Bile firing and incoming bile dodge.

CorrosiveBileTactic
-------------------
Fires the ravager's EFFECT_CORROSIVEBILE ability (range 9) at the
highest-priority target in range.  Priority order:

  1. High-value enemy buildings (command centres, production, tech)
  2. Slow / stationary / burrowed units (sieged tanks, lurkers, etc.)
  3. Any non-flying enemy unit in range (direct damage is still good)

Per-unit cooldown tracking ensures we fire again as soon as the ~7-second
cooldown expires without wasting ticks polling already-fired ravagers.

Friendly-fire guard: if any friendly unit is within FRIENDLY_FIRE_RADIUS of
the chosen target the shot is skipped — bile hits friendlies too.

Confidence:
  vs high-value building  → 0.82  (high; buildings can't dodge)
  vs slow / burrowed unit → 0.74
  vs any unit             → 0.66

DodgeBileTactic
---------------
Detects incoming enemy bile shots via bot.state.effects
(EffectId.RAVAGERCORROSIVEBILECP).  Ares' GridManager already marks active
bile positions on the ground avoidance grid each frame, so dodging is simply
KeepUnitSafe with mediator.get_ground_avoidance_grid — no custom pathfinding
needed.

Fires on all non-flying, non-structure units that are within
DODGE_ALERT_RADIUS tiles of any active bile landing zone.  Confidence 0.90
so it overrides most other decisions (attack, rally, patrol) except
ExtractorShield (0.92) and BaseDefense (0.85-0.95).

Bile positions are cached per frame on the tactic instance so we only
scan bot.state.effects once regardless of how many units trigger the check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet, List, Optional, Set

from sc2.ids.ability_id import AbilityId
from sc2.ids.effect_id import EffectId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2

from ares.behaviors.combat.individual import KeepUnitSafe

from ManifestorBot.abilities.ability import Ability, AbilityContext
from ManifestorBot.abilities.ability_registry import ability_registry
from ManifestorBot.manifests.tactics.base import TacticIdea, TacticModule
from ManifestorBot.logger import get_logger

if TYPE_CHECKING:
    from sc2.unit import Unit
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy

log = get_logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BILE_RANGE:           float = 9.0    # ravager ability range (tiles)
BILE_COOLDOWN:        int   = 160    # frames between shots (~7.2 s at fast speed)
BILE_RADIUS:          float = 0.25   # impact radius

# Trigger dodge when a bile centre is within this many tiles of the unit.
# Bile radius is 0.25; units need a full body-width of clearance beyond that.
DODGE_ALERT_RADIUS:   float = 1.5

# Don't fire bile if a friendly is this close to the intended target.
FRIENDLY_FIRE_RADIUS: float = 0.8

# ---------------------------------------------------------------------------
# Priority sets for target selection
# ---------------------------------------------------------------------------

# Enemy structures worth prioritising — can't move, high mineral/gas value.
_HIGH_VALUE_STRUCTURES: FrozenSet[UnitID] = frozenset({
    UnitID.COMMANDCENTER, UnitID.ORBITALCOMMAND, UnitID.PLANETARYFORTRESS,
    UnitID.NEXUS,
    UnitID.HATCHERY, UnitID.LAIR, UnitID.HIVE,
    UnitID.BARRACKS, UnitID.FACTORY, UnitID.STARPORT,
    UnitID.GATEWAY, UnitID.ROBOTICSFACILITY, UnitID.STARGATE,
    UnitID.SPAWNINGPOOL, UnitID.ROACHWARREN, UnitID.HYDRALISKDEN,
    UnitID.BANELINGNEST, UnitID.INFESTATIONPIT,
    UnitID.BUNKER, UnitID.PHOTONCANNON,
})

# Enemy unit types that can't (or barely) move — maximum bile value.
_PRIORITY_UNIT_TYPES: FrozenSet[UnitID] = frozenset({
    UnitID.SIEGETANKSIEGED,
    UnitID.LIBERATORAG,          # Liberator in siege mode
    UnitID.LURKERMPBURROWED,
    UnitID.WIDOWMINEBURROWED,
    UnitID.THOR,
    UnitID.ULTRALISK,
    UnitID.BROODLORD,
})

# Unit types excluded from "army" when measuring safe distance for friendly fire.
_SUPPLY_TYPES: FrozenSet[UnitID] = frozenset({
    UnitID.OVERLORD, UnitID.OVERSEER, UnitID.OVERLORDTRANSPORT,
})


# ---------------------------------------------------------------------------
# Ability: corrosive bile
# ---------------------------------------------------------------------------

class CorrosiveBileAbility(Ability):
    """Issues EFFECT_CORROSIVEBILE at the position carried in context."""

    UNIT_TYPES: Set[UnitID] = {UnitID.RAVAGER}
    GOAL:       str = "corrosive_bile"
    priority:   int = 80   # above general combat (70), below emergency (90+)

    def can_use(self, unit: "Unit", context: AbilityContext,
                bot: "ManifestorBot") -> bool:
        return context.target_position is not None

    def execute(self, unit: "Unit", context: AbilityContext,
                bot: "ManifestorBot") -> bool:
        unit(AbilityId.EFFECT_CORROSIVEBILE, context.target_position)
        context.ability_used = self.name
        context.command_issued = True
        log.debug(
            "CorrosiveBileAbility: ravager %d fires bile at %s",
            unit.tag, context.target_position,
            frame=bot.state.game_loop,
        )
        return True


def register_bile_abilities() -> None:
    """Register bile ability into the global registry.  Call once from on_start."""
    ability_registry.register(UnitID.RAVAGER, CorrosiveBileAbility())
    log.info("CorrosiveBileAbility registered for RAVAGER")


# ---------------------------------------------------------------------------
# Tactic 1: CorrosiveBileTactic
# ---------------------------------------------------------------------------

class CorrosiveBileTactic(TacticModule):
    """
    Fires corrosive bile at the best in-range target whenever the cooldown
    has elapsed.

    Target priority (in order):
      1. High-value enemy structures (can't dodge; every hit counts)
      2. Slow / stationary / burrowed enemy units
      3. Any non-flying enemy unit in range
    """

    name = "CorrosiveBileTactic"
    is_group_tactic = False

    def __init__(self) -> None:
        super().__init__()
        # tag → frame when bile was last issued for this ravager
        self._last_fired: dict[int, int] = {}

    def is_applicable(self, unit: "Unit", bot: "ManifestorBot") -> bool:
        return unit.type_id == UnitID.RAVAGER and not unit.is_burrowed

    def generate_idea(
        self,
        unit: "Unit",
        bot:  "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        frame = bot.state.game_loop

        # Cooldown guard
        if frame - self._last_fired.get(unit.tag, 0) < BILE_COOLDOWN:
            return None

        target, confidence = self._pick_target(unit, bot)
        if target is None:
            return None

        # Record the shot so we don't spam the same ravager this frame.
        self._last_fired[unit.tag] = frame

        evidence = {
            "target_tag":  getattr(target, "tag", "pos"),
            "target_type": getattr(target.type_id, "name", "?") if hasattr(target, "type_id") else "pos",
            "confidence":  round(confidence, 3),
        }

        context = AbilityContext(
            goal="corrosive_bile",
            target_position=target.position,
            confidence=confidence,
        )

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=target,
            context=context,
        )

    def create_behavior(self, unit: "Unit", idea: TacticIdea,
                        bot: "ManifestorBot"):
        # Execution goes through the ability registry (CorrosiveBileAbility).
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_target(self, unit: "Unit",
                     bot: "ManifestorBot") -> tuple[Optional["Unit"], float]:
        """Return (best_target, confidence) or (None, 0)."""

        enemies_in_range = [
            e for e in bot.enemy_units
            if unit.position.distance_to(e.position) <= BILE_RANGE
        ]
        if not enemies_in_range:
            return None, 0.0

        # --- Pass 1: high-value structures ---
        priority_structures = [
            e for e in enemies_in_range
            if e.is_structure and e.type_id in _HIGH_VALUE_STRUCTURES
        ]
        if priority_structures:
            target = min(priority_structures,
                         key=lambda s: unit.position.distance_to(s.position))
            if not self._friendly_near(target.position, bot):
                return target, 0.82

        # --- Pass 2: all enemy structures ---
        any_structures = [e for e in enemies_in_range if e.is_structure]
        if any_structures:
            target = min(any_structures,
                         key=lambda s: unit.position.distance_to(s.position))
            if not self._friendly_near(target.position, bot):
                return target, 0.78

        # --- Pass 3: slow / burrowed / priority unit types ---
        slow_units = [
            e for e in enemies_in_range
            if not e.is_flying and (e.is_burrowed or e.type_id in _PRIORITY_UNIT_TYPES)
        ]
        if slow_units:
            target = min(slow_units,
                         key=lambda e: unit.position.distance_to(e.position))
            if not self._friendly_near(target.position, bot):
                return target, 0.74

        # --- Pass 4: any non-flying enemy ---
        ground_enemies = [e for e in enemies_in_range if not e.is_flying]
        if ground_enemies:
            target = min(ground_enemies,
                         key=lambda e: unit.position.distance_to(e.position))
            if not self._friendly_near(target.position, bot):
                return target, 0.66

        return None, 0.0

    @staticmethod
    def _friendly_near(pos: Point2, bot: "ManifestorBot") -> bool:
        """True if any friendly unit (except supply units) is very close to pos."""
        return any(
            u.position.distance_to(pos) < FRIENDLY_FIRE_RADIUS
            for u in bot.units
            if u.type_id not in _SUPPLY_TYPES and not u.is_structure
        )


# ---------------------------------------------------------------------------
# Tactic 2: DodgeBileTactic
# ---------------------------------------------------------------------------

class DodgeBileTactic(TacticModule):
    """
    Moves any ground unit out of the path of an incoming enemy bile shot.

    Detection: scans bot.state.effects each frame for
    EffectId.RAVAGERCORROSIVEBILECP.  Each active bile has a set of Point2
    positions; any unit within DODGE_ALERT_RADIUS of one triggers this tactic.

    Execution: KeepUnitSafe with the ground avoidance grid.  Ares' GridManager
    already marks bile positions on that grid, so the unit is automatically
    pathed to the nearest safe tile.

    Confidence: 0.90 — high enough to override attack, rally, and patrol, but
    below ExtractorShield (0.92) and emergency defence (0.95).
    """

    name = "DodgeBileTactic"
    is_group_tactic = False

    def __init__(self) -> None:
        super().__init__()
        # Per-frame cache so bot.state.effects is scanned once, not per-unit.
        self._cached_bile_positions: List[Point2] = []
        self._cache_frame: int = -1

    def is_applicable(self, unit: "Unit", bot: "ManifestorBot") -> bool:
        # Quick structural gate — no effect scanning here.
        return not unit.is_flying and not unit.is_structure

    def generate_idea(
        self,
        unit:             "Unit",
        bot:              "ManifestorBot",
        heuristics:       "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        bile_positions = self._get_bile_positions(bot)
        if not bile_positions:
            return None

        # Find the closest bile to this unit.
        nearest_dist = float("inf")
        nearest_pos: Optional[Point2] = None
        for pos in bile_positions:
            d = unit.position.distance_to(pos)
            if d < nearest_dist:
                nearest_dist = d
                nearest_pos  = pos

        if nearest_dist >= DODGE_ALERT_RADIUS:
            return None  # bile is not close enough to worry about

        evidence = {
            "bile_distance": round(nearest_dist, 2),
            "bile_position": str(nearest_pos),
        }

        return TacticIdea(
            tactic_module=self,
            confidence=0.90,
            evidence=evidence,
            target=nearest_pos,
        )

    def create_behavior(self, unit: "Unit", idea: TacticIdea,
                        bot: "ManifestorBot"):
        """Use Ares KeepUnitSafe with the avoidance grid (bile is already on it)."""
        try:
            grid = bot.mediator.get_ground_avoidance_grid
            return KeepUnitSafe(unit=unit, grid=grid)
        except Exception as exc:
            log.debug("DodgeBileTactic.create_behavior failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Bile position cache
    # ------------------------------------------------------------------

    def _get_bile_positions(self, bot: "ManifestorBot") -> List[Point2]:
        """Return all active enemy bile positions, cached for the current frame."""
        frame = bot.state.game_loop
        if frame == self._cache_frame:
            return self._cached_bile_positions

        self._cache_frame = frame
        self._cached_bile_positions = []
        try:
            for eff in bot.state.effects:
                if eff.id == EffectId.RAVAGERCORROSIVEBILECP and eff.is_enemy:
                    self._cached_bile_positions.extend(eff.positions)
        except Exception as exc:
            log.debug("DodgeBileTactic: failed to read effects — %s", exc)

        return self._cached_bile_positions
