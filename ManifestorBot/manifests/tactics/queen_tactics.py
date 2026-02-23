"""
Queen Tactic Module — Inject + Creep Spread

Queens are divided into 3 squads on first assignment:
  - INJECTORS  (~1/3): stay near hatcheries, inject larva
  - CREEP_LEFT (~1/3): spread creep in one lateral direction toward enemy
  - CREEP_RIGHT(~1/3): spread creep in the other lateral direction toward enemy

The directional split is based on spawn location:
  - Top/bottom spawn  → left queen spreads west, right queen spreads east
  - Left/right spawn  → left queen spreads north, right queen spreads south
In all cases the creep squads angle toward the enemy base rather than away.

Tumor behavior:
  All mature creep tumors (CREEPTUMORBURROWED with BUILD_CREEPTUMOR_TUMOR
  available) are handled here too — they spread toward the enemy start
  location automatically via TumorSpreadCreep.

"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit

from cython_extensions import cy_distance_to_squared
from cython_extensions.general_utils import cy_has_creep
from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)

from ManifestorBot.manifests.tactics.base import TacticIdea, TacticModule
from ManifestorBot.logger import get_logger

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy

log = get_logger()


# ---------------------------------------------------------------------------
# Squad assignment
# ---------------------------------------------------------------------------

class QueenRole(str, Enum):
    INJECTOR    = "INJECTOR"
    CREEP_LEFT  = "CREEP_LEFT"
    CREEP_RIGHT = "CREEP_RIGHT"


# Module-level registry: queen tag → QueenRole.
# Persists across ticks so assignment is stable.
_queen_roles: dict[int, QueenRole] = {}


def _assign_queen_roles(bot: "ManifestorBot") -> None:
    """
    Assign every unassigned queen a role.

    The first time we have ≥1 queens we snapshot all current queens and
    divide them into three buckets.  New queens that arrive later join the
    smallest bucket to keep the squads balanced.
    """
    queens = bot.units(UnitID.QUEEN).ready
    if not queens:
        return

    for queen in queens:
        if queen.tag in _queen_roles:
            continue  # already assigned

        # Count current squad sizes
        counts = {r: 0 for r in QueenRole}
        for role in _queen_roles.values():
            counts[role] += 1

        total = sum(counts.values())

        # First queen → injector. Then fill creep squads to balance.
        if counts[QueenRole.INJECTOR] == 0:
            role = QueenRole.INJECTOR
        elif counts[QueenRole.CREEP_LEFT] <= counts[QueenRole.CREEP_RIGHT]:
            # Bias: we want injectors to be ~1/3 of the force.
            # Once both creep squads are non-empty and injectors are at
            # parity, rebalance by total share.
            injector_share = counts[QueenRole.INJECTOR] / max(1, total + 1)
            if injector_share < 0.34:
                role = QueenRole.INJECTOR
            else:
                role = QueenRole.CREEP_LEFT
        else:
            injector_share = counts[QueenRole.INJECTOR] / max(1, total + 1)
            if injector_share < 0.34:
                role = QueenRole.INJECTOR
            else:
                role = QueenRole.CREEP_RIGHT

        _queen_roles[queen.tag] = role
        log.info(
            "QueenRole assigned: tag=%d → %s  [I=%d L=%d R=%d]",
            queen.tag,
            role.value,
            counts[QueenRole.INJECTOR] + (1 if role == QueenRole.INJECTOR else 0),
            counts[QueenRole.CREEP_LEFT] + (1 if role == QueenRole.CREEP_LEFT else 0),
            counts[QueenRole.CREEP_RIGHT] + (1 if role == QueenRole.CREEP_RIGHT else 0),
        )


def _creep_target_for_role(role: QueenRole, bot: "ManifestorBot") -> Point2:
    """
    Return the map position a creep squad should spread toward.

    The heuristic:
      - Determine whether we spawned in a top/bottom corner or a
        left/right corner by comparing x and y coordinates relative to
        the map centre.
      - CREEP_LEFT spreads in the direction that eventually angles toward
        the enemy; CREEP_RIGHT does the same on the other axis.

    In all cases both targets are offset away from the start location AND
    toward the enemy half of the map so creep marches toward the enemy
    rather than spreading uselessly behind the base.
    """
    start:  Point2 = bot.start_location
    enemy:  Point2 = bot.enemy_start_locations[0]
    centre: Point2 = bot.game_info.map_center

    # Unit vector from start toward enemy (used to lean targets forward)
    dx = enemy.x - start.x
    dy = enemy.y - start.y
    length = max(0.01, (dx ** 2 + dy ** 2) ** 0.5)
    fwd_x, fwd_y = dx / length, dy / length

    # Perpendicular (rotated 90° left/right)
    perp_left  = Point2((-fwd_y,  fwd_x))
    perp_right = Point2(( fwd_y, -fwd_x))

    spread_lateral = 30.0   # how far sideways to push the target
    spread_forward = 20.0   # how far forward to lean both targets

    if role == QueenRole.CREEP_LEFT:
        perp = perp_left
    else:
        perp = perp_right

    target = Point2((
        start.x + fwd_x * spread_forward + perp.x * spread_lateral,
        start.y + fwd_y * spread_forward + perp.y * spread_lateral,
    ))

    # Clamp to playable map area
    w = bot.game_info.pathing_grid.width
    h = bot.game_info.pathing_grid.height
    target = Point2((max(2.0, min(w - 2.0, target.x)),
                     max(2.0, min(h - 2.0, target.y))))

    return target


# ---------------------------------------------------------------------------
# Inject behavior (inline — no Ares equivalent for queen inject)
# ---------------------------------------------------------------------------

class _InjectBehavior(CombatIndividualBehavior):
    """Issue EFFECT_INJECTLARVA on the nearest hatchery that needs it."""

    def __init__(self, queen: Unit):
        self.unit = queen

    # Energy threshold for inject. Queens need exactly 25 energy.
    # We avoid queen.abilities entirely — that property requires a prior
    # async GetAvailableAbilities call and will be empty/raise on a raw
    # proto Unit object. Energy is always available synchronously.
    INJECT_ENERGY = 25

    def execute(self, ai, config, mediator) -> bool:
        queen = self.unit

        # Already injecting — leave it alone
        if queen.is_using_ability(AbilityId.EFFECT_INJECTLARVA):
            log.debug(
                "_InjectBehavior: queen tag=%d already injecting — skip",
                queen.tag,
            )
            return True

        # Energy gate — no async abilities lookup needed
        if queen.energy < self.INJECT_ENERGY:
            log.debug(
                "_InjectBehavior: queen tag=%d energy=%.0f < %d — pre-positioning near hatchery",
                queen.tag,
                queen.energy,
                self.INJECT_ENERGY,
            )
            # Pre-position: move toward nearest hatchery while waiting for energy
            if ai.townhalls.ready:
                closest_hatch = ai.townhalls.ready.closest_to(queen)
                if queen.distance_to(closest_hatch) > 5.0:
                    queen.move(closest_hatch.position)
            return True  # claim the action slot so we don't fall through

        hatcheries = [
            h for h in ai.townhalls.ready
            if not h.has_buff(sc2_buff_injectlarva())
        ]

        if not hatcheries:
            log.debug(
                "_InjectBehavior: queen tag=%d — no hatchery needs inject right now",
                queen.tag,
            )
            return False

        closest = min(hatcheries, key=lambda h: queen.distance_to(h))
        log.info(
            "_InjectBehavior: queen tag=%d injecting hatchery tag=%d (dist=%.1f energy=%.0f)",
            queen.tag,
            closest.tag,
            queen.distance_to(closest),
            queen.energy,
        )
        queen(AbilityId.EFFECT_INJECTLARVA, closest)
        return True


def sc2_buff_injectlarva():
    """Late import to avoid circular issues at module load time."""
    from sc2.ids.buff_id import BuffId
    return BuffId.QUEENSPAWNLARVATIMER


# ---------------------------------------------------------------------------
# Inline creep spread behaviors — avoid ares QueenSpreadCreep/TumorSpreadCreep
# which both call unit.abilities (not available on this project's local sc2).
#
# Queen creep spread energy cost: BUILD_CREEPTUMOR_QUEEN = 25 energy
# Tumor spread: idle burrowed tumor (no orders) can spread once
# ---------------------------------------------------------------------------

class _QueenCreepBehavior(CombatIndividualBehavior):
    """
    Spread creep from a queen toward a directional target.

    Logic mirrors Ares' QueenSpreadCreep but uses only energy (never .abilities):
    - energy < 25 → pre-move toward the nearest creep edge to pre-position
    - energy >= 25 → find a tumor placement via mediator and either move to it
                     or issue BUILD_CREEPTUMOR_QUEEN directly
    """

    def __init__(self, queen: Unit, target: Point2):
        self.unit = queen
        self.target = target

    def execute(self, ai, config, mediator) -> bool:
        queen = self.unit

        # Already in the middle of placing a tumor — leave it alone
        if queen.is_using_ability(AbilityId.BUILD_CREEPTUMOR):
            log.debug(
                "_QueenCreepBehavior: queen tag=%d already placing tumor — skip",
                queen.tag,
            )
            return True

        TUMOR_ENERGY = 25
        can_place = queen.energy >= TUMOR_ENERGY

        # Find best edge position near the queen (large search radius so queens
        # that are far from creep still get a pre-move target)
        try:
            edge_position: Point2 = mediator.find_nearby_creep_edge_position(
                position=queen.position,
                search_radius=200.0,
                unit_tag=queen.tag,
                cache_result=not can_place,
            )
        except Exception as exc:
            log.warning(
                "_QueenCreepBehavior: find_nearby_creep_edge_position raised: %s", exc
            )
            edge_position = None

        if not can_place:
            # Pre-position: move toward the creep edge while energy builds
            if edge_position and cy_distance_to_squared(queen.position, edge_position) > 9.0:
                queen.move(edge_position)
                log.debug(
                    "_QueenCreepBehavior: queen tag=%d energy=%.0f — pre-moving to edge (%.1f,%.1f)",
                    queen.tag, queen.energy, edge_position.x, edge_position.y,
                )
            return True  # claim the slot — nothing else should interrupt pre-positioning

        # We have energy — find a placement spot
        creep_spot: Point2 | None = None
        try:
            grid = mediator.get_ground_grid
            # Use get_next_tumor_on_path to spread directionally toward target
            creep_spot = mediator.get_next_tumor_on_path(
                grid=grid,
                from_pos=queen.position,
                to_pos=self.target,
                find_alternative=True,
            )
        except Exception as exc:
            log.warning(
                "_QueenCreepBehavior: get_next_tumor_on_path raised: %s", exc
            )

        # Fall back to edge position if path-based placement failed
        if not creep_spot and edge_position:
            creep_spot = edge_position

        if creep_spot:
            dist_sq = cy_distance_to_squared(queen.position, creep_spot)
            if dist_sq > 25.0:
                queen.move(creep_spot)
                log.debug(
                    "_QueenCreepBehavior: queen tag=%d energy=%.0f moving to spot (%.1f,%.1f) dist_sq=%.1f",
                    queen.tag, queen.energy, creep_spot.x, creep_spot.y, dist_sq,
                )
            else:
                queen(AbilityId.BUILD_CREEPTUMOR_QUEEN, creep_spot)
                log.info(
                    "_QueenCreepBehavior: queen tag=%d placing tumor at (%.1f,%.1f) energy=%.0f",
                    queen.tag, creep_spot.x, creep_spot.y, queen.energy,
                )
            return True

        log.debug(
            "_QueenCreepBehavior: queen tag=%d no valid creep spot found",
            queen.tag,
        )
        return False


class _TumorBehavior(CombatIndividualBehavior):
    """
    Spread creep from a burrowed tumor toward a target.

    Mirrors Ares' TumorSpreadCreep but uses only unit.orders (never .abilities):
    - Strategy 1: spread to a nearby creep edge (within tumor range ~10 tiles)
    - Strategy 2: random creep position as fallback
    """

    def __init__(self, tumor: Unit, target: Point2):
        self.unit = tumor
        self.target = target

    def execute(self, ai, config, mediator) -> bool:
        tumor = self.unit

        # If it already has an order it's in the process of spreading — done
        if tumor.orders:
            log.debug(
                "_TumorBehavior: tumor tag=%d has orders — already spreading",
                tumor.tag,
            )
            return True

        # Strategy 1: spread to nearby creep edge
        try:
            chosen_pos: Point2 = mediator.find_nearby_creep_edge_position(
                position=tumor.position,
                search_radius=10.2,
                closest_valid=False,
                spread_dist=1.0,
            )
            if chosen_pos:
                tumor(AbilityId.BUILD_CREEPTUMOR, chosen_pos)
                log.info(
                    "_TumorBehavior: tumor tag=%d spreading to edge (%.1f,%.1f)",
                    tumor.tag, chosen_pos.x, chosen_pos.y,
                )
                return True
        except Exception as exc:
            log.warning("_TumorBehavior: find_nearby_creep_edge_position raised: %s", exc)

        # Strategy 2: random creep position fallback
        try:
            random_pos: Point2 = mediator.get_random_creep_position(
                position=tumor.position
            )
            if random_pos:
                tumor(AbilityId.BUILD_CREEPTUMOR, random_pos)
                log.info(
                    "_TumorBehavior: tumor tag=%d spreading to random pos (%.1f,%.1f)",
                    tumor.tag, random_pos.x, random_pos.y,
                )
                return True
        except Exception as exc:
            log.warning("_TumorBehavior: get_random_creep_position raised: %s", exc)

        log.debug("_TumorBehavior: tumor tag=%d no spread position found", tumor.tag)
        return False


# ---------------------------------------------------------------------------
# 1. QueenInjectTactic
# ---------------------------------------------------------------------------

class QueenInjectTactic(TacticModule):
    """
    INJECTOR squad: queens with enough energy inject the nearest hatchery.
    Queens with < 25 energy move toward the nearest un-injected hatchery
    to pre-position.
    """

    INJECT_ENERGY = 25  # minimum energy to issue inject

    def is_applicable(self, unit: Unit, bot: "ManifestorBot") -> bool:
        if unit.type_id != UnitID.QUEEN:
            return False
        _assign_queen_roles(bot)
        role = _queen_roles.get(unit.tag)
        if role != QueenRole.INJECTOR:
            return False
        if not bot.townhalls.ready:
            log.debug(
                "QueenInjectTactic: no ready townhalls — skip queen tag=%d",
                unit.tag,
            )
            return False
        return True

    def generate_idea(
        self,
        unit: Unit,
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        role = _queen_roles.get(unit.tag)
        log.debug(
            "QueenInjectTactic.generate_idea: queen tag=%d role=%s energy=%.0f",
            unit.tag,
            role,
            unit.energy,
        )

        # High base confidence — inject is infrastructure, always useful
        confidence = 0.90
        evidence: dict = {"role": "INJECTOR", "energy": unit.energy}

        # Bonus if we have inject ready now
        if unit.energy >= self.INJECT_ENERGY:
            confidence += 0.08
            evidence["inject_ready"] = 0.08

        return TacticIdea(
            tactic_module=self,
            confidence=min(1.0, confidence),
            evidence=evidence,
            target=None,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: "ManifestorBot",
    ) -> Optional[CombatIndividualBehavior]:
        log.info(
            "QueenInjectTactic.create_behavior: queen tag=%d energy=%.0f",
            unit.tag,
            unit.energy,
        )
        return _InjectBehavior(unit)


# ---------------------------------------------------------------------------
# 2. QueenCreepSpreadTactic
# ---------------------------------------------------------------------------

class QueenCreepSpreadTactic(TacticModule):
    """
    CREEP_LEFT / CREEP_RIGHT squads: queens spread creep toward a
    directional target using Ares' QueenSpreadCreep behavior.

    Queens lay the first tumor when they have ≥ 25 energy and creep is
    available under them (or they move toward the creep edge first).
    """

    def is_applicable(self, unit: Unit, bot: "ManifestorBot") -> bool:
        if unit.type_id != UnitID.QUEEN:
            return False
        _assign_queen_roles(bot)
        role = _queen_roles.get(unit.tag)
        if role not in (QueenRole.CREEP_LEFT, QueenRole.CREEP_RIGHT):
            log.debug(
                "QueenCreepSpreadTactic: queen tag=%d role=%s — not a creep queen, skip",
                unit.tag,
                role,
            )
            return False
        return True

    def generate_idea(
        self,
        unit: Unit,
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        role = _queen_roles.get(unit.tag)
        target = _creep_target_for_role(role, bot)

        log.debug(
            "QueenCreepSpreadTactic.generate_idea: queen tag=%d role=%s energy=%.0f target=(%.1f,%.1f)",
            unit.tag,
            role,
            unit.energy,
            target.x,
            target.y,
        )

        confidence = 0.85
        evidence: dict = {
            "role": role.value,
            "energy": unit.energy,
            "target_x": round(target.x, 1),
            "target_y": round(target.y, 1),
        }

        if unit.energy >= 25:
            confidence += 0.10
            evidence["tumor_ready"] = 0.10

        return TacticIdea(
            tactic_module=self,
            confidence=min(1.0, confidence),
            evidence=evidence,
            target=target,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: "ManifestorBot",
    ) -> Optional[CombatIndividualBehavior]:
        role = _queen_roles.get(unit.tag)
        target = idea.target or _creep_target_for_role(role, bot)

        log.info(
            "QueenCreepSpreadTactic.create_behavior: queen tag=%d role=%s → target=(%.1f,%.1f) energy=%.0f",
            unit.tag,
            role,
            target.x,
            target.y,
            unit.energy,
        )

        # Use our inline behavior — avoids QueenSpreadCreep which calls unit.abilities
        return _QueenCreepBehavior(unit, target)


# ---------------------------------------------------------------------------
# 3. TumorSpreadTactic
# ---------------------------------------------------------------------------

class TumorSpreadTactic(TacticModule):
    """
    All burrowed creep tumors with BUILD_CREEPTUMOR_TUMOR available should
    immediately spread toward the enemy base.

    Ares' TumorSpreadCreep handles the pathfinding; we just need to fire it
    whenever the tumor is ready.
    """

    def is_applicable(self, unit: Unit, bot: "ManifestorBot") -> bool:
        if unit.type_id != UnitID.CREEPTUMORBURROWED:
            return False
        # Tumors spread by issuing BUILD_CREEPTUMOR_TUMOR — they can only do
        # this once, after which they become inactive (CREEPTUMOR type).
        # We don't use unit.abilities (requires async query). Instead we check
        # that the tumor has no current orders — an idle burrowed tumor is
        # ready to spread. Once it issues the command it will have an order.
        is_idle = not unit.orders
        log.debug(
            "TumorSpreadTactic.is_applicable: tumor tag=%d is_idle=%s",
            unit.tag,
            is_idle,
        )
        return is_idle

    def generate_idea(
        self,
        unit: Unit,
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        enemy = bot.enemy_start_locations[0]

        log.debug(
            "TumorSpreadTactic.generate_idea: tumor tag=%d pos=(%.1f,%.1f) → enemy=(%.1f,%.1f)",
            unit.tag,
            unit.position.x,
            unit.position.y,
            enemy.x,
            enemy.y,
        )

        return TacticIdea(
            tactic_module=self,
            confidence=0.95,  # tumors spread the moment they're ready — always
            evidence={"tumor_ready": True},
            target=enemy,
        )

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: "ManifestorBot",
    ) -> Optional[CombatIndividualBehavior]:
        target = idea.target or bot.enemy_start_locations[0]

        log.info(
            "TumorSpreadTactic.create_behavior: tumor tag=%d spreading toward (%.1f,%.1f)",
            unit.tag,
            target.x,
            target.y,
        )

        # Use our inline behavior — avoids TumorSpreadCreep which calls unit.abilities
        return _TumorBehavior(unit, target)