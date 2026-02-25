"""
DefendableTerritory — army-strength-gated territorial boundary.

Models what the bot can actually defend at any given moment, based on army
supply and active threat level.  This is intentionally separate from:

  - TerritoryBorderMap   — creep/vision edge for overlord placement
  - PheromoneMap         — threat-scent for combat decisions
  - HeuristicState       — aggregate signals for strategy switching

DefendableTerritory answers the question: "Given our army right now, what
territory can we hold?"  It provides two primary services:

  1. is_safe_to_expand(bot, h, target_base_count)
     Returns True when all three safety conditions are met:
       - threat_level is below the expansion-safe ceiling
       - army supply meets the minimum for this expansion tier
       - army_value_ratio is not clearly losing

  2. patrol_edge_point(bot, unit_position)
     Returns a patrol Point2 on the boundary of our defended territory.
     Returned to OpportunisticPatrolTactic when army is small so units stay
     near home rather than drifting forward toward the enemy.

Territory geometry
------------------
For each ready hatchery, the "defended radius" is:

    raw_radius  = BASE_RADIUS + supply_army * RADIUS_PER_SUPPLY
    effective   = raw_radius  * max(THREAT_SHRINK_FLOOR, 1.0 - threat_level)

The territory edge patrol point is:

    nearest_hatch.position.towards(enemy_start, effective_radius)

so it always faces the expected threat direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sc2.position import Point2

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState


# ── Geometry constants ─────────────────────────────────────────────────────────

# Minimum defended radius regardless of army size (queens alone hold this)
BASE_RADIUS: float = 8.0

# Each 1 point of army supply extends the perimeter by this many tiles
RADIUS_PER_SUPPLY: float = 0.30

# Cap so even a 200-supply army doesn't claim the entire map
MAX_RADIUS: float = 30.0

# Under maximum threat (1.0) the radius shrinks to this fraction
THREAT_SHRINK_FLOOR: float = 0.40


# ── Expansion safety constants ─────────────────────────────────────────────────

# Don't expand when enemy is this threatening (0.30 ≈ enemy within 70 tiles of base)
SAFE_EXPAND_MAX_THREAT: float = 0.30

# Don't expand when we're losing the army fight (only enforced post early-game)
SAFE_EXPAND_MIN_RATIO: float = 0.80

# Minimum army supply (supply_army) before each expansion level.
# Key = target base count after the expansion completes.
# These values account for Zerg's need to hold their economy during an expansion:
#   2nd base: 1 queen (2) + ~8 lings (8) = 10 minimum
#   3rd base: queens + roach/ling core
#   4th+:     substantial defensive force
_MIN_ARMY_FOR_EXPANSION: dict[int, int] = {
    2: 10,   # Second hatchery: needs at minimum queens + some lings
    3: 22,   # Third hatchery: proper defensive core
    4: 32,   # Fourth hatchery: multi-front capable force
    5: 40,   # Fifth hatchery: late-game economy expansion
}
_DEFAULT_MIN_ARMY: int = 45  # 6th base and beyond


# ── Patrol boundary constant ───────────────────────────────────────────────────

# When army supply is below this, patrol_edge_point() stays close to home
ARMY_PATROL_THRESHOLD: int = 20


class DefendableTerritory:
    """
    Lightweight defensive perimeter tracker.

    Updated once per step (after heuristics). No grid operations — pure
    arithmetic on existing heuristic values.

    ManifestorBot holds one instance as ``self.defendable_territory``.
    """

    def __init__(self) -> None:
        self._effective_radius: float = BASE_RADIUS
        self._last_supply: int = 0

    def update(self, bot: "ManifestorBot", h: "HeuristicState") -> None:
        """Recompute perimeter. Call once per step after heuristics.update()."""
        supply = bot.supply_army
        raw_radius = min(MAX_RADIUS, BASE_RADIUS + supply * RADIUS_PER_SUPPLY)
        shrink = max(THREAT_SHRINK_FLOOR, 1.0 - h.threat_level)
        self._effective_radius = raw_radius * shrink
        self._last_supply = supply

    # ── Expansion safety ───────────────────────────────────────────────────────

    def is_safe_to_expand(
        self,
        bot: "ManifestorBot",
        h: "HeuristicState",
        target_base_count: int,
    ) -> tuple[bool, str]:
        """
        Return (safe: bool, reason: str) for whether expansion is safe.

        The reason string is logged when expansion is blocked, making it
        easy to see exactly why the bot isn't expanding.
        """
        # Gate 1: active threat
        if h.threat_level >= SAFE_EXPAND_MAX_THREAT:
            return False, f"threat_level={h.threat_level:.2f} >= {SAFE_EXPAND_MAX_THREAT}"

        # Gate 2: minimum army for this expansion tier
        min_army = _MIN_ARMY_FOR_EXPANSION.get(target_base_count, _DEFAULT_MIN_ARMY)
        if bot.supply_army < min_army:
            return False, f"supply_army={bot.supply_army} < {min_army} required for base #{target_base_count}"

        # Gate 3: not clearly losing the army fight (skip in very early game)
        if h.game_phase >= 0.05 and h.army_value_ratio < SAFE_EXPAND_MIN_RATIO:
            return False, (
                f"army_value_ratio={h.army_value_ratio:.2f} < {SAFE_EXPAND_MIN_RATIO} "
                f"(phase={h.game_phase:.2f})"
            )

        return True, "ok"

    @property
    def effective_radius(self) -> float:
        """Current defended radius in tiles."""
        return self._effective_radius

    # ── Patrol target ──────────────────────────────────────────────────────────

    def patrol_edge_point(
        self,
        bot: "ManifestorBot",
        unit_position: Point2,
    ) -> Optional[Point2]:
        """
        Return a patrol point on the territory boundary.

        The point is at effective_radius tiles from the nearest hatchery,
        projected toward the enemy start location.  This keeps units on the
        edge of what we can hold rather than drifting into enemy territory.
        Returns None if no hatcheries are ready.
        """
        if not bot.townhalls.ready:
            return None

        nearest_th = bot.townhalls.ready.closest_to(unit_position)
        if not bot.enemy_start_locations:
            # No known enemy start — orbit the main at patrol radius
            return nearest_th.position.towards(
                bot.game_info.map_center, self._effective_radius
            )

        return nearest_th.position.towards(
            bot.enemy_start_locations[0], self._effective_radius
        )

    def should_patrol_territory_edge(self, bot: "ManifestorBot") -> bool:
        """
        True when the army is small enough that units should stay near home
        rather than pushing forward for map control.
        """
        return bot.supply_army < ARMY_PATROL_THRESHOLD
