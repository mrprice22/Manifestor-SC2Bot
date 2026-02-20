"""
AbilityRegistry — maps unit types to their ability lists.

Usage
-----
The registry is a singleton populated at bot start. Each unit type gets an
ordered list of Ability instances. The AbilitySelector iterates this list
(highest-priority first) and fires the first ability that passes can_use().

Registration
------------
Use register() to add abilities at startup. Abilities are always kept sorted
by descending priority within each unit type bucket.

    registry = AbilityRegistry()
    registry.register(UnitID.DRONE, MineAbility())
    registry.register(UnitID.DRONE, AttackAbility())  # lower priority

Fallback chain
--------------
If a unit has no registered abilities, the selector falls through to the
old TacticModule.create_behavior() path, ensuring full backward compatibility
during the migration.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.unit import Unit

from ManifestorBot.abilities.ability import Ability, AbilityContext

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot


class AbilityRegistry:
    """
    Per-unit-type ability registry.

    Thread-safety: not needed — SC2 bots are single-threaded.
    """

    def __init__(self) -> None:
        # UnitID → [Ability, ...] sorted by descending priority
        self._registry: Dict[UnitID, List[Ability]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def register(self, unit_type: UnitID, ability: Ability) -> None:
        """
        Register an ability for a unit type.

        If the same ability class is already registered for this type, the old
        entry is replaced (prevents duplicate registration on hot-reload).
        """
        bucket = self._registry[unit_type]
        # Remove existing entry of the same class to prevent duplicates
        self._registry[unit_type] = [
            a for a in bucket if type(a) is not type(ability)
        ]
        self._registry[unit_type].append(ability)
        # Keep highest priority first
        self._registry[unit_type].sort(key=lambda a: a.priority, reverse=True)

    def register_many(self, unit_type: UnitID, *abilities: Ability) -> None:
        """Convenience batch registration."""
        for ability in abilities:
            self.register(unit_type, ability)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get(self, unit_type: UnitID) -> List[Ability]:
        """Return the ability list for a unit type (may be empty)."""
        return self._registry.get(unit_type, [])

    def has_abilities(self, unit_type: UnitID) -> bool:
        """True if at least one ability is registered for this type."""
        return bool(self._registry.get(unit_type))

    def first_applicable(
        self,
        unit: Unit,
        context: AbilityContext,
        bot: "ManifestorBot",
    ) -> Optional[Ability]:
        """
        Return the highest-priority ability that passes can_use(), or None.

        Goal filtering: if an ability declares GOAL != "any", it is only
        eligible when context.goal matches that GOAL string.
        """
        for ability in self.get(unit.type_id):
            # Goal filter
            if ability.GOAL != "any" and ability.GOAL != context.goal:
                continue
            # Eligibility check
            if ability.can_use(unit, context, bot):
                return ability
        return None

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable registry listing (for startup logs)."""
        lines = []
        for unit_type, abilities in sorted(self._registry.items(), key=lambda x: x[0].name):
            names = ", ".join(f"{a.name}(p={a.priority})" for a in abilities)
            lines.append(f"  {unit_type.name}: [{names}]")
        return "\n".join(lines) if lines else "  (empty)"


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
ability_registry = AbilityRegistry()
