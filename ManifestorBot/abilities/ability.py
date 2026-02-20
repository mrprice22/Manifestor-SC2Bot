"""
Ability — the atomic unit of what a unit *does*.

Design principles
-----------------
- An Ability is the final translation layer between a tactic decision and
  an SC2 command. Tactics reason about goals; abilities issue the wire command.
- Abilities are stateless. All runtime information flows in via AbilityContext.
- execute() returns True on success, False if the command could not be issued
  (e.g. not enough energy, cooldown not ready). The selector uses this to try
  fallbacks if needed.
- Abilities declare unit-type affinity via UNIT_TYPES. The registry uses this
  to filter candidates without needing to instantiate every ability.

AbilityContext
--------------
This is the shared "blackboard" that flows down from the strategy layer through
tactics and into ability execution. Tactics modify context fields to express
intent; they no longer issue commands directly.

    strategy → sets context.goal, context.aggression
    tactic   → sets context.target, context.priority_mode
    ability  → reads context, issues SC2 command
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Set

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.unit import Unit

if TYPE_CHECKING:
    from sc2.position import Point2
    from ManifestorBot.manifestor_bot import ManifestorBot


# ---------------------------------------------------------------------------
# AbilityContext — the tactic-to-ability communication channel
# ---------------------------------------------------------------------------

@dataclass
class AbilityContext:
    """
    A mutable blackboard set by tactics and consumed by abilities.

    The strategy layer sets coarse fields (goal, aggression).
    The tactic layer narrows them (target, priority_mode).
    The ability layer reads and executes.

    Tactics should set only the fields they understand. Unset fields default
    to None or False, and abilities must handle those gracefully.
    """

    # ---- Goal-level flags (set by strategy / tactic layer) ----
    goal: str = "idle"           # "mine", "attack", "defend", "retreat", "harass"
    aggression: float = 0.5      # 0.0 = full retreat, 1.0 = all-in

    # ---- Target fields (set by tactic) ----
    target_unit: Optional[Unit] = None
    target_position: Optional["Point2"] = None

    # ---- Priority mode (tactic hints to ability) ----
    priority_mode: str = "default"  # "harass", "kite", "suicide", etc.

    # ---- Contextual data forwarded from the tactic idea ----
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)

    # ---- Runtime state (set by ability, read by loop for logging) ----
    ability_used: Optional[str] = None   # name of ability that fired
    command_issued: bool = False


# ---------------------------------------------------------------------------
# Ability — base class
# ---------------------------------------------------------------------------

class Ability(ABC):
    """
    One concrete thing a unit can do.

    Subclass and implement:
      - UNIT_TYPES  — which unit types this ability is registered for
      - GOAL        — which context.goal values this ability responds to
      - priority    — higher priority abilities are tried first by selector
      - can_use()   — fast eligibility check (cooldown, energy, targets)
      - execute()   — issue the SC2 command; return True on success

    Keep execute() as close to one API call as possible. Business logic
    (when to use this ability) belongs in can_use() or the tactic layer.
    """

    # Subclasses declare these as class attributes
    UNIT_TYPES: Set[UnitID] = set()  # empty = universal
    GOAL: str = "any"                # matches context.goal; "any" = always eligible
    priority: int = 0                # higher wins ties in selector

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def can_use(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        """
        Fast eligibility check. Return False if this ability cannot fire right now.

        Check cooldowns, energy levels, whether a target exists, etc.
        Do NOT do confidence math here — that belongs in tactics.
        """

    @abstractmethod
    def execute(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        """
        Issue the SC2 command.

        Return True if a command was issued (even if it turns out to be a no-op
        on the SC2 side), False if the ability bailed without issuing anything.
        On success, set context.ability_used = self.name.
        """
