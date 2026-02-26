"""
AbilitySelector — the decision layer between tactic ideas and SC2 commands.

Position in the stack
---------------------
    Strategy Layer
        ↓  (sets goal, aggression in context)
    Tactics modify context (target, priority_mode)
        ↓
    AbilitySelector.select_and_execute()
        ↓  (iterates registry, calls can_use / execute)
    Ability.execute()
        ↓
    SC2 command

How it fits with the existing loop
-----------------------------------
_execute_idea() currently calls tactic.create_behavior() and hands the
result to register_behavior(). With this refactor:

  1. _execute_idea() builds an AbilityContext from the TacticIdea.
  2. It calls AbilitySelector.select_and_execute(unit, context, bot).
  3. If a registered ability fires → done.
  4. If no registered ability covers this unit type → fall back to the
     old create_behavior() path (backward-compat bridge).

This means tactics that have NOT been ported yet continue to work as-is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ManifestorBot.abilities.ability import AbilityContext
from ManifestorBot.abilities.ability_registry import ability_registry
from ManifestorBot.logger import get_logger

if TYPE_CHECKING:
    from sc2.unit import Unit
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.tactics.base import TacticIdea

log = get_logger()


class AbilitySelector:
    """
    Selects and executes the best ability for a unit given a tactic idea.

    Responsibilities
    ----------------
    - Build an AbilityContext from the incoming TacticIdea.
    - Query the AbilityRegistry for the first eligible ability.
    - Call execute() and record the result.
    - Fall back to the legacy create_behavior() path if no ability matches.

    The selector itself is stateless — it holds no per-unit data.
    """

    def select_and_execute(
        self,
        unit: "Unit",
        idea: "TacticIdea",
        bot: "ManifestorBot",
    ) -> bool:
        """
        Main entry point called from _execute_idea().

        Returns True if a command was issued (via either path), False otherwise.
        """
        context = self._build_context(idea)

        # --- Ability registry path ---
        if ability_registry.has_abilities(unit.type_id):
            ability = ability_registry.first_applicable(unit, context, bot)
            if ability:
                success = ability.execute(unit, context, bot)
                if success:
                    log.debug(
                        "Ability fired: %s on %s (goal=%s)",
                        ability.name,
                        unit.type_id.name,
                        context.goal,
                        frame=bot.state.game_loop,
                    )
                    return True
                else:
                    log.debug(
                        "Ability %s.execute() returned False for %s — falling through",
                        ability.name,
                        unit.type_id.name,
                        frame=bot.state.game_loop,
                    )

        # --- Legacy fallback path ---
        return self._legacy_execute(unit, idea, bot)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, idea: "TacticIdea") -> AbilityContext:
        """
        Translate a TacticIdea into an AbilityContext.

        The tactic module name is used to infer the goal string. As tactics
        are ported, they can set idea.context directly instead of relying on
        this heuristic mapping.
        """
        # If the idea already carries a pre-built context (ported tactics),
        # use it directly.
        if hasattr(idea, "context") and isinstance(idea.context, AbilityContext):
            return idea.context

        # Heuristic mapping from tactic name → goal string.
        tactic_name = idea.tactic_module.name
        goal = _TACTIC_TO_GOAL.get(tactic_name, "attack")

        return AbilityContext(
            goal=goal,
            aggression=idea.confidence,          # confidence ≈ aggression level
            target_unit=idea.target if hasattr(idea.target, "tag") else None,
            target_position=idea.target if not hasattr(idea.target, "tag") else None,
            confidence=idea.confidence,
            evidence=idea.evidence,
            
        )

    def _legacy_execute(
        self,
        unit: "Unit",
        idea: "TacticIdea",
        bot: "ManifestorBot",
    ) -> bool:
        """
        Backward-compatible execution via TacticModule.create_behavior().

        Used for tactics not yet ported to the ability system.
        """
        from ares.behaviors.combat import CombatManeuver

        behavior = idea.tactic_module.create_behavior(unit, idea, bot)
        if behavior:
            maneuver = CombatManeuver()
            maneuver.add(behavior)
            bot.register_behavior(maneuver)
            return True
        return False


# ---------------------------------------------------------------------------
# Heuristic goal mapping — tactic name → goal string
# Remove entries as tactics are fully ported and set idea.context themselves.
# ---------------------------------------------------------------------------
_TACTIC_TO_GOAL: dict[str, str] = {
    "KeepUnitSafeTactic":     "retreat",
    "StutterForwardTactic":   "attack",
    "HarassWorkersTactic":    "harass",
    "FlankTactic":            "attack",
    "HoldChokePointTactic":   "defend",
    "RallyToArmyTactic":      "rally",
    "CitizensArrestTactic":   "attack",
    "MiningTactic":           "mine",
    "QueenInjectTactic":      "inject",
    "QueenCreepSpreadTactic": "spread_creep",
    "TumorSpreadTactic":      "spread_creep",
    "CrawlerMoveTactic":      "reposition",
    "ExtractorShieldTactic":  "extractor_shield",
    "BanelingMorphTactic":    "morph_baneling",
    "RavagerMorphTactic":     "morph_ravager",
    "CorrosiveBileTactic":    "corrosive_bile",
}


# Module-level singleton
ability_selector = AbilitySelector()
