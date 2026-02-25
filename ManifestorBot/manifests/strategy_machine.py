"""
StrategyMachine — heuristic-driven strategy state machine.

Evaluates game conditions every ~22 frames and selects the most
appropriate strategy from the priority table. Switches the bot's
active strategy when a new one is warranted.

Anti-thrash design
------------------
Two mechanisms prevent the bot from flip-flopping between strategies:

1. Lockout timer (MIN_FRAMES_BETWEEN_SWITCHES, ~60 s):
   After any switch, no further switch is allowed for this many frames.
   This is the primary safeguard — it means strategies get at least a
   minute to play out before we reconsider.

2. Confirmation gate (CONFIRMATION_COUNT = 3 evaluations, ~3 s):
   A candidate strategy must be selected in N consecutive evaluations
   before the switch commits. This filters transient heuristic spikes
   (one anomalous frame doesn't trigger a strategy change).

Emergency override
------------------
Rules with emergency=True bypass both mechanisms. Currently only
DRONE_ONLY_FORTRESS qualifies: if we're taking damage at our base
AND badly outgunned, that's not a recoverable situation — act now.

Priority table
--------------
Rules are checked in order; the first match wins.
STOCK_STANDARD is always last and always matches (default fallback).

  1. DRONE_ONLY_FORTRESS  — emergency turtle (enemy at gates, losing badly)
  2. ALL_IN               — crushing army advantage, close it out now
  3. BLEED_OUT            — losing army fight, pivot to guerrilla
  4. WAR_OF_ATTRITION     — under pressure but economically ahead, hold/grind
  5. JUST_GO_PUNCH_EM     — army advantage + positive momentum, press forward
  6. WAR_ON_SANITY        — economically dominant, maximum multi-front chaos
  7. KEEP_EM_BUSY         — initiative-based, harassment when in their half
  8. STOCK_STANDARD       — balanced default
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from ManifestorBot.manifests.strategy import Strategy

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState

log = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────

# Frames between switches (~60 s at 22.4 fps). Emergency rules bypass this.
MIN_FRAMES_BETWEEN_SWITCHES: int = 1344

# How many consecutive evaluations must agree before committing a switch.
# Machine runs every ~22 frames, so 3 = ~3 s of agreement required.
CONFIRMATION_COUNT: int = 3

# Machine evaluation cadence in game_loop frames
EVAL_CADENCE: int = 22


# ── Rule dataclass ────────────────────────────────────────────────────────────

@dataclass
class _StrategyRule:
    """
    One entry in the priority table.

    enter(h) → True  means conditions are right to switch to this strategy.
    emergency         means it bypasses the lockout and confirmation gate.
    """
    strategy: Strategy
    enter: Callable[['HeuristicState'], bool]
    emergency: bool = False


# ── Priority table ────────────────────────────────────────────────────────────

_RULES: list[_StrategyRule] = [
    # ── 1. Emergency turtle ─────────────────────────────────────────────────
    # Enemy is threatening our base AND we're badly outgunned.
    # threat_level 0.65 ≈ enemy army ~35 tiles from our townhall.
    # army_value_ratio < 0.65 = we have significantly less army.
    # game_phase >= 0.30: don't trigger before Lair/spines are viable —
    #   at phase < 0.30 we have no static defence to hold the fortress.
    _StrategyRule(
        strategy=Strategy.DRONE_ONLY_FORTRESS,
        enter=lambda h: (
            h.threat_level >= 0.65
            and h.army_value_ratio < 0.65
            and h.game_phase >= 0.30
        ),
        emergency=True,
    ),

    # ── 2. All-in overwhelm ─────────────────────────────────────────────────
    # Our army is worth 1.75× theirs and we're past early game.
    # Don't drag it out — go end it.
    _StrategyRule(
        strategy=Strategy.ALL_IN,
        enter=lambda h: (
            h.army_value_ratio >= 1.75
            and h.game_phase >= 0.20
        ),
    ),

    # ── 3. Guerrilla/harassment pivot ───────────────────────────────────────
    # Losing the army fight at mid-game or later.
    # Stop taking fair trades; switch to muta-bane harassment to bleed them.
    _StrategyRule(
        strategy=Strategy.BLEED_OUT,
        enter=lambda h: (
            h.army_value_ratio < 0.80
            and h.game_phase >= 0.25
        ),
    ),

    # ── 4. Hold and grind ───────────────────────────────────────────────────
    # Being pushed but have a significant economic lead.
    # Lurker/infestor/ravager lines grind them down; don't throw it away.
    _StrategyRule(
        strategy=Strategy.WAR_OF_ATTRITION,
        enter=lambda h: (
            h.threat_level >= 0.50
            and h.economic_health >= 1.15
            and h.game_phase >= 0.35
        ),
    ),

    # ── 5. Press army advantage ─────────────────────────────────────────────
    # Winning the army fight and have momentum. Commit and push.
    _StrategyRule(
        strategy=Strategy.JUST_GO_PUNCH_EM,
        enter=lambda h: (
            h.army_value_ratio >= 1.30
            and h.momentum > 0.5
            and h.game_phase >= 0.20
        ),
    ),

    # ── 6. Economic dominance — all-front chaos ──────────────────────────────
    # Significantly more workers and army lead. Make the opponent deal with
    # pressure from every direction simultaneously.
    _StrategyRule(
        strategy=Strategy.WAR_ON_SANITY,
        enter=lambda h: (
            h.economic_health >= 1.30
            and h.army_value_ratio >= 1.10
            and h.game_phase >= 0.30
        ),
    ),

    # ── 7. Initiative-based harassment ─────────────────────────────────────
    # Our army is in their half and roughly even. Poke, harass, force splits.
    _StrategyRule(
        strategy=Strategy.KEEP_EM_BUSY,
        enter=lambda h: (
            h.initiative > 0.15
            and h.army_value_ratio >= 0.90
            and h.game_phase >= 0.28
        ),
    ),

    # ── 8. Balanced default ─────────────────────────────────────────────────
    # Always matches — textbook macro Zerg until something else fires.
    _StrategyRule(
        strategy=Strategy.STOCK_STANDARD,
        enter=lambda h: True,
    ),
]


# ── State machine ─────────────────────────────────────────────────────────────

class StrategyMachine:
    """
    Heuristic-driven strategy selection with anti-thrash protection.

    Call update() from on_step after heuristics have been computed.
    The machine calls bot.change_strategy() when a switch is warranted.
    """

    def __init__(self) -> None:
        self._last_switch_frame: int = 0
        self._candidate: Optional[Strategy] = None
        self._candidate_count: int = 0

    def update(self, bot: 'ManifestorBot', h: 'HeuristicState') -> None:
        """
        Evaluate the priority table and switch strategies if warranted.
        Call this once per on_step after heuristics.update().
        """
        # Guard: only run if we have townhalls (game is in progress)
        if not bot.townhalls:
            return

        frame = bot.state.game_loop

        # Only evaluate on our cadence to avoid excessive processing
        if frame % EVAL_CADENCE != 0:
            return

        target = self._select_target(h)

        # Already on the right strategy — reset any pending candidate
        if target == bot.current_strategy:
            self._candidate = None
            self._candidate_count = 0
            return

        # Check if this is an emergency rule
        rule = next((r for r in _RULES if r.strategy == target), None)
        is_emergency = rule is not None and rule.emergency

        if is_emergency:
            log.warning(
                "StrategyMachine EMERGENCY: %s → %s (frame=%d, avr=%.2f, threat=%.2f)",
                bot.current_strategy.value, target.value, frame,
                h.army_value_ratio, h.threat_level,
            )
            bot.change_strategy(target, reason="emergency")
            self._last_switch_frame = frame
            self._candidate = None
            self._candidate_count = 0
            return

        # Lockout: still within cooldown window, skip
        frames_since_switch = frame - self._last_switch_frame
        if frames_since_switch < MIN_FRAMES_BETWEEN_SWITCHES:
            return

        # Confirmation: accumulate consecutive agreements
        if target == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = target
            self._candidate_count = 1

        if self._candidate_count >= CONFIRMATION_COUNT:
            log.info(
                "StrategyMachine: %s → %s (frame=%d, avr=%.2f econ=%.2f threat=%.2f phase=%.2f)",
                bot.current_strategy.value, target.value, frame,
                h.army_value_ratio, h.economic_health, h.threat_level, h.game_phase,
            )
            bot.change_strategy(target, reason="state_machine")
            self._last_switch_frame = frame
            self._candidate = None
            self._candidate_count = 0

    def candidate_summary(self) -> str:
        """Human-readable description of current candidate state (for logs)."""
        if self._candidate is None:
            return "no candidate"
        return (
            f"candidate={self._candidate.value} "
            f"({self._candidate_count}/{CONFIRMATION_COUNT})"
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _select_target(self, h: 'HeuristicState') -> Strategy:
        """Return the highest-priority strategy whose enter() condition fires."""
        for rule in _RULES:
            try:
                if rule.enter(h):
                    return rule.strategy
            except Exception as exc:
                log.warning("StrategyMachine: rule %s raised %s", rule.strategy, exc)
                continue
        return Strategy.STOCK_STANDARD
