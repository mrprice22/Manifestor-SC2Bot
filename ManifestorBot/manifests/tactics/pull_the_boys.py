"""
PullTheBoysTactic — last-resort full worker mobilisation.

When a real attack (≥ MIN_ATTACKERS enemy combat units) reaches one of our
bases AND we have fewer than MAX_DEFENDERS friendly army units there to stop
it, every eligible drone at that base immediately drops its mineral patch and
attacks.

This is the nuclear option.  It fires when the army isn't home and the base
is going to die unless the workers buy time or drive off the raiders.

How it differs from CitizensArrestTactic
-----------------------------------------
CitizensArrest   Handles a lone scout / single intruder inside the mineral
                 line.  Posse of 3 nearby workers mob one unit.
                 → light harassment response.

PullTheBoys      Handles a real attack wave (3+ combat units).  ALL workers
                 within PULL_RADIUS of the threatened base mobilise, each
                 independently targeting the nearest enemy.  No minimum
                 posse — if even one worker is eligible, it fights.
                 → last-resort base survival.

Confidence: 0.82 – 0.92  (always beats MiningTactic 0.45–0.55).

The tactic is an individual tactic (not group) so each drone picks its own
nearest target. Natural focus-fire emerges because all nearby enemies are
a small cluster; each drone attacks the zergling closest to it, and the
pack gets shredded one-by-one.

Workers return to mining automatically once is_applicable() stops returning
True — no cleanup needed.

Chat announcement: "PULL THE BOYS: N drones vs M attackers!" fires at most
once every ~10 seconds to signal the emergency to the observer without spam.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, FrozenSet

from ares.behaviors.combat.individual import AttackTarget
from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.unit import Unit

from ManifestorBot.manifests.tactics.base import TacticModule, TacticIdea
from ManifestorBot.logger import get_logger

log = get_logger()

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.manifests.heuristics import HeuristicState
    from ManifestorBot.manifests.strategy import Strategy


# ── Tuning constants ───────────────────────────────────────────────────────────

# Minimum attacking enemies at the base before workers mobilise.
# At 1–2 we rely on CitizensArrestTactic (nearby mob).
# At 3+ it's a real attack wave → pull every drone.
MIN_ATTACKERS: int = 3

# If we already have this many army units at the base, don't pull workers —
# let the army handle it.
MAX_DEFENDERS: int = 4

# Scan radius around each townhall for enemy attackers.
ATTACK_RADIUS: float = 25.0

# Workers within this radius of the threatened townhall are eligible.
# Wide enough to pull the whole mineral line, not just nearby workers.
PULL_RADIUS: float = 28.0

# Scan radius for friendly army units when deciding if army is "home".
DEFENDER_SCAN_RADIUS: float = 20.0

# Chat announcement throttle — frames between "PULL THE BOYS" messages.
CHAT_THROTTLE: int = 224   # ≈ 10 seconds

# Non-combatant enemy types we don't mobilise workers to fight.
_NON_COMBATANT_TYPES: frozenset = frozenset({
    UnitID.OVERLORD, UnitID.OVERLORDTRANSPORT, UnitID.OVERSEER,
    UnitID.OVERSEERSIEGEMODE,
    UnitID.SCV, UnitID.PROBE, UnitID.DRONE,
})


# ── Module-level helpers ───────────────────────────────────────────────────────

def _find_attack_wave(bot: "ManifestorBot"):
    """
    Find the townhall currently under a real attack wave.

    Returns (townhall, attacker_list) if MIN_ATTACKERS or more enemy combat
    units are within ATTACK_RADIUS of any ready townhall AND we have fewer
    than MAX_DEFENDERS friendly army units near that base.

    Returns (None, []) if no such attack is in progress.
    """
    for th in bot.townhalls.ready:
        nearby_enemies = bot.enemy_units.closer_than(ATTACK_RADIUS, th.position)
        attackers = [
            e for e in nearby_enemies
            if not e.is_structure
            and not e.is_memory
            and e.type_id not in _NON_COMBATANT_TYPES
        ]
        if len(attackers) < MIN_ATTACKERS:
            continue

        # Check whether the army is already home to deal with it
        nearby_army = bot.units.closer_than(DEFENDER_SCAN_RADIUS, th.position)
        defenders = sum(
            1 for u in nearby_army
            if not u.is_structure
            and u.type_id not in {
                bot.worker_type, bot.supply_type,
                UnitID.OVERLORD, UnitID.OVERSEER, UnitID.QUEEN,
            }
        )
        if defenders >= MAX_DEFENDERS:
            continue   # army is home — let them handle it

        return th, attackers

    return None, []


def _nearest_attacker(worker: Unit, attackers: list) -> Optional[Unit]:
    """Return the attacker closest to this worker (natural spread = pseudo-focus-fire)."""
    if not attackers:
        return None
    return min(attackers, key=lambda e: worker.distance_to(e))


# ── Tactic ─────────────────────────────────────────────────────────────────────

class PullTheBoysTactic(TacticModule):
    """
    Last-resort drone mobilisation against a real attack wave.

    Individual tactic — each eligible drone independently attacks its
    nearest attacker.  No group consolidation required.
    """

    # Never blocked — base survival is non-negotiable
    @property
    def blocked_strategies(self) -> "FrozenSet[Strategy]":
        return frozenset()

    def __init__(self) -> None:
        super().__init__()
        self._last_chat_frame: int = -CHAT_THROTTLE  # allow chat on first trigger

    # ------------------------------------------------------------------ #
    # Structural gate
    # ------------------------------------------------------------------ #

    def is_applicable(self, unit: Unit, bot: "ManifestorBot") -> bool:
        if unit.type_id != bot.worker_type:
            return False
        # Don't yank a drone that's mid-construction
        if getattr(unit, "is_constructing_scv", False):
            return False
        if not bot.townhalls.ready:
            return False

        th, attackers = _find_attack_wave(bot)
        if th is None:
            return False

        # Worker must be close enough to the threatened base to help
        return unit.distance_to(th.position) <= PULL_RADIUS

    # ------------------------------------------------------------------ #
    # Confidence scoring
    # ------------------------------------------------------------------ #

    def generate_idea(
        self,
        unit: Unit,
        bot: "ManifestorBot",
        heuristics: "HeuristicState",
        current_strategy: "Strategy",
    ) -> Optional[TacticIdea]:
        th, attackers = _find_attack_wave(bot)
        if th is None or not attackers:
            return None

        target = _nearest_attacker(unit, attackers)
        if target is None:
            return None

        attacker_count = len(attackers)
        workers_nearby = sum(
            1 for w in bot.workers
            if w.distance_to(th.position) <= PULL_RADIUS
        )

        confidence = 0.82
        evidence: dict = {
            "attacker_count":  attacker_count,
            "workers_pulling": workers_nearby,
            "base_conf":       0.82,
        }

        # --- sub-signal: more attackers = more urgent ---
        if attacker_count > MIN_ATTACKERS:
            sig = min(0.06, (attacker_count - MIN_ATTACKERS) * 0.02)
            confidence += sig
            evidence["attacker_urgency"] = round(sig, 3)

        # --- sub-signal: no army at all = truly desperate ---
        nearby_army = bot.units.closer_than(DEFENDER_SCAN_RADIUS, th.position)
        real_defenders = sum(
            1 for u in nearby_army
            if not u.is_structure
            and u.type_id not in {
                bot.worker_type, bot.supply_type,
                UnitID.OVERLORD, UnitID.OVERSEER, UnitID.QUEEN,
            }
        )
        if real_defenders == 0:
            confidence += 0.04
            evidence["no_army_bonus"] = 0.04

        confidence = min(0.92, confidence)

        # --- announce the first activation (throttled) ---
        frame = bot.state.game_loop
        if frame - self._last_chat_frame >= CHAT_THROTTLE:
            self._last_chat_frame = frame
            bot._pending_chat.append(
                f"PULL THE BOYS: {workers_nearby} drones vs {attacker_count} attackers!"
            )
            log.game_event(
                "PULL_THE_BOYS",
                f"{workers_nearby} drones mobilised vs {attacker_count} attackers "
                f"at {th.position} | conf={confidence:.2f}",
                frame=frame,
            )

        return TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=target,
        )

    # ------------------------------------------------------------------ #
    # Behavior
    # ------------------------------------------------------------------ #

    def create_behavior(
        self,
        unit: Unit,
        idea: TacticIdea,
        bot: "ManifestorBot",
    ) -> Optional[CombatIndividualBehavior]:
        target = idea.target
        if target is None or target not in bot.enemy_units:
            return None
        return AttackTarget(unit=unit, target=target)
