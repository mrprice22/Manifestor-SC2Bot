"""
GameStatsTracker — end-of-game performance metrics.

Accumulates statistics throughout the match and writes a formatted summary
to the log at game end.  The summary is emitted as a GAME_STATS game-event
so it appears at GAME level in the log file and is easy to grep.

Tracked statistics
------------------
Losses (things that went wrong — friendly units only):
  overlords_lost     Friendly Overlords destroyed.
  bases_lost         Friendly hatcheries / lairs / hives destroyed without
                     cancel — full mineral cost forfeited.
  units_lost         All other friendly non-structure units destroyed.
  buildings_lost     All friendly structures destroyed (including bases).

Peaks (best values achieved at any single moment):
  peak_army_supply   Largest combat army supply-cost alive simultaneously.
                     Workers, queens, overlords excluded.
  peak_drone_count   Most drones alive at one time.
  peak_game_phase    Highest game-phase heuristic value reached (0.0–1.0).

Duration:
  Computed as end_frame / 22.4 (SC2 fast speed).

Ares engine stats (read from bot.state.score at game end):
  killed_value_units      Mineral+gas value of enemy units killed.
  killed_value_structures Mineral+gas value of enemy structures killed.
  collected_minerals      Total minerals harvested.
  collected_vespene       Total vespene harvested.
  idle_worker_time        Cumulative seconds workers sat idle.

Composite score
---------------
A single per-minute value.  Rewards reaching late-game strength;
penalises losing the game, resource losses, and unit attrition.

    positive = peak_army_supply * 1.0   (0–200 supply → 0–200 pts)
             + peak_drone_count * 2.0   (0–66 drones  → 0–132 pts)
             + peak_game_phase  * 75    (0–1 phase     → 0–75  pts)

    negative = overlords_lost * 5
             + bases_lost      * 25
             + defeat_penalty          (75 pts if Result.Defeat)

    composite = (positive − negative) / max(8, duration_minutes)

    Denominator floor of 8 minutes prevents short games from
    inflating scores; the defeat penalty ensures a loss never
    rates above STRUGGLING regardless of how many drones were alive.

Score labels:  ≥ 12 EXCELLENT · 6–12 SOLID · 2–6 STRUGGLING
                0–2 POOR · < 0 COLLAPSE

Integration
-----------
    # __init__
    self.game_stats = GameStatsTracker()

    # on_step (every frame — tracker self-throttles internally)
    self.game_stats.update(self)

    # on_unit_destroyed
    if dead_unit is not None:
        self.game_stats.record_unit_destroyed(dead_unit)

    # on_end
    self.game_stats.finalize(self, game_result, end_frame)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId as UnitID

from ManifestorBot.logger import get_logger

log = get_logger()

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from sc2.unit import Unit


# ── Constants ──────────────────────────────────────────────────────────────────

SC2_FPS: float = 22.4

# Peak sampling cadence — 112 frames ≈ 5 seconds at fast speed.
SAMPLE_CADENCE: int = 112

# Unit types excluded from the "army supply" peak measurement.
_NON_ARMY_TYPES: frozenset = frozenset({
    UnitID.DRONE, UnitID.SCV, UnitID.PROBE,
    UnitID.QUEEN,
    UnitID.OVERLORD, UnitID.OVERLORDTRANSPORT,
    UnitID.OVERSEER, UnitID.OVERSEERSIEGEMODE,
})

# Friendly command structures — destroyed = "base lost".
_BASE_TYPES: frozenset = frozenset({
    UnitID.HATCHERY, UnitID.LAIR, UnitID.HIVE,
    UnitID.COMMANDCENTER, UnitID.ORBITALCOMMAND, UnitID.PLANETARYFORTRESS,
    UnitID.NEXUS,
})

# ── Composite score weights / penalties ───────────────────────────────────────
_W_ARMY        = 1.0    # pts per supply point of peak army
_W_DRONE       = 2.0    # pts per drone at peak eco
_W_PHASE       = 75.0   # pts for reaching peak game phase (×phase 0–1)
_P_OVERLORD    = 5.0    # penalty pts per overlord lost
_P_BASE        = 25.0   # penalty pts per base destroyed without cancel
_P_DEFEAT      = 75.0   # flat penalty pts added on any defeat
_MIN_DENOM     = 8.0    # minimum "minutes" in denominator — prevents short
                        # games from inflating the score


# ── Tracker ────────────────────────────────────────────────────────────────────

class GameStatsTracker:
    """
    Lightweight accumulator for all end-game statistics.

    Call update()              every on_step tick (self-throttled).
    Call record_unit_destroyed() from on_unit_destroyed.
    Call finalize()            from on_end.
    """

    def __init__(self) -> None:
        # ── Loss counters (friendly units only — guarded by is_mine check) ──
        self.overlords_lost: int = 0
        self.bases_lost: int = 0
        self.units_lost: int = 0       # all other non-structure friendly units
        self.buildings_lost: int = 0   # all friendly structures (incl. bases)

        # ── Peak stats ─────────────────────────────────────────────────────
        self.peak_army_supply: int = 0
        self.peak_drone_count: int = 0
        self.peak_game_phase: float = 0.0

        # ── Internal ───────────────────────────────────────────────────────
        self._last_sample_frame: int = -SAMPLE_CADENCE

    # ── Per-step sampling ──────────────────────────────────────────────────────

    def update(self, bot: "ManifestorBot") -> None:
        """Sample peak values every SAMPLE_CADENCE frames."""
        frame = bot.state.game_loop
        if frame - self._last_sample_frame < SAMPLE_CADENCE:
            return
        self._last_sample_frame = frame

        drone_count = len(bot.workers)
        if drone_count > self.peak_drone_count:
            self.peak_drone_count = drone_count

        army_supply = self._sample_army_supply(bot)
        if army_supply > self.peak_army_supply:
            self.peak_army_supply = army_supply

        try:
            phase = bot.heuristic_manager.get_state().game_phase
            if phase > self.peak_game_phase:
                self.peak_game_phase = phase
        except Exception:
            pass

    # ── Destruction events ────────────────────────────────────────────────────

    def record_unit_destroyed(self, unit: "Unit") -> None:
        """
        Called from on_unit_destroyed for every unit/structure death.

        Only counts friendly (is_mine) units to avoid crediting enemy kills
        against us.  _all_units_previous_map contains both sides so this
        guard is essential.
        """
        if not getattr(unit, "is_mine", True):
            return  # enemy or neutral unit died — not our loss

        t = unit.type_id

        if unit.is_structure:
            self.buildings_lost += 1          # all structures
            if t in _BASE_TYPES:
                self.bases_lost += 1          # subset: command structures
        else:
            self.units_lost += 1              # all non-structure units
            if t == UnitID.OVERLORD:
                self.overlords_lost += 1      # subset: supply providers

    # ── Finalization ──────────────────────────────────────────────────────────

    def finalize(
        self,
        bot: "ManifestorBot",
        game_result: object,
        end_frame: int,
    ) -> None:
        """
        Compute composite score, read Ares engine stats, log full report.
        Called once from on_end().
        """
        duration_seconds = end_frame / SC2_FPS
        duration_minutes = duration_seconds / 60.0
        is_defeat = (game_result == Result.Defeat)
        composite = self._compute_composite(duration_minutes, is_defeat)

        # Read Ares / python-sc2 score object for economy and combat stats
        score = _read_score(bot)

        report = self._format_report(
            game_result=game_result,
            duration_seconds=duration_seconds,
            composite=composite,
            score=score,
        )

        log.game_event("GAME_STATS", "\n" + report, frame=end_frame)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sample_army_supply(self, bot: "ManifestorBot") -> int:
        total = 0
        try:
            army_dict = bot.mediator.get_own_army_dict
            for unit_type, units in army_dict.items():
                if unit_type in _NON_ARMY_TYPES:
                    continue
                supply = bot.calculate_supply_cost(unit_type) or 1
                total += supply * len(units)
        except Exception:
            try:
                for u in bot.units:
                    if u.is_structure or u.type_id in _NON_ARMY_TYPES:
                        continue
                    total += bot.calculate_supply_cost(u.type_id) or 1
            except Exception:
                pass
        return total

    def _compute_composite(self, duration_minutes: float, is_defeat: bool) -> float:
        positive = (
            self.peak_army_supply * _W_ARMY
            + self.peak_drone_count * _W_DRONE
            + self.peak_game_phase  * _W_PHASE
        )
        negative = (
            self.overlords_lost * _P_OVERLORD
            + self.bases_lost   * _P_BASE
            + (_P_DEFEAT if is_defeat else 0.0)
        )
        return (positive - negative) / max(_MIN_DENOM, duration_minutes)

    def _format_report(
        self,
        game_result: object,
        duration_seconds: float,
        composite: float,
        score: Optional[object],
    ) -> str:
        mins = int(duration_seconds // 60)
        secs = int(duration_seconds % 60)
        W = 52

        def row(label: str, value: str) -> str:
            return f"  {label:<24} : {value}"

        sep_thick = "═" * W
        sep_thin  = "─" * W

        # ── header ──────────────────────────────────────────────────────
        lines = [
            sep_thick,
            "  END-OF-GAME STATS",
            sep_thick,
            row("Result",            str(game_result)),
            row("Duration",          f"{mins}m {secs:02d}s"),
        ]

        # ── peaks ────────────────────────────────────────────────────────
        lines += [
            sep_thin,
            "  PEAKS  (best values achieved during match)",
            row("Largest Army",      f"{self.peak_army_supply} supply"),
            row("Largest Eco",       f"{self.peak_drone_count} drones"),
            row("Peak Game Phase",   f"{self.peak_game_phase:.2f}"),
        ]

        # ── losses ───────────────────────────────────────────────────────
        lines += [
            sep_thin,
            "  LOSSES",
            row("Overlords Lost",    str(self.overlords_lost)),
            row("Bases Lost",        str(self.bases_lost)),
            row("Units Lost",        str(self.units_lost)),
            row("Buildings Lost",    str(self.buildings_lost)),
        ]

        # ── ares / engine stats ──────────────────────────────────────────
        if score is not None:
            kvu  = _fmt_val(getattr(score, "killed_value_units",      None))
            kvs  = _fmt_val(getattr(score, "killed_value_structures",  None))
            cm   = _fmt_val(getattr(score, "collected_minerals",       None))
            cv   = _fmt_val(getattr(score, "collected_vespene",        None))
            iwt  = _fmt_time(getattr(score, "idle_worker_time",        None))

            lines += [
                sep_thin,
                "  COMBAT  (engine values)",
                row("Killed (units)",    kvu),
                row("Killed (structs)",  kvs),
                sep_thin,
                "  ECONOMY  (engine values)",
                row("Minerals Collected", cm),
                row("Vespene Collected",  cv),
                row("Idle Worker Time",   iwt),
            ]

        # ── composite ────────────────────────────────────────────────────
        label = _score_label(composite)
        lines += [
            sep_thin,
            row("COMPOSITE SCORE",   f"{composite:+.1f} pts/min  [{label}]"),
            sep_thick,
        ]

        return "\n".join(lines)


# ── Module-level helpers ───────────────────────────────────────────────────────

def _read_score(bot: "ManifestorBot") -> Optional[object]:
    """Safely return bot.state.score, or None if unavailable."""
    try:
        return bot.state.score
    except Exception:
        return None


def _fmt_val(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.0f}"


def _fmt_time(v: Optional[float]) -> str:
    """Format seconds as Xm Ys, or raw seconds if under 60."""
    if v is None:
        return "n/a"
    mins = int(v // 60)
    secs = int(v % 60)
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def _score_label(score: float) -> str:
    if score >= 12.0:
        return "EXCELLENT"
    if score >= 6.0:
        return "SOLID"
    if score >= 2.0:
        return "STRUGGLING"
    if score >= 0.0:
        return "POOR"
    return "COLLAPSE"
