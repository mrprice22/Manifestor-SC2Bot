"""
ManifestorBot Logger - Persistent file-based logging.

Provides structured, levelled logging to rotating log files so you can
review exactly what the bot was thinking long after the game ends —
no need to rely on in-game chat or a live console window.

Usage
-----
    from ManifestorBot.logger import get_logger

    log = get_logger()          # module-level logger
    log.info("Game started")
    log.debug("Heuristic state: %s", state)
    log.warning("Map file not found, falling back")
    log.error("Unexpected exception: %s", exc)

    # Game-specific helpers
    log.game_event("PIVOT", "STOCK_STANDARD → AGGRESSIVE", frame=1234)
    log.tactic("StutterForward", unit_tag=12345, confidence=0.82, frame=1234)

The log file lives at  logs/manifestor_<timestamp>.log  next to run.py.
Old log files are kept for up to LOG_BACKUP_COUNT runs before being deleted.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Configuration ────────────────────────────────────────────────────────────

LOG_DIR         = Path("logs")          # Relative to CWD (i.e. project root)
LOG_LEVEL       = logging.DEBUG         # File log level  (very verbose)
CONSOLE_LEVEL   = logging.INFO          # Console level   (INFO and above)
LOG_BACKUP_COUNT = 10                   # How many old log files to keep
MAX_BYTES        = 5 * 1024 * 1024      # 5 MB per file before rotating


# ── Custom log levels ─────────────────────────────────────────────────────────

GAME_EVENT_LEVEL = 25   # between INFO (20) and WARNING (30)
TACTIC_LEVEL     = 15   # between DEBUG (10) and INFO (20)

logging.addLevelName(GAME_EVENT_LEVEL, "GAME")
logging.addLevelName(TACTIC_LEVEL,     "TACTIC")


# ── Custom formatter ──────────────────────────────────────────────────────────

class ManifestorFormatter(logging.Formatter):
    """
    Adds a [frame] column when a 'frame' extra field is present, so log lines
    can be correlated directly to a specific game loop iteration.

    Example output:
        2026-02-19 21:14:03.412 | INFO    |       - | Game started
        2026-02-19 21:14:05.001 | GAME    |    1280 | PIVOT: STOCK_STANDARD → AGGRESSIVE (army_value_ratio > 1.4)
        2026-02-19 21:14:05.002 | TACTIC  |    1280 | StutterForward | tag=12345 conf=0.82
    """

    BASE_FMT  = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(frame_col)8s | %(message)s"
    DATE_FMT  = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        # Inject a right-aligned frame column (or dash if not provided)
        record.frame_col = str(getattr(record, "frame", "-"))
        record.levelname = record.levelname[:7]  # keep column width fixed
        return super().format(record)


# ── Logger factory ────────────────────────────────────────────────────────────

_logger_instance: Optional["ManifestorLogger"] = None


def get_logger(name: str = "manifestor") -> "ManifestorLogger":
    """
    Return the singleton ManifestorLogger, creating it on first call.

    Call this once at module level in each file that needs logging:

        log = get_logger()
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ManifestorLogger(name)
    return _logger_instance


class ManifestorLogger:
    """
    Thin wrapper around Python's standard logging that adds game-specific
    helpers and wires up both a rotating file handler and a console handler.
    """

    def __init__(self, name: str = "manifestor") -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(LOG_LEVEL)

        # Avoid adding duplicate handlers if the logger is somehow re-initialised
        if self._logger.handlers:
            return

        self._setup_handlers()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_handlers(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file   = LOG_DIR / f"manifestor_{timestamp}.log"

        formatter = ManifestorFormatter(
            fmt     = ManifestorFormatter.BASE_FMT,
            datefmt = ManifestorFormatter.DATE_FMT,
        )

        # ── Rotating file handler ──────────────────────────────────────────
        file_handler = logging.handlers.RotatingFileHandler(
            filename    = log_file,
            maxBytes    = MAX_BYTES,
            backupCount = LOG_BACKUP_COUNT,
            encoding    = "utf-8",
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)

        # ── Console (stderr) handler ───────────────────────────────────────
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(CONSOLE_LEVEL)
        console_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

        self._logger.info(
            "Logger initialised — writing to %s",
            log_file.resolve(),
        )

    # ── Standard log levels ───────────────────────────────────────────────────

    def debug(self, msg: str, *args, frame: Optional[int] = None, **kwargs) -> None:
        self._logger.debug(msg, *args, extra={"frame": frame}, **kwargs)

    def info(self, msg: str, *args, frame: Optional[int] = None, **kwargs) -> None:
        self._logger.info(msg, *args, extra={"frame": frame}, **kwargs)

    def warning(self, msg: str, *args, frame: Optional[int] = None, **kwargs) -> None:
        self._logger.warning(msg, *args, extra={"frame": frame}, **kwargs)

    def error(self, msg: str, *args, frame: Optional[int] = None, **kwargs) -> None:
        self._logger.error(msg, *args, extra={"frame": frame}, **kwargs)

    def critical(self, msg: str, *args, frame: Optional[int] = None, **kwargs) -> None:
        self._logger.critical(msg, *args, extra={"frame": frame}, **kwargs)

    def exception(self, msg: str, *args, frame: Optional[int] = None, **kwargs) -> None:
        self._logger.exception(msg, *args, extra={"frame": frame}, **kwargs)

    # ── Game-specific helpers ─────────────────────────────────────────────────

    def game_event(
        self,
        event_type: str,
        detail: str,
        frame: Optional[int] = None,
    ) -> None:
        """
        Log a significant named game event (strategy pivots, game start/end, etc.).

        Example:
            log.game_event("PIVOT", "STOCK_STANDARD → AGGRESSIVE", frame=1280)
            log.game_event("GAME_END", "Result.Victory", frame=45000)
        """
        self._logger.log(
            GAME_EVENT_LEVEL,
            "%s | %s",
            event_type.upper(),
            detail,
            extra={"frame": frame},
        )

    def tactic(
        self,
        tactic_name: str,
        unit_tag: int,
        confidence: float,
        frame: Optional[int] = None,
        target_tag: Optional[int] = None,
        suppressed: bool = False,
    ) -> None:
        """
        Log an individual tactic idea (executed or suppressed).

        Example:
            log.tactic("StutterForward", unit_tag=12345, confidence=0.82, frame=1280)
        """
        tag_str = f"tag={unit_tag}"
        conf_str = f"conf={confidence:.2f}"
        target_str = f" → target={target_tag}" if target_tag is not None else ""
        status_str = " [SUPPRESSED]" if suppressed else ""
        self._logger.log(
            TACTIC_LEVEL,
            "%s | %s %s%s%s",
            tactic_name,
            tag_str,
            conf_str,
            target_str,
            status_str,
            extra={"frame": frame},
        )

    def heuristics(self, state, frame: Optional[int] = None) -> None:
        """
        Log the full HeuristicState snapshot at DEBUG level.

        Example (called once per heuristic update, not every frame):
            log.heuristics(self.heuristic_manager.get_state(), frame=iteration)
        """
        self._logger.debug(
            "Heuristics | mom=%.2f army_val=%.2f agg=%.0f threat=%.2f econ=%.2f",
            getattr(state, "momentum", 0),
            getattr(state, "army_value_ratio", 0),
            getattr(state, "aggression_dial", 0),
            getattr(state, "threat_level", 0),
            getattr(state, "economic_health", 0),
            extra={"frame": frame},
        )
