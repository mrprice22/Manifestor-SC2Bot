"""
ResourcePressureManager — overbanking guard.

Produces a confidence boost for ZergArmyProductionTactic when minerals
bank past the soft threshold, and forces spending at panic level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot


class ResourcePressureManager:
    SOFT_THRESHOLD = 800    # minerals: begin ramping
    PANIC_THRESHOLD = 1500  # minerals: force spending

    def army_production_boost(self, bot: "ManifestorBot") -> float:
        mins = bot.minerals
        if mins >= self.PANIC_THRESHOLD:
            return 0.40          # panic → confidence will be forced to 0.92
        elif mins >= self.SOFT_THRESHOLD:
            frac = (mins - self.SOFT_THRESHOLD) / (self.PANIC_THRESHOLD - self.SOFT_THRESHOLD)
            return frac * 0.20   # linear ramp 0.0 → 0.20
        return 0.0

    def is_panic_mode(self, bot: "ManifestorBot") -> bool:
        return bot.minerals >= self.PANIC_THRESHOLD
