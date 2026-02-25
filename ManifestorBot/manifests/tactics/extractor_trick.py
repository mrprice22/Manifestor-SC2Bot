"""
ExtractorTrick — early-game supply-block bypass via extractor cancel.

When supply-blocked at 14/14 (overlord pending but not yet hatched), the bot
wastes valuable seconds doing nothing. The extractor trick exploits SC2
mechanics:

  1. Start building an Extractor (costs 1 drone → frees 1 supply slot)
  2. Use that supply slot to train a Drone from larva
  3. Cancel the Extractor before it completes → drone pops back out, 56 minerals
     refunded (75% of 75)

Net cost: 19 minerals for ~10 seconds of unblocked production. This is standard
pro Zerg play in the first 2-3 minutes.

This is implemented as a standalone on_step hook (not a building/unit tactic)
because it needs frame-precise coordination across multiple game objects and
spans several frames of state.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID

from ManifestorBot.logger import get_logger

log = get_logger()

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot


class _Phase(Enum):
    IDLE = auto()           # waiting for supply block
    BUILDING = auto()       # extractor ordered, waiting for it to appear
    WAITING_DRONE = auto()  # extractor morphing, drone queued, wait for cancel window
    DONE = auto()           # cooldown / finished for this game


# Maximum game phase (heuristic) at which the trick is relevant
_MAX_GAME_PHASE: float = 0.10

# Maximum frame at which we'll attempt the trick (safety net: ~3 min at fastest)
_MAX_FRAME: int = 22.4 * 180  # ~4032 frames = 3 minutes at fastest speed

# How many frames to wait after ordering the extractor before we give up
_BUILD_TIMEOUT_FRAMES: int = 112  # ~5 seconds

# Geyser search radius from main base
_GEYSER_SEARCH_RADIUS: float = 12.0

# Radius to check if geyser already has a structure on it
_GEYSER_OCCUPIED_RADIUS: float = 2.0


class ExtractorTrick:
    """
    Stateful on_step hook for the early-game extractor trick.

    Usage in manifestor_bot.py:
        self.extractor_trick = ExtractorTrick()
        # in on_step, before building ideas:
        await self.extractor_trick.update(self)
    """

    def __init__(self) -> None:
        self._phase: _Phase = _Phase.IDLE
        self._extractor_tag: Optional[int] = None
        self._ordered_frame: int = 0
        self._drone_queued: bool = False

    async def update(self, bot: "ManifestorBot") -> None:
        """Call once per on_step. Drives the state machine."""
        frame = bot.state.game_loop

        # Once done, never fire again
        if self._phase == _Phase.DONE:
            return

        # Safety: don't attempt after early game
        if frame > _MAX_FRAME:
            self._phase = _Phase.DONE
            return

        if self._phase == _Phase.IDLE:
            self._try_start(bot, frame)
        elif self._phase == _Phase.BUILDING:
            self._check_extractor_started(bot, frame)
        elif self._phase == _Phase.WAITING_DRONE:
            self._try_cancel(bot, frame)

    def _try_start(self, bot: "ManifestorBot", frame: int) -> None:
        """Phase IDLE → BUILDING: trigger when supply-blocked with overlord pending."""
        # Only when fully supply blocked
        if bot.supply_left > 0:
            return

        # Only if an overlord is already on the way
        if bot.already_pending(UnitID.OVERLORD) < 1:
            return

        # Only in very early game
        heuristics = bot.heuristic_manager.get_state()
        if heuristics.game_phase >= _MAX_GAME_PHASE:
            self._phase = _Phase.DONE
            return

        # Need larva to queue the drone we're making room for
        if not bot.larva:
            return

        # Need minerals for extractor (75) + drone (50) = 125
        if bot.minerals < 125:
            return

        # Find a free geyser near main base
        if not bot.townhalls:
            return
        main = bot.townhalls.first
        geyser = self._find_free_geyser(bot, main.position)
        if geyser is None:
            return

        # Find the nearest available drone
        workers = bot.workers.closer_than(15, main.position)
        if not workers:
            return
        drone = workers.closest_to(geyser.position)

        # Order the drone to build the extractor
        drone(AbilityId.ZERGBUILD_EXTRACTOR, geyser)
        self._ordered_frame = frame
        self._phase = _Phase.BUILDING
        self._drone_queued = False

        log.game_event(
            "EXTRACTOR_TRICK",
            f"START: drone tag={drone.tag} → geyser at {geyser.position} "
            f"(supply={bot.supply_used}/{bot.supply_cap})",
            frame=frame,
        )

    def _check_extractor_started(self, bot: "ManifestorBot", frame: int) -> None:
        """Phase BUILDING → WAITING_DRONE: detect when the extractor appears."""
        # Timeout: if the extractor never appeared, abort
        if frame - self._ordered_frame > _BUILD_TIMEOUT_FRAMES:
            log.warning(
                "ExtractorTrick: BUILD timeout after %d frames — aborting",
                frame - self._ordered_frame,
                frame=frame,
            )
            self._phase = _Phase.DONE
            return

        # Look for an in-progress extractor (build_progress < 1.0)
        for ext in bot.structures(UnitID.EXTRACTOR):
            if ext.build_progress < 1.0:
                self._extractor_tag = ext.tag
                self._phase = _Phase.WAITING_DRONE

                # The drone morphed → we have 1 free supply. Queue a drone now.
                if bot.larva and bot.minerals >= 50:
                    larva = bot.larva.random
                    larva.train(UnitID.DRONE)
                    self._drone_queued = True
                    log.game_event(
                        "EXTRACTOR_TRICK",
                        f"DRONE QUEUED: extractor tag={ext.tag} "
                        f"(supply={bot.supply_used}/{bot.supply_cap})",
                        frame=frame,
                    )
                else:
                    log.warning(
                        "ExtractorTrick: extractor started but no larva/minerals for drone",
                        frame=frame,
                    )
                return

    def _try_cancel(self, bot: "ManifestorBot", frame: int) -> None:
        """Phase WAITING_DRONE → DONE: cancel the extractor to get the drone back."""
        if self._extractor_tag is None:
            self._phase = _Phase.DONE
            return

        # Find the extractor by tag
        ext = None
        for s in bot.structures(UnitID.EXTRACTOR):
            if s.tag == self._extractor_tag:
                ext = s
                break

        if ext is None:
            # Extractor disappeared (maybe killed?) — we're done
            log.warning(
                "ExtractorTrick: extractor tag=%d disappeared before cancel",
                self._extractor_tag,
                frame=frame,
            )
            self._phase = _Phase.DONE
            return

        # If extractor is complete, it's too late to cancel for free — leave it
        if ext.build_progress >= 1.0:
            log.warning(
                "ExtractorTrick: extractor completed before cancel — keeping it",
                frame=frame,
            )
            self._phase = _Phase.DONE
            return

        # Cancel the extractor (returns 75% minerals, drone pops back)
        ext(AbilityId.CANCEL_BUILDINPROGRESS)
        log.game_event(
            "EXTRACTOR_TRICK",
            f"CANCEL: extractor tag={ext.tag} progress={ext.build_progress:.0%} "
            f"drone_queued={self._drone_queued}",
            frame=frame,
        )
        self._phase = _Phase.DONE

    @staticmethod
    def _find_free_geyser(bot: "ManifestorBot", position) -> Optional["object"]:
        """Find the nearest unoccupied vespene geyser near a position."""
        candidates = bot.vespene_geyser.closer_than(_GEYSER_SEARCH_RADIUS, position)
        if not candidates:
            return None

        for geyser in candidates.sorted_by_distance_to(position):
            # Skip geysers with existing gas buildings
            if bot.gas_buildings.closer_than(_GEYSER_OCCUPIED_RADIUS, geyser.position):
                continue
            # Skip geysers with any structure on them
            if bot.structures.closer_than(_GEYSER_OCCUPIED_RADIUS, geyser.position):
                continue
            # Skip geysers used as shield extractors
            if hasattr(bot, "_shield_extractor_positions"):
                already_shielded = any(
                    geyser.position.distance_to(pos) < _GEYSER_OCCUPIED_RADIUS
                    for pos in bot._shield_extractor_positions
                )
                if already_shielded:
                    continue
            return geyser
        return None
