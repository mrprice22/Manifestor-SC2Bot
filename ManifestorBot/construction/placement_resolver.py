"""
PlacementResolver — wraps Ares placement APIs for the construction system.

Design philosophy
-----------------
We do NOT reinvent placement logic. Ares already has:
  - ``mediator.get_placements_dict``  — pre-computed grid-based placements
  - ``ai.request_zerg_placement()``   — async deferred placement (Zerg only)
  - ``mediator.request_building_placement()`` — sync placement query

For Zerg, ``request_zerg_placement`` is the right API: it defers to Ares'
ZergPlacementManager which handles worker selection and pathfinding internally.
Our job is to decide *what* to build and *near which base* — Ares figures out
the precise tile.

What we add
-----------
1. A synchronous fallback for cases where we need the placement now (e.g. to
   show the intended build site in commentary or for pre-flight validation).
2. A ``can_place_near()`` query so BuildAbility.can_use() can gate itself
   without triggering a morph command.
3. Centralized logging of all placement requests.

Usage
-----
    resolver = PlacementResolver()

    # Async deferred (preferred for Zerg — lets Ares handle worker selection)
    resolver.request_async(bot, UnitID.SPAWNINGPOOL, base_location=bot.start_location)

    # Synchronous query (for validation, not for issuing commands)
    pos = resolver.find_placement(bot, UnitID.SPAWNINGPOOL, near=bot.start_location)
    if pos:
        drone.build(UnitID.SPAWNINGPOOL, pos)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2

from ManifestorBot.logger import get_logger

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot

log = get_logger()


class PlacementResolver:
    """
    Placement resolution helper.

    Stateless — all context is passed in per-call.
    One instance lives on the bot as ``self.placement_resolver``.
    """

    # ------------------------------------------------------------------
    # Primary API: async deferred request (Zerg preferred path)
    # ------------------------------------------------------------------

    def request_async(
        self,
        bot: "ManifestorBot",
        structure_type: UnitID,
        base_location: Optional[Point2] = None,
        frame: int = 0,
    ) -> Optional["object"]:
        """
        Submit a placement request and return the actual dispatched worker, if known.

        For most structures: calls ``bot.request_zerg_placement()``, which queues
        the request for Ares' ZergPlacementManager. Ares selects its own worker
        asynchronously, so we return None (MorphTracker will detect whichever drone
        disappears near base_location and handle tracking).

        For EXTRACTOR: geysers require a Unit target, not a tile position, so Ares'
        placement API cannot be used. We call ``bot.select_build_worker()`` and
        ``worker.build_gas(geyser)`` directly, and return the worker so that
        BuildAbility can register it with MorphTracker under the correct drone tag.

        Returns
        -------
        Unit | None
            The dispatched worker (extractor only) or None (all other structures,
            or if the extractor dispatch failed).
        """
        if base_location is None:
            base_location = bot.start_location

        if structure_type == UnitID.EXTRACTOR:
            return self._build_extractor(bot, base_location, frame)

        log.debug(
            "PlacementResolver: async request %s near %s",
            structure_type.name,
            base_location,
            frame=frame,
        )
        bot.request_zerg_placement(base_location, structure_type)
        return None

    def _build_extractor(
        self,
        bot: "ManifestorBot",
        near: Point2,
        frame: int,
    ) -> Optional["object"]:
        """
        Find a free geyser near `near`, select a worker, and issue build_gas.

        Returns the worker Unit so the caller can register it with MorphTracker,
        or None if no geyser / worker was available (caller should abort the order).
        """
        existing_gas = bot.gas_buildings
        free_geysers = [
            g for g in bot.vespene_geyser.closer_than(12, near)
            if not existing_gas.closer_than(1.0, g)
        ]
        if not free_geysers:
            log.debug(
                "PlacementResolver: no free geyser near %s for EXTRACTOR",
                near,
                frame=frame,
            )
            return None

        geyser = min(free_geysers, key=lambda g: g.distance_to(near))
        worker = bot.select_build_worker(geyser.position)
        if worker is None:
            log.debug("PlacementResolver: no build worker available for EXTRACTOR", frame=frame)
            return None

        worker.build_gas(geyser)
        log.debug(
            "PlacementResolver: EXTRACTOR dispatched | drone=%s geyser=%s",
            worker.tag,
            geyser.tag,
            frame=frame,
        )
        return worker

    # ------------------------------------------------------------------
    # Secondary API: synchronous query (for validation / commentary)
    # ------------------------------------------------------------------

    def find_placement(
        self,
        bot: "ManifestorBot",
        structure_type: UnitID,
        near: Optional[Point2] = None,
    ) -> Optional[Point2]:
        """
        Find a valid placement tile for ``structure_type`` near ``near``.

        Does NOT issue any command. Returns the tile or None if no valid
        placement exists right now (e.g. all spots occupied).

        Uses ``mediator.request_building_placement()`` which is a synchronous
        query into the Ares placement solver.

        This is used by:
        - BuildAbility.can_use() to verify a placement exists before claiming
          a ConstructionOrder (avoids claiming an order we can't fulfil).
        - Commentary / logging to describe where a building will go.
        """
        if near is None:
            near = bot.start_location

        try:
            placement: Optional[Point2] = bot.mediator.request_building_placement(
                base_location=near,
                structure_type=structure_type,
            )
            return placement
        except Exception as exc:
            # Ares may raise if no placements are registered for this type yet.
            log.debug(
                "PlacementResolver.find_placement(%s) raised: %s",
                structure_type.name,
                exc,
            )
            return None

    def can_place_near(
        self,
        bot: "ManifestorBot",
        structure_type: UnitID,
        near: Optional[Point2] = None,
    ) -> bool:
        """
        Return True if at least one valid placement tile exists for
        ``structure_type`` near ``near``.

        Cheap guard used in BuildAbility.can_use().
        """
        return self.find_placement(bot, structure_type, near) is not None

    # ------------------------------------------------------------------
    # Best base selection
    # ------------------------------------------------------------------

    def best_base_for(
        self,
        bot: "ManifestorBot",
        structure_type: UnitID,
    ) -> Point2:
        """
        Choose the most appropriate base location to build ``structure_type`` at.

        Current heuristic:
        - Prefer the main base (start_location) for tech / eco buildings.
        - Prefer the nearest expansion that has a townhall for expansions.
        - Fall back to start_location if nothing better exists.

        This is intentionally simple and can be extended with strategy-specific
        logic (e.g. building a Spine Crawler at a threatened expansion).
        """
        # Tech and production buildings → main base
        _MAIN_BASE_ONLY = {
            UnitID.SPAWNINGPOOL,
            UnitID.EVOLUTIONCHAMBER,
            UnitID.LAIR,
            UnitID.HIVE,
            UnitID.SPIRE,
            UnitID.GREATERSPIRE,
            UnitID.HYDRALISKDEN,
            UnitID.ROACHWARREN,
            UnitID.BANELINGNEST,
            UnitID.ULTRALISKCAVERN,
            UnitID.INFESTATIONPIT,
            UnitID.NYDUSNETWORK,
            UnitID.LURKERDENMP,
        }
        if structure_type in _MAIN_BASE_ONLY:
            return bot.start_location

        # Extractors → nearest base with available gas
        if structure_type == UnitID.EXTRACTOR:
            for th in bot.townhalls.ready:
                nearby_gas = bot.vespene_geyser.closer_than(10, th.position)
                already_built = bot.gas_buildings.closer_than(10, th.position)
                if len(nearby_gas) > len(already_built):
                    return th.position
            return bot.start_location

        # Hatcheries → nearest unoccupied expansion to our existing bases.
        # Prefer expanding close to home (safer, more likely on creep).
        if structure_type == UnitID.HATCHERY:
            taken = {th.position for th in bot.townhalls}
            # Also exclude locations where a hatchery is already pending.
            pending_hatch = bot.units_and_structures.of_type(UnitID.HATCHERY).not_ready
            for ph in pending_hatch:
                taken.add(ph.position)

            free = [exp for exp in bot.expansion_locations_list if exp not in taken]
            if free:
                # Sort by distance to our closest existing townhall.
                if bot.townhalls:
                    free.sort(
                        key=lambda e: min(
                            e.distance_to(th.position) for th in bot.townhalls
                        )
                    )
                return free[0]

        return bot.start_location
