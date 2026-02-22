"""
MorphTracker — the cleanup contract for drone morphs.

Why this exists
---------------
When a Zerg drone morphs into a building, it ceases to exist as a unit.
Its tag disappears from ``bot.units`` and reappears (eventually) as a
structure tag. Without explicit tracking, the bot will:

  - Have a ghost entry in ``suppressed_ideas`` forever (leaks memory, skews
    cooldown logic for the next drone to take that slot).
  - Have an incorrect worker count (Ares' saturation math will think we still
    have that drone and under-produce replacements).
  - Leave a ConstructionOrder stuck in CLAIMED state if the drone dies mid-morph.

MorphTracker.update() is called once per on_step() tick, before the idea loop,
so that by the time tactics run the world state is already clean.

Integration with Ares
---------------------
For Zerg, Ares' own building tracker (``not_started_but_in_building_tracker``)
already counts drones on route. We shadow it with our own records so that the
ConstructionQueue can do its lifecycle management without needing to poke Ares
internals. The two systems are complementary, not competing.

Lifecycle events detected
-------------------------
1. Drone disappears from bot.units while its order is CLAIMED
   → mark_building() on its ConstructionOrder
   → clean up suppressed_ideas entry

2. A new structure of the expected type appears near the base_location
   → mark_done() on the corresponding BUILDING order

3. A CLAIMED order times out (drone dispatched but nothing happened)
   → mark_failed() and log a warning; re-queued automatically

4. A CLAIMED drone is still in bot.units but has lost its build order
   (worker was attacked and retreated, or the order was cancelled)
   → mark_failed() so a replacement worker can be dispatched
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Set

from sc2.ids.unit_typeid import UnitTypeId as UnitID

from ManifestorBot.construction.construction_queue import ConstructionOrder, OrderStatus
from ManifestorBot.logger import get_logger

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot
    from ManifestorBot.construction.construction_queue import ConstructionQueue

log = get_logger()

# How close (game units) a new structure must appear to the target base_location
# to be considered the completion of an order.
_COMPLETION_RADIUS: float = 15.0

# After this many frames a CLAIMED order with no activity is failed.
_CLAIM_TIMEOUT_FRAMES: int = 448   # ~20 seconds


class MorphTracker:
    """
    Watches drone→building transitions and keeps the ConstructionQueue
    and bot state consistent.

    One instance lives on the bot as ``self.morph_tracker``.
    Call ``update(bot)`` once per on_step(), before the idea loop.
    """

    def __init__(self) -> None:
        # drone_tag → ConstructionOrder (for fast lookup when a tag disappears)
        self._pending_morphs: Dict[int, ConstructionOrder] = {}

        # structure_types we're watching for completion
        # Maps structure_type → set of active ConstructionOrders
        self._watching_for: Dict[UnitID, list] = {}

        # Tags we've seen as completed structures (to avoid double-processing)
        self._completed_structure_tags: Set[int] = set()

    # ------------------------------------------------------------------
    # Main update — call once per on_step() before the idea loop
    # ------------------------------------------------------------------

    def update(self, bot: "ManifestorBot") -> None:
        frame = bot.state.game_loop
        queue: ConstructionQueue = bot.construction_queue

        # 1. Register any newly claimed orders into our tracking dicts
        self._sync_claimed_orders(queue, bot)

        # 2. Detect drones that have disappeared (morph started)
        self._detect_morph_starts(bot, queue, frame)

        # 3. Detect completed structures (morph finished)
        self._detect_morph_completions(bot, queue, frame)

        # 4. Detect failed claims (worker still alive but lost build order)
        self._detect_failed_claims(bot, queue, frame)

        # 5. Timeout any orders stuck in CLAIMED too long
        timed_out = queue.timeout_stuck_orders(frame, _CLAIM_TIMEOUT_FRAMES)
        for order in timed_out:
            log.warning(
                "Construction order timed out: %s (claimed_by=%s, dispatched=%s)",
                order.structure_type.name,
                order.claimed_by,
                order.dispatched_frame,
                frame=frame,
            )
            # Clean up tracking dicts
            if order.claimed_by in self._pending_morphs:
                del self._pending_morphs[order.claimed_by]

        # 6. Prune completed/failed orders periodically
        if frame % 224 == 0:
            queue.prune_done_and_failed()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_claimed_orders(
        self,
        queue: "ConstructionQueue",
        bot: "ManifestorBot",
    ) -> None:
        """Register newly CLAIMED orders into _pending_morphs."""
        for order in queue.active():
            if (
                order.status == OrderStatus.CLAIMED
                and order.claimed_by is not None
                and order.claimed_by not in self._pending_morphs
            ):
                self._pending_morphs[order.claimed_by] = order
                # Start watching for this structure type
                self._watching_for.setdefault(order.structure_type, [])
                if order not in self._watching_for[order.structure_type]:
                    self._watching_for[order.structure_type].append(order)

    def _detect_morph_starts(
        self,
        bot: "ManifestorBot",
        queue: "ConstructionQueue",
        frame: int,
    ) -> None:
        """
        Check which tracked drone tags have disappeared from bot.units.
        A disappeared drone means the morph has started.
        """
        current_unit_tags: Set[int] = {u.tag for u in bot.units}

        for drone_tag, order in list(self._pending_morphs.items()):
            if order.status != OrderStatus.CLAIMED:
                continue
            if drone_tag not in current_unit_tags:
                # Drone is gone — morph started (or drone died)
                queue.mark_building(order)

                # Clean up suppressed_ideas so the slot isn't leaked
                bot.suppressed_ideas.pop(drone_tag, None)

                log.game_event(
                    "MORPH_STARTED",
                    f"{order.structure_type.name} | drone={drone_tag}",
                    frame=frame,
                )
                # Remove from pending_morphs; now watching for completion
                del self._pending_morphs[drone_tag]

    def _detect_morph_completions(
        self,
        bot: "ManifestorBot",
        queue: "ConstructionQueue",
        frame: int,
    ) -> None:
        """
        Check bot.structures for new structures that match BUILDING orders.
        """
        if not self._watching_for:
            return

        for structure_type, orders in list(self._watching_for.items()):
            building_orders = [o for o in orders if o.status == OrderStatus.BUILDING]
            if not building_orders:
                continue

            # Look for a newly appeared structure of this type
            matching_structures = bot.structures(structure_type)
            for struct in matching_structures:
                if struct.tag in self._completed_structure_tags:
                    continue

                # Find the closest BUILDING order to this structure
                closest_order = self._find_matching_order(struct, building_orders)
                if closest_order is None:
                    continue

                queue.mark_done(closest_order)
                self._completed_structure_tags.add(struct.tag)
                orders.remove(closest_order)

                log.game_event(
                    "MORPH_COMPLETE",
                    f"{structure_type.name} tag={struct.tag}",
                    frame=frame,
                )

    def _detect_failed_claims(
        self,
        bot: "ManifestorBot",
        queue: "ConstructionQueue",
        frame: int,
    ) -> None:
        """
        Detect workers that are still alive but have abandoned their build order.
        This happens when a drone is attacked while traveling to the build site
        and retreats, or when the player (or another system) cancels the order.
        """
        _GRACE_FRAMES = 15  # allow Ares time to issue the command
        current_unit_tags: Set[int] = {u.tag for u in bot.units}

        for order in queue.active():
            if order.status != OrderStatus.CLAIMED:
                continue
            if order.claimed_by is None:
                continue
            if order.claimed_by not in current_unit_tags:
                continue  # handled by _detect_morph_starts as disappeared

            #Don't check until the drone has had time to receive the command
            if order.dispatched_frame and (frame - order.dispatched_frame) < _GRACE_FRAMES:
                continue

            # Drone is still alive — check if it still has a build order
            drone = bot.units.find_by_tag(order.claimed_by)
            if drone is None:
                continue

            if not self._drone_has_build_order(drone):
                # Drone abandoned the build
                log.warning(
                    "Drone %s abandoned build order for %s — marking failed",
                    order.claimed_by,
                    order.structure_type.name,
                    frame=frame,
                )
                queue.mark_failed(order)
                if order.claimed_by in self._pending_morphs:
                    del self._pending_morphs[order.claimed_by]

    # ------------------------------------------------------------------
    # External API — called by BuildAbility after dispatching a worker
    # ------------------------------------------------------------------

    def register_claim(self, order: ConstructionOrder, drone_tag: int) -> None:
        """
        Called by BuildAbility immediately after dispatching a drone.
        Ensures the tracker knows about this claim before update() runs.
        """
        self._pending_morphs[drone_tag] = order
        self._watching_for.setdefault(order.structure_type, [])
        if order not in self._watching_for[order.structure_type]:
            self._watching_for[order.structure_type].append(order)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _find_matching_order(
        self,
        structure,
        orders: list,
    ) -> Optional[ConstructionOrder]:
        """
        Match a completed structure to the closest BUILDING order within
        _COMPLETION_RADIUS of the structure's base_location.
        """
        best: Optional[ConstructionOrder] = None
        best_dist = float("inf")

        for order in orders:
            dist = structure.position.distance_to(order.base_location)
            if dist < _COMPLETION_RADIUS and dist < best_dist:
                best_dist = dist
                best = order

        return best

    @staticmethod
    def _drone_has_build_order(drone) -> bool:
        """
        True if the drone currently has a build/morph command in its order queue.
        Build orders use abilities whose names contain "BUILD" or "ZERGBUILD".
        """
        from sc2.ids.ability_id import AbilityId
        _BUILD_ABILITY_NAMES = {"BUILD", "ZERGBUILD", "MORPH"}
        for order in drone.orders:
            ability_name = order.ability.id.name.upper()
            if any(kw in ability_name for kw in _BUILD_ABILITY_NAMES):
                return True
        return False
