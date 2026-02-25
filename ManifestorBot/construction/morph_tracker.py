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

        # order._order_id → frame when the order transitioned to BUILDING
        # Used for the grace period in _detect_cancelled_buildings so we don't
        # fire before the in-progress structure has had a frame to appear.
        self._building_start_frames: Dict[int, int] = {}

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

        # 4b. Detect BUILDING orders whose in-progress structure was cancelled
        self._detect_cancelled_buildings(bot, queue, frame)

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
                log.debug(
                    "MorphTracker: registered claim — %s by drone=%d",
                    order.structure_type.name,
                    order.claimed_by,
                    frame=bot.state.game_loop,
                )

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
                self._building_start_frames[order._order_id] = frame

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

        FIX (2026-02-23): Previously crashed with `list.remove(x): x not in list`
        when two structures of the same type appeared in the same tick (e.g. a
        BANELINGNEST that was already present in bot.structures when the tracker
        first saw it). The inner struct loop could match the same `closest_order`
        twice — the first removal succeeded, the second raised ValueError.

        Two guards added:
        1. Re-filter `building_orders` inside the struct loop to exclude orders
           whose status is already DONE (mutated by mark_done() on a prior struct).
        2. Check `closest_order in orders` before calling `orders.remove()` as a
           belt-and-suspenders defence against any other path that might remove
           the order first.
        """
        if not self._watching_for:
            return

        for structure_type, orders in list(self._watching_for.items()):
            # Look for a newly appeared structure of this type
            matching_structures = bot.structures(structure_type)
            for struct in matching_structures:
                if struct.tag in self._completed_structure_tags:
                    continue

                # Re-filter each iteration so we only consider orders that are
                # still genuinely BUILDING (not already marked DONE this tick).
                building_orders = [o for o in orders if o.status == OrderStatus.BUILDING]
                if not building_orders:
                    log.debug(
                        "MorphTracker: %s tag=%d appeared but no BUILDING orders — "
                        "may be a pre-existing structure not tracked by this system",
                        structure_type.name,
                        struct.tag,
                        frame=frame,
                    )
                    # Still mark it seen so we don't log this every tick
                    self._completed_structure_tags.add(struct.tag)
                    continue

                # Find the closest BUILDING order to this structure
                closest_order = self._find_matching_order(struct, building_orders)
                if closest_order is None:
                    log.debug(
                        "MorphTracker: %s tag=%d matched no order within radius %.0f — skipping",
                        structure_type.name,
                        struct.tag,
                        _COMPLETION_RADIUS,
                        frame=frame,
                    )
                    continue

                queue.mark_done(closest_order)
                self._completed_structure_tags.add(struct.tag)
                self._building_start_frames.pop(closest_order._order_id, None)

                # Guard: only remove if still present (prevents ValueError if
                # another code path already removed it this tick)
                if closest_order in orders:
                    orders.remove(closest_order)
                else:
                    log.warning(
                        "MorphTracker: closest_order for %s tag=%d was already removed "
                        "from _watching_for — double-completion avoided",
                        structure_type.name,
                        struct.tag,
                        frame=frame,
                    )

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

        NOTE — Ares worker-selection mismatch:
        BuildAbility records the *requesting* drone's tag as claimed_by, but
        Ares' async placement resolver may dispatch a *different* (closer) drone.
        The requesting drone has no build order and would otherwise trigger a
        false FAILED transition every ~15 frames.

        Guard: before marking FAILED, check whether a structure of the expected
        type is already appearing near the base_location.  If one is found, Ares
        already sent a worker and the morph is in progress — transition directly
        to BUILDING instead of FAILED.
        """
        _GRACE_FRAMES = 200  # allow Ares async placement time (~9s) — the real drone
        # may still be walking; structures appear 28–52 frames after false-abandon
        current_unit_tags: Set[int] = {u.tag for u in bot.units}

        for order in queue.active():
            if order.status != OrderStatus.CLAIMED:
                continue
            if order.claimed_by is None:
                continue
            if order.claimed_by not in current_unit_tags:
                continue  # handled by _detect_morph_starts as disappeared

            # Don't check until the drone has had time to receive the command
            if order.dispatched_frame and (frame - order.dispatched_frame) <= _GRACE_FRAMES:
                continue

            # Drone is still alive — check if it still has a build order
            drone = bot.units.find_by_tag(order.claimed_by)
            if drone is None:
                continue

            if not self._drone_has_build_order(drone):
                # Before marking FAILED: check whether Ares sent a different drone
                # and the build has already started (structure appearing in bot.structures).
                in_progress = any(
                    s.build_progress < 1.0
                    and s.position.distance_to(order.base_location) < _COMPLETION_RADIUS
                    for s in bot.structures(order.structure_type)
                )
                if in_progress:
                    # Ares dispatched a different worker — morph is underway.
                    # Transition to BUILDING so _detect_morph_completions handles it.
                    queue.mark_building(order)
                    self._building_start_frames[order._order_id] = frame
                    if order.claimed_by in self._pending_morphs:
                        del self._pending_morphs[order.claimed_by]
                    log.debug(
                        "MorphTracker: CLAIMED drone %s has no build order but structure "
                        "%s is already in-progress — Ares sent a different worker; "
                        "transitioning to BUILDING",
                        order.claimed_by,
                        order.structure_type.name,
                        frame=frame,
                    )
                    continue

                # Genuine failure — drone abandoned the build with no structure appearing
                log.warning(
                    "Drone %s abandoned build order for %s — marking failed",
                    order.claimed_by,
                    order.structure_type.name,
                    frame=frame,
                )
                queue.mark_failed(order)
                if order.claimed_by in self._pending_morphs:
                    del self._pending_morphs[order.claimed_by]

    def _detect_cancelled_buildings(
        self,
        bot: "ManifestorBot",
        queue: "ConstructionQueue",
        frame: int,
    ) -> None:
        """
        Detect BUILDING orders whose in-progress structure was cancelled or destroyed.

        When a Zerg building is cancelled mid-construction (e.g. by
        CancelDyingBuildingTactic), the structure disappears from bot.structures
        and the drone pops back out.  Since _detect_morph_completions only fires
        when a structure APPEARS, a cancelled BUILDING order would otherwise stay
        stuck forever, permanently blocking re-queuing of that structure type.

        We scan all BUILDING orders; if no in-progress structure of the expected
        type exists near the base_location, the build was cancelled/destroyed.
        We mark it FAILED so the tactic loop can re-queue on the next cycle.

        Grace period: we skip orders that JUST transitioned to BUILDING (within
        _BUILDING_GRACE_FRAMES) to avoid a false positive before the in-progress
        structure has appeared in bot.structures for the first time.
        """
        _BUILDING_GRACE_FRAMES = 5  # ~0.25 s at fastest speed

        for structure_type, orders in list(self._watching_for.items()):
            for order in list(orders):
                if order.status != OrderStatus.BUILDING:
                    continue

                # Grace: structure may not have appeared yet this frame
                start_frame = self._building_start_frames.get(order._order_id, 0)
                if frame - start_frame < _BUILDING_GRACE_FRAMES:
                    continue

                nearby = list(bot.structures(structure_type))

                # Still morphing — all good
                still_in_progress = any(
                    s.build_progress < 1.0
                    and s.position.distance_to(order.base_location) < _COMPLETION_RADIUS
                    for s in nearby
                )
                if still_in_progress:
                    continue

                # Check if the structure already completed (build_progress == 1.0) but
                # was never marked DONE (e.g. tag was added to _completed_structure_tags
                # as "pre-existing" before the order reached BUILDING state).
                already_complete = next(
                    (
                        s for s in nearby
                        if s.build_progress >= 1.0
                        and s.position.distance_to(order.base_location) < _COMPLETION_RADIUS
                        and s.tag not in self._completed_structure_tags
                    ),
                    None,
                )
                if already_complete is not None:
                    queue.mark_done(order)
                    self._completed_structure_tags.add(already_complete.tag)
                    self._building_start_frames.pop(order._order_id, None)
                    if order in orders:
                        orders.remove(order)
                    log.debug(
                        "MorphTracker: %s BUILDING order completed without DONE event "
                        "(tag=%d) — marked DONE retroactively",
                        structure_type.name,
                        already_complete.tag,
                        frame=frame,
                    )
                    continue

                # No in-progress structure and no untracked complete structure —
                # the build was cancelled or the building was destroyed.
                log.warning(
                    "MorphTracker: %s BUILDING order (id=%d near %s) has no matching "
                    "in-progress structure — assumed cancelled; marking FAILED for re-queue",
                    structure_type.name,
                    order._order_id,
                    order.base_location,
                    frame=frame,
                )
                queue.mark_failed(order)
                self._building_start_frames.pop(order._order_id, None)
                if order in orders:
                    orders.remove(order)

    # ------------------------------------------------------------------
    # External API — called by BuildAbility after dispatching a worker
    # ------------------------------------------------------------------

    def register_claim(self, order: ConstructionOrder, drone_tag: int) -> None:
        """
        Called by BuildAbility immediately after dispatching a drone.
        Ensures the tracker knows about this claim before update() runs.
        Note: drone_tag is the *requesting* drone; Ares may send a different
        worker.  _detect_failed_claims handles the mismatch gracefully.
        """
        self._pending_morphs[drone_tag] = order
        self._watching_for.setdefault(order.structure_type, [])
        if order not in self._watching_for[order.structure_type]:
            self._watching_for[order.structure_type].append(order)
        log.debug(
            "MorphTracker.register_claim: %s claimed by drone=%d",
            order.structure_type.name,
            drone_tag,
        )

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