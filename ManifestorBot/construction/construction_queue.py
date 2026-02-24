"""
ConstructionQueue — the pending build order ledger.

Responsibility
--------------
This is the single source of truth for "what the strategy layer wants built
but has not yet dispatched to a worker". It decouples the *decision* layer
(BuildingTacticModule saying "we need a Spawning Pool") from the *execution*
layer (a drone actually morphing into one).

Lifecycle of a ConstructionOrder
---------------------------------
  PENDING   → created by a BuildingTacticModule via enqueue()
  CLAIMED   → a BuildAbility (worker-level) has selected this order and
               dispatched the drone; claimed_by is set to the drone's tag
  BUILDING  → MorphTracker confirmed the drone left the unit pool (morph started);
               the drone tag is now tracked as a disappearing unit
  DONE      → MorphTracker confirmed the finished structure appeared in bot.structures

Only PENDING orders are visible to the BuildingTactic. CLAIMED, BUILDING,
and DONE orders are managed by the MorphTracker.

Deduplication
-------------
enqueue() silently drops a new order if an order for the same structure type
is already PENDING or CLAIMED. This prevents the strategy layer from queuing
the same building ten times before a worker responds.

The queue is intentionally simple — a list, not a heap. There will never be
more than a handful of pending orders at once, so O(n) scans are free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2


# ---------------------------------------------------------------------------
# Order state machine
# ---------------------------------------------------------------------------

class OrderStatus(Enum):
    PENDING  = auto()   # Waiting for a worker to claim it
    CLAIMED  = auto()   # Worker dispatched; drone tag in claimed_by
    BUILDING = auto()   # Drone disappeared — morph in progress
    DONE     = auto()   # Finished structure confirmed
    FAILED   = auto()   # Worker died before completing; needs re-queue


# ---------------------------------------------------------------------------
# ConstructionOrder
# ---------------------------------------------------------------------------

@dataclass
class ConstructionOrder:
    """
    A single pending or in-progress build request.

    Fields
    ------
    structure_type : UnitTypeId
        What to build (e.g. UnitID.SPAWNINGPOOL).
    base_location : Point2
        Which base to build near. Passed to Ares placement resolver.
    priority : int
        Higher = more urgent. BuildingTactic uses this to order candidates
        when multiple orders are pending and multiple drones are free.
    claimed_by : int | None
        Tag of the drone that has been dispatched to build this. None = PENDING.
    status : OrderStatus
        Current lifecycle state.
    created_frame : int
        Game loop frame when the order was enqueued. Used for timeout detection.
    dispatched_frame : int | None
        Frame when a worker was sent. Used to detect stuck workers.
    """
    structure_type: UnitID
    base_location: Point2
    priority: int = 50
    claimed_by: Optional[int] = None
    status: OrderStatus = OrderStatus.PENDING
    created_frame: int = 0
    dispatched_frame: Optional[int] = None

    # Internal: unique id for logging (auto-assigned by queue)
    _order_id: int = field(default=0, repr=False, compare=False)

    @property
    def is_available(self) -> bool:
        """True if a worker can claim this order right now."""
        return self.status == OrderStatus.PENDING

    @property
    def is_active(self) -> bool:
        """True if this order is in-flight (claimed or building)."""
        return self.status in (OrderStatus.CLAIMED, OrderStatus.BUILDING)

    def __str__(self) -> str:
        return (
            f"Order#{self._order_id}({self.structure_type.name} "
            f"@ {self.base_location} | {self.status.name} "
            f"claimed_by={self.claimed_by})"
        )


# ---------------------------------------------------------------------------
# ConstructionQueue
# ---------------------------------------------------------------------------

class ConstructionQueue:
    """
    Ordered list of pending construction orders.

    Thread-safety: not required (SC2 bots are single-threaded).

    Intended usage
    --------------
    Bot holds one instance as ``self.construction_queue``.

    BuildingTacticModules write:
        bot.construction_queue.enqueue(ConstructionOrder(...))

    BuildingTactic / BuildAbility read:
        order = bot.construction_queue.next_pending()

    MorphTracker updates:
        bot.construction_queue.mark_claimed(order, drone_tag, frame)
        bot.construction_queue.mark_building(order)
        bot.construction_queue.mark_done(order)
        bot.construction_queue.mark_failed(order)     # re-queues automatically
    """

    def __init__(self) -> None:
        self._orders: List[ConstructionOrder] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Write API — called by BuildingTacticModules
    # ------------------------------------------------------------------

    def enqueue(self, order: ConstructionOrder) -> bool:
        """
        Add an order to the queue, unless a duplicate is already active.

        Returns True if the order was accepted, False if it was silently
        dropped as a duplicate.

        Duplicate definition: same structure_type, status is PENDING or CLAIMED
        (we allow multiple BUILDING/DONE of the same type — e.g. two pools).
        """
        for existing in self._orders:
            if (
                existing.structure_type == order.structure_type
                and existing.status in (OrderStatus.PENDING, OrderStatus.CLAIMED)
            ):
                return False  # already queued

        order._order_id = self._next_id
        self._next_id += 1
        self._orders.append(order)
        return True

    # ------------------------------------------------------------------
    # Read API — called by BuildingTactic / BuildAbility
    # ------------------------------------------------------------------

    def pending(self) -> List[ConstructionOrder]:
        """All orders with status PENDING, sorted by descending priority."""
        return sorted(
            [o for o in self._orders if o.status == OrderStatus.PENDING],
            key=lambda o: o.priority,
            reverse=True,
        )

    def next_pending(self) -> Optional[ConstructionOrder]:
        """Highest-priority PENDING order, or None."""
        p = self.pending()
        return p[0] if p else None

    def active(self) -> List[ConstructionOrder]:
        """All CLAIMED or BUILDING orders."""
        return [o for o in self._orders if o.is_active]

    def claimed_by_drone(self, drone_tag: int) -> Optional[ConstructionOrder]:
        """Return the order claimed by a specific drone tag, if any."""
        for o in self._orders:
            if o.claimed_by == drone_tag and o.is_active:
                return o
        return None

    def has_pending(self) -> bool:
        return any(o.status == OrderStatus.PENDING for o in self._orders)

    def count_active_of_type(self, structure_type: UnitID) -> int:
        """How many orders for this type are PENDING, CLAIMED, or BUILDING."""
        return sum(
            1 for o in self._orders
            if o.structure_type == structure_type
            and o.status in (OrderStatus.PENDING, OrderStatus.CLAIMED, OrderStatus.BUILDING)
        )

    # ------------------------------------------------------------------
    # Lifecycle transitions — called by MorphTracker
    # ------------------------------------------------------------------

    def mark_claimed(
        self,
        order: ConstructionOrder,
        drone_tag: int,
        frame: int,
    ) -> None:
        """Worker dispatched to this order."""
        order.claimed_by = drone_tag
        order.status = OrderStatus.CLAIMED
        order.dispatched_frame = frame

    def mark_building(self, order: ConstructionOrder) -> None:
        """Drone has disappeared — morph confirmed started."""
        order.status = OrderStatus.BUILDING

    def mark_done(self, order: ConstructionOrder) -> None:
        """Finished structure confirmed in bot.structures."""
        order.status = OrderStatus.DONE

    def mark_failed(self, order: ConstructionOrder) -> None:
        """
        Worker died or lost its build order before completing.

        Sets status to FAILED so the order is no longer active.
        Does NOT auto-requeue — the tactic loop (generate_idea) will
        notice the structure count is below cap and create a fresh order
        with proper cap checking.  Auto-requeuing here previously bypassed
        the cap check, causing runaway building (e.g. 10 Evolution Chambers
        instead of the intended max of 2).
        """
        order.status = OrderStatus.FAILED

    # ------------------------------------------------------------------
    # Maintenance — called by MorphTracker.update()
    # ------------------------------------------------------------------

    def prune_done_and_failed(self, keep_last: int = 10) -> None:
        """
        Remove DONE and FAILED orders from the ledger to prevent unbounded growth.

        Keeps the last `keep_last` completed orders for post-game inspection.
        """
        completed = [
            o for o in self._orders
            if o.status in (OrderStatus.DONE, OrderStatus.FAILED)
        ]
        active = [
            o for o in self._orders
            if o.status not in (OrderStatus.DONE, OrderStatus.FAILED)
        ]
        self._orders = active + completed[-keep_last:]

    def timeout_stuck_orders(self, current_frame: int, timeout_frames: int = 448) -> List[ConstructionOrder]:
        """
        Detect CLAIMED orders where the drone has been dispatched but nothing
        happened for too long. Returns timed-out orders (already marked FAILED).

        448 frames ≈ 20 seconds at normal speed — ample time for a drone to
        reach a build site and morph.
        """
        timed_out = []
        for order in self._orders:
            if order.status != OrderStatus.CLAIMED:
                continue
            if order.dispatched_frame is None:
                continue
            if current_frame - order.dispatched_frame > timeout_frames:
                order.status = OrderStatus.FAILED
                timed_out.append(order)
        return timed_out

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def summary(self) -> str:
        if not self._orders:
            return "(empty)"
        lines = []
        for o in self._orders:
            if o.status in (OrderStatus.DONE, OrderStatus.FAILED):
                continue
            lines.append(f"  {o}")
        return "\n".join(lines) if lines else "(all complete)"
