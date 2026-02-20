"""
ManifestorBot.construction — drone morphing and building placement system.

Public API
----------
    from ManifestorBot.construction import (
        ConstructionQueue,
        ConstructionOrder,
        OrderStatus,
        MorphTracker,
        PlacementResolver,
    )
    from ManifestorBot.construction.build_ability import (
        BuildAbility,
        BuildingTactic,
        register_construction_abilities,
    )
"""

from ManifestorBot.construction.construction_queue import (
    ConstructionQueue,
    ConstructionOrder,
    OrderStatus,
)
from ManifestorBot.construction.morph_tracker import MorphTracker
from ManifestorBot.construction.placement_resolver import PlacementResolver

__all__ = [
    "ConstructionQueue",
    "ConstructionOrder",
    "OrderStatus",
    "MorphTracker",
    "PlacementResolver",
]
