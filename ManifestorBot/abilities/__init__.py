"""
ManifestorBot.abilities — unit ability system.

Public API
----------
    from ManifestorBot.abilities import Ability, AbilityContext
    from ManifestorBot.abilities import ability_registry, ability_selector
    from ManifestorBot.abilities.worker_abilities import register_worker_abilities
"""

from ManifestorBot.abilities.ability import Ability, AbilityContext
from ManifestorBot.abilities.ability_registry import ability_registry, AbilityRegistry
from ManifestorBot.abilities.ability_selector import ability_selector, AbilitySelector

__all__ = [
    "Ability",
    "AbilityContext",
    "AbilityRegistry",
    "ability_registry",
    "AbilitySelector",
    "ability_selector",
]
