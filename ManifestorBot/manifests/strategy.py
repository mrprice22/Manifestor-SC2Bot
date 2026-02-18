"""
Named strategy definitions.

Each strategy represents a high-level plan that governs all unit decisions.
"""

from enum import Enum


class Strategy(Enum):
    """
    The complete strategy library.
    
    At any moment, the bot is operating under exactly one of these.
    Units know which strategy is active and can sacrifice themselves
    for its success if needed.
    """
    
    # Aggressive strategies
    JUST_GO_PUNCH_EM = "Just Go Punch 'Em"
    ALL_IN = "All-In"
    KEEP_EM_BUSY = "Keep 'Em Busy"
    WAR_ON_SANITY = "War on Sanity"
    
    # Balanced strategies
    STOCK_STANDARD = "Stock Standard"
    WAR_OF_ATTRITION = "War of Attrition"
    BLEED_OUT = "Bleed Out"
    
    # Defensive/Economic strategies
    DRONE_ONLY_FORTRESS = "Drone Only (Fortress)"
    
    def is_aggressive(self) -> bool:
        """Is this an aggressive strategy?"""
        return self in {
            Strategy.JUST_GO_PUNCH_EM,
            Strategy.ALL_IN,
            Strategy.KEEP_EM_BUSY,
            Strategy.WAR_ON_SANITY,
        }
        
    def is_defensive(self) -> bool:
        """Is this a defensive/economic strategy?"""
        return self in {
            Strategy.DRONE_ONLY_FORTRESS,
        }
        
    def is_balanced(self) -> bool:
        """Is this a balanced/reactive strategy?"""
        return not (self.is_aggressive() or self.is_defensive())
