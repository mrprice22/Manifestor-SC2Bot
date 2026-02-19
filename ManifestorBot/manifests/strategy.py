"""
Named strategy definitions and their tactical profiles.

Each strategy represents a high-level plan that governs all unit decisions.
Strategies publish a TacticalProfile — a set of additive bias values that
tactics consume when calculating confidence. Tactics never inspect the
strategy enum directly; they only see the profile.

This keeps the relationship clean:
  - Adding a new strategy = define its TacticalProfile
  - Adding a new tactic = consume the relevant bias fields
  - No combinatorial if/else chains anywhere
"""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class TacticalProfile:
    """
    A strategy's published preferences for unit behavior.

    All values are additive biases applied to tactic confidence scores.
    Positive = this strategy encourages this behavior.
    Negative = this strategy discourages this behavior.

    Typical range is -0.4 to +0.4. Values outside ±0.5 are intentional
    and signal that this strategy strongly commits to a direction.

    Fields:
        engage_bias:    Bias toward fighting (StutterForward, FlankTactic)
        retreat_bias:   Bias toward retreating (KeepUnitSafe)
        harass_bias:    Bias toward worker/poke harassment (HarassWorkers)
        cohesion_bias:  Bias toward staying grouped (RallyToArmy)
        hold_bias:      Bias toward holding positions/chokes (HoldChoke)
        sacrifice_ok:   Whether units can trade themselves freely.
                        When True, retreat confidence is further reduced
                        and engage confidence gets a small bonus for
                        units at low health.
    """
    engage_bias:   float = 0.0
    retreat_bias:  float = 0.0
    harass_bias:   float = 0.0
    cohesion_bias: float = 0.0
    hold_bias:     float = 0.0
    sacrifice_ok:  bool  = False


class Strategy(Enum):
    """
    The complete strategy library.

    At any moment, the bot is operating under exactly one of these.
    Call .profile() to get the TacticalProfile for the current strategy.
    """

    # Aggressive strategies
    JUST_GO_PUNCH_EM = "Just Go Punch 'Em"
    ALL_IN           = "All-In"
    KEEP_EM_BUSY     = "Keep 'Em Busy"
    WAR_ON_SANITY    = "War on Sanity"

    # Balanced strategies
    STOCK_STANDARD   = "Stock Standard"
    WAR_OF_ATTRITION = "War of Attrition"
    BLEED_OUT        = "Bleed Out"

    # Defensive/Economic strategies
    DRONE_ONLY_FORTRESS = "Drone Only (Fortress)"

    # ------------------------------------------------------------------ #
    # Classification helpers
    # ------------------------------------------------------------------ #

    def is_aggressive(self) -> bool:
        return self in {
            Strategy.JUST_GO_PUNCH_EM,
            Strategy.ALL_IN,
            Strategy.KEEP_EM_BUSY,
            Strategy.WAR_ON_SANITY,
        }

    def is_defensive(self) -> bool:
        return self in {Strategy.DRONE_ONLY_FORTRESS}

    def is_balanced(self) -> bool:
        return not (self.is_aggressive() or self.is_defensive())

    # ------------------------------------------------------------------ #
    # Tactical profile
    # ------------------------------------------------------------------ #

    def profile(self) -> TacticalProfile:
        """
        Return the TacticalProfile for this strategy.

        This is the *only* place where strategy identity leaks into
        the tactical layer. Everything downstream works on the profile,
        not the enum value.
        """
        return _PROFILES[self]


# ------------------------------------------------------------------ #
# Profile definitions — one per strategy
# Keeping these outside the enum body avoids forward-reference issues.
# ------------------------------------------------------------------ #

_PROFILES: dict[Strategy, TacticalProfile] = {

    # ---- JUST_GO_PUNCH_EM ----------------------------------------- #
    # Full commitment. March in, fight, win or die. No patience for
    # dancing around or preserving units for later.
    Strategy.JUST_GO_PUNCH_EM: TacticalProfile(
        engage_bias   = +0.35,
        retreat_bias  = -0.40,
        harass_bias   = -0.10,  # don't waste time poking, commit fully
        cohesion_bias = +0.10,  # stay together for the push
        hold_bias     = -0.20,  # don't hold — go forward
        sacrifice_ok  = True,
    ),

    # ---- ALL_IN --------------------------------------------------- #
    # Everything committed right now. Even more aggressive than PUNCH_EM
    # because we can't recover from this — units are pure currency.
    Strategy.ALL_IN: TacticalProfile(
        engage_bias   = +0.45,
        retreat_bias  = -0.50,
        harass_bias   = -0.20,  # no side missions, go for the kill
        cohesion_bias = -0.10,  # splitting pressure is fine
        hold_bias     = -0.30,
        sacrifice_ok  = True,
    ),

    # ---- KEEP_EM_BUSY --------------------------------------------- #
    # Constant pressure and harassment to prevent them from droning up.
    # Engage when favorable, poke otherwise, never fully commit.
    Strategy.KEEP_EM_BUSY: TacticalProfile(
        engage_bias   = +0.20,
        retreat_bias  = -0.15,
        harass_bias   = +0.35,  # harassment is the whole point
        cohesion_bias = -0.15,  # spread out to maximize harassment vectors
        hold_bias     = -0.10,
        sacrifice_ok  = False,
    ),

    # ---- WAR_ON_SANITY -------------------------------------------- #
    # Multi-pronged simultaneous pressure everywhere at once.
    # Units should split and cause chaos, not ball up.
    Strategy.WAR_ON_SANITY: TacticalProfile(
        engage_bias   = +0.25,
        retreat_bias  = -0.20,
        harass_bias   = +0.25,
        cohesion_bias = -0.30,  # actively want units spread out
        hold_bias     = -0.15,
        sacrifice_ok  = False,
    ),

    # ---- STOCK_STANDARD ------------------------------------------- #
    # Neutral. React to the game state. No strong biases.
    Strategy.STOCK_STANDARD: TacticalProfile(
        engage_bias   = 0.0,
        retreat_bias  = 0.0,
        harass_bias   = 0.0,
        cohesion_bias = 0.0,
        hold_bias     = 0.0,
        sacrifice_ok  = False,
    ),

    # ---- WAR_OF_ATTRITION ----------------------------------------- #
    # Grind them down slowly. Preserve our army, trade favorably.
    # Pick fights only when the math is clearly in our favor.
    Strategy.WAR_OF_ATTRITION: TacticalProfile(
        engage_bias   = -0.10,
        retreat_bias  = +0.15,
        harass_bias   = +0.10,  # poke and run is fine, full commits are not
        cohesion_bias = +0.20,  # don't let units die in isolation
        hold_bias     = +0.15,  # hold ground, let them come to us
        sacrifice_ok  = False,
    ),

    # ---- BLEED_OUT ------------------------------------------------ #
    # Slowly drain their economy and units. Harass constantly.
    # Never fully commit — preserve army to keep the bleeding going.
    Strategy.BLEED_OUT: TacticalProfile(
        engage_bias   = -0.20,
        retreat_bias  = +0.25,
        harass_bias   = +0.40,  # this strategy lives and dies on harassment
        cohesion_bias = +0.10,
        hold_bias     = +0.10,
        sacrifice_ok  = False,
    ),

    # ---- DRONE_ONLY_FORTRESS -------------------------------------- #
    # Survive. Protect the economy. Don't attack. Army exists only
    # to defend mineral lines, not to go looking for fights.
    Strategy.DRONE_ONLY_FORTRESS: TacticalProfile(
        engage_bias   = -0.45,
        retreat_bias  = +0.40,
        harass_bias   = -0.45,  # absolutely do not harass — stay home
        cohesion_bias = +0.35,  # cluster defensively near bases
        hold_bias     = +0.40,  # hold chokes and stay put
        sacrifice_ok  = False,
    ),
}
