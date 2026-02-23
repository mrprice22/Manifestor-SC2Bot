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
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

@dataclass
class CompositionTarget:
    """
    A desired army composition at a specific game phase.
    
    ratios: What fraction of total army supply each unit type should occupy.
           Doesn't need to sum to 1.0 — anything unnormalized works.
           "ZERGLING: 0.5, ROACH: 0.3" means "more zerglings than roaches"
    
    army_supply_target: How many supply of combat units we want at this phase.
                        Drives *scale*, not composition.
    
    max_hatcheries: Expansion cap at this phase.
    """
    ratios: dict[UnitID, float]
    army_supply_target: int
    max_hatcheries: int

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
    opening:       str   = "StandardOpener"  #maps to build name in zerg_builds.yml
    # Ordered list of (min_phase_threshold, CompositionTarget)
    # The last entry whose threshold <= current game_phase is active.
    composition_curve: list[tuple[float, CompositionTarget]] = field(
        default_factory=list
    )
    
    # Returns the active CompositionTarget for the given game phase.
    def active_composition(self, game_phase: float) -> Optional[CompositionTarget]:
        active = None
        for threshold, target in self.composition_curve:
            if game_phase >= threshold:
                active = target
        return active


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
# All curves share the same phase band structure:
#   0.00  early  (pool/nat, queens, lings)
#   0.30  mid    (lair complete, first tech units, lurkers possible)
#   0.55  late   (hive started, vipers, brood lords unlock)
#   0.75  supreme late (hive army, full upgrades, maxing)
# ------------------------------------------------------------------ #

_PROFILES: dict[Strategy, TacticalProfile] = {

    # ─────────────────────────────────────────────────────────────────────────────
    # JUST_GO_PUNCH_EM
    # Philosophy: roach/ling flood as fast as possible, transition into ravagers
    # and then lurkers to crack static defence. No mutas — no time for air.
    # Late game is irrelevant: we should have won or lost.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.JUST_GO_PUNCH_EM: TacticalProfile(
        engage_bias   = +0.35,
        retreat_bias  = -0.40,
        harass_bias   = -0.10,
        cohesion_bias = +0.10,
        hold_bias     = -0.20,
        sacrifice_ok  = True,
        opening      = "EarlyAggression",
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.20,
                    UnitID.ZERGLING: 0.80,   # mass lings, go now
                },
                army_supply_target=24,
                max_hatcheries=2,
            )),
            (0.25, CompositionTarget(
                ratios={
                    UnitID.ZERGLING: 0.40,
                    UnitID.ROACH:    0.60,   # roach backbone
                },
                army_supply_target=60,
                max_hatcheries=3,
            )),
            (0.45, CompositionTarget(
                ratios={
                    UnitID.ROACH:    0.50,
                    UnitID.RAVAGER:  0.20,   # bile for walls / corrosive
                    UnitID.LURKERMP: 0.30,   # siege support
                },
                army_supply_target=100,
                max_hatcheries=4,
            )),
            (0.70, CompositionTarget(
                ratios={
                    UnitID.ULTRALISK: 0.30,  # if we're still going, go big
                    UnitID.LURKERMP:  0.35,
                    UnitID.ROACH:     0.35,
                },
                army_supply_target=150,
                max_hatcheries=5,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # ALL_IN
    # Philosophy: everything committed right now. Hard ling-roach all-in.
    # No transition plan. max_hatcheries stays low — resources go to units.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.ALL_IN: TacticalProfile(
        engage_bias   = +0.45,
        retreat_bias  = -0.50,
        harass_bias   = -0.20,
        cohesion_bias = -0.10,
        hold_bias     = -0.30,
        sacrifice_ok  = True,
        opening      = "EarlyAggression",
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.10,
                    UnitID.ZERGLING: 0.90,   # pure ling flood
                },
                army_supply_target=30,
                max_hatcheries=2,
            )),
            (0.20, CompositionTarget(
                ratios={
                    UnitID.ZERGLING: 0.35,
                    UnitID.ROACH:    0.65,   # roach-ling all-in
                },
                army_supply_target=80,
                max_hatcheries=3,
            )),
            (0.40, CompositionTarget(
                ratios={
                    UnitID.ROACH:    0.50,
                    UnitID.RAVAGER:  0.30,
                    UnitID.ZERGLING: 0.20,
                },
                army_supply_target=120,
                max_hatcheries=3,   # still low — go, don't drone
            )),
            # No late band: if game_phase hits 0.70, the all-in failed
            # and we're just surviving on existing curve.
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # KEEP_EM_BUSY
    # Philosophy: constant multi-angle pressure. Mutas are ideal here —
    # mobile, harass-capable, force defensive splits from opponent.
    # Lurkers back the mid-game to hold map without full commitment.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.KEEP_EM_BUSY: TacticalProfile(
        engage_bias   = +0.20,
        retreat_bias  = -0.15,
        harass_bias   = +0.35,
        cohesion_bias = -0.15,
        hold_bias     = -0.10,
        sacrifice_ok  = False,
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.30,
                    UnitID.ZERGLING: 0.70,
                },
                army_supply_target=20,
                max_hatcheries=2,
            )),
            (0.28, CompositionTarget(
                ratios={
                    UnitID.ZERGLING:  0.30,
                    UnitID.ROACH:     0.20,
                    UnitID.MUTALISK:  0.50,  # muta harassment pack
                },
                army_supply_target=55,
                max_hatcheries=4,
            )),
            (0.50, CompositionTarget(
                ratios={
                    UnitID.MUTALISK:  0.30,
                    UnitID.LURKERMP:  0.30,  # ground harassment + siege
                    UnitID.ROACH:     0.25,
                    UnitID.ZERGLING:  0.15,
                },
                army_supply_target=90,
                max_hatcheries=5,
            )),
            (0.72, CompositionTarget(
                ratios={
                    UnitID.VIPER:        0.15,  # pull, blinding cloud
                    UnitID.BROODLORD:    0.25,
                    UnitID.LURKERMP:     0.35,
                    UnitID.MUTALISK:     0.25,
                },
                army_supply_target=140,
                max_hatcheries=6,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # WAR_ON_SANITY
    # Philosophy: everything, everywhere, all at once. Mutas for drops and
    # harass, lings for runbys, lurkers to hold map positions simultaneously.
    # Opponent can't be everywhere. We don't need to win any single fight.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.WAR_ON_SANITY: TacticalProfile(
        engage_bias   = +0.25,
        retreat_bias  = -0.20,
        harass_bias   = +0.25,
        cohesion_bias = -0.30,
        hold_bias     = -0.15,
        sacrifice_ok  = False,
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.25,
                    UnitID.ZERGLING: 0.75,
                },
                army_supply_target=22,
                max_hatcheries=2,
            )),
            (0.28, CompositionTarget(
                ratios={
                    UnitID.ZERGLING: 0.35,
                    UnitID.MUTALISK: 0.40,   # muta pack for chaos
                    UnitID.ROACH:    0.25,
                },
                army_supply_target=60,
                max_hatcheries=4,
            )),
            (0.48, CompositionTarget(
                ratios={
                    UnitID.MUTALISK:  0.30,
                    UnitID.LURKERMP:  0.25,
                    UnitID.ZERGLING:  0.25,
                    UnitID.ROACH:     0.20,
                },
                army_supply_target=95,
                max_hatcheries=5,
            )),
            (0.70, CompositionTarget(
                ratios={
                    UnitID.BROODLORD: 0.20,
                    UnitID.VIPER:     0.15,
                    UnitID.LURKERMP:  0.30,
                    UnitID.MUTALISK:  0.20,
                    UnitID.ZERGLING:  0.15,  # ling runbys still relevant
                },
                army_supply_target=150,
                max_hatcheries=6,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # STOCK_STANDARD
    # Philosophy: textbook Zerg macro. Roach-ling-bane mid, lurker transition,
    # hive tech late. No mutas — solid predictable execution.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.STOCK_STANDARD: TacticalProfile(
        engage_bias   = 0.2,
        retreat_bias  = 0.1,
        harass_bias   = 0.2,
        cohesion_bias = 0.0,
        hold_bias     = 0.0,
        sacrifice_ok  = False,
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.40,
                    UnitID.ZERGLING: 0.60,
                },
                army_supply_target=20,
                max_hatcheries=2,
            )),
            (0.30, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.15,
                    UnitID.ZERGLING: 0.30,
                    UnitID.ROACH:    0.55,
                },
                army_supply_target=60,
                max_hatcheries=4,
            )),
            (0.50, CompositionTarget(
                ratios={
                    UnitID.ROACH:    0.40,
                    UnitID.LURKERMP: 0.40,
                    UnitID.ZERGLING: 0.20,
                },
                army_supply_target=100,
                max_hatcheries=5,
            )),
            (0.72, CompositionTarget(
                ratios={
                    UnitID.ULTRALISK: 0.30,
                    UnitID.LURKERMP:  0.35,
                    UnitID.VIPER:     0.15,
                    UnitID.ZERGLING:  0.20,  # surround assist
                },
                army_supply_target=160,
                max_hatcheries=6,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # WAR_OF_ATTRITION
    # Philosophy: trade favorably, hold ground, let them come.
    # Lurkers are perfect here — siege power without committing to attack.
    # Late game: ultra/viper is the grind comp of choice.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.WAR_OF_ATTRITION: TacticalProfile(
        engage_bias   = -0.10,
        retreat_bias  = +0.15,
        harass_bias   = +0.10,
        cohesion_bias = +0.20,
        hold_bias     = +0.15,
        sacrifice_ok  = False,
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.50,   # queens hold early with strong AA
                    UnitID.ZERGLING: 0.50,
                },
                army_supply_target=18,
                max_hatcheries=2,
            )),
            (0.30, CompositionTarget(
                ratios={
                    UnitID.ROACH:    0.60,
                    UnitID.QUEEN:    0.20,
                    UnitID.ZERGLING: 0.20,
                },
                army_supply_target=55,
                max_hatcheries=4,
            )),
            (0.50, CompositionTarget(
                ratios={
                    UnitID.LURKERMP: 0.55,   # lurker core — let them die on us
                    UnitID.ROACH:    0.30,
                    UnitID.ZERGLING: 0.15,
                },
                army_supply_target=100,
                max_hatcheries=5,
            )),
            (0.72, CompositionTarget(
                ratios={
                    UnitID.ULTRALISK: 0.35,
                    UnitID.LURKERMP:  0.35,
                    UnitID.VIPER:     0.20,  # parasitic bond, blinding cloud
                    UnitID.ZERGLING:  0.10,
                },
                army_supply_target=160,
                max_hatcheries=6,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # BLEED_OUT
    # Philosophy: never commit, always be harassing somewhere.
    # Mutas are core to this strategy. Lurkers hold a defensive line while
    # muta packs roam. Brood lords appear late to maintain the pressure siege.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.BLEED_OUT: TacticalProfile(
        engage_bias   = -0.20,
        retreat_bias  = +0.25,
        harass_bias   = +0.40,
        cohesion_bias = +0.10,
        hold_bias     = +0.10,
        sacrifice_ok  = False,
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.40,
                    UnitID.ZERGLING: 0.60,
                },
                army_supply_target=18,
                max_hatcheries=2,
            )),
            (0.28, CompositionTarget(
                ratios={
                    UnitID.MUTALISK:  0.55,  # muta roam starts immediately on lair
                    UnitID.ZERGLING:  0.30,
                    UnitID.ROACH:     0.15,  # minimal ground presence
                },
                army_supply_target=50,
                max_hatcheries=4,
            )),
            (0.50, CompositionTarget(
                ratios={
                    UnitID.MUTALISK:  0.35,
                    UnitID.LURKERMP:  0.35,  # defensive lurkers, muta still roam
                    UnitID.ZERGLING:  0.20,
                    UnitID.ROACH:     0.10,
                },
                army_supply_target=90,
                max_hatcheries=5,
            )),
            (0.72, CompositionTarget(
                ratios={
                    UnitID.BROODLORD: 0.30,  # broodlords = permanent harassment siege
                    UnitID.VIPER:     0.20,
                    UnitID.LURKERMP:  0.30,
                    UnitID.MUTALISK:  0.20,
                },
                army_supply_target=150,
                max_hatcheries=6,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # DRONE_ONLY_FORTRESS
    # Philosophy: survive. Grow economy behind queens and static defence.
    # Lurkers replace roaches as the primary defensive unit mid-game.
    # If we reach late game intact, ultralisk/viper/broodlord should end it.
    # No mutas — they require the economy risk of a Spire with no ground response.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.DRONE_ONLY_FORTRESS: TacticalProfile(
        engage_bias   = -0.45,
        retreat_bias  = +0.40,
        harass_bias   = -0.45,
        cohesion_bias = +0.35,
        hold_bias     = +0.40,
        sacrifice_ok  = False,
        opening     = "TurtleEco",
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.70,   # queens are the defence
                    UnitID.ZERGLING: 0.30,
                },
                army_supply_target=14,
                max_hatcheries=3,   # expand aggressively — that's the whole point
            )),
            (0.30, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.30,
                    UnitID.ROACH:    0.50,
                    UnitID.ZERGLING: 0.20,
                },
                army_supply_target=40,
                max_hatcheries=5,
            )),
            (0.50, CompositionTarget(
                ratios={
                    UnitID.LURKERMP: 0.60,   # static lurker lines
                    UnitID.ROACH:    0.25,
                    UnitID.QUEEN:    0.15,
                },
                army_supply_target=80,
                max_hatcheries=6,
            )),
            (0.72, CompositionTarget(
                ratios={
                    UnitID.ULTRALISK:  0.35,
                    UnitID.BROODLORD:  0.30,  # now we finally strike
                    UnitID.VIPER:      0.20,
                    UnitID.LURKERMP:   0.15,
                },
                army_supply_target=170,
                max_hatcheries=7,
            )),
        ],
    ),
}
