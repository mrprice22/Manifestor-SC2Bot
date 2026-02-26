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

Zerg unit roster across strategies
-----------------------------------
Ground:  ZERGLING, BANELING, ROACH, RAVAGER, HYDRALISK, LURKERMP,
         INFESTOR, SWARMHOSTMP, ULTRALISK, QUEEN
Air:     MUTALISK, CORRUPTOR, BROODLORD, VIPER

Tech phase landmarks (approximate game_phase values)
  0.00  early    – pool up, queens, ling speed underway
  0.25  mid-early– roach warren + first gas units
  0.30  mid      – lair complete; hydras, mutas, ravagers online
  0.50  mid-late – lurker den, infest pit, hive started
  0.70  late     – hive army: ultra, viper, greater spire unlocked
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

        Economy biases (additive to confidence in building tactics):
        drone_bias:     Bias toward droning.  +0.30 = fortress drones hard;
                        -0.50 = all-in stops droning entirely.
        expand_bias:    Bias toward expansion.  Lowers the saturation
                        threshold needed to take a new base and lifts
                        expansion confidence.  +0.25 = fortress expands
                        early; -0.30 = all-in never expands.
        gas_ratio_bias: Bias toward filling gas buildings.  +0.20 = muta/
                        hydra strategies saturate gas ASAP; -0.20 = ling-
                        flood/all-in routes most workers to minerals.
        bank_bias:      Bias toward saving minerals rather than spending them.
                        Reduces confidence of army production and upgrade
                        research so minerals accumulate for expansion.
                        Dynamically boosted by bot._ldm_pressure (overflow
                        drones signal we need a new base, not more army).
                        +0.10 = mild saving; +0.30 = strong eco hold;
                        -0.20 = all-in, spend every mineral immediately.
        scout_bias:     Bias toward sending overlords into scout slots
                        (vision-edge positions deep toward enemy territory).
                        +0.25 = harassment strategies need map intel;
                        -0.30 = fortress/all-in hides overlords safely.
                        Values below -0.05 disable scout slots entirely and
                        recall any scouts already deployed to rear-guard.
                        Default 0.0 = neutral (allow scouts up to cap).
    """
    engage_bias:    float = 0.0
    retreat_bias:   float = 0.0
    harass_bias:    float = 0.0
    cohesion_bias:  float = 0.0
    hold_bias:      float = 0.0
    sacrifice_ok:   bool  = False
    drone_bias:     float = 0.0
    expand_bias:    float = 0.0
    gas_ratio_bias: float = 0.0
    bank_bias:      float = 0.0
    scout_bias:     float = 0.0
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
#
# Phase band structure used throughout:
#   0.00  early      (pool/nat, queens, lings, ling speed)
#   0.25+ mid-early  (roach warren, ravagers, banelings come online)
#   0.28+ mid        (lair complete; mutas, hydras available)
#   0.30+ mid        (lair + first tier-2 ground tech)
#   0.48+ mid-late   (lurker den, infest pit, hive underway)
#   0.70+ late       (hive army: ultra, viper, greater spire → broodlords)
# ------------------------------------------------------------------ #

_PROFILES: dict[Strategy, TacticalProfile] = {

    # ─────────────────────────────────────────────────────────────────────────────
    # STOCK_STANDARD
    # Philosophy: textbook Zerg macro. Ling-bane-roach-ravager mid game for
    # flexible response; lurker/hydra transition once lair and den are up;
    # ultra-viper-infestor hive army to close. Solid and predictable execution.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.STOCK_STANDARD: TacticalProfile(
        engage_bias   = 0.16,
        retreat_bias  = 0.0,
        harass_bias   = 0.15,
        bank_bias     = +0.10,  # mild saving — keep minerals available for expansions
        cohesion_bias = 0.11,
        hold_bias     = 0.05,
        sacrifice_ok  = False,
        drone_bias    = +0.06,  # drone priority: hit supply quickly
        expand_bias   = +0.24,  # expand earlier (threshold ~70% vs default 75%)
        gas_ratio_bias= -0.10,  # early comp is ling/queen (0 gas); don't over-collect
        scout_bias    = -0.15,  # don't send overlords deep — too many die to early queens
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.35,
                    UnitID.ZERGLING: 0.50,
                    UnitID.BANELING: 0.15,   # early bio answer
                },
                army_supply_target=20,
                max_hatcheries=2,
            )),
            (0.3, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.10,
                    UnitID.ZERGLING: 0.20,
                    UnitID.BANELING: 0.15,
                    UnitID.ROACH:    0.40,
                    UnitID.RAVAGER:  0.15,   # corrosive bile on bio clumps/walls
                },
                army_supply_target=80,
                max_hatcheries=4,
            )),
            (0.5, CompositionTarget(
                ratios={
                    UnitID.ROACH:     0.25,
                    UnitID.HYDRALISK: 0.15,  # anti-air + harassment response
                    UnitID.LURKERMP:  0.40,  # lurker core
                    UnitID.ZERGLING:  0.20,  # surround assist
                },
                army_supply_target=120,
                max_hatcheries=5,
            )),
            (0.6, CompositionTarget(
                ratios={
                    UnitID.ULTRALISK: 0.30,  # tanky front-line
                    UnitID.LURKERMP:  0.25,
                    UnitID.VIPER:     0.15,  # parasitic bond, blinding cloud
                    UnitID.INFESTOR:  0.15,  # fungal + neural parasite
                    UnitID.ZERGLING:  0.15,  # surround fill
                },
                army_supply_target=140,
                max_hatcheries=6,
            )),
        ],
    ),
    
    # ─────────────────────────────────────────────────────────────────────────────
    # JUST_GO_PUNCH_EM
    # Philosophy: roach/ling flood as fast as possible, transition into ravagers
    # for corrosive bile and lurkers to crack static defence.
    # Banelings shred bio; ravagers handle walls and bunkers.
    # Late game is gravy: ultralisk/viper closes it out if we're still going.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.JUST_GO_PUNCH_EM: TacticalProfile(
        engage_bias   = +0.35,
        retreat_bias  = -0.40,
        harass_bias   = -0.10,
        cohesion_bias = +0.10,
        bank_bias     = -0.10,  # spend freely — army is the priority, not bases
        hold_bias     = -0.20,
        sacrifice_ok  = True,
        drone_bias    = -0.20,   # prioritise army over workers
        expand_bias   = -0.10,   # hold expansions; army is the priority
        gas_ratio_bias=  0.00,   # neutral — roach/ling needs some gas
        scout_bias    = +0.10,   # need to see targets; send scouts to find army position
        opening      = "EarlyAggression",
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.15,
                    UnitID.ZERGLING: 0.85,   # mass lings, go now
                },
                army_supply_target=24,
                max_hatcheries=2,
            )),
            (0.25, CompositionTarget(
                ratios={
                    UnitID.ZERGLING: 0.35,
                    UnitID.BANELING: 0.15,   # banelings shred bio armies
                    UnitID.ROACH:    0.35,   # roach backbone
                    UnitID.RAVAGER:  0.15,   # bile for walls / bunkers
                },
                army_supply_target=60,
                max_hatcheries=3,
            )),
            (0.45, CompositionTarget(
                ratios={
                    UnitID.ROACH:    0.35,
                    UnitID.RAVAGER:  0.20,   # corrosive bile on static defence
                    UnitID.LURKERMP: 0.30,   # siege support
                    UnitID.BANELING: 0.15,   # ling-bane flanks
                },
                army_supply_target=100,
                max_hatcheries=4,
            )),
            (0.70, CompositionTarget(
                ratios={
                    UnitID.ULTRALISK: 0.30,  # if we're still going, go big
                    UnitID.LURKERMP:  0.35,
                    UnitID.VIPER:     0.10,  # blinding cloud, abduct colossi/thors
                    UnitID.ROACH:     0.25,
                },
                army_supply_target=150,
                max_hatcheries=5,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # ALL_IN
    # Philosophy: everything committed right now. Hard ling-bane-roach all-in.
    # Banelings are the key bio-killer. Ravagers handle mineral lines and ramps.
    # No transition plan — resources go to units, not infrastructure.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.ALL_IN: TacticalProfile(
        engage_bias   = +0.45,
        retreat_bias  = -0.50,
        harass_bias   = -0.20,
        cohesion_bias = -0.10,
        bank_bias     = -0.20,  # never save — every mineral goes to units now
        hold_bias     = -0.30,
        sacrifice_ok  = True,
        drone_bias    = -0.50,   # stop droning — every larva is army
        expand_bias   = -0.30,   # never expand; commit everything now
        gas_ratio_bias= -0.20,   # mineral-heavy; ling-bane-roach all-in
        scout_bias    = -0.20,   # pull overlords back — intel irrelevant, just commit
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
                    UnitID.ZERGLING: 0.30,
                    UnitID.BANELING: 0.20,   # ling-bane all-in vs bio
                    UnitID.ROACH:    0.50,   # roach wall-punch
                },
                army_supply_target=80,
                max_hatcheries=3,
            )),
            (0.40, CompositionTarget(
                ratios={
                    UnitID.ROACH:    0.40,
                    UnitID.RAVAGER:  0.35,   # ravagers break fortified positions
                    UnitID.BANELING: 0.10,
                    UnitID.ZERGLING: 0.15,
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
    # Philosophy: constant multi-angle pressure. Mutas force defensive splits.
    # Hydras back them up for anti-air and mobile ground response.
    # Lurkers hold map positions while mutas roam. Late: corruptors into broodlords
    # so the siege never stops.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.KEEP_EM_BUSY: TacticalProfile(
        engage_bias   = +0.20,
        retreat_bias  = -0.15,
        harass_bias   = +0.35,
        cohesion_bias = -0.15,
        bank_bias     = +0.05,  # slight saving — more bases fuel continuous harassment
        hold_bias     = -0.10,
        sacrifice_ok  = False,
        drone_bias    =  0.00,   # keep pace with army; neither over-drones nor starves
        expand_bias   = +0.10,   # more bases fuel continuous harassment
        gas_ratio_bias= +0.20,   # mutas/hydras are gas-hungry; fill extractors fast
        scout_bias    = +0.25,   # harassment needs map vision; keep scouts on their expos
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
                    UnitID.ZERGLING:  0.25,
                    UnitID.ROACH:     0.15,
                    UnitID.HYDRALISK: 0.15,  # anti-air + ground support
                    UnitID.MUTALISK:  0.45,  # muta harassment pack
                },
                army_supply_target=55,
                max_hatcheries=4,
            )),
            (0.50, CompositionTarget(
                ratios={
                    UnitID.MUTALISK:  0.25,
                    UnitID.HYDRALISK: 0.20,  # hydra-lurker ground backbone
                    UnitID.LURKERMP:  0.25,  # siege + multi-angle pressure
                    UnitID.ROACH:     0.15,
                    UnitID.ZERGLING:  0.15,
                },
                army_supply_target=90,
                max_hatcheries=5,
            )),
            (0.72, CompositionTarget(
                ratios={
                    UnitID.VIPER:     0.10,  # abduct, blinding cloud
                    UnitID.CORRUPTOR: 0.20,  # anti-massive; morphs into broodlords
                    UnitID.BROODLORD: 0.20,  # permanent creep-push siege
                    UnitID.LURKERMP:  0.30,
                    UnitID.MUTALISK:  0.20,
                },
                army_supply_target=140,
                max_hatcheries=6,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # WAR_ON_SANITY
    # Philosophy: everything, everywhere, all at once. Mutas for drops and
    # harass, banelings for ling runbys, lurkers to hold map positions, infestors
    # for fungal chaos. Opponent can't be everywhere. We don't need to win any
    # single fight — we need them to lose all of them simultaneously.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.WAR_ON_SANITY: TacticalProfile(
        engage_bias   = +0.25,
        retreat_bias  = -0.20,
        harass_bias   = +0.25,
        cohesion_bias = -0.30,
        bank_bias     = +0.05,  # slight saving — multi-front chaos needs base count
        hold_bias     = -0.15,
        sacrifice_ok  = False,
        drone_bias    = +0.10,   # rich economy fuels multi-front chaos
        expand_bias   = +0.15,   # more bases = more production capacity for chaos
        gas_ratio_bias= +0.10,   # mutas + banes + lurkers all need gas
        scout_bias    = +0.15,   # need to know which fronts to hit simultaneously
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
                    UnitID.ZERGLING: 0.20,
                    UnitID.BANELING: 0.15,   # ling-bane runbys everywhere
                    UnitID.MUTALISK: 0.35,   # muta pack for chaos
                    UnitID.ROACH:    0.30,
                },
                army_supply_target=60,
                max_hatcheries=4,
            )),
            (0.48, CompositionTarget(
                ratios={
                    UnitID.MUTALISK:  0.25,
                    UnitID.LURKERMP:  0.25,  # siege multiple map positions
                    UnitID.INFESTOR:  0.15,  # fungal on clumped armies
                    UnitID.ZERGLING:  0.20,
                    UnitID.ROACH:     0.15,
                },
                army_supply_target=95,
                max_hatcheries=5,
            )),
            (0.70, CompositionTarget(
                ratios={
                    UnitID.BROODLORD: 0.15,  # creep-push siege
                    UnitID.CORRUPTOR: 0.10,  # anti-massive + broodlord supply
                    UnitID.VIPER:     0.15,  # pull, blinding cloud, caustic spray
                    UnitID.LURKERMP:  0.30,
                    UnitID.MUTALISK:  0.15,
                    UnitID.ZERGLING:  0.15,  # ling runbys still relevant
                },
                army_supply_target=150,
                max_hatcheries=6,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # WAR_OF_ATTRITION
    # Philosophy: trade favorably, hold ground, let them come to us.
    # Ravagers punish forward pushes with bile. Lurkers are the core — siege power
    # without committing. Infestors + swarm hosts turn our lines into a meat grinder.
    # Late: ultra/viper/lurker grinds down any sustained assault.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.WAR_OF_ATTRITION: TacticalProfile(
        engage_bias   = -0.10,
        retreat_bias  = +0.15,
        harass_bias   = +0.10,
        cohesion_bias = +0.20,
        bank_bias     = +0.12,  # steady saving — sustained attrition needs income
        hold_bias     = +0.15,
        sacrifice_ok  = False,
        drone_bias    = +0.10,   # steady droning; rich economy sustains the grind
        expand_bias   = +0.05,   # modest push; expand when safe, not eagerly
        gas_ratio_bias=  0.00,   # lurkers/infestors need gas but so do ravagers; neutral
        scout_bias    = +0.05,   # mild scouting; need to see pushes forming
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
                    UnitID.ROACH:    0.45,
                    UnitID.RAVAGER:  0.15,   # bile punishes enemy pushes
                    UnitID.QUEEN:    0.20,
                    UnitID.ZERGLING: 0.20,
                },
                army_supply_target=55,
                max_hatcheries=4,
            )),
            (0.50, CompositionTarget(
                ratios={
                    UnitID.LURKERMP:  0.45,  # lurker core — let them die on us
                    UnitID.INFESTOR:  0.15,  # fungal locks armies in lurker range
                    UnitID.ROACH:     0.25,
                    UnitID.ZERGLING:  0.15,
                },
                army_supply_target=100,
                max_hatcheries=5,
            )),
            (0.72, CompositionTarget(
                ratios={
                    UnitID.ULTRALISK:  0.25,  # unstoppable front-line
                    UnitID.LURKERMP:   0.25,
                    UnitID.VIPER:      0.20,  # parasitic bond, blinding cloud
                    UnitID.SWARMHOSTMP:0.15,  # free locust waves = attrition heaven
                    UnitID.INFESTOR:   0.10,  # fungal + neural on key targets
                    UnitID.ZERGLING:   0.05,
                },
                army_supply_target=160,
                max_hatcheries=6,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # BLEED_OUT
    # Philosophy: never commit to a decisive fight, always be harassing somewhere.
    # Mutas are the core — roam constantly, drain resources and attention.
    # Banelings supplement ling runbys into undefended expansions.
    # Lurkers hold a defensive line while mutas roam. Brood lords + vipers appear
    # late to siege without ever allowing a fair trade.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.BLEED_OUT: TacticalProfile(
        engage_bias   = -0.20,
        retreat_bias  = +0.25,
        harass_bias   = +0.40,
        cohesion_bias = +0.10,
        bank_bias     = +0.08,  # mild saving — harassment bases need funding
        hold_bias     = +0.10,
        sacrifice_ok  = False,
        drone_bias    = -0.05,   # slight drone de-emphasis; units keep the pressure on
        expand_bias   = +0.10,   # more bases fuel sustained harassment
        gas_ratio_bias= +0.20,   # mutas/broodlords/vipers are all gas-hungry
        scout_bias    = +0.10,   # mutas provide some vision; mild scout support
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
                    UnitID.MUTALISK:  0.45,  # muta roam starts immediately on lair
                    UnitID.ZERGLING:  0.25,
                    UnitID.BANELING:  0.15,  # runbys into undefended expansions
                    UnitID.ROACH:     0.15,  # minimal ground presence
                },
                army_supply_target=50,
                max_hatcheries=4,
            )),
            (0.50, CompositionTarget(
                ratios={
                    UnitID.MUTALISK:  0.30,
                    UnitID.LURKERMP:  0.25,  # defensive lurkers, mutas still roam
                    UnitID.INFESTOR:  0.15,  # fungal grounds fleeing units for mutas
                    UnitID.ZERGLING:  0.20,
                    UnitID.ROACH:     0.10,
                },
                army_supply_target=90,
                max_hatcheries=5,
            )),
            (0.72, CompositionTarget(
                ratios={
                    UnitID.BROODLORD: 0.25,  # permanent harassment siege
                    UnitID.CORRUPTOR: 0.15,  # anti-massive; morphs into broodlords
                    UnitID.VIPER:     0.20,  # abduct, blinding cloud, caustic spray
                    UnitID.LURKERMP:  0.25,
                    UnitID.MUTALISK:  0.15,
                },
                army_supply_target=150,
                max_hatcheries=6,
            )),
        ],
    ),


    # ─────────────────────────────────────────────────────────────────────────────
    # DRONE_ONLY_FORTRESS
    # Philosophy: survive, grow economy behind queens and static defence.
    # Hydras give early anti-air coverage. Lurkers replace roaches as primary
    # defensive unit mid-game. Infestors + swarm hosts turn the fortress into a
    # nightmare to assault. If we reach late game intact, ultra/viper/broodlord
    # finally strikes back.
    # ─────────────────────────────────────────────────────────────────────────────
    Strategy.DRONE_ONLY_FORTRESS: TacticalProfile(
        engage_bias   = -0.45,
        retreat_bias  = +0.40,
        harass_bias   = -0.45,
        cohesion_bias = +0.35,
        bank_bias     = +0.30,  # strong saving — economy IS the strategy; fund every hatch
        hold_bias     = +0.40,
        sacrifice_ok  = False,
        drone_bias    = +0.30,   # drone as hard as possible; economy IS the strategy
        expand_bias   = +0.25,   # expand early and often; more bases = more drones
        gas_ratio_bias= -0.25,   # cut gas workers — drones should mine minerals for spines
        scout_bias    = -0.30,   # pull all overlords home; we can't afford to lose any
        opening     = "TurtleEco",
        composition_curve = [
            (0.00, CompositionTarget(
                ratios={
                    UnitID.QUEEN:    0.70,   # queens are the entire defence
                    UnitID.ZERGLING: 0.30,
                },
                army_supply_target=14,
                max_hatcheries=3,   # expand aggressively — that's the whole point
            )),
            (0.30, CompositionTarget(
                ratios={
                    UnitID.QUEEN:     0.25,
                    UnitID.ROACH:     0.40,
                    UnitID.HYDRALISK: 0.15,  # anti-air; answers early air harassment
                    UnitID.ZERGLING:  0.20,
                },
                army_supply_target=40,
                max_hatcheries=5,
            )),
            (0.50, CompositionTarget(
                ratios={
                    UnitID.LURKERMP:   0.45,  # static lurker lines at each choke
                    UnitID.INFESTOR:   0.20,  # fungal locks pushes; neural key units
                    UnitID.SWARMHOSTMP:0.15,  # free locust damage without risking army
                    UnitID.ROACH:      0.15,
                    UnitID.QUEEN:      0.05,
                },
                army_supply_target=80,
                max_hatcheries=6,
            )),
            (0.72, CompositionTarget(
                ratios={
                    UnitID.ULTRALISK:  0.25,  # now we finally strike
                    UnitID.BROODLORD:  0.20,  # siege from safety
                    UnitID.CORRUPTOR:  0.10,  # anti-massive; morphs into broodlords
                    UnitID.VIPER:      0.20,  # abduct, blinding cloud, caustic spray
                    UnitID.LURKERMP:   0.15,
                    UnitID.INFESTOR:   0.10,
                },
                army_supply_target=170,
                max_hatcheries=7,
            )),
        ],
    ),
}
