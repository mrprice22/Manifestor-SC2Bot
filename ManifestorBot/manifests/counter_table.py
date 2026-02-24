# ManifestorBot/manifests/counter_table.py

from dataclasses import dataclass
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId


@dataclass(frozen=True)
class CounterPrescription:
    """What the bot should lean toward when it sees a specific enemy unit."""
    # Units to prioritise training (in order of preference)
    train_priority: tuple[UnitID, ...] = ()
    # Upgrades that become more valuable against this threat
    research_priority: tuple[UpgradeId, ...] = ()
    # Additive confidence bonus applied to building idea modules that match
    production_confidence_bonus: float = 0.0
    research_confidence_bonus: float = 0.0
    # Additive bonus applied to unit tactic confidence (e.g. engage vs retreat)
    engage_bias_mod: float = 0.0
    retreat_bias_mod: float = 0.0


COUNTER_TABLE: dict[UnitID, CounterPrescription] = {

    # =========================
    # ====== PROTOSS ==========
    # =========================

    # Void Rays → sustained anti-air
    UnitID.VOIDRAY: CounterPrescription(
        train_priority=(UnitID.HYDRALISK, UnitID.QUEEN),
        research_priority=(
            UpgradeId.ZERGMISSILEWEAPONSLEVEL1,
            UpgradeId.EVOLVEGROOVEDSPINES,
        ),
        production_confidence_bonus=0.30,
        research_confidence_bonus=0.20,
    ),

    # Carriers → Hydras early, Corruptors late
    UnitID.CARRIER: CounterPrescription(
        train_priority=(UnitID.CORRUPTOR, UnitID.HYDRALISK),
        research_priority=(
            UpgradeId.ZERGMISSILEWEAPONSLEVEL2,
            UpgradeId.ZERGFLYERWEAPONSLEVEL1,
        ),
        production_confidence_bonus=0.35,
        research_confidence_bonus=0.25,
        retreat_bias_mod=0.10,  # don’t a-move into interceptor cloud blindly
    ),

    # Tempests → Corruptor focus, possibly Viper later
    UnitID.TEMPEST: CounterPrescription(
        train_priority=(UnitID.CORRUPTOR,),
        research_priority=(UpgradeId.ZERGFLYERWEAPONSLEVEL1,),
        production_confidence_bonus=0.30,
    ),

    # Stalkers → Roach core
    UnitID.STALKER: CounterPrescription(
        train_priority=(UnitID.ROACH,),
        research_priority=(UpgradeId.GLIALRECONSTITUTION,),
        production_confidence_bonus=0.20,
    ),

    # Zealots → Roach wall or Baneling
    UnitID.ZEALOT: CounterPrescription(
        train_priority=(UnitID.ROACH, UnitID.BANELING),
        production_confidence_bonus=0.15,
    ),

    # Immortals → Swarm or Ling surround
    UnitID.IMMORTAL: CounterPrescription(
        train_priority=(UnitID.ZERGLING, UnitID.ULTRALISK),
        research_priority=(UpgradeId.ZERGLINGMOVEMENTSPEED,),
        production_confidence_bonus=0.25,
        engage_bias_mod=0.15,
    ),

    # Colossus → Corruptor or Viper transition
    UnitID.COLOSSUS: CounterPrescription(
        train_priority=(UnitID.CORRUPTOR, UnitID.HYDRALISK),
        research_priority=(UpgradeId.ZERGFLYERWEAPONSLEVEL1,),
        production_confidence_bonus=0.30,
    ),

    # High Templar → split-capable units, Viper later
    UnitID.HIGHTEMPLAR: CounterPrescription(
        train_priority=(UnitID.ROACH, UnitID.ULTRALISK),
        production_confidence_bonus=0.15,
        retreat_bias_mod=0.10,
    ),

    # Dark Templar → detection + roach safety
    UnitID.DARKTEMPLAR: CounterPrescription(
        train_priority=(UnitID.ROACH,),
        production_confidence_bonus=0.15,
    ),


    # =========================
    # ====== TERRAN ===========
    # =========================

    # Marines → Banelings
    UnitID.MARINE: CounterPrescription(
        train_priority=(UnitID.BANELING,),
        research_priority=(
            UpgradeId.ZERGLINGMOVEMENTSPEED,
            UpgradeId.CENTRIFICALHOOKS,
        ),
        production_confidence_bonus=0.25,
        research_confidence_bonus=0.25,
    ),

    # Marauders → Roach heavy
    UnitID.MARAUDER: CounterPrescription(
        train_priority=(UnitID.ROACH,),
        research_priority=(UpgradeId.GLIALRECONSTITUTION,),
        production_confidence_bonus=0.20,
    ),

    # Siege Tanks → flank hard
    UnitID.SIEGETANKSIEGED: CounterPrescription(
        train_priority=(UnitID.ZERGLING, UnitID.ULTRALISK),
        research_priority=(UpgradeId.ZERGLINGMOVEMENTSPEED,),
        production_confidence_bonus=0.25,
        engage_bias_mod=0.20,
    ),

    # Thors → Swarm or Ling flood
    UnitID.THOR: CounterPrescription(
        train_priority=(UnitID.ZERGLING, UnitID.ULTRALISK),
        research_priority=(UpgradeId.ZERGLINGMOVEMENTSPEED,),
        production_confidence_bonus=0.25,
        engage_bias_mod=0.10,
    ),

    # Hellions → Roach safety
    UnitID.HELLION: CounterPrescription(
        train_priority=(UnitID.ROACH,),
        production_confidence_bonus=0.25,
    ),

    # Battlecruiser → Corruptor + Queen
    UnitID.BATTLECRUISER: CounterPrescription(
        train_priority=(UnitID.CORRUPTOR, UnitID.QUEEN),
        research_priority=(UpgradeId.ZERGFLYERWEAPONSLEVEL1,),
        production_confidence_bonus=0.35,
        research_confidence_bonus=0.20,
    ),

    # Liberator → Hydras
    UnitID.LIBERATOR: CounterPrescription(
        train_priority=(UnitID.HYDRALISK,),
        research_priority=(UpgradeId.EVOLVEGROOVEDSPINES,),
        production_confidence_bonus=0.20,
    ),


    # =========================
    # ======== ZERG ===========
    # =========================

    # Roach → Ravager artillery
    UnitID.ROACH: CounterPrescription(
        train_priority=(UnitID.RAVAGER, UnitID.HYDRALISK),
        production_confidence_bonus=0.20,
    ),

    # Mutalisk → Hydras + Queens
    UnitID.MUTALISK: CounterPrescription(
        train_priority=(UnitID.HYDRALISK, UnitID.QUEEN),
        research_priority=(UpgradeId.EVOLVEGROOVEDSPINES,),
        production_confidence_bonus=0.30,
        research_confidence_bonus=0.15,
    ),

    # Ultralisks → Broodlord tech path
    UnitID.ULTRALISK: CounterPrescription(
        train_priority=(UnitID.BROODLORD,),
        research_priority=(UpgradeId.ZERGFLYERWEAPONSLEVEL2,),
        production_confidence_bonus=0.30,
    ),

    # Broodlords → Corruptor response
    UnitID.BROODLORD: CounterPrescription(
        train_priority=(UnitID.CORRUPTOR,),
        research_priority=(UpgradeId.ZERGFLYERWEAPONSLEVEL1,),
        production_confidence_bonus=0.30,
    ),

    # Banelings → Roach stability
    UnitID.BANELING: CounterPrescription(
        train_priority=(UnitID.ROACH,),
        production_confidence_bonus=0.20,
    ),
}