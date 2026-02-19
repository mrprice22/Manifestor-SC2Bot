# ManifestorBot/manifests/counter_table.py

from dataclasses import dataclass, field
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
    # Against heavy air (Void Rays, Carriers): make Hydras + missile upgrades
    UnitID.VOIDRAY: CounterPrescription(
        train_priority=(UnitID.HYDRALISK, UnitID.QUEEN),
        research_priority=(UpgradeId.ZERGMISSILEWEAPONSLEVEL1,),
        production_confidence_bonus=0.25,
        research_confidence_bonus=0.15,
    ),
    # Against tanks: Zerglings to flank, Ultralisks to tank
    UnitID.SIEGETANKSIEGED: CounterPrescription(
        train_priority=(UnitID.ZERGLING, UnitID.ULTRALISK),
        research_priority=(UpgradeId.ZERGLINGMOVEMENTSPEED,),
        production_confidence_bonus=0.20,
        engage_bias_mod=0.15,   # commit to the flank — don't sit in siege range
    ),
    # Against Marines/bio-ball: Banelings + speed
    UnitID.MARINE: CounterPrescription(
        train_priority=(UnitID.BANELING,),
        research_priority=(UpgradeId.ZERGLINGMOVEMENTSPEED, UpgradeId.CENTRIFICALHOOKS),
        production_confidence_bonus=0.20,
        research_confidence_bonus=0.20,
    ),
    # Against Roaches: Ravagers + Hydras
    UnitID.ROACH: CounterPrescription(
        train_priority=(UnitID.RAVAGER, UnitID.HYDRALISK),
        production_confidence_bonus=0.15,
    ),
    # Against Stalkers: Roaches (armored, tanky)
    UnitID.STALKER: CounterPrescription(
        train_priority=(UnitID.ROACH,),
        research_priority=(UpgradeId.GLIALRECONSTITUTION,),
        production_confidence_bonus=0.15,
    ),
}