# ManifestorBot/manifests/scout_ledger.py
from dataclasses import dataclass, field
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId
from ManifestorBot.manifests.counter_table import COUNTER_TABLE, CounterPrescription

@dataclass
class EnemyObservation:
    unit_type: UnitID
    first_seen_frame: int
    last_seen_frame: int
    count_peak: int          # highest simultaneous count ever seen
    count_current: int       # alive units we can see right now

@dataclass
class CounterContext:
    """Aggregated counter-play signals for this frame."""
    # Flat set of units the counter table says to prioritise training
    priority_train_types: set[UnitID] = field(default_factory=set)
    # Flat set of upgrades the counter table says to prioritise
    priority_upgrades: set[UpgradeId] = field(default_factory=set)
    # Summed confidence bonuses (multiple threats stack)
    production_bonus: float = 0.0
    research_bonus: float = 0.0
    engage_bias_mod: float = 0.0
    retreat_bias_mod: float = 0.0

class ScoutLedger:
    def __init__(self, bot):
        self.bot = bot
        self._observations: dict[UnitID, EnemyObservation] = {}
        self._seen_structures: set[UnitID] = set()

    def update(self, iteration: int) -> None:
        """Call once per on_step, before heuristics."""
        current_frame = self.bot.state.game_loop

        # Reset current counts
        current_counts: dict[UnitID, int] = {}
        for unit in self.bot.enemy_units:
            current_counts[unit.type_id] = current_counts.get(unit.type_id, 0) + 1

        for unit_type, count in current_counts.items():
            if unit_type not in self._observations:
                self._observations[unit_type] = EnemyObservation(
                    unit_type=unit_type,
                    first_seen_frame=current_frame,
                    last_seen_frame=current_frame,
                    count_peak=count,
                    count_current=count,
                )
            else:
                obs = self._observations[unit_type]
                obs.last_seen_frame = current_frame
                obs.count_peak = max(obs.count_peak, count)
                obs.count_current = count

        # Zero out units we can no longer see
        for unit_type, obs in self._observations.items():
            if unit_type not in current_counts:
                obs.count_current = 0

        # Track structures separately (they tell you about tech paths)
        for struct in self.bot.enemy_structures:
            self._seen_structures.add(struct.type_id)

    # --- Query API ---

    def has_seen(self, unit_type: UnitID) -> bool:
        return unit_type in self._observations

    def has_seen_structure(self, struct_type: UnitID) -> bool:
        return struct_type in self._seen_structures

    def peak_count(self, unit_type: UnitID) -> int:
        obs = self._observations.get(unit_type)
        return obs.count_peak if obs else 0

    def frames_since_seen(self, unit_type: UnitID, current_frame: int) -> int:
        obs = self._observations.get(unit_type)
        if obs is None:
            return 999999
        return current_frame - obs.last_seen_frame

    def is_recently_active(self, unit_type: UnitID, current_frame: int, window: int = 448) -> bool:
        """True if we saw this unit type within the last ~20 seconds."""
        return self.frames_since_seen(unit_type, current_frame) < window

    def get_counter_context(self, current_frame: int) -> CounterContext:
        """
        Aggregate all active counter prescriptions into one context object.
        A prescription is 'active' if we've seen that unit recently.
        """
        ctx = CounterContext()
        for unit_type, prescription in COUNTER_TABLE.items():
            if not self.is_recently_active(unit_type, current_frame):
                continue
            ctx.priority_train_types.update(prescription.train_priority)
            ctx.priority_upgrades.update(prescription.research_priority)
            ctx.production_bonus  += prescription.production_confidence_bonus
            ctx.research_bonus    += prescription.research_confidence_bonus
            ctx.engage_bias_mod   += prescription.engage_bias_mod
            ctx.retreat_bias_mod  += prescription.retreat_bias_mod
        # Cap bonuses so multiple threats don't compound to absurd values
        ctx.production_bonus = min(ctx.production_bonus, 0.4)
        ctx.research_bonus   = min(ctx.research_bonus, 0.4)
        return ctx



    