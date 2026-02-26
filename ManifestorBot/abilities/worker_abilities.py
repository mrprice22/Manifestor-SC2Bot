"""
Worker abilities — Drone (and future SCV / Probe) ability implementations.

MineAbility
-----------
Refactors mining from implicit Ares macro behavior into an explicit, inspectable
ability. The ability:
  1. Finds the worker's assigned mineral field / gas via Ares mediator.
  2. Issues the gather command if the worker is idle or returning cargo.
  3. Returns False (so the selector falls through) if the worker already has
     a valid gather order — Ares' own mining loop handles the fine-grained
     saturation logic and we don't want to fight it.

Registration
------------
Call register_worker_abilities() once from on_start(). This populates the
ability_registry with the drone's ability list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.unit import Unit

from ManifestorBot.abilities.ability import Ability, AbilityContext
from ManifestorBot.abilities.ability_registry import ability_registry

if TYPE_CHECKING:
    from ManifestorBot.manifestor_bot import ManifestorBot


# ---------------------------------------------------------------------------
# MineAbility
# ---------------------------------------------------------------------------

class MineAbility(Ability):
    """
    Send a drone to mine minerals or gas.

    Triggers when context.goal == "mine".
    Priority 100 — mining is the default drone behavior and should win unless
    the tactic layer has explicitly set a higher-confidence override.

    Thundering-herd guard
    ---------------------
    ``surplus_harvesters`` / ``assigned_harvesters`` are game-engine values
    that do NOT update within a single game loop tick.  When N drones at an
    oversaturated base all evaluate in the same frame they all see the same
    headroom at the destination, all get dispatched there, and overshoot —
    causing the bases to swap roles and the whole cycle to reverse next frame.

    Fix: ``_pending_inbound`` tracks how many drones have already been
    committed to each base or geyser tag this frame.  Effective headroom is
    reduced by the pending count so later drones in the same frame see a
    virtually-full destination and fall through to their next best idea.
    The dict is reset at the start of each new game loop tick.
    """

    UNIT_TYPES: Set[UnitID] = {UnitID.DRONE}
    GOAL: str = "mine"
    priority: int = 100

    # Thundering-herd guard — shared across all MineAbility instances.
    _pending_inbound: dict = {}   # resource_tag (int) -> pending drone count
    _pending_frame: int = -1      # game_loop tick when dict was last reset

    # ------------------------------------------------------------------
    # Thundering-herd helpers
    # ------------------------------------------------------------------

    @classmethod
    def _refresh_pending(cls, bot: "ManifestorBot") -> None:
        """Reset pending counters when a new game frame begins."""
        frame = bot.state.game_loop
        if frame != cls._pending_frame:
            cls._pending_inbound = {}
            cls._pending_frame = frame
            # Reset the long-distance-mining pressure counter so
            # _maybe_expand always sees a fresh count for this frame.
            bot._ldm_pressure = 0

    @classmethod
    def _reserve(cls, tag: int) -> None:
        """Mark one drone as inbound to the resource/base with *tag*."""
        cls._pending_inbound[tag] = cls._pending_inbound.get(tag, 0) + 1

    @classmethod
    def _pending_for(cls, tag: int) -> int:
        return cls._pending_inbound.get(tag, 0)

    def can_use(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        """
        A drone can mine if:
        - It is idle, OR its current gather target is at an over-saturated
          location and a better destination exists.
        - There is at least one mineral field or gas building accessible.

        Ghost-oscillation fix
        ---------------------
        The previous implementation checked the nearest townhall to the
        drone's *physical position* to detect over-saturation.  Mid-transit
        drones (just dispatched to a new base but not yet physically there)
        were still closest to the old base, so they were re-redirected every
        frame, bouncing the bases back and forth.

        The fix: check the saturation of the base nearest to the drone's
        *gather target resource*, not the drone's current position.  Once a
        drone has been given gather(mineral_B), its target is in base B's
        mineral line.  If B is not over-saturated, can_use returns False and
        the drone is left alone to complete its transit.
        """
        if unit.orders:
            order = unit.orders[0]
            if order.ability.id in _GATHER_ABILITIES:

                # HARVEST_RETURN: drone is returning cargo to base.  Never
                # interrupt — it auto-resumes harvest after delivery and
                # we'd just be adding a redundant gather command.
                if order.ability.id in _RETURN_ABILITIES:
                    return False

                # HARVEST_GATHER: decide using the *target resource*, not the
                # drone's current position, to avoid ghost-oscillation.
                target_tag = order.target
                if isinstance(target_tag, int) and bot.townhalls.ready:
                    # Gas target: only redirect if the geyser itself is over-assigned.
                    gas = bot.gas_buildings.find_by_tag(target_tag)
                    if gas is not None:
                        return gas.assigned_harvesters > gas.ideal_harvesters

                    # Mineral target: redirect only if the base closest to
                    # *that mineral* is effectively over-saturated, and there's
                    # a better destination available (local or long-distance).
                    mineral = bot.mineral_field.find_by_tag(target_tag)
                    if mineral is not None:
                        self._refresh_pending(bot)
                        target_th = bot.townhalls.ready.closest_to(mineral.position)
                        eff_surplus = (target_th.surplus_harvesters
                                       + self._pending_for(target_th.tag))
                        if eff_surplus <= 0:
                            return False  # target base is fine — leave drone alone

                        # Target base is over-saturated.  A valid destination
                        # is either a local under-saturated base OR a free
                        # expansion location for long-distance mining.
                        has_local = any(
                            (th.surplus_harvesters + self._pending_for(th.tag)) < 0
                            and bool(bot.mineral_field.closer_than(10, th.position))
                            for th in bot.townhalls.ready
                        )
                        if has_local:
                            return True
                        # Long-distance-mining check.
                        # Only consider LDM when genuinely over-saturated:
                        # total assigned workers must exceed total ideal by at
                        # least _LDM_SURPLUS_THRESHOLD.  This prevents premature
                        # LDM that would drain drones from a still-filling base.
                        total_surplus = sum(
                            th.surplus_harvesters for th in bot.townhalls.ready
                        )
                        if total_surplus < _LDM_SURPLUS_THRESHOLD:
                            return False
                        # Cap: don't dispatch more LDM drones than the actual
                        # surplus.  _ldm_pressure counts drones already committed
                        # to LDM this frame (incremented in _find_gather_target).
                        if getattr(bot, "_ldm_pressure", 0) >= total_surplus:
                            return False
                        # Use ares's precomputed ground-pathing-sorted expansion
                        # list so we prefer the closest reachable site, not the
                        # closest as-the-crow-flies.  get_own_expansions is
                        # List[Tuple[Point2, float]] sorted by path distance
                        # from our main base, computed once at game start.
                        our_pos = {th.position for th in bot.townhalls}
                        for exp, _dist in bot.mediator.get_own_expansions:
                            if exp in our_pos:
                                continue
                            nearby = bot.mineral_field.closer_than(10, exp)
                            if not nearby:
                                continue
                            if self._pending_for(exp) < len(nearby) * 2:
                                return True  # LDM slot available
                        return False  # no destinations at all

                # Fallback: target tag not in known resources (mid-build,
                # returning, etc.) — use position-based check as before.
                if bot.townhalls.ready:
                    closest_th = bot.townhalls.ready.closest_to(unit.position)
                    if not bot.mineral_field.closer_than(10, closest_th.position):
                        return True  # mined-out base — reassign
                    if closest_th.surplus_harvesters > 0:
                        self._refresh_pending(bot)
                        return any(
                            (th.surplus_harvesters + self._pending_for(th.tag)) < 0
                            and bool(bot.mineral_field.closer_than(10, th.position))
                            for th in bot.townhalls.ready
                        )
                return False  # gather order is fine — let Ares handle it

        # Drone is idle — needs somewhere to mine.
        return bool(bot.mineral_field or bot.gas_buildings.ready)

    def execute(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        """
        Issue a gather command to the nearest accessible resource.

        Prefers the mineral line at the nearest townhall to keep drones
        local. Falls back to the global mineral list if no townhall found.

        Cargo-return on LDM
        -------------------
        If the selected target is a long-distance mining site (no hatchery
        within 12 tiles) and the drone is carrying resources, issue a
        HARVEST_RETURN first and queue the LDM gather behind it.  Without
        this the drone would walk all the way to the LDM site, then walk
        all the way back to drop off, then walk back again — three trips
        instead of one.
        """
        target = self._find_gather_target(unit, bot)
        if target is None:
            return False

        # Detect LDM: target mineral is not near any of our bases.
        is_ldm = bot.townhalls.ready and not any(
            target.position.distance_to(th.position) < 12
            for th in bot.townhalls.ready
        )
        if is_ldm and (unit.is_carrying_minerals or unit.is_carrying_vespene):
            unit(AbilityId.HARVEST_RETURN_DRONE)
            unit.gather(target, queue=True)
            context.ability_used = self.name
            context.command_issued = True
            return True

        unit.gather(target)
        context.ability_used = self.name
        context.command_issued = True
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_gather_target(self, unit: Unit, bot: "ManifestorBot"):
        """
        Find the best resource for this drone to gather.

        Priority order:
        1. Gas buildings that are ready and under-saturated (≤ 2 workers),
           accounting for drones already dispatched there this frame.
        2. The nearest *under-saturated* townhall's mineral line, accounting
           for drones already dispatched there this frame (thundering-herd
           guard — see class docstring).
        3. Any mineral field as a global fallback (early game only).

        When a target is chosen a virtual slot is reserved via _reserve() so
        that the next drone evaluated in the same frame sees reduced headroom
        and won't overshoot the destination.  Drones for which no slot remains
        get None back, execute() returns False, and the ability selector lets
        them fall through to the next best idea (e.g. mine minerals instead of
        gas, long-distance mine, or just keep their current gather order).
        """
        self._refresh_pending(bot)

        # 1. Unsaturated gas — first geyser with effective headroom.
        for geyser in bot.gas_buildings.ready:
            pending = self._pending_for(geyser.tag)
            if geyser.assigned_harvesters + pending < geyser.ideal_harvesters:
                self._reserve(geyser.tag)
                return geyser

        # 2. Minerals at the nearest effectively-under-saturated base.
        #    We rank townhalls by effective headroom (most under-saturated
        #    first) and break ties by distance so drones prefer nearby bases.
        if bot.townhalls.ready:
            best_target: Optional[Unit] = None
            best_th_tag: Optional[int] = None
            best_score: float = float("inf")

            for th in bot.townhalls.ready:
                nearby_minerals = bot.mineral_field.closer_than(10, th.position)
                if not nearby_minerals:
                    continue  # mined-out base — skip entirely

                # Effective surplus = game surplus + drones already dispatched
                # here this frame.  Negative = still has room.
                pending = self._pending_for(th.tag)
                effective_surplus = th.surplus_harvesters + pending
                if effective_surplus >= 0:
                    continue  # virtually full this frame — skip

                headroom = abs(effective_surplus)
                distance = unit.position.distance_to(th.position)
                score = distance / max(headroom, 1)

                if score < best_score:
                    best_score = score
                    best_th_tag = th.tag
                    best_target = nearby_minerals.closest_to(unit.position)

            if best_target is not None:
                # Reserve a virtual slot so later drones in the same frame
                # see the reduced headroom and don't pile on.
                self._reserve(best_th_tag)
                return best_target

            # All local bases are at/above effective saturation this frame.
            # Try long-distance mining at the next free expansion location.
            #
            # Normally we require a meaningful local surplus (>= _LDM_SURPLUS_THRESHOLD)
            # before dispatching to LDM, so we don't steal drones from a still-filling
            # base.  However, when a base mines out the SC2 engine drops ideal_harvesters
            # to 0 and the workers there lose their gather orders — they become unassigned
            # and no longer appear in surplus_harvesters.  This makes total_real_surplus
            # look artificially low even though there are 20+ idle drones.
            #
            # Stranded-drone bypass: a drone with no active orders and no local minerals
            # to mine is sitting completely idle.  Let it LDM immediately rather than
            # waiting for the surplus threshold.
            total_real_surplus = sum(
                th.surplus_harvesters for th in bot.townhalls.ready
            )
            drone_is_stranded = not unit.orders
            if not drone_is_stranded and total_real_surplus < _LDM_SURPLUS_THRESHOLD:
                return None  # not saturated enough; keep drones local

            # Cap: only move as many drones to LDM as the actual surplus,
            # unless this drone is stranded (idle with nowhere local to mine),
            # in which case we let it through regardless — any LDM is better
            # than sitting completely idle.
            if not drone_is_stranded and getattr(bot, "_ldm_pressure", 0) >= total_real_surplus:
                return None

            # Use ares's precomputed ground-pathing-sorted expansion list so
            # we prefer the closest reachable site, not closest as-the-crow-flies.
            our_positions = {th.position for th in bot.townhalls}
            for exp, _dist in bot.mediator.get_own_expansions:
                if exp in our_positions:
                    continue  # already owned — skip
                nearby_minerals = bot.mineral_field.closer_than(10, exp)
                if not nearby_minerals:
                    continue  # expansion site is mined out
                # Cap: 2 workers per mineral patch (same as normal saturation).
                ldm_cap = len(nearby_minerals) * 2
                if self._pending_for(exp) >= ldm_cap:
                    continue  # this LDM site is virtually full this frame
                # Found a viable long-distance target — reserve a slot and
                # increment the expansion pressure counter on the bot.
                self._reserve(exp)
                bot._ldm_pressure = getattr(bot, "_ldm_pressure", 0) + 1
                return nearby_minerals.closest_to(unit.position)

            # Nowhere productive to send this drone this frame.
            return None

        # 3. No ready townhalls yet (early game — hatchery still morphing).
        #    Mine the nearest mineral so drones don't sit idle.
        if bot.mineral_field:
            return bot.mineral_field.closest_to(unit.position)

        return None


# ---------------------------------------------------------------------------
# WorkerBuildAbility
# ---------------------------------------------------------------------------

class WorkerBuildAbility(Ability):
    """
    Placeholder ability for when a drone is ordered to construct a building.

    Not yet fully implemented — exists to claim the "build" goal slot so that
    future building-placement tactics can route through the ability layer.
    """

    UNIT_TYPES: Set[UnitID] = {UnitID.DRONE, UnitID.SCV, UnitID.PROBE}
    GOAL: str = "build"
    priority: int = 90

    def can_use(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        # Needs a target position to build at
        return context.target_position is not None

    def execute(self, unit: Unit, context: AbilityContext, bot: "ManifestorBot") -> bool:
        # Delegated to the building tactic — return False so it falls through.
        return False


# ---------------------------------------------------------------------------
# MiningTactic — a thin TacticModule that wraps MineAbility
# ---------------------------------------------------------------------------
# This makes mining visible in the tactic layer so it participates in the
# confidence comparison. Workers can be "out-bid" by higher-confidence tactics
# (e.g. CitizensArrestTactic) and switch off mining for that tick.

class MiningTactic:
    """
    Tactic wrapper that makes mining participate in the idea auction.

    This is NOT a full TacticModule subclass — it produces a TacticIdea with
    a pre-built AbilityContext so the ability selector uses the registry path
    directly, bypassing create_behavior().
    """

    name = "MiningTactic"
    is_group_tactic = False
    blocked_strategies = frozenset()

    def is_applicable(self, unit: Unit, bot: "ManifestorBot") -> bool:
        if unit.type_id not in {UnitID.DRONE, UnitID.SCV, UnitID.PROBE}:
            return False
        # Don't redirect drones that are already claimed to build something
        cq = getattr(bot, "construction_queue", None)
        if cq is not None and cq.claimed_by_drone(unit.tag) is not None:
            return False
        return True

    def generate_idea(self, unit: Unit, bot, heuristics, current_strategy):
        from ManifestorBot.manifests.tactics.base import TacticIdea

        # Mining confidence: moderate baseline, boosted when economy heuristic
        # is high (strategy wants more income) and depressed when threat is low
        # and other tactics should drive workers to fight.
        profile = current_strategy.profile()
        confidence = 0.45  # baseline — mining is the default worker job
        evidence = {"baseline": 0.45}

        econ_bias = getattr(profile, "econ_bias", 0.0)
        confidence += econ_bias
        evidence["econ_bias"] = econ_bias

        # Suppress mining if threat is high (enables CitizensArrest to out-bid)
        threat = heuristics.threat_level
        if threat > 0.6:
            penalty = (threat - 0.6) * 0.3
            confidence -= penalty
            evidence["threat_penalty"] = -round(penalty, 3)

        if confidence < 0.20:
            return None

        # Attach a pre-built AbilityContext so the selector uses the registry
        context = AbilityContext(
            goal="mine",
            aggression=1.0 - confidence,
            confidence=confidence,
            evidence=evidence,
        )
        idea = TacticIdea(
            tactic_module=self,
            confidence=confidence,
            evidence=evidence,
            target=None,
        )
        idea.context = context  # selector checks for this attribute
        return idea

    def create_behavior(self, unit, idea, bot):
        # Should not be called — ability selector intercepts first.
        # Returning None is safe (no-op).
        return None


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_worker_abilities() -> None:
    """
    Populate the ability registry with all worker abilities.

    Call once from ManifestorBot.on_start().
    """
    ability_registry.register(UnitID.DRONE, MineAbility())
    ability_registry.register(UnitID.DRONE, WorkerBuildAbility())
    # Future: SCV, Probe, and race-specific abilities registered here.


# ---------------------------------------------------------------------------
# SC2 gather ability IDs (used in can_use to detect active mining)
# ---------------------------------------------------------------------------

# All gather/return abilities — used to detect that a drone is currently mining.
_GATHER_ABILITIES: frozenset[AbilityId] = frozenset({
    AbilityId.HARVEST_GATHER,
    AbilityId.HARVEST_GATHER_DRONE,
    AbilityId.HARVEST_GATHER_SCV,
    AbilityId.HARVEST_GATHER_PROBE,
    AbilityId.HARVEST_RETURN,
    AbilityId.HARVEST_RETURN_DRONE,
    AbilityId.HARVEST_RETURN_SCV,
    AbilityId.HARVEST_RETURN_PROBE,
})

# Minimum total real surplus (sum of surplus_harvesters across all ready
# townhalls) before any drones are sent to long-distance mine.
# surplus_harvesters = assigned - ideal, so 3 means we have 3 workers more
# than the game's recommended saturation across all current bases combined.
# This prevents LDM from stealing drones from a still-filling local base.
_LDM_SURPLUS_THRESHOLD: int = 3

# Return-only subset — drones with these orders are mid-transit back to base
# and should never be interrupted (they auto-resume harvest on delivery).
_RETURN_ABILITIES: frozenset[AbilityId] = frozenset({
    AbilityId.HARVEST_RETURN,
    AbilityId.HARVEST_RETURN_DRONE,
    AbilityId.HARVEST_RETURN_SCV,
    AbilityId.HARVEST_RETURN_PROBE,
})
