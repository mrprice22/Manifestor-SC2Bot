# Manifestor Bot — Tactics Module

> **Scope:** Everything under `ManifestorBot/manifests/tactics/` plus the strategy and heuristic layers that feed it.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Game Phase — The Missing Dimension](#2-game-phase--the-missing-dimension)
3. [Strategy Layer](#3-strategy-layer)
4. [Composition Curves](#4-composition-curves)
5. [Unit Tactics (Mobile Units)](#5-unit-tactics-mobile-units)
6. [Building Tactics (Structures)](#6-building-tactics-structures)
7. [The Counter-Play Table](#7-the-counter-play-table)
8. [The Opening Build Runner](#8-the-opening-build-runner)
9. [Per-Frame Data Flow (Full Pipeline)](#9-per-frame-data-flow-full-pipeline)
10. [Adding a New Unit Tactic](#10-adding-a-new-unit-tactic)
11. [Adding a New Building Module](#11-adding-a-new-building-module)
12. [Adding a New Strategy](#12-adding-a-new-strategy)
13. [Design Rules & Anti-Patterns](#13-design-rules--anti-patterns)
14. [Future Integration Points](#14-future-integration-points)

---

## 1. Architecture Overview

The tactics module is a **two-track idea system** — one track for mobile units, one for structures — both feeding into the same confidence → suppression → execute pipeline.

```
                        ┌─────────────────────────────────────────────────┐
                        │             STAGE 1: Heuristics                 │
                        │  Pure math on game state. Runs every frame.     │
                        │  Outputs: HeuristicState (~200 named signals)   │
                        │  Includes: game_phase (0.0 → 1.0)              │
                        └────────────────────┬────────────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────────────┐
                        │          STAGE 2-6: Strategy Classification     │
                        │  Current named Strategy (e.g. STOCK_STANDARD).  │
                        │  Calls .profile() to get TacticalProfile.       │
                        │  Profile.active_composition(game_phase) gives   │
                        │  the live CompositionTarget for this moment.    │
                        └──────────┬──────────────────────────┬───────────┘
                                   │                          │
              ┌────────────────────▼───┐      ┌──────────────▼────────────┐
              │   STAGE 7A             │      │   STAGE 7B                │
              │   Unit Ideas           │      │   Building Ideas          │
              │   Every 10 frames      │      │   Every 20 frames         │
              │   Mobile units only    │      │   Structures only         │
              └────────────┬───────────┘      └──────────────┬────────────┘
                           │                                 │
              ┌────────────▼─────────────────────────────────▼────────────┐
              │         Confidence → Suppression → Execution               │
              │  threshold: 0.40 | cooldown: 50 frames (units)            │
              │                                 20 frames (buildings)     │
              └────────────────────────────────────────────────────────────┘
```

**Core principle:** Tactics never touch the `Strategy` enum directly. They call `current_strategy.profile()` and operate on the `TacticalProfile` dataclass. This keeps the coupling inspectable and keeps individual tactics ignorant of strategy-switching logic.

---

## 2. Game Phase — The Missing Dimension

### What it is

`game_phase` is a continuous float from **0.0 (early game) to 1.0 (late game)** stored in `HeuristicState`. It rises organically as observable game facts accumulate — not from a timer.

```python
# In HeuristicState:
game_phase: float = 0.0
```

### How it is computed

```python
def _update_game_phase(self) -> None:
    score = 0.0

    # Base count — the strongest signal (4 bases → +0.30)
    bases = len(self.bot.townhalls)
    score += min(0.30, bases * 0.075)

    # Tech tier
    if self.bot.structures(UnitID.HIVE).ready:
        score += 0.40
    elif self.bot.structures(UnitID.LAIR).ready:
        score += 0.20

    # Completed upgrades (each → small bump, max +0.20)
    score += min(0.20, len(self.bot.state.upgrades) * 0.03)

    # Army supply as fraction of 200 (max +0.10)
    army_supply = self.bot.supply_used - len(self.bot.workers) - len(self.bot.structures)
    score += min(0.10, army_supply / 200.0)

    self.current_state.game_phase = min(1.0, score)
```

### Why this beats time-based thresholds

Casters say "early game" / "mid game" / "late game" based on what they *see* — tech tier, base count, army composition — not the game clock. A fast cheese is already in "mid game" at 3 minutes. A greedy macro game might still be "early game" at 7 minutes. `game_phase` reflects observable reality, not elapsed time.

### Approximate landmark values

| `game_phase` | Typical state |
|---|---|
| 0.00 – 0.20 | 1–2 bases, Hatchery only, Spawning Pool just built |
| 0.20 – 0.40 | Natural taken, Lair started, first Queens up |
| 0.40 – 0.60 | 3 bases, Lair complete, Roach Warren / Hydralisk Den |
| 0.60 – 0.80 | 4 bases, Hive started, upgrades rolling |
| 0.80 – 1.00 | Hive complete, Ultralisk Cavern, maxing on 5+ bases |

---

## 3. Strategy Layer

### Named Strategies

The bot operates under exactly one `Strategy` at a time, set in `self.current_strategy`. Strategies are descriptive names for the high-level intent:

| Strategy | Intent | Posture |
|---|---|---|
| `JUST_GO_PUNCH_EM` | Full commit, march in and fight | Aggressive |
| `ALL_IN` | Everything now, no recovery | Aggressive |
| `KEEP_EM_BUSY` | Constant harassment, never fully commit | Aggressive |
| `WAR_ON_SANITY` | Multi-front chaos, split attention | Aggressive |
| `STOCK_STANDARD` | Neutral, react to game state | Balanced |
| `WAR_OF_ATTRITION` | Grind slowly, trade favorably | Balanced |
| `BLEED_OUT` | Economic harassment, preserve army | Balanced |
| `DRONE_ONLY_FORTRESS` | Survive, defend economy, never attack | Defensive |

### TacticalProfile

Each Strategy maps to a `TacticalProfile` dataclass — the **only** place strategy identity leaks into the tactical layer:

```python
@dataclass
class TacticalProfile:
    # Combat biases — additive modifiers on tactic confidence scores
    engage_bias:   float = 0.0   # +0.35 = press forward; -0.45 = hold back
    retreat_bias:  float = 0.0   # +0.40 = preserve units; -0.50 = sacrifice ok
    harass_bias:   float = 0.0   # +0.40 = harass constantly; -0.20 = no side missions
    cohesion_bias: float = 0.0   # +0.35 = stay grouped; -0.30 = spread out
    hold_bias:     float = 0.0   # +0.40 = hold chokes; -0.30 = go forward
    sacrifice_ok:  bool  = False # If True, units fight even at low health

    # Composition curve — how army should look across game phases
    composition_curve: list[tuple[float, CompositionTarget]] = field(
        default_factory=list
    )
```

### Getting the active profile

```python
profile = current_strategy.profile()
# profile is a TacticalProfile — consume its fields, never inspect strategy enum
```

---

## 4. Composition Curves

### Why not a static desired_army dict

A static `{ZERGLING: 20, ROACH: 10}` dict fails in two ways:

1. **No phase awareness** — Zerglings are the correct unit at `game_phase=0.1`. Ultralisks are correct at `game_phase=0.85`. A single dict can't express this.
2. **Count vs ratio** — "I want 60% Roaches" scales correctly as your army grows. "I want 10 Roaches" becomes wasteful at 150 supply or insufficient at 50.

### CompositionTarget

```python
@dataclass
class CompositionTarget:
    ratios: dict[UnitID, float]
    # Desired fractional share of total army supply per unit type.
    # Does NOT need to sum to 1.0 — the system normalises internally.
    # "ZERGLING: 0.5, ROACH: 0.3" means "more zerglings than roaches".

    army_supply_target: int
    # How many supply of combat units we want at this phase.
    # Drives SCALE (how big is the army?) separately from composition (what's in it?).

    max_hatcheries: int
    # Expansion cap at this phase. Prevents runaway hatchery spam before
    # the Dream system can take over expansion decisions.
```

### Composition curve structure

`TacticalProfile.composition_curve` is an ordered list of `(min_phase_threshold, CompositionTarget)` pairs. The active target is the **last entry whose threshold ≤ current game_phase**:

```python
def active_composition(self, game_phase: float) -> Optional[CompositionTarget]:
    active = None
    for threshold, target in self.composition_curve:
        if game_phase >= threshold:
            active = target
    return active
```

### Example — STOCK_STANDARD

```python
Strategy.STOCK_STANDARD: TacticalProfile(
    engage_bias = 0.0,
    composition_curve = [
        (0.0, CompositionTarget(           # Early: queens anchor, lings harass
            ratios={
                UnitID.QUEEN:    0.40,
                UnitID.ZERGLING: 0.60,
            },
            army_supply_target=20,
            max_hatcheries=2,
        )),
        (0.30, CompositionTarget(          # Mid: roach core established
            ratios={
                UnitID.QUEEN:    0.20,
                UnitID.ZERGLING: 0.30,
                UnitID.ROACH:    0.50,
            },
            army_supply_target=60,
            max_hatcheries=4,
        )),
        (0.70, CompositionTarget(          # Late: hive tech transition
            ratios={
                UnitID.ROACH:     0.30,
                UnitID.HYDRALISK: 0.40,
                UnitID.ULTRALISK: 0.30,
            },
            army_supply_target=140,
            max_hatcheries=6,
        )),
    ],
),
```

### How unit selection uses the curve

`ZergArmyProductionTactic._pick_unit()` walks the active `CompositionTarget.ratios` and selects the unit type we are **furthest below our desired ratio**:

```python
def _pick_unit(self, building, bot) -> Optional[UnitID]:
    target = bot.current_strategy.profile().active_composition(
        bot.heuristic_manager.get_state().game_phase
    )
    if not target:
        return None

    total_army_supply = sum(
        bot.units(uid).amount * SUPPLY_COST[uid]
        for uid in target.ratios
    )

    best_unit, biggest_deficit = None, -999.0

    for unit_type, desired_ratio in target.ratios.items():
        if not self._can_afford_train(unit_type, bot):
            continue
        if bot.tech_requirement_progress(unit_type) < 1.0:
            continue
        current_supply = bot.units(unit_type).amount * SUPPLY_COST.get(unit_type, 1)
        current_ratio = current_supply / max(1, total_army_supply)
        deficit = desired_ratio - current_ratio
        if deficit > biggest_deficit:
            biggest_deficit = deficit
            best_unit = unit_type

    return best_unit
```

Units not yet unlocked (tech requirement < 1.0) are skipped automatically — they never appear until the tech path catches up with `game_phase`. You do **not** need to manually gate Ultralisks; they gate themselves.

### Supply target and expansion cap

`army_supply_target` is consumed by `ZergArmyProductionTactic.generate_idea()` as a modifier on confidence — if our current army supply is well below the target for this phase, confidence rises:

```python
current_army_supply = bot.supply_used - len(bot.workers)
gap = target.army_supply_target - current_army_supply
supply_sig = min(0.3, gap * 0.005)  # small signal, not a hard gate
confidence += supply_sig
evidence["supply_gap"] = supply_sig
```

`max_hatcheries` is a hard gate in `ZergStructureBuildTactic.generate_idea()`:

```python
active = bot.current_strategy.profile().active_composition(
    bot.heuristic_manager.get_state().game_phase
)
if active:
    current = bot.townhalls.amount + bot.already_pending(UnitID.HATCHERY)
    if current >= active.max_hatcheries:
        return None   # don't build another hatchery yet
```

---

## 5. Unit Tactics (Mobile Units)

### File: `tactics/base.py` — `TacticModule`

All mobile-unit tactics inherit from `TacticModule` and implement three methods:

```
is_applicable(unit, bot)   → bool
    Fast structural gate. Never do confidence math here.

generate_idea(unit, bot, heuristics, current_strategy) → Optional[TacticIdea]
    Score the situation. Build confidence additively with an evidence trail.
    Return None if confidence < floor (~0.15).

create_behavior(unit, idea, bot) → Optional[CombatIndividualBehavior]
    Convert an approved idea into an Ares behavior.
    Return None if the target has died between generation and execution.
```

### `TacticIdea`

```python
@dataclass
class TacticIdea:
    tactic_module: TacticModule
    confidence: float       # 0.0 – 1.0
    evidence: dict          # Named sub-signal contributions (full paper trail)
    target: Optional[object] = None   # Unit, Point2, or None
    context: Optional[AbilityContext] = None
```

### Registered unit tactic modules

| Module | Applies to | Fires when |
|---|---|---|
| `BuildingTactic` | Drones | Drone has a pending construction order to execute |
| `MiningTactic` | Workers | Worker is idle or displaced from minerals |
| `StutterForwardTactic` | Army units | Engaged enemy, unit can kite or stutter-step |
| `HarassWorkersTactic` | Army units | Enemy workers visible, strategy has harass bias |
| `FlankTactic` | Army units | Enemy engaged frontally, flanking angle available |
| `HoldChokePointTactic` | Army units | Army should hold a known choke, not advance |
| `RallyToArmyTactic` | Army units | Unit is isolated far from army centroid |
| `KeepUnitSafeTactic` | Army units | Unit at low health under fire, retreat viable |
| `CitizensArrestTactic` | Workers | Enemy raider in mineral line, enough posse to mob |

### Suppression

After all ideas for a unit are collected and sorted by confidence, the **best** idea is tested against suppression:

```python
def _should_suppress_idea(self, unit, idea) -> bool:
    if idea.confidence < 0.40:          # Hard threshold
        return True
    if unit.tag in self.suppressed_ideas:
        if frames_since < 50:           # 50-frame cooldown (~2.2 sec)
            return True
    if len(unit.orders) > 1:            # Already busy with multiple orders
        return True
    return False
```

### Group tactics

Some tactics (`CitizensArrestTactic`) set `is_group_tactic = True`. These are collected separately in Phase 1 and consolidated in Phase 2: if enough units share the same group idea targeting the same enemy, one coordinated `give_same_action` is issued. If the group is too small, **all** ideas in that bucket are dropped — no lone worker suicide charges.

### Blocked strategies

Override `blocked_strategies` to prevent a tactic from ever firing under specific strategies:

```python
@property
def blocked_strategies(self):
    from ManifestorBot.manifests.strategy import Strategy
    return frozenset({Strategy.DRONE_ONLY_FORTRESS})
```

---

## 6. Building Tactics (Structures)

### File: `tactics/building_base.py` — `BuildingTacticModule`

Structures run a parallel pipeline to unit tactics. Buildings have three possible actions:

```python
class BuildingAction(Enum):
    TRAIN     # Queue a unit (Drone, Zergling, Roach, …)
    RESEARCH  # Start an upgrade (Metabolic Boost, +1 attack, …)
    SET_RALLY # Correct the rally point
```

All building modules inherit from `BuildingTacticModule`:

```
is_applicable(building, bot)   → bool
generate_idea(building, bot, heuristics, current_strategy, counter_ctx) → Optional[BuildingIdea]
execute(building, idea, bot)   → bool
```

### `BuildingIdea`

```python
@dataclass
class BuildingIdea:
    building_module: BuildingTacticModule
    action: BuildingAction
    confidence: float
    evidence: dict = field(default_factory=dict)
    train_type: Optional[UnitID] = None
    upgrade: Optional[UpgradeId] = None
    rally_point: Optional[Point2] = None
```

### Registered building modules (execution order)

| Module | `BUILDING_TYPES` | Action | Core logic |
|---|---|---|---|
| `ZergRallyTactic` | Hatchery / Lair / Hive | `SET_RALLY` | Army centroid has drifted >20 units from last rally point |
| `ZergWorkerProductionTactic` | Hatchery / Lair / Hive | `TRAIN` (Drone) | `saturation_delta > 0`, building idle, supply available |
| `ZergArmyProductionTactic` | Hatchery / Lair / Hive | `TRAIN` (army) | Ratio-deficit against active `CompositionTarget`; counter-play bonus applied on top |
| `ZergUpgradeResearchTactic` | Spawning Pool, Evo Chamber, Roach Warren, etc. | `RESEARCH` | Priority list, affordable, not already researched or in progress |
| `ZergStructureBuildTactic` | Hatchery / Lair / Hive | `TRAIN` (structure dispatch) | Tech building needed per `_STRUCTURE_PRIORITY`; `max_hatcheries` gate |

### Building suppression

Buildings share the same `suppressed_ideas` dict as units (keyed by tag), with a tighter cooldown of **20 frames** (buildings act less frequently than units). The 0.40 confidence threshold applies identically.

### Execution helpers

`BuildingTacticModule` provides three helpers callable from `execute()`:

- `_execute_train(building, idea, bot)` — handles Zerg larva routing automatically
- `_execute_research(building, idea, bot)` — calls `building.research(idea.upgrade)`
- `_execute_rally(building, idea, bot)` — sets rally; falls back to army centroid if `rally_point` is None

---

## 7. The Counter-Play Table

### What it does

The counter-play table (`scout_ledger.py`) watches what the opponent is building and prescribes reactive adjustments. It produces a `CounterContext` each frame:

```python
@dataclass
class CounterContext:
    priority_train_types: set[UnitID]  # Units we should bias toward training
    production_bonus: float            # Extra confidence bonus for those units
    engage_bias_mod: float             # Additive modifier on aggression_dial
```

### Where it plugs in

Counter-play operates as a **bonus layer on top of** the composition curve — it does not replace or mutate the `CompositionTarget`. The composition curve represents strategic intent. Counter-play represents tactical adaptation.

In `ZergArmyProductionTactic.generate_idea()`:

```python
if train_type in counter_ctx.priority_train_types:
    counter_sig = counter_ctx.production_bonus
    confidence += counter_sig
    evidence["counter_play_bonus"] = counter_sig
```

If the composition curve says "40% Hydras" and the counter table sees enemy Mutalisks and prescribes "train Corruptors," the Corruptor gets a confidence surge that can outbid the Hydralisk for this particular training slot — but the underlying curve is unchanged. Next time the bot has a free hatchery and no Mutalisk threat, it reverts to Hydra training naturally.

### Aggression dial modification

`CounterContext.engage_bias_mod` is applied directly to `HeuristicState.aggression_dial` in `HeuristicManager.update()`:

```python
self.current_state.aggression_dial += ctx.engage_bias_mod * 10
self.current_state.aggression_dial = max(0, min(100, self.current_state.aggression_dial))
```

This means a scouted Cannon rush can spike the dial toward 25 (defensive) even if the current strategy is `JUST_GO_PUNCH_EM`.

---

## 8. The Opening Build Runner

### What it is

`AresBot.build_order_runner` is a deterministic supply-gated to-do list for the first ~3–5 minutes. It runs inside `AresBot.on_step()` before `ManifestorBot`'s own logic, and completes when all steps are executed or `set_build_completed()` is called.

It is **not** a strategy expression system. It is a precise opening script.

### Relationship to strategies

Strategies and openings are **many-to-one**: a small number of opening archetypes (3–4 builds) are selected based on the current strategy at game start. Several strategies can share the same opener.

| Opening | Strategies that use it |
|---|---|
| `StandardOpener` | `STOCK_STANDARD`, `WAR_OF_ATTRITION`, `BLEED_OUT` |
| `EarlyAggression` | `JUST_GO_PUNCH_EM`, `ALL_IN`, `KEEP_EM_BUSY` |
| `MultiProng` | `WAR_ON_SANITY` |
| `TurtleEco` | `DRONE_ONLY_FORTRESS` |

Opening selection happens in `ManifestorBot.on_start()` by mapping `self.current_strategy` to an opening name string and calling `self.build_order_runner.switch_opening(name)`.

### Handoff to the dynamic system

Once `build_order_runner.build_completed` is `True`, the dynamic building and unit tactic systems take full control. The composition curve picks up from wherever `game_phase` currently sits — there is no gap or awkward transition. The build runner's final state (structures built, resources spent, tech tier) naturally pushes `game_phase` to the right starting value for the composition curve.

---

## 9. Per-Frame Data Flow (Full Pipeline)

```
Every frame:
┌──────────────────────────────────────────────────────────────────┐
│  1. HeuristicManager.update()                                    │
│     → Recalculates all ~200 signals including game_phase         │
│     → CounterContext applied: aggression_dial modified           │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────┐
│  2. AresBot.on_step() base class                                 │
│     → build_order_runner.run_build() if not completed            │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
Every 10 frames:
┌─────────────────────────────────▼────────────────────────────────┐
│  3. _generate_unit_ideas()                                       │
│     Phase 1: for each eligible unit:                             │
│       → for each TacticModule: is_applicable? generate_idea?     │
│       → sort ideas by confidence, take best                      │
│       → suppress if < 0.40 or cooldown active                   │
│       → bucket into group vs individual                          │
│     Phase 2a: group consolidation (CitizensArrest etc.)          │
│     Phase 2b: individual execution → Ares behavior               │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
Every 20 frames:
┌─────────────────────────────────▼────────────────────────────────┐
│  4. _generate_building_ideas()                                   │
│     for each structure:                                          │
│       → get active CompositionTarget from profile.active_composition(game_phase) │
│       → for each BuildingTacticModule: is_applicable? generate_idea?             │
│       → sort building ideas by confidence, take best             │
│       → suppress if < 0.40 or 20-frame cooldown active          │
│       → execute: TRAIN / RESEARCH / SET_RALLY                    │
│       → stamp suppression cooldown                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 10. Adding a New Unit Tactic

1. Create your file under `manifests/tactics/` (or add to an existing one).
2. Subclass `TacticModule`, implement the three required methods.
3. Register an instance in `ManifestorBot._load_tactic_modules()`.

```python
from ManifestorBot.manifests.tactics.base import TacticModule, TacticIdea
from sc2.ids.unit_typeid import UnitTypeId as UnitID

class BurrowRetreatTactic(TacticModule):
    """Burrow a badly wounded roach to recover."""

    @property
    def blocked_strategies(self):
        from ManifestorBot.manifests.strategy import Strategy
        # Never burrow in an all-in — every unit should fight to the death
        return frozenset({Strategy.ALL_IN, Strategy.JUST_GO_PUNCH_EM})

    def is_applicable(self, unit, bot) -> bool:
        if unit.type_id != UnitID.ROACH:
            return False
        if not bot.state.upgrades:  # need burrow researched
            return False
        # UpgradeId.BURROW in bot.state.upgrades is the real check; simplified here
        return True

    def generate_idea(self, unit, bot, heuristics, current_strategy):
        confidence = 0.0
        evidence = {}

        # Sub-signal: unit is badly hurt
        hp_ratio = unit.health / unit.health_max if unit.health_max > 0 else 1.0
        if hp_ratio > 0.35:
            return None  # not hurt enough to care
        hurt_sig = (0.35 - hp_ratio) * 1.5
        confidence += hurt_sig
        evidence["hp_ratio"] = hurt_sig

        # Sub-signal: under fire (enemies nearby)
        nearby = bot.enemy_units.closer_than(8, unit.position)
        if nearby:
            fire_sig = min(0.3, len(nearby) * 0.08)
            confidence += fire_sig
            evidence["under_fire"] = fire_sig

        # Sub-signal: strategy retreat bias
        profile = current_strategy.profile()
        confidence += profile.retreat_bias
        evidence["strategy_retreat_bias"] = profile.retreat_bias

        if confidence < 0.15:
            return None

        return TacticIdea(self, confidence, evidence)

    def create_behavior(self, unit, idea, bot):
        from ares.behaviors.combat.individual import BurrowUnit
        return BurrowUnit()
```

Then in `_load_tactic_modules`:

```python
self.tactic_modules = [
    ...
    BurrowRetreatTactic(),
]
```

---

## 11. Adding a New Building Module

1. Subclass `BuildingTacticModule`.
2. Set `BUILDING_TYPES`.
3. Implement `is_applicable`, `generate_idea`, `execute`.
4. Add an instance to `ManifestorBot._load_building_modules()`.

```python
from ManifestorBot.manifests.tactics.building_base import (
    BuildingTacticModule, BuildingIdea, BuildingAction
)
from sc2.ids.unit_typeid import UnitTypeId as UnitID

class ZergQueenInjectTactic(BuildingTacticModule):
    """Inject larva at hatcheries when a queen is idle nearby."""

    BUILDING_TYPES = frozenset({UnitID.HATCHERY, UnitID.LAIR, UnitID.HIVE})

    def is_applicable(self, building, bot) -> bool:
        if building.type_id not in self.BUILDING_TYPES:
            return False
        if not self._building_is_ready(building):
            return False
        # A queen must be nearby and idle to make this worth scoring
        queens = bot.units(UnitID.QUEEN).closer_than(5, building.position)
        return any(q.is_idle for q in queens)

    def generate_idea(self, building, bot, heuristics, current_strategy, counter_ctx):
        confidence = 0.60  # Injecting is almost always correct

        # Defensive strategies are slightly more conservative with queens
        profile = current_strategy.profile()
        if profile.hold_bias > 0.2:
            confidence -= 0.05  # Tiny drag — queens might be needed for creep

        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,  # Reuse TRAIN action for the inject ability
            confidence=confidence,
            evidence={"base_inject": 0.60, "hold_drag": confidence - 0.60},
            train_type=None,   # Handled manually in execute()
        )

    def execute(self, building, idea, bot) -> bool:
        queens = bot.units(UnitID.QUEEN).closer_than(5, building.position)
        idle_queen = next((q for q in queens if q.is_idle), None)
        if idle_queen is None:
            return False
        from sc2.ids.ability_id import AbilityId
        idle_queen(AbilityId.EFFECT_INJECTLARVA, building)
        return True
```

Register in `_load_building_modules`:

```python
self.building_modules = [
    ZergRallyTactic(),
    ZergQueenInjectTactic(),   # before worker production so queens fire first
    ZergWorkerProductionTactic(),
    ZergArmyProductionTactic(),
    ZergUpgradeResearchTactic(),
    ZergStructureBuildTactic(),
]
```

---

## 12. Adding a New Strategy

1. Add a member to the `Strategy` enum in `strategy.py`.
2. Add a `TacticalProfile` entry to `_PROFILES`.
3. Add a `composition_curve` that reflects the strategy's army philosophy.
4. Optionally add an opening name to `zerg_builds.yml` and map it in `on_start`.

```python
# In Strategy enum:
ECONOMIC_BLITZ = "Economic Blitz"

# In _PROFILES:
Strategy.ECONOMIC_BLITZ: TacticalProfile(
    engage_bias   = +0.15,
    retreat_bias  = -0.10,
    harass_bias   = +0.10,
    cohesion_bias = +0.05,
    hold_bias     = 0.0,
    sacrifice_ok  = False,
    composition_curve = [
        (0.0, CompositionTarget(
            ratios={UnitID.QUEEN: 0.30, UnitID.ZERGLING: 0.70},
            army_supply_target=16,
            max_hatcheries=3,          # Fast third hatchery
        )),
        (0.25, CompositionTarget(
            ratios={UnitID.ROACH: 0.60, UnitID.ZERGLING: 0.40},
            army_supply_target=80,
            max_hatcheries=5,
        )),
        (0.65, CompositionTarget(
            ratios={
                UnitID.ROACH:     0.25,
                UnitID.HYDRALISK: 0.50,
                UnitID.ULTRALISK: 0.25,
            },
            army_supply_target=160,
            max_hatcheries=7,
        )),
    ],
),
```

---

## 13. Design Rules & Anti-Patterns

### Rules

| ✅ Do | ❌ Don't |
|---|---|
| Call `current_strategy.profile()` and consume the `TacticalProfile` | Inspect `current_strategy == Strategy.ALL_IN` inside a tactic |
| Add every sub-signal to `evidence` dict | Compute confidence and discard the breakdown |
| Return `None` from `generate_idea` when confidence is structurally impossible | Return a `TacticIdea` with confidence 0.0 |
| Use `is_applicable` as a cheap structural gate only | Run expensive queries in `is_applicable` |
| Use `active_composition(game_phase)` for unit type selection | Hard-code unit priorities as a static list |
| Let `game_phase` unlock tech units naturally via the tech-requirement check | Manually time-gate Ultralisks or Mutalisks |
| Use `max_hatcheries` from `CompositionTarget` for expansion caps | Hard-code a hatchery count anywhere in tactic code |

### On composition ratios and supply

`CompositionTarget.ratios` express **relative preference**, not exact population counts. They do **not** need to sum to 1.0. The system normalizes internally when computing deficits. Writing `{ZERGLING: 0.5, ROACH: 0.3}` means "I want roughly 5 Zerglings for every 3 Roaches" — the total army size is controlled by `army_supply_target` separately.

Do not try to encode a 200-supply endgame army in the ratios. At `game_phase=0.9` with `army_supply_target=140`, a `{ROACH: 0.3, HYDRA: 0.4, ULTRA: 0.3}` ratio set will naturally produce roughly `42 / 56 / 42` supply of each — which maps to about 21 Roaches, 28 Hydralisks, and 7 Ultralisks. Correct proportions emerge from the math; you declare intent.

---

## 14. Future Integration Points

### The Dream (Phase 5 of roadmap)

The Dream system will produce phase-aware projections of the game state 2–3 minutes into the future. Its main hook into this module is:

**Hot-swapping the active `CompositionTarget`.** The Dream can synthesize a custom `CompositionTarget` for the current moment — based on projected opponent army, map state, and historical game data — and inject it as a synthetic curve entry that overrides the profile default:

```python
# Dream injects a custom target, bypassing the static curve
bot.dream_composition_override = CompositionTarget(
    ratios={UnitID.CORRUPTOR: 0.6, UnitID.ROACH: 0.4},
    army_supply_target=80,
    max_hatcheries=4,
)
```

`active_composition()` checks for this override before walking the curve. When the Dream retracts its override (e.g. Mutalisk threat resolved), the system falls back to the profile curve seamlessly.

### The Ant Algorithm (Phase 2 of roadmap)

Pheromone maps will feed into `HeuristicState` as additional signals — `danger_density`, `resource_denial_pct`, `harassment_opportunity_score`. These slot into confidence sub-signals in existing tactics without requiring structural changes. `HoldChokePointTactic` and `HarassWorkersTactic` are the primary consumers.

The pheromone map will also improve `game_phase` computation by adding a "map pressure" component: heavy enemy activity across the map pushes phase upward faster, reflecting the practical reality that a map-control game ages faster than a passive macro game.

### Strategy switching (future classifier)

Currently `self.current_strategy` is set at game start and changed manually via `change_strategy()`. When the opponent modeling system (Phase 4) arrives, it will call `change_strategy()` automatically based on classified opponent intent. The tactics layer requires zero changes — it already reads `current_strategy` dynamically every idea-generation cycle.
