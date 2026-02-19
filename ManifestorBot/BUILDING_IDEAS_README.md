# Building Ideas Framework

## What was missing

The existing Manifestor cognitive loop generates **unit ideas** every 10 frames — each mobile unit runs through all registered `TacticModule`s, the best idea is selected, suppression is applied, and the surviving idea is executed via an Ares behavior.

Buildings were explicitly excluded:

```python
# manifestor_bot.py  _generate_unit_ideas()
if unit_role in {UnitRole.BUILDING, UnitRole.GATHERING}:
    continue
```

There was no equivalent pipeline for the three things buildings can do on their own initiative:

| Category         | Examples                                              |
|------------------|-------------------------------------------------------|
| **Train / morph**   | Hatchery → Drone, Barracks → Marine                |
| **Research upgrade**| Lair → Metabolic Boost, Evo Chamber → +1 attack   |
| **Set rally point** | Point new units toward the army or back to base    |

---

## New files

```
ManifestorBot/manifests/tactics/
    building_base.py          # Base class, idea dataclass, action enum
    building_tactics.py       # Concrete Zerg modules

ManifestorBot/
    manifestor_bot_patch.py   # Methods / wiring to add to manifestor_bot.py
```

---

## How the system works

It mirrors the unit tactic system exactly:

```
Building
   │
   ▼
BuildingTacticModule.is_applicable(building, bot)
   │   Fast gate — wrong type, busy, blocked strategy → skip
   ▼
BuildingTacticModule.generate_idea(building, bot, heuristics, strategy)
   │   Confidence scoring with additive evidence trail
   │   Returns BuildingIdea or None
   ▼
_should_suppress_building_idea(structure, idea)
   │   Threshold check + cooldown timer (20-frame cooldown vs 50 for units)
   ▼
BuildingTacticModule.execute(building, idea, bot)
   │   Calls _execute_train / _execute_research / _execute_rally
   ▼
Log + commentary + stamp suppressed_ideas[structure.tag]
```

---

## Applying the patch

Open `ManifestorBot/manifestor_bot.py` and make four small changes:

### 1. Imports (top of file)
```python
from ManifestorBot.manifests.tactics.building_base import BuildingTacticModule, BuildingIdea
```

### 2. `__init__` — two new attributes
```python
self.building_modules: List[BuildingTacticModule] = []
self._building_rally_cache: dict = {}
```

### 3. `on_start` — one new call
```python
self._load_building_modules()   # after self._load_tactic_modules()
```

### 4. `on_step` — one new call
```python
await self._generate_building_ideas()   # after await self._generate_unit_ideas()
```

### 5. Paste in the new methods from `manifestor_bot_patch.py`
Copy these five methods into the `ManifestorBot` class body:
- `_load_building_modules`
- `_generate_building_ideas`
- `_should_suppress_building_idea`
- `_building_idea_summary` (standalone helper — can live outside the class)

---

## Bundled modules (Zerg)

| Module | BUILDING_TYPES | Action | Fires when... |
|--------|---------------|--------|---------------|
| `ZergRallyTactic` | Hatchery / Lair / Hive | SET_RALLY | Army centroid has drifted >20 units from last rally |
| `ZergWorkerProductionTactic` | Hatchery / Lair / Hive | TRAIN (Drone) | saturation_delta > 0, building idle |
| `ZergArmyProductionTactic` | Hatchery / Lair / Hive | TRAIN (Zergling/Roach/…) | strategy is aggressive or army value is behind |
| `ZergUpgradeResearchTactic` | Spawning Pool, Evo Chamber, etc. | RESEARCH | Upgrade is unresearched, affordable, building idle |

---

## Adding new modules

Subclass `BuildingTacticModule`, set `BUILDING_TYPES`, implement the three abstract methods, and add an instance to `_load_building_modules`:

```python
class MyCustomModule(BuildingTacticModule):
    BUILDING_TYPES = frozenset({UnitID.BARRACKS})

    def is_applicable(self, building, bot):
        return (
            building.type_id in self.BUILDING_TYPES
            and self._building_is_ready(building)
            and self._building_is_idle(building)
        )

    def generate_idea(self, building, bot, heuristics, current_strategy):
        ...
        return BuildingIdea(
            building_module=self,
            action=BuildingAction.TRAIN,
            confidence=0.6,
            evidence={"reason": 0.6},
            train_type=UnitID.MARINE,
        )

    def execute(self, building, idea, bot):
        return self._execute_train(building, idea, bot)
```

Then in `_load_building_modules`:
```python
self.building_modules.append(MyCustomModule())
```

---

## Extending TacticalProfile for production biases

Right now, strategy biases on building modules are inferred from the existing `engage_bias` and `retreat_bias` fields. When you want explicit production control from strategy, add two fields to `TacticalProfile` in `strategy.py`:

```python
@dataclass(frozen=True)
class TacticalProfile:
    ...
    produce_bias:  float = 0.0   # +0.3 = build more army; -0.3 = drone up
    research_bias: float = 0.0   # +0.3 = prioritise upgrades aggressively
```

Then consume them in `generate_idea`:
```python
profile = current_strategy.profile()
confidence += profile.produce_bias * 0.3
evidence["strategy_produce_bias"] = profile.produce_bias * 0.3
```
