# Manifestor Bot 🧬

> *"Don't react to what is happening. Decide what happens next."*

A StarCraft II Zerg bot built on [Ares-SC2](https://aressc2.github.io/ares-sc2/) that takes a fundamentally different approach to bot AI — thinking deeply before acting, manifesting a dream of the future, and letting that vision govern every decision from macro to the individual unit.

---

## Philosophy

Most SC2 bots maximize actions per minute. Manifestor Bot minimizes them.

Rather than reacting to the current game state as fast as possible, Manifestor operates on the principle that a decision made now should be evaluated based on how it shapes a **projected future state** — not on what is happening in the current frame. Every unit participates in a continuous ideation loop, generating candidate tactical ideas and suppressing most of them, acting only when the expected value of intervention clearly outweighs the cost of disrupting the natural flow of the game.

The bot is named for this core idea: it does not respond to the game. It **manifests** the game it expects to win.

---

## Architecture Overview

### The Two-Speed Cognitive Loop

Each unit runs on two separate clocks:

- **Idea generation** — every frame, each unit evaluates its local context and proposes a candidate tactic from the applicable tactic library
- **Action execution** — every ~10 frames, the bot decides whether to act on any idea or suppress it and allow automation to continue

The default state is **non-interference**. Built-in SC2 automation — patrol paths, attack-move, command queuing — handles routine execution. The bot intervenes only when a course correction is clearly necessary.

### The Dream of Future

At any moment the bot maintains a live **Dream** — a projected series of future game states representing where it intends to be. The Dream is decomposed into named phases, and all decision-making is oriented toward advancing the current phase rather than reacting to the present.

The Dream is never rigid. Scouting data, opponent modeling, and combat outcomes continuously update its viability. When a better Dream exists, the bot pivots.

### Named Strategy at All Times

The bot always operates under exactly one named strategy. Every unit knows both its individual tactic role and the strategy it serves. This allows deliberate sacrifice — a unit can die for a greater strategic purpose rather than always optimizing for its own survival.

### Bottom-Up Ideation

Strategic decisions emerge from unit-level observations upward. There are no hard-coded thresholds like "pull drones at 40 supply." Instead, units recognize conditions locally and surface context-appropriate tactics, which are weighed against the current strategy and global heuristics before any action is taken.

---

## Core Systems

### Heuristic Layer

A suite of continuously calculated game-state signals, visible to all decision-making layers:

| Category | Signals |
|---|---|
| Combat | Momentum, Initiative, Threat Level, Tempo, Army Cohesion |
| Economy | Economic Health, Spend Efficiency, Worker Safety Index, Saturation Delta |
| Map | Creep Coverage %, Vision Dominance, Expansion Race Index, Choke Control |
| Army | Army Value Ratio, Upgrade Advantage, Reinforcement Proximity |
| Meta | Opponent Reaction Speed, Opponent Risk Profile, Strategy Confidence Score |

These feed into a composite **Aggression Dial** (0–100) that all units read, automatically adjusting tactic selection without each unit needing to evaluate every signal independently.

### Tactic Library

Named, coded tactic modules — each with defined trigger conditions, input parameters, and expected outcomes. Tactics are not chosen by a single decision-maker; they emerge from individual units cycling through applicable options and surfacing the best local fit.

**Offensive:** Flank, Full Surround, Hammer and Anvil, Cheese, Kill Their Economy, Slow Their Tech, Headhunter, Maximize AoE, Upgrade Timing Strike, Lead/Kite Enemy, Baneling Bust

**Zerg-Specific:** Nydus Tactical Insertion, Decoy Nydus, Ghetto Blink (Overlord drops), Split/Pre-split Forces, Redirect Widow Mines, Body Block, Changeling Body Block

**Defensive:** Deny Flank, Deny Full Surround, Mislead/Distract

### Tactic Detection (Rule-Based Ensemble)

Enemy tactics are identified by an ensemble of independent rule-based detectors, each returning a confidence score built from geometric and behavioral sub-signals. All detectors run simultaneously. The classifier returns a ranked list of active signals — tactics can co-occur — and logs everything in a format ready for future ML training.

### Opponent Modeling

- Continuous strategy hypothesis updated from scouting and unit composition
- Named label assigned to opponent's current strategy at all times
- Best-next-move projection evaluated each step
- Rock-paper-scissors tactic relationship awareness
- Pheromone map system tracking enemy activity, combat history, and economic threat — all decaying over time

### Persistent Memory (SQLite)

The bot learns across games without retraining. Between-game storage includes:

- **Opponent profiles** — observed strategies, reaction speed, aggression score, last seen
- **Game records** — full strategy timeline per player, key moments anchored to strategic context
- **Tactic success rates** — by tactic, vs. opponent strategy, by map
- **Heuristic baselines** — per map and matchup, so the bot knows if it is ahead or behind relative to normal

Strategy state is recorded as a timeline, not a snapshot — capturing the full narrative of pivots, what triggered them, and whether they succeeded.

---

## Strategy Library

| Strategy | Intent |
|---|---|
| Just Go Punch 'Em | Overwhelming direct aggression |
| War of Attrition | Cost-efficient trading over time |
| All-In | High risk, high reward commitment |
| Keep 'Em Busy | Constant multi-front harassment |
| Stock Standard | Safe, reliable macro opener |
| War on Sanity | Tax opponent attention and APM |
| Bleed Out | Relentless economic pressure |
| Drone Only (Fortress) | Full economic buildup behind static defense |

---

## Built On

**[Ares-SC2](https://aressc2.github.io/ares-sc2/)** provides the infrastructure layer:

- Unit role assignment and tracking
- Behavior + CombatManeuver composition system (tactic modules plug directly in here)
- Influence grids and safe pathfinding
- Fog-of-war memory units
- Combat sim / fight outcome prediction
- Build order runner for opening strategy
- Custom manager registration for extending with new systems

Manifestor Bot uses Ares for execution and infrastructure. All strategic thinking — heuristics, tactic detection, opponent modeling, the Dream — lives in custom managers and behaviors built on top.

---

## Inspirations

- **Dark and Hero** (pro SC2 players) — the playstyle this bot aspires to emulate: deep reads, creative positioning, and a willingness to do the unexpected
- **Serral's Infestor micro** — pre-aiming fungals at projected future positions rather than current ones; the original inspiration for future-state decision making
- **Ant Colony Optimization** — the pheromone map system for persistent environmental memory
- **The Art of War** — named strategic heuristics encoded as high-level priority filters over tactic selection

---

## Project Status

🚧 Early development — infrastructure setup and core loop in progress.

### Roadmap

| Phase | Focus |
|---|---|
| 1 | Core loop — idea generation, action suppression, tactic module structure, named strategy state |
| 2 | Pheromone maps — layered decay maps for enemy activity, combat history, danger zones |
| 3 | Tactic modules — all tactics implemented as Ares `CombatIndividualBehavior` modules with confidence scoring |
| 4 | Opponent modeling — strategy recognition, best-next-move projection, counter-strategy selection |
| 5 | Dream of Future — phase decomposition, step tracking, pivot scoring fed by historical DB |
| 6 | Persistent memory — SQLite game history, strategy timelines, tactic success rates |
| 7 | Arena testing — SC2 AI Arena deployment, loss pattern analysis, weight tuning |

---

## Development Setup

```bash
git clone https://github.com/mrprice22/SC2-Noob-Zerg-Bot.git
cd SC2-Noob-Zerg-Bot

python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac / Linux

pip install burnysc2
```

Requires Python 3.9+ and a local StarCraft II installation.

If you encounter protobuf errors on startup:

```bash
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

---

*Manifestor Bot is a personal research project exploring abstract, future-state oriented AI in real-time strategy games.*
