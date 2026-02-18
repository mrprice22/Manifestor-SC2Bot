# Manifestor Bot — Per-Frame Data Flow

Every game step, the following pipeline runs in order. Each stage feeds into the next, building a richer picture of the game state before any unit acts.

---

## Pipeline Overview

```
[DETERMINISTIC] → [ENSEMBLE] → [CLASSIFIER] → [DIFFUSION] → [DIFFUSION] → [MLP] → [DETERMINISTIC] → [DETERMINISTIC]
   Heuristics       Tactic ID    Strategy ID    Dream         Opp. Move    Weights   Unit Ideas        Execution
```

---

## Stage 1 — Heuristic Calculation `[DETERMINISTIC]`

**Input:** Raw game observation (unit positions, health, resources, upgrades, map state)

**Output:** A ~200-dimensional vector of named heuristic values, globally visible to all subsequent stages

All values are pure math on the observation object — no ML, no ambiguity. This stage is fast, interpretable, and runs every frame without exception.

Signals calculated include:

| Category | Examples |
|---|---|
| Combat | Momentum (rolling score of kills, retreats, map control gained), Initiative, Threat Level, Tempo |
| Economy | Economic Health, Spend Efficiency, Worker Safety Index, Saturation Delta |
| Map | Creep Coverage %, Vision Dominance, Expansion Race Index, Choke Control |
| Army | Army Value Ratio, Upgrade Advantage Delta, Reinforcement Proximity, Army Cohesion |
| Meta | Opponent Reaction Speed estimate, Opponent Risk Profile, Strategy Confidence Score |

**Momentum** deserves special mention: it is a rolling signed score driven by kills, structures destroyed, map control gained, and enemy retreats observed — decaying over time. High positive momentum triggers pressing behaviors; negative momentum shifts the bot toward stabilization or harassment to reclaim it.

All heuristics feed into a composite **Aggression Dial** (0–100) used downstream by the unit idea layer.

---

## Stage 2 — Tactic Recognition `[ENSEMBLE / DETERMINISTIC]`

**Input:** Sliding window of enemy unit positions, velocities, and commands over the last ~10 seconds

**Output:** Ranked list of `TacticSignal(tactic, confidence, evidence)` — one per detector that fired above threshold

Rather than a single classifier, an **ensemble of independent rule-based detectors** runs simultaneously — one per known tactic. Each detector evaluates its own geometric and behavioral sub-signals and returns a confidence score between 0.0 and 1.0, built from weighted contributions of those sub-signals rather than an arbitrary number.

```
SurroundDetector   → confidence: 0.81  (angular coverage: 0.9, inward motion: 0.8, escape blocked: 0.6)
FlankDetector      → confidence: 0.44  (angle separation: 0.6, main force engaged: 0.5, flank velocity: 0.2)
HammerAnvilDetector→ confidence: 0.12
...
```

All signals above a minimum confidence threshold are returned — tactics frequently co-occur (a Hammer and Anvil *is* simultaneously a flank and a body block). The strategy layer receives the full ranked list, not just the winner.

All classification records are logged with raw window snapshots, creating a self-generating labeled dataset for potential future ML training.

---

## Stage 3 — Strategy Labeling `[CLASSIFIER / DETERMINISTIC]`

**Input:** Last ~2 minutes of opponent behavior (unit production observed, expansion timing, aggression patterns, tactic signals from Stage 2)

**Output:** Named strategy label + confidence score

```
→ "opponent strategy: Bio Timing Push, confidence: 0.74"
```

The bot maintains a running hypothesis about which named strategy the opponent is executing, updated continuously. Rather than a hard label swap, confidence scores decay and rebuild as new evidence arrives — the bot can hold two competing hypotheses simultaneously when scouting data is ambiguous.

Recognized strategy archetypes include race-specific patterns (e.g. Bio Timing, Skytoss Transition, 2-base All-in, Macro Expand) as well as general patterns (Economic Greed, Defensive Turtle, Multi-Prong Harassment). New archetypes are added as they are encountered in arena play.

The strategy label feeds directly into the persistent memory system — opponent profiles accumulate labeled strategy histories across games, allowing the bot to load a prior hypothesis when facing a known opponent.

---

## Stage 4 — Dream of Future `[DIFFUSION]`

**Input:** Current heuristic vector (Stage 1) + current named strategy + historical DB priors

**Output:** Best projected future game state from a sampled distribution of 8 plausible futures

The Dream of Future is the bot's imagination. A small diffusion model operates not on raw game pixels but on the compressed heuristic feature space — making it tractable within the 100MB storage budget.

**How it works:**

1. The current heuristic vector is encoded into a 64-dimensional latent representation
2. The diffusion model samples 8 candidate future states by starting from random noise and iteratively denoising, conditioned on the current context
3. Each sampled future is scored against the current strategy's goals using historical DB data as a prior (e.g. "last time momentum dropped this sharply during Bleed Out, what happened next?")
4. The highest-scoring future becomes the **active Dream** — the projected game state the bot is working toward
5. The Dream is re-sampled periodically or whenever a significant game event invalidates it (major army wipe, base lost, strategy pivot)

This produces futures that are both **imaginative** (diffusion explores the possibility space) and **historically grounded** (DB priors weight plausible outcomes over fantasy). The bot is not predicting one fixed future — it is selecting the best path from a distribution.

---

## Stage 5 — Opponent Next-Move Prediction `[DIFFUSION]`

**Input:** Opponent strategy label + confidence (Stage 3) + opponent heuristic estimates + pheromone map state

**Output:** Probability-weighted prediction of the enemy's most likely tactical action in the next ~30 seconds

```
→ "high probability of drop in next 30s (0.71)"
→ "moderate probability of third base expansion (0.44)"
```

A second, smaller diffusion model samples over the opponent's likely action space given their current strategy and observed state. This is not predicting individual unit commands — it is predicting **tactical moves** at the same abstraction level as the tactic library.

This feeds a **course correction evaluation**: given what the opponent is probably about to do, does the current Dream still hold? Does the current strategy need to change? Do any units need reassignment now, before the threat materializes?

This is the mechanism that allows Manifestor to act *ahead* of threats rather than reacting to them.

---

## Stage 6 — Heuristic Weight Adjustment `[MLP]`

**Input:** All outputs from Stages 1–5 combined

**Output:** A weight vector that modulates how much each heuristic influences tactic scoring in Stage 7

A small multi-layer perceptron (~1–2MB) learns that different heuristics should carry different importance depending on the strategic context. For example:

- When the Dream predicts an opponent drop, **Worker Safety Index** weight spikes
- When executing an All-In strategy, **Economic Health** weight drops (we don't care about sustainability)
- When army cohesion is low, **Reinforcement Proximity** weight rises before any aggressive tactic is scored

This replaces hard-coded conditional logic that would otherwise require extensive manual tuning. The MLP learns these contextual adjustments from game outcome data.

The output is a simple scaling vector applied to the heuristic values before they reach the unit idea layer.

---

## Stage 7 — Unit Idea Generation `[DETERMINISTIC]`

**Input:** Weighted heuristic vector (Stage 6) + active Dream (Stage 4) + current named strategy

**Output:** Per-unit candidate tactic ideas, most suppressed, some emitted for execution

Each unit runs its own local evaluation loop:

1. Observe local context (nearby units, distances, current orders)
2. Filter tactic library to applicable tactics given current unit type and position
3. Score each applicable tactic against the weighted heuristic vector
4. Compare against the current Dream — does this tactic advance the active phase?
5. Apply **action suppression**: only emit an idea if the expected value of acting clearly exceeds the expected value of allowing automation to continue

The default answer is **do nothing**. Built-in SC2 automation (attack-move, patrol paths, queued commands) continues uninterrupted unless a unit's idea crosses the suppression threshold.

Units emit ideas with a priority score. Ideas bubble up to the strategy layer, which evaluates them in the context of the current named strategy — a unit can be overridden if its idea conflicts with the greater strategic purpose, even at the cost of that unit's survival.

---

## Stage 8 — Action Execution `[DETERMINISTIC]`

**Input:** Emitted unit ideas from Stage 7

**Output:** Actual SC2 commands issued via the Ares behavior system

Tactic execution is handled by coded, deterministic Ares `CombatIndividualBehavior` modules. Once a tactic is selected, *how* it executes is not an AI problem — it is a correctly implemented behavior.

Execution patterns include:

- **Direct commands** — attack, move, use ability
- **Patrol paths** — set and forget harassment routes
- **Queued commands** — pre-positioned attacks at anticipated enemy locations (Serral-style Infestor pre-aim)
- **Command groups** — coordinated multi-unit actions issued via `give_same_action`
- **Ares drop system** — Overlord ghetto blink via `PickUpCargo` / `PathUnitToTarget` / `DropCargo`

The Ares behavior system batches and executes these efficiently, avoiding redundant API calls.

---

## Summary Table

| Stage | Type | Runs Every | Key Output |
|---|---|---|---|
| 1 — Heuristics | Deterministic | Frame | ~200 heuristic values + Aggression Dial |
| 2 — Tactic Recognition | Rule-based Ensemble | Frame | Ranked tactic signals with confidence + evidence |
| 3 — Strategy Labeling | Rule-based Classifier | Frame | Named strategy label + confidence score |
| 4 — Dream of Future | Diffusion Model | Periodic / on event | Best projected future state from 8 samples |
| 5 — Opponent Next-Move | Diffusion Model | Periodic / on event | Probability-weighted enemy action prediction |
| 6 — Weight Adjustment | MLP (~1–2MB) | Frame | Contextual heuristic weight vector |
| 7 — Unit Ideas | Deterministic | ~10 frames per unit | Suppressed or emitted tactic ideas |
| 8 — Execution | Deterministic (Ares) | On idea emission | SC2 unit commands |
