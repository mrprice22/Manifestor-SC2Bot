#!/usr/bin/env python3
"""
SC2 Bot Log Analyzer
Reads logs from C:\\Git\\Manifestor-SC2Bot\\logs, detects new sessions,
generates charts, and updates a baseline CSV.

Fully dynamic: new heuristic fields, log levels, and end-game stat rows are
discovered automatically from the logs — no code changes needed.
"""

import os
import re
import csv
import math
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── optional deps (graceful import) ──────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed – charts will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
_HERE              = Path(__file__).parent
DEFAULT_LOG_DIR    = str(_HERE / "logs")
BASELINE_FILE      = str(_HERE / "baseline.csv")
SEEN_SESSIONS_FILE = str(_HERE / "seen_sessions.json")
CHARTS_DIR         = str(_HERE / "charts")

# ── Core log-line structure (timestamp | level | frame | message) ─────────────
LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"   # timestamp
    r"\s*\|\s*(\w+)"                                    # level
    r"\s*\|\s*(\S+)"                                    # frame
    r"\s*\|(.*)"                                        # message
)

# ── Heuristics: parse any sequence of key=value after "Heuristics |" ─────────
# Matches lines like:
#   Heuristics | mom=0.00 army_val=1.77 agg=71 threat=0.20 econ=1.25 phase=0.39
# Any new fields added to the heuristics line are captured automatically.
HEURISTIC_LINE_RE  = re.compile(r"Heuristics\s*\|")
HEURISTIC_KV_RE    = re.compile(r"([\w]+)=([\-\d.]+)")

# ── GAME_END structured line ──────────────────────────────────────────────────
GAME_END_RE = re.compile(r"GAME_END \| result=(\S+) \| final_strategy=(.+)")

# ── End-of-game stats block: capture "  Label (optional parens)  : value" ────
# This regex is intentionally broad so it picks up any label/value pair inside
# the stats block regardless of whether we've seen it before.
STAT_LINE_RE = re.compile(
    r"^\s{1,6}"                         # leading indent (stats lines are indented)
    r"([A-Za-z][A-Za-z0-9 /()%\-]+?)"  # label  (letters, spaces, parens, slash…)
    r"\s*:\s*"                           # colon separator
    r"(.+?)\s*$"                         # value
)

# Lines inside the stats block that are NOT stat rows (decorative / section headers).
# IMPORTANT: "COMPOSITE SCORE" contains a colon and IS a stat row, so it must NOT
# be in this filter.  Only pure decoration lines and section-header-only lines belong here.
STAT_NOISE_RE = re.compile(r"^[\s═─\-=]+$|END-OF-GAME STATS|^\s*(PEAKS|LOSSES|COMBAT|ECONOMY)\s*")

# ── Known section headers that appear in the stats block (no colon) ──────────
# These are skipped by STAT_LINE_RE already (no colon), but listed for clarity.


def _label_to_key(label: str) -> str:
    """Convert a human-readable stat label to a safe CSV column key.
    e.g. 'Largest Army' → 'stat_largest_army'
         'Killed (units)' → 'stat_killed_units'
         'COMPOSITE SCORE' → 'stat_composite_score'
    """
    key = label.lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)   # non-alphanum → underscore
    key = key.strip("_")
    return f"stat_{key}"

# ─────────────────────────────────────────────────────────────────────────────
# FILE GROUPING
# ─────────────────────────────────────────────────────────────────────────────

def group_log_files(log_dir: Path) -> dict[str, list[Path]]:
    """Group log files by session key (basename without .N suffix)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(log_dir.glob("manifestor_*.log*")):
        # base = everything up to first ".log"
        name = f.name
        m = re.match(r"(manifestor_\d{8}_\d{6}\.log)", name)
        if m:
            groups[m.group(1)].append(f)
    # sort files within each group numerically (.log < .log.1 < .log.2 …)
    for key in groups:
        groups[key].sort(key=lambda p: (
            0 if p.suffix == ".log" else int(p.suffix.lstrip(".") or 0)
            if re.match(r"\.\d+$", p.suffix) else 999
        ))
    return dict(groups)


def load_seen_sessions(path: str) -> set:
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_seen_sessions(seen: set, path: str):
    with open(path, "w") as f:
        json.dump(sorted(seen), f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# LOG PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_session(files: list[Path]) -> dict:
    """Parse all files for a session and return structured data.

    Fully dynamic — discovers heuristic fields, end-game stat rows, and log
    levels on the fly.  No code changes needed when new fields are added to
    the bot.
    """
    lines_raw = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                lines_raw.extend(fh.readlines())
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")

    # frame-indexed series
    tactic_by_frame:    dict[int, dict[str, int]] = {}
    heuristics_by_frame: dict[int, dict[str, float]] = {}

    # aggregates
    tactic_totals:  dict[str, int]   = defaultdict(int)
    level_counts:   dict[str, int]   = defaultdict(int)   # e.g. DEBUG/INFO/TACTIC/GAME/…
    game_events:    list[dict]        = []                 # structured GAME-level lines
    end_stats:      dict[str, str]    = {}                 # label_key → value string

    exit_condition: str = "crash_or_incomplete"
    strategy:       str = ""
    final_strategy: str = ""
    in_stats_block: bool = False

    for raw in lines_raw:
        line = raw.rstrip("\n")
        m = LOG_LINE_RE.match(line)

        if not m:
            # Continuation line — only relevant inside the stats block
            if in_stats_block:
                _try_parse_stat_line(line, end_stats)
            continue

        ts_str, level, frame_str, msg = (
            m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
        )

        level_counts[level] += 1

        try:
            frame = int(frame_str)
        except ValueError:
            frame = None

        # ── GAME_START ────────────────────────────────────────────────────────
        gs = re.search(r"GAME_START \| Strategy: (.+)", msg)
        if gs:
            strategy = gs.group(1).strip()
            game_events.append({"type": "GAME_START", "frame": frame,
                                 "strategy": strategy, "ts": ts_str})

        # ── GAME_END ──────────────────────────────────────────────────────────
        ge = GAME_END_RE.search(msg)
        if ge:
            result_raw = ge.group(1)
            final_strategy = ge.group(2).strip()
            game_events.append({"type": "GAME_END", "frame": frame,
                                 "result": result_raw, "ts": ts_str})
            if "Victory" in result_raw:
                exit_condition = "victory"
            elif "Defeat" in result_raw:
                exit_condition = "defeat"
            else:
                exit_condition = f"result_{result_raw}"

        # ── TACTIC ────────────────────────────────────────────────────────────
        if level == "TACTIC" and frame is not None:
            # message format: "TacticName | tag=… conf=…"
            tactic_name = msg.split("|")[0].strip()
            tactic_totals[tactic_name] += 1
            if frame not in tactic_by_frame:
                tactic_by_frame[frame] = defaultdict(int)
            tactic_by_frame[frame][tactic_name] += 1

        # ── HEURISTICS ────────────────────────────────────────────────────────
        # Dynamically extract ALL key=value pairs after "Heuristics |"
        # so new fields are captured with zero code changes.
        if HEURISTIC_LINE_RE.search(msg) and frame is not None:
            kv_part = msg[msg.index("|") + 1:]
            parsed = {
                k: float(v)
                for k, v in HEURISTIC_KV_RE.findall(kv_part)
            }
            if parsed:
                heuristics_by_frame[frame] = parsed

        # ── GAME_STATS block ──────────────────────────────────────────────────
        if "GAME_STATS" in msg:
            in_stats_block = True

        if in_stats_block:
            _try_parse_stat_line(msg, end_stats)
            if "Game finished" in msg:
                in_stats_block = False

    return {
        "strategy":             strategy or final_strategy,
        "exit_condition":       exit_condition,
        "end_stats":            end_stats,          # {stat_label_key: value_str}
        "tactic_totals":        dict(tactic_totals),
        "tactic_by_frame":      tactic_by_frame,
        "heuristics_by_frame":  heuristics_by_frame,
        "level_counts":         dict(level_counts),
        "game_events":          game_events,
    }


def _try_parse_stat_line(line: str, stats: dict):
    """Attempt to extract a 'Label : Value' pair from a stats-block line.

    Uses a broad regex rather than a hardcoded lookup table, so any new stat
    rows are captured automatically.  Noise lines (decorators, section headers)
    are filtered by STAT_NOISE_RE before matching.
    """
    if STAT_NOISE_RE.search(line):
        return
    m = STAT_LINE_RE.match(line)
    if not m:
        return
    label, value = m.group(1).strip(), m.group(2).strip()
    if not label or not value:
        return
    # Strip trailing bracket annotations like "[COLLAPSE]"
    value = re.sub(r"\s*\[.*?\]\s*$", "", value).strip()
    # Remove thousands commas from numeric values
    clean_value = value.replace(",", "")
    key = _label_to_key(label)
    if key not in stats:          # first occurrence wins (handles duplicate labels)
        stats[key] = clean_value


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def make_charts(session_key: str, data: dict, charts_dir: Path):
    if not HAS_MPL:
        return
    charts_dir.mkdir(parents=True, exist_ok=True)
    session_label = session_key.replace(".log", "")

    # ── 1. Tactics over time ─────────────────────────────────────────────────
    tbf = data["tactic_by_frame"]
    if tbf:
        all_tactics = set()
        for counts in tbf.values():
            all_tactics.update(counts.keys())
        all_tactics = sorted(all_tactics)

        frames = sorted(tbf.keys())
        # rolling window: accumulate counts up to each frame (cumulative)
        cum: dict[str, list] = {t: [] for t in all_tactics}
        running = defaultdict(int)
        for fr in frames:
            for t, c in tbf[fr].items():
                running[t] += c
            for t in all_tactics:
                cum[t].append(running[t])

        fig, ax = plt.subplots(figsize=(14, 6))
        for t in all_tactics:
            ax.plot(frames, cum[t], label=t)
        ax.set_title(f"Cumulative Tactic Usage Over Time\n{session_label}")
        ax.set_xlabel("Game Frame")
        ax.set_ylabel("Cumulative Count")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        out = charts_dir / f"{session_label}_tactics.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Chart: {out}")

    # ── 2. Tactic frequency bar chart ────────────────────────────────────────
    totals = data["tactic_totals"]
    if totals:
        names = sorted(totals, key=totals.get, reverse=True)
        values = [totals[n] for n in names]
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 5))
        bars = ax.bar(range(len(names)), values, color="steelblue")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Total Tactic Usage\n{session_label}")
        ax.set_ylabel("Count")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(v), ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        out = charts_dir / f"{session_label}_tactic_totals.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Chart: {out}")

    # ── 3. Heuristics over time (fully dynamic: any fields, any count) ──────────
    hbf = data["heuristics_by_frame"]
    if hbf:
        frames = sorted(hbf.keys())
        # Discover all metric names seen across all frames, preserve insertion order
        metrics_seen: dict[str, None] = {}
        for fd in hbf.values():
            metrics_seen.update({k: None for k in fd})
        metrics = list(metrics_seen)

        n = len(metrics)
        ncols = 2
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(14, max(4, nrows * 3)),
                                 sharex=True)
        # Flatten and handle the case where subplots returns a 1-D array or scalar
        if n == 1:
            axes = [axes]
        else:
            axes = list(getattr(axes, "flatten", lambda: axes)())

        cmap = plt.get_cmap("tab10")
        for i, metric in enumerate(metrics):
            values = [hbf[fr].get(metric, float("nan")) for fr in frames]
            axes[i].plot(frames, values, color=cmap(i % 10), linewidth=1.2)
            axes[i].set_title(metric, fontsize=10)
            axes[i].set_ylabel(metric)
            axes[i].xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
            )

        # Hide any unused subplot panels
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        for ax in axes[:n]:
            ax.set_xlabel("Game Frame")

        fig.suptitle(f"Heuristics Over Time\n{session_label}", fontsize=12)
        fig.tight_layout()
        out = charts_dir / f"{session_label}_heuristics.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Chart: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE CSV
# ─────────────────────────────────────────────────────────────────────────────

# Fixed prefix columns that always appear first for readability.
# Everything else (stat_* keys, tactic_totals, level_counts) is appended
# dynamically in the order it is first encountered.
BASELINE_PREFIX_COLS = [
    "session_key", "datetime", "strategy", "exit_condition",
]


def load_baseline(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_baseline(rows: list[dict], path: str):
    """Write baseline CSV, deriving column order from actual data.

    Prefix columns come first; all other columns are appended in the order
    they are first seen across all rows — so new stat fields or heuristic
    metrics appear automatically without any code changes.
    """
    ordered: list[str] = list(BASELINE_PREFIX_COLS)
    seen_cols: set[str] = set(ordered)
    for row in rows:
        for col in row:
            if col not in seen_cols:
                ordered.append(col)
                seen_cols.add(col)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def session_to_row(session_key: str, data: dict) -> dict:
    """Build a flat CSV row from parsed session data.

    All end_stats keys (e.g. stat_result, stat_duration, stat_largest_army…)
    are written directly — whatever the parser discovered is what goes in.
    """
    m = re.search(r"manifestor_(\d{8})_(\d{6})", session_key)
    dt_str = ""
    if m:
        try:
            dt_str = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").isoformat()
        except ValueError:
            pass

    row: dict = {
        "session_key":    session_key,
        "datetime":       dt_str,
        "strategy":       data["strategy"],
        "exit_condition": data["exit_condition"],
    }

    # All discovered end-of-game stat fields (stat_* keys)
    row.update(data["end_stats"])

    # Tactic totals as JSON blob (preserves full detail; easy to query later)
    row["tactic_totals"] = json.dumps(data["tactic_totals"])

    # Log-level event counts (e.g. level_count_DEBUG, level_count_TACTIC, …)
    for lvl, cnt in data.get("level_counts", {}).items():
        row[f"level_count_{lvl}"] = cnt

    return row


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SC2 Bot Log Analyzer")
    parser.add_argument("--log-dir",    default=DEFAULT_LOG_DIR,      help="Path to log directory")
    parser.add_argument("--baseline",   default=BASELINE_FILE,        help="Baseline CSV path")
    parser.add_argument("--seen",       default=SEEN_SESSIONS_FILE,   help="Seen-sessions JSON path")
    parser.add_argument("--charts-dir", default=CHARTS_DIR,           help="Output directory for charts")
    parser.add_argument("--force-all",  action="store_true",          help="Re-process all sessions (ignore seen)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"[ERROR] Log directory not found: {log_dir}")
        return

    charts_dir = Path(args.charts_dir)
    seen = set() if args.force_all else load_seen_sessions(args.seen)
    baseline_rows = load_baseline(args.baseline)
    existing_keys = {r["session_key"] for r in baseline_rows}

    groups = group_log_files(log_dir)
    new_sessions = [k for k in groups if k not in seen]

    if not new_sessions:
        print("No new sessions to process.")
        return

    print(f"Found {len(new_sessions)} new session(s) to process.")

    for session_key in sorted(new_sessions):
        print(f"\n── Processing: {session_key} ──")
        files = groups[session_key]
        print(f"  Files: {[f.name for f in files]}")

        data = parse_session(files)

        print(f"  Exit condition : {data['exit_condition']}")
        print(f"  Strategy       : {data['strategy']}")
        print(f"  Tactic types   : {len(data['tactic_totals'])}")
        print(f"  Heuristic pts  : {len(data['heuristics_by_frame'])}")
        print(f"  Stat fields    : {sorted(data['end_stats'].keys())}")
        print(f"  Log levels     : {dict(data['level_counts'])}")

        make_charts(session_key, data, charts_dir)

        row = session_to_row(session_key, data)
        if session_key in existing_keys:
            baseline_rows = [r for r in baseline_rows if r["session_key"] != session_key]
        baseline_rows.append(row)
        seen.add(session_key)

    save_baseline(baseline_rows, args.baseline)
    save_seen_sessions(seen, args.seen)

    print(f"\n✓ Baseline updated → {args.baseline}  ({len(baseline_rows)} total rows)")
    print(f"✓ Seen sessions  → {args.seen}")
    if HAS_MPL:
        print(f"✓ Charts saved   → {charts_dir}/")


if __name__ == "__main__":
    main()
