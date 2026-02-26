# ===== BOT SETTINGS =====
# Your bot's name and race (use plain strings)
BOT_NAME = "James' Noob Bot"
BOT_RACE = "Zerg"  # Options: Terran, Protoss, Zerg, Random

# ===== GAME SETTINGS =====
# Maps configuration
# Set to None to use default SC2 maps, or specify the full path to the Maps directory
# Examples (include the Maps folder in the path):
#   MAP_PATH = "C:/Program Files (x86)/StarCraft II/Maps"  # Standard Windows path
#   MAP_PATH = "/Applications/StarCraft II/Maps"          # Mac
#   MAP_PATH = "~/StarCraftII/Maps"                      # Linux
# Specifying the full path to Maps directory helps avoid case sensitivity issues
MAP_PATH = "C:/Program Files (x86)/StarCraft II/Maps"  # Default Windows path - modify as needed

# List of maps to play on (randomly selected if not specified)
MAP_POOL = [
    "PersephoneAIE_v4",
    "PylonAIE_v4",
    "TorchesAIE_v4"
]

# ===== OPPONENT SETTINGS =====
# Computer opponent settings (for local games)
OPPONENT_RACE = "Zerg"  # Terran, Zerg, Protoss, Random
OPPONENT_DIFFICULTY = "VeryHard"  # VeryEasy, Easy, Medium, Hard, VeryHard, etc.

# ===== GAME MODE =====
# Set to True to play in realtime (like a human), False for faster simulation
REALTIME = False

# ===== STRATEGY MACHINE =====
# Set to None to let the state machine run normally (recommended for live games).
# Set to a Strategy name string to lock the bot to that strategy for the entire
# game — useful for testing a specific strategy at different game stages.
#
# Valid values (use the exact string):
#   None                   — normal state machine (default)
#   "STOCK_STANDARD"       — textbook macro Zerg
#   "JUST_GO_PUNCH_EM"     — press army advantage
#   "ALL_IN"               — all-in overwhelm
#   "KEEP_EM_BUSY"         — initiative harassment
#   "WAR_ON_SANITY"        — multi-front economic chaos
#   "WAR_OF_ATTRITION"     — hold-and-grind defensive
#   "BLEED_OUT"            — guerrilla/harassment pivot
#   "DRONE_ONLY_FORTRESS"  — emergency turtle
FORCE_STRATEGY = "STOCK_STANDARD"

# ===== POST-GAME LOG ANALYSIS =====
# When True, sc2_log_analyzer.py runs automatically after every game.
# It parses the latest session's logs, generates charts (requires matplotlib),
# and appends a row to baseline.csv so you can track performance over time.
# Output files land in the project root (baseline.csv, seen_sessions.json, charts/).
# Set to False to skip analysis (e.g. when running ladder games or batch tests).
RUN_LOG_ANALYZER = True
