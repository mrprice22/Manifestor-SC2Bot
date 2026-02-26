"""
Run script for Manifestor Bot using config.py settings.
"""

import os
import random
from pathlib import Path
import sys
sys.path.append('ares-sc2/src/ares')
sys.path.append('ares-sc2/src')
sys.path.append('ares-sc2')
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty, AIBuild
from ManifestorBot.manifestor_bot import ManifestorBot
from ManifestorBot.logger import get_logger
from config import BOT_NAME, BOT_RACE, MAP_POOL, MAP_PATH, OPPONENT_RACE, OPPONENT_DIFFICULTY, REALTIME, RUN_LOG_ANALYZER

log = get_logger()


def main():
    """Run a single game for testing"""

    log.info("=" * 50)
    log.info("%s (%s)", BOT_NAME, BOT_RACE)
    log.info("=" * 50)
    
    # Convert string race to enum
    try:
        bot_race = Race[BOT_RACE.capitalize()]
    except KeyError:
        log.warning("Invalid bot race: %s. Using Zerg.", BOT_RACE)
        bot_race = Race.Zerg
        
    try:
        opponent_race = Race[OPPONENT_RACE.capitalize()]
    except KeyError:
        log.warning("Invalid opponent race: %s. Using Zerg.", OPPONENT_RACE)
        opponent_race = Race.Zerg
        
    try:
        difficulty = Difficulty[OPPONENT_DIFFICULTY]
    except KeyError:
        log.warning("Invalid difficulty: %s. Using Hard.", OPPONENT_DIFFICULTY)
        difficulty = Difficulty.Hard
    
    # Select random map
    map_name = random.choice(MAP_POOL)
    log.info("Map: %s", map_name)
    log.info("Opponent: %s %s", OPPONENT_RACE, OPPONENT_DIFFICULTY)
    log.info("Realtime: %s", REALTIME)
    
    # Load map
    if MAP_PATH and os.path.exists(MAP_PATH):
        try:
            log.info("Loading map from: %s", MAP_PATH)
            map_file = Path(MAP_PATH) / f"{map_name}.SC2Map"
            if map_file.exists():
                map_obj = maps.Map(map_file)
            else:
                log.warning("Map not found: %s — falling back to default SC2 maps", map_file)
                map_obj = maps.get(map_name)
        except Exception as e:
            log.error("Error loading custom map: %s — falling back to default SC2 maps", e)
            map_obj = maps.get(map_name)
    else:
        map_obj = maps.get(map_name)
    
    log.info("Starting game on %s ...", map_name)
    run_game(
        map_obj,
        [
            Bot(bot_race, ManifestorBot()),
            Computer(opponent_race, difficulty, AIBuild.Rush)
        ],
        realtime=REALTIME,
        save_replay_as="manifestor_test.SC2Replay"
    )
    
    log.info("Game finished!")

    if RUN_LOG_ANALYZER:
        import subprocess
        analyzer = Path(__file__).parent / "sc2_log_analyzer.py"
        log.info("Running post-game log analyzer...")
        result = subprocess.run(
            [sys.executable, str(analyzer)],
            cwd=str(Path(__file__).parent),
        )
        if result.returncode != 0:
            log.warning("Log analyzer exited with code %d", result.returncode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Game stopped by user")
    except Exception as e:
        log.exception("Unexpected error in main: %s", e)
