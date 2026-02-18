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
from config import BOT_NAME, BOT_RACE, MAP_POOL, MAP_PATH, OPPONENT_RACE, OPPONENT_DIFFICULTY, REALTIME


def main():
    """Run a single game for testing"""
    
    print("=" * 50)
    print(f"{BOT_NAME} ({BOT_RACE})")
    print("=" * 50)
    
    # Convert string race to enum
    try:
        bot_race = Race[BOT_RACE.capitalize()]
    except KeyError:
        print(f"Invalid bot race: {BOT_RACE}. Using Zerg.")
        bot_race = Race.Zerg
        
    try:
        opponent_race = Race[OPPONENT_RACE.capitalize()]
    except KeyError:
        print(f"Invalid opponent race: {OPPONENT_RACE}. Using Zerg.")
        opponent_race = Race.Zerg
        
    try:
        difficulty = Difficulty[OPPONENT_DIFFICULTY]
    except KeyError:
        print(f"Invalid difficulty: {OPPONENT_DIFFICULTY}. Using Hard.")
        difficulty = Difficulty.Hard
    
    # Select random map
    map_name = random.choice(MAP_POOL)
    print(f"Map: {map_name}")
    print(f"Opponent: {OPPONENT_RACE} {OPPONENT_DIFFICULTY} Rush")
    print(f"Realtime: {REALTIME}")
    print()
    
    # Load map
    if MAP_PATH and os.path.exists(MAP_PATH):
        try:
            print(f"Loading map from: {MAP_PATH}")
            map_file = Path(MAP_PATH) / f"{map_name}.SC2Map"
            if map_file.exists():
                map_obj = maps.Map(map_file)
            else:
                print(f"Map not found: {map_file}")
                print("Falling back to default SC2 maps...")
                map_obj = maps.get(map_name)
        except Exception as e:
            print(f"Error loading custom map: {e}")
            print("Falling back to default SC2 maps...")
            map_obj = maps.get(map_name)
    else:
        map_obj = maps.get(map_name)
    
    # Run the game
    print(f"\nStarting game on {map_name}...\n")
    run_game(
        map_obj,
        [
            Bot(bot_race, ManifestorBot()),
            Computer(opponent_race, difficulty, AIBuild.Rush)
        ],
        realtime=REALTIME,
        save_replay_as="manifestor_test.SC2Replay"
    )
    
    print("\nGame finished!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame stopped by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()