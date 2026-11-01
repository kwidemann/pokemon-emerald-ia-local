"""Agente aleatorio: prueba el entorno y mide la línea base.

Uso:
    uv run python scripts/random_agent.py --savestate C:/AI/pokemon-ia-states/lab_mudkip.ss0 --steps 300 --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.pokemon_env import ACTION_NAMES, PokemonEmeraldEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--savestate", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--new-tile-bonus", type=float, default=0.0)
    args = parser.parse_args()

    env = PokemonEmeraldEnv(args.savestate, port=args.port, max_steps=args.steps, new_tile_bonus=args.new_tile_bonus)
    obs, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    print(f"inicio: mapa=({info['state']['map_group']},{info['state']['map_num']}) hitos={info['milestones']}")

    total, t0 = 0.0, time.time()
    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if reward:
            print(f"paso {info['steps']}: +{reward} con {ACTION_NAMES[action]} -> hitos={info['milestones']}")
        if terminated or truncated:
            break
    dt = time.time() - t0
    s = info["state"]
    print(
        f"fin: {info['steps']} pasos en {dt:.1f}s ({info['steps'] / dt:.1f} pasos/s), recompensa={total}, "
        f"mapa=({s['map_group']},{s['map_num']}) pos=({s['x']},{s['y']}), hitos={info['milestones']}"
    )
    env.close()


if __name__ == "__main__":
    main()
