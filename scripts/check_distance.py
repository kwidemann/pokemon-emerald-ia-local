"""Comprueba en vivo el campo de distancias de `env/distance.py` (R3).

Para cada savestate de partida: carga el estado en un mGBA ya abierto, muestra la distancia a la meta y camina por el mejor
camino del grafo (siempre al vecino con menor distancia), comprobando que cada pulsación mueve al personaje y que la
distancia baja en cada paso. Si el grafo piensa que se puede pasar por un sitio que el juego bloquea, se ve aquí.

    uv run python scripts/check_distance.py --port 8888 runs/lab_mudkip.ss0 runs/ruta101_sur.ss0 runs/ruta101_norte.ss0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator.client import Key, MGBAClient  # noqa: E402
from env.distance import distance_field  # noqa: E402
from env.milestones import OLDALE  # noqa: E402
from env.state import read_state  # noqa: E402

KEYS = {"up": (Key.UP, (0, -1)), "down": (Key.DOWN, (0, 1)), "left": (Key.LEFT, (-1, 0)), "right": (Key.RIGHT, (1, 0))}


def walk(emu: MGBAClient, field: dict, max_steps: int) -> None:
    s = read_state(emu)
    pos = (s.map_group, s.map_num, s.x, s.y)
    print(f"  inicio {pos}: distancia {field.get(pos)}")
    stuck = 0
    for i in range(max_steps):
        s = read_state(emu)
        pos = (s.map_group, s.map_num, s.x, s.y)
        d = field.get(pos)
        if d is None:
            print(f"  paso {i}: {pos} fuera del campo")
            return
        if d == 0:
            print(f"  paso {i}: llegó a la meta {pos}")
            return
        # candidatos: vecinos en el mismo mapa con menor distancia (los saltos y warps no se prueban aquí)
        best = None
        for name, (key, (dx, dy)) in KEYS.items():
            nd = field.get((*pos[:2], pos[2] + dx, pos[3] + dy))
            if nd is not None and nd < d and (best is None or nd < best[1]):
                best = (name, nd)
        if best is None:
            print(f"  paso {i}: {pos} d={d}: sin vecino mejor en el mapa (warp o conexión o salto): se prueba cada tecla")
            for name, (key, _) in KEYS.items():
                emu.step(key, 4)
                emu.step(0, 16)
                t = read_state(emu)
                np_ = (t.map_group, t.map_num, t.x, t.y)
                if np_ != pos and field.get(np_, 10**9) < d:
                    break
            else:
                print("    ninguna tecla acerca: el grafo y el juego discrepan aquí")
                return
            continue
        name, nd = best
        key, _ = KEYS[name]
        emu.step(key, 4)
        emu.step(0, 16)
        # el primer toque puede solo girar: se repite una vez
        t = read_state(emu)
        if (t.map_group, t.map_num, t.x, t.y) == pos:
            emu.step(key, 4)
            emu.step(0, 16)
            t = read_state(emu)
            if (t.map_group, t.map_num, t.x, t.y) == pos:
                stuck += 1
                print(
                    f"  paso {i}: {pos} d={d}: el juego no deja ir a {name} (el grafo dice d={nd}); "
                    f"combate={t.in_battle} diálogo={t.script_active} en_mundo={t.in_overworld}"
                )
                if stuck >= 3:
                    return
    print("  se acabaron los pasos")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("savestates", nargs="+")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--max-steps", type=int, default=150)
    args = p.parse_args()
    emu = MGBAClient("127.0.0.1", args.port)
    field = distance_field(OLDALE)
    for ss in args.savestates:
        print(ss)
        emu.load_state(Path(ss).resolve().as_posix())
        emu.step(0, 2)
        walk(emu, field, args.max_steps)


if __name__ == "__main__":
    main()
