"""Corre el agente de reglas desde un savestate y va marcando hitos.

Uso:
    uv run python scripts/run_rules.py --savestate runs/lab_mudkip.ss0 --goals 0,10 0,18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.rules.agent import RulesAgent  # noqa: E402
from emulator.client import MGBAClient  # noqa: E402
from env.milestones import TRAMO_1, MilestoneMonitor  # noqa: E402
from env.state import flag_diff, read_flags  # noqa: E402
from env.world import World, load_rom  # noqa: E402
from scripts.drive import Driver  # noqa: E402


# Guion del Tramo 1 (specs.md, 9.1): cada paso sigue el orden de la historia de Esmeralda.
TRAMO1_GOALS = [
    "0,10",  # Oldale (hito 6)
    "grind:0,18:8",  # algo de nivel antes del rival
    "talk:0,18:240:0.9",  # rival en la Ruta 103 (hito 7)
    "1,4",  # Pokédex en el laboratorio (hito 8)
    "0,0",  # Petalburg (hito 9)
    "talk:8,1:129",  # Norman y Wally (hito 10)
    "0,3",  # Bosque Petalburgo (hito 11) y Rustboro (hito 12)
    "catch:0,19:2:11,7",  # compra Poké Balls en la tienda de Rustboro y captura el segundo Pokémon (hito 13)
    "talk:11,3:125:0.95",  # gimnasio y Roxanne (hitos 14, 15, 16)
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--savestate", default="runs/lab_mudkip.ss0")
    parser.add_argument("--goals", nargs="+", default=["0,10"], help="mapas destino 'grupo,número', en orden")
    parser.add_argument("--tramo1", action="store_true", help="el guion completo del Tramo 1 hasta la medalla 1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--no-load", action="store_true", help="seguir desde el estado actual de mGBA")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--save-at-end", help="guardar un savestate al terminar (ruta)")
    args = parser.parse_args()
    if args.tramo1:
        args.goals = TRAMO1_GOALS

    world = World(load_rom())
    with MGBAClient(port=args.port) as emu:
        d = Driver(emu)
        if not args.no_load:
            emu.load_state(Path(args.savestate).resolve().as_posix())
            emu.step(0, 2)
        monitor = MilestoneMonitor()
        achieved: set[int] = set()
        first_frame: dict[int, int] = {}
        start_frame = emu.frame()

        def on_state(state) -> None:
            for ms in monitor.update(state):
                achieved.add(ms.id)
                first_frame[ms.id] = state.frame - start_frame
                print(f"  HITO {ms.id}: {ms.name} (frame {state.frame})", flush=True)

        agent = RulesAgent(d, world, on_state=on_state)
        flags0 = read_flags(emu)
        for goal in args.goals:
            print(f"-> {goal}", flush=True)
            before = read_flags(emu)
            if goal.startswith("talk:"):  # talk:grupo,número:gráfico_del_NPC
                parts = goal.split(":")  # talk:grupo,número:gráfico[:hp_mínimo]
                ok = agent.talk_to(
                    tuple(int(v) for v in parts[1].split(",")), int(parts[2]), min_hp=float(parts[3]) if len(parts) > 3 else 0.0
                )
            elif goal.startswith("catch:"):  # catch:grupo,número:tamaño_del_equipo[:tienda grupo,número]
                _, map_txt, size, *mart = goal.split(":")
                ok = agent.catch(
                    tuple(int(v) for v in map_txt.split(",")),
                    int(size),
                    mart=tuple(int(v) for v in mart[0].split(",")) if mart else None,
                )
            elif goal.startswith("grind:"):  # grind:grupo,número:nivel
                _, map_txt, lvl = goal.split(":")
                ok = agent.grind(tuple(int(v) for v in map_txt.split(",")), int(lvl))
            else:
                ok = agent.goto(tuple(int(v) for v in goal.split(",")), args.iterations)
            on, off = flag_diff(before, read_flags(emu))
            if on or off:
                print(f"  banderas: +{[hex(f) for f in on]} -{[hex(f) for f in off]}", flush=True)
            if not ok:
                break
        total = emu.frame() - start_frame
        print(f"hitos: {sorted(achieved)} de {[m.id for m in TRAMO_1]}; frames por hito: {dict(sorted(first_frame.items()))}; total {total} frames ({total / 3600:.1f} min de juego)")
        if args.save_at_end:
            emu.save_state(Path(args.save_at_end).resolve().as_posix())
            print(f"savestate guardado en {args.save_at_end}")
        s = d.state()
        print(
            f"fin: mapa={(s.map_group, s.map_num)} pos=({s.x},{s.y}) batallas={agent.stats.battles} "
            f"movimientos={agent.stats.moves} diálogo={agent.stats.dialogue_presses} salidas_bloqueadas={agent.stats.blocked_exits} equipo={[(m.species, m.level, m.hp, m.max_hp) for m in s.party]}"
        )


if __name__ == "__main__":
    main()
