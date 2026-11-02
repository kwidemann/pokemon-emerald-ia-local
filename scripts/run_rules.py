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
from env.milestones import TRAMO_1, Tracker  # noqa: E402
from env.state import flag_diff, read_flags  # noqa: E402
from env.world import World, load_rom  # noqa: E402
from scripts.drive import Driver  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--savestate", default="runs/lab_mudkip.ss0")
    parser.add_argument("--goals", nargs="+", default=["0,10"], help="mapas destino 'grupo,número', en orden")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--no-load", action="store_true", help="seguir desde el estado actual de mGBA")
    parser.add_argument("--iterations", type=int, default=600)
    args = parser.parse_args()

    world = World(load_rom())
    with MGBAClient(port=args.port) as emu:
        d = Driver(emu)
        if not args.no_load:
            emu.load_state(Path(args.savestate).resolve().as_posix())
            emu.step(0, 2)
        tracker, achieved = Tracker(), set()

        def on_state(state) -> None:
            tracker.update(state)
            for ms in TRAMO_1:
                if ms.id not in achieved and ms.check is not None and ms.check(state, tracker):
                    achieved.add(ms.id)
                    print(f"  HITO {ms.id}: {ms.name} (frame {state.frame})", flush=True)

        agent = RulesAgent(d, world, on_state=on_state)
        flags0 = read_flags(emu)
        for goal in args.goals:
            print(f"-> {goal}", flush=True)
            before = read_flags(emu)
            if goal.startswith("talk:"):  # talk:grupo,número:gráfico_del_NPC
                _, map_txt, gfx = goal.split(":")
                ok = agent.talk_to(tuple(int(v) for v in map_txt.split(",")), int(gfx))
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
        s = d.state()
        print(
            f"fin: mapa={(s.map_group, s.map_num)} pos=({s.x},{s.y}) batallas={agent.stats.battles} "
            f"movimientos={agent.stats.moves} diálogo={agent.stats.dialogue_presses} salidas_bloqueadas={agent.stats.blocked_exits} equipo={[(m.species, m.level, m.hp, m.max_hp) for m in s.party]}"
        )


if __name__ == "__main__":
    main()
