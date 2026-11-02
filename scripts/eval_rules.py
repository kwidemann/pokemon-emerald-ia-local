"""Evalúa el agente de reglas: N ejecuciones del guion del Tramo 1 desde el mismo savestate.

Cada ejecución carga el savestate y corre `TRAMO1_GOALS`; se anotan los hitos alcanzados (frame de juego de cada
uno), si llegó a la medalla 1, el tiempo y el motivo del fallo si lo hubo. El emulador sigue corriendo entre comandos,
así que las ejecuciones difieren (RNG del juego, combates, encuentros).

Uso:
    uv run python scripts/eval_rules.py --runs 10 [--savestate runs/lab_mudkip.ss0] [--max-minutes 12]

Escribe `runs/eval/rules_<fecha>.json` (ignorado por git) y imprime una tabla.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.rules.agent import RulesAgent  # noqa: E402
from emulator.client import MGBAClient  # noqa: E402
from env.milestones import TRAMO_1, MilestoneMonitor  # noqa: E402
from env.world import World, load_rom  # noqa: E402
from scripts.drive import Driver  # noqa: E402
from scripts.run_rules import TRAMO1_GOALS  # noqa: E402

FINAL_MILESTONE = 16


class OutOfTime(RuntimeError):
    pass


def run_goal(agent: RulesAgent, goal: str) -> bool:
    if goal.startswith("talk:"):
        parts = goal.split(":")
        return agent.talk_to(tuple(int(v) for v in parts[1].split(",")), int(parts[2]), min_hp=float(parts[3]) if len(parts) > 3 else 0.0)
    if goal.startswith("catch:"):
        _, m, size, *mart = goal.split(":")
        return agent.catch(
            tuple(int(v) for v in m.split(",")), int(size), mart=tuple(int(v) for v in mart[0].split(",")) if mart else None
        )
    if goal.startswith("grind:"):
        _, m, lvl = goal.split(":")
        return agent.grind(tuple(int(v) for v in m.split(",")), int(lvl))
    return agent.goto(tuple(int(v) for v in goal.split(",")))


def one_run(emu: MGBAClient, world: World, savestate: Path, max_seconds: float) -> dict:
    d = Driver(emu)
    emu.load_state(savestate.as_posix())
    emu.step(0, 2)
    start_frame, t0 = emu.frame(), time.time()
    monitor = MilestoneMonitor()
    first: dict[int, int] = {}

    def on_state(state) -> None:
        for ms in monitor.update(state):
            first[ms.id] = state.frame - start_frame
        if time.time() - t0 > max_seconds:
            raise OutOfTime(f"más de {max_seconds / 60:.0f} min de reloj")

    agent = RulesAgent(d, world, verbose=False, on_state=on_state)
    error, failed_goal = None, None
    try:
        for goal in TRAMO1_GOALS:
            if not run_goal(agent, goal):
                failed_goal = goal
                break
            if FINAL_MILESTONE in first:
                break
    except Exception as exc:  # noqa: BLE001 - se registra el motivo y se sigue con la siguiente ejecución
        error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, (OutOfTime, RuntimeError, TimeoutError)):
            error += "\n" + traceback.format_exc(limit=3)
    s = d.state()
    return {
        "success": FINAL_MILESTONE in first,
        "milestones": first,
        "frames": emu.frame() - start_frame,
        "seconds": round(time.time() - t0, 1),
        "battles": agent.stats.battles,
        "failed_goal": failed_goal,
        "error": error,
        "final": {
            "map": [s.map_group, s.map_num],
            "party": [(m.species, m.level, m.hp, m.max_hp) for m in s.party],
            "in_battle": s.in_battle,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--savestate", default="runs/lab_mudkip.ss0")
    parser.add_argument("--max-minutes", type=float, default=12.0, help="tope de reloj por ejecución")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    world = World(load_rom())
    savestate = Path(args.savestate).resolve()
    out_dir = Path("runs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"rules_{datetime.now():%Y%m%d_%H%M%S}.json"

    results = []
    with MGBAClient(port=args.port) as emu:
        for i in range(args.runs):
            r = one_run(emu, world, savestate, args.max_minutes * 60)
            results.append(r)
            ms = sorted(r["milestones"])
            print(
                f"[{i + 1}/{args.runs}] {'ÉXITO' if r['success'] else 'FALLO '} {r['seconds']:>6.1f}s {r['frames']:>7} frames "
                f"hitos={ms} batallas={r['battles']}" + (f" meta_fallida={r['failed_goal']}" if r["failed_goal"] else "")
                + (f" error={r['error'].splitlines()[0]}" if r["error"] else ""),
                flush=True,
            )
            out.write_text(json.dumps({"savestate": str(savestate), "results": results}, indent=1), encoding="utf-8")

    ok = [r for r in results if r["success"]]
    print(f"\nÉxito: {len(ok)}/{len(results)} ({100 * len(ok) / len(results):.0f} %)")
    if ok:
        frames = [r["frames"] for r in ok]
        secs = [r["seconds"] for r in ok]
        print(f"Frames hasta la medalla (éxitos): mediana {statistics.median(frames):.0f}, mín {min(frames)}, máx {max(frames)}")
        print(f"Reloj (éxitos): mediana {statistics.median(secs):.0f} s")
    print("Alcance por hito (ejecuciones que lo lograron):")
    for ms in TRAMO_1:
        n = sum(1 for r in results if ms.id in r["milestones"])
        print(f"  {ms.id:>2} {ms.name:<45} {n}/{len(results)}")
    print(f"Resultados en {out}")


if __name__ == "__main__":
    main()
