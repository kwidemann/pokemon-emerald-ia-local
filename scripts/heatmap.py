"""Mapa de calor de las casillas que pisa el agente mientras entrena.

`watch` abre un cliente de solo lectura por cada instancia de mGBA (el servidor admite varios clientes, y solo el que
lanzó un `step` espera respuesta), lee la posición cada `--interval` segundos y cuenta las visitas por mapa y casilla.
Cuenta tiempo de reloj, no pasos de PPO. Solo cuenta si se controla al personaje en el mundo (no en batalla ni
menús) y la lectura es válida. `render` dibuja un PNG por mapa sobre los gráficos reales del juego (`env/maprender.py`, leídos de la ROM).

Uso:
    uv run python scripts/heatmap.py watch --ports 8888 8889 8890 --name ppo_500k --minutes 150
    uv run python scripts/heatmap.py render --name ppo_500k [--port 8890]

Salidas en `runs/heatmaps/<name>.json` y `runs/heatmaps/<name>/map_<grupo>_<número>.png`, o `<name>_puerto<N>/` con `--port` (ignorado por git).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emulator.client import MGBAClient  # noqa: E402
from env.state import read_state  # noqa: E402

OUT = ROOT / "runs" / "heatmaps"


def watch(args) -> None:
    out = OUT / f"{args.name}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    if out.exists() and not args.fresh:  # seguir sumando sobre lo ya guardado
        for port, maps in json.loads(out.read_text())["counts"].items():
            for m, tiles in maps.items():
                for t, n in tiles.items():
                    counts[port][m][t] = n
    clients: dict[int, MGBAClient | None] = {p: None for p in args.ports}
    deadline = time.time() + args.minutes * 60
    last_save, failures, samples = time.time(), 0, 0

    def save() -> None:
        out.write_text(json.dumps({"interval": args.interval, "counts": counts}))

    print(f"muestreando {args.ports} cada {args.interval} s hasta {args.minutes} min; salida {out}", flush=True)
    try:
        while time.time() < deadline and failures < 20:
            ok_any = False
            for port in args.ports:
                try:
                    if clients[port] is None:
                        clients[port] = MGBAClient(port=port, timeout=5)
                    s = read_state(clients[port])
                except (OSError, ValueError):
                    clients[port] = None
                    continue
                ok_any = True
                if s.in_transition or not s.in_overworld or s.in_battle:
                    continue
                counts[str(port)][f"{s.map_group},{s.map_num}"][f"{s.x},{s.y}"] += 1
                samples += 1
            failures = 0 if ok_any else failures + 1
            if time.time() - last_save > 30:
                save()
                last_save = time.time()
                print(f"  {samples} muestras", flush=True)
            time.sleep(args.interval)
    finally:
        save()
        print(f"guardado {out} ({samples} muestras)", flush=True)


def heat_color(t: float) -> tuple[int, int, int]:
    """t en [0, 1] -> azul oscuro, morado, naranja, amarillo."""
    stops = [(0.0, (30, 20, 90)), (0.35, (150, 40, 130)), (0.7, (245, 120, 30)), (1.0, (255, 235, 90))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if t <= b:
            k = (t - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * k) for i in range(3))
    return stops[-1][1]


def render(args) -> None:
    from PIL import Image, ImageDraw

    from env.maprender import MapRenderer
    from env.world import World, load_rom

    data = json.loads((OUT / f"{args.name}.json").read_text())["counts"]
    ports = [str(args.port)] if args.port else list(data)
    merged: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for port in ports:
        for m, tiles in data.get(port, {}).items():
            for t, n in tiles.items():
                merged[m][t] += n
    if not merged:
        sys.exit("no hay muestras (¿puerto o nombre incorrectos?)")
    renderer = MapRenderer(World(load_rom()))
    folder = OUT / (f"{args.name}_puerto{args.port}" if args.port else args.name)
    folder.mkdir(parents=True, exist_ok=True)
    for m, tiles in merged.items():
        g, n = (int(v) for v in m.split(","))
        if (g, n) not in renderer.world.maps:
            continue  # lectura espuria (p. ej. (-1,-1) durante una transición)
        base = renderer.render((g, n)).convert("RGBA")
        scale = max(1, min(3, 1400 // base.width))
        base = base.resize((base.width * scale, base.height * scale), Image.NEAREST)
        base = Image.blend(base, Image.new("RGBA", base.size, (0, 0, 0, 255)), 0.35)  # oscurece para que resalte el calor
        tile = 16 * scale
        heat = Image.new("RGBA", base.size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(heat)
        top = max(tiles.values())
        for t, c in tiles.items():
            x, y = (int(v) for v in t.split(","))
            k = math.log1p(c) / math.log1p(top)
            hd.rectangle([x * tile, y * tile, (x + 1) * tile - 1, (y + 1) * tile - 1], fill=(*heat_color(k), int(120 + 120 * k)))
        img = Image.alpha_composite(base, heat)
        bar_h = 22
        out = Image.new("RGB", (img.width, img.height + bar_h), (20, 20, 20))
        out.paste(img.convert("RGB"), (0, 0))
        d = ImageDraw.Draw(out)
        for i in range(img.width // 2):  # barra de color: poco -> mucho
            d.line([(8 + i, img.height + 4), (8 + i, img.height + 12)], fill=heat_color(i / max(1, img.width // 2 - 1)))
        d.text((img.width // 2 + 16, img.height + 4), f"mapa ({g},{n}) | {sum(tiles.values())} muestras | maximo {top} en una casilla", fill=(230, 230, 230))
        path = folder / f"map_{g}_{n}.png"
        out.save(path)
        print(f"{path.relative_to(ROOT)}: mapa ({g},{n}), {sum(tiles.values())} muestras, máximo {top} en una casilla")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--ports", type=int, nargs="+", default=[8888, 8889, 8890])
    w.add_argument("--name", required=True)
    w.add_argument("--interval", type=float, default=0.25)
    w.add_argument("--minutes", type=float, default=150)
    w.add_argument("--fresh", action="store_true", help="empezar de cero aunque exista el archivo")
    r = sub.add_parser("render")
    r.add_argument("--name", required=True)
    r.add_argument("--port", type=int, help="solo ese puerto (instancia); por defecto, todos sumados")
    args = parser.parse_args()
    watch(args) if args.cmd == "watch" else render(args)


if __name__ == "__main__":
    main()
