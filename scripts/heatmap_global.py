"""Mapa de calor global: une los mapas de exteriores que se tocan por sus bordes y dibuja un solo PNG por corrida.

Usa los mismos datos que `heatmap.py render` (`runs/heatmaps/<name>.json`, todos los puertos sumados) pero deja fuera los
interiores (laboratorio, casas, Centro Pokémon...): solo entran los mapas con conexiones por el borde que el agente pisó en
alguna de las corridas indicadas. La posición de cada mapa en el plano sale de las conexiones de la ROM (`World._conn_offsets`).
La escala de color es la de cada corrida (logarítmica, máximo de toda la imagen), así que los colores no se comparan entre
corridas; para eso sirve el número de muestras que lleva cada imagen.

Uso:
    uv run python scripts/heatmap_global.py [--names ppo_500k ppo_r1_castigos ...]

Salidas en `runs/heatmaps/global/<name>.png` y `runs/heatmaps/global/_todas.png` (ignorados por git).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.heatmap import OUT, heat_color  # noqa: E402

NAMES = {(0, 9): "Littleroot", (0, 10): "Oldale", (0, 16): "Ruta 101", (0, 17): "Ruta 102", (0, 18): "Ruta 103", (0, 0): "Petalburg"}
RUNS = [
    "ppo_500k",
    "ppo_r1_castigos",
    "ppo_r3_acercamiento",
    "ppo_r3b_choques_escalados",
    "ppo_r3c_castigo_giros",
    "ppo_r3d_castigo_dialogos",
    "ppo_r4a_terreno",
    "ppo_r4b_memoria",
    "ppo_r4c_terreno_memoria",
]
SCALE = 3  # píxeles por píxel del juego (una casilla = 16 px de juego)


def load_counts(name: str) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    data = json.loads((OUT / f"{name}.json").read_text())["counts"]
    merged: dict[tuple[int, int], dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    for maps in data.values():
        for m, tiles in maps.items():
            g, n = (int(v) for v in m.split(","))
            for t, c in tiles.items():
                x, y = (int(v) for v in t.split(","))
                merged[(g, n)][(x, y)] += c
    return merged


def layout(world, maps: set) -> dict[tuple[int, int], tuple[int, int]]:
    """Origen (en casillas) de cada mapa del conjunto, colocando por conexiones desde el primero."""
    def neighbours(a):
        return {ex.dest for ex in world.maps[a].exits if ex.kind == "conn" and ex.dest in maps}

    comps, seen = [], set()
    for m in sorted(maps):  # componentes conexas por conexiones de borde; se queda la más grande
        if m in seen:
            continue
        comp, stack = set(), [m]
        while stack:
            a = stack.pop()
            if a not in comp:
                comp.add(a)
                stack.extend(neighbours(a) - comp)
        seen |= comp
        comps.append(comp)
    maps = max(comps, key=len)
    start = min(maps)
    pos = {start: (0, 0)}
    queue = [start]
    while queue:
        a = queue.pop(0)
        ax, ay = pos[a]
        ia = world.maps[a]
        for ex in ia.exits:
            if ex.kind != "conn" or ex.dest not in maps or ex.dest in pos:
                continue
            ib = world.maps[ex.dest]
            off = world._conn_offsets.get((a, ex), 0)
            origin = {
                "up": (ax + off, ay - ib.height),
                "down": (ax + off, ay + ia.height),
                "left": (ax - ib.width, ay + off),
                "right": (ax + ia.width, ay + off),
            }[ex.direction]
            pos[ex.dest] = origin
            queue.append(ex.dest)
    return pos


def render_run(name, counts, renderer, pos, crop_tiles) -> "Image.Image":
    from PIL import Image, ImageDraw

    tile = 16 * SCALE
    min_x = min(x for x, _ in pos.values())
    min_y = min(y for _, y in pos.values())
    width = max(x + renderer.world.maps[m].width for m, (x, _) in pos.items()) - min_x
    height = max(y + renderer.world.maps[m].height for m, (_, y) in pos.items()) - min_y
    width = min(width, crop_tiles)  # se recorta a la zona pisada en alguna corrida (igual para todas)
    canvas = Image.new("RGBA", (width * tile, height * tile), (20, 20, 20, 255))
    top = max((c for m in pos for c in counts.get(m, {}).values()), default=1)
    total = sum(c for m in pos for c in counts.get(m, {}).values())
    d_labels = []
    for m, (ox, oy) in pos.items():
        base = renderer.render(m).convert("RGBA")
        base = base.resize((base.width * SCALE, base.height * SCALE), Image.NEAREST)
        base = Image.blend(base, Image.new("RGBA", base.size, (0, 0, 0, 255)), 0.35)
        heat = Image.new("RGBA", base.size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(heat)
        for (x, y), c in counts.get(m, {}).items():
            k = math.log1p(c) / math.log1p(top)
            hd.rectangle([x * tile, y * tile, (x + 1) * tile - 1, (y + 1) * tile - 1], fill=(*heat_color(k), int(120 + 120 * k)))
        px, py = (ox - min_x) * tile, (oy - min_y) * tile
        piece = Image.alpha_composite(base, heat)
        piece = piece.crop((0, 0, min(piece.width, canvas.width - px), piece.height))
        canvas.alpha_composite(piece, (px, py))
        d_labels.append((px, py, f"{NAMES.get(m, m)} {m}"))
    d = ImageDraw.Draw(canvas)
    for px, py, text in d_labels:
        d.rectangle([px + 4, py + 4, px + 4 + 7 * len(text) + 8, py + 22], fill=(0, 0, 0, 200))
        d.text((px + 8, py + 8), text, fill=(240, 240, 240))
    bar_h = 30
    out = Image.new("RGB", (canvas.width, canvas.height + bar_h), (20, 20, 20))
    out.paste(canvas.convert("RGB"), (0, 0))
    d = ImageDraw.Draw(out)
    half = canvas.width // 4
    for i in range(half):
        d.line([(8 + i, canvas.height + 8), (8 + i, canvas.height + 18)], fill=heat_color(i / max(1, half - 1)))
    d.text((half + 20, canvas.height + 4), f"{name} | {total} muestras | maximo {top} en una casilla", fill=(230, 230, 230))
    d.text((half + 20, canvas.height + 16), "color logaritmico, propio de cada imagen", fill=(170, 170, 170))
    return out


def main() -> None:
    from PIL import Image, ImageDraw

    from env.maprender import MapRenderer
    from env.world import World, load_rom

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--names", nargs="+", default=RUNS)
    parser.add_argument("--sheet", default="_todas.png", help="nombre de la hoja con todas las imágenes")
    args = parser.parse_args()

    renderer = MapRenderer(World(load_rom()))
    world = renderer.world
    all_counts = {n: load_counts(n) for n in args.names}
    samples = defaultdict(int)  # muestras de todas las corridas por mapa; las lecturas espurias (1 o 2 muestras) no cuentan
    for c in all_counts.values():
        for m, tiles in c.items():
            samples[m] += sum(tiles.values())
    visited = {m for m, n in samples.items() if m in world.maps and n >= 50}
    outdoor = {m for m in visited if any(ex.kind == "conn" for ex in world.maps[m].exits)}
    pos = layout(world, outdoor)
    min_x = min(x for x, _ in pos.values())
    right = max(ox + x + 1 for c in all_counts.values() for m, (ox, _) in pos.items() for x, _ in c.get(m, {}))
    crop_tiles = right - min_x + 3
    missing = outdoor - set(pos)
    if missing:
        print(f"aviso: mapas de exterior que no se tocan con el resto, quedan fuera: {sorted(missing)}")
    print("mapas en el plano:", {NAMES.get(m, m): p for m, p in pos.items()})
    folder = OUT / "global"
    folder.mkdir(parents=True, exist_ok=True)
    images = []
    for name, counts in all_counts.items():
        img = render_run(name, counts, renderer, pos, crop_tiles)
        path = folder / f"{name}.png"
        img.save(path)
        images.append((name, img))
        print(f"{path.relative_to(ROOT)}: {img.width}x{img.height}")
    # hoja con todas las corridas, una columna por corrida y a media escala, con el nombre arriba
    from PIL import ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    prefix = os.path.commonprefix([n for n, _ in images]) if len(images) > 1 else ""
    w, h = images[0][1].size
    tw, th, title = w // 2, h // 2, 44
    sheet = Image.new("RGB", (tw * len(images), th + title), (20, 20, 20))
    sd = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(images):
        sheet.paste(img.resize((tw, th), Image.LANCZOS), (i * tw, title))
        sd.text((i * tw + 10, 6), name[len(prefix):], fill=(240, 240, 240), font=font)
    sheet_path = folder / args.sheet
    sheet.save(sheet_path)
    print(f"{sheet_path.relative_to(ROOT)}: {sheet.width}x{sheet.height}")


if __name__ == "__main__":
    main()
