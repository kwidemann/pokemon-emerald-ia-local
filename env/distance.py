"""Distancia a pie, en pasos, desde cualquier casilla del juego hasta un mapa meta, calculada solo con la ROM.

Sirve para premiar el acercamiento a un hito (R3): la recompensa es `coef * (distancia anterior - distancia nueva)`,
una función potencial, así que ir y volver suma cero y no se puede explotar dando vueltas.

Grafo (nodos = casillas `(grupo, número, x, y)` con las coordenadas que lee `read_state`, sin el borde de 7):
  - pasos a las 4 casillas vecinas si la de destino es transitable (metatile definido, sin colisión) y no hay un cambio
    de elevación (misma regla que `World.region_of` y `MapView.walkable`);
  - un desnivel (comportamiento 0x38-0x3B) no es una casilla: solo se salta en su dirección, de la casilla de antes a la de
    después, en un paso;
  - el agua (comportamientos 0x10-0x15) no se pisa: no se modela surf;
  - warps: la casilla de la puerta lleva a la casilla de llegada en el mapa destino (los warps dinámicos 127/127 se ignoran);
  - conexiones: salir por el borde del mapa lleva a la casilla correspondiente del vecino (misma fórmula que `World._edges`).
No se modelan NPC, árboles, rocas ni puertas con condiciones: la distancia es un mínimo optimista, no un camino garantizado.
Los comportamientos de casilla vienen de `mapdata.LEDGES` (verificados solo a medias, ver su docstring); la comprobación de
este módulo es que las distancias de los puntos de partida de entrenamiento salgan finitas y bajen a lo largo del camino que
sigue el agente de reglas (scripts/check_distance.py).
"""

from __future__ import annotations

import struct
from collections import deque

from env.mapdata import LEDGES, MAPGRID_UNDEFINED, STEP
from env.world import ROM_BASE, MapId, World

WATER_BEHAVIORS = set(range(0x10, 0x16))
Node = tuple[int, int, int, int]  # grupo, número, x, y


class WalkGraph:
    """Grafo de casillas de todo el mundo, con aristas inversas para calcular distancias hacia una meta."""

    def __init__(self, world: World):
        self.world = world
        self._attrs: dict[tuple[int, int], tuple[int, ...]] = {}
        self._cells: dict[MapId, tuple[int, int, tuple[int, ...], tuple[int, ...]]] = {}
        self._warps: dict[MapId, set[tuple[int, int]]] = {}
        self.reverse: dict[Node, list[Node]] = {}
        self._build()

    # --- lectura de la ROM ---------------------------------------------------------------------
    def _tileset_attrs(self, prim_ptr: int, sec_ptr: int) -> tuple[int, ...]:
        key = (prim_ptr, sec_ptr)
        if key not in self._attrs:
            rom = self.world.rom
            parts = []
            for ptr in (prim_ptr, sec_ptr):
                (attr_ptr,) = struct.unpack_from("<I", rom, ptr - ROM_BASE + 16)
                off = attr_ptr - ROM_BASE
                parts.append(struct.unpack_from("<512H", rom, off))
            self._attrs[key] = parts[0] + parts[1]
        return self._attrs[key]

    def _map_cells(self, mid: MapId):
        if mid not in self._cells:
            info = self.world.maps[mid]
            res = None
            if info.width > 0 and info.height > 0 and self.world._valid(info.layout_ptr):
                w, h, _b, map_ptr, prim, sec = struct.unpack_from("<iiIIII", self.world.rom, info.layout_ptr - ROM_BASE)
                if 0 < w * h < 40000 and self.world._valid(map_ptr) and self.world._valid(prim) and self.world._valid(sec):
                    cells = struct.unpack_from(f"<{w * h}H", self.world.rom, map_ptr - ROM_BASE)
                    res = (w, h, cells, self._tileset_attrs(prim, sec))
            self._cells[mid] = res
        return self._cells[mid]

    def tile_kind(self, mid: MapId, x: int, y: int) -> str:
        """Clase de una casilla según la ROM: "libre", "puerta", "desnivel_<dirección>" o "bloqueada".

        "bloqueada" incluye pared, agua, fuera del mapa y casillas sin metatile. No se modelan NPC, árboles ni rocas
        (misma limitación que el grafo). Las coordenadas son las que lee `read_state`.
        """
        if mid not in self.world.maps:
            return "bloqueada"
        data = self._map_cells(mid)
        if data is None:
            return "bloqueada"
        w, h, cells, attrs = data
        if not (0 <= x < w and 0 <= y < h):
            return "bloqueada"
        if mid not in self._warps:
            self._warps[mid] = set(self.world.maps[mid].warp_pos)
        if (x, y) in self._warps[mid]:
            return "puerta"
        v = cells[y * w + x]
        behavior = attrs[v & 0x3FF] & 0xFF
        if behavior in LEDGES:
            return f"desnivel_{LEDGES[behavior]}"
        if (v & 0x3FF) != MAPGRID_UNDEFINED and not (v >> 10) & 3 and behavior not in WATER_BEHAVIORS:
            return "libre"
        return "bloqueada"

    # --- construcción ---------------------------------------------------------------------------
    def _build(self) -> None:
        world = self.world
        forward: dict[Node, list[Node]] = {}

        def add(a: Node, b: Node) -> None:
            forward.setdefault(a, []).append(b)

        for mid, info in world.maps.items():
            data = self._map_cells(mid)
            if data is None:
                continue
            w, h, cells, attrs = data
            warp_tiles = set(info.warp_pos)

            def behavior(x: int, y: int) -> int:
                return attrs[cells[y * w + x] & 0x3FF] & 0xFF

            def passable(x: int, y: int) -> bool:
                v = cells[y * w + x]
                if (x, y) in warp_tiles:
                    return True
                return (v & 0x3FF) != MAPGRID_UNDEFINED and not (v >> 10) & 3 and behavior(x, y) not in WATER_BEHAVIORS and behavior(x, y) not in LEDGES

            for y in range(h):
                for x in range(w):
                    if not passable(x, y):
                        continue
                    ea = cells[y * w + x] >> 12
                    here = (*mid, x, y)
                    for name, (dx, dy) in STEP.items():
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < w and 0 <= ny < h):
                            continue  # el cruce de bordes se añade con las conexiones
                        if behavior(nx, ny) in LEDGES:
                            if LEDGES[behavior(nx, ny)] == name:  # salto: cae dos casillas más allá
                                jx, jy = nx + dx, ny + dy
                                if 0 <= jx < w and 0 <= jy < h and passable(jx, jy):
                                    add(here, (*mid, jx, jy))
                            continue
                        if not passable(nx, ny):
                            continue
                        eb = cells[ny * w + nx] >> 12
                        if ea and eb and ea != eb and 15 not in (ea, eb):
                            continue
                        add(here, (*mid, nx, ny))
            # warps
            for ex in info.exits:
                if ex.kind != "warp":
                    continue
                dest = world.maps.get(ex.dest)
                if dest is None or not 0 <= ex.warp_id < len(dest.warp_pos) or self._map_cells(ex.dest) is None:
                    continue
                dx_, dy_ = dest.warp_pos[ex.warp_id]
                if 0 <= ex.x < w and 0 <= ex.y < h:
                    add((*mid, ex.x, ex.y), (*ex.dest, dx_, dy_))
            # conexiones: se sale pisando fuera del borde del mapa
            for ex in info.exits:
                if ex.kind != "conn":
                    continue
                dest = world.maps.get(ex.dest)
                dest_data = self._map_cells(ex.dest) if dest else None
                if dest_data is None:
                    continue
                off = world._conn_offsets.get((mid, ex), 0)
                dw, dh = dest_data[0], dest_data[1]
                if ex.direction in ("up", "down"):
                    ys, dy_ = (0, dh - 1) if ex.direction == "up" else (h - 1, 0)
                    for x in range(w):
                        tx = x - off
                        if passable(x, ys) and 0 <= tx < dw:
                            add((*mid, x, ys), (*ex.dest, tx, dy_))
                else:
                    xs, dx_ = (0, dw - 1) if ex.direction == "left" else (w - 1, 0)
                    for y in range(h):
                        ty = y - off
                        if passable(xs, y) and 0 <= ty < dh:
                            add((*mid, xs, y), (*ex.dest, dx_, ty))

        for a, targets in forward.items():
            for b in targets:
                self.reverse.setdefault(b, []).append(a)

    # --- distancias -----------------------------------------------------------------------------
    def distances_to_map(self, goal: MapId) -> dict[Node, int]:
        """Pasos mínimos desde cada casilla hasta pisar cualquier casilla del mapa `goal` (0 dentro de él)."""
        dist: dict[Node, int] = {}
        queue: deque[Node] = deque()
        data = self._map_cells(goal)
        if data is None:
            return dist
        w, h = data[0], data[1]
        for y in range(h):
            for x in range(w):
                node = (*goal, x, y)
                dist[node] = 0
                queue.append(node)
        while queue:
            cur = queue.popleft()
            d = dist[cur] + 1
            for prev in self.reverse.get(cur, ()):
                if prev not in dist:
                    dist[prev] = d
                    queue.append(prev)
        return dist

    def distances_to_tile(self, goal: Node) -> dict[Node, int]:
        """Pasos mínimos desde cada casilla hasta la casilla `goal` (mapa, x, y); vacío si no está en el grafo."""
        dist: dict[Node, int] = {goal: 0}
        queue: deque[Node] = deque([goal])
        while queue:
            cur = queue.popleft()
            d = dist[cur] + 1
            for prev in self.reverse.get(cur, ()):
                if prev not in dist:
                    dist[prev] = d
                    queue.append(prev)
        return dist


_GRAPH: WalkGraph | None = None
_FIELDS: dict[MapId, dict[Node, int]] = {}
_TILE_FIELDS: dict[Node, dict[Node, int]] = {}


def get_graph() -> WalkGraph:
    """Grafo del mundo, construido una vez por proceso (≈1,7 s)."""
    global _GRAPH
    if _GRAPH is None:
        from env.world import load_rom

        _GRAPH = WalkGraph(World(load_rom()))
    return _GRAPH


def distance_field(goal: MapId) -> dict[Node, int]:
    """Campo de distancias hacia el mapa `goal`, con el grafo y los campos en caché del proceso."""
    if goal not in _FIELDS:
        _FIELDS[goal] = get_graph().distances_to_map(goal)
    return _FIELDS[goal]


def tile_distance_field(goal: Node) -> dict[Node, int]:
    """Campo de distancias hacia una casilla concreta (mapa, x, y), en caché por proceso."""
    if goal not in _TILE_FIELDS:
        _TILE_FIELDS[goal] = get_graph().distances_to_tile(goal)
    return _TILE_FIELDS[goal]
