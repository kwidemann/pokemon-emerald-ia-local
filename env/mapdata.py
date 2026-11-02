"""Mapa del lugar actual leído de la RAM: colisiones, objetos (NPC), warps y conexiones, y búsqueda de caminos.

Estructuras (fuente: pret/pokeemerald `include/global.fieldmap.h`; direcciones: pokeemerald.sym):
  gBackupMapLayout {s32 width, s32 height, u16 *map}: la cuadrícula incluye 7 casillas de borde por lado
    (width = ancho + 15, height = alto + 14). Cada u16: metatile (bits 0-9), colisión (10-11), elevación (12-15).
    Las coordenadas del jugador en SaveBlock1 son sin el borde: celda = (y + 7) * width + (x + 7).
  gObjectEvents[16] de 0x24 bytes: bit 0 de flags = activo, bit 16 = es el jugador; coordenadas actuales en
    +0x10 (con el borde de 7), dirección en +0x18 (1 sur, 2 norte, 3 oeste, 4 este).
  gMapHeader -> events (conteos + punteros) -> warps de 8 bytes {s16 x, s16 y, u8 elev, u8 warpId, u8 mapNum, u8 mapGroup}.
  gMapHeader -> connections {s32 count, ptr} de 12 bytes {u8 dir, s32 offset, u8 group, u8 num}.

  Comportamientos de casilla (hierba, desniveles, puertas): MapLayout {s32 w, s32 h, u16 *border, u16 *map,
    Tileset *primary (+16), Tileset *secondary (+20)}; Tileset.metatileAttributes en +16 (u16 por metatile;
    comportamiento = byte bajo). Metatile < 512 usa el tileset primario, el resto el secundario.
    Valores de comportamiento (pokeemerald `metatile_behaviors.h`, de memoria; verificar en vivo): hierba alta 2,
    hierba larga 3, saltos de desnivel 0x38 este / 0x39 oeste / 0x3A norte / 0x3B sur.

Verificado en vivo en el laboratorio de Birch: ancho 28, alto 27 (13x13 + borde), el jugador y dos NPC en
sus casillas, y dos warps en (6,12) y (7,12) hacia Littleroot. NO verificado: saltos de desnivel (ledges),
agua/surf (se aproxima con la regla de elevación) y cualquier cosa fuera del laboratorio.
"""

from __future__ import annotations

import struct
from collections import deque
from dataclasses import dataclass, field

G_BACKUP_MAP_LAYOUT = 0x03005DC0
G_OBJECT_EVENTS = 0x02037350
G_MAP_HEADER = 0x02037318
OBJECT_EVENT_SIZE = 0x24
OBJECT_EVENT_COUNT = 16
MAP_OFFSET = 7  # casillas de borde a cada lado dentro de la cuadrícula
MAPGRID_UNDEFINED = 0x03FF

# Direcciones de gObjectEvents[].facingDirection y de las conexiones.
FACING = {1: "down", 2: "up", 3: "left", 4: "right"}
STEP = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
# Valores de MapConnection.direction en pokeemerald (DIRECTION_*): 1 sur, 2 norte, 3 oeste, 4 este
CONN_DIR = {1: "down", 2: "up", 3: "left", 4: "right"}


GRASS_BEHAVIORS = {2, 3}
LEDGES = {0x38: "right", 0x39: "left", 0x3A: "up", 0x3B: "down"}
_ATTR_CACHE: dict[int, tuple[int, ...]] = {}


@dataclass(frozen=True)
class Warp:
    x: int
    y: int
    dest_group: int
    dest_num: int
    warp_id: int


@dataclass(frozen=True)
class Connection:
    direction: str  # "up" | "down" | "left" | "right"
    offset: int
    dest_group: int
    dest_num: int


@dataclass(frozen=True)
class MapObject:
    index: int
    graphics_id: int
    local_id: int
    x: int
    y: int
    facing: str
    is_player: bool


@dataclass
class MapView:
    group: int
    num: int
    width: int  # ancho de la cuadrícula completa (incluye bordes)
    height: int
    grid: tuple[int, ...]
    objects: list[MapObject] = field(default_factory=list)
    warps: list[Warp] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    attrs: tuple[int, ...] = ()  # atributos de los 1024 metatiles posibles (primario + secundario)

    def behavior(self, x: int, y: int) -> int | None:
        v = self.cell(x, y)
        if v is None or not self.attrs:
            return None
        mt = v & 0x3FF
        return self.attrs[mt] & 0xFF if mt < len(self.attrs) else None

    def is_grass(self, x: int, y: int) -> bool:
        return self.behavior(x, y) in GRASS_BEHAVIORS

    @property
    def inner_width(self) -> int:
        return self.width - 15

    @property
    def inner_height(self) -> int:
        return self.height - 14

    @property
    def player(self) -> MapObject:
        return next(o for o in self.objects if o.is_player)

    def cell(self, x: int, y: int) -> int | None:
        gx, gy = x + MAP_OFFSET, y + MAP_OFFSET
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return None
        return self.grid[gy * self.width + gx]

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.inner_width and 0 <= y < self.inner_height

    def npc_tiles(self) -> set[tuple[int, int]]:
        return {(o.x, o.y) for o in self.objects if not o.is_player}

    def walkable(self, x: int, y: int, from_xy: tuple[int, int] | None = None, blocked: set | None = None) -> bool:
        v = self.cell(x, y)
        if v is None or (v & 0x3FF) == MAPGRID_UNDEFINED or (v >> 10) & 3:
            return False
        if blocked and (x, y) in blocked:
            return False
        if self.behavior(x, y) in LEDGES:  # no se puede quedar parado sobre un desnivel: solo se salta
            return False
        if from_xy is not None:
            a = (self.cell(*from_xy) or 0) >> 12
            b = v >> 12
            if a and b and a != b and 15 not in (a, b):  # desnivel/agua: elevaciones distintas bloquean
                return False
        return True

    def ascii(self, radius: int = 7) -> str:
        p = self.player
        rows = []
        npcs = self.npc_tiles()
        warps = {(w.x, w.y) for w in self.warps}
        for y in range(p.y - radius, p.y + radius + 1):
            row = ""
            for x in range(p.x - radius, p.x + radius + 1):
                if (x, y) == (p.x, p.y):
                    row += "@"
                elif (x, y) in npcs:
                    row += "N"
                elif (x, y) in warps:
                    row += "W"
                elif self.cell(x, y) is None:
                    row += " "
                elif self.behavior(x, y) in LEDGES:
                    row += {"right": ">", "left": "<", "up": "^", "down": "v"}[LEDGES[self.behavior(x, y)]]
                elif not self.walkable(x, y):
                    row += "#"
                elif self.is_grass(x, y):
                    row += "g"
                else:
                    row += "." if self.inside(x, y) else ","
            rows.append(row)
        return "\n".join(rows)


def _read_attrs(emu, layout_ptr: int) -> tuple[int, ...]:
    """Atributos de metatile (comportamiento en el byte bajo) del tileset primario y el secundario; con caché."""
    if layout_ptr in _ATTR_CACHE:
        return _ATTR_CACHE[layout_ptr]
    _, (layout,) = emu.read_multi([(layout_ptr, 24)])
    primary, secondary = struct.unpack_from("<II", layout, 16)
    _, tilesets = emu.read_multi([(primary, 24), (secondary, 24)])
    (prim_attr,) = struct.unpack_from("<I", tilesets[0], 16)
    (sec_attr,) = struct.unpack_from("<I", tilesets[1], 16)
    _, (a, b) = emu.read_multi([(prim_attr, 1024), (sec_attr, 1024)])
    attrs = struct.unpack("<1024H", a + b)
    _ATTR_CACHE[layout_ptr] = attrs
    return attrs


def read_map(emu, group: int, num: int) -> MapView:
    """Lee el mapa actual. `group`/`num` vienen de `GameState` (se usan solo para etiquetar la vista)."""
    _, (layout, objs, header) = emu.read_multi(
        [
            (G_BACKUP_MAP_LAYOUT, 12),
            (G_OBJECT_EVENTS, OBJECT_EVENT_SIZE * OBJECT_EVENT_COUNT),
            (G_MAP_HEADER, 28),
        ]
    )
    width, height, grid_ptr = struct.unpack("<iiI", layout)
    header_layout, events_ptr, _, conns_ptr = struct.unpack_from("<IIII", header, 0)

    ranges = [(grid_ptr, width * height * 2), (events_ptr, 20)]
    if conns_ptr:
        ranges.append((conns_ptr, 8))
    _, got = emu.read_multi(ranges)
    grid = struct.unpack(f"<{width * height}H", got[0])
    n_warps = got[1][1]
    (warps_ptr,) = struct.unpack_from("<I", got[1], 8)

    more = []
    if n_warps:
        more.append((warps_ptr, 8 * n_warps))
    n_conns = conns_cptr = 0
    if conns_ptr:
        n_conns, conns_cptr = struct.unpack("<iI", got[2])
        if 0 < n_conns <= 16:
            more.append((conns_cptr, 12 * n_conns))
    _, tail = emu.read_multi(more) if more else (0, [])
    warps, conns = [], []
    i = 0
    if n_warps:
        for k in range(n_warps):
            x, y, _el, warp_id, num_, group_ = struct.unpack_from("<hhBBBB", tail[i], 8 * k)
            warps.append(Warp(x, y, group_, num_, warp_id))
        i += 1
    if conns_ptr and 0 < n_conns <= 16:
        for k in range(n_conns):
            d, off, group_, num_ = struct.unpack_from("<BxxxiBBxx", tail[i], 12 * k)
            if d in CONN_DIR:
                conns.append(Connection(CONN_DIR[d], off, group_, num_))

    objects = []
    for k in range(OBJECT_EVENT_COUNT):
        o = objs[k * OBJECT_EVENT_SIZE : (k + 1) * OBJECT_EVENT_SIZE]
        (flags,) = struct.unpack_from("<I", o, 0)
        if not flags & 1:
            continue
        cx, cy = struct.unpack_from("<hh", o, 0x10)
        objects.append(
            MapObject(k, o[5], o[8], cx - MAP_OFFSET, cy - MAP_OFFSET, FACING.get(o[0x18] & 0xF, "?"), bool((flags >> 16) & 1))
        )
    attrs = _read_attrs(emu, header_layout)
    return MapView(group, num, width, height, grid, objects, warps, conns, attrs)


def find_path(
    view: MapView,
    goals: set[tuple[int, int]],
    start: tuple[int, int] | None = None,
    avoid_npcs: bool = True,
    allow_outside: bool = True,
) -> list[str] | None:
    """BFS en 4 direcciones hasta cualquiera de las casillas `goals`. Devuelve la lista de movimientos o None.

    `allow_outside` deja pisar el borde de 7 casillas (contiene la zona del mapa vecino, por donde se
    cruza una conexión). Las casillas de NPC se evitan, salvo que la propia meta sea una de ellas.
    """
    start = start or (view.player.x, view.player.y)
    blocked = view.npc_tiles() - goals if avoid_npcs else None
    prev: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur in goals and cur != start:
            path = []
            while prev[cur] is not None:
                cur, move = prev[cur][0], prev[cur][1]
                path.append(move)
            return path[::-1]
        for move, (dx, dy) in STEP.items():
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in prev:
                continue
            if view.behavior(*nxt) in LEDGES:
                if LEDGES[view.behavior(*nxt)] != move:
                    continue  # un desnivel solo se cruza en su sentido
                nxt = (nxt[0] + dx, nxt[1] + dy)  # el salto aterriza una casilla más allá
                if nxt in prev:
                    continue
            if not allow_outside and not view.inside(*nxt):
                continue
            if not view.walkable(*nxt, from_xy=cur, blocked=blocked):
                continue
            prev[nxt] = (cur, move)
            queue.append(nxt)
    return None
