"""Grafo de mapas del juego leído directamente de la ROM (sin emulador): warps y conexiones de cada mapa.

Sirve para decidir por qué salida ir en cada mapa para llegar a otro (`World.route`).

Estructuras de la ROM (fuente: pret/pokeemerald `include/global.fieldmap.h`; ver también env/mapdata.py):
  gMapGroups (0x08486578): tabla de punteros, uno por grupo, a un arreglo de punteros a `MapHeader`.
  Los arreglos de grupos ocupan [0x08485D60, 0x08486578) en la ROM verificada (SHA-1 en ram_map.py);
  el tamaño de cada grupo se deduce de las direcciones de arranque del arreglo siguiente.
  MapHeader (28 bytes): layout*, events*, scripts*, connections*, ...; MapLayout: s32 ancho, s32 alto, ...

Los IDs `(grupo, número)` salen de esa tabla y son los mismos que lee `read_state` en vivo.
NO verificado: el recorrido completo de la tabla (se contrastó solo el laboratorio `(1,4)` en vivo,
con la misma lectura por la RAM) y los destinos dinámicos (warps a mapa 127/127 se ignoran).
"""

from __future__ import annotations

import struct
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROM_BASE = 0x08000000
G_MAP_GROUPS = 0x08486578
GROUP_ARRAYS_START = 0x08485D60
DYNAMIC_MAP = (127, 127)  # destino "dinámico": lo decide el script (p. ej. el camión)

MapId = tuple[int, int]


@dataclass(frozen=True)
class Exit:
    kind: str  # "warp" | "conn"
    dest: MapId
    x: int = 0  # warp: casilla de la puerta
    y: int = 0
    direction: str = ""  # conn: lado del mapa por el que se sale ("up", "down", "left", "right")
    warp_id: int = 0


@dataclass(frozen=True)
class NpcTemplate:
    """Objeto de mapa definido en la ROM (ObjectEventTemplate, 24 bytes). La posición es la inicial: puede moverse."""

    local_id: int
    graphics_id: int
    x: int
    y: int
    movement_type: int
    trainer_type: int  # 0 = NPC normal; 1 y 2 = entrenador que reta al verte
    script: int
    flag: int  # si esta bandera está activa el objeto no aparece


@dataclass
class MapInfo:
    id: MapId
    width: int
    height: int
    exits: list[Exit] = field(default_factory=list)
    npcs: list[NpcTemplate] = field(default_factory=list)
    layout_ptr: int = 0  # puntero a MapLayout en la ROM (cuadrícula estática de colisiones)
    warp_pos: list[tuple[int, int]] = field(default_factory=list)  # (x, y) de cada warp, por su índice


def load_rom(roms_dir: str | Path = "roms") -> bytes:
    roms = Path(roms_dir)
    for gba in roms.glob("*.gba"):
        return gba.read_bytes()
    for zp in roms.glob("*.zip"):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.lower().endswith(".gba"):
                    return z.read(name)
    raise FileNotFoundError(f"no hay una ROM .gba (ni un .zip con una) en {roms.resolve()}")


class World:
    def __init__(self, rom: bytes):
        self.rom = rom
        self.maps: dict[MapId, MapInfo] = {}
        self._grids: dict = {}
        self._comps: dict = {}
        self._edge_cache: dict = {}
        self._conn_offsets: dict = {}
        self._parse()

    def _u32(self, ptr: int) -> int:
        off = ptr - ROM_BASE
        return struct.unpack_from("<I", self.rom, off)[0]

    def _valid(self, ptr: int) -> bool:
        return ROM_BASE <= ptr < ROM_BASE + len(self.rom)

    def _parse(self) -> None:
        group_ptrs = []
        ptr = G_MAP_GROUPS
        while True:
            val = self._u32(ptr)
            if not (GROUP_ARRAYS_START <= val < G_MAP_GROUPS):
                break
            group_ptrs.append(val)
            ptr += 4
        bounds = sorted(set(group_ptrs)) + [G_MAP_GROUPS]
        for g, start in enumerate(group_ptrs):
            end = bounds[bounds.index(start) + 1]
            for n in range((end - start) // 4):
                header = self._u32(start + 4 * n)
                if self._valid(header):
                    self._parse_map((g, n), header)

    def _parse_map(self, mid: MapId, header: int) -> None:
        off = header - ROM_BASE
        layout, events, _scripts, conns = struct.unpack_from("<IIII", self.rom, off)
        width = height = 0
        if self._valid(layout):
            width, height = struct.unpack_from("<ii", self.rom, layout - ROM_BASE)
        info = MapInfo(mid, width, height, layout_ptr=layout)
        if self._valid(events):
            ev = events - ROM_BASE
            n_objects = self.rom[ev]
            (objects_ptr,) = struct.unpack_from("<I", self.rom, ev + 4)
            if n_objects and self._valid(objects_ptr):
                for k in range(n_objects):
                    o = objects_ptr - ROM_BASE + 24 * k
                    x, y = struct.unpack_from("<hh", self.rom, o + 4)
                    (trainer_type,) = struct.unpack_from("<H", self.rom, o + 12)
                    script, flag = struct.unpack_from("<IH", self.rom, o + 16)
                    info.npcs.append(
                        NpcTemplate(self.rom[o], self.rom[o + 1], x, y, self.rom[o + 9], trainer_type, script, flag)
                    )
            n_warps = self.rom[ev + 1]
            (warps_ptr,) = struct.unpack_from("<I", self.rom, ev + 8)
            if n_warps and self._valid(warps_ptr):
                for k in range(n_warps):
                    x, y, _el, warp_id, num, group = struct.unpack_from("<hhBBBB", self.rom, warps_ptr - ROM_BASE + 8 * k)
                    info.warp_pos.append((x, y))
                    if (group, num) != DYNAMIC_MAP:
                        info.exits.append(Exit("warp", (group, num), x=x, y=y, warp_id=warp_id))
        if self._valid(conns):
            count, cptr = struct.unpack_from("<iI", self.rom, conns - ROM_BASE)
            if 0 < count <= 16 and self._valid(cptr):
                names = {1: "down", 2: "up", 3: "left", 4: "right"}
                for k in range(count):
                    d, off, group, num = struct.unpack_from("<BxxxiBBxx", self.rom, cptr - ROM_BASE + 12 * k)
                    if d in names:
                        ex = Exit("conn", (group, num), direction=names[d])
                        info.exits.append(ex)
                        self._conn_offsets[(mid, ex)] = off
        self.maps[mid] = info

    NURSE_GRAPHICS_ID = 58  # visto en (2,2) (Centro Pokémon de Oldale); 18 mapas lo tienen

    def pokemon_centers(self) -> set[MapId]:
        """Mapas con una enfermera (gráfico 58): los Centros Pokémon de la planta baja, la Liga y el Frente de Batalla."""
        return {mid for mid, m in self.maps.items() if any(n.graphics_id == self.NURSE_GRAPHICS_ID for n in m.npcs)}

    # --- regiones (zonas conectadas a pie) -----------------------------------------------------
    # Un mapa como la Ruta 104 tiene dos zonas separadas (sur y norte) que solo se unen por el Bosque Petalburgo;
    # planificar solo por mapas no lo ve. Se calculan componentes conexas de la cuadrícula de la ROM (colisión
    # y desnivel de elevación; NO se consideran NPC ni saltos de desnivel, que se resuelven en vivo).

    def _grid(self, mid: MapId) -> tuple[int, int, tuple[int, ...]] | None:
        if mid in self._grids:
            return self._grids[mid]
        info = self.maps.get(mid)
        res = None
        if info and self._valid(info.layout_ptr):
            w, h, _border, map_ptr = struct.unpack_from("<iiII", self.rom, info.layout_ptr - ROM_BASE)
            if 0 < w * h < 40000 and self._valid(map_ptr):
                res = (w, h, struct.unpack_from(f"<{w * h}H", self.rom, map_ptr - ROM_BASE))
        self._grids[mid] = res
        return res

    @staticmethod
    def _passable(v: int) -> bool:
        return (v & 0x3FF) != 0x3FF and not (v >> 10) & 3

    def _components(self, mid: MapId) -> dict[tuple[int, int], int]:
        if mid in self._comps:
            return self._comps[mid]
        comp: dict[tuple[int, int], int] = {}
        g = self._grid(mid)
        if g:
            w, h, cells = g
            n = 0
            for sy in range(h):
                for sx in range(w):
                    if (sx, sy) in comp or not self._passable(cells[sy * w + sx]):
                        continue
                    comp[(sx, sy)] = n
                    queue = deque([(sx, sy)])
                    while queue:
                        cx, cy = queue.popleft()
                        ea = cells[cy * w + cx] >> 12
                        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                            if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in comp:
                                continue
                            v = cells[ny * w + nx]
                            eb = v >> 12
                            if not self._passable(v) or (ea and eb and ea != eb and 15 not in (ea, eb)):
                                continue
                            comp[(nx, ny)] = n
                            queue.append((nx, ny))
                    n += 1
        self._comps[mid] = comp
        return comp

    def region_of(self, mid: MapId, x: int, y: int) -> int | None:
        """Zona conexa a la que pertenece la casilla (o una vecina caminable, para puertas con colisión)."""
        comp = self._components(mid)
        if (x, y) in comp:
            return comp[(x, y)]
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if (x + dx, y + dy) in comp:
                return comp[(x + dx, y + dy)]
        return None

    def _edges(self, mid: MapId) -> list[tuple[int | None, MapId, int | None, Exit]]:
        """Aristas salientes: (zona de origen, mapa destino, zona de llegada o None si no se conoce, salida)."""
        if mid in self._edge_cache:
            return self._edge_cache[mid]
        info = self.maps[mid]
        comp = self._components(mid)
        out: list[tuple[int | None, MapId, int | None, Exit]] = []
        for ex in info.exits:
            dest = self.maps.get(ex.dest)
            if dest is None:
                continue
            if ex.kind == "warp":
                src_region = self.region_of(mid, ex.x, ex.y)
                dst_region = None
                if 0 <= ex.warp_id < len(dest.warp_pos):
                    dst_region = self.region_of(ex.dest, *dest.warp_pos[ex.warp_id])
                out.append((src_region, ex.dest, dst_region, ex))
            else:
                w, h = info.width, info.height
                off = self._conn_offsets.get((mid, ex), 0)
                seen_pairs: set[tuple[int, int | None]] = set()
                for (x, y), r in comp.items():
                    if not {"up": y == 0, "down": y == h - 1, "left": x == 0, "right": x == w - 1}[ex.direction]:
                        continue
                    d = {
                        "up": (x - off, dest.height - 1),
                        "down": (x - off, 0),
                        "left": (dest.width - 1, y - off),
                        "right": (0, y - off),
                    }[ex.direction]
                    pair = (r, self.region_of(ex.dest, *d))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        out.append((pair[0], ex.dest, pair[1], ex))
        self._edge_cache[mid] = out
        return out

    def route(
        self,
        src: MapId,
        dst: MapId,
        avoid: set[MapId] | None = None,
        blocked: set[tuple[MapId, Exit]] | None = None,
        pos: tuple[int, int] | None = None,
    ) -> list[Exit] | None:
        """Camino más corto, en número de salidas, de `src` a `dst`, contando las zonas conexas de cada mapa.

        `pos`: casilla actual en `src` (define la zona de partida; sin ella se parte de todas). `blocked`: pares
        (mapa, salida) que se descubrió que NO se pueden usar desde esa zona (restringe solo la primera salida).
        Un mapa puede repetirse si es otra zona (Ruta 104: sur -> Bosque -> norte).
        """
        if src == dst:
            return []
        avoid = avoid or set()
        blocked = blocked or set()
        all_regions = set(self._components(src).values()) or {None}
        start = {self.region_of(src, *pos)} - {None} if pos is not None else set()
        start = start or all_regions
        queue: deque[tuple[MapId, int | None, list[Exit]]] = deque()
        seen: set[tuple[MapId, int | None]] = set()
        for r in start:
            seen.add((src, r))
            queue.append((src, r, []))
        while queue:
            cur, region, path = queue.popleft()
            for r, dmap, dregion, ex in self._edges(cur):
                if r is not None and region is not None and r != region:
                    continue
                if not path and (cur, ex) in blocked:
                    continue
                if dmap in avoid:
                    continue
                if dmap == dst:
                    return path + [ex]
                if dregion is None:  # llegada desconocida: se prueba desde todas las zonas del mapa destino
                    nxt = set(self._components(dmap).values()) or {None}
                else:
                    nxt = {dregion}
                for rr in nxt:
                    if (dmap, rr) not in seen:
                        seen.add((dmap, rr))
                        queue.append((dmap, rr, path + [ex]))
        return None
