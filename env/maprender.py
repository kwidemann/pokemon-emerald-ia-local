"""Dibuja el mapa de un lugar con los gráficos reales del juego, leyéndolos de la ROM (sin emulador).

Estructuras (pokeemerald; los offsets de `MapLayout` y de `metatileAttributes` coinciden con los que ya usa
`env/mapdata.py`, verificados en vivo; el resto sale del formato de Gen III y se comprueba mirando la imagen):
  MapLayout: s32 ancho, s32 alto, u16* borde, u16* mapa, Tileset* primario (+16), Tileset* secundario (+20).
  Tileset:   u8 comprimido (+0), u8 secundario (+1), tiles* (+4, LZ77 de 4 bits por píxel), paletas* (+8, 16 x 16 colores
             BGR555), metatiles* (+12, 8 entradas u16 por metatile).
  Entrada de metatile: bits 0-9 tile, 10 volteo horizontal, 11 vertical, 12-15 paleta.
  Los tiles 0-511 y los metatiles 0-511 son del tileset primario; el resto, del secundario. Las paletas 0-5 son del
  primario y las 6-12 del secundario. Las entradas 0-3 de un metatile son la capa de abajo y las 4-7, la de arriba (el
  color 0 de la de arriba es transparente).

Limitaciones: las animaciones (agua, flores) salen en su primer fotograma y no se dibujan los objetos ni el jugador.
"""

from __future__ import annotations

import struct

from PIL import Image

from env.world import ROM_BASE, MapId, World

PRIMARY_TILES = 512
PRIMARY_METATILES = 512
PRIMARY_PALETTES = 6


def lz77(rom: bytes, offset: int) -> bytes:
    """Descompresión LZ77 de la BIOS de GBA (tipo 0x10)."""
    header = struct.unpack_from("<I", rom, offset)[0]
    if header & 0xFF != 0x10:
        raise ValueError(f"no es LZ77 (cabecera {header:#x})")
    size = header >> 8
    out = bytearray()
    pos = offset + 4
    while len(out) < size:
        flags = rom[pos]
        pos += 1
        for bit in range(8):
            if len(out) >= size:
                break
            if flags & (0x80 >> bit):
                b1, b2 = rom[pos], rom[pos + 1]
                pos += 2
                length, disp = (b1 >> 4) + 3, ((b1 & 0xF) << 8 | b2) + 1
                for _ in range(length):
                    out.append(out[-disp])
            else:
                out.append(rom[pos])
                pos += 1
    return bytes(out[:size])


class _Tileset:
    def __init__(self, rom: bytes, ptr: int):
        base = ptr - ROM_BASE
        compressed = rom[base]
        tiles_ptr, pal_ptr, meta_ptr = struct.unpack_from("<III", rom, base + 4)
        tiles_off = tiles_ptr - ROM_BASE
        self.tiles = lz77(rom, tiles_off) if compressed else rom[tiles_off : tiles_off + PRIMARY_TILES * 32]
        self.palettes = [
            [self._color(struct.unpack_from("<H", rom, pal_ptr - ROM_BASE + p * 32 + c * 2)[0]) for c in range(16)]
            for p in range(16)
        ]
        # el número de metatiles no está en la estructura: se leen los 512 posibles (el borde de la ROM no se usa)
        self.metatiles = struct.unpack_from("<4096H", rom, meta_ptr - ROM_BASE)

    @staticmethod
    def _color(v: int) -> tuple[int, int, int]:
        return ((v & 31) * 255 // 31, (v >> 5 & 31) * 255 // 31, (v >> 10 & 31) * 255 // 31)


class MapRenderer:
    """Dibuja mapas completos a 16 píxeles por casilla; guarda en caché los tilesets ya leídos."""

    def __init__(self, world: World):
        self.world = world
        self._tilesets: dict[int, _Tileset] = {}
        self._meta_images: dict[tuple[int, int, int], Image.Image] = {}

    def _tileset(self, ptr: int) -> _Tileset:
        if ptr not in self._tilesets:
            self._tilesets[ptr] = _Tileset(self.world.rom, ptr)
        return self._tilesets[ptr]

    def _tile_pixels(self, prim: _Tileset, sec: _Tileset, entry: int) -> Image.Image:
        tile, hflip, vflip, pal = entry & 0x3FF, entry >> 10 & 1, entry >> 11 & 1, entry >> 12 & 15
        src = prim.tiles if tile < PRIMARY_TILES else sec.tiles
        idx = tile if tile < PRIMARY_TILES else tile - PRIMARY_TILES
        data = src[idx * 32 : idx * 32 + 32]
        palette = (prim if pal < PRIMARY_PALETTES else sec).palettes[pal]
        img = Image.new("RGBA", (8, 8))
        px = img.load()
        for i, byte in enumerate(data):
            y, x = divmod(i * 2, 8)
            for k, c in enumerate((byte & 15, byte >> 4)):
                px[x + k, y] = (*palette[c], 0 if c == 0 else 255)
        if hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if vflip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def _metatile(self, key: tuple[int, int], prim: _Tileset, sec: _Tileset, mt: int) -> Image.Image:
        cache_key = (*key, mt)
        if cache_key in self._meta_images:
            return self._meta_images[cache_key]
        entries = (prim.metatiles[mt * 8 : mt * 8 + 8] if mt < PRIMARY_METATILES
                   else sec.metatiles[(mt - PRIMARY_METATILES) * 8 : (mt - PRIMARY_METATILES) * 8 + 8])
        img = Image.new("RGBA", (16, 16), (0, 0, 0, 255))
        for layer in (0, 1):
            for i in range(4):
                tile = self._tile_pixels(prim, sec, entries[layer * 4 + i])
                if layer == 0:  # la de abajo es opaca: el color 0 es el fondo
                    base = Image.new("RGBA", (8, 8), (*tile.getpixel((0, 0))[:3], 255))
                    base.alpha_composite(tile)
                    tile = base
                img.alpha_composite(tile, ((i % 2) * 8, (i // 2) * 8))
        self._meta_images[cache_key] = img
        return img

    def render(self, mid: MapId) -> Image.Image:
        """Imagen RGB del mapa (ancho*16 x alto*16)."""
        world = self.world
        info = world.maps[mid]
        w, h, _border, map_ptr, prim_ptr, sec_ptr = struct.unpack_from("<iiIIII", world.rom, info.layout_ptr - ROM_BASE)
        prim, sec = self._tileset(prim_ptr), self._tileset(sec_ptr)
        cells = struct.unpack_from(f"<{w * h}H", world.rom, map_ptr - ROM_BASE)
        out = Image.new("RGBA", (w * 16, h * 16))
        key = (prim_ptr, sec_ptr)
        for y in range(h):
            for x in range(w):
                out.paste(self._metatile(key, prim, sec, cells[y * w + x] & 0x3FF), (x * 16, y * 16))
        return out.convert("RGB")
