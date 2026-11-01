"""Lectura del estado del juego desde la RAM de mGBA.

`read_state(emu)` devuelve un `GameState` con lo mínimo que necesitan los cuatro enfoques.
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass, field

from env import ram_map as m


@dataclass
class PartyMon:
    species: int  # índice interno de especie de Gen III (NO es el número de la Pokédex nacional)
    level: int
    hp: int
    max_hp: int
    status: int
    moves: tuple[int, int, int, int]
    pp: tuple[int, int, int, int]
    held_item: int
    experience: int
    checksum_ok: bool  # False => el descifrado no coincide con el checksum del juego: no fiarse de species/moves


@dataclass
class GameState:
    frame: int
    map_group: int
    map_num: int
    x: int
    y: int
    script_active: bool  # hay un script/diálogo en ejecución
    in_battle: bool
    battle_type_flags: int
    money: int
    badges: int  # cantidad de medallas
    player_name: str
    party: list[PartyMon] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# Tabla de caracteres de Gen III: solo letras y dígitos (suficiente para el nombre del jugador).
def _decode_name(raw: bytes) -> str:
    out = []
    for b in raw:
        if b == 0xFF:
            break
        if 0xBB <= b <= 0xD4:
            out.append(chr(ord("A") + b - 0xBB))
        elif 0xD5 <= b <= 0xEE:
            out.append(chr(ord("a") + b - 0xD5))
        elif 0xA1 <= b <= 0xAA:
            out.append(chr(ord("0") + b - 0xA1))
        else:
            out.append("?")
    return "".join(out)


def parse_party_mon(raw: bytes) -> PartyMon | None:
    """Interpreta los 100 bytes de un `struct Pokemon`. Devuelve None si el hueco está vacío."""
    if len(raw) != m.POKEMON_SIZE:
        raise ValueError(f"se esperaban {m.POKEMON_SIZE} bytes, llegaron {len(raw)}")
    if not any(raw):
        return None
    personality, ot_id = struct.unpack_from("<II", raw, m.MON_PERSONALITY)
    (checksum,) = struct.unpack_from("<H", raw, m.MON_CHECKSUM)

    key = personality ^ ot_id
    words = struct.unpack_from("<12I", raw, m.MON_SECURE)
    secure = struct.pack("<12I", *(w ^ key for w in words))
    # El juego guarda como checksum la suma de las 24 palabras de 16 bits ya descifradas.
    checksum_ok = (sum(struct.unpack("<24H", secure)) & 0xFFFF) == checksum

    growth_i, attacks_i, _evs_i, _misc_i = m.SUBSTRUCT_ORDER[personality % 24]
    growth = secure[12 * growth_i : 12 * growth_i + 12]
    attacks = secure[12 * attacks_i : 12 * attacks_i + 12]
    species, held_item, experience = struct.unpack_from("<HHI", growth, 0)
    moves = struct.unpack_from("<4H", attacks, 0)
    pp = struct.unpack_from("<4B", attacks, 8)

    (status,) = struct.unpack_from("<I", raw, m.MON_STATUS)
    level = raw[m.MON_LEVEL]
    hp, max_hp = struct.unpack_from("<HH", raw, m.MON_HP)
    return PartyMon(species, level, hp, max_hp, status, moves, pp, held_item, experience, checksum_ok)


def read_state(emu) -> GameState:
    sb1, sb2 = struct.unpack("<II", emu.read_range(m.G_SAVE_BLOCK1_PTR, 8))

    x, y, map_group, map_num = struct.unpack("<hhbb", emu.read_range(sb1 + m.SB1_POS, 6))
    key = struct.unpack("<I", emu.read_range(sb2 + m.SB2_ENCRYPTION_KEY, 4))[0]
    (money_raw,) = struct.unpack("<I", emu.read_range(sb1 + m.SB1_MONEY, 4))
    name = _decode_name(emu.read_range(sb2 + m.SB2_PLAYER_NAME, 8))

    # Medallas: flags 0x867..0x86E dentro del arreglo de flags de SaveBlock1.
    first_byte = m.FLAG_BADGE01_GET >> 3
    flag_bytes = emu.read_range(sb1 + m.SB1_FLAGS + first_byte, 2)
    badges = sum(
        (flag_bytes[((m.FLAG_BADGE01_GET + i) >> 3) - first_byte] >> ((m.FLAG_BADGE01_GET + i) & 7)) & 1
        for i in range(m.NUM_BADGES)
    )

    count = min(emu.read8(m.G_PLAYER_PARTY_COUNT), m.PARTY_MAX)
    party_raw = emu.read_range(m.G_PLAYER_PARTY, m.POKEMON_SIZE * count) if count else b""
    party = [
        mon
        for i in range(count)
        if (mon := parse_party_mon(party_raw[i * m.POKEMON_SIZE : (i + 1) * m.POKEMON_SIZE])) is not None
    ]

    return GameState(
        frame=emu.frame(),
        map_group=map_group,
        map_num=map_num,
        x=x,
        y=y,
        script_active=emu.read8(m.S_GLOBAL_SCRIPT_CONTEXT_STATUS) != m.SCRIPT_CONTEXT_DISABLED,
        in_battle=bool(emu.read8(m.G_MAIN_IN_BATTLE) & m.IN_BATTLE_MASK),
        battle_type_flags=emu.read32(m.G_BATTLE_TYPE_FLAGS),
        money=money_raw ^ key,
        badges=badges,
        player_name=name,
        party=party,
    )
