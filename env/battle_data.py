"""Datos de combate leídos de la ROM: movimientos, tipos de las especies y tabla de efectividad.

Direcciones (pokeemerald.sym, ROM verificada): gTypeEffectiveness 0x0831ACE8 (tripletes u8 atacante, defensor,
multiplicador; 10 = x1, 20 = x2, 5 = x0.5, 0 = inmune; 0xFF termina), gBattleMoves 0x0831C898 (12 bytes por
movimiento: efecto, poder, tipo, precisión, PP, ...), gSpeciesInfo 0x083203CC (28 bytes por especie: HP base,
ataque, defensa, velocidad, ataque esp., defensa esp., tipo 1, tipo 2, ...). Los índices de especie son los
internos de Gen III (los mismos que `PartyMon.species`).

Tipos (pokeemerald `constants/pokemon.h`): 0 normal, 1 lucha, 2 volador, 3 veneno, 4 tierra, 5 roca, 6 bicho,
7 fantasma, 8 acero, 9 ???, 10 fuego, 11 agua, 12 planta, 13 eléctrico, 14 psíquico, 15 hielo, 16 dragón, 17 siniestro.

Verificado en vivo: nada todavía; los valores se leen de la ROM sin emulador y se contrastan en `scripts/`
imprimiendo, por ejemplo, el Placaje (poder 35, tipo normal) y el tipo de Mudkip (agua).
"""

from __future__ import annotations

import struct

ROM_BASE = 0x08000000
G_TYPE_EFFECTIVENESS = 0x0831ACE8
G_BATTLE_MOVES = 0x0831C898
G_SPECIES_INFO = 0x083203CC
MOVE_SIZE = 12
SPECIES_SIZE = 28
NUM_MOVES = 355
NUM_SPECIES = 412

# Tipos especiales en Gen III (el resto es físico). Sirve para elegir entre ataque y ataque especial.
SPECIAL_TYPES = {10, 11, 12, 13, 14, 15, 16, 17}


class BattleData:
    def __init__(self, rom: bytes):
        self.rom = rom
        self._chart: dict[tuple[int, int], int] = {}
        off = G_TYPE_EFFECTIVENESS - ROM_BASE
        while off < len(rom) and rom[off] != 0xFF:
            atk, dfn, mult = rom[off : off + 3]
            if atk != 0xFE:  # 0xFE separa las entradas que se ignoran con Previsión
                self._chart[(atk, dfn)] = mult
            off += 3

    def move(self, move_id: int) -> tuple[int, int, int, int]:
        """(poder, tipo, precisión, PP) del movimiento; (0, 0, 0, 0) si el ID no es válido."""
        if not 0 < move_id < NUM_MOVES:
            return (0, 0, 0, 0)
        o = G_BATTLE_MOVES - ROM_BASE + MOVE_SIZE * move_id
        _effect, power, mtype, accuracy, pp = struct.unpack_from("<5B", self.rom, o)
        return power, mtype, accuracy, pp

    def species_types(self, species: int) -> tuple[int, int]:
        if not 0 < species < NUM_SPECIES:
            return (0, 0)
        o = G_SPECIES_INFO - ROM_BASE + SPECIES_SIZE * species
        return self.rom[o + 6], self.rom[o + 7]

    def effectiveness(self, attack_type: int, defender_types: tuple[int, int]) -> float:
        mult = 1.0
        for t in set(defender_types):
            mult *= self._chart.get((attack_type, t), 10) / 10
        return mult

    def move_score(self, move_id: int, attacker_species: int, defender_species: int) -> float:
        """Daño esperado relativo: poder x STAB x efectividad x precisión. 0 para movimientos sin daño."""
        power, mtype, accuracy, _pp = self.move(move_id)
        if not power:
            return 0.0
        stab = 1.5 if mtype in self.species_types(attacker_species) else 1.0
        eff = self.effectiveness(mtype, self.species_types(defender_species))
        return power * stab * eff * (accuracy or 100) / 100
