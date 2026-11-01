"""Catálogo de hitos del Tramo 1 (specs.md, 9.1) y el seguimiento del historial que algunos necesitan.

Un hito es "enganchado": una vez alcanzado en un episodio se mantiene. Los que ya se cumplen al
hacer `reset()` se marcan sin dar recompensa (el savestate inicial los cubre).

Los IDs de mapa de ciudades y rutas están INFERIDOS del orden de pokeemerald (Littleroot `(0,9)` y
Ruta 101 `(0,16)` sí se verificaron en vivo; el resto no). Un ID equivocado hace que el hito no se
dispare nunca, no que se dispare mal. Se confirman al pisar cada mapa.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from env.state import GameState

BATTLE_TYPE_TRAINER = 0x08  # bit de pokeemerald; aún sin ver en una batalla real

# (grupo, número) de mapa. Verificados: LAB, LITTLEROOT, ROUTE_101. El resto, inferidos.
LAB = (1, 4)
LITTLEROOT = (0, 9)
ROUTE_101 = (0, 16)
OLDALE = (0, 10)  # inferido
ROUTE_103 = (0, 18)  # inferido
PETALBURG = (0, 0)  # inferido
RUSTBORO = (0, 3)  # inferido

SPECIES_MARSHTOMP = 284  # índice interno de Gen III; Mudkip = 283 (verificado en vivo)


@dataclass
class Tracker:
    """Historial del episodio que no se puede deducir de un solo estado."""

    maps_visited: set[tuple[int, int]] = field(default_factory=set)
    trainer_battles_won: int = 0
    _prev: GameState | None = None

    def update(self, state: GameState) -> None:
        self.maps_visited.add((state.map_group, state.map_num))
        prev = self._prev
        if prev is not None and prev.in_battle and not state.in_battle:
            was_trainer = bool(prev.battle_type_flags & BATTLE_TYPE_TRAINER)
            if was_trainer and any(mon.hp > 0 for mon in state.party):
                self.trainer_battles_won += 1
        self._prev = state


Check = Callable[[GameState, Tracker], bool]


@dataclass(frozen=True)
class Milestone:
    id: int  # número del hito en specs.md, 9.1
    name: str
    reward: float
    check: Check | None  # None = todavía sin detector; nunca se dispara
    verified: bool  # True si el detector se probó en vivo


def _visited(map_id: tuple[int, int]) -> Check:
    return lambda s, t: map_id in t.maps_visited


TRAMO_1: tuple[Milestone, ...] = (
    Milestone(3, "starter conseguido", 1.0, lambda s, t: len(s.party) >= 1, True),
    Milestone(5, "laboratorio de Birch", 1.0, _visited(LAB), True),
    Milestone(6, "Oldale Town", 1.0, _visited(OLDALE), False),
    Milestone(
        7, "rival vencido en la Ruta 103", 1.0,
        lambda s, t: ROUTE_103 in t.maps_visited and t.trainer_battles_won >= 1, False,
    ),
    Milestone(8, "Pokédex y Poké Balls", 1.0, None, False),
    Milestone(9, "Petalburg City", 1.0, _visited(PETALBURG), False),
    Milestone(10, "Wally capturado", 1.0, None, False),
    Milestone(11, "recluta de Team Aqua vencido", 1.0, None, False),
    Milestone(12, "Rustboro City", 1.0, _visited(RUSTBORO), False),
    Milestone(13, "segundo Pokémon capturado", 1.0, lambda s, t: len(s.party) >= 2, False),
    Milestone(14, "starter evoluciona a Marshtomp", 1.0,
              lambda s, t: any(mon.species == SPECIES_MARSHTOMP for mon in s.party), False),
    Milestone(15, "gimnasio de Rustboro", 1.0, None, False),
    Milestone(16, "medalla 1 (Roxanne)", 5.0, lambda s, t: s.badges >= 1, False),
)
