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

BATTLE_TYPE_TRAINER = 0x08  # verificado en vivo: el combate con el rival dio flags 0xC = IS_MASTER | TRAINER
BATTLE_TYPE_WALLY_TUTORIAL = 0x200  # visto en vivo: la escena de Wally dio flags 0x204 (el bit exacto es de memoria)
WOODS = (24, 11)  # Bosque Petalburgo (del grafo de la ROM; se cruzó en vivo)

# (grupo, número) de mapa. Verificados en vivo: LAB, LITTLEROOT, ROUTE_101, OLDALE, ROUTE_103, PETALBURG, RUSTBORO.
LAB = (1, 4)
LITTLEROOT = (0, 9)
ROUTE_101 = (0, 16)
OLDALE = (0, 10)  # verificado en vivo (hito 6)
ROUTE_103 = (0, 18)  # verificado en vivo (hito 7)
PETALBURG = (0, 0)  # verificado en vivo (hito 9)
RUSTBORO = (0, 3)  # verificado en vivo (hito 12)
RUSTBORO_GYM = (11, 3)  # verificado en vivo (hito 15)

# Tope de nivel del premio por experiencia (R5f): el primer hito sin cumplir de esta lista fija hasta qué nivel del líder se
# premia ganar experiencia. Los niveles son los de los savestates de `runs/etapas/` (e3 = 8..9 antes del rival, e6 = 11, e8 = 13,
# e10 = 14); tras el gimnasio (hito 15) hasta el 16, donde evoluciona Mudkip (hito 14) y se espera pelear con Roxanne.
EXP_LEVEL_CAPS: tuple[tuple[int, int], ...] = ((7, 9), (9, 11), (12, 13), (15, 14), (16, 16))


def exp_level_cap(achieved) -> int:
    """Nivel hasta el que se premia la experiencia, según los hitos ya cumplidos (`achieved`: contenedor de ids)."""
    for ms_id, cap in EXP_LEVEL_CAPS:
        if ms_id not in achieved:
            return cap
    return EXP_LEVEL_CAPS[-1][1]


def exp_medium_slow(level: int) -> int:
    """Experiencia total de un Pokémon del grupo «medio lento» (Mudkip, Torchic, Treecko y sus evoluciones) en ese nivel."""
    return max(0, int(6 * level**3 / 5 - 15 * level**2 + 100 * level - 140))


RIVAL_TILE = (10, 3)  # casilla del rival en la Ruta 103 (verificado en vivo: NPC 240; se le habla desde una casilla vecina)

# Meta del premio por acercarse (R3): el primer par cuyo hito aún no se ha cumplido manda. Se omite el hito 5 (laboratorio): los
# puntos de partida de entrenamiento ya lo cumplieron en la partida real y volver atrás sería un desvío. Los hitos sin lugar
# (8, 10, 11, 13, 14, 16) no tienen meta: tras 15 no queda ninguna y el premio vale 0.
SHAPING_GOALS: tuple[tuple[int, tuple[int, int]], ...] = (
    (6, OLDALE), (7, ROUTE_103), (9, PETALBURG), (12, RUSTBORO), (15, RUSTBORO_GYM),
)

FLAG_SYS_POKEDEX_GET = 0x861  # verificada en vivo: se activó al hablar con Birch tras vencer al rival
SPECIES_MARSHTOMP = 284  # índice interno de Gen III; Mudkip = 283. Verificado en vivo: evolucionó en el combate con Roxanne


@dataclass
class Tracker:
    """Historial del episodio que no se puede deducir de un solo estado."""

    maps_visited: set[tuple[int, int]] = field(default_factory=set)
    trainer_battles_won: int = 0
    trainer_wins_on: set[tuple[int, int]] = field(default_factory=set)  # mapas donde se ganó a un entrenador
    wally_tutorial_seen: bool = False
    _prev: GameState | None = None

    def update(self, state: GameState) -> None:
        here = (state.map_group, state.map_num)
        # Durante un warp el mapa se lee como (0,0) en un paso (ver GameState.in_transition) y Petalburg es (0,0): en
        # entrenamiento dio el hito 9 en 2 de 99 episodios sin haber salido de Littleroot. (0,0) solo cuenta tras dos
        # lecturas seguidas.
        prev = self._prev
        if here != (0, 0) or (prev is not None and (prev.map_group, prev.map_num) == (0, 0)):
            self.maps_visited.add(here)
        if state.in_battle and state.battle_type_flags & BATTLE_TYPE_WALLY_TUTORIAL:
            self.wally_tutorial_seen = True
        prev = self._prev
        if prev is not None and prev.in_battle and not state.in_battle:
            was_trainer = bool(prev.battle_type_flags & BATTLE_TYPE_TRAINER)
            if was_trainer and any(mon.hp > 0 for mon in state.party):
                self.trainer_battles_won += 1
                self.trainer_wins_on.add((prev.map_group, prev.map_num))
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
    Milestone(6, "Oldale Town", 1.0, _visited(OLDALE), True),
    Milestone(
        7, "rival vencido en la Ruta 103", 1.0,
        lambda s, t: ROUTE_103 in t.maps_visited and t.trainer_battles_won >= 1, True,
    ),
    Milestone(8, "Pokédex recibida", 1.0, lambda s, t: s.flag(FLAG_SYS_POKEDEX_GET), True),
    Milestone(9, "Petalburg City", 1.0, _visited(PETALBURG), True),
    Milestone(10, "tutorial de captura de Wally", 1.0, lambda s, t: t.wally_tutorial_seen, True),
    Milestone(11, "entrenador vencido en el Bosque Petalburgo", 1.0, lambda s, t: WOODS in t.trainer_wins_on, True),
    Milestone(12, "Rustboro City", 1.0, _visited(RUSTBORO), True),
    Milestone(13, "segundo Pokémon capturado", 1.0, lambda s, t: len(s.party) >= 2, True),
    Milestone(14, "starter evoluciona a Marshtomp", 1.0,
              lambda s, t: any(mon.species == SPECIES_MARSHTOMP for mon in s.party), True),
    Milestone(15, "gimnasio de Rustboro", 1.0, _visited(RUSTBORO_GYM), True),
    Milestone(16, "medalla 1 (Roxanne)", 5.0, lambda s, t: s.badges >= 1, True),
)


RIVAL_ENCOUNTER_ID = 17  # hito extra y opcional (no está en specs.md): ver `rival_encounter`


def rival_encounter(reward: float) -> Milestone:
    """Hito opcional para entrenar la etapa del rival: se cumple al empezar un combate de entrenador en la Ruta 103."""
    return Milestone(
        RIVAL_ENCOUNTER_ID, "combate con el rival iniciado", reward,
        lambda s, t: (s.map_group, s.map_num) == ROUTE_103 and s.in_battle and bool(s.battle_type_flags & BATTLE_TYPE_TRAINER),
        False,
    )


class MilestoneMonitor:
    """Sigue los hitos de una ejecución. Un hito solo cuenta cuando su condición se cumple en `confirm` lecturas
    seguidas: una ejecución de evaluación dio por buena la medalla 1 en Petalburg con un Mudkip de nivel 10, por una
    lectura puntual de banderas que no era real (causa no identificada)."""

    def __init__(self, milestones: tuple[Milestone, ...] = TRAMO_1, confirm: int = 3):
        self.milestones = milestones
        self.confirm = confirm
        self.tracker = Tracker()
        self.achieved: dict[int, int] = {}  # id -> frame en que se confirmó
        self._streak: dict[int, int] = {}

    def update(self, state: GameState) -> list[Milestone]:
        """Procesa un estado; devuelve los hitos confirmados con él."""
        if state.in_transition:
            return []
        self.tracker.update(state)
        new = []
        for ms in self.milestones:
            if ms.id in self.achieved or ms.check is None:
                continue
            self._streak[ms.id] = self._streak.get(ms.id, 0) + 1 if ms.check(state, self.tracker) else 0
            if self._streak[ms.id] >= self.confirm:
                self.achieved[ms.id] = state.frame
                new.append(ms)
        return new
