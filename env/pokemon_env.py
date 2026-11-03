"""Entorno Gymnasium sobre mGBA: una acción = una pulsación corta + espera (~1 casilla de movimiento).

Requiere mGBA abierto con `emulator/server.lua` cargado y un savestate de partida.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from emulator.client import Key, MGBAClient
from env.distance import distance_field, get_graph
from env.milestones import BATTLE_TYPE_TRAINER, SHAPING_GOALS, TRAMO_1, Milestone, MilestoneMonitor
from env.state import GameState, read_state

# Acción discreta -> tecla. NOOP deja pasar el tiempo (útil para animaciones y texto).
ACTIONS: tuple[Key | int, ...] = (Key.A, Key.B, Key.UP, Key.DOWN, Key.LEFT, Key.RIGHT, Key.START, 0)
ACTION_NAMES = ("A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "NOOP")
DIRECTIONS = {2, 3, 4, 5}  # índices de UP, DOWN, LEFT, RIGHT en ACTIONS
FACING_OF_ACTION = {2: 2, 3: 1, 4: 3, 5: 4}  # acción -> código de GameState.facing (1 abajo, 2 arriba, 3 izq., 4 der.)

REVISIT_WINDOW = 30  # posiciones recordadas para detectar vueltas (solo para contar; el castigo se calibra aparte)

OBS_SIZE = 12

# Bloques opcionales de la observación (se añaden detrás de los 12 valores básicos; ver `PokemonEmeraldEnv`).
TERRAIN_RADIUS = 2  # parche de (2*radio+1)^2 casillas alrededor del jugador, con el norte arriba
TERRAIN_KINDS = ("libre", "puerta", "desnivel_down", "desnivel_up", "desnivel_left", "desnivel_right")
TERRAIN_PATCH = (2 * TERRAIN_RADIUS + 1) ** 2
LAST_ACTIONS = 4  # acciones recientes que se recuerdan, en one-hot
MEMORY_SIZE = TERRAIN_PATCH + LAST_ACTIONS * len(ACTIONS) + 4 + 4  # visitadas, acciones, orientación, resultado del último paso
_TERRAIN_CACHE: dict[tuple[int, int, int, int], np.ndarray] = {}


def observation_size(terrain: bool = False, memory: bool = False) -> int:
    return OBS_SIZE + (len(TERRAIN_KINDS) * TERRAIN_PATCH if terrain else 0) + (MEMORY_SIZE if memory else 0)


def terrain_patch(map_group: int, map_num: int, x: int, y: int) -> np.ndarray:
    """Clase de cada casilla del parche (one-hot por clase, forma `(clases, lado, lado)`), leída de la ROM y en caché."""
    key = (map_group, map_num, x, y)
    if key not in _TERRAIN_CACHE:
        graph = get_graph()
        side = 2 * TERRAIN_RADIUS + 1
        out = np.zeros((len(TERRAIN_KINDS), side, side), dtype=np.float32)
        for j in range(side):
            for i in range(side):
                kind = graph.tile_kind((map_group, map_num), x + i - TERRAIN_RADIUS, y + j - TERRAIN_RADIUS)
                if kind in TERRAIN_KINDS:
                    out[TERRAIN_KINDS.index(kind), j, i] = 1.0
        _TERRAIN_CACHE[key] = out
    return _TERRAIN_CACHE[key]


def encode_observation(s: GameState) -> np.ndarray:
    """Vector de 12 valores aproximadamente en [0, 1]. Los IDs de mapa van escalados, no en one-hot.

    Es una codificación provisional: los IDs discretos escalados dan una falsa noción de cercanía
    (riesgo anotado en specs.md, 5.2). Se cambiará cuando se mida su efecto.
    """
    lead = s.party[0] if s.party else None
    return np.array(
        [
            s.map_group / 32,
            s.map_num / 128,
            s.x / 64,
            s.y / 64,
            float(s.script_active),
            float(s.in_battle),
            float(bool(s.battle_type_flags & BATTLE_TYPE_TRAINER)),
            len(s.party) / 6,
            (lead.hp / lead.max_hp) if lead and lead.max_hp else 0.0,
            (lead.level / 100) if lead else 0.0,
            s.badges / 8,
            math.log1p(s.money) / 12,
        ],
        dtype=np.float32,
    )


class PokemonEmeraldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        savestate: str | Path,
        host: str = "127.0.0.1",
        port: int = 8888,
        hold_frames: int = 4,
        wait_frames: int = 16,
        max_steps: int = 2000,
        milestones: tuple[Milestone, ...] = TRAMO_1,
        terminal_milestone: int | None = 16,
        new_tile_bonus: float = 0.0,
        confirm: int = 3,
        bump_penalty: float = 0.0,
        idle_penalty: float = 0.0,
        loop_penalty: float = 0.0,
        menu_penalty: float = 0.0,
        shaping_coef: float = 0.0,
        track_distance: bool = False,
        bump_escalation_cap: int = 1,
        turn_penalty: float = 0.0,
        dialog_penalty: float = 0.0,
        obs_terrain: bool = False,
        obs_memory: bool = False,
    ):
        """
        hold_frames/wait_frames: medido en vivo, una pulsación de 1-6 frames mueve exactamente una casilla.
        terminal_milestone: id del hito que termina el episodio con éxito (None = nunca).
        new_tile_bonus: recompensa de ayuda por pisar una casilla nueva (0 = solo hitos).
        confirm: lecturas seguidas que necesita un hito para contarse (`MilestoneMonitor`): una lectura puntual falsa
            de banderas ya dio por buena una medalla que no existía. La recompensa llega `confirm - 1` pasos tarde.
        bump_penalty, idle_penalty, loop_penalty, menu_penalty: castigos (se restan; 0 = desactivado). Los contadores de
            `info["contadores"]` se llevan siempre, con o sin castigo, para poder calibrarlos:
            - choques: se apretó la dirección hacia la que ya miraba el personaje y no se movió (pared, desnivel, NPC...).
            - giros: se apretó una dirección distinta de hacia la que miraba: la pulsación solo gira (no se castiga).
            - quieto: se controla al personaje y no se movió con una tecla que no es dirección (A, B, START o nada).
            - bucles: volver a una casilla que ya se pisó 2 veces o más en las últimas `REVISIT_WINDOW` casillas; así volver
              a un sitio a propósito (Centro Pokémon, un puente) no cuenta, pero dar vueltas en el mismo lugar sí.
            - rev3, rev10, rev30: volver a una de las últimas 3, 10 o 30 casillas (para ver qué ventana separa mejor).
            - menu, dialogo, batalla: pasos en cada tipo de pantalla.
        shaping_coef: premio por acercarse a la meta del siguiente hito (`SHAPING_GOALS`): `coef * (distancia anterior -
            distancia nueva)`, con la distancia a pie de `env/distance.py`; cada paso se limita a +-3 casillas. Es una función
            potencial: ir y volver suma cero. 0 = desactivado.
        bump_escalation_cap: el castigo por choque se multiplica por los choques seguidos en la misma casilla (la racha se
            reinicia al moverse a otra), hasta este tope: con 10, el décimo choque seguido cuesta 10 veces `bump_penalty`.
            1 = castigo fijo. El contador `racha_max` guarda la racha más larga del episodio.
        turn_penalty: castigo fijo por cada giro en el lugar (la pulsación solo gira al personaje; 0 = desactivado). No
            reinicia ni aumenta la racha de choques.
        dialog_penalty: castigo fijo por cada paso con un diálogo o script activo (0 = desactivado). Sin él, quedarse en un
            diálogo es gratis y el agente puede esconderse ahí cuando los demás castigos superan los premios (ver R3c).
        track_distance: lleva los contadores de distancia (`dist_ini`, `dist_min`, `dist_fin`: casillas que faltan hasta la
            meta) aunque no haya premio; se activa solo si `shaping_coef > 0`.
        obs_terrain: añade a la observación el terreno alrededor del jugador: un parche de 5x5 casillas con el norte arriba
            y 6 clases en one-hot (libre, puerta, desnivel hacia cada dirección; todo lo demás es 0), leído de la ROM. Así
            se ve una pared o un desnivel antes de chocar. No incluye NPC, árboles ni rocas.
        obs_memory: añade a la observación memoria corta de lo que pasó: casillas del parche ya pisadas en el episodio (25),
            las últimas 4 acciones en one-hot (32), hacia dónde mira el personaje (4) y el resultado del último paso:
            se movió, chocó, giró y racha de choques seguidos / 10 (4).
        Con las dos opciones en False la observación es la original de 12 valores (los modelos anteriores siguen sirviendo).
        """
        super().__init__()
        # mGBA resuelve las rutas relativas desde su propio directorio, no el nuestro.
        self.savestate = Path(savestate).resolve().as_posix()
        self.emu = MGBAClient(host, port)
        self.hold_frames = hold_frames
        self.wait_frames = wait_frames
        self.max_steps = max_steps
        self.milestones = milestones
        self.terminal_milestone = terminal_milestone
        self.new_tile_bonus = new_tile_bonus
        self.bump_penalty = bump_penalty
        self.idle_penalty = idle_penalty
        self.loop_penalty = loop_penalty
        self.menu_penalty = menu_penalty
        self.shaping_coef = shaping_coef
        self.bump_escalation_cap = max(1, bump_escalation_cap)
        self.turn_penalty = turn_penalty
        self.dialog_penalty = dialog_penalty
        self._bump_streak = 0
        self.track_distance = track_distance or shaping_coef > 0
        self._last_pos: tuple[int, int, int, int] | None = None  # última posición válida (no de un warp)

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.obs_terrain = obs_terrain
        self.obs_memory = obs_memory
        self._last_actions: deque = deque(maxlen=LAST_ACTIONS)
        self._outcome = (0.0, 0.0, 0.0)  # se movió, chocó, giró en el último paso
        self.observation_space = spaces.Box(0.0, 1.0, shape=(observation_size(obs_terrain, obs_memory),), dtype=np.float32)

        self.confirm = confirm
        self._monitor = MilestoneMonitor(milestones, confirm)
        self._tiles: set[tuple[int, int, int, int]] = set()
        self._steps = 0
        self._state: GameState | None = None
        self._history: deque = deque(maxlen=REVISIT_WINDOW)
        self.counters = self._new_counters()

    @staticmethod
    def _new_counters() -> dict[str, int]:
        keys = ("choques", "giros", "quieto", "bucles", "rev3", "rev10", "rev30", "menu", "dialogo", "batalla", "racha_max")
        return dict.fromkeys(keys, 0)

    def _goal_field(self) -> dict | None:
        """Campo de distancias hacia la meta del siguiente hito sin cumplir, o None si no queda ninguna."""
        for ms_id, goal in SHAPING_GOALS:
            if ms_id not in self._monitor.achieved:
                return distance_field(goal)
        return None

    def _shape(self, state: GameState) -> float:
        """Premio por acercarse a la meta (o 0). También actualiza `dist_ini`, `dist_min` y `dist_fin`."""
        pos = (state.map_group, state.map_num, state.x, state.y)
        prev, self._last_pos = self._last_pos, pos
        goal_field = self._goal_field()
        if goal_field is None or pos not in goal_field:
            return 0.0
        now = goal_field[pos]
        c = self.counters
        if "dist_ini" not in c:
            c["dist_ini"] = c["dist_min"] = now
        c["dist_min"] = min(c["dist_min"], now)
        c["dist_fin"] = now
        if prev is None or prev not in goal_field:
            return 0.0
        return self.shaping_coef * max(-3, min(3, goal_field[prev] - now))

    @staticmethod
    def _controls_player(s: GameState) -> bool:
        return not s.in_transition and s.in_overworld and not s.in_battle and not s.script_active

    def _count(self, prev: GameState | None, state: GameState, action: int) -> float:
        """Actualiza los contadores y devuelve el castigo (negativo) que corresponda, 0 si están desactivados."""
        c = self.counters
        if state.in_battle:
            c["batalla"] += 1
            return 0.0
        if state.script_active:
            c["dialogo"] += 1
            return -self.dialog_penalty
        if not state.in_overworld:
            c["menu"] += 1
            return -self.menu_penalty
        if prev is None or not self._controls_player(prev):
            return 0.0  # primer paso tras un menú, un diálogo o una batalla: no se sabe de dónde se partía
        before = (prev.map_group, prev.map_num, prev.x, prev.y)
        here = (state.map_group, state.map_num, state.x, state.y)
        if here == before:
            if action in DIRECTIONS:
                if prev.facing and prev.facing != FACING_OF_ACTION[action]:
                    c["giros"] += 1  # miraba hacia otro lado: la pulsación solo gira al personaje
                    self._outcome = (0.0, 0.0, 1.0)
                    return -self.turn_penalty
                c["choques"] += 1
                self._outcome = (0.0, 1.0, 0.0)
                self._bump_streak += 1
                c["racha_max"] = max(c["racha_max"], self._bump_streak)
                return -self.bump_penalty * min(self._bump_streak, self.bump_escalation_cap)
            c["quieto"] += 1
            return -self.idle_penalty
        self._bump_streak = 0  # se movió a otra casilla
        self._outcome = (1.0, 0.0, 0.0)
        recent = list(self._history)
        penalty = 0.0
        if here in recent[-3:]:
            c["rev3"] += 1
        if here in recent[-10:]:
            c["rev10"] += 1
        if here in recent:
            c["rev30"] += 1
        if recent.count(here) >= 2:
            c["bucles"] += 1
            penalty -= self.loop_penalty
        self._history.append(here)
        return penalty

    def _observe(self, state: GameState) -> tuple[np.ndarray, dict]:
        info = {
            "state": state.to_dict(),
            "milestones": sorted(self._monitor.achieved),
            "steps": self._steps,
            "contadores": {**self.counters, "casillas": len(self._tiles)},
        }
        return self._encode(state), info

    def _memory_block(self, state: GameState) -> np.ndarray:
        side = 2 * TERRAIN_RADIUS + 1
        visited = np.zeros((side, side), dtype=np.float32)
        for j in range(side):
            for i in range(side):
                tile = (state.map_group, state.map_num, state.x + i - TERRAIN_RADIUS, state.y + j - TERRAIN_RADIUS)
                visited[j, i] = tile in self._tiles
        actions = np.zeros((LAST_ACTIONS, len(ACTIONS)), dtype=np.float32)
        for k, a in enumerate(reversed(self._last_actions)):  # fila 0 = la más reciente
            actions[k, a] = 1.0
        facing = np.zeros(4, dtype=np.float32)
        if 1 <= state.facing <= 4:
            facing[state.facing - 1] = 1.0
        streak = min(self._bump_streak, 10) / 10
        return np.concatenate([visited.ravel(), actions.ravel(), facing, np.array([*self._outcome, streak], dtype=np.float32)])

    def _encode(self, state: GameState) -> np.ndarray:
        parts = [encode_observation(state)]
        if self.obs_terrain:
            parts.append(terrain_patch(state.map_group, state.map_num, state.x, state.y).ravel())
        if self.obs_memory:
            parts.append(self._memory_block(state))
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.emu.load_state(self.savestate)
        self.emu.step(0, 2)  # deja que la RAM se asiente tras cargar
        self._steps = 0
        self._monitor = MilestoneMonitor(self.milestones, self.confirm)
        self._tiles = set()
        state = read_state(self.emu)
        # Lo que ya se cumple en el savestate no da recompensa: el mismo estado se procesa `confirm` veces.
        for _ in range(self.confirm):
            self._monitor.update(state)
        self._tiles.add((state.map_group, state.map_num, state.x, state.y))
        self._history = deque([(state.map_group, state.map_num, state.x, state.y)], maxlen=REVISIT_WINDOW)
        self.counters = self._new_counters()
        self._last_pos = None
        self._bump_streak = 0
        self._last_actions.clear()
        self._outcome = (0.0, 0.0, 0.0)
        if self.track_distance:
            self._shape(state)
        self._state = state
        return self._observe(state)

    def step(self, action: int):
        self.emu.step(ACTIONS[int(action)], self.hold_frames)
        self.emu.step(0, self.wait_frames)
        self._steps += 1

        state = read_state(self.emu)

        reward = 0.0
        self._last_actions.append(int(action))
        self._outcome = (0.0, 0.0, 0.0)
        if not state.in_transition:  # en un warp la ubicación es basura: no cuenta para hitos ni historial
            for ms in self._monitor.update(state):
                reward += ms.reward
            tile = (state.map_group, state.map_num, state.x, state.y)
            if self.new_tile_bonus and tile not in self._tiles:
                reward += self.new_tile_bonus
            self._tiles.add(tile)
            reward += self._count(self._state, state, int(action))
            if self.track_distance:
                reward += self._shape(state)

        terminated = self.terminal_milestone is not None and self.terminal_milestone in self._monitor.achieved
        truncated = self._steps >= self.max_steps
        self._state = state
        obs, info = self._observe(state)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.emu.close()
