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
from env.milestones import BATTLE_TYPE_TRAINER, TRAMO_1, Milestone, MilestoneMonitor
from env.state import GameState, read_state

# Acción discreta -> tecla. NOOP deja pasar el tiempo (útil para animaciones y texto).
ACTIONS: tuple[Key | int, ...] = (Key.A, Key.B, Key.UP, Key.DOWN, Key.LEFT, Key.RIGHT, Key.START, 0)
ACTION_NAMES = ("A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "NOOP")
DIRECTIONS = {2, 3, 4, 5}  # índices de UP, DOWN, LEFT, RIGHT en ACTIONS
FACING_OF_ACTION = {2: 2, 3: 1, 4: 3, 5: 4}  # acción -> código de GameState.facing (1 abajo, 2 arriba, 3 izq., 4 der.)

REVISIT_WINDOW = 30  # posiciones recordadas para detectar vueltas (solo para contar; el castigo se calibra aparte)

OBS_SIZE = 12


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

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self.confirm = confirm
        self._monitor = MilestoneMonitor(milestones, confirm)
        self._tiles: set[tuple[int, int, int, int]] = set()
        self._steps = 0
        self._state: GameState | None = None
        self._history: deque = deque(maxlen=REVISIT_WINDOW)
        self.counters = self._new_counters()

    @staticmethod
    def _new_counters() -> dict[str, int]:
        keys = ("choques", "giros", "quieto", "bucles", "rev3", "rev10", "rev30", "menu", "dialogo", "batalla")
        return dict.fromkeys(keys, 0)

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
            return 0.0
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
                    return 0.0
                c["choques"] += 1
                return -self.bump_penalty
            c["quieto"] += 1
            return -self.idle_penalty
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
        return encode_observation(state), info

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
        self._state = state
        return self._observe(state)

    def step(self, action: int):
        self.emu.step(ACTIONS[int(action)], self.hold_frames)
        self.emu.step(0, self.wait_frames)
        self._steps += 1

        state = read_state(self.emu)

        reward = 0.0
        if not state.in_transition:  # en un warp la ubicación es basura: no cuenta para hitos ni historial
            for ms in self._monitor.update(state):
                reward += ms.reward
            tile = (state.map_group, state.map_num, state.x, state.y)
            if self.new_tile_bonus and tile not in self._tiles:
                reward += self.new_tile_bonus
            self._tiles.add(tile)
            reward += self._count(self._state, state, int(action))

        terminated = self.terminal_milestone is not None and self.terminal_milestone in self._monitor.achieved
        truncated = self._steps >= self.max_steps
        self._state = state
        obs, info = self._observe(state)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.emu.close()
