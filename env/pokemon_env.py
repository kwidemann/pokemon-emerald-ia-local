"""Entorno Gymnasium sobre mGBA: una acción = una pulsación corta + espera (~1 casilla de movimiento).

Requiere mGBA abierto con `emulator/server.lua` cargado y un savestate de partida.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from emulator.client import Key, MGBAClient
from env.milestones import BATTLE_TYPE_TRAINER, TRAMO_1, Milestone, Tracker
from env.state import GameState, read_state

# Acción discreta -> tecla. NOOP deja pasar el tiempo (útil para animaciones y texto).
ACTIONS: tuple[Key | int, ...] = (Key.A, Key.B, Key.UP, Key.DOWN, Key.LEFT, Key.RIGHT, Key.START, 0)
ACTION_NAMES = ("A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "NOOP")

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
    ):
        """
        hold_frames/wait_frames: medido en vivo, una pulsación de 1-6 frames mueve exactamente una casilla.
        terminal_milestone: id del hito que termina el episodio con éxito (None = nunca).
        new_tile_bonus: recompensa de ayuda por pisar una casilla nueva (0 = solo hitos).
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

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self._tracker = Tracker()
        self._achieved: set[int] = set()
        self._tiles: set[tuple[int, int, int, int]] = set()
        self._steps = 0
        self._state: GameState | None = None

    def _observe(self, state: GameState) -> tuple[np.ndarray, dict]:
        info = {
            "state": state.to_dict(),
            "milestones": sorted(self._achieved),
            "steps": self._steps,
        }
        return encode_observation(state), info

    def _new_milestones(self, state: GameState) -> list[Milestone]:
        return [
            ms
            for ms in self.milestones
            if ms.id not in self._achieved and ms.check is not None and ms.check(state, self._tracker)
        ]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.emu.load_state(self.savestate)
        self.emu.step(0, 2)  # deja que la RAM se asiente tras cargar
        self._steps = 0
        self._tracker = Tracker()
        self._tiles = set()
        state = read_state(self.emu)
        self._tracker.update(state)
        # Lo que ya se cumple en el savestate no da recompensa.
        self._achieved = {ms.id for ms in self._new_milestones(state)}
        self._tiles.add((state.map_group, state.map_num, state.x, state.y))
        self._state = state
        return self._observe(state)

    def step(self, action: int):
        self.emu.step(ACTIONS[int(action)], self.hold_frames)
        self.emu.step(0, self.wait_frames)
        self._steps += 1

        state = read_state(self.emu)

        reward = 0.0
        if not state.in_transition:  # en un warp la ubicación es basura: no cuenta para hitos ni historial
            self._tracker.update(state)
            for ms in self._new_milestones(state):
                self._achieved.add(ms.id)
                reward += ms.reward
            tile = (state.map_group, state.map_num, state.x, state.y)
            if self.new_tile_bonus and tile not in self._tiles:
                reward += self.new_tile_bonus
            self._tiles.add(tile)

        terminated = self.terminal_milestone is not None and self.terminal_milestone in self._achieved
        truncated = self._steps >= self.max_steps
        self._state = state
        obs, info = self._observe(state)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.emu.close()
