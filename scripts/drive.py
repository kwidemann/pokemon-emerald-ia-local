"""Utilidades para manejar el juego por RAM + teclas (usadas para automatizar la intro).

Uso interactivo desde otros scripts:
    from scripts.drive import Driver
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emulator.client import Key, MGBAClient  # noqa: E402
from env.state import GameState, read_state  # noqa: E402

DIRS = {"up": Key.UP, "down": Key.DOWN, "left": Key.LEFT, "right": Key.RIGHT}


class Driver:
    def __init__(self, emu: MGBAClient):
        self.emu = emu

    def state(self) -> GameState:
        return read_state(self.emu)

    def press(self, key: Key, hold: int = 4, release: int = 8) -> None:
        self.emu.step(key, hold)
        self.emu.step(0, release)

    def advance_dialogue(self, max_presses: int = 80) -> int:
        """Pulsa A hasta que no haya script activo. Devuelve cuántas pulsaciones hicieron falta."""
        for n in range(max_presses):
            if not self.state().script_active:
                return n
            self.press(Key.A, hold=4, release=20)
        return max_presses

    def pos(self) -> tuple[int, int, int, int]:
        s = self.state()
        for _ in range(20):  # durante un warp la ubicación se lee como (0,0) en (0,0): esperar
            if not s.in_transition:
                break
            self.emu.step(0, 10)
            s = self.state()
        return s.map_group, s.map_num, s.x, s.y

    def move(self, direction: str, steps: int = 1) -> tuple[int, int, int, int]:
        """Avanza exactamente `steps` casillas.

        Medido en vivo: una pulsación corta (1-6 frames) mueve UNA casilla y la posición en RAM se
        actualiza unos frames después de soltar. Si el personaje no miraba hacia `direction`, la primera
        pulsación solo gira, así que se reintenta. Se detiene si cambia el mapa o aparece un script.
        """
        key = DIRS[direction]
        for _ in range(steps):
            before = self.pos()
            for _ in range(3):
                self.emu.step(key, 4)
                self.emu.step(0, 20)
                if self.pos() != before:
                    break
            now = self.pos()
            if now[:2] != before[:2] or now == before or self.state().script_active:
                break
        return self.pos()

    def tap(self, key: Key, wait: int = 30, hold: int = 4) -> None:
        self.emu.step(key, hold)
        self.emu.step(0, wait)

    def settle(self, max_presses: int = 300, quiet_frames: int = 90) -> int:
        """Avanza diálogos hasta que no haya script activo durante `quiet_frames` seguidos.

        Un script puede dejar `script_active` en False unos frames y volver a arrancar (p. ej. mamá
        caminando entre frases), por eso se espera y se vuelve a comprobar. Devuelve las pulsaciones usadas.
        """
        used = 0
        while used < max_presses:
            used += self.advance_dialogue(max_presses - used)
            self.emu.step(0, quiet_frames)
            if not self.state().script_active:
                return used
        raise TimeoutError(f"el script no terminó tras {max_presses} pulsaciones; estado: {self.state().to_dict()}")

    def move_until_map_change(self, direction: str, max_steps: int = 12) -> tuple[int, int, int, int]:
        """Sigue en una dirección hasta cruzar a otro mapa (puertas, escaleras, salidas del pueblo)."""
        start = self.pos()[:2]
        for _ in range(max_steps):
            now = self.move(direction, 1)
            if now[:2] != start:
                self.emu.step(0, 60)  # deja terminar el fundido
                return self.pos()
            if self.state().script_active:
                return now
        raise TimeoutError(f"no se cambió de mapa yendo '{direction}' desde {start}")

    def walk(self, direction: str, steps: int = 1) -> tuple[int, int]:
        """Camina `steps` casillas (16 frames por casilla; el primer toque solo gira si no miraba hacia allí)."""
        key = DIRS[direction]
        for _ in range(steps):
            self.emu.step(key, 16)
        self.emu.step(0, 8)
        s = self.state()
        return s.x, s.y
