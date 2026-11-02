"""Cliente Python para el servidor Lua de mGBA (emulator/server.lua)."""

from __future__ import annotations

import socket
from enum import IntFlag


class Key(IntFlag):
    """Bits en el mismo orden que el registro KEYINPUT de la GBA."""

    A = 1 << 0
    B = 1 << 1
    SELECT = 1 << 2
    START = 1 << 3
    RIGHT = 1 << 4
    LEFT = 1 << 5
    UP = 1 << 6
    DOWN = 1 << 7
    R = 1 << 8
    L = 1 << 9


class EmulatorError(RuntimeError):
    pass


class MGBAClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8888, timeout: float = 30.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._buffer = b""
        self._has_multi = True

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> "MGBAClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _command(self, line: str) -> str:
        self._sock.sendall(line.encode() + b"\n")
        while b"\n" not in self._buffer:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise EmulatorError("el emulador cerró la conexión")
            self._buffer += chunk
        raw, self._buffer = self._buffer.split(b"\n", 1)
        text = raw.decode().strip()
        if text.startswith("ERR"):
            raise EmulatorError(text)
        return text[3:].strip()  # quita el prefijo "OK"

    def ping(self) -> bool:
        return self._command("ping") == "pong"

    def frame(self) -> int:
        return int(self._command("frame"))

    def read8(self, addr: int) -> int:
        return int(self._command(f"read8 {addr}"))

    def read16(self, addr: int) -> int:
        return int(self._command(f"read16 {addr}"))

    def read32(self, addr: int) -> int:
        return int(self._command(f"read32 {addr}"))

    def read_range(self, addr: int, length: int) -> bytes:
        return bytes.fromhex(self._command(f"readrange {addr} {length}"))

    def read_multi(self, ranges: list[tuple[int, int]]) -> tuple[int, list[bytes]]:
        """Lee varios rangos `(dirección, largo)` en una sola llamada. Devuelve `(frame, [bytes, ...])`.

        Si el servidor Lua cargado es una versión anterior sin `readmulti`, cae a una lectura por rango.
        """
        if self._has_multi:
            args = " ".join(f"{addr}:{length}" for addr, length in ranges)
            try:
                parts = self._command(f"readmulti {args}").split()
            except EmulatorError as err:
                if "desconocido" not in str(err):
                    raise
                self._has_multi = False
            else:
                return int(parts[0]), [bytes.fromhex(p) for p in parts[1:]]
        return self.frame(), [self.read_range(addr, length) for addr, length in ranges]

    def step(self, keys: Key | int = 0, frames: int = 1) -> int:
        """Mantiene `keys` pulsadas durante `frames` frames y devuelve el frame final."""
        return int(self._command(f"step {int(keys)} {frames}"))

    def save_state(self, path: str) -> None:
        self._command(f"savestate {path}")

    def load_state(self, path: str) -> None:
        self._command(f"loadstate {path}")

    def reset(self) -> None:
        """Reinicia la consola (arranca desde el logo). No borra el .sav."""
        self._command("reset")

    def reg(self, name: str) -> int:
        """Registro de la CPU (pc, lr, sp, r0..r15, cpsr); sirve para ver dónde está un cuelgue."""
        return int(self._command(f"reg {name}"))

    def screenshot(self, path: str) -> None:
        self._command(f"screenshot {path}")
