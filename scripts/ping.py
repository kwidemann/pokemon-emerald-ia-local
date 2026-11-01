"""Prueba del puente con mGBA: ping, avanzar frames, pulsar un botón y leer memoria.

Uso (con mGBA abierto, ROM cargada y emulator/server.lua cargado):
    uv run python scripts/ping.py [--port 8888]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emulator.client import Key, MGBAClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    with MGBAClient(args.host, args.port) as emu:
        print("ping:", emu.ping())
        start = emu.frame()
        print("frame inicial:", start)
        end = emu.step(Key.A, 30)
        print(f"step(A, 30) -> frame {end} (avanzó {end - start})")
        # 0x03000000 = inicio de IWRAM: solo comprueba que la lectura de memoria responde
        print("IWRAM[0x03000000] 16 bytes:", emu.read_range(0x03000000, 16).hex())


if __name__ == "__main__":
    main()
