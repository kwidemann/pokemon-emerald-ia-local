"""Imprime el estado actual del juego leído de la RAM (mGBA abierto con emulator/server.lua cargado).

Uso:
    uv run python scripts/inspect_state.py [--port 8888]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emulator.client import MGBAClient  # noqa: E402
from env.state import read_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    with MGBAClient(args.host, args.port) as emu:
        print(json.dumps(read_state(emu).to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
