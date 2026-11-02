"""Abre N instancias de mGBA listas para entrenar en paralelo: cada una con su copia de la ROM y del .sav, su puerto,
`emulator/server.lua` cargado y el avance rápido sin límite.

mGBA 0.10.5 no tiene opción de línea de comandos para scripts, así que `emulator/setup_instance.ps1` maneja sus menús
con UI Automation de Windows. Eso usa el teclado y el primer plano: mientras se prepara cada instancia (unos 10 s)
no toques el teclado ni el ratón. Las instancias se preparan una por una.

Uso:
    uv run python scripts/launch_emulators.py --n 4 --base-port 8888
    uv run python scripts/launch_emulators.py --stop

Las copias viven en `runs/instances/inst<i>/` (ignorado por git). La ROM y el .sav originales de `roms/` no se tocan.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emulator.client import MGBAClient  # noqa: E402

MGBA = Path(r"C:\Program Files\mGBA\mGBA.exe")
INSTANCES = ROOT / "runs" / "instances"
PIDS = INSTANCES / "pids.json"
SETUP = ROOT / "emulator" / "setup_instance.ps1"
SERVER_LUA = ROOT / "emulator" / "server.lua"


def find_rom() -> Path:
    roms = [p for p in (ROOT / "roms").iterdir() if p.suffix.lower() in (".zip", ".gba")]
    if not roms:
        sys.exit("no hay ROM en roms/")
    return roms[0]


def port_alive(port: int) -> bool:
    try:
        with MGBAClient(port=port, timeout=2) as emu:
            return emu.ping()
    except OSError:
        return False


def measure_fps(port: int, frames: int = 3000) -> float:
    with MGBAClient(port=port) as emu:
        t = time.time()
        emu.step(0, frames)
        return frames / (time.time() - t)


def launch_one(index: int, port: int, rom: Path) -> int:
    folder = INSTANCES / f"inst{index}"
    folder.mkdir(parents=True, exist_ok=True)
    rom_copy = folder / rom.name
    if not rom_copy.exists():
        shutil.copy2(rom, rom_copy)
    sav = rom.with_suffix(".sav")
    sav_copy = rom_copy.with_suffix(".sav")
    if sav.exists() and not sav_copy.exists():
        shutil.copy2(sav, sav_copy)
    env = {**os.environ, "POKEMON_IA_PORT": str(port)}
    proc = subprocess.Popen([str(MGBA), str(rom_copy)], env=env)
    return proc.pid


def setup_one(pid: int) -> None:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SETUP), "-ProcessId", str(pid),
         "-ScriptPath", str(SERVER_LUA)],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0 or "OK" not in out.stdout:
        raise RuntimeError(f"falló la preparación de la instancia pid {pid}: {out.stdout.strip()} {out.stderr.strip()}")


def stop() -> None:
    if not PIDS.exists():
        print("no hay instancias registradas")
        return
    for pid in json.loads(PIDS.read_text()).get("pids", []):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        print(f"cerrada la instancia pid {pid}")
    PIDS.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=4, help="cantidad de instancias")
    parser.add_argument("--base-port", type=int, default=8888)
    parser.add_argument("--stop", action="store_true", help="cerrar las instancias abiertas por este script")
    args = parser.parse_args()
    if args.stop:
        stop()
        return

    rom = find_rom()
    INSTANCES.mkdir(parents=True, exist_ok=True)
    pids: list[int] = json.loads(PIDS.read_text()).get("pids", []) if PIDS.exists() else []
    print(f"ROM: {rom.name}. No toques teclado ni ratón mientras se preparan las instancias.", flush=True)
    for i in range(args.n):
        port = args.base_port + i
        if port_alive(port):
            print(f"[{i}] el puerto {port} ya responde: se reutiliza", flush=True)
            continue
        pid = launch_one(i, port, rom)
        pids.append(pid)
        PIDS.write_text(json.dumps({"pids": pids}))
        time.sleep(3)  # que la ventana exista antes de manejar sus menús
        try:
            setup_one(pid)
        except Exception:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)  # no dejar diálogos abiertos
            pids.remove(pid)
            PIDS.write_text(json.dumps({"pids": pids}))
            raise
        for _ in range(30):
            if port_alive(port):
                break
            time.sleep(0.5)
        else:
            sys.exit(f"[{i}] el puerto {port} no responde tras cargar el script")
        print(f"[{i}] pid {pid} puerto {port}: {measure_fps(port):.0f} frames/s", flush=True)
    print(f"listas {args.n} instancias en los puertos {args.base_port} a {args.base_port + args.n - 1}")


if __name__ == "__main__":
    main()
