"""Genera el savestate inicial: juega la intro de Esmeralda desde el logo hasta el laboratorio de Birch.

Reinicia la consola, elige NUEVO JUEGO (chico, nombre "AAAA"), pone el reloj, visita al vecino y al
rival, rescata a Birch eligiendo a Mudkip, gana la primera batalla y guarda el savestate en el laboratorio.

No guarda dentro del juego, así que el .sav de `roms/` no se modifica (mGBA sí lo reescribe al cargar
savestates). Cada tramo se comprueba contra la RAM y el script se detiene con un error si algo no coincide.

Uso (mGBA abierto con emulator/server.lua cargado y el avance rápido activado):
    uv run python scripts/make_savestate.py [--out runs/lab_mudkip.ss0]

El resultado NO es idéntico bit a bit entre ejecuciones: entre comandos el emulador sigue corriendo
(a velocidad libre), así que el RNG del juego cambia (personalidad del Mudkip, combate).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from emulator.client import Key, MGBAClient  # noqa: E402
from scripts.drive import Driver  # noqa: E402

# Valores de gMain.callback2 observados en vivo (ROM verificada). Solo los dos que distinguen
# pantallas que comparten el mismo callback con el menú.
G_MAIN_CALLBACK2 = 0x030022C4
CB2_NAMING_SCREEN = 0x080E4F59
CB2_OVERWORLD = 0x08085E5D
CB2_CLOCK_SCREEN = 0x08134C9D

MAP_TRUCK = (25, 40)
MAP_LITTLEROOT = (0, 9)
MAP_HOUSE_1F = (1, 0)
MAP_BEDROOM = (1, 1)
MAP_NEIGHBOR_1F = (1, 2)
MAP_NEIGHBOR_ROOM = (1, 3)
MAP_LAB = (1, 4)
MAP_ROUTE_101 = (0, 16)

SPECIES_MUDKIP = 283
PLAYER_NAME = "AAAA"


def check(cond: bool, msg: str, d: Driver) -> None:
    if not cond:
        s = d.state().to_dict()
        s["party"] = f"{len(s['party'])} Pokémon"
        raise RuntimeError(f"{msg}\nestado: {s}")


def at(d: Driver, map_id: tuple[int, int], what: str) -> tuple[int, int, int, int]:
    pos = d.pos()
    check(pos[:2] == map_id, f"se esperaba el mapa {map_id} ({what}) y se está en {pos}", d)
    return pos


def screenshot_pixels(emu: MGBAClient) -> Image.Image:
    path = Path(tempfile.gettempdir(), "pokemon_ia_boot.png").as_posix()
    if " " in path:
        raise RuntimeError(f"la ruta temporal tiene espacios ({path}); mGBA las cortaría")
    emu.screenshot(path)
    return Image.open(path).convert("RGB")


def is_main_menu(img: Image.Image) -> bool:
    """El menú principal tiene el fondo lila (140,148,255) en las esquinas inferiores."""
    return all(abs(a - b) <= 8 for p in ((235, 150), (5, 150)) for a, b in zip(img.getpixel(p), (140, 148, 255)))


def has_continue(img: Image.Image) -> bool:
    """Con una partida guardada aparece CONTINUE: un panel blanco alto en la parte superior."""
    return img.getpixel((100, 45)) == (255, 255, 255)


def yes_no_box_visible(img: Image.Image) -> bool:
    """El cuadro SÍ/NO de la pantalla del reloj es blanco en la esquina derecha; sin él es verde azulado."""
    return img.getpixel((230, 85)) == (255, 255, 255) and img.getpixel((228, 100)) == (255, 255, 255)


def boot_to_new_game(d: Driver) -> None:
    emu = d.emu
    emu.reset()
    emu.step(0, 120)
    for _ in range(30):  # START/A saltan el logo y el título hasta el menú principal
        d.tap(Key.START, wait=90)
        if is_main_menu(screenshot_pixels(emu)):
            break
    else:
        raise RuntimeError("no se llegó al menú principal")
    if has_continue(screenshot_pixels(emu)):
        d.tap(Key.DOWN, wait=20)  # CONTINUE -> NEW GAME
    d.tap(Key.A, wait=60)


def create_character(d: Driver) -> None:
    emu = d.emu
    # Discurso de Birch y "¿chico o chica?" (el cursor empieza en CHICO): A hasta la pantalla de nombre.
    for _ in range(80):
        if emu.read32(G_MAIN_CALLBACK2) & ~1 == CB2_NAMING_SCREEN & ~1:
            break
        d.tap(Key.A, wait=40)
    else:
        raise RuntimeError("no apareció la pantalla de nombre")
    emu.step(0, 90)  # deja terminar el fundido: una pulsación antes se pierde (salió "AAA")
    for _ in range(len(PLAYER_NAME)):  # la 'A' ya está seleccionada al abrir el teclado
        d.tap(Key.A, wait=15)
    d.tap(Key.START, wait=20)  # mueve el cursor a OK
    d.tap(Key.A, wait=60)  # confirma el nombre
    for _ in range(80):  # "¿Así que es AAAA?" -> SÍ, y el resto del discurso hasta el camión
        d.tap(Key.A, wait=40)
        if emu.read32(G_MAIN_CALLBACK2) & ~1 == CB2_OVERWORLD & ~1 and d.pos()[:2] == MAP_TRUCK:
            break
    else:
        raise RuntimeError("no se llegó al camión")
    name = d.state().player_name
    check(name == PLAYER_NAME, f"el nombre es '{name}', no '{PLAYER_NAME}'", d)


def go_through(d: Driver, direction: str, max_steps: int = 12) -> tuple[int, int, int, int]:
    """Cruza a otro mapa yendo en `direction`; si un script interrumpe el camino lo termina y sigue."""
    start = d.pos()[:2]
    for _ in range(6):
        pos = d.move_until_map_change(direction, max_steps)
        if pos[:2] != start and not d.state().script_active:
            return pos
        d.settle()
        if d.pos()[:2] != start:
            return d.pos()
    raise RuntimeError(f"no se pudo salir de {start} yendo '{direction}'")


def leave_truck_and_enter_house(d: Driver) -> None:
    at(d, MAP_TRUCK, "camión")
    d.emu.step(0, 900)  # el camión se sacude un rato; hasta que termina no se puede mover al jugador
    d.move_until_map_change("right", 8)
    at(d, MAP_LITTLEROOT, "salida del camión")
    d.move("down", 1)  # dispara a mamá
    d.settle()
    at(d, MAP_HOUSE_1F, "casa del jugador")


def set_clock(d: Driver) -> None:
    # Subir hasta el televisor: mamá habla y el juego manda al jugador al dormitorio.
    d.move("up", 6)
    d.settle()
    at(d, MAP_BEDROOM, "dormitorio")
    d.move("left", 2)
    d.move("up", 1)  # solo gira hacia el reloj (hay pared)
    d.tap(Key.A, wait=40)  # "El reloj marca..."
    for _ in range(30):  # A hasta que se abra la pantalla del reloj
        if d.emu.read32(G_MAIN_CALLBACK2) & ~1 == CB2_CLOCK_SCREEN & ~1:
            break
        d.tap(Key.A, wait=30)
    else:
        raise RuntimeError("no se abrió la pantalla del reloj")
    d.emu.step(0, 60)
    d.tap(Key.A, wait=10)  # confirma la hora por defecto (10:00 AM)
    for _ in range(40):  # espera el cuadro SÍ/NO ("¿Es la hora correcta?"), que arranca en NO
        if yes_no_box_visible(screenshot_pixels(d.emu)):
            break
        d.emu.step(0, 15)
    else:
        raise RuntimeError("no apareció el cuadro SÍ/NO del reloj")
    d.tap(Key.UP, wait=20)
    d.tap(Key.A, wait=40)
    d.settle()
    pos = at(d, MAP_BEDROOM, "tras el reloj")
    check(pos[2:] == (5, 2), f"se esperaba estar en (5,2) frente al reloj y se está en {pos[2:]}", d)


def leave_house(d: Driver) -> None:
    d.move("right", 2)
    d.move_until_map_change("up", 4)  # escaleras a la planta baja
    d.settle()
    at(d, MAP_HOUSE_1F, "planta baja tras el reloj")
    # Rodear la mesa hasta la puerta (abajo a la derecha) y salir.
    d.move("right", 6)
    d.move("down", 3)
    d.move("left", 1)
    d.move_until_map_change("down", 6)
    at(d, MAP_LITTLEROOT, "afuera de la casa")


def visit_neighbor(d: Driver) -> None:
    d.move("down", 1)
    d.move("right", 9)
    d.move_until_map_change("up", 4)  # puerta de la casa del vecino
    d.settle()
    at(d, MAP_NEIGHBOR_1F, "casa del vecino")
    d.move_until_map_change("up", 10)  # escalera
    d.settle()
    at(d, MAP_NEIGHBOR_ROOM, "cuarto del rival")
    d.move("down", 2)
    d.move("right", 3)  # el rival aparece al acercarse
    d.settle()
    d.move("left", 3)
    go_through(d, "up", 4)
    at(d, MAP_NEIGHBOR_1F, "bajando del cuarto")
    d.settle()
    go_through(d, "down", 10)
    at(d, MAP_LITTLEROOT, "afuera de la casa del vecino")


def rescue_birch(d: Driver) -> None:
    d.move("down", 1)
    d.move("left", 3)
    d.move("up", 10)  # el rival habla en (11,1); no se deja pasar hasta haber visitado al vecino
    d.settle()
    d.move_until_map_change("up", 6)
    d.settle()  # "¡Ayúdame!" de Birch en la Ruta 101
    at(d, MAP_ROUTE_101, "Ruta 101")
    d.move("up", 1)
    d.move("left", 3)
    d.tap(Key.A, wait=60)  # el bolso con los tres Pokémon
    d.tap(Key.RIGHT, wait=30)  # el cursor arranca en Torchic (centro); Mudkip está a la derecha
    d.tap(Key.A, wait=40)  # "¿Elegir este Pokémon?"
    d.tap(Key.A, wait=60)  # SÍ
    s = d.state()
    check(len(s.party) == 1 and s.party[0].species == SPECIES_MUDKIP, "no se consiguió a Mudkip", d)


def win_first_battle(d: Driver) -> None:
    for _ in range(400):
        if not d.state().in_battle:
            break
        d.tap(Key.A, wait=30)
    else:
        raise RuntimeError("la primera batalla no terminó")
    check(any(mon.hp > 0 for mon in d.state().party), "el Mudkip quedó sin HP", d)
    d.settle(max_presses=400)
    pos = at(d, MAP_LAB, "laboratorio de Birch")
    check(not d.state().script_active, "hay un script activo en el laboratorio", d)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="runs/lab_mudkip.ss0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with MGBAClient(port=args.port) as emu:
        d = Driver(emu)
        steps = [
            ("arrancar y elegir NUEVO JUEGO", boot_to_new_game),
            ("crear personaje", create_character),
            ("salir del camión y entrar a la casa", leave_truck_and_enter_house),
            ("poner el reloj", set_clock),
            ("salir de la casa", leave_house),
            ("visitar al vecino y al rival", visit_neighbor),
            ("rescatar a Birch y elegir a Mudkip", rescue_birch),
            ("ganar la primera batalla", win_first_battle),
        ]
        for name, fn in steps:
            print(f"- {name}...", flush=True)
            fn(d)
            print(f"  ok: {d.pos()}", flush=True)
        emu.save_state(out.as_posix())
        s = d.state()
        print(f"savestate guardado en {out}")
        print(f"jugador={s.player_name} mapa={d.pos()} equipo={[(m.species, m.level, m.hp) for m in s.party]}")


if __name__ == "__main__":
    main()
