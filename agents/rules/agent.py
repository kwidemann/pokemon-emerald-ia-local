"""Agente de reglas (enfoque 1): una máquina de estados con lógica explícita, sin aprendizaje.

Prioridad en cada iteración:
  1. batalla abierta        -> pelear (A en FIGHT y A en el primer movimiento; A para avanzar el texto)
  2. pantalla que no es el mundo (menú, fundido, evolución...) -> B para salir
  3. diálogo / script activo -> A para avanzar
  4. en el mapa objetivo     -> terminado
  5. si no                   -> seguir la ruta del grafo de mapas (env/world.py) hasta la salida siguiente,
                                 con búsqueda de caminos sobre las colisiones leídas de la RAM (env/mapdata.py)

Limitaciones conocidas (se irán cubriendo): no cura en Centros Pokémon, no usa objetos, elige siempre el
primer movimiento, no sabe de puzles ni de NPC que bloquean hasta que pase un evento de historia.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field

from emulator.client import Key
from env.mapdata import STEP, MapView, find_path, read_map
from env.battle_data import BattleData
from env.milestones import BATTLE_TYPE_TRAINER
from env.ram_map import G_ENEMY_PARTY, PARTY_MAX, POKEMON_SIZE
from env.state import GameState, parse_party_mon
from env.world import Exit, MapId, World
from scripts.drive import Driver

G_BATTLER_PARTY_INDEXES = 0x0202406E  # u16[4]: índice en el equipo de cada combatiente (0 = jugador, 1 = rival); [SYM], sin verificar en vivo
ALL_DIRS = ("down", "up", "left", "right")


@dataclass
class Stats:
    battles: int = 0
    dialogue_presses: int = 0
    moves: int = 0
    menu_exits: int = 0
    blocked_exits: int = 0
    log: list[str] = field(default_factory=list)


class RulesAgent:
    def __init__(
        self,
        driver: Driver,
        world: World,
        verbose: bool = True,
        on_state: callable | None = None,
        heal_below: float = 0.5,
    ):
        self.heal_below = heal_below  # fracción de HP por debajo de la cual se va a curar
        self._healing = False
        self._heal_failed = False
        self._blocked_for: MapId | None = None
        self._last_map: MapId | None = None
        self._text_key = Key.B
        self._text_run = 0
        self.on_state = on_state  # se llama con cada GameState leído (p. ej. para seguir hitos)
        self.d = driver
        self.world = world
        self.battle_data = BattleData(world.rom)
        self.verbose = verbose
        self.stats = Stats()
        self.blocked: set[tuple[MapId, Exit]] = set()
        self._was_in_battle = False

    def say(self, msg: str) -> None:
        self.stats.log.append(msg)
        if self.verbose:
            print(msg, flush=True)

    # --- acciones simples -------------------------------------------------------------------

    def _pick_move(self, s: GameState) -> int:
        """Índice (0-3) del movimiento con más daño esperado contra el rival activo. 0 si no se puede decidir."""
        try:
            _, (idx, enemy_raw) = self.d.emu.read_multi([(G_BATTLER_PARTY_INDEXES, 8), (G_ENEMY_PARTY, POKEMON_SIZE * PARTY_MAX)])
            mine_i, enemy_i = struct.unpack_from("<HH", idx, 0)
            mine = s.party[mine_i]
            enemy = parse_party_mon(enemy_raw[enemy_i * POKEMON_SIZE : (enemy_i + 1) * POKEMON_SIZE])
        except (IndexError, struct.error, ValueError):
            return 0
        if enemy is None or not enemy.checksum_ok or not mine.checksum_ok:
            return 0
        scores = [
            self.battle_data.move_score(mv, mine.species, enemy.species) if mv and pp > 0 else -1.0
            for mv, pp in zip(mine.moves, mine.pp)
        ]
        best = max(range(4), key=lambda k: scores[k])
        return best if scores[best] > 0 else next((k for k in range(4) if scores[k] >= 0), 0)

    def _battle_turn(self) -> None:
        """Una "ronda" de pulsaciones: FIGHT, mover el cursor al movimiento elegido y aceptar.

        No se sabe en qué parte del menú se está (texto, acciones, movimientos): las flechas no hacen nada durante
        el texto y las A avanzan, así que repetir la secuencia converge. El cursor de movimientos recuerda el último
        usado, por eso se lleva primero a la esquina superior izquierda (arriba, izquierda)."""
        k = self._pick_move(self.d.state())
        # El cursor del menú de acciones recuerda la última opción (visto sobre HUIR): se lleva a LUCHAR antes de aceptar.
        self.d.tap(Key.UP, wait=6)
        self.d.tap(Key.LEFT, wait=6)
        self.d.tap(Key.A, wait=15)
        self.d.tap(Key.UP, wait=6)
        self.d.tap(Key.LEFT, wait=6)
        if k % 2:
            self.d.tap(Key.RIGHT, wait=6)
        if k >= 2:
            self.d.tap(Key.DOWN, wait=6)
        self.d.tap(Key.A, wait=25)

    def _leave_screen(self) -> None:
        self.d.tap(Key.B, wait=25)
        self.stats.menu_exits += 1

    def _advance_text(self) -> None:
        """Avanza un cuadro de texto. B por defecto: con A, el último toque vuelve a "hablar" con el NPC de enfrente
        y el diálogo se repite sin fin (visto en Petalburg). A solo cuando se espera responder SÍ (`_text_key`), o si B
        no consigue cerrar el script tras muchas pulsaciones (p. ej. una pregunta SÍ/NO)."""
        self._text_run += 1
        use_a = self._text_key == Key.A or (self._text_run > 40 and self._text_run % 40 < 15)
        self.d.tap(Key.A if use_a else Key.B, wait=20)
        self.stats.dialogue_presses += 1

    # --- navegación -------------------------------------------------------------------------

    def _exit_goals(self, view: MapView, ex: Exit) -> set[tuple[int, int]]:
        if ex.kind == "warp":
            if view.walkable(ex.x, ex.y):  # alfombra/escalera: se pisa
                return {(ex.x, ex.y)}
            # Puerta de edificio: su casilla tiene colisión; el warp se activa al empujarla desde una vecina.
            return {(ex.x + dx, ex.y + dy) for dx, dy in STEP.values() if view.walkable(ex.x + dx, ex.y + dy)}
        goals = set()
        w, h = view.inner_width, view.inner_height
        for x in range(-1, w + 1):
            for y in range(-1, h + 1):
                outside = {"up": y < 0, "down": y >= h, "left": x < 0, "right": x >= w}[ex.direction]
                if outside and view.walkable(x, y):
                    goals.add((x, y))
        return goals

    def _step_out(self, view: MapView, start_map: MapId) -> bool:
        """De pie sobre una casilla de warp: pulsar la dirección que la activa (las alfombras de salida piden ABAJO)."""
        p = view.player
        order = sorted(ALL_DIRS, key=lambda dr: view.walkable(p.x + STEP[dr][0], p.y + STEP[dr][1]))
        for direction in order:
            self.d.move(direction, 1)
            self.d.emu.step(0, 60)
            if self.d.pos()[:2] != start_map:
                return True
            if self.d.state().script_active:
                return False
        return False

    def _enter_warp(self, view: MapView, ex: Exit, start_map: MapId) -> None:
        p = view.player
        if (p.x, p.y) == (ex.x, ex.y):
            self._step_out(view, start_map)  # de pie sobre la alfombra
            return
        toward = next(dr for dr, (dx, dy) in STEP.items() if (p.x + dx, p.y + dy) == (ex.x, ex.y))
        self.d.move(toward, 1)  # empuja la puerta
        self.d.emu.step(0, 60)

    def _follow(self, exit_: Exit, view: MapView, start_map: MapId) -> bool:
        """Lleva al jugador hacia `exit_`. Devuelve False si no hay camino."""
        goals = self._exit_goals(view, exit_)
        p = view.player
        if exit_.kind == "warp" and (p.x, p.y) in goals:
            self._enter_warp(view, exit_, start_map)
            return True
        path = find_path(view, goals) or find_path(view, goals, avoid_npcs=False)
        if not path:
            return False
        for move in path:
            before = self.d.pos()
            self.d.move(move, 1)
            self.stats.moves += 1
            s = self.d.state()
            if s.in_battle or s.script_active or not s.in_overworld or self.d.pos()[:2] != start_map:
                return True
            if self.d.pos() == before:  # algo bloquea: volver a calcular el camino con el estado nuevo
                return True
        if exit_.kind == "warp":
            view = read_map(self.d.emu, *start_map)
            if (view.player.x, view.player.y) in goals:
                self._enter_warp(view, exit_, start_map)
        else:
            self.d.move(exit_.direction, 1)  # un paso más hacia afuera cruza la conexión
            self.d.emu.step(0, 30)
        return True

    def _interrupts(self, s: GameState) -> bool:
        """Atiende lo que tiene prioridad sobre navegar (batalla, menús, diálogos). True si hizo algo."""
        if s.in_battle:
            if not self._was_in_battle:
                self.stats.battles += 1
                kind = "entrenador" if s.battle_type_flags & BATTLE_TYPE_TRAINER else "salvaje"
                self.say(f"  batalla {kind} (flags={s.battle_type_flags:#x}) en {(s.map_group, s.map_num)}")
            self._was_in_battle = True
            self._battle_turn()
            return True
        if self._was_in_battle:
            self._was_in_battle = False
            lead = s.party[0] if s.party else None
            self.say(f"  fin de la batalla: líder nv{lead.level if lead else '?'} hp {lead.hp if lead else '?'}")
        if not s.in_overworld:
            self._leave_screen()
            return True
        if s.script_active:
            self._advance_text()
            return True
        self._text_run = 0
        return False

    def _hurt(self, s: GameState) -> bool:
        fracs = [m.hp / m.max_hp for m in s.party if m.max_hp]
        return bool(fracs) and min(fracs) < self.heal_below

    def heal(self, target: MapId | None = None) -> bool:
        """Va al Centro Pokémon más cercano (por número de salidas) y habla con la enfermera."""
        here = self.d.pos()[:2]
        pos = self.d.pos()[2:]
        options = []
        for pc in self.world.pokemon_centers():
            route = self.world.route(here, pc, blocked=self.blocked, pos=pos)
            if route is None:
                continue
            onward = self.world.route(pc, target) if target else []  # desvío: ir al Centro y luego al destino
            if onward is not None:
                options.append((len(route) + len(onward), pc))
        if not options:
            self.say("  no hay Centro Pokémon alcanzable")
            return False
        _, pc = min(options)
        self.say(f"  HP bajo: voy a curarme a {pc}")
        self._healing = True
        try:
            return self.talk_to(pc, self.world.NURSE_GRAPHICS_ID)
        finally:
            self._healing = False

    def grind(self, map_id: MapId, level: int, max_iterations: int = 3000) -> bool:
        """Camina por la hierba alta de `map_id` provocando combates salvajes hasta que el líder llegue a `level`."""
        if not self.goto(map_id):
            return False
        for it in range(max_iterations):
            s = self.d.state()
            if self.on_state and not s.in_transition:
                self.on_state(s)
            if self._interrupts(s):
                continue
            if s.party and s.party[0].level >= level:
                self.say(f"  líder nv{s.party[0].level} tras {it} iteraciones de entrenamiento")
                return True
            here = (s.map_group, s.map_num)
            if self._hurt(s) and here not in self.world.pokemon_centers() and not self._heal_failed:
                if not self.heal():
                    self._heal_failed = True
                continue
            if here != map_id:
                if not self.goto(map_id):
                    return False
                continue
            view = read_map(self.d.emu, *here)
            p = view.player
            grass = {
                (x, y)
                for x in range(view.inner_width)
                for y in range(view.inner_height)
                if view.is_grass(x, y) and view.walkable(x, y)
            }
            grass.discard((p.x, p.y))
            if not grass:
                self.say(f"  no hay hierba alta alcanzable en {map_id}")
                return False
            path = find_path(view, grass)
            if not path:
                self.say("  no hay camino hasta la hierba")
                return False
            self.d.move(path[0], 1)
            self.stats.moves += 1
        self.say("  entrenamiento: se acabaron las iteraciones")
        return False

    def settle(self, max_iterations: int = 300) -> None:
        """Resuelve diálogos y batallas hasta quedar libre en el mundo (con una pausa para scripts que se reanudan)."""
        free = 0
        for _ in range(max_iterations):
            s = self.d.state()
            if self.on_state and not s.in_transition:
                self.on_state(s)
            if self._interrupts(s):
                free = 0
                continue
            free += 1
            if free >= 3:
                return
            self.d.emu.step(0, 60)
        self.say("  settle: no se quedó libre")

    def talk_to(self, map_id: MapId, graphics_id: int, max_iterations: int = 400) -> bool:
        """Va al mapa, camina hasta el NPC con ese gráfico y le habla (A). Los entrenadores también cuentan."""
        if not self.goto(map_id):
            return False
        template = next((n for n in self.world.maps[map_id].npcs if n.graphics_id == graphics_id), None)
        for _ in range(max_iterations):
            s = self.d.state()
            if self.on_state and not s.in_transition:
                self.on_state(s)
            if self._interrupts(s):
                continue
            if (s.map_group, s.map_num) != map_id:
                self.say(f"  se salió del mapa {map_id} mientras buscaba al NPC")
                return False
            view = read_map(self.d.emu, *map_id)
            live = next((o for o in view.objects if o.graphics_id == graphics_id and not o.is_player), None)
            if live is None and template is None:
                self.say(f"  no hay NPC con gráfico {graphics_id} en {map_id}")
                return False
            nx, ny = (live.x, live.y) if live else (template.x, template.y)
            p = view.player
            # Casillas desde las que se puede hablar: pegado al NPC, o a dos casillas con un mostrador en medio.
            spots: dict[tuple[int, int], str] = {}
            for dr, (dx, dy) in STEP.items():
                near = (nx - dx, ny - dy)  # casilla desde la que se mira hacia `dr` y se llega al NPC
                if view.walkable(*near):
                    spots[near] = dr
                far = (nx - 2 * dx, ny - 2 * dy)
                if not view.walkable(*near) and view.walkable(*far) and view.cell(*near) is not None:
                    spots[far] = dr  # mostrador (colisión) entre medias
            if (p.x, p.y) in spots:
                self.d.move(spots[(p.x, p.y)], 1)  # solo gira hacia el NPC o el mostrador
                self.d.tap(Key.A, wait=40)
                self.say(f"  habló con el NPC {graphics_id} en ({nx},{ny})")
                self._text_key = Key.A  # en la conversación buscada se responde SÍ (enfermera, etc.)
                try:
                    self.settle()
                finally:
                    self._text_key = Key.B
                return True
            path = find_path(view, set(spots))
            if not path:
                self.say(f"  no hay camino hasta el NPC {graphics_id} en ({nx},{ny})")
                return False
            for move in path[:8]:  # tramos cortos: el NPC puede moverse
                before = self.d.pos()
                self.d.move(move, 1)
                self.stats.moves += 1
                st = self.d.state()
                if st.in_battle or st.script_active or not st.in_overworld or self.d.pos() == before:
                    break
        return False

    def goto(self, target: MapId, max_iterations: int = 600, until: callable | None = None) -> bool:
        """Va al mapa `target`. Devuelve True si llegó, False si se rindió o se acabaron las iteraciones."""
        t0 = time.time()
        last_sig, same = None, 0
        for it in range(max_iterations):
            s = self.d.state()
            if self.on_state and not s.in_transition:
                self.on_state(s)
            sig = (s.map_group, s.map_num, s.x, s.y, s.script_active, s.in_battle, s.callback2, s.party[0].hp if s.party else 0)
            # En batalla la firma casi no cambia (el HP del enemigo no está en ella): no cuenta como atasco.
            same = same + 1 if sig == last_sig and not s.in_battle else 0
            last_sig = sig
            if same > 60:
                self.say(f"  atascado en {sig}")
                return False

            if self._interrupts(s):
                continue
            here = (s.map_group, s.map_num)
            if here != self._last_map:
                self.say(f"  mapa {self._last_map} -> {here} en ({s.x},{s.y})")
                self._last_map = here
            if not self._healing and not self._heal_failed and self._hurt(s) and here not in self.world.pokemon_centers():
                if not self.heal(target):
                    self._heal_failed = True  # no insistir: se sigue sin curar
                continue
            if here == target or (until and until(s)):
                self.say(f"  llegó a {here} en {time.time() - t0:.0f}s ({it} iteraciones)")
                return True

            if here != self._blocked_for:  # los bloqueos valen solo para la zona del mapa donde se descubrieron
                self.blocked.clear()
                self._blocked_for = here
            route = self.world.route(here, target, blocked=self.blocked, pos=(s.x, s.y))
            if not route:
                self.say(f"  sin ruta de {here} a {target}")
                return False
            ex = route[0]
            view = read_map(self.d.emu, *here)
            if not self._follow(ex, view, here):
                self.blocked.add((here, ex))
                self.stats.blocked_exits += 1
                self.say(f"  sin camino local en {here} hacia {ex.kind} {ex.dest}; se prueba otra salida")
        self.say("  se acabaron las iteraciones")
        return False
