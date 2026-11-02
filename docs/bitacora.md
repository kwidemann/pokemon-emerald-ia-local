# Bitácora de pokemon-ia

Registro cronológico de lo que se hizo, lo que se decidió y lo que se descubrió. Las especificaciones y el diseño viven en [`specs.md`](../specs.md); aquí va el "qué pasó y cuándo".

**Cómo se mantiene:** al terminar cada paso se agrega una entrada al final de su día (más reciente abajo) con *Hecho*, *Verificado*, *Decisiones* y *Pendiente*. Lo que no se comprobó en vivo se dice. Cada entrada de un paso que cambió archivos termina con el commit que la contiene, cuando el usuario decide hacerlo.

---

## 2026-11-01

### Diseño y decisiones iniciales
- **Objetivo:** proyecto educativo local en el que agentes de IA juegan Pokémon Esmeralda (GBA), orquestados con scripts de Python invocables desde Claude Code, aprovechando la GPU y la CPU.
- **Enfoques a comparar:** (1) reglas / máquina de estados, (2) aprendizaje por refuerzo (PPO), (3) modelo de mundo tipo JEPA sobre la RAM, (4) híbrido: ejecutor + Laya (decisor rápido local) + Ollama (supervisor lento).
- **Decisiones del usuario:** emulador mGBA con puente Lua por socket; observación estructurada leída de la RAM; meta a largo plazo de completar el juego mediante hitos; JEPA sobre RAM y no sobre píxeles; supervisor textual con `qwen2.5:7b` o `gemma:7b`; se acepta el diseño híbrido de dos niveles.
- **Hardware:** RTX 2060 Max-Q de 6 GB, i7-10750H (6 núcleos), 11,8 GB de RAM.

### Fase 1: fundaciones
- **Hecho:** proyecto `uv` con Python 3.12 (el 3.13 del sistema se evitó); mGBA 0.10.5 instalado con winget; `emulator/server.lua` (servidor TCP en el puerto 8888), `emulator/client.py` (`MGBAClient`) y `scripts/ping.py`.
- **Verificado con la ROM real:** `ping`, `frame`, `step`, `readrange`, `savestate`, `loadstate` y `screenshot`.
- **Descubierto:** `mGBA.exe --script` no existe en 0.10.5 y el programa se cierra, así que el Lua se carga a mano en cada sesión (*Tools > Scripting > File > Load script*). `step N` avanza N+1 frames, una constante que no afecta el determinismo.
- **Incidente:** el parser de rutas cortaba en el primer espacio y mGBA escribió un archivo suelto llamado `Repositorios` en `Documents`. Se borró y se corrigió `server.lua`; la corrección **aún no está cargada** en mGBA, y mientras tanto las capturas de pantalla vuelven a dejar ese archivo suelto, que se borra a mano.

### Fase 2: mapa de RAM
- **Hecho:** `env/ram_map.py` (direcciones con su fuente y estado de verificación), `env/state.py` (`read_state` y `parse_party_mon`) y `scripts/inspect_state.py`.
- **ROM:** SHA-1 `f3ae0881…07b7`, idéntico al de pret/pokeemerald (v1.0 USA/Europa, `BPEE`).
- **Descubierto:** los save blocks cambian de dirección en cada arranque; se leen siempre a través de `gSaveBlock1Ptr` y `gSaveBlock2Ptr`.
- **Verificado en vivo:** posición, mapa, dinero (3000, con el XOR de la clave de cifrado), nombre del jugador y estado de script (0 con texto abierto, 2 sin script).

### Automatización de la intro y verificación del equipo
- **Hecho:** `scripts/drive.py` (`Driver` con `advance_dialogue`, `move`, `press`). Con él el agente llevó la partida desde Littleroot hasta el laboratorio de Birch: casa, reloj de pared, casa del vecino y su cuarto, Ruta 101, elección de **Mudkip** (elegido por el agente), primera batalla y laboratorio.
- **Verificado en vivo:** el descifrado del Pokémon (`species=283`, nivel 5, HP 21/21, Placaje y Gruñido, `checksum_ok=True`); los cambios tras la batalla (PP 35→32, HP 21→18, experiencia 135→152); `in_battle` pasa a `True` al empezar y a `False` al terminar; `gBattleTypeFlags = 0x14` en la primera batalla; `loadstate` devuelve la posición original.
- **Descubierto:** un NPC bloquea la salida norte de Littleroot ("Um, um, um!") hasta visitar la casa del vecino. En la pantalla del reloj el cursor arranca en NO. Una pulsación corta (1–6 frames) mueve exactamente una casilla; mantener la tecla mientras se lee la posición hace avanzar varias. El emulador corre a 60 fps, tiempo real.
- **Savestate inicial:** `C:\AI\pokemon-ia-states\lab_mudkip.ss0` (laboratorio, `(6,5)`, Mudkip nivel 5). Está fuera del proyecto por el problema de las rutas con espacios.

### Fase 3: entorno Gymnasium (primera versión)
- **Hecho:** `readmulti` en `server.lua` y `MGBAClient.read_multi` (con caída automática al método antiguo si el Lua cargado es viejo); `read_state` ahora usa 2 llamadas; `env/milestones.py`, `env/pokemon_env.py` y `scripts/random_agent.py`.
- **Verificado:** `reset()` desde el savestate marca los hitos 3 y 5 sin recompensa; 60 pasos del agente aleatorio sin errores; una acción mueve una casilla si el personaje ya mira en esa dirección, y si no solo lo gira.
- **Medido:** 1,7 pasos/s a velocidad real (20 frames por acción, 60 fps). Con tiempo real no se puede entrenar.
- **Sin verificar:** `readmulti` en el servidor (el Lua cargado en mGBA es el viejo y se usó la caída automática), y los IDs de mapa inferidos de Oldale, Ruta 103, Petalburg y Rustboro.
- **Después de recargar el Lua y activar el avance rápido:** `readmulti` y las rutas con espacios funcionan; el savestate se movió a `runs/lab_mudkip.ss0`. Velocidad: ~52 pasos/s (antes 1,7). Se descubrió que al cruzar una puerta la ubicación se lee un paso como `(0,0)` en `(0,0)`, lo que disparó el hito de Petalburg por error; se añadió `GameState.in_transition` y el entorno lo ignora. Las rutas relativas del savestate fallaban porque mGBA las resuelve desde su directorio; el entorno ahora las vuelve absolutas.
- **Generador del savestate (`scripts/make_savestate.py`):** comando `reset` en el Lua; el script juega la intro desde el logo hasta el laboratorio en ~16 s y se comprueba contra la RAM en cada tramo. Dos ejecuciones seguidas funcionaron y el `.sav` quedó intacto (mismo MD5). Se descubrió que con el avance rápido el emulador sigue corriendo (~2200 fps) entre comandos, así que el RNG no es reproducible: el flujo sí, los valores no. Problemas resueltos por el camino: la pantalla de nombre perdía la primera pulsación (se espera el fundido), el camión no deja moverse hasta que termina su sacudida (~900 frames), el SÍ/NO del reloj arranca en NO y se detecta por píxeles, y mamá del vecino vuelve a hablar al bajar la escalera.
- **Sin verificar:** el arranque sin `.sav` previo; la reproducibilidad exacta.
- **Pendiente:** verificar los detectores de hitos; paralelizar emuladores.

### Fase 4: agente de reglas
- **Hecho:** `env/mapdata.py` (colisiones, NPC, warps y conexiones desde la RAM, BFS, hierba y desniveles por comportamiento de casilla), `env/world.py` (grafo de 518 mapas desde la ROM con zonas conexas, `route`, NPC y centros Pokémon), `env/battle_data.py` (movimientos y tipos desde la ROM), `agents/rules/agent.py` y `scripts/run_rules.py`. `GameState` ganó `callback2`, `in_overworld` y `flags` (banderas de eventos).
- **Verificado:** el agente, desde el savestate del laboratorio, llegó solo a Oldale, venció al rival en la Ruta 103, recibió la Pokédex (flag `0x861`), llegó a Petalburg, activó la escena de Wally (flags de batalla `0x204`), cruzó el Bosque Petalburgo y llegó a Rustboro. Se curó en dos Centros Pokémon. Datos de la ROM contrastados con el juego.
- **Problemas:** un cuelgue del juego sin causa conocida en un combate de entrenador (se perdió ese progreso); el agente aún no llega a la medalla 1.
- **Pendiente:** hitos 13 a 16; investigar el cuelgue (añadir un comando para leer registros de la CPU); repetir ejecuciones para medir la tasa de éxito.

### Control de versiones y organización
- **Hecho:** `git init` con rama `main`. Se creó esta bitácora y se añadió a `specs.md` la sección de hitos propuestos. A pedido del usuario se hizo el primer commit del proyecto (el hash se ve con `git log`).
- **Publicación en GitHub:** el equipo no tiene `gh` instalado, así que el repositorio público lo crea el usuario en GitHub y después se hace el `push` desde aquí.
- **Hitos:** el usuario aprobó la lista de 16 hitos del Tramo 1 (`specs.md`, 9.1). El orden de las rutas viene de Bulbapedia, pero la página no se pudo descargar completa (403), así que los detalles marcados **[memoria]** (Pokédex, Wally, Team Aqua, nivel de evolución y equipo de Roxanne) quedan por verificar jugando.
- **Decisiones:** se usará **solo Laya** (no Jev real por API); se versiona el proyecto con git; no se hacen commits sin que el usuario lo pida.
- **Pendiente:**
  - Elegir hitos definitivos y su orden (propuesta en `specs.md`, sección 9.1).
  - Avance rápido sin límite en mGBA y nueva medición de velocidad.
  - Recargar `server.lua` y mover el savestate a `runs/`.
  - Comando `readmulti` en el Lua.
  - Verificar medallas y batallas contra entrenadores.
  - Fase 3: entorno Gymnasium.
